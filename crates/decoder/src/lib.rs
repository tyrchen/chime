//! Joint dual-track lattice decoder (Pinyin + Fuzzy + Typo + English Code-Switching)
//! with Beam Search, step-by-step prefix word candidates, `NeuralReranker` hook,
//! and 60ms Anti-Jitter Freeze Window.

#![forbid(unsafe_code)]

use chime_lexicon::{Lexicon, UserStore, WordEntry};
use chime_phonetics::{
    CodeSwitchSignal, FuzzyConfig, ShuangpinScheme, SyllableId, classify_code_switch,
    expand_shuangpin_pair, generate_typo_variants, lookup_syllable,
};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Provenance of a decoded candidate span, used for UI hints (e.g. subtle typo underline).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchKind {
    /// Exact Pinyin, Shuangpin, or English match.
    Exact,
    /// Matched via enabled fuzzy Pinyin equivalence.
    Fuzzy,
    /// Matched via initial-letter abbreviation skeleton (`zgr` -> `中国人`).
    Abbreviation,
    /// Recovered via QWERTY spatial neighbor or transposition typo correction.
    TypoCorrected,
    /// Modeless English / raw code pass-through.
    CodeSwitch,
}

/// Decoded candidate returned to the UI layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    /// Rendered surface string (with Pangu spacing applied when configured).
    pub text: String,
    /// Total log-score (higher is better).
    pub score: f32,
    /// Dominant match kind across the candidate's spans.
    pub kind: MatchKind,
    /// Byte length of the raw input consumed by this candidate (`< raw_input.len()` for partial
    /// prefix selection).
    pub consumed_bytes: usize,
}

/// Request context supplied on every keystroke decode.
#[derive(Debug, Clone)]
pub struct DecodeContext<'a> {
    /// Left surrounding text before the cursor (up to ~64 chars for prefix KV-cache / bigram).
    pub left_context: &'a str,
    /// Active `macOS` host application bundle identifier (e.g. `com.microsoft.VSCode`).
    pub app_bundle_id: &'a str,
    /// Active keyboard scheme (Full Pinyin or Shuangpin).
    pub scheme: ShuangpinScheme,
    /// Fuzzy Pinyin configuration.
    pub fuzzy: FuzzyConfig,
    /// Whether to insert Pangu half-width spaces at Chinese-English boundaries.
    pub pangu_spacing: bool,
    /// Current UNIX timestamp in seconds (for user habit decay).
    pub now_epoch_secs: u64,
}

impl Default for DecodeContext<'_> {
    fn default() -> Self {
        Self {
            left_context: "",
            app_bundle_id: "",
            scheme: ShuangpinScheme::Full,
            fuzzy: FuzzyConfig::default(),
            pangu_spacing: true,
            now_epoch_secs: 0,
        }
    }
}

/// Pluggable Stage-2 neural language model reranker (`SafetensorsReranker` or `macOS` 15 `CoreML
/// MLState`).
pub trait NeuralReranker: std::fmt::Debug + Send + Sync {
    /// Scores a batch of candidate surface texts given immutable `left_context` and
    /// `app_bundle_id`. Returns one additive log-likelihood score per candidate.
    fn score_batch(
        &self,
        left_context: &str,
        app_bundle_id: &str,
        candidates: &[Candidate],
    ) -> Vec<f32>;
}

#[derive(Debug, Clone)]
struct LatticeEdge {
    end: usize,
    surface: String,
    score: f32,
    kind: MatchKind,
    is_english: bool,
}

#[derive(Debug, Clone)]
struct BeamState {
    text: String,
    last_word: String,
    last_is_english: bool,
    score: f32,
    kind: MatchKind,
}

/// Stage-1 + optional synchronous Stage-2 decoder over raw preedit keystrokes.
#[derive(Debug, Default)]
pub struct LatticeDecoder {
    reranker: Option<Box<dyn NeuralReranker>>,
}

impl LatticeDecoder {
    /// Creates a new lattice decoder without an attached synchronous neural reranker.
    #[must_use]
    pub fn new() -> Self {
        Self { reranker: None }
    }

    /// Attaches a synchronous Stage-2 neural reranker.
    #[must_use]
    pub fn with_reranker(mut self, reranker: Box<dyn NeuralReranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Decodes `raw_input` into ranked full-sentence candidates followed by step-by-step prefix
    /// word candidates.
    #[must_use]
    pub fn decode(
        &self,
        raw_input: &str,
        lexicon: &Lexicon,
        user_store: &UserStore,
        ctx: &DecodeContext<'_>,
    ) -> Vec<Candidate> {
        if raw_input.is_empty() {
            return Vec::new();
        }

        if classify_code_switch(raw_input) == CodeSwitchSignal::HardRawCode {
            return vec![Candidate {
                text: raw_input.to_string(),
                score: 0.0,
                kind: MatchKind::CodeSwitch,
                consumed_bytes: raw_input.len(),
            }];
        }

        let is_dev_app = UserStore::is_developer_app(ctx.app_bundle_id);
        let edges_by_start = self.build_edges(raw_input, lexicon, user_store, ctx, is_dev_app);
        let n = raw_input.len();

        let mut beams: Vec<Vec<BeamState>> = vec![Vec::new(); n + 1];
        beams[0].push(BeamState {
            text: String::new(),
            last_word: extract_tail_word(ctx.left_context),
            last_is_english: false,
            score: 0.0,
            kind: MatchKind::CodeSwitch,
        });

        for i in 0..n {
            if beams[i].is_empty() {
                continue;
            }
            if raw_input.as_bytes()[i] == b'\'' {
                let carried = beams[i].clone();
                Self::merge_and_prune(&mut beams[i + 1], carried, 12);
                continue;
            }

            let current_states = beams[i].clone();
            for edge in &edges_by_start[i] {
                let mut next_states = Vec::with_capacity(current_states.len());
                for st in &current_states {
                    let mut composed = st.text.clone();
                    if !composed.is_empty()
                        && ctx.pangu_spacing
                        && !is_dev_app
                        && (st.last_is_english != edge.is_english)
                    {
                        composed.push(' ');
                    }
                    composed.push_str(&edge.surface);

                    let next_kind = if st.text.is_empty() {
                        edge.kind
                    } else {
                        combine_beam_kind(st.kind, edge.kind)
                    };
                    let bigram = lexicon.bigram_bonus(&st.last_word, &edge.surface);
                    next_states.push(BeamState {
                        text: composed,
                        last_word: edge.surface.clone(),
                        last_is_english: edge.is_english,
                        score: st.score + edge.score + bigram - 0.55,
                        kind: next_kind,
                    });
                }
                Self::merge_and_prune(&mut beams[edge.end], next_states, 12);
            }
        }

        self.finalize_candidates(
            raw_input,
            &beams[n],
            &edges_by_start[0],
            lexicon,
            user_store,
            ctx,
        )
    }

    fn finalize_candidates(
        &self,
        raw_input: &str,
        terminal_beams: &[BeamState],
        leading_edges: &[LatticeEdge],
        lexicon: &Lexicon,
        user_store: &UserStore,
        ctx: &DecodeContext<'_>,
    ) -> Vec<Candidate> {
        let n = raw_input.len();
        let mut full_candidates: Vec<Candidate> = terminal_beams
            .iter()
            .map(|st| {
                let user_delta = user_store.score_boost(
                    raw_input,
                    &st.text,
                    ctx.app_bundle_id,
                    ctx.now_epoch_secs,
                );
                Candidate {
                    text: st.text.clone(),
                    score: st.score + user_delta,
                    kind: st.kind,
                    consumed_bytes: n,
                }
            })
            .collect();

        // Include custom user-coined full phrases from L1 Cedar trie!
        for (surf, weight) in user_store.lookup_custom_phrases(&raw_input.to_ascii_lowercase()) {
            if !full_candidates.iter().any(|c| c.text == *surf) {
                full_candidates.push(Candidate {
                    text: surf.clone(),
                    score: -2.0 + (1.0 + weight).ln(),
                    kind: MatchKind::Exact,
                    consumed_bytes: n,
                });
            }
        }

        // Append English prefix completions for long pure-English inputs (`kubern` ->
        // `Kubernetes`).
        let lower_all = raw_input.to_ascii_lowercase();
        for comp in lexicon.complete_english_prefix(&lower_all, 2) {
            if !full_candidates.iter().any(|c| c.text == comp.surface) {
                full_candidates.push(Candidate {
                    text: comp.surface,
                    score: comp.log_prob - 1.2,
                    kind: MatchKind::CodeSwitch,
                    consumed_bytes: n,
                });
            }
        }

        full_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        full_candidates.truncate(12);

        if let Some(reranker) = &self.reranker {
            let bonuses =
                reranker.score_batch(ctx.left_context, ctx.app_bundle_id, &full_candidates);
            for (cand, bonus) in full_candidates.iter_mut().zip(bonuses) {
                cand.score += bonus;
            }
            full_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        }

        // Append top partial prefix word candidates (`consumed_bytes < n`) for step-by-step phrase
        // selection!
        let mut prefix_edges: Vec<&LatticeEdge> =
            leading_edges.iter().filter(|e| e.end < n).collect();
        prefix_edges.sort_by(|a, b| b.end.cmp(&a.end).then_with(|| b.score.total_cmp(&a.score)));

        for pe in prefix_edges {
            if full_candidates.len() >= 18 {
                break;
            }
            if !full_candidates.iter().any(|c| c.text == pe.surface) {
                full_candidates.push(Candidate {
                    text: pe.surface.clone(),
                    score: pe.score - 6.0,
                    kind: pe.kind,
                    consumed_bytes: pe.end,
                });
            }
        }

        // Always keep raw input fallback so user is never trapped (and reachable via Tab
        // rollback!).
        if !full_candidates.iter().any(|c| c.text == raw_input) {
            #[allow(clippy::cast_precision_loss)]
            let fallback_score = -10.0 - 2.5 * (n as f32);
            full_candidates.push(Candidate {
                text: raw_input.to_string(),
                score: fallback_score,
                kind: MatchKind::CodeSwitch,
                consumed_bytes: n,
            });
        }

        full_candidates
    }

    #[allow(clippy::too_many_lines)]
    fn build_edges(
        &self,
        raw_input: &str,
        lexicon: &Lexicon,
        user_store: &UserStore,
        ctx: &DecodeContext<'_>,
        is_dev_app: bool,
    ) -> Vec<Vec<LatticeEdge>> {
        let _ = self;
        let n = raw_input.len();
        let lower = raw_input.to_ascii_lowercase();
        let syl_edges = Self::build_syllable_spans(&lower, ctx);
        let mut edges: Vec<Vec<LatticeEdge>> = vec![Vec::new(); n];

        Self::populate_mandarin_edges(
            &lower,
            &syl_edges,
            lexicon,
            user_store,
            ctx.app_bundle_id,
            ctx.now_epoch_secs,
            is_dev_app,
            &mut edges,
        );

        // Super-abbreviation initial skeletons (`zgr` -> `中国人`, `wsm` -> `为什么`).
        if ctx.scheme == ShuangpinScheme::Full {
            for start in 0..n {
                for len in 2..=6.min(n - start) {
                    let sub = &lower[start..start + len];
                    if !sub.bytes().all(|b| b.is_ascii_lowercase()) {
                        continue;
                    }
                    for entry in lexicon.lookup_skeleton(sub).into_iter().take(4) {
                        edges[start].push(LatticeEdge {
                            end: start + len,
                            surface: entry.surface,
                            score: entry.log_prob - 1.15,
                            kind: MatchKind::Abbreviation,
                            is_english: false,
                        });
                    }
                }
            }
        }

        // Modeless English / Code-Switch track (`macos` -> `macOS`, `github` -> `GitHub`, `pr` ->
        // `PR`).
        for start in 0..n {
            for end in (start + 2)..=n {
                let raw_sub = &raw_input[start..end];
                let lower_sub = &lower[start..end];
                if let Some(eng) = lexicon.lookup_english(lower_sub) {
                    let dev_bonus = if is_dev_app && eng.is_tech_term {
                        0.95
                    } else {
                        0.0
                    };
                    let phonotactic_bonus =
                        if classify_code_switch(raw_sub) == CodeSwitchSignal::PhonotacticEnglish {
                            1.2
                        } else {
                            0.0
                        };
                    edges[start].push(LatticeEdge {
                        end,
                        surface: eng.surface,
                        score: eng.log_prob + dev_bonus + phonotactic_bonus,
                        kind: MatchKind::CodeSwitch,
                        is_english: true,
                    });
                } else if classify_code_switch(raw_sub) != CodeSwitchSignal::AmbiguousAscii {
                    #[allow(clippy::cast_precision_loss)]
                    let per_char_cost = -1.95 * (raw_sub.len() as f32);
                    edges[start].push(LatticeEdge {
                        end,
                        surface: raw_sub.to_string(),
                        score: per_char_cost,
                        kind: MatchKind::CodeSwitch,
                        is_english: true,
                    });
                }
            }
        }

        edges
    }

    fn build_syllable_spans(
        lower: &str,
        ctx: &DecodeContext<'_>,
    ) -> Vec<Vec<(usize, SyllableId, f32, MatchKind)>> {
        let n = lower.len();
        let bytes = lower.as_bytes();
        let mut spans: Vec<Vec<(usize, SyllableId, f32, MatchKind)>> = vec![Vec::new(); n];

        for i in 0..n {
            if bytes[i] == b'\'' {
                continue;
            }
            if ctx.scheme != ShuangpinScheme::Full {
                if i + 2 <= n {
                    for id in expand_shuangpin_pair(ctx.scheme, bytes[i], bytes[i + 1]) {
                        spans[i].push((i + 2, id, 0.0, MatchKind::Exact));
                        let canonical = chime_phonetics::canonical_syllable(id);
                        for (fid, pen) in ctx.fuzzy.expand(canonical) {
                            spans[i].push((i + 2, fid, -pen, MatchKind::Fuzzy));
                        }
                    }
                }
                continue;
            }

            let max_len = (n - i).min(6);
            for len in 1..=max_len {
                let sub = &lower[i..i + len];
                if sub.contains('\'') {
                    break;
                }
                if let Some(id) = lookup_syllable(sub) {
                    spans[i].push((i + len, id, 0.0, MatchKind::Exact));
                    for (fid, pen) in ctx.fuzzy.expand(sub) {
                        spans[i].push((i + len, fid, -pen, MatchKind::Fuzzy));
                    }
                }
                if len >= 2 {
                    for (tid, edit) in generate_typo_variants(sub) {
                        if !spans[i]
                            .iter()
                            .any(|&(e, existing, _, _)| e == i + len && existing == tid)
                        {
                            spans[i].push((
                                i + len,
                                tid,
                                -(1.65 + edit.cost()),
                                MatchKind::TypoCorrected,
                            ));
                        }
                    }
                }
            }
        }
        spans
    }

    #[allow(clippy::too_many_arguments)]
    fn populate_mandarin_edges(
        lower: &str,
        syl_edges: &[Vec<(usize, SyllableId, f32, MatchKind)>],
        lexicon: &Lexicon,
        user_store: &UserStore,
        app_bundle_id: &str,
        now_secs: u64,
        is_dev_app: bool,
        edges: &mut [Vec<LatticeEdge>],
    ) {
        let n = lower.len();
        for start in 0..n {
            let mut stack: Vec<(usize, SmallVec<[SyllableId; 4]>, f32, MatchKind)> = Vec::new();
            for &(end1, id1, cost1, kind1) in &syl_edges[start] {
                let mut seq = SmallVec::new();
                seq.push(id1);
                stack.push((end1, seq, cost1, kind1));
            }

            while let Some((cur_end, seq, acc_cost, acc_kind)) = stack.pop() {
                let raw_slice = &lower[start..cur_end];
                let entries: Vec<WordEntry> = lexicon.lookup_exact(&seq);
                for entry in entries.into_iter().take(6) {
                    let tech_bonus = if is_dev_app && entry.is_tech_term {
                        0.45
                    } else {
                        0.0
                    };
                    let user_bonus =
                        user_store.score_boost(raw_slice, &entry.surface, app_bundle_id, now_secs);
                    #[allow(clippy::cast_precision_loss)]
                    let word_len_bonus = ((seq.len().saturating_sub(1)) as f32) * 0.95;
                    edges[start].push(LatticeEdge {
                        end: cur_end,
                        surface: entry.surface,
                        score: entry.log_prob + acc_cost + word_len_bonus + tech_bonus + user_bonus,
                        kind: acc_kind,
                        is_english: false,
                    });
                }

                if seq.len() < 4 {
                    let next_pos = if cur_end < n && lower.as_bytes()[cur_end] == b'\'' {
                        cur_end + 1
                    } else {
                        cur_end
                    };
                    if next_pos < n {
                        for &(end2, id2, cost2, kind2) in &syl_edges[next_pos] {
                            let mut next_seq = seq.clone();
                            next_seq.push(id2);
                            let merged_kind = merge_match_kind(acc_kind, kind2);
                            stack.push((end2, next_seq, acc_cost + cost2, merged_kind));
                        }
                    }
                }
            }
        }
    }

    fn merge_and_prune(target: &mut Vec<BeamState>, incoming: Vec<BeamState>, beam_width: usize) {
        for state in incoming {
            if let Some(existing) = target.iter_mut().find(|s| s.text == state.text) {
                if state.score > existing.score {
                    *existing = state;
                }
            } else {
                target.push(state);
            }
        }
        target.sort_by(|a, b| b.score.total_cmp(&a.score));
        target.truncate(beam_width);
    }
}

const fn combine_beam_kind(prev: MatchKind, next: MatchKind) -> MatchKind {
    if matches!(prev, MatchKind::TypoCorrected) || matches!(next, MatchKind::TypoCorrected) {
        MatchKind::TypoCorrected
    } else if matches!(prev, MatchKind::Fuzzy) || matches!(next, MatchKind::Fuzzy) {
        MatchKind::Fuzzy
    } else if matches!(prev, MatchKind::CodeSwitch) && matches!(next, MatchKind::CodeSwitch) {
        MatchKind::CodeSwitch
    } else {
        MatchKind::Exact
    }
}

const fn merge_match_kind(a: MatchKind, b: MatchKind) -> MatchKind {
    if matches!(a, MatchKind::TypoCorrected) || matches!(b, MatchKind::TypoCorrected) {
        MatchKind::TypoCorrected
    } else if matches!(a, MatchKind::Fuzzy) || matches!(b, MatchKind::Fuzzy) {
        MatchKind::Fuzzy
    } else {
        MatchKind::Exact
    }
}

fn extract_tail_word(left_context: &str) -> String {
    let trimmed = left_context.trim_end();
    if trimmed.is_empty() {
        return String::new();
    }
    if let Some(last_ascii) = trimmed
        .rsplit(|c: char| !c.is_ascii_alphanumeric())
        .next()
        .filter(|s| !s.is_empty())
    {
        return last_ascii.to_string();
    }
    let chars: Vec<char> = trimmed.chars().collect();
    let start = chars.len().saturating_sub(2);
    chars[start..].iter().collect()
}

/// Applies the 60ms Anti-Jitter Freeze Window when an asynchronous Stage-2 rerank arrives.
#[must_use]
pub fn apply_anti_jitter_freeze_window(
    displayed: &[Candidate],
    reranked: Vec<Candidate>,
    elapsed_ms: u64,
    freeze_after_ms: u64,
) -> Vec<Candidate> {
    if elapsed_ms <= freeze_after_ms || displayed.is_empty() || reranked.is_empty() {
        return reranked;
    }
    let frozen_top = &displayed[0];
    if reranked[0].text == frozen_top.text {
        return reranked;
    }

    let mut stabilized = Vec::with_capacity(reranked.len() + 1);
    stabilized.push(frozen_top.clone());
    for cand in reranked {
        if cand.text != frozen_top.text {
            stabilized.push(cand);
        }
    }
    stabilized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_decode_modeless_chinese_english_mixed_sentence() {
        let lex = Lexicon::with_builtin_seed();
        let store = UserStore::new();
        let dec = LatticeDecoder::new();
        let ctx = DecodeContext::default();

        let cands = dec.decode("woshiyongmacoshegithub", &lex, &store, &ctx);
        assert_eq!(
            cands.first().map(|c| c.text.as_str()),
            Some("我使用 macOS 和 GitHub")
        );
        // Should also expose step-by-step prefix word candidates (`我`).
        assert!(
            cands
                .iter()
                .any(|c| c.text == "我" && c.consumed_bytes < 22)
        );
    }

    #[test]
    fn test_should_auto_correct_transposition_and_neighbor_typos() {
        let lex = Lexicon::with_builtin_seed();
        let store = UserStore::new();
        let dec = LatticeDecoder::new();
        let ctx = DecodeContext::default();

        let cands = dec.decode("shaungpin", &lex, &store, &ctx);
        assert_eq!(cands[0].text, "双拼");
        assert_eq!(cands[0].kind, MatchKind::TypoCorrected);

        let cands2 = dec.decode("nohao", &lex, &store, &ctx);
        assert_eq!(cands2[0].text, "你好");
    }

    #[test]
    fn test_should_freeze_first_candidate_after_anti_jitter_window() {
        let initial = vec![
            Candidate {
                text: "亮化".into(),
                score: -4.0,
                kind: MatchKind::Exact,
                consumed_bytes: 8,
            },
            Candidate {
                text: "量化".into(),
                score: -4.2,
                kind: MatchKind::Exact,
                consumed_bytes: 8,
            },
        ];
        let late_rerank = vec![
            Candidate {
                text: "量化".into(),
                score: -2.0,
                kind: MatchKind::Exact,
                consumed_bytes: 8,
            },
            Candidate {
                text: "亮化".into(),
                score: -4.0,
                kind: MatchKind::Exact,
                consumed_bytes: 8,
            },
        ];

        let guarded = apply_anti_jitter_freeze_window(&initial, late_rerank, 75, 60);
        assert_eq!(guarded[0].text, "亮化");
        assert_eq!(guarded[1].text, "量化");
    }
}
