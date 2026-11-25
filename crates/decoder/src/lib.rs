//! Joint dual-track lattice decoder (Pinyin + Fuzzy + Typo + English Code-Switching)
//! with Beam Search, step-by-step prefix word candidates, `NeuralReranker` hook,
//! and 60ms Anti-Jitter Freeze Window.

#![forbid(unsafe_code)]

use std::collections::HashMap;

use chime_lexicon::{Lexicon, UserStore, WordEntry, annotate_chinese_word};
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
    /// Prefix verse, idiom, proverb, or English auto-completion (`tianshengwocai` ->
    /// `天生我材必有用`).
    Completion,
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
    prev_content_word: String,
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

        let (seed_prev_content, seed_last_word) = extract_context_anchors(ctx.left_context);
        let mut beams: Vec<Vec<BeamState>> = vec![Vec::new(); n + 1];
        beams[0].push(BeamState {
            text: String::new(),
            prev_content_word: seed_prev_content,
            last_word: seed_last_word,
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
                    // Combine immediate Bigram (static + online user habit!), Trigram, and
                    // particle-transparent Skip-Bigram across `的/了/在/地`!
                    let direct_bg = lexicon.bigram_bonus(&st.last_word, &edge.surface)
                        + user_store.bigram_boost(&st.last_word, &edge.surface);
                    let skip_bg = if is_transparent_particle(&st.last_word) {
                        (lexicon.bigram_bonus(&st.prev_content_word, &edge.surface)
                            + user_store.bigram_boost(&st.prev_content_word, &edge.surface))
                            * 0.9
                    } else {
                        0.0
                    };
                    let tg =
                        lexicon.trigram_bonus(&st.prev_content_word, &st.last_word, &edge.surface);
                    let next_prev_content = if is_transparent_particle(&edge.surface) {
                        if is_transparent_particle(&st.last_word) {
                            st.prev_content_word.clone()
                        } else {
                            st.last_word.clone()
                        }
                    } else if is_transparent_particle(&st.last_word) {
                        st.prev_content_word.clone()
                    } else {
                        st.last_word.clone()
                    };

                    let boundary_cost = if is_grammatical_affix(&edge.surface)
                        || is_grammatical_affix(&st.last_word)
                    {
                        0.14
                    } else {
                        0.55
                    };

                    next_states.push(BeamState {
                        text: composed,
                        prev_content_word: next_prev_content,
                        last_word: edge.surface.clone(),
                        last_is_english: edge.is_english,
                        score: st.score + edge.score + direct_bg.max(skip_bg) + tg - boundary_cost,
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

    #[allow(clippy::too_many_lines)]
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
        let lower_all = raw_input.to_ascii_lowercase();
        let single_exact_syl = if ctx.scheme == ShuangpinScheme::Full && !raw_input.contains('\'') {
            lookup_syllable(&lower_all)
        } else {
            None
        };
        let is_single_syllable_input = single_exact_syl.is_some();

        let mut full_candidates: Vec<Candidate> = terminal_beams
            .iter()
            .filter(|st| {
                // Never split a valid single syllable (`gen`) into multi-char vowel fragments
                // (`g + en = 个嗯`) when no apostrophe `'` was typed!
                !(is_single_syllable_input && !st.last_is_english && st.text.chars().count() > 1)
            })
            .map(|st| {
                let user_delta = if st.text.chars().count() == 1 && !is_single_syllable_input {
                    0.0
                } else {
                    user_store.score_boost(
                        raw_input,
                        &st.text,
                        ctx.app_bundle_id,
                        ctx.now_epoch_secs,
                    )
                };
                // Demote rare single characters (`哏`, `艮`, `揯`) below top everyday 2-character
                // words (`根据`, `根本`) in the horizontal Top-5 bar; rare chars stay in the `5x5`
                // `↓` character grid!
                let rare_single_pen = if is_single_syllable_input
                    && !st.last_is_english
                    && st.kind == MatchKind::Exact
                    && st.text.chars().count() == 1
                    && st.score < -4.35
                {
                    -3.20
                } else {
                    0.0
                };
                Candidate {
                    text: st.text.clone(),
                    score: st.score + user_delta + rare_single_pen,
                    kind: st.kind,
                    consumed_bytes: n,
                }
            })
            .collect();

        // Include custom user-coined full phrases from L1 Cedar trie!
        for (surf, weight) in user_store.lookup_custom_phrases(&lower_all) {
            if !full_candidates.iter().any(|c| c.text == *surf) {
                full_candidates.push(Candidate {
                    text: surf.clone(),
                    score: -2.0 + (1.0 + weight).ln(),
                    kind: MatchKind::Exact,
                    consumed_bytes: n,
                });
            }
        }

        // On single-syllable inputs (`gen`), populate high-utility 2-syllable everyday words
        // (`根据`, `根本`) into slots #4/#5 instead of cold Unihan single characters!
        if let Some(syl_id) = single_exact_syl {
            for phrase in lexicon.complete_2syl_from_single(syl_id, 3) {
                if !full_candidates.iter().any(|c| c.text == phrase.surface) {
                    let bg = if ctx.left_context.is_empty() {
                        0.0
                    } else {
                        let (_, prev_w) = extract_context_anchors(ctx.left_context);
                        lexicon.bigram_bonus(&prev_w, &phrase.surface) * 0.5
                    };
                    full_candidates.push(Candidate {
                        text: phrase.surface,
                        score: phrase.log_prob - 1.25 + bg,
                        kind: MatchKind::Completion,
                        consumed_bytes: n,
                    });
                }
            }
        }

        // Append English prefix completions (`chatg` -> `ChatGPT`, `kubern` -> `Kubernetes`).
        for comp in lexicon.complete_english_prefix(&lower_all, 3) {
            if !full_candidates.iter().any(|c| c.text == comp.surface) {
                full_candidates.push(Candidate {
                    text: comp.surface,
                    score: comp.log_prob - 0.8,
                    kind: MatchKind::Completion,
                    consumed_bytes: n,
                });
            }
        }

        // Append classical poetry, tongue-twister, idiom & proverb prefix completions
        // (`tianshengwocai` / fuzzy `tianshenwocai` -> `天生我材必有用`, `babaibiaobing` ->
        // `八百标兵奔北坡`) when 3+ syllables are typed!
        if let Some(exact_syls) = extract_greedy_syllables(&lower_all, ctx.scheme)
            && exact_syls.len() >= 3
        {
            for (q_syls, fz_pen) in expand_fuzzy_prefix_queries(&exact_syls, &ctx.fuzzy) {
                for verse in lexicon.complete_pinyin_prefix(&q_syls, 6) {
                    if !full_candidates.iter().any(|c| c.text == verse.surface) {
                        #[allow(clippy::cast_precision_loss)]
                        let comp_score = verse.log_prob
                            + ((q_syls.len().saturating_sub(2)) as f32) * 0.75
                            - fz_pen;
                        full_candidates.push(Candidate {
                            text: verse.surface,
                            score: comp_score,
                            kind: MatchKind::Completion,
                            consumed_bytes: n,
                        });
                    }
                }
            }
        }

        // Context-guided next-verse continuation promotion
        // (`君不见黄河之水天上来，奔流到海不复回。` + `junbujian` -> `#1 君不见高堂明镜悲白发`) AND
        // anti-repeat penalty against repeating an already-typed 5+ character verse verbatim!
        if !ctx.left_context.is_empty() && !is_single_syllable_input {
            if let Some(next_raw) = lexicon.next_clause_completion(ctx.left_context) {
                let clean_next = next_raw.trim_start_matches(|c: char| {
                    matches!(
                        c,
                        '，' | '。' | '！' | '？' | '；' | '：' | ',' | '.' | '!' | '?' | ';'
                    )
                });
                if !clean_next.is_empty()
                    && matches_expected_clause_prefix(&lower_all, clean_next, ctx)
                {
                    if let Some(existing) =
                        full_candidates.iter_mut().find(|c| c.text == clean_next)
                    {
                        existing.score += 7.5;
                    } else {
                        full_candidates.push(Candidate {
                            text: clean_next.to_string(),
                            score: 4.5,
                            kind: MatchKind::Completion,
                            consumed_bytes: n,
                        });
                    }
                }
            }

            for cand in &mut full_candidates {
                if cand.text.chars().count() >= 5 && ctx.left_context.contains(&cand.text) {
                    cand.score -= 5.5;
                }
            }
        }

        full_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        full_candidates.truncate(15);

        if let Some(reranker) = &self.reranker {
            let mut non_comp_indices: Vec<usize> = full_candidates
                .iter()
                .enumerate()
                .filter(|(_, c)| c.kind != MatchKind::Completion)
                .map(|(i, _)| i)
                .take(6)
                .collect();
            if non_comp_indices.len() >= 2 {
                let batch: Vec<Candidate> = non_comp_indices
                    .iter()
                    .map(|&i| full_candidates[i].clone())
                    .collect();
                let bonuses = reranker.score_batch(ctx.left_context, ctx.app_bundle_id, &batch);
                for (idx, bonus) in non_comp_indices.drain(..).zip(bonuses) {
                    full_candidates[idx].score += bonus;
                }
                full_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            }
        }

        // Guarantee the highest-ranked `Completion` candidate (`… ⇥`) sits in the visible Top-2
        // of the compact 1x5 bar on multi-syllable / English prefixes so `Tab` completion is
        // immediately discoverable, while keeping #1-#3 on 1-syllable inputs as primary chars!
        if !is_single_syllable_input {
            promote_top_completion_into_bar(&mut full_candidates);
        }

        // 1) Append multi-character partial prefix word candidates (`consumed_bytes < n`, `chars >=
        //    2`)
        // for step-by-step phrase composition!
        let mut multi_prefix_edges: Vec<&LatticeEdge> = leading_edges
            .iter()
            .filter(|e| e.end < n && !e.is_english && e.surface.chars().count() >= 2)
            .collect();
        multi_prefix_edges
            .sort_by(|a, b| b.end.cmp(&a.end).then_with(|| b.score.total_cmp(&a.score)));

        for pe in multi_prefix_edges.into_iter().take(10) {
            if !full_candidates.iter().any(|c| c.text == pe.surface) {
                full_candidates.push(Candidate {
                    text: pe.surface.clone(),
                    score: pe.score - 6.0,
                    kind: pe.kind,
                    consumed_bytes: pe.end,
                });
            }
        }

        // 2) Always append rich single-character candidates (`chars == 1`, up to 35 chars) for the
        // leading syllable so users can expand the `5x5` grid and pick characters one-by-one!
        let mut single_char_edges: Vec<&LatticeEdge> = leading_edges
            .iter()
            .filter(|e| !e.is_english && e.surface.chars().count() == 1)
            .collect();
        single_char_edges.sort_by(|a, b| {
            let a_exact = u8::from(a.kind == MatchKind::Exact);
            let b_exact = u8::from(b.kind == MatchKind::Exact);
            b_exact
                .cmp(&a_exact)
                .then_with(|| b.score.total_cmp(&a.score))
        });

        for sc in single_char_edges {
            if full_candidates.len() >= 55 {
                break;
            }
            if !full_candidates.iter().any(|c| c.text == sc.surface) {
                full_candidates.push(Candidate {
                    text: sc.surface.clone(),
                    score: sc.score - 9.0,
                    kind: sc.kind,
                    consumed_bytes: sc.end,
                });
            }
        }

        // Always keep raw input fallback so user is never trapped.
        if !full_candidates.iter().any(|c| c.text == raw_input) {
            #[allow(clippy::cast_precision_loss)]
            let fallback_score = -15.0 - 2.5 * (n as f32);
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
        // Never trigger on spans that are already valid canonical single syllables (`gen`, `ban`)!
        if ctx.scheme == ShuangpinScheme::Full {
            for start in 0..n {
                for len in 2..=6.min(n - start) {
                    let sub = &lower[start..start + len];
                    if !sub.bytes().all(|b| b.is_ascii_lowercase())
                        || chime_phonetics::is_valid_syllable(sub)
                    {
                        continue;
                    }
                    for entry in lexicon.lookup_skeleton(sub).into_iter().take(3) {
                        edges[start].push(LatticeEdge {
                            end: start + len,
                            surface: entry.surface,
                            score: entry.log_prob - 1.35,
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
                } else if start == 0
                    && end == n
                    && classify_code_switch(raw_sub) != CodeSwitchSignal::AmbiguousAscii
                {
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

            let max_len = (n - i).min(7);
            for len in 1..=max_len {
                let sub = &lower[i..i + len];
                if sub.contains('\'') {
                    break;
                }
                let is_exact_syl = lookup_syllable(sub);
                if let Some(id) = is_exact_syl {
                    spans[i].push((i + len, id, 0.0, MatchKind::Exact));
                    for (fid, pen) in ctx.fuzzy.expand(sub) {
                        spans[i].push((i + len, fid, -pen, MatchKind::Fuzzy));
                    }
                }
                // Geminate boundary consonant elision (`chunuanhuakai` -> `chun` + `nuan`,
                // `xiniankuaile` -> `xin` + `nian`, `changjiangungunlai` -> `jiang` + `gun`):
                // when a typist presses a shared boundary `n` or `g` once instead of twice!
                if len >= 2
                    && i + len + 1 < n
                    && matches!(bytes[i + len], b'n' | b'g')
                    && matches!(bytes[i + len + 1], b'a' | b'e' | b'i' | b'o' | b'u' | b'v')
                {
                    let extended = &lower[i..=i + len];
                    if !extended.contains('\'')
                        && let Some(gem_id) = lookup_syllable(extended)
                        && !spans[i]
                            .iter()
                            .any(|&(e, existing, _, _)| e == i + len && existing == gem_id)
                    {
                        spans[i].push((i + len, gem_id, -0.85, MatchKind::TypoCorrected));
                    }
                }
                if len >= 2 {
                    // When `sub` is ALREADY a valid syllable (`duo`, `da`, `gen`), penalize typo
                    // neighbors heavily (`-2.75`) so real words win cleanly without blocking 3-syl
                    // lexical typo recovery (`shulufa -> 输入法`); when `sub` is NOT a valid
                    // syllable (`shaung`, `yign` in `mohuyign`), recover fast (`-1.45`) AND allow
                    // one fuzzy rhyme step (`yign -> ying -> yin` -> `模糊音`)!
                    let base_typo_pen = if is_exact_syl.is_some() { 2.75 } else { 1.45 };
                    for (tid, edit) in generate_typo_variants(sub) {
                        let t_cost = -(base_typo_pen + edit.cost());
                        if !spans[i]
                            .iter()
                            .any(|&(e, existing, _, _)| e == i + len && existing == tid)
                        {
                            spans[i].push((i + len, tid, t_cost, MatchKind::TypoCorrected));
                        }
                        if is_exact_syl.is_none() && len >= 3 {
                            let canonical_t = chime_phonetics::canonical_syllable(tid);
                            for (fid, fpen) in ctx.fuzzy.expand(canonical_t) {
                                if !spans[i]
                                    .iter()
                                    .any(|&(e, existing, _, _)| e == i + len && existing == fid)
                                {
                                    spans[i].push((
                                        i + len,
                                        fid,
                                        t_cost - fpen,
                                        MatchKind::TypoCorrected,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        spans
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
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
        type EdgeVisit = (usize, usize, SmallVec<[SyllableId; 16]>, f32, MatchKind);
        let n = lower.len();
        for start in 0..n {
            let mut stack: Vec<EdgeVisit> = Vec::new();
            for &(end1, id1, cost1, kind1) in &syl_edges[start] {
                let mut seq = SmallVec::new();
                seq.push(id1);
                stack.push((end1, start, seq, cost1, kind1));
            }

            while let Some((cur_end, last_syl_start, seq, acc_cost, acc_kind)) = stack.pop() {
                let raw_slice = &lower[start..cur_end];
                let entries: Vec<WordEntry> = lexicon.lookup_exact(&seq);
                let take_limit = if start == 0 && seq.len() == 1 { 32 } else { 8 };
                for entry in entries.into_iter().take(take_limit) {
                    let is_single_char = entry.surface.chars().count() == 1;
                    // Anti-Cold-Word & Single-Syllable Exact Dominance:
                    // 1) Never inject cold fuzzy single characters (`耕`, `梗`, `绠`, `郠`) when
                    //    typing a single syllable (`gen`); keep only ultra-common chars (`更`) with
                    //    a `-1.65` penalty so exact chars (`跟`, `根`) always take #1 and #2!
                    // 2) Suppress 1-char typo guesses on already-valid single-syllable inputs!
                    let single_non_exact_pen = if is_single_char && seq.len() == 1 {
                        match acc_kind {
                            MatchKind::Fuzzy => {
                                if entry.log_prob < -4.15 {
                                    continue;
                                }
                                -1.65
                            }
                            MatchKind::TypoCorrected => {
                                if start == 0
                                    && cur_end == n
                                    && chime_phonetics::is_valid_syllable(raw_slice)
                                {
                                    continue;
                                }
                                if entry.log_prob < -4.30 {
                                    continue;
                                }
                                -0.85
                            }
                            _ => 0.0,
                        }
                    } else {
                        0.0
                    };

                    let tech_bonus = if is_dev_app && entry.is_tech_term {
                        0.45
                    } else {
                        0.0
                    };
                    // Scope single-character personal habits strictly to 1-syllable queries so a
                    // single-char pick (`成`, `和`, `补`) never corrupts multi-syllable words!
                    let user_bonus = if is_single_char && !(start == 0 && cur_end == n) {
                        0.0
                    } else {
                        let raw_boost = user_store.score_boost(
                            raw_slice,
                            &entry.surface,
                            app_bundle_id,
                            now_secs,
                        );
                        if is_single_char {
                            raw_boost.clamp(-1.5, 1.35)
                        } else {
                            raw_boost
                        }
                    };
                    #[allow(clippy::cast_precision_loss)]
                    let word_len_bonus = ((seq.len().saturating_sub(1)).min(3) as f32) * 0.95
                        + (seq.len().saturating_sub(4) as f32) * 1.18;
                    edges[start].push(LatticeEdge {
                        end: cur_end,
                        surface: entry.surface,
                        score: entry.log_prob
                            + acc_cost
                            + single_non_exact_pen
                            + word_len_bonus
                            + tech_bonus
                            + user_bonus,
                        kind: acc_kind,
                        is_english: false,
                    });
                }

                // Allow up to 16-syllable exact idioms, poems, merged Song Ci clauses & tongue
                // twisters (`念去去，千里烟波`, `君不见黄河之水天上来`,
                // `蓦然回首那人却在灯火阑珊处`), gated at every step by $O(\log N)$ prefix
                // existence check!
                if seq.len() < 16 && lexicon.has_longer_pinyin_prefix(&seq) {
                    let has_sep = cur_end < n && lower.as_bytes()[cur_end] == b'\'';
                    let next_pos = if has_sep { cur_end + 1 } else { cur_end };
                    if next_pos < n {
                        for &(end2, id2, cost2, kind2) in &syl_edges[next_pos] {
                            // Penalize splitting a valid single syllable (`jie`, `xian`, `tian`)
                            // across a vowel hiatus (`ji+e` `饥饿`, `ti+an` `提案`) when the user
                            // did NOT type an apostrophe `'`!
                            let hiatus_pen = if !has_sep
                                && chime_phonetics::canonical_syllable(id2)
                                    .bytes()
                                    .next()
                                    .is_some_and(|b| matches!(b, b'a' | b'o' | b'e'))
                                && lookup_syllable(&lower[last_syl_start..end2]).is_some()
                            {
                                -1.35
                            } else {
                                0.0
                            };
                            let mut next_seq = seq.clone();
                            next_seq.push(id2);
                            let merged_kind = merge_match_kind(acc_kind, kind2);
                            stack.push((
                                end2,
                                next_pos,
                                next_seq,
                                acc_cost + cost2 + hiatus_pen,
                                merged_kind,
                            ));
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

fn extract_greedy_syllables(
    lower: &str,
    scheme: ShuangpinScheme,
) -> Option<SmallVec<[SyllableId; 8]>> {
    if lower.is_empty() {
        return None;
    }
    let mut out = SmallVec::new();
    if scheme != ShuangpinScheme::Full {
        let bytes = lower.as_bytes();
        if !bytes.len().is_multiple_of(2) {
            return None;
        }
        let mut i = 0usize;
        while i < bytes.len() {
            let first = expand_shuangpin_pair(scheme, bytes[i], bytes[i + 1])
                .into_iter()
                .next()?;
            out.push(first);
            i += 2;
        }
        return Some(out);
    }

    let n = lower.len();
    let mut pos = 0usize;
    while pos < n {
        if lower.as_bytes()[pos] == b'\'' {
            pos += 1;
            continue;
        }
        let max_len = (n - pos).min(6);
        let mut matched = None;
        for len in (1..=max_len).rev() {
            let sub = &lower[pos..pos + len];
            if !sub.contains('\'')
                && let Some(id) = lookup_syllable(sub)
            {
                // Avoid greedy split failure like `tian` + `g` when `ti` + `ang` is valid:
                let rem = &lower[pos + len..];
                if rem.is_empty() || chime_phonetics::can_segment_exact(rem) {
                    matched = Some((id, len));
                    break;
                }
            }
        }
        let (id, len) = matched?;
        out.push(id);
        pos += len;
    }
    Some(out)
}

/// Promotes the highest-ranked `MatchKind::Completion` candidate into slot #2 (`index 1`) so
/// poetry/proverb/idiom `Tab` completions are always visible on Page 1 of the compact 1x5 bar.
pub fn promote_top_completion_into_bar(candidates: &mut [Candidate]) {
    if let Some(comp_pos) = candidates
        .iter()
        .position(|c| c.kind == MatchKind::Completion)
        && comp_pos >= 2
    {
        candidates[1..=comp_pos].rotate_right(1);
    }
}

fn matches_expected_clause_prefix(
    lower_raw: &str,
    expected_clause: &str,
    ctx: &DecodeContext<'_>,
) -> bool {
    let clean_raw: String = lower_raw
        .bytes()
        .filter(u8::is_ascii_lowercase)
        .map(char::from)
        .collect();
    if clean_raw.len() < 4 {
        return false;
    }
    let cjk_only: String = expected_clause
        .chars()
        .filter(|c| ('\u{4E00}'..='\u{9FA5}').contains(c))
        .collect();
    let Some(py_spaced) = annotate_chinese_word(&cjk_only, &HashMap::new()) else {
        return false;
    };
    let joined_py: String = py_spaced.split_whitespace().collect();
    if ctx.scheme == ShuangpinScheme::Full && joined_py.starts_with(&clean_raw) {
        return true;
    }
    let Some(typed_syls) = extract_greedy_syllables(lower_raw, ctx.scheme) else {
        return false;
    };
    if typed_syls.len() < 2 {
        return false;
    }
    let expected_syls: Vec<SyllableId> = py_spaced
        .split_whitespace()
        .filter_map(lookup_syllable)
        .collect();
    if expected_syls.len() < typed_syls.len() {
        return false;
    }
    expand_fuzzy_prefix_queries(&typed_syls, &ctx.fuzzy)
        .iter()
        .any(|(q, _)| expected_syls.starts_with(q))
}

fn expand_fuzzy_prefix_queries(
    exact: &[SyllableId],
    fuzzy: &FuzzyConfig,
) -> Vec<(SmallVec<[SyllableId; 8]>, f32)> {
    let mut variants: Vec<(SmallVec<[SyllableId; 8]>, f32)> =
        vec![(SmallVec::from_slice(exact), 0.0)];
    for (pos, &syl_id) in exact.iter().enumerate() {
        let canonical = chime_phonetics::canonical_syllable(syl_id);
        for (alt_id, pen) in fuzzy.expand(canonical).into_iter().take(2) {
            if variants.len() >= 6 {
                break;
            }
            let mut copy = SmallVec::from_slice(exact);
            copy[pos] = alt_id;
            variants.push((copy, pen));
        }
    }
    variants
}

fn is_grammatical_affix(word: &str) -> bool {
    matches!(
        word,
        "的" | "地"
            | "得"
            | "了"
            | "着"
            | "过"
            | "中"
            | "里"
            | "上"
            | "下"
            | "内"
            | "外"
            | "前"
            | "后"
            | "时"
            | "们"
            | "不"
            | "没"
            | "最"
            | "很"
            | "太"
            | "更"
            | "挺"
            | "也"
            | "都"
            | "就"
            | "才"
            | "还"
            | "又"
            | "再"
            | "把"
            | "被"
            | "给"
            | "让"
            | "对"
            | "和"
            | "与"
            | "个"
            | "些"
            | "本"
            | "这"
            | "那"
            | "该"
            | "每"
            | "各"
    )
}

fn is_transparent_particle(word: &str) -> bool {
    matches!(
        word,
        "的" | "地"
            | "得"
            | "了"
            | "着"
            | "过"
            | "在"
            | "是"
            | "很"
            | "太"
            | "更"
            | "最"
            | "被"
            | "把"
            | "对"
    )
}

fn extract_context_anchors(left_context: &str) -> (String, String) {
    let trimmed = left_context.trim_end_matches(|c: char| {
        c.is_whitespace() || matches!(c, '，' | '。' | '！' | '？' | '：' | ',' | '.')
    });
    if trimmed.is_empty() {
        return (String::new(), String::new());
    }
    if let Some(last_ascii) = trimmed
        .rsplit(|c: char| !c.is_ascii_alphanumeric())
        .next()
        .filter(|s| !s.is_empty())
    {
        return (String::new(), last_ascii.to_string());
    }

    let chars: Vec<char> = trimmed.chars().collect();
    let n = chars.len();
    let last_ch = chars[n - 1].to_string();
    if is_transparent_particle(&last_ch) && n >= 3 {
        let prev_content: String = chars[n - 3..n - 1].iter().collect();
        return (prev_content, last_ch);
    }
    let last_two: String = chars[n.saturating_sub(2)..].iter().collect();
    let prev_two: String = if n >= 4 {
        chars[n - 4..n - 2].iter().collect()
    } else {
        String::new()
    };
    (prev_two, last_two)
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
