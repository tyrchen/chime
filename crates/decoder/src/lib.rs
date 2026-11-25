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
    /// Treat leading uppercase input (`Hello`, `Mac`) as direct English input.
    pub uppercase_triggers_english: bool,
    /// Automatically recover misspelled English words in standalone and mixed-language spans.
    pub english_auto_correct: bool,
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
            uppercase_triggers_english: true,
            english_auto_correct: true,
            now_epoch_secs: 0,
        }
    }
}

/// Pluggable Stage-2 neural language model reranker (`SafetensorsReranker` or `macOS` 15 `CoreML
/// MLState`).
pub trait NeuralReranker: std::fmt::Debug + Send + Sync {
    /// Scores a batch of complete candidate surface texts given immutable `left_context` and
    /// `app_bundle_id`. Returns one additive log-likelihood score per candidate.
    fn score_batch(
        &self,
        left_context: &str,
        app_bundle_id: &str,
        candidates: &[Candidate],
    ) -> Vec<f32>;

    /// Scores open mid-sentence beam prefixes (disabling right-boundary punctuation closure) to
    /// guide long-sentence beam search without early homophone collapse.
    fn score_open_prefix_batch(
        &self,
        left_context: &str,
        app_bundle_id: &str,
        candidates: &[Candidate],
    ) -> Vec<f32> {
        let _ = (left_context, app_bundle_id);
        vec![0.0; candidates.len()]
    }
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

    /// Updates or clears the attached neural reranker in place.
    pub fn set_reranker(&mut self, reranker: Option<Box<dyn NeuralReranker>>) {
        self.reranker = reranker;
    }

    /// Decodes `raw_input` into ranked full-sentence candidates followed by step-by-step prefix
    /// word candidates.
    #[must_use]
    #[allow(clippy::too_many_lines)]
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

        let is_ms_shuangpin_ing =
            ctx.scheme == ShuangpinScheme::Microsoft && raw_input.contains(';');
        let starts_upper_english =
            ctx.uppercase_triggers_english && raw_input.as_bytes()[0].is_ascii_uppercase();
        if !is_ms_shuangpin_ing
            && (starts_upper_english
                || classify_code_switch(raw_input) == CodeSwitchSignal::HardRawCode)
        {
            let mut out = vec![Candidate {
                text: raw_input.to_string(),
                score: 0.0,
                kind: MatchKind::CodeSwitch,
                consumed_bytes: raw_input.len(),
            }];
            if raw_input.len() >= 2 && raw_input.bytes().all(|b| b.is_ascii_alphabetic()) {
                let lower_q = raw_input.to_ascii_lowercase();
                let mut added_comp = false;
                for comp in lexicon.complete_english_prefix(&lower_q, 3) {
                    if comp.surface != raw_input
                        && (comp.is_tech_term || comp.log_prob > -4.5)
                        && !out.iter().any(|c| c.text == comp.surface)
                    {
                        out.push(Candidate {
                            text: comp.surface,
                            score: comp.log_prob,
                            kind: MatchKind::Completion,
                            consumed_bytes: raw_input.len(),
                        });
                        added_comp = true;
                    }
                }
                if ctx.english_auto_correct
                    && !added_comp
                    && (3..=18).contains(&raw_input.len())
                    && !raw_input[1..].bytes().any(|b| b.is_ascii_uppercase())
                    && !lexicon.is_known_valid_english(&lower_q)
                {
                    for (idx_c, (corr, r_score, _)) in lexicon
                        .correct_english_word(&lower_q, 2)
                        .into_iter()
                        .enumerate()
                    {
                        let surf = format_corrected_english_surface(raw_input, &corr.surface);
                        if surf != raw_input && !out.iter().any(|c| c.text == surf) {
                            let cand = Candidate {
                                text: surf,
                                score: if idx_c == 0 { 0.50 } else { r_score },
                                kind: MatchKind::TypoCorrected,
                                consumed_bytes: raw_input.len(),
                            };
                            if idx_c == 0 {
                                out.insert(0, cand);
                            } else {
                                out.push(cand);
                            }
                        }
                    }
                }
            }
            return out;
        }

        let is_dev_app = UserStore::is_developer_app(ctx.app_bundle_id);
        let edges_by_start = self.build_edges(raw_input, lexicon, user_store, ctx, is_dev_app);
        let n = raw_input.len();
        let checkpoints = select_neural_checkpoints(n);
        let mut next_cp_idx = 0usize;

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
                Self::merge_and_prune(&mut beams[i + 1], carried, 40);
                continue;
            }

            // Mid-sentence open-prefix neural beam guidance on long sentences (`>= 20` bytes):
            // wait until a genuine multi-syllable word boundary (never firing mid-word when an
            // already-populated span `j > i` has a stronger per-byte score!) and rescore up to 12
            // competing prefixes while shifting unscored tail beams down proportionately.
            if let Some(&cp_target) = checkpoints.get(next_cp_idx)
                && i >= cp_target
                && !edges_by_start[i].is_empty()
            {
                #[allow(clippy::cast_precision_loss)]
                let cur_rate = beams[i][0].score / (i.max(1) as f32);
                let has_better_span_ahead = ((i + 1)..=n.min(i + 6)).any(|j| {
                    #[allow(clippy::cast_precision_loss)]
                    let ahead_ok = !beams[j].is_empty()
                        && (!edges_by_start.get(j).is_none_or(Vec::is_empty) || j == n)
                        && (beams[j][0].score / (j as f32)) > cur_rate + 0.05;
                    ahead_ok
                });
                if !has_better_span_ahead {
                    next_cp_idx += 1;
                    if let Some(reranker) = &self.reranker
                        && beams[i].len() >= 2
                    {
                        let take_k = beams[i].len().min(12);
                        let prefix_batch: Vec<Candidate> = beams[i][..take_k]
                            .iter()
                            .map(|st| Candidate {
                                text: st.text.clone(),
                                score: st.score,
                                kind: st.kind,
                                consumed_bytes: i,
                            })
                            .collect();
                        let deltas = reranker.score_open_prefix_batch(
                            ctx.left_context,
                            ctx.app_bundle_id,
                            &prefix_batch,
                        );
                        if deltas.iter().any(|&d| d.abs() > 1e-4) {
                            let env_before = beams[i][0].score;
                            let min_delta = deltas.iter().copied().fold(0.0_f32, f32::min);
                            let mut env_after = f32::NEG_INFINITY;
                            for (st, delta) in beams[i][..take_k].iter_mut().zip(deltas) {
                                st.score += delta;
                                env_after = env_after.max(st.score);
                            }
                            let tail_shift = (min_delta - 0.35).min(-0.35);
                            for st in &mut beams[i][take_k..] {
                                st.score += tail_shift;
                            }
                            // DAG potential-preserving anchor: never inflate `beams[i]` above
                            // `env_before`, otherwise paths splitting at vertex `i` (`容易到+至`)
                            // gain free nats over multi-syllable words straddling `i` (`导致`)!
                            let drift = env_after - env_before;
                            if drift.is_finite() {
                                for st in &mut beams[i] {
                                    st.score -= drift;
                                }
                            }
                            beams[i].sort_by(|a, b| b.score.total_cmp(&a.score));
                        }
                    }
                }
            }

            let current_states = beams[i].clone();
            for edge in &edges_by_start[i] {
                let mut next_states = Vec::with_capacity(current_states.len());
                let is_utterance_end = edge.end == n;
                for st in &current_states {
                    let mut composed = st.text.clone();
                    if !composed.is_empty()
                        && ctx.pangu_spacing
                        && (st.last_is_english || edge.is_english)
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

                    let clitic_bonus =
                        prosodic_clitic_cohesion(&st.last_word, &edge.surface, st.kind, edge.kind);
                    let boundary_cost = if clitic_bonus > 0.0 {
                        0.0
                    } else if is_grammatical_affix(&edge.surface)
                        || is_grammatical_affix(&st.last_word)
                    {
                        0.08
                    } else {
                        0.40
                    };
                    let syntax_pen = syntactic_transition_penalty(
                        &st.prev_content_word,
                        &st.last_word,
                        &edge.surface,
                        edge.kind,
                        is_utterance_end,
                        n,
                    );

                    next_states.push(BeamState {
                        text: composed,
                        prev_content_word: next_prev_content,
                        last_word: edge.surface.clone(),
                        last_is_english: edge.is_english,
                        score: st.score + edge.score + direct_bg.max(skip_bg) + tg - boundary_cost
                            + clitic_bonus
                            + syntax_pen,
                        kind: next_kind,
                    });
                }
                Self::merge_and_prune(&mut beams[edge.end], next_states, 40);
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

        // Append curated English/tech prefix completions (`chatg` -> `ChatGPT`, `kubern` ->
        // `Kubernetes`), ignoring low-prior `/usr/share/dict/words` entries (`mingelen`) so valid
        // Mandarin words (`minge` -> `名额`) are never displaced!
        if lower_all.len() >= 4 {
            for comp in lexicon.complete_english_prefix(&lower_all, 4) {
                if (comp.is_tech_term || comp.log_prob > -4.5)
                    && !full_candidates.iter().any(|c| c.text == comp.surface)
                {
                    full_candidates.push(Candidate {
                        text: comp.surface,
                        score: comp.log_prob - 0.8,
                        kind: MatchKind::Completion,
                        consumed_bytes: n,
                    });
                }
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
        full_candidates.truncate(36);

        if let Some(reranker) = &self.reranker {
            let best_non_comp = full_candidates
                .iter()
                .find(|c| c.kind != MatchKind::Completion)
                .map_or(f32::NEG_INFINITY, |c| c.score);
            let mut non_comp_indices: Vec<usize> = full_candidates
                .iter()
                .enumerate()
                .filter(|(_, c)| c.kind != MatchKind::Completion && c.score >= best_non_comp - 6.8)
                .map(|(i, _)| i)
                .take(22)
                .collect();
            if non_comp_indices.len() >= 2 {
                let env_before = non_comp_indices
                    .iter()
                    .map(|&i| full_candidates[i].score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let batch: Vec<Candidate> = non_comp_indices
                    .iter()
                    .map(|&i| full_candidates[i].clone())
                    .collect();
                let bonuses = reranker.score_batch(ctx.left_context, ctx.app_bundle_id, &batch);
                let min_bonus = bonuses.iter().copied().fold(0.0_f32, f32::min);
                let mut scored_mask = vec![false; full_candidates.len()];
                let mut env_after = f32::NEG_INFINITY;
                for (&idx, &bonus) in non_comp_indices.iter().zip(bonuses.iter()) {
                    full_candidates[idx].score += bonus;
                    env_after = env_after.max(full_candidates[idx].score);
                    scored_mask[idx] = true;
                }
                let unscored_shift = (min_bonus - 0.40).min(-0.40);
                for (idx, cand) in full_candidates.iter_mut().enumerate() {
                    if !scored_mask[idx] && cand.kind != MatchKind::Completion {
                        cand.score += unscored_shift;
                    }
                }
                // Envelope-preserving calibration: keep exact pairwise differences among all
                // non-completion candidates while anchoring their maximum score to `env_before` so
                // Stage-2 zero-mean shifts never inflate above `Completion` candidates!
                let drift = env_after - env_before;
                if drift.is_finite() {
                    for cand in &mut full_candidates {
                        if cand.kind != MatchKind::Completion {
                            cand.score -= drift;
                        }
                    }
                }
                non_comp_indices.clear();
                full_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            }
        }

        if has_explicit_feminine_antecedent(ctx.left_context) {
            promote_or_synthesize_pronoun_twin(&mut full_candidates, n, '他', '她', 0.35);
        } else if !has_feminine_or_neuter_cue(ctx.left_context) {
            apply_unmarked_pronoun_symmetry(&mut full_candidates, n);
        }

        // Slot-1 Exactness Invariant: `MatchKind::Completion` represents untyped future syllables
        // for `Tab` auto-completion, never usurping Slot #1 (`Space` key) when a full-span match
        // for the actually typed syllables exists (unless explicitly promoted by left-context
        // couplet continuation `score > 2.0`)!
        if let Some((top_full, top_kind)) = full_candidates
            .iter()
            .find(|c| c.kind != MatchKind::Completion)
            .map(|c| (c.score, c.kind))
        {
            let mut clamped_comp = false;
            for cand in &mut full_candidates {
                let full_is_plausible =
                    top_kind == MatchKind::Exact || top_full > cand.score - 2.20;
                if full_is_plausible
                    && cand.kind == MatchKind::Completion
                    && cand.score < 2.0
                    && cand.score >= top_full
                {
                    cand.score = top_full - 0.05;
                    clamped_comp = true;
                }
            }
            if clamped_comp {
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

        // Super-abbreviation initial skeletons (`zgr` -> `中国人`, `wsm` -> `为什么`, `bzd` ->
        // `不知道`): only trigger on whole-input abbreviation queries (or explicit `'`-delimited
        // spans), NEVER mid-syllable inside a valid full-Pinyin sentence where nasal codas (`ng` +
        // `d/z/s/c/l/h`) would otherwise hallucinate `戈达尔(gde)` / `工作(gz)` / `能够(ng)`!
        if ctx.scheme == ShuangpinScheme::Full {
            let full_is_canonical = n >= 4 && chime_phonetics::can_segment_exact(&lower);
            for start in 0..n {
                let left_delimited = start == 0 || lower.as_bytes()[start - 1] == b'\'';
                if full_is_canonical && start > 0 {
                    break;
                }
                for len in 2..=6.min(n - start) {
                    let end = start + len;
                    let right_delimited = end == n || lower.as_bytes()[end] == b'\'';
                    if full_is_canonical && (!left_delimited || !right_delimited) {
                        continue;
                    }
                    let sub = &lower[start..end];
                    if !sub.bytes().all(|b| b.is_ascii_lowercase())
                        || chime_phonetics::is_valid_syllable(sub)
                    {
                        continue;
                    }
                    for entry in lexicon.lookup_skeleton(sub).into_iter().take(3) {
                        edges[start].push(LatticeEdge {
                            end,
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
        let full_is_canonical =
            ctx.scheme == ShuangpinScheme::Full && chime_phonetics::can_segment_exact(&lower);
        for start in 0..n {
            for end in (start + 2)..=n {
                let raw_sub = &raw_input[start..end];
                let lower_sub = &lower[start..end];
                if ctx.scheme != ShuangpinScheme::Full && raw_sub == lower_sub && (end - start) <= 4
                {
                    continue;
                }
                if let Some(eng) = lexicon.lookup_english(lower_sub) {
                    let is_generic_lower =
                        !eng.is_tech_term && eng.surface == lower_sub && raw_sub == lower_sub;
                    // Suppress generic lowercase English words (`and`, `any`, `chile`, `wang`,
                    // `song`) from tearing apart Shuangpin pairs or internal Mandarin syllables
                    // (`jiandan` -> `ji + and + an`) when the user typed no uppercase cue and the
                    // word is neither a tech term nor proper-cased!
                    if is_generic_lower {
                        if ctx.scheme != ShuangpinScheme::Full {
                            continue;
                        }
                        if !(start == 0 && end == n)
                            && (full_is_canonical
                                || (end - start) < 5
                                || chime_phonetics::can_segment_exact(lower_sub))
                        {
                            continue;
                        }
                    }
                    // On standalone canonical Pinyin inputs (`women` -> `我们`, `name` -> `那么`,
                    // `like` -> `立刻`), cap non-tech lowercase English homographs below Chinese
                    // words while keeping their true Zipf priors for `En` mode & typo recovery.
                    let base_eng_lp = if start == 0
                        && end == n
                        && !eng.is_tech_term
                        && eng.surface == lower_sub
                        && raw_sub == lower_sub
                        && chime_phonetics::can_segment_exact(lower_sub)
                    {
                        eng.log_prob.min(-5.80)
                    } else {
                        eng.log_prob
                    };
                    let dev_bonus = if is_dev_app && eng.is_tech_term {
                        0.95
                    } else {
                        0.0
                    };
                    let phonotactic_bonus = if !chime_phonetics::can_segment_exact(lower_sub)
                        && classify_code_switch(raw_sub) == CodeSwitchSignal::PhonotacticEnglish
                    {
                        1.2
                    } else {
                        0.0
                    };
                    edges[start].push(LatticeEdge {
                        end,
                        surface: eng.surface,
                        score: base_eng_lp + dev_bonus + phonotactic_bonus,
                        kind: MatchKind::CodeSwitch,
                        is_english: true,
                    });
                } else if ctx.scheme == ShuangpinScheme::Full {
                    Self::populate_english_typo_edges(
                        raw_sub,
                        lower_sub,
                        start,
                        end,
                        n,
                        lexicon,
                        ctx.english_auto_correct,
                        is_dev_app,
                        &mut edges[start],
                    );
                    if start == 0
                        && end == n
                        && classify_code_switch(raw_sub) != CodeSwitchSignal::AmbiguousAscii
                    {
                        #[allow(clippy::cast_precision_loss)]
                        let per_char_cost = if raw_sub == lower_sub {
                            -3.20 - 1.95 * (raw_sub.len() as f32)
                        } else {
                            -1.95 * (raw_sub.len() as f32)
                        };
                        edges[start].push(LatticeEdge {
                            end,
                            surface: raw_sub.to_string(),
                            score: per_char_cost,
                            kind: MatchKind::CodeSwitch,
                            is_english: true,
                        });
                    }
                } else if start == 0
                    && end == n
                    && classify_code_switch(raw_sub) != CodeSwitchSignal::AmbiguousAscii
                {
                    #[allow(clippy::cast_precision_loss)]
                    let per_char_cost = if raw_sub == lower_sub {
                        -3.20 - 1.95 * (raw_sub.len() as f32)
                    } else {
                        -1.95 * (raw_sub.len() as f32)
                    };
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

    #[allow(clippy::too_many_arguments)]
    fn populate_english_typo_edges(
        raw_sub: &str,
        lower_sub: &str,
        start: usize,
        end: usize,
        total_len: usize,
        lexicon: &Lexicon,
        english_auto_correct: bool,
        is_dev_app: bool,
        out_edges: &mut Vec<LatticeEdge>,
    ) {
        let span_len = end - start;
        if !english_auto_correct
            || !(4..=18).contains(&span_len)
            || !lower_sub.bytes().all(|b| b.is_ascii_lowercase())
            || chime_phonetics::can_segment_exact(lower_sub)
            || lexicon.is_known_valid_english(lower_sub)
        {
            return;
        }

        let is_full_query = start == 0 && end == total_len;
        if is_full_query {
            // Do not hijack clean prefixes of longer English words (`pytho` -> `Python ⇥`).
            if lexicon
                .complete_english_prefix(lower_sub, 1)
                .into_iter()
                .any(|c| c.is_tech_term || c.log_prob > -4.5)
            {
                return;
            }
            for (corr, ranked_score, _) in lexicon.correct_english_word(lower_sub, 2) {
                let corr_lower = corr.surface.to_ascii_lowercase();
                if !corr.is_tech_term
                    && corr.surface == corr_lower
                    && chime_phonetics::can_segment_exact(&corr_lower)
                {
                    continue;
                }
                let dev_bonus = if is_dev_app && corr.is_tech_term {
                    0.85
                } else {
                    0.0
                };
                out_edges.push(LatticeEdge {
                    end,
                    surface: format_corrected_english_surface(raw_sub, &corr.surface),
                    score: ranked_score - 0.25 + dev_bonus,
                    kind: MatchKind::TypoCorrected,
                    is_english: true,
                });
            }
        } else if (5..=14).contains(&span_len)
            && classify_code_switch(raw_sub) == CodeSwitchSignal::PhonotacticEnglish
        {
            for (corr, ranked_score, _) in lexicon.correct_english_word(lower_sub, 1) {
                if corr.log_prob < -4.25 {
                    continue;
                }
                let corr_lower = corr.surface.to_ascii_lowercase();
                if !corr.is_tech_term
                    && corr.surface == corr_lower
                    && chime_phonetics::can_segment_exact(&corr_lower)
                {
                    continue;
                }
                let dev_bonus = if is_dev_app && corr.is_tech_term {
                    0.85
                } else {
                    0.0
                };
                out_edges.push(LatticeEdge {
                    end,
                    surface: format_corrected_english_surface(raw_sub, &corr.surface),
                    score: ranked_score + 0.95 + dev_bonus,
                    kind: MatchKind::TypoCorrected,
                    is_english: true,
                });
            }
        }
    }

    fn build_syllable_spans(
        lower: &str,
        ctx: &DecodeContext<'_>,
    ) -> Vec<Vec<(usize, SyllableId, f32, MatchKind)>> {
        let n = lower.len();
        let bytes = lower.as_bytes();
        let full_already_exact = n >= 15 && chime_phonetics::can_segment_exact(lower);
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
                // GB/T 16159 Phonotactic Integrity:
                // 1) Bare nasal interjections `n`/`m`/`ng` (`嗯`/`呣`) only occur as standalone
                //    queries or `'`-delimited syllables, never splitting nasal codas (`zhen` ->
                //    `zhe+n` `这嗯`, `yijian` -> `yijia+n` `一家嗯`)!
                if matches!(sub, "n" | "m" | "ng") {
                    let left_ok = i == 0 || bytes[i - 1] == b'\'';
                    let right_ok = i + len == n || bytes[i + len] == b'\'';
                    if !(left_ok && right_ok && n <= 3) {
                        continue;
                    }
                }
                // 2) Penalize splitting a valid preceding syllable across an un-apostrophized
                //    1-letter vowel (`jihuo` -> `ji+hu+o` `几乎哦`, `yijia` -> `yi+ji+a` `一级啊`,
                //    `xiaorenwu` -> `xia+o` `下哦`)!
                let cross_vowel_pen = if len == 1
                    && i > 0
                    && bytes[i - 1] != b'\''
                    && matches!(bytes[i], b'a' | b'o' | b'e')
                    && (1..=3.min(i)).any(|back| {
                        !lower[i - back..=i].contains('\'')
                            && lookup_syllable(&lower[i - back..=i]).is_some()
                    }) {
                    if i + 1 < n { -3.40 } else { -1.80 }
                } else {
                    0.0
                };
                let is_exact_syl = lookup_syllable(sub);
                if let Some(id) = is_exact_syl {
                    spans[i].push((i + len, id, cross_vowel_pen, MatchKind::Exact));
                }
                if cross_vowel_pen == 0.0 && (is_exact_syl.is_some() || len >= 2) {
                    for (fid, pen) in ctx.fuzzy.expand(sub) {
                        if !spans[i]
                            .iter()
                            .any(|&(e, existing, _, _)| e == i + len && existing == fid)
                        {
                            spans[i].push((i + len, fid, -pen, MatchKind::Fuzzy));
                        }
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
                if len >= 2 && !full_already_exact {
                    Self::expand_typo_spans(
                        sub,
                        i + len,
                        is_exact_syl.is_some(),
                        ctx,
                        &mut spans[i],
                    );
                }
            }
        }
        spans
    }

    fn expand_typo_spans(
        sub: &str,
        end: usize,
        is_exact_syl: bool,
        ctx: &DecodeContext<'_>,
        out: &mut Vec<(usize, SyllableId, f32, MatchKind)>,
    ) {
        if !is_exact_syl && starts_with_tech_acronym_plus_pinyin(sub) {
            return;
        }
        let sub_already_segments = !is_exact_syl && chime_phonetics::can_segment_exact(sub);
        let base_typo_pen = if is_exact_syl { 2.75 } else { 1.45 };
        let typo_cap = if is_exact_syl { 8 } else { 16 };
        for (tid, edit) in generate_typo_variants(sub).into_iter().take(typo_cap) {
            if sub_already_segments && matches!(edit, chime_phonetics::TypoEdit::NeighborKey { .. })
            {
                continue;
            }
            let t_cost = -(base_typo_pen + edit.cost());
            if !out
                .iter()
                .any(|&(e, existing, _, _)| e == end && existing == tid)
            {
                out.push((end, tid, t_cost, MatchKind::TypoCorrected));
            }
            if !is_exact_syl && sub.len() >= 3 {
                let canonical_t = chime_phonetics::canonical_syllable(tid);
                for (fid, fpen) in ctx.fuzzy.expand(canonical_t) {
                    if !out
                        .iter()
                        .any(|&(e, existing, _, _)| e == end && existing == fid)
                    {
                        out.push((end, fid, t_cost - fpen, MatchKind::TypoCorrected));
                    }
                }
            }
        }
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
        let full_exact_seg = n >= 6 && chime_phonetics::can_segment_exact(lower);
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
                // High-density Chinese syllables (`ji`, `zhi`, `shi`, `yi`, `du`, `ban`, `ping`)
                // have 15-25 everyday single characters; keep 22 exact single-char candidates at
                // internal positions so characters ranked #9-#22 (`堵`, `朵`, `浇`, `瓶`, `搬`,
                // `寄`, `帧`, `熵`) are never dropped before beam search!
                let take_limit = if start == 0 && seq.len() == 1 {
                    34
                } else if seq.len() == 1 {
                    if acc_kind == MatchKind::Exact { 26 } else { 4 }
                } else {
                    8
                };
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
                                let min_lp = if start == 0 && cur_end == n {
                                    -4.15
                                } else {
                                    -3.85
                                };
                                if entry.log_prob < min_lp {
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
                                if entry.log_prob < -4.15 {
                                    continue;
                                }
                                -0.95
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
                    // Information-theoretic frequency-calibrated lexical cohesion bonus:
                    // Genuine common compounds (`现在 -3.3`, `方案 -3.7`, `安静 -4.3`) receive
                    // strong multi-syllable cohesion, while rare proper names (`李德 -5.45`, `得数
                    // -5.50`, `龋齿 -6.10`, `童一剑 -6.42`) receive near-zero bonus so they never
                    // hijack adjacent grammatical clitics (`里 + 的`, `去 + 吃`, `同 + 一间`)!
                    #[allow(clippy::cast_precision_loss)]
                    let word_len_bonus = match seq.len() {
                        0 | 1 => 0.0,
                        2 | 3 => {
                            let cohesion = ((entry.log_prob + 5.55) / 1.65).clamp(0.05, 1.0);
                            let embedded_non_exact =
                                if !(start == 0 && cur_end == n) && acc_kind != MatchKind::Exact {
                                    if full_exact_seg { -1.35 } else { -0.45 }
                                } else {
                                    0.0
                                };
                            ((seq.len() - 1) as f32) * 0.72 * cohesion + embedded_non_exact
                        }
                        len => {
                            let base = 2.05 + ((len - 4) as f32) * 1.08;
                            if !(start == 0 && cur_end == n) && acc_kind != MatchKind::Exact {
                                base - 1.65
                            } else {
                                base
                            }
                        }
                    };
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
                            // Never chain multiple motor-slip typos inside a single word (`O(K^2)`
                            // explosion)!
                            if acc_kind == MatchKind::TypoCorrected
                                && kind2 == MatchKind::TypoCorrected
                            {
                                continue;
                            }
                            // Mild hiatus preference (`-0.35`) when a multi-syllable dictionary
                            // word splits a potential single syllable (`xian` -> `西安`, `jie` ->
                            // `饥饿`) without an apostrophe, so single-syllable words (`先`, `姐`)
                            // win standalone while established compounds (`去西安旅行`, `忍受饥饿`)
                            // remain competitive in context!
                            let hiatus_pen = if !has_sep
                                && chime_phonetics::canonical_syllable(id2)
                                    .bytes()
                                    .next()
                                    .is_some_and(|b| matches!(b, b'a' | b'o' | b'e'))
                                && lookup_syllable(&lower[last_syl_start..end2]).is_some()
                            {
                                -0.35
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
        if target.len() <= beam_width {
            return;
        }

        // Multi-scale diverse beam pruning (prevents mid-sentence homophone fan-out from evicting
        // competing earlier interpretations before right-context arrives):
        // 1) Cap any 2-char opening (`head2`) at 10 slots;
        // 2) Cap any `(char_len - 2)` prefix (`pre2`, when `char_len >= 5`) at 4 slots so a single
        //    mid-sentence choice (`...主宰` vs `...住在`, `...飞刀` vs `...飞到`) cannot spawn 10
        //    2-char children and monopolize the beam;
        // 3) Cap `(char_len - 1)` `stem` at 5 variants for the #1 stem and 3 for secondary stems!
        let mut kept = Vec::with_capacity(beam_width);
        let mut overflow = Vec::new();
        let mut stem_counts: HashMap<String, usize> = HashMap::new();
        let mut pre2_counts: HashMap<String, usize> = HashMap::new();
        let mut head_counts: HashMap<String, usize> = HashMap::new();
        let mut top_stem: Option<String> = None;
        let mut top_pre2: Option<String> = None;

        for state in target.drain(..) {
            if kept.len() >= beam_width {
                break;
            }
            let char_count = state.text.chars().count();
            if char_count >= 4 {
                let head2: String = state.text.chars().take(2).collect();
                let hc = head_counts.entry(head2).or_insert(0);
                if *hc >= 12 {
                    overflow.push(state);
                    continue;
                }
                *hc += 1;
            }
            if char_count >= 5 {
                let pre2: String = state.text.chars().take(char_count - 2).collect();
                if top_pre2.is_none() {
                    top_pre2 = Some(pre2.clone());
                }
                let pre2_cap = if top_pre2.as_deref() == Some(pre2.as_str()) {
                    8
                } else {
                    3
                };
                let pc = pre2_counts.entry(pre2).or_insert(0);
                if *pc >= pre2_cap {
                    overflow.push(state);
                    continue;
                }
                *pc += 1;
            }
            if char_count >= 3 {
                let stem: String = state.text.chars().take(char_count - 1).collect();
                if top_stem.is_none() {
                    top_stem = Some(stem.clone());
                }
                let cap = if top_stem.as_deref() == Some(stem.as_str()) {
                    8
                } else if char_count <= 4 {
                    1
                } else {
                    2
                };
                let count = stem_counts.entry(stem).or_insert(0);
                if *count >= cap {
                    overflow.push(state);
                    continue;
                }
                *count += 1;
            }
            kept.push(state);
        }

        for state in overflow {
            if kept.len() >= beam_width {
                break;
            }
            kept.push(state);
        }
        *target = kept;
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
            | "吧"
            | "吗"
            | "呢"
            | "啊"
            | "呀"
            | "啦"
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

fn is_numeral_or_dem_classifier(word: &str) -> bool {
    let mut chars = word.chars();
    let (Some(c1), Some(c2), None) = (chars.next(), chars.next(), chars.next()) else {
        return false;
    };
    matches!(
        c1,
        '一' | '二'
            | '两'
            | '三'
            | '四'
            | '五'
            | '六'
            | '七'
            | '八'
            | '九'
            | '十'
            | '百'
            | '千'
            | '几'
            | '每'
            | '这'
            | '那'
            | '哪'
            | '半'
            | '各'
            | '整'
            | '首'
    ) && matches!(
        c2,
        '个' | '位'
            | '只'
            | '条'
            | '张'
            | '件'
            | '种'
            | '块'
            | '杯'
            | '本'
            | '部'
            | '台'
            | '碗'
            | '次'
            | '场'
            | '段'
            | '套'
            | '份'
            | '家'
            | '双'
            | '顿'
            | '朵'
            | '颗'
            | '棵'
            | '辆'
            | '座'
            | '篇'
            | '层'
            | '句'
            | '步'
            | '名'
            | '员'
            | '把'
            | '副'
            | '幅'
            | '首'
            | '封'
            | '顶'
            | '支'
            | '束'
            | '盘'
            | '壶'
            | '粒'
            | '栋'
            | '门'
            | '项'
            | '笔'
            | '间'
            | '瓶'
            | '盒'
            | '页'
            | '格'
    )
}

fn is_non_nominal_functor(word: &str) -> bool {
    matches!(
        word,
        "并" | "且"
            | "或"
            | "及"
            | "与"
            | "而"
            | "但"
            | "却"
            | "把"
            | "被"
            | "从"
            | "向"
            | "往"
            | "替"
            | "了"
            | "着"
            | "过"
            | "也"
            | "都"
            | "就"
            | "才"
            | "又"
            | "还"
            | "再"
            | "更"
    )
}

fn is_dangling_final_functor(word: &str) -> bool {
    matches!(
        word,
        "把" | "被" | "并" | "且" | "及" | "与" | "从" | "往" | "向" | "对于" | "关于"
    )
}

fn has_feminine_or_neuter_cue(left_context: &str) -> bool {
    left_context.chars().any(|c| {
        matches!(
            c,
            '她' | '女'
                | '妈'
                | '姐'
                | '妹'
                | '姑'
                | '娘'
                | '妻'
                | '婆'
                | '姨'
                | '嫂'
                | '妇'
                | '母'
                | '护'
                | '闺'
                | '它'
                | '猫'
                | '狗'
        )
    })
}

fn has_explicit_feminine_antecedent(left_context: &str) -> bool {
    let last_fem = left_context.rfind(|c| {
        matches!(
            c,
            '她' | '女'
                | '妈'
                | '姐'
                | '妹'
                | '姑'
                | '娘'
                | '妻'
                | '婆'
                | '姨'
                | '嫂'
                | '妇'
                | '母'
        )
    });
    let last_masc = left_context.rfind(|c| {
        matches!(
            c,
            '他' | '男' | '爸' | '父' | '哥' | '弟' | '爷' | '叔' | '伯' | '舅' | '兄'
        )
    });
    match (last_fem, last_masc) {
        (Some(f), Some(m)) => f > m,
        (Some(_), None) => true,
        _ => left_context.contains("护士") || left_context.contains("闺蜜"),
    }
}

fn promote_or_synthesize_pronoun_twin(
    candidates: &mut Vec<Candidate>,
    full_bytes: usize,
    from_ch: char,
    to_ch: char,
    margin: f32,
) {
    let mut additions: Vec<Candidate> = Vec::new();
    for i in 0..candidates.len().min(6) {
        if candidates[i].consumed_bytes == full_bytes && candidates[i].text.contains(from_ch) {
            let twin_text: String = candidates[i]
                .text
                .chars()
                .map(|ch| if ch == from_ch { to_ch } else { ch })
                .collect();
            let target_score = candidates[i].score + margin;
            if let Some(existing) = candidates
                .iter_mut()
                .find(|c| c.consumed_bytes == full_bytes && c.text == twin_text)
            {
                if existing.score < target_score {
                    existing.score = target_score;
                }
            } else {
                additions.push(Candidate {
                    text: twin_text,
                    score: target_score,
                    kind: candidates[i].kind,
                    consumed_bytes: full_bytes,
                });
            }
        }
    }
    candidates.extend(additions);
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
}

fn has_inherent_feminine_predicate(text: &str) -> bool {
    text.contains("妈妈")
        || text.contains("母亲")
        || text.contains("姐姐")
        || text.contains("妹妹")
        || text.contains("姑娘")
        || text.contains("妻子")
        || text.contains("老婆")
        || text.contains("阿姨")
        || text.contains("奶奶")
        || text.contains("外婆")
        || text.contains("女生")
        || text.contains("女孩")
        || text.contains("女士")
}

/// Ranks unmarked masculine `他` immediately ahead of its exact character-minimal-pair `她`/`它`
/// twin when `left_context` carries no feminine or neuter cue.
pub fn apply_unmarked_pronoun_symmetry(candidates: &mut Vec<Candidate>, full_bytes: usize) {
    let mut additions: Vec<Candidate> = Vec::new();
    for i in 0..candidates.len().min(6) {
        if candidates[i].consumed_bytes == full_bytes
            && candidates[i].text.contains('她')
            && !has_inherent_feminine_predicate(&candidates[i].text)
        {
            let masc_text: String = candidates[i]
                .text
                .chars()
                .map(|ch| if ch == '她' { '他' } else { ch })
                .collect();
            if !candidates
                .iter()
                .any(|c| c.consumed_bytes == full_bytes && c.text == masc_text)
            {
                additions.push(Candidate {
                    text: masc_text,
                    score: candidates[i].score + 0.15,
                    kind: candidates[i].kind,
                    consumed_bytes: full_bytes,
                });
            }
        }
    }
    candidates.extend(additions);

    let mut masc_scores: HashMap<String, f32> = HashMap::new();
    for c in candidates.iter() {
        if c.consumed_bytes == full_bytes && c.text.contains('他') {
            masc_scores
                .entry(c.text.clone())
                .and_modify(|s| *s = s.max(c.score))
                .or_insert(c.score);
        }
    }
    if masc_scores.is_empty() {
        return;
    }
    let mut changed = false;
    for c in candidates.iter_mut() {
        if c.consumed_bytes == full_bytes
            && (c.text.contains('她') || c.text.contains('它'))
            && !has_inherent_feminine_predicate(&c.text)
        {
            let is_neuter = c.text.contains('它');
            let masc_key: String = c
                .text
                .chars()
                .map(|ch| {
                    if matches!(ch, '她' | '它') {
                        '他'
                    } else {
                        ch
                    }
                })
                .collect();
            if let Some(&m_score) = masc_scores.get(&masc_key)
                && c.score - m_score <= 3.5
            {
                let margin = if is_neuter { 0.30 } else { 0.15 };
                if c.score >= m_score - margin {
                    c.score = m_score - margin;
                    changed = true;
                }
            }
        }
    }
    if changed {
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    }
}

fn starts_with_tech_acronym_plus_pinyin(sub: &str) -> bool {
    for acr in [
        "aws", "cpu", "gpu", "api", "ssh", "sql", "ios", "llm", "sdk", "ide", "pdf",
    ] {
        if let Some(rem) = sub.strip_prefix(acr)
            && !rem.is_empty()
            && chime_phonetics::can_segment_exact(rem)
        {
            return true;
        }
    }
    false
}

fn select_neural_checkpoints(total_bytes: usize) -> SmallVec<[usize; 3]> {
    let mut out = SmallVec::new();
    if total_bytes >= 36 {
        out.push(total_bytes / 3);
        out.push((total_bytes * 2) / 3);
    } else if total_bytes >= 26 {
        out.push((total_bytes * 2) / 5);
        out.push((total_bytes * 4) / 5);
    } else if total_bytes >= 18 {
        out.push(total_bytes / 2);
    }
    out
}

/// Prosodic clitic cohesion (`韵律附着内聚度`) for productive Mandarin `[Stem + Closed-Class
/// Clitic]` constructions that word-based corpora split into separate tokens (`Verb +
/// 了/着/过/得/起来/出来`, `Pronoun + Copula/Modal`, `Numeral/Dem + Classifier`). Defined strictly
/// over closed grammatical classes so every open-domain verb/adjective generalizes without lexical
/// hardcoding.
fn prosodic_clitic_cohesion(
    prev_word: &str,
    next_word: &str,
    prev_kind: MatchKind,
    next_kind: MatchKind,
) -> f32 {
    if prev_word.is_empty()
        || prev_kind != MatchKind::Exact
        || next_kind != MatchKind::Exact
        || is_non_nominal_functor(prev_word)
    {
        return 0.0;
    }
    let prev_len = prev_word.chars().count();
    // 1. Monosyllabic/disyllabic predicate + closed-class aspect / complement / modal clitic
    // (`得` complementizer primarily attaches to monosyllabic predicates `做得/跑得/写得`, while
    // `的` attaches freely to both 1- and 2-char modifiers `好的/使用的`)!
    if next_word == "得"
        && prev_len == 1
        && !matches!(prev_word, "是" | "的" | "地" | "不" | "没" | "很" | "最")
    {
        return 0.36;
    }
    if (1..=2).contains(&prev_len)
        && !matches!(
            prev_word,
            "是" | "我是" | "你是" | "他是" | "她是" | "它是" | "这是" | "那是" | "的" | "地"
        )
        && matches!(
            next_word,
            "的" | "了"
                | "着"
                | "过"
                | "吧"
                | "吗"
                | "呢"
                | "起来"
                | "出来"
                | "下来"
                | "过来"
                | "回来"
                | "下去"
                | "过去"
                | "回去"
                | "不到"
                | "不了"
                | "不起"
                | "不住"
                | "不懂"
                | "不开"
                | "不完"
                | "不着"
                | "一下"
        )
    {
        return if prev_len == 1 { 0.42 } else { 0.24 };
    }
    // 2. Closed-class pronoun + modal / directional / copula verb (`我想`, `我去`, `他有`, `你在`)
    if matches!(
        prev_word,
        "我" | "你" | "他" | "她" | "它" | "我们" | "你们" | "他们"
    ) && matches!(
        next_word,
        "是" | "有" | "在" | "想" | "要" | "会" | "能" | "去" | "来" | "看" | "说" | "做"
    ) {
        return 0.36;
    }
    // 3. Numeral / demonstrative + classifier (`三碗`, `两块`, `这把`, `这朵`, `哪位`)
    if prev_len == 1 && next_word.chars().count() == 1 {
        let mut combined = String::with_capacity(prev_word.len() + next_word.len());
        combined.push_str(prev_word);
        combined.push_str(next_word);
        if is_numeral_or_dem_classifier(&combined) {
            return 0.38;
        }
    }
    0.0
}

fn syntactic_transition_penalty(
    prev_content_word: &str,
    prev_word: &str,
    next_word: &str,
    next_kind: MatchKind,
    is_utterance_end: bool,
    total_bytes: usize,
) -> f32 {
    let mut pen = 0.0_f32;
    if is_numeral_or_dem_classifier(prev_word) && is_non_nominal_functor(next_word) {
        pen -= 3.20;
    }
    // Productive Mandarin `A不A` (`忙不忙`, `累不累`, `好不好`) and `X一X` (`看一看`, `试一试`)
    // morphosyntactic reduplication frames!
    if next_kind == MatchKind::Exact
        && next_word.chars().count() == 1
        && prev_content_word == next_word
    {
        if prev_word == "不" {
            pen += 1.65;
        } else if prev_word == "一" {
            pen += 1.35;
        }
    }
    // Copula `是` cannot take perfective aspect clitic `了` (`*我是了一下` vs `我试了一下`)!
    if matches!(
        prev_word,
        "是" | "我是" | "你是" | "他是" | "她是" | "它是" | "这是" | "那是"
    ) && next_word == "了"
    {
        pen -= 2.60;
    }
    // Degree adverbs (`很/太/最/非常/...`) select gradable predicates, never pronoun+copula
    // (`*很他是` vs `很踏实`)!
    if matches!(
        prev_word,
        "很" | "太" | "最" | "更" | "挺" | "真" | "非常" | "特别" | "十分" | "相当"
    ) && matches!(
        next_word,
        "我是" | "你是" | "他是" | "她是" | "它是" | "是谁"
    ) {
        pen -= 3.00;
    }
    // Degree complementizer `得` takes verbal/adjectival predicates, never postpositional locative
    // nominals (`*使用得过程中` vs `使用的过程中`)!
    if prev_word == "得" && next_word.chars().count() >= 2 && next_word.ends_with('中') {
        pen -= 1.80;
    }
    if is_utterance_end && total_bytes >= 6 && is_dangling_final_functor(next_word) {
        pen -= 1.50;
    }
    pen
}

fn format_corrected_english_surface(typed_raw: &str, canonical_surface: &str) -> String {
    let lower = canonical_surface.to_ascii_lowercase();
    if canonical_surface != lower {
        return canonical_surface.to_string();
    }
    if typed_raw
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_uppercase)
        && !typed_raw[1..].bytes().any(|b| b.is_ascii_uppercase())
    {
        let mut chars = canonical_surface.chars();
        if let Some(first) = chars.next() {
            let mut out = String::with_capacity(canonical_surface.len());
            out.push(first.to_ascii_uppercase());
            out.push_str(chars.as_str());
            return out;
        }
    }
    canonical_surface.to_string()
}

fn is_transparent_particle(word: &str) -> bool {
    matches!(
        word,
        "的" | "地"
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

    #[test]
    fn test_should_enforce_classifier_syntax_and_prefix_diverse_beam_for_woshiyigebing() {
        let mut lex = Lexicon::with_builtin_seed();
        lex.insert_pinyin_word("卧室", "wo shi", -4.62, false);
        lex.insert_pinyin_word("我试", "wo shi", -4.91, false);
        lex.insert_pinyin_word("我市", "wo shi", -5.17, false);
        lex.insert_pinyin_word("并", "bing", -2.82, false);
        lex.insert_english_term("chile", -5.40, false);

        let store = UserStore::new();
        let dec = LatticeDecoder::new();
        let ctx = DecodeContext::default();

        let cands = dec.decode("woshiyigebing", &lex, &store, &ctx);
        assert_eq!(cands[0].text, "我是一个兵");
        let top6: Vec<&str> = cands.iter().take(6).map(|c| c.text.as_str()).collect();
        assert!(
            top6.contains(&"我是一个饼"),
            "expected 我是一个饼 in top-6 candidates, got {top6:?}"
        );
        assert!(
            !top6.iter().any(|t| t.ends_with('并')),
            "dangling conjunction 并 after classifier 一个 must not enter top-6: {top6:?}"
        );

        let cands_chile = dec.decode("chilelianggebing", &lex, &store, &ctx);
        assert!(
            !cands_chile[0].text.contains("chile"),
            "embedded /usr/share/dict/words entry chile must not hijack 吃了: {:?}",
            cands_chile[0].text
        );
    }

    #[test]
    fn test_should_auto_correct_english_typos_in_chinese_and_mixed_sentences() {
        let mut lex = Lexicon::with_builtin_seed();
        lex.insert_pinyin_word("这个", "zhe ge", -2.55, false);
        lex.insert_pinyin_word("部署", "bu shu", -3.45, true);
        lex.insert_pinyin_word("到", "dao", -2.65, false);
        lex.insert_pinyin_word("我们", "wo men", -2.20, false);

        let store = UserStore::new();
        let dec = LatticeDecoder::new();
        let ctx = DecodeContext::default();

        // 1. Standalone English misspellings in Chinese mode (`reivew`, `definately`, `kubernets`)
        assert_eq!(dec.decode("reivew", &lex, &store, &ctx)[0].text, "review");
        assert_eq!(
            dec.decode("definately", &lex, &store, &ctx)[0].text,
            "definitely"
        );
        assert_eq!(
            dec.decode("kubernets", &lex, &store, &ctx)[0].text,
            "Kubernetes"
        );
        assert_eq!(dec.decode("Teh", &lex, &store, &ctx)[0].text, "The");

        // 2. Mixed Chinese-English sentences with misspelled English spans!
        assert_eq!(
            dec.decode("bangworeivewzhegepr", &lex, &store, &ctx)[0].text,
            "帮我 review 这个 PR"
        );
        assert_eq!(
            dec.decode("bushudaokubernets", &lex, &store, &ctx)[0].text,
            "部署到 Kubernetes"
        );

        // 3. Canonical Chinese Pinyin homographs (`women` -> `我们`) must never be displaced!
        assert_eq!(dec.decode("women", &lex, &store, &ctx)[0].text, "我们");
    }
}
