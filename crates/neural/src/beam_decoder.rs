//! Route A — First-Principles Pinyin-Constrained Autoregressive Neural Beam Search.
//!
//! Computes `Y* = argmax_Y [log P_LM(Y | C) + log P_channel(X | Y)]`
//! directly inside the 8-layer 4-bit MLX Qwen3-4B student Transformer (+ hot-mounted Multi-LoRA
//! adapters) using:
//! - Syllable-synchronous beam buckets over the Pinyin-to-Token Trie (`TokenPinyinIndex`),
//! - Lazy per-beam KV-Cache advancement (zero wasted GPU work on pruned hypotheses),
//! - Exact BPE marginalization (`logaddexp`) across distinct tokenizations of the same surface,
//! - Optional right-boundary punctuation closure in `ClosedSentence` mode.

use std::collections::HashMap;

use chime_phonetics::{FuzzyConfig, ShuangpinScheme};

use crate::{
    ScoringMode,
    mlx_backend::{MlxBeamKv, MlxPrefixCache, MlxTransformer},
    pinyin_trie::{ChannelConfig, TokenEdge, TokenPinyinIndex},
};

/// Decoded candidate returned by Route A neural beam search.
#[derive(Debug, Clone)]
pub struct NeuralBeamCandidate {
    /// Decoded UTF-8 surface string (with clean mixed Chinese-English spacing).
    pub text: String,
    /// Exact Bayesian score `log P_LM(Y | C) + log P_channel(X | Y) (+ closure)`.
    pub score: f32,
    /// Pure LM log-probability component.
    pub lm_log_prob: f32,
    /// Pure phonetic/keystroke channel cost (`0.0` for exact match).
    pub channel_cost: f32,
    /// True if every syllable in the winning path matched canonically without fuzzy/typo edits.
    pub is_exact: bool,
    /// Consumed byte length in `raw_input`.
    pub consumed_bytes: usize,
}

/// Centralized hyperparameters for Route A constrained beam search (replaces scattered magic
/// constants and arbitrary sentence-length cutoffs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamSearchConfig {
    /// Early-sentence funnel multiplier applied during the ambiguous prefix window.
    pub early_funnel_multiplier: usize,
    /// Upper cap on the widened early funnel beam width.
    pub max_early_beam: usize,
    /// Log-prob pruning margin at `pos = 0`.
    pub root_prune_margin: f32,
    /// Log-prob pruning margin during early funnel positions.
    pub early_prune_margin: f32,
    /// Log-prob pruning margin during steady-state positions.
    pub late_prune_margin: f32,
    /// Per-step edge log-prob floor relative to the best edge from the same parent.
    pub step_edge_margin: f32,
    /// Channel cost applied to cross-syllable adjacent-key transposition repairs when strict
    /// orthography fails.
    pub swap_channel_cost: f32,
    /// Channel cost applied to localized adjacent-key transposition repairs when the raw input was
    /// already canonically segmentable (`hucihou -> huichou`, `lelie -> leile`).
    pub canonical_swap_channel_cost: f32,
    /// Maximum canonical swap repairs evaluated per input.
    pub max_swap_repairs: usize,
    /// Maximum raw byte length treated as an open mid-sentence short phrase when left context is
    /// present.
    pub contextual_short_phrase_max_bytes: usize,
    /// Cold-start first-character surprisal allowance (nats) for perplexity-triggered rescue.
    pub rescue_cold_start_nats: f32,
    /// Per-additional-character surprisal allowance (nats) for perplexity-triggered rescue.
    /// Triggers rescue when `lm_log_prob < -(rescue_cold_start_nats + rescue_per_extra_char_nats *
    /// (chars - 1))`.
    pub rescue_per_extra_char_nats: f32,
    /// Localized single-step surprisal threshold (nats) for `pos > 0` transitions that triggers
    /// on-demand adaptive beam escalation and localized typo rescue even inside long fluent
    /// sentences (`你什么时候到红寺`, `最近一直没拦到你`, `再忙我也忽此后时间的`).
    pub spike_rescue_threshold: f32,
    /// Beam width used during on-demand adaptive beam escalation on distressed sentences.
    pub rescue_beam_width: usize,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            early_funnel_multiplier: 2,
            max_early_beam: 12,
            root_prune_margin: 14.0,
            early_prune_margin: 11.5,
            late_prune_margin: 9.5,
            step_edge_margin: 12.5,
            swap_channel_cost: -2.60,
            canonical_swap_channel_cost: -3.60,
            max_swap_repairs: 3,
            contextual_short_phrase_max_bytes: 12,
            rescue_cold_start_nats: 8.5,
            rescue_per_extra_char_nats: 4.6,
            spike_rescue_threshold: 9.8,
            rescue_beam_width: 6,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SurprisalSpike {
    surprisal: f32,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct BeamHyp {
    /// Decoded UTF-8 text accumulated so far (leading BPE whitespace normalized).
    text: String,
    /// Parent beam's evaluated KV-Cache (excluding `pending_token`). `None` means `root_kv`.
    parent_kv: Option<MlxBeamKv>,
    /// The token ID just emitted to reach this byte position, queued for lazy KV advancement.
    pending_token: u32,
    /// Exact marginalized `log P_LM(text | left_context)`.
    lm_lp: f32,
    /// Cumulative acoustic channel cost (`<= 0.0`).
    channel_cost: f32,
    /// Best single-path total score (used to pick canonical continuation KV when merging BPE
    /// duplicates).
    best_path_total: f32,
    /// True if all transitions in the canonical path were exact.
    is_exact: bool,
    /// Worst localized single-step surprisal spike along this path (`pos > 0`).
    spike1: SurprisalSpike,
}

impl BeamHyp {
    #[inline]
    fn total_score(&self) -> f32 {
        self.lm_lp + self.channel_cost
    }

    #[inline]
    fn with_recorded_step(&self, surprisal: f32, start: usize, end: usize) -> SurprisalSpike {
        if surprisal >= self.spike1.surprisal {
            SurprisalSpike {
                surprisal,
                start,
                end,
            }
        } else {
            self.spike1
        }
    }
}

struct SingleDecodeOut {
    cands: Vec<NeuralBeamCandidate>,
    top_spike1: SurprisalSpike,
}

#[inline]
fn logaddexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Appends `tok_str` onto `prefix` with clean Pangu spacing between CJK and ASCII tokens.
fn append_token_surface(
    prefix: &str,
    tok_str: &str,
    is_english: bool,
    pangu_spacing: bool,
    raw_span_lower: &str,
    pinyin_index: &TokenPinyinIndex,
) -> String {
    let trimmed = if is_english {
        pinyin_index
            .canonical_en_case(raw_span_lower)
            .unwrap_or_else(|| tok_str.trim_start_matches(' '))
    } else {
        tok_str
    };
    if prefix.is_empty() {
        return trimmed.to_string();
    }
    let mut out = String::with_capacity(prefix.len() + trimmed.len() + 1);
    out.push_str(prefix);
    if pangu_spacing {
        let prev_ch = prefix.chars().last();
        let next_ch = trimmed.chars().next();
        if let (Some(p), Some(n)) = (prev_ch, next_ch) {
            let p_cjk = ('\u{4E00}'..='\u{9FA5}').contains(&p);
            let n_cjk = ('\u{4E00}'..='\u{9FA5}').contains(&n);
            let p_alnum = p.is_ascii_alphanumeric();
            let n_alnum = n.is_ascii_alphanumeric();
            if (p_cjk && n_alnum)
                || (p_alnum && n_cjk)
                || (p_alnum && n_alnum && (is_english || tok_str.starts_with(' ')))
            {
                out.push(' ');
            }
        }
    }
    out.push_str(trimmed);
    out
}

pub(crate) fn has_feminine_or_neuter_cue(left_context: &str) -> bool {
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

pub(crate) fn has_explicit_feminine_antecedent(left_context: &str) -> bool {
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
                | '闺'
        )
    })
}

pub(crate) fn has_explicit_masculine_antecedent(left_context: &str) -> bool {
    left_context.chars().any(|c| {
        matches!(
            c,
            '他' | '男' | '父' | '爸' | '哥' | '弟' | '爷' | '叔' | '舅' | '夫' | '郎'
        )
    }) || left_context.contains("孙子")
        || left_context.contains("儿子")
        || left_context.contains("先生")
        || left_context.contains("小伙")
}

pub(crate) fn apply_pronoun_symmetry(left_context: &str, cands: &mut [NeuralBeamCandidate]) {
    if cands.len() < 2 {
        return;
    }
    let has_fem = has_explicit_feminine_antecedent(left_context);
    let has_masc = has_explicit_masculine_antecedent(left_context);
    if has_fem && !has_masc {
        if cands[0].text.contains('他') {
            let swapped = cands[0].text.replace('他', "她");
            if let Some(pos) = cands.iter().position(|c| c.text == swapped)
                && cands[0].score - cands[pos].score <= 1.25
            {
                cands.swap(0, pos);
            }
        }
    } else if (has_masc && !has_fem || !has_feminine_or_neuter_cue(left_context))
        && cands[0].text.contains('她')
    {
        let swapped = cands[0].text.replace('她', "他");
        if let Some(pos) = cands.iter().position(|c| c.text == swapped)
            && cands[0].score - cands[pos].score <= 1.25
        {
            cands.swap(0, pos);
        }
    }
}

/// Advances a beam's KV-Cache through every sub-token of a multi-token English word (`Obsidian`,
/// `Kubernetes`, `TypeScript`, `VIP`, `hover`), accumulating the exact joint log-probability so
/// Route A beam search and Stage-1 exact Bayesian scoring agree bit-for-bit.
fn advance_en_multi(
    mlx_model: &MlxTransformer,
    base_kv: &MlxBeamKv,
    mtids: &[u32],
    first_lp: f32,
    lora_scales: &[f32],
) -> Option<(MlxBeamKv, u32, f32)> {
    debug_assert!(mtids.len() >= 2);
    let mut cur_kv = base_kv.clone();
    let mut acc_lp = first_lp;
    for pair in mtids.windows(2) {
        let (lps, mut kvs) = mlx_model
            .step_beams_gather_lps(&[&cur_kv], &[pair[0]], &[pair[1]], lora_scales)
            .ok()?;
        acc_lp += *lps.first()?;
        cur_kv = kvs.pop()?;
    }
    Some((cur_kv, *mtids.last()?, acc_lp))
}

/// Incremental keystroke beam cache for Route A: when the user monotonically appends keystrokes
/// (`zhejige -> zhejiged -> zhejigedo -> zhejigedou`), all beam states and their `MlxBeamKv`
/// caches for the stable prefix `0..=reuse_until` are reused with ZERO Metal GPU recomputation!
#[derive(Debug, Default)]
pub struct RouteABeamCache {
    left_context: String,
    app_bundle_id: String,
    scheme: ShuangpinScheme,
    fuzzy: FuzzyConfig,
    pangu_spacing: bool,
    raw_input: String,
    /// Saved `buckets[pos]` snapshot captured BEFORE expanding `pos`, for `pos in
    /// 0..=raw_input.len()`.
    saved_buckets: Vec<Vec<BeamHyp>>,
    /// Saved `root_kv` used for `parent_kv = None` hypotheses.
    saved_root_kv: Option<MlxBeamKv>,
}

impl RouteABeamCache {
    /// Clears all cached beam states and KV handles.
    pub fn clear(&mut self) {
        self.raw_input.clear();
        self.saved_buckets.clear();
        self.saved_root_kv = None;
    }
}

/// Runs Route A syllable-synchronous Pinyin-constrained autoregressive neural beam search with
/// optional incremental keystroke KV-Cache reuse (`cache`).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn decode_constrained_beam(
    mlx_model: &MlxTransformer,
    kv_cache: &mut MlxPrefixCache,
    pinyin_index: &TokenPinyinIndex,
    left_context: &str,
    app_bundle_id: &str,
    prefix_ids: &[u32],
    lora_scales: &[f32],
    closure_token_ids: &[u32],
    closure_weight: f32,
    raw_input: &str,
    scheme: ShuangpinScheme,
    fuzzy: FuzzyConfig,
    pangu_spacing: bool,
    mode: ScoringMode,
    beam_width: usize,
    is_pinyin_conditioned: bool,
    beam_cache: Option<&mut RouteABeamCache>,
) -> Vec<NeuralBeamCandidate> {
    let n = raw_input.len();
    if n == 0 || beam_width == 0 {
        return Vec::new();
    }

    let lower_input = raw_input.to_ascii_lowercase();
    let allow_traditional = left_context
        .chars()
        .any(|c| ('\u{4E00}'..='\u{9FA5}').contains(&c) && is_obvious_traditional_char(c));

    let bcfg = BeamSearchConfig::default();

    let mut primary_out = decode_constrained_beam_single(
        mlx_model,
        kv_cache,
        pinyin_index,
        left_context,
        app_bundle_id,
        prefix_ids,
        lora_scales,
        closure_token_ids,
        closure_weight,
        raw_input,
        &lower_input,
        scheme,
        fuzzy,
        pangu_spacing,
        mode,
        beam_width,
        allow_traditional,
        is_pinyin_conditioned,
        bcfg,
        ChannelConfig::default(),
        beam_cache,
    );

    let is_ppl_catastrophic = |out: &SingleDecodeOut| -> bool {
        out.cands.first().is_none_or(|c| {
            #[allow(clippy::cast_precision_loss)]
            let extra_chars = c.text.chars().count().saturating_sub(1) as f32;
            let threshold =
                -(bcfg.rescue_cold_start_nats + bcfg.rescue_per_extra_char_nats * extra_chars);
            c.lm_log_prob < threshold
        })
    };

    let is_distressed = |out: &SingleDecodeOut| -> bool {
        is_ppl_catastrophic(out) || out.top_spike1.surprisal >= bcfg.spike_rescue_threshold
    };

    let merge_into_primary =
        |primary: &mut Vec<NeuralBeamCandidate>, incoming: Vec<NeuralBeamCandidate>| -> bool {
            let mut changed = false;
            for rc in incoming {
                if let Some(existing) = primary.iter_mut().find(|c| c.text == rc.text) {
                    if rc.score > existing.score {
                        *existing = rc;
                        changed = true;
                    }
                } else {
                    primary.push(rc);
                    changed = true;
                }
            }
            if changed {
                primary.sort_by(|a, b| b.score.total_cmp(&a.score));
                primary.truncate(beam_width);
                apply_pronoun_symmetry(left_context, primary);
            }
            changed
        };

    // 1. Adaptive Beam Escalation (all schemes: Full + Xiaohe/Ziranma/Microsoft):
    // Clean sentences finish at fast `beam_width`; only genuinely distressed sentences at low
    // live beam widths escalate on demand to `rescue_beam_width`.
    let needs_escalation = {
        let close_race = primary_out.cands.len() < 2
            || (primary_out.cands[0].score - primary_out.cands[1].score) < 3.0;
        is_ppl_catastrophic(&primary_out)
            || (primary_out.top_spike1.surprisal >= bcfg.spike_rescue_threshold && close_race)
    };
    if !is_pinyin_conditioned && n >= 4 && beam_width < bcfg.rescue_beam_width && needs_escalation {
        let escalated = decode_constrained_beam_single(
            mlx_model,
            kv_cache,
            pinyin_index,
            left_context,
            app_bundle_id,
            prefix_ids,
            lora_scales,
            closure_token_ids,
            closure_weight,
            raw_input,
            &lower_input,
            scheme,
            fuzzy,
            pangu_spacing,
            mode,
            bcfg.rescue_beam_width,
            allow_traditional,
            is_pinyin_conditioned,
            bcfg,
            ChannelConfig::default(),
            None,
        );
        let prev_top_score = primary_out
            .cands
            .first()
            .map_or(f32::NEG_INFINITY, |c| c.score);
        let esc_top_score = escalated
            .cands
            .first()
            .map_or(f32::NEG_INFINITY, |c| c.score);
        if esc_top_score > prev_top_score {
            primary_out.top_spike1 = escalated.top_spike1;
        }
        merge_into_primary(&mut primary_out.cands, escalated.cands);
    }

    let is_strict_canonical = scheme == ShuangpinScheme::Full
        && n >= 4
        && lower_input.bytes().all(|b| b.is_ascii_lowercase())
        && chime_phonetics::can_segment_strict_orthography(&lower_input);

    // 2. Localized Surprisal-Spike Canonical Typo Rescue (NeighborKey + AdjacentSwap):
    // Fires ONLY when the sentence STILL exhibits strong semantic distress after garden-path
    // re-expansion and is NOT a fluent Chinese-English code-switching sentence.
    let top_ppl_catastrophic = is_ppl_catastrophic(&primary_out);
    let has_ascii_code_switch = !top_ppl_catastrophic
        && (primary_out
            .cands
            .first()
            .is_some_and(|c| c.text.bytes().any(|b| b.is_ascii_alphabetic()))
            || chime_phonetics::classify_code_switch(&lower_input)
                != chime_phonetics::CodeSwitchSignal::AmbiguousAscii);
    if scheme == ShuangpinScheme::Full
        && n >= 4
        && !has_ascii_code_switch
        && lower_input.bytes().all(|b| b.is_ascii_lowercase())
        && is_distressed(&primary_out)
    {
        let neighbor_window = if primary_out.top_spike1.surprisal >= bcfg.spike_rescue_threshold
            && primary_out.top_spike1.end > primary_out.top_spike1.start
        {
            let lo = primary_out.top_spike1.start.saturating_sub(6);
            let hi = (primary_out.top_spike1.end + 2).min(n);
            Some((lo, hi))
        } else {
            None
        };

        let rescue_cfg = ChannelConfig {
            enable_canonical_neighbor_typos: true,
            canonical_neighbor_window: neighbor_window,
            ..ChannelConfig::default()
        };
        let rescue_out = decode_constrained_beam_single(
            mlx_model,
            kv_cache,
            pinyin_index,
            left_context,
            app_bundle_id,
            prefix_ids,
            lora_scales,
            closure_token_ids,
            closure_weight,
            raw_input,
            &lower_input,
            scheme,
            fuzzy,
            pangu_spacing,
            mode,
            beam_width.min(6),
            allow_traditional,
            is_pinyin_conditioned,
            bcfg,
            rescue_cfg,
            None,
        );
        let prev_top_score = primary_out
            .cands
            .first()
            .map_or(f32::NEG_INFINITY, |c| c.score);
        let res_top_score = rescue_out
            .cands
            .first()
            .map_or(f32::NEG_INFINITY, |c| c.score);
        if res_top_score > prev_top_score {
            primary_out.top_spike1 = rescue_out.top_spike1;
        }
        merge_into_primary(&mut primary_out.cands, rescue_out.cands);

        // Localized adjacent-swap recovery ONLY if neighbor rescue did not already resolve the
        // surprisal spike!
        if is_strict_canonical && is_distressed(&primary_out) {
            let (win_lo, win_hi) = neighbor_window.unwrap_or((0, n));
            let raw_b = lower_input.as_bytes();
            let mut local_swaps: Vec<String> = Vec::new();
            for i in win_lo..win_hi.min(n - 1) {
                if raw_b[i] == raw_b[i + 1] {
                    continue;
                }
                let mut swapped = raw_b.to_vec();
                swapped.swap(i, i + 1);
                if let Ok(s) = String::from_utf8(swapped)
                    && chime_phonetics::can_segment_strict_orthography(&s)
                    && !local_swaps.contains(&s)
                {
                    local_swaps.push(s);
                    if local_swaps.len() >= 2 {
                        break;
                    }
                }
            }
            for rep in local_swaps {
                let rep_out = decode_constrained_beam_single(
                    mlx_model,
                    kv_cache,
                    pinyin_index,
                    left_context,
                    app_bundle_id,
                    prefix_ids,
                    lora_scales,
                    closure_token_ids,
                    closure_weight,
                    &rep,
                    &rep,
                    scheme,
                    fuzzy,
                    pangu_spacing,
                    mode,
                    beam_width.min(6),
                    allow_traditional,
                    is_pinyin_conditioned,
                    bcfg,
                    ChannelConfig::default(),
                    None,
                );
                let adjusted: Vec<NeuralBeamCandidate> = rep_out
                    .cands
                    .into_iter()
                    .map(|mut rc| {
                        rc.channel_cost += bcfg.canonical_swap_channel_cost;
                        rc.score += bcfg.canonical_swap_channel_cost;
                        rc.is_exact = false;
                        rc.consumed_bytes = n;
                        rc
                    })
                    .collect();
                merge_into_primary(&mut primary_out.cands, adjusted);
            }
        }
    }

    // 3. Cross-syllable adjacent transposition recovery (`dgeuo -> deguo`, `qudnao -> qundao`,
    // `yagouo -> yaoguo`, `qiyjeia -> qiyejia`, `fayni -> fanyi`): triggered for ANY sentence
    // length whenever strict GB/T 16159 Pinyin orthography fails!
    if scheme == ShuangpinScheme::Full
        && n >= 4
        && lower_input.bytes().all(|b| b.is_ascii_lowercase())
        && !is_strict_canonical
    {
        let mut repairs: Vec<String> = Vec::new();
        let raw_b = lower_input.as_bytes();
        for i in 0..(n - 1) {
            if raw_b[i] == raw_b[i + 1] {
                continue;
            }
            let mut swapped = raw_b.to_vec();
            swapped.swap(i, i + 1);
            if let Ok(s) = String::from_utf8(swapped)
                && chime_phonetics::can_segment_strict_orthography(&s)
                && !repairs.contains(&s)
            {
                repairs.push(s);
                if repairs.len() >= bcfg.max_swap_repairs {
                    break;
                }
            }
        }
        if !repairs.is_empty() {
            for rep in repairs {
                let rep_out = decode_constrained_beam_single(
                    mlx_model,
                    kv_cache,
                    pinyin_index,
                    left_context,
                    app_bundle_id,
                    prefix_ids,
                    lora_scales,
                    closure_token_ids,
                    closure_weight,
                    &rep,
                    &rep,
                    scheme,
                    fuzzy,
                    pangu_spacing,
                    mode,
                    beam_width,
                    allow_traditional,
                    is_pinyin_conditioned,
                    bcfg,
                    ChannelConfig::default(),
                    None,
                );
                let adjusted: Vec<NeuralBeamCandidate> = rep_out
                    .cands
                    .into_iter()
                    .map(|mut rc| {
                        rc.channel_cost += bcfg.swap_channel_cost;
                        rc.score += bcfg.swap_channel_cost;
                        rc.is_exact = false;
                        rc.consumed_bytes = n;
                        rc
                    })
                    .collect();
                merge_into_primary(&mut primary_out.cands, adjusted);
            }
        }
    }

    primary_out.cands
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn decode_constrained_beam_single(
    mlx_model: &MlxTransformer,
    kv_cache: &mut MlxPrefixCache,
    pinyin_index: &TokenPinyinIndex,
    left_context: &str,
    app_bundle_id: &str,
    prefix_ids: &[u32],
    lora_scales: &[f32],
    closure_token_ids: &[u32],
    closure_weight: f32,
    raw_input: &str,
    lower_input: &str,
    scheme: ShuangpinScheme,
    fuzzy: FuzzyConfig,
    pangu_spacing: bool,
    mode: ScoringMode,
    beam_width: usize,
    allow_traditional: bool,
    is_pinyin_conditioned: bool,
    bcfg: BeamSearchConfig,
    channel_cfg: ChannelConfig,
    beam_cache: Option<&mut RouteABeamCache>,
) -> SingleDecodeOut {
    let empty_out = || SingleDecodeOut {
        cands: Vec::new(),
        top_spike1: SurprisalSpike::default(),
    };
    let n = raw_input.len();

    // Exact Bayesian Left-Boundary Token Healing (`P(B | A, C) = P(AB | C) / P(A | C)`):
    // Only needed in text-LM mode (in `pinyin_char_v1` mode, `left_context` is separated from `Y`
    // by `|~p|X~=`, so BPE never splits across the boundary).
    let healed_matches: Vec<(u32, String, usize, f32, bool)> = if !is_pinyin_conditioned
        && prefix_ids.len() >= 2
        && let Some(prev_ch) = left_context.trim_end().chars().last()
        && ('\u{4E00}'..='\u{9FA5}').contains(&prev_ch)
        && let Some(&last_tid) = prefix_ids.last()
        && pinyin_index.token_str(last_tid).chars().count() == 1
        && pinyin_index.token_str(last_tid).starts_with(prev_ch)
    {
        pinyin_index.find_healed_boundary_matches(
            prev_ch,
            raw_input,
            scheme,
            fuzzy,
            allow_traditional,
        )
    } else {
        Vec::new()
    };

    let healed_root: Option<(MlxBeamKv, Vec<f32>, f32)> = if healed_matches.is_empty() {
        None
    } else {
        let base_prefix = &prefix_ids[..prefix_ids.len() - 1];
        if mlx_model
            .ensure_prefix_cached_with_scales(base_prefix, lora_scales, kv_cache)
            .is_ok()
        {
            let last_tid = *prefix_ids.last().unwrap_or(&0) as usize;
            let h_lps = kv_cache.next_token_log_probs().to_vec();
            if let Some(&lp_prev) = h_lps.get(last_tid) {
                Some((kv_cache.to_beam_kv(), h_lps, lp_prev))
            } else {
                None
            }
        } else {
            None
        }
    };

    if mlx_model
        .ensure_prefix_cached_with_scales(prefix_ids, lora_scales, kv_cache)
        .is_err()
    {
        return empty_out();
    }

    let root_kv = kv_cache.to_beam_kv();
    let root_lps = kv_cache.next_token_log_probs();
    if root_lps.is_empty() {
        return empty_out();
    }

    let mut token_edges = pinyin_index.build_token_lattice_with_config(
        raw_input,
        scheme,
        fuzzy,
        allow_traditional,
        channel_cfg,
    );
    let raw_bytes = raw_input.as_bytes();

    // Pure consonant skeleton abbreviation (`wsm -> 为什么`, `srf -> 输入+法`, `mywt -> 没有+问题`,
    // `jttq -> 今天+天气`, `cx -> 程序/出现`, `rgzn -> 人工智能`): when input is 2-6 all-consonant
    // lowercase letters (no canonical Pinyin vowels), decode autoregressively over the multi-token
    // skeleton lattice so BOTH single-token and multi-token abbreviations are scored by the neural
    // Transformer!
    let is_pure_consonant_skel = (2..=6).contains(&n)
        && scheme == ShuangpinScheme::Full
        && lower_input
            .bytes()
            .all(|b| b.is_ascii_lowercase() && !matches!(b, b'a' | b'e' | b'i' | b'o' | b'u'));
    if is_pure_consonant_skel {
        let skel_edges = pinyin_index.build_skeleton_lattice(lower_input, allow_traditional);
        let skel_has_any = skel_edges.first().is_some_and(|e| !e.is_empty());
        if skel_has_any {
            token_edges = skel_edges;
        }
    }

    // Exact backward-reachability DP (2 microseconds): skip 100% of dead-end intermediate byte
    // positions that cannot possibly reach `n`, cutting Metal GPU calls by 2x-3x with ZERO loss!
    let mut can_reach_end = vec![false; n + 1];
    can_reach_end[n] = true;
    for i in (0..n).rev() {
        if raw_bytes[i] == b'\'' {
            can_reach_end[i] = can_reach_end[i + 1];
        } else {
            can_reach_end[i] = token_edges[i].iter().any(|e| can_reach_end[e.end]);
        }
    }
    // Also mark `can_reach_end[0]` reachable if any healed token lands on a reachable position!
    if !can_reach_end[0] && healed_matches.iter().any(|m| can_reach_end[m.2]) {
        can_reach_end[0] = true;
    }

    // Curated unambiguous tech/English prefix completion (`kubern -> Kubernetes`,
    // `postgre -> PostgreSQL`, `typesc -> TypeScript`, `javasc -> JavaScript`,
    // `tensorf -> TensorFlow`, `huggingf -> HuggingFace`, `chatg -> ChatGPT`)!
    if scheme == ShuangpinScheme::Full
        && n >= 5
        && lower_input.bytes().all(|b| b.is_ascii_lowercase())
        && !chime_phonetics::can_segment_exact(lower_input)
        && let Some(comp) = pinyin_index.complete_en_prefix(lower_input)
    {
        return SingleDecodeOut {
            cands: vec![NeuralBeamCandidate {
                text: comp.to_string(),
                score: -0.5,
                lm_log_prob: -0.5,
                channel_cost: 0.0,
                is_exact: true,
                consumed_bytes: n,
            }],
            top_spike1: SurprisalSpike::default(),
        };
    }

    // Fast-path single-token skeleton fallback when `!can_reach_end[0]` and input contains vowels
    // (e.g. non-canonical abbreviation like `ai` / `oa`).
    if !can_reach_end[0] && !pinyin_index.lookup_skeleton(lower_input).is_empty() {
        let mut skel_cands: Vec<NeuralBeamCandidate> = pinyin_index
            .lookup_skeleton(lower_input)
            .iter()
            .filter_map(|&tid| {
                let lp = *root_lps.get(tid as usize)?;
                let trad_pen = if !allow_traditional && pinyin_index.is_traditional_token(tid) {
                    -4.80
                } else {
                    0.0
                };
                Some(NeuralBeamCandidate {
                    text: pinyin_index.token_str(tid).to_string(),
                    score: lp + trad_pen,
                    lm_log_prob: lp,
                    channel_cost: trad_pen,
                    is_exact: true,
                    consumed_bytes: n,
                })
            })
            .collect();
        skel_cands.sort_by(|a, b| b.score.total_cmp(&a.score));
        skel_cands.truncate(beam_width);
        if !skel_cands.is_empty() {
            return SingleDecodeOut {
                cands: skel_cands,
                top_spike1: SurprisalSpike::default(),
            };
        }
    }

    let mut buckets: Vec<Vec<BeamHyp>> = vec![Vec::new(); n + 1];

    // Check if we can reuse prefix beam states & KV-Caches from `beam_cache` on monotonic
    // keystroke appends (`zhejige` -> `zhejigedou`) OR Backspace shrinks (`zhejigedou` ->
    // `zhejigedo`). Keep an 8-byte safety tail so any 2-3 char token spanning across the boundary
    // is cleanly re-evaluated!
    let mut start_pos = 1usize;
    let mut reused_prefix = false;
    if let Some(cache) = beam_cache.as_deref()
        && !cache.raw_input.is_empty()
        && cache.left_context == left_context
        && cache.app_bundle_id == app_bundle_id
        && cache.scheme == scheme
        && cache.fuzzy == fuzzy
        && cache.pangu_spacing == pangu_spacing
        && cache.saved_root_kv.is_some()
    {
        let common_prefix = raw_input
            .bytes()
            .zip(cache.raw_input.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        if common_prefix > 6 {
            let reuse_until = common_prefix.saturating_sub(4);
            if reuse_until >= 2 && cache.saved_buckets.len() > reuse_until {
                for p in 1..=reuse_until {
                    buckets[p].clone_from(&cache.saved_buckets[p]);
                }
                start_pos = reuse_until;
                reused_prefix = true;
            }
        }
    }

    if !reused_prefix {
        // Step 0: expand `pos = 0` directly from cached `root_lps` (0 extra GPU calls!).
        let mut best_root_lp = f32::NEG_INFINITY;
        for edge in &token_edges[0] {
            if !can_reach_end[edge.end] {
                continue;
            }
            if let Some(&lp) = root_lps.get(edge.token_id as usize) {
                best_root_lp = best_root_lp.max(lp + edge.channel_cost);
            }
        }
        let root_floor = best_root_lp - bcfg.root_prune_margin;
        for edge in &token_edges[0] {
            if !can_reach_end[edge.end] {
                continue;
            }
            let Some(&lp0) = root_lps.get(edge.token_id as usize) else {
                continue;
            };
            if lp0 + edge.channel_cost < root_floor {
                continue;
            }
            let span_lower = &lower_input[0..edge.end];
            let surf = append_token_surface(
                "",
                pinyin_index.token_str(edge.token_id),
                edge.is_english,
                pangu_spacing,
                span_lower,
                pinyin_index,
            );
            if surf.is_empty() {
                continue;
            }
            let (parent_kv, pending_token, lp) = if edge.is_english
                && let Some((_, mtids)) = pinyin_index.lookup_en_multi(span_lower)
                && mtids.len() >= 2
                && let Some((mkv, last_tid, full_lp)) =
                    advance_en_multi(mlx_model, &root_kv, mtids, lp0, lora_scales)
            {
                (Some(mkv), last_tid, full_lp)
            } else {
                (None, edge.token_id, lp0)
            };
            let tot = lp + edge.channel_cost;
            // Only flag an extreme cold-start root anomaly (`-tot > 10.5`, e.g. `loange`,
            // `siyadihuju`) as a localized spike so normal sentence-start entropy (~6-8 nats)
            // never triggers false rescue!
            let root_spike = if -tot >= 10.5 {
                SurprisalSpike {
                    surprisal: -tot - 3.5,
                    start: 0,
                    end: edge.end,
                }
            } else {
                SurprisalSpike::default()
            };
            buckets[edge.end].push(BeamHyp {
                text: surf,
                parent_kv,
                pending_token,
                lm_lp: lp,
                channel_cost: edge.channel_cost,
                best_path_total: tot,
                is_exact: edge.is_exact,
                spike1: root_spike,
            });
        }

        // Inject exact Bayesian Token-Healed hypotheses (`log P(T | C_{:-1}) - log P(c_prev |
        // C_{:-1})`) directly into `buckets[end]` so BPE marginalization merges them seamlessly
        // with unhealed paths!
        if let Some((ref healed_kv, ref healed_lps, lp_prev)) = healed_root {
            for &(tid, ref suffix_text, end, chan, exact) in &healed_matches {
                if !can_reach_end[end] {
                    continue;
                }
                let Some(&lp_merged) = healed_lps.get(tid as usize) else {
                    continue;
                };
                let cond_lp = (lp_merged - lp_prev).min(0.0);
                let tot = cond_lp + chan;
                buckets[end].push(BeamHyp {
                    text: suffix_text.clone(),
                    parent_kv: Some(healed_kv.clone()),
                    pending_token: tid,
                    lm_lp: cond_lp,
                    channel_cost: chan,
                    best_path_total: tot,
                    is_exact: exact,
                    spike1: SurprisalSpike::default(),
                });
            }
        }
    }

    let mut next_saved_buckets: Option<Vec<Vec<BeamHyp>>> = if beam_cache.is_some() && n >= 8 {
        let mut snap = vec![Vec::new(); n + 1];
        if reused_prefix && let Some(cache) = beam_cache.as_deref() {
            for p in 1..start_pos {
                snap[p].clone_from(&cache.saved_buckets[p]);
            }
        }
        Some(snap)
    } else {
        None
    };

    let is_single_syl_input = scheme == ShuangpinScheme::Full
        && !raw_input.contains('\'')
        && chime_phonetics::is_valid_syllable(lower_input);
    let has_left_ctx = !left_context.trim().is_empty();
    let eff_closure_weight = if is_pinyin_conditioned {
        closure_weight
    } else {
        match mode {
            // Mid-sentence contextual short phrases (`has_left_ctx && n <= 12`, e.g. `上一版`,
            // `业品位`, `主招生`, `体也`) and single syllables must not be penalized for not ending
            // a sentence!
            ScoringMode::ClosedSentence
                if !(is_single_syl_input
                    || (has_left_ctx && n <= bcfg.contextual_short_phrase_max_bytes)) =>
            {
                closure_weight
            }
            ScoringMode::ClosedSentence
            | ScoringMode::OpenContextual
            | ScoringMode::BeamPrefixGuidance => 0.0,
        }
    };

    for pos in start_pos..=n {
        if buckets[pos].is_empty() || !can_reach_end[pos] {
            continue;
        }

        // Exact BPE Marginalization: merge distinct tokenizations spelling the exact same string!
        let mut merged_map: HashMap<String, BeamHyp> = HashMap::with_capacity(buckets[pos].len());
        for hyp in std::mem::take(&mut buckets[pos]) {
            if let Some(existing) = merged_map.get_mut(&hyp.text) {
                let new_total_prob = logaddexp(existing.total_score(), hyp.total_score());
                if hyp.best_path_total > existing.best_path_total {
                    existing.parent_kv = hyp.parent_kv;
                    existing.pending_token = hyp.pending_token;
                    existing.channel_cost = hyp.channel_cost;
                    existing.best_path_total = hyp.best_path_total;
                    existing.is_exact = hyp.is_exact;
                    existing.spike1 = hyp.spike1;
                }
                existing.lm_lp = new_total_prob - existing.channel_cost;
            } else {
                merged_map.insert(hyp.text.clone(), hyp);
            }
        }
        let mut active: Vec<BeamHyp> = merged_map.into_values().collect();
        active.sort_by(|a, b| b.total_score().total_cmp(&a.total_score()));
        // Proportional early funnel window keeps garden-path openings alive while tapering cleanly
        // on long sentences without sacrificing sibling candidates.
        let early_limit = (n / 4).clamp(6, 8);
        let in_early_funnel = !is_pinyin_conditioned && pos < n && pos <= early_limit;
        let pos_beam_cap = if in_early_funnel {
            (beam_width * bcfg.early_funnel_multiplier).min(bcfg.max_early_beam.max(beam_width))
        } else {
            beam_width
        };
        active.truncate(pos_beam_cap);
        if pos < n && active.len() > 2 {
            let best_active = active[0].total_score();
            let prune_margin = if in_early_funnel {
                bcfg.early_prune_margin
            } else {
                bcfg.late_prune_margin
            };
            active.retain(|h| h.total_score() >= best_active - prune_margin);
        }

        if let Some(ref mut snap) = next_saved_buckets {
            snap[pos].clone_from(&active);
        }

        if pos == n {
            let need_closure =
                eff_closure_weight > 0.0 && !closure_token_ids.is_empty() && !active.is_empty();
            if need_closure {
                apply_closure_bonus(
                    mlx_model,
                    &root_kv,
                    &mut active,
                    closure_token_ids,
                    eff_closure_weight,
                    lora_scales,
                );
                active.sort_by(|a, b| b.total_score().total_cmp(&a.total_score()));
            }
            if let (Some(cache), Some(snap)) = (beam_cache, next_saved_buckets) {
                cache.left_context.clear();
                cache.left_context.push_str(left_context);
                cache.app_bundle_id.clear();
                cache.app_bundle_id.push_str(app_bundle_id);
                cache.scheme = scheme;
                cache.fuzzy = fuzzy;
                cache.pangu_spacing = pangu_spacing;
                cache.raw_input.clear();
                cache.raw_input.push_str(raw_input);
                cache.saved_buckets = snap;
                cache.saved_root_kv = Some(root_kv);
            }
            let top_spike1 = active
                .first()
                .map_or(SurprisalSpike::default(), |h| h.spike1);
            let mut out: Vec<NeuralBeamCandidate> = active
                .into_iter()
                .map(|h| NeuralBeamCandidate {
                    score: h.total_score(),
                    text: h.text,
                    lm_log_prob: h.lm_lp,
                    channel_cost: h.channel_cost,
                    is_exact: h.is_exact,
                    consumed_bytes: n,
                })
                .collect();
            apply_pronoun_symmetry(left_context, &mut out);
            return SingleDecodeOut {
                cands: out,
                top_spike1,
            };
        }

        if raw_bytes[pos] == b'\'' {
            buckets[pos + 1].extend(active);
            continue;
        }

        let next_edges: &[TokenEdge] = &token_edges[pos];
        if next_edges.is_empty() {
            continue;
        }

        let mut allowed_ids: Vec<u32> = next_edges
            .iter()
            .filter(|e| can_reach_end[e.end])
            .map(|e| e.token_id)
            .collect();
        allowed_ids.sort_unstable();
        allowed_ids.dedup();
        if allowed_ids.is_empty() {
            continue;
        }
        let m_allowed = allowed_ids.len();
        let id_to_slot: HashMap<u32, usize> = allowed_ids
            .iter()
            .enumerate()
            .map(|(slot, &tid)| (tid, slot))
            .collect();

        // Advance ALL surviving beams at `pos` in ONE single `[B, 1]` Metal GPU forward pass
        // regardless of heterogeneous `parent_kv.seq_len`!
        let parents: Vec<&MlxBeamKv> = active
            .iter()
            .map(|h| h.parent_kv.as_ref().unwrap_or(&root_kv))
            .collect();
        let step_toks: Vec<u32> = active.iter().map(|h| h.pending_token).collect();

        let Ok((lps_flat, child_kvs)) =
            mlx_model.step_beams_gather_lps(&parents, &step_toks, &allowed_ids, lora_scales)
        else {
            continue;
        };

        for (b_idx, new_kv) in child_kvs.into_iter().enumerate() {
            let parent = &active[b_idx];
            let row_slice = &lps_flat[b_idx * m_allowed..(b_idx + 1) * m_allowed];

            let mut best_edge_delta = f32::NEG_INFINITY;
            for edge in next_edges {
                if !can_reach_end[edge.end] {
                    continue;
                }
                if let Some(&slot) = id_to_slot.get(&edge.token_id) {
                    best_edge_delta = best_edge_delta.max(row_slice[slot] + edge.channel_cost);
                }
            }
            let edge_floor = best_edge_delta - bcfg.step_edge_margin;

            for edge in next_edges {
                if !can_reach_end[edge.end] {
                    continue;
                }
                let Some(&slot) = id_to_slot.get(&edge.token_id) else {
                    continue;
                };
                let step_lp0 = row_slice[slot];
                if step_lp0 + edge.channel_cost < edge_floor {
                    continue;
                }
                let span_lower = &lower_input[pos..edge.end];
                let (next_kv, pending_token, step_lp) = if edge.is_english
                    && let Some((_, mtids)) = pinyin_index.lookup_en_multi(span_lower)
                    && mtids.len() >= 2
                    && let Some((mkv, last_tid, full_lp)) =
                        advance_en_multi(mlx_model, &new_kv, mtids, step_lp0, lora_scales)
                {
                    (mkv, last_tid, full_lp)
                } else {
                    (new_kv.clone(), edge.token_id, step_lp0)
                };
                let new_lm_lp = parent.lm_lp + step_lp;
                let new_chan = parent.channel_cost + edge.channel_cost;
                let new_tot = new_lm_lp + new_chan;
                let text = append_token_surface(
                    &parent.text,
                    pinyin_index.token_str(edge.token_id),
                    edge.is_english,
                    pangu_spacing,
                    span_lower,
                    pinyin_index,
                );
                let step_surprisal = -(step_lp + edge.channel_cost);
                let spike1 = parent.with_recorded_step(step_surprisal, pos, edge.end);
                buckets[edge.end].push(BeamHyp {
                    text,
                    parent_kv: Some(next_kv),
                    pending_token,
                    lm_lp: new_lm_lp,
                    channel_cost: new_chan,
                    best_path_total: new_tot,
                    is_exact: parent.is_exact && edge.is_exact,
                    spike1,
                });
            }
        }
    }

    // Fallback for curated English/tech prefix completions (`chatg -> ChatGPT`, `kubern ->
    // Kubernetes`) when no full-span Pinyin parse reached `n`.
    if let Some(comp) = pinyin_index.complete_en_prefix(lower_input) {
        return SingleDecodeOut {
            cands: vec![NeuralBeamCandidate {
                text: comp.to_string(),
                score: -1.0,
                lm_log_prob: -1.0,
                channel_cost: 0.0,
                is_exact: true,
                consumed_bytes: n,
            }],
            top_spike1: SurprisalSpike::default(),
        };
    }

    empty_out()
}

fn is_obvious_traditional_char(c: char) -> bool {
    matches!(
        c,
        '這' | '個'
            | '來'
            | '時'
            | '說'
            | '為'
            | '會'
            | '沒'
            | '過'
            | '點'
            | '長'
            | '經'
            | '進'
            | '開'
            | '關'
            | '學'
            | '機'
            | '體'
            | '國'
            | '發'
            | '動'
            | '對'
            | '當'
            | '還'
            | '從'
            | '與'
            | '樣'
            | '們'
    )
}

fn apply_closure_bonus(
    mlx_model: &MlxTransformer,
    root_kv: &MlxBeamKv,
    active: &mut [BeamHyp],
    closure_token_ids: &[u32],
    closure_weight: f32,
    lora_scales: &[f32],
) {
    let m = closure_token_ids.len();
    let parents: Vec<&MlxBeamKv> = active
        .iter()
        .map(|h| h.parent_kv.as_ref().unwrap_or(root_kv))
        .collect();
    let step_toks: Vec<u32> = active.iter().map(|h| h.pending_token).collect();
    let Ok((lps_flat, _)) =
        mlx_model.step_beams_gather_lps(&parents, &step_toks, closure_token_ids, lora_scales)
    else {
        return;
    };
    for (b_idx, hyp) in active.iter_mut().enumerate() {
        let p_slice = &lps_flat[b_idx * m..(b_idx + 1) * m];
        let max_lp = p_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_lp.is_finite() {
            let sum_exp: f32 = p_slice.iter().map(|&x| (x - max_lp).exp()).sum();
            let closure_lp = (max_lp + sum_exp.max(1e-12).ln()).clamp(-9.5, 0.0);
            hyp.lm_lp += closure_weight * closure_lp;
        }
    }
}
