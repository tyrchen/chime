//! Pinyin-to-Token Trie over the 36,864-token Qwen3 student vocabulary for Route A
//! (syllable-synchronous Pinyin-constrained autoregressive neural beam search).

use std::{collections::HashMap, fs, path::Path};

use chime_phonetics::{
    FuzzyConfig, ShuangpinScheme, SyllableId, TypoEdit, can_segment_exact,
    can_segment_strict_orthography, canonical_syllable, expand_shuangpin_pair,
    generate_typo_variants, lookup_syllable,
};
use serde::Deserialize;
use smallvec::SmallVec;

use crate::NeuralError;

/// Acoustic and keystroke channel log-probability parameters for [`TokenPinyinIndex`] lattice
/// construction (centralized to eliminate scattered magic numbers and length cliffs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelConfig {
    /// Additional log-prob cost layered on top of `FuzzyConfig::fuzzy_penalty`.
    pub fuzzy_extra_cost: f32,
    /// Base channel cost for typo recovery on non-canonical spans (`shaung`, `meo`, `tonh`).
    pub typo_noncanonical_base_cost: f32,
    /// Base channel cost for immediate QWERTY neighbor recovery on canonical syllables.
    pub typo_canonical_neighbor_base_cost: f32,
    /// Maximum Euclidean keyboard edit cost accepted for canonical-syllable neighbor recovery.
    pub max_canonical_neighbor_dist_cost: f32,
    /// Channel cost for shared-boundary nasal/velar consonant elision (`chunuan -> chun+nuan`).
    pub geminate_elision_cost: f32,
    /// Log-prob penalty applied to Traditional-only CJK tokens when left context is Simplified.
    pub traditional_penalty: f32,
    /// Maximum motor-typo syllables permitted inside a single multi-syllable vocabulary token.
    pub max_typos_per_token: u8,
    /// When `true`, also expand immediate QWERTY neighbor-key substitutions on already-valid
    /// canonical syllables (`shuruga -> shurufa`). Enabled on-demand via surprisal-spike rescue so
    /// clean sentences pay zero fan-out overhead regardless of sentence length.
    pub enable_canonical_neighbor_typos: bool,
    /// Optional byte window `(start, end)` restricting canonical neighbor-key expansion to the
    /// localized surprisal-spike region (`None` expands across the full buffer).
    pub canonical_neighbor_window: Option<(usize, usize)>,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            fuzzy_extra_cost: 0.85,
            typo_noncanonical_base_cost: 2.10,
            typo_canonical_neighbor_base_cost: 3.80,
            max_canonical_neighbor_dist_cost: 1.35,
            geminate_elision_cost: 2.40,
            traditional_penalty: 4.80,
            max_typos_per_token: 1,
            enable_canonical_neighbor_typos: false,
            canonical_neighbor_window: None,
        }
    }
}

/// A single phonetically valid token transition spanning `start..end` in the raw keystroke buffer.
#[derive(Debug, Clone, Copy)]
pub struct TokenEdge {
    /// Target vocabulary token ID (`0..vocab_size`).
    pub token_id: u32,
    /// Exclusive byte offset in `raw_input` consumed by this token.
    pub end: usize,
    /// Acoustic / keystroke channel log-probability penalty (`0.0` for exact match, `< 0.0` for
    /// fuzzy pinyin or motor typos).
    pub channel_cost: f32,
    /// True if this token matched via exact canonical Pinyin / Shuangpin / exact English.
    pub is_exact: bool,
    /// True if this is an ASCII/English token.
    pub is_english: bool,
}

#[derive(Debug, Deserialize)]
struct RawTokenPinyinFile {
    cjk: Vec<(u32, Vec<Vec<String>>)>,
    #[serde(default)]
    trad: Vec<u32>,
    en: Vec<(u32, String)>,
    #[serde(default)]
    en_multi: Vec<(String, String, Vec<u32>)>,
}

#[derive(Debug, Clone, Default)]
struct TrieNode {
    children: HashMap<SyllableId, usize>,
    token_ids: Vec<u32>,
}

/// A multi-character CJK vocabulary token that heals a 1-char left-context boundary
/// (e.g. `left_context` ending in `'一'` + token `"一些"` matching suffix `"xie"` -> `"些"`).
#[derive(Debug, Clone)]
pub struct HealedTokenEntry {
    /// Vocabulary token ID of the full merged token `T` (e.g. `tok("一些")`, `tok("珊瑚")`).
    pub token_id: u32,
    /// Suffix string emitted after `first_char` (e.g. `"些"`, `"瑚"`, `"业"`).
    pub suffix_text: String,
    /// Canonical Mandarin syllable sequence of `suffix_text`.
    pub suffix_syls: SmallVec<[SyllableId; 3]>,
}

/// Pre-indexed Pinyin-to-Token Trie and English surface index built from `token_pinyin.json` +
/// `vocab.json`.
#[derive(Debug, Clone)]
pub struct TokenPinyinIndex {
    id_to_token: Vec<String>,
    is_trad: Vec<bool>,
    nodes: Vec<TrieNode>,
    /// First CJK character of every 2-4 char vocab token -> healed suffix entries for exact
    /// Bayesian left-boundary Token Healing (`P(B | A, C) = P(AB | C) / P(A | C)`).
    by_first_char: HashMap<char, Vec<HealedTokenEntry>>,
    /// Initial-letter skeleton (e.g. `"wsm"` -> `为什么`, `"rgzn"` -> `人工智能`) -> token IDs.
    skeleton_exact: HashMap<String, Vec<u32>>,
    /// Single-character simplified CJK tokens indexed by first Pinyin letter (`'a'..='z'`) for
    /// autoregressive multi-token skeleton decoding (`srf` -> `输入` + `法`, `mywt` -> `没有` +
    /// `问题`).
    skeleton_single: [Vec<u32>; 26],
    /// Lowercase ASCII word -> list of vocabulary token IDs.
    en_exact: HashMap<String, Vec<u32>>,
    /// Lowercase tech/brand term -> canonical display casing (`"github" -> "GitHub"`).
    en_canonical_case: HashMap<String, String>,
    /// Lowercase multi-token tech term -> `(display_surface, token_ids)`.
    en_multi: HashMap<String, (String, Vec<u32>)>,
}

impl TokenPinyinIndex {
    /// Loads `token_pinyin.json` alongside `vocab_tokens` from `model_dir`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError`] if `token_pinyin.json` cannot be read or parsed.
    #[allow(clippy::too_many_lines)]
    pub fn load_from_dir(model_dir: &Path, vocab_tokens: Vec<String>) -> Result<Self, NeuralError> {
        let raw_str = fs::read_to_string(model_dir.join("token_pinyin.json"))?;
        let parsed: RawTokenPinyinFile = serde_json::from_str(&raw_str)?;

        let mut is_trad = vec![false; vocab_tokens.len()];
        for tid in parsed.trad {
            if let Some(slot) = is_trad.get_mut(tid as usize) {
                *slot = true;
            }
        }

        let mut nodes = vec![TrieNode::default()];
        let mut by_first_char: HashMap<char, Vec<HealedTokenEntry>> = HashMap::new();
        let mut skeleton_exact: HashMap<String, Vec<u32>> = HashMap::new();
        let mut skeleton_single: [Vec<u32>; 26] = std::array::from_fn(|_| Vec::new());
        for (tid, readings) in parsed.cjk {
            if (tid as usize) >= vocab_tokens.len() {
                continue;
            }
            let tok_str = &vocab_tokens[tid as usize];
            let tok_chars: SmallVec<[char; 4]> = tok_str.chars().collect();
            let tid_is_trad = is_trad.get(tid as usize).copied().unwrap_or(false);
            for syl_list in readings {
                if syl_list.is_empty() {
                    continue;
                }
                let mut cur = 0usize;
                let mut sids: SmallVec<[SyllableId; 4]> = SmallVec::new();
                let mut ok = true;
                for s in &syl_list {
                    if let Some(sid) = lookup_syllable(s) {
                        sids.push(sid);
                        let next = if let Some(&child) = nodes[cur].children.get(&sid) {
                            child
                        } else {
                            let new_idx = nodes.len();
                            nodes.push(TrieNode::default());
                            nodes[cur].children.insert(sid, new_idx);
                            new_idx
                        };
                        cur = next;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok && !nodes[cur].token_ids.contains(&tid) {
                    nodes[cur].token_ids.push(tid);
                    if (2..=4).contains(&tok_chars.len()) && sids.len() == tok_chars.len() {
                        let first_ch = tok_chars[0];
                        let suffix_text: String = tok_chars[1..].iter().collect();
                        let suffix_syls: SmallVec<[SyllableId; 3]> =
                            sids[1..].iter().copied().collect();
                        let bucket = by_first_char.entry(first_ch).or_default();
                        if !bucket
                            .iter()
                            .any(|e| e.token_id == tid && e.suffix_syls == suffix_syls)
                        {
                            bucket.push(HealedTokenEntry {
                                token_id: tid,
                                suffix_text,
                                suffix_syls,
                            });
                        }
                    }
                    if syl_list.len() == 1 && !tid_is_trad {
                        if let Some(b) = syl_list[0].bytes().next()
                            && b.is_ascii_lowercase()
                        {
                            let slot = &mut skeleton_single[(b - b'a') as usize];
                            if !slot.contains(&tid) {
                                slot.push(tid);
                            }
                        }
                    } else if (2..=4).contains(&syl_list.len()) {
                        let skel: String = syl_list
                            .iter()
                            .filter_map(|s| s.bytes().next().map(char::from))
                            .collect();
                        if skel.len() == syl_list.len() {
                            let list = skeleton_exact.entry(skel).or_default();
                            if !list.contains(&tid) {
                                list.push(tid);
                            }
                        }
                    }
                }
            }
        }

        let mut en_exact: HashMap<String, Vec<u32>> = HashMap::with_capacity(parsed.en.len());
        for (tid, lower) in parsed.en {
            if (tid as usize) >= vocab_tokens.len() {
                continue;
            }
            en_exact.entry(lower).or_default().push(tid);
        }

        let mut en_canonical_case: HashMap<String, String> =
            HashMap::with_capacity(parsed.en_multi.len());
        let mut en_multi: HashMap<String, (String, Vec<u32>)> =
            HashMap::with_capacity(parsed.en_multi.len());
        for (lower, display, tids) in parsed.en_multi {
            en_canonical_case.insert(lower.clone(), display.clone());
            if tids.len() == 1 {
                let list = en_exact.entry(lower).or_default();
                if !list.contains(&tids[0]) {
                    list.push(tids[0]);
                }
            } else if !tids.is_empty() {
                en_multi.insert(lower, (display, tids));
            }
        }

        Ok(Self {
            id_to_token: vocab_tokens,
            is_trad,
            nodes,
            by_first_char,
            skeleton_exact,
            skeleton_single,
            en_exact,
            en_canonical_case,
            en_multi,
        })
    }

    /// Finds all multi-character vocabulary tokens starting with `prev_char` whose remaining
    /// suffix matches a valid Pinyin prefix `raw_input[0..end]`, returning
    /// `(token_id, suffix_text, end_byte, channel_cost, is_exact)`.
    #[must_use]
    pub fn find_healed_boundary_matches(
        &self,
        prev_char: char,
        raw_input: &str,
        scheme: ShuangpinScheme,
        fuzzy: FuzzyConfig,
        allow_traditional: bool,
    ) -> Vec<(u32, String, usize, f32, bool)> {
        let Some(entries) = self.by_first_char.get(&prev_char) else {
            return Vec::new();
        };
        let n = raw_input.len();
        if n == 0 {
            return Vec::new();
        }
        let lower = raw_input.to_ascii_lowercase();
        let bytes = lower.as_bytes();

        // Build syllable edges for `0..n.min(18)` (suffix of a 2-4 char token is at most 3 syls).
        let cap = n.min(18);
        let mut syl_edges: Vec<Vec<(usize, SyllableId, f32, bool)>> = vec![Vec::new(); cap];
        for start in 0..cap {
            if bytes[start] == b'\'' {
                continue;
            }
            if scheme != ShuangpinScheme::Full {
                if start + 2 <= cap && bytes[start + 1] != b'\'' {
                    for sid in expand_shuangpin_pair(scheme, bytes[start], bytes[start + 1]) {
                        syl_edges[start].push((start + 2, sid, 0.0, true));
                    }
                }
                continue;
            }
            let max_end = (start + 6).min(cap);
            for end in (start + 1)..=max_end {
                if bytes[end - 1] == b'\'' {
                    break;
                }
                let slice = &lower[start..end];
                if let Some(sid) = lookup_syllable(slice) {
                    syl_edges[start].push((end, sid, 0.0, true));
                    for (fid, fcost) in fuzzy.expand(slice) {
                        if fid != sid {
                            syl_edges[start].push((end, fid, -fcost - 0.85, false));
                        }
                    }
                }
            }
        }

        let mut out = Vec::new();
        for entry in entries {
            if !allow_traditional && self.is_traditional_token(entry.token_id) {
                continue;
            }
            // Match `entry.suffix_syls` sequentially from `pos = 0`.
            let mut frontier: SmallVec<[(usize, f32, bool); 4]> = SmallVec::new();
            frontier.push((0, 0.0, true));
            for &need_sid in &entry.suffix_syls {
                let mut next_frontier: SmallVec<[(usize, f32, bool); 4]> = SmallVec::new();
                for (pos, cost, exact) in frontier {
                    let p = if pos < cap && bytes[pos] == b'\'' {
                        pos + 1
                    } else {
                        pos
                    };
                    if p >= cap {
                        continue;
                    }
                    for &(end, sid, scost, sexact) in &syl_edges[p] {
                        if sid == need_sid {
                            next_frontier.push((end, cost + scost, exact && sexact));
                        }
                    }
                }
                frontier = next_frontier;
                if frontier.is_empty() {
                    break;
                }
            }
            for (end, cost, exact) in frontier {
                out.push((entry.token_id, entry.suffix_text.clone(), end, cost, exact));
            }
        }
        out
    }

    /// Looks up exact initial-letter skeleton matches (`"wsm"` -> `为什么`, `"rgzn"` ->
    /// `人工智能`).
    #[must_use]
    pub fn lookup_skeleton(&self, lower: &str) -> &[u32] {
        self.skeleton_exact.get(lower).map_or(&[], Vec::as_slice)
    }

    /// Builds a multi-token initial-letter skeleton lattice over `lower` (e.g. `srf` -> `输入` +
    /// `法`, `mywt` -> `没有` + `问题`, `jttq` -> `今天` + `天气`) for autoregressive neural beam
    /// decoding of arbitrary 2-6 letter Pinyin abbreviations.
    #[must_use]
    pub fn build_skeleton_lattice(
        &self,
        lower: &str,
        allow_traditional: bool,
    ) -> Vec<Vec<TokenEdge>> {
        let n = lower.len();
        let bytes = lower.as_bytes();
        let mut edges_by_start: Vec<Vec<TokenEdge>> = vec![Vec::new(); n];
        for start in 0..n {
            if !bytes[start].is_ascii_lowercase() {
                continue;
            }
            let max_end = (start + 4).min(n);
            for end in (start + 1)..=max_end {
                let slice = &lower[start..end];
                if end - start == 1 {
                    let b = bytes[start];
                    for &tid in &self.skeleton_single[(b - b'a') as usize] {
                        edges_by_start[start].push(TokenEdge {
                            token_id: tid,
                            end,
                            channel_cost: -0.35,
                            is_exact: true,
                            is_english: false,
                        });
                    }
                } else if let Some(tids) = self.skeleton_exact.get(slice) {
                    for &tid in tids {
                        let trad_pen = if !allow_traditional && self.is_traditional_token(tid) {
                            -4.80
                        } else {
                            0.0
                        };
                        edges_by_start[start].push(TokenEdge {
                            token_id: tid,
                            end,
                            channel_cost: trad_pen,
                            is_exact: true,
                            is_english: false,
                        });
                    }
                }
            }
            edges_by_start[start].sort_unstable_by_key(|e| (e.end, e.token_id));
        }
        edges_by_start
    }

    /// Returns true if `token_id` is flagged as a Traditional-only CJK variant.
    #[must_use]
    pub fn is_traditional_token(&self, token_id: u32) -> bool {
        self.is_trad
            .get(token_id as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Returns canonical brand/tech casing for a lowercase English token (e.g. `"github"` ->
    /// `"GitHub"`, `"macos"` -> `"macOS"`).
    #[must_use]
    pub fn canonical_en_case(&self, lower: &str) -> Option<&str> {
        self.en_canonical_case.get(lower).map(String::as_str)
    }

    /// Returns any multi-token curated tech/English terms matching lowercase `slice`.
    #[must_use]
    pub fn lookup_en_multi(&self, lower: &str) -> Option<&(String, Vec<u32>)> {
        self.en_multi.get(lower)
    }

    /// Returns curated tech/English prefix completions (e.g. `"chatg"` -> `"ChatGPT"`,
    /// `"kubern"` -> `"Kubernetes"`).
    #[must_use]
    pub fn complete_en_prefix(&self, lower: &str) -> Option<&str> {
        if lower.len() < 4 {
            return None;
        }
        let mut best: Option<&str> = None;
        for (k, disp) in &self.en_canonical_case {
            if k.starts_with(lower)
                && k.len() > lower.len()
                && best.is_none_or(|b| disp.len() < b.len())
            {
                best = Some(disp.as_str());
            }
        }
        best
    }

    /// Returns the UTF-8 token surface string for `token_id`.
    #[must_use]
    pub fn token_str(&self, token_id: u32) -> &str {
        self.id_to_token
            .get(token_id as usize)
            .map_or("", String::as_str)
    }

    /// Builds the acoustic syllable lattice using default [`ChannelConfig`] and expands it through
    /// the Pinyin-to-Token Trie to produce `edges_by_start[pos]` for `0..raw_input.len()`.
    #[must_use]
    pub fn build_token_lattice(
        &self,
        raw_input: &str,
        scheme: ShuangpinScheme,
        fuzzy: FuzzyConfig,
        allow_traditional: bool,
    ) -> Vec<Vec<TokenEdge>> {
        self.build_token_lattice_with_config(
            raw_input,
            scheme,
            fuzzy,
            allow_traditional,
            ChannelConfig::default(),
        )
    }

    /// Builds the acoustic syllable lattice with explicit [`ChannelConfig`] parameters.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn build_token_lattice_with_config(
        &self,
        raw_input: &str,
        scheme: ShuangpinScheme,
        fuzzy: FuzzyConfig,
        allow_traditional: bool,
        cfg: ChannelConfig,
    ) -> Vec<Vec<TokenEdge>> {
        type SylEdge = (usize, SyllableId, f32, bool, bool);
        type TrieWalkState = (usize, usize, f32, bool, u8, u8);

        let n = raw_input.len();
        let mut edges_by_start: Vec<Vec<TokenEdge>> = vec![Vec::new(); n];
        if n == 0 {
            return edges_by_start;
        }
        let lower = raw_input.to_ascii_lowercase();
        let bytes = lower.as_bytes();
        let full_is_canonical =
            scheme == ShuangpinScheme::Full && can_segment_strict_orthography(&lower);

        // 1. Build syllable-level acoustic channel edges: `(end, SyllableId, channel_cost,
        //    is_exact, is_typo)`
        let mut syl_edges: Vec<Vec<SylEdge>> = vec![Vec::new(); n];
        for start in 0..n {
            if bytes[start] == b'\'' {
                continue;
            }
            if scheme != ShuangpinScheme::Full {
                if start + 2 <= n && bytes[start] != b'\'' && bytes[start + 1] != b'\'' {
                    for sid in expand_shuangpin_pair(scheme, bytes[start], bytes[start + 1]) {
                        syl_edges[start].push((start + 2, sid, 0.0, true, false));
                    }
                }
                continue;
            }

            let max_end = (start + 7).min(n);
            for end in (start + 1)..=max_end {
                if bytes[end - 1] == b'\'' {
                    break;
                }
                let slice = &lower[start..end];
                let canonical = if slice.len() <= 6 {
                    lookup_syllable(slice)
                } else {
                    None
                };
                if let Some(sid) = canonical {
                    syl_edges[start].push((end, sid, 0.0, true, false));
                    // Shared-boundary nasal/velar elision (`chunuan -> chun + nuan`,
                    // `zhonguoren -> zhong + guo + ren`, `bujinchangjiangungunlai`): when `slice`
                    // ends in `n` or `ng` and is immediately followed by a vowel, allow sharing
                    // the final consonant (`n` or `g`) with the next syllable at a clean channel
                    // penalty!
                    if (slice.ends_with('n') || slice.ends_with("ng"))
                        && end < n
                        && matches!(bytes[end], b'a' | b'e' | b'i' | b'o' | b'u' | b'v')
                    {
                        syl_edges[start].push((
                            end - 1,
                            sid,
                            -cfg.geminate_elision_cost,
                            false,
                            true,
                        ));
                    }
                    // Tight single-key QWERTY neighbor recovery on valid syllables (only when
                    // explicitly enabled by surprisal-spike rescue — zero overhead on clean
                    // sentences and zero arbitrary sentence-length cutoff!).
                    let in_neighbor_window = cfg
                        .canonical_neighbor_window
                        .is_none_or(|(lo, hi)| end > lo && start < hi);
                    if cfg.enable_canonical_neighbor_typos && in_neighbor_window && slice.len() >= 2
                    {
                        for (tid_syl, edit) in generate_typo_variants(slice) {
                            if tid_syl != sid
                                && matches!(
                                    edit,
                                    TypoEdit::NeighborKey { cost }
                                        if cost <= cfg.max_canonical_neighbor_dist_cost
                                )
                            {
                                let cost = -edit.cost() - cfg.typo_canonical_neighbor_base_cost;
                                syl_edges[start].push((end, tid_syl, cost, false, true));
                            } else if tid_syl != sid
                                && matches!(edit, TypoEdit::AdjacentSwap { .. })
                            {
                                let cost =
                                    -edit.cost() - (cfg.typo_canonical_neighbor_base_cost - 0.35);
                                syl_edges[start].push((end, tid_syl, cost, false, true));
                            }
                        }
                    }
                }
                // Always run fuzzy expansion for `2..=6` EVEN when `slice` itself is NOT a
                // canonical Mandarin syllable (e.g. `suang -> shuang`, `diang -> dian`,
                // `zuang -> zhuang`, `cuang -> chuang`, `tiang -> tian`)!
                // On a standalone canonical single-syllable input (`gen`), exact-pinyin characters
                // must strongly lead fuzzy equivalents (`geng -> 更`) so cold-start unigram skew
                // never usurps Slot #1.
                if (2..=6).contains(&slice.len()) {
                    let single_syl_exact_extra = if start == 0 && end == n && canonical.is_some() {
                        2.0
                    } else {
                        0.0
                    };
                    for (fid, fcost) in fuzzy.expand(slice) {
                        if Some(fid) != canonical {
                            syl_edges[start].push((
                                end,
                                fid,
                                -fcost - cfg.fuzzy_extra_cost - single_syl_exact_extra,
                                false,
                                false,
                            ));
                        }
                    }
                }
                if canonical.is_none() && slice.len() >= 2 {
                    // Always allow transposition (`tina -> tian`, `yuo -> you`, `zia -> zai`,
                    // `yign -> ying`) and extra-repeat typos on non-syllable spans; allow QWERTY
                    // neighbor substitutions whenever strict orthography fails (`suoyicaimeoqu`)
                    // OR inside an on-demand localized surprisal-spike rescue window (e.g.
                    // `qianean` falsely segmenting as `qian+e+an` while intending `qian+wan`).
                    let in_neighbor_window = cfg
                        .canonical_neighbor_window
                        .is_none_or(|(lo, hi)| end > lo && start < hi);
                    let allow_neighbor = !full_is_canonical
                        || (cfg.enable_canonical_neighbor_typos && in_neighbor_window);
                    for (tid_syl, edit) in generate_typo_variants(slice) {
                        let is_swap_or_rep = matches!(
                            edit,
                            TypoEdit::AdjacentSwap { .. } | TypoEdit::ExtraRepeat { .. }
                        );
                        if allow_neighbor || is_swap_or_rep {
                            let cost = -edit.cost() - cfg.typo_noncanonical_base_cost;
                            syl_edges[start].push((end, tid_syl, cost, false, true));
                            for (fid, fcost) in fuzzy.expand(canonical_syllable(tid_syl)) {
                                if fid != tid_syl {
                                    syl_edges[start].push((end, fid, cost - fcost, false, true));
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Walk the Pinyin-to-Token Trie from each `start` over `syl_edges`.
        for start in 0..n {
            if bytes[start] == b'\'' {
                continue;
            }
            // Stack: `(cur_byte, trie_node_idx, acc_cost, acc_exact, typo_cnt, depth)`
            let mut stack: SmallVec<[TrieWalkState; 64]> = SmallVec::new();
            for &(end1, sid1, cost1, exact1, is_typo1) in &syl_edges[start] {
                if let Some(&child) = self.nodes[0].children.get(&sid1) {
                    stack.push((end1, child, cost1, exact1, u8::from(is_typo1), 1));
                }
            }

            // Dedup best `(channel_cost, is_exact)` per `(token_id, end)` at this `start`.
            let mut best_map: HashMap<(u32, usize), (f32, bool, bool)> = HashMap::new();

            while let Some((cur_end, node_idx, acc_cost, acc_exact, typo_cnt, depth)) = stack.pop()
            {
                let node = &self.nodes[node_idx];
                for &tid in &node.token_ids {
                    let trad_pen = if !allow_traditional && self.is_traditional_token(tid) {
                        -cfg.traditional_penalty
                    } else {
                        0.0
                    };
                    let eff_cost = acc_cost + trad_pen;
                    let entry =
                        best_map
                            .entry((tid, cur_end))
                            .or_insert((f32::NEG_INFINITY, false, false));
                    if eff_cost > entry.0
                        || ((eff_cost - entry.0).abs() < 1e-6 && acc_exact && !entry.1)
                    {
                        *entry = (eff_cost, acc_exact, false);
                    }
                }

                if depth < 5 && cur_end < n {
                    let next_pos = if bytes[cur_end] == b'\'' {
                        cur_end + 1
                    } else {
                        cur_end
                    };
                    if next_pos < n {
                        for &(end2, sid2, cost2, exact2, is_typo2) in &syl_edges[next_pos] {
                            // Allow multiple fuzzy syllables (`zenzeng -> 真正`, `senzen -> 深圳`,
                            // `zici -> 支持`) inside a multi-syllable token, while capping motor
                            // typos to at most 1 per token!
                            let next_typos = typo_cnt + u8::from(is_typo2);
                            if next_typos > cfg.max_typos_per_token {
                                continue;
                            }
                            if let Some(&child) = node.children.get(&sid2) {
                                stack.push((
                                    end2,
                                    child,
                                    acc_cost + cost2,
                                    acc_exact && exact2,
                                    next_typos,
                                    depth + 1,
                                ));
                            }
                        }
                    }
                }
            }

            // 3. Match exact English/ASCII vocabulary tokens spanning `start..end` (for mixed
            //    Chinese-English sentences like `bangworeviewzhegepr` or `woshiyongmacoshegithub`).
            if bytes[start].is_ascii_alphabetic() {
                let max_en = (start + 16).min(n);
                for end in (start + 2)..=max_en {
                    if !bytes[end - 1].is_ascii_alphanumeric() {
                        break;
                    }
                    let slice = &lower[start..end];
                    let is_known_tech = self.en_canonical_case.contains_key(slice);
                    if let Some((_, mtids)) = self.en_multi.get(slice)
                        && let Some(&first_tid) = mtids.first()
                    {
                        // Multi-token tech/English word (`Obsidian`, `ChatGPT`, `Kubernetes`,
                        // `TypeScript`, `PyTorch`, `VIP`, `hover`, ...): beam search advances
                        // the KV-cache through every token in `mtids`, so score with the exact
                        // same known-tech prior as single-token tech terms!
                        let entry = best_map.entry((first_tid, end)).or_insert((
                            f32::NEG_INFINITY,
                            false,
                            true,
                        ));
                        if 1.15 > entry.0 {
                            *entry = (1.15, true, true);
                        }
                    } else if let Some(tids) = self.en_exact.get(slice) {
                        let is_embedded_lower =
                            !(start == 0 && end == n) && &raw_input[start..end] == slice;
                        let en_pen = if is_known_tech {
                            0.35
                        } else if slice.len() <= 2 && lookup_syllable(slice).is_some() {
                            -2.80
                        } else if slice.len() <= 3 && can_segment_exact(slice) {
                            -1.80
                        } else if is_embedded_lower
                            && chime_phonetics::classify_code_switch(slice)
                                == chime_phonetics::CodeSwitchSignal::AmbiguousAscii
                        {
                            -2.20
                        } else {
                            -0.65
                        };
                        for &tid in tids {
                            let entry = best_map.entry((tid, end)).or_insert((
                                f32::NEG_INFINITY,
                                false,
                                true,
                            ));
                            if en_pen > entry.0 {
                                *entry = (en_pen, true, true);
                            }
                        }
                    }
                }
            }

            let edges = &mut edges_by_start[start];
            edges.reserve(best_map.len());
            for ((token_id, end), (channel_cost, is_exact, is_english)) in best_map {
                edges.push(TokenEdge {
                    token_id,
                    end,
                    channel_cost,
                    is_exact,
                    is_english,
                });
            }
            // Deterministic ordering by `(end, token_id)`
            edges.sort_unstable_by_key(|e| (e.end, e.token_id));
        }

        edges_by_start
    }

    /// Aligns `candidate_text` against a pre-built token lattice, returning the minimum phonetic /
    /// keystroke channel cost `(channel_cost, is_exact)`.
    #[must_use]
    pub fn exact_channel_cost_with_lattice(
        &self,
        raw_input: &str,
        candidate_text: &str,
        lattice: &[Vec<TokenEdge>],
    ) -> Option<(f32, bool)> {
        let n = raw_input.len();
        if n == 0 || candidate_text.is_empty() || lattice.len() != n {
            return None;
        }
        // Strip Pangu ASCII spaces inserted between CJK/ASCII boundaries so pure token strings
        // align cleanly against `self.id_to_token`.
        let compact: String = candidate_text.chars().filter(|&c| c != ' ').collect();
        let m = compact.len();
        if m == 0 {
            return None;
        }
        let lower_raw = raw_input.to_ascii_lowercase();
        let raw_b = raw_input.as_bytes();
        // Sparse DP over `(raw_pos, text_byte_pos) -> (best_cost, is_exact)`.
        let mut dp: HashMap<(usize, usize), (f32, bool)> = HashMap::new();
        dp.insert((0, 0), (0.0, true));
        for pos in 0..n {
            let mut cur_states: SmallVec<[(usize, f32, bool); 8]> = SmallVec::new();
            for (&(rp, tp), &(cost, ex)) in &dp {
                if rp == pos {
                    cur_states.push((tp, cost, ex));
                }
            }
            if cur_states.is_empty() {
                continue;
            }
            if raw_b[pos] == b'\'' {
                for (tp, cost, ex) in cur_states {
                    let e = dp
                        .entry((pos + 1, tp))
                        .or_insert((f32::NEG_INFINITY, false));
                    if cost > e.0 {
                        *e = (cost, ex);
                    }
                }
                continue;
            }
            for (tp, cost, ex) in cur_states {
                let rem = &compact[tp..];
                if rem.is_empty() {
                    continue;
                }
                for edge in &lattice[pos] {
                    let matched_len = if edge.is_english {
                        let surf = self
                            .canonical_en_case(&lower_raw[pos..edge.end])
                            .unwrap_or_else(|| {
                                self.token_str(edge.token_id).trim_start_matches(' ')
                            });
                        if !surf.is_empty()
                            && rem.len() >= surf.len()
                            && rem.is_char_boundary(surf.len())
                            && rem[..surf.len()].eq_ignore_ascii_case(surf)
                        {
                            Some(surf.len())
                        } else {
                            None
                        }
                    } else {
                        let tok = self.token_str(edge.token_id);
                        if !tok.is_empty() && rem.starts_with(tok) {
                            Some(tok.len())
                        } else {
                            None
                        }
                    };
                    if let Some(mlen) = matched_len {
                        let next_tp = tp + mlen;
                        let next_cost = cost + edge.channel_cost;
                        let next_ex = ex && edge.is_exact;
                        let e = dp
                            .entry((edge.end, next_tp))
                            .or_insert((f32::NEG_INFINITY, false));
                        if next_cost > e.0 || ((next_cost - e.0).abs() < 1e-6 && next_ex && !e.1) {
                            *e = (next_cost, next_ex);
                        }
                    }
                }
            }
        }
        dp.get(&(n, m)).copied()
    }

    /// Computes the exact minimum acoustic/keystroke channel cost `(channel_cost, is_exact)` for
    /// aligning `candidate_text` to `raw_input` under the exact same [`ChannelConfig`] &
    /// [`FuzzyConfig`] used by Route A beam search.
    #[must_use]
    pub fn exact_channel_cost(
        &self,
        raw_input: &str,
        candidate_text: &str,
        scheme: ShuangpinScheme,
        fuzzy: FuzzyConfig,
        allow_traditional: bool,
        cfg: ChannelConfig,
    ) -> Option<(f32, bool)> {
        let primary_lattice =
            self.build_token_lattice_with_config(raw_input, scheme, fuzzy, allow_traditional, cfg);
        if let Some(res) =
            self.exact_channel_cost_with_lattice(raw_input, candidate_text, &primary_lattice)
        {
            return Some(res);
        }
        if !cfg.enable_canonical_neighbor_typos && scheme == ShuangpinScheme::Full {
            let neighbor_lattice = self.build_token_lattice_with_config(
                raw_input,
                scheme,
                fuzzy,
                allow_traditional,
                ChannelConfig {
                    enable_canonical_neighbor_typos: true,
                    canonical_neighbor_window: None,
                    ..cfg
                },
            );
            return self.exact_channel_cost_with_lattice(
                raw_input,
                candidate_text,
                &neighbor_lattice,
            );
        }
        None
    }
}
