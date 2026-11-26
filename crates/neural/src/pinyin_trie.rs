//! Pinyin-to-Token Trie over the 36,864-token Qwen3 student vocabulary for Route A
//! (syllable-synchronous Pinyin-constrained autoregressive neural beam search).

use std::{collections::HashMap, fs, path::Path};

use chime_phonetics::{
    FuzzyConfig, ShuangpinScheme, SyllableId, TypoEdit, can_segment_exact, canonical_syllable,
    expand_shuangpin_pair, generate_typo_variants, lookup_syllable,
};
use serde::Deserialize;
use smallvec::SmallVec;

use crate::NeuralError;

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

    /// Builds the acoustic syllable lattice and expands it through the Pinyin-to-Token Trie to
    /// produce `edges_by_start[pos]` for every byte position `0..raw_input.len()`.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn build_token_lattice(
        &self,
        raw_input: &str,
        scheme: ShuangpinScheme,
        fuzzy: FuzzyConfig,
        allow_traditional: bool,
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
        let full_is_canonical = scheme == ShuangpinScheme::Full && can_segment_exact(&lower);

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

            let max_end = (start + 6).min(n);
            for end in (start + 1)..=max_end {
                if bytes[end - 1] == b'\'' {
                    break;
                }
                let slice = &lower[start..end];
                let canonical = lookup_syllable(slice);
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
                        syl_edges[start].push((end - 1, sid, -2.40, false, true));
                    }
                    // Tight single-key QWERTY neighbor recovery on valid syllables (`shulufa ->
                    // shurufa`, `shuruga -> shurufa`, `biyao -> buyao`, `landaoni -> kandaoni`) at
                    // a strong `-4.45` channel cost, restricted to immediate physical neighbors
                    // (`edit.cost() <= 1.35`) so clean long sentences stay fast and never misfire!
                    if n <= 28 && slice.len() >= 2 {
                        for (tid_syl, edit) in generate_typo_variants(slice) {
                            if tid_syl != sid
                                && matches!(edit, TypoEdit::NeighborKey { cost } if cost <= 1.35)
                            {
                                let cost = -edit.cost() - 4.55;
                                syl_edges[start].push((end, tid_syl, cost, false, true));
                            }
                        }
                    }
                }
                // Always run fuzzy expansion for `len >= 2` EVEN when `slice` itself is NOT a
                // canonical Mandarin syllable (e.g. `suang -> shuang`, `diang -> dian`,
                // `zuang -> zhuang`, `cuang -> chuang`, `tiang -> tian`)!
                if slice.len() >= 2 {
                    for (fid, fcost) in fuzzy.expand(slice) {
                        if Some(fid) != canonical {
                            syl_edges[start].push((end, fid, -fcost - 0.85, false, false));
                        }
                    }
                }
                if canonical.is_none() && slice.len() >= 2 {
                    // Always allow transposition (`tina -> tian`, `yuo -> you`, `zia -> zai`,
                    // `yign -> ying`) and extra-repeat typos on non-syllable spans; allow QWERTY
                    // neighbor substitutions on non-canonical inputs. Also chain fuzzy expansion
                    // after typo repair so `mohuyign -> ying (typo) -> yin (fuzzy)` recovers!
                    for (tid_syl, edit) in generate_typo_variants(slice) {
                        let is_swap_or_rep = matches!(
                            edit,
                            TypoEdit::AdjacentSwap { .. } | TypoEdit::ExtraRepeat { .. }
                        );
                        if !full_is_canonical || is_swap_or_rep {
                            let cost = -edit.cost() - 2.10;
                            syl_edges[start].push((end, tid_syl, cost, false, true));
                            for (fid, fcost) in fuzzy.expand(canonical_syllable(tid_syl)) {
                                if fid != tid_syl {
                                    syl_edges[start].push((
                                        end,
                                        fid,
                                        cost - fcost - 0.85,
                                        false,
                                        true,
                                    ));
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
                        -4.80
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
                            if next_typos > 1 {
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
                        // Multi-token tech proper noun (`ChatGPT`, `Kubernetes`, `TypeScript`,
                        // `Redis`, `PyTorch`, `WebSocket`, `Protobuf`, `Safetensors`, `Colab`,
                        // `Cloudflare`, `Swift`, `macOS`): prefer canonical full-word span over
                        // fragmented subwords (`py` + `torch`, `saf` + `特` + `ensors`).
                        let entry = best_map.entry((first_tid, end)).or_insert((
                            f32::NEG_INFINITY,
                            false,
                            true,
                        ));
                        if 3.05 > entry.0 {
                            *entry = (3.05, true, true);
                        }
                    } else if let Some(tids) = self.en_exact.get(slice) {
                        let en_pen = if is_known_tech {
                            0.35
                        } else if slice.len() <= 2 && lookup_syllable(slice).is_some() {
                            -2.80
                        } else if slice.len() <= 3 && can_segment_exact(slice) {
                            -1.80
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
}
