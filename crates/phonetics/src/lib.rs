//! Mandarin syllable algebra, zero-bloat fuzzy expansion, Shuangpin mapping,
//! QWERTY keyboard typo modeling, and Chinese-English code-switch phonotactics.

#![forbid(unsafe_code)]

pub mod shuangpin;
pub mod syllables;
pub mod typo;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

pub use crate::{
    shuangpin::{ShuangpinScheme, expand_shuangpin_pair},
    syllables::{SyllableId, canonical_syllable, is_valid_syllable, lookup_syllable},
    typo::{TypoEdit, generate_typo_variants, keyboard_substitution_cost},
};

/// Configurable fuzzy Pinyin equivalence flags.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct FuzzyConfig {
    /// Equate `z` and `zh`.
    pub z_zh: bool,
    /// Equate `c` and `ch`.
    pub c_ch: bool,
    /// Equate `s` and `sh`.
    pub s_sh: bool,
    /// Equate initial `n` and `l`.
    pub n_l: bool,
    /// Equate initial `f` and `h`.
    pub f_h: bool,
    /// Equate initial `r` and `l`.
    pub r_l: bool,
    /// Equate final `an` and `ang`.
    pub an_ang: bool,
    /// Equate final `en` and `eng`.
    pub en_eng: bool,
    /// Equate final `in` and `ing`.
    pub in_ing: bool,
    /// Equate final `ian` and `iang`.
    pub ian_iang: bool,
    /// Equate final `uan` and `uang`.
    pub uan_uang: bool,
    /// Penalty added per fuzzy substitution (lower cost = higher rank).
    pub fuzzy_penalty: f32,
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        Self {
            z_zh: true,
            c_ch: true,
            s_sh: true,
            n_l: false,
            f_h: false,
            r_l: false,
            an_ang: true,
            en_eng: true,
            in_ing: true,
            ian_iang: true,
            uan_uang: true,
            fuzzy_penalty: 0.52,
        }
    }
}

impl FuzzyConfig {
    /// Expands a canonical Pinyin syllable into enabled fuzzy equivalents and their penalties.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn expand(&self, syllable: &str) -> SmallVec<[(SyllableId, f32); 4]> {
        let mut out = SmallVec::new();
        let mut initial_variants: SmallVec<[(&str, &str, u32); 3]> = SmallVec::new();

        if let Some(rest) = syllable.strip_prefix("zh") {
            initial_variants.push(("zh", rest, 0));
            if self.z_zh {
                initial_variants.push(("z", rest, 1));
            }
        } else if let Some(rest) = syllable.strip_prefix("ch") {
            initial_variants.push(("ch", rest, 0));
            if self.c_ch {
                initial_variants.push(("c", rest, 1));
            }
        } else if let Some(rest) = syllable.strip_prefix("sh") {
            initial_variants.push(("sh", rest, 0));
            if self.s_sh {
                initial_variants.push(("s", rest, 1));
            }
        } else if let Some(rest) = syllable.strip_prefix('z') {
            initial_variants.push(("z", rest, 0));
            if self.z_zh {
                initial_variants.push(("zh", rest, 1));
            }
        } else if let Some(rest) = syllable.strip_prefix('c') {
            initial_variants.push(("c", rest, 0));
            if self.c_ch {
                initial_variants.push(("ch", rest, 1));
            }
        } else if let Some(rest) = syllable.strip_prefix('s') {
            initial_variants.push(("s", rest, 0));
            if self.s_sh {
                initial_variants.push(("sh", rest, 1));
            }
        } else if let Some(rest) = syllable.strip_prefix('n') {
            initial_variants.push(("n", rest, 0));
            if self.n_l {
                initial_variants.push(("l", rest, 1));
            }
        } else if let Some(rest) = syllable.strip_prefix('l') {
            initial_variants.push(("l", rest, 0));
            if self.n_l {
                initial_variants.push(("n", rest, 1));
            }
            if self.r_l {
                initial_variants.push(("r", rest, 1));
            }
        } else if let Some(rest) = syllable.strip_prefix('r') {
            initial_variants.push(("r", rest, 0));
            if self.r_l {
                initial_variants.push(("l", rest, 1));
            }
        } else if let Some(rest) = syllable.strip_prefix('f') {
            initial_variants.push(("f", rest, 0));
            if self.f_h {
                initial_variants.push(("h", rest, 1));
            }
        } else if let Some(rest) = syllable.strip_prefix('h') {
            initial_variants.push(("h", rest, 0));
            if self.f_h {
                initial_variants.push(("f", rest, 1));
            }
        } else {
            // Strip all remaining standard Mandarin initials (`b, p, m, d, t, g, k, j, q, x, y, w`)
            // so final-rhyme fuzzy pairs (`yin<->ying`, `jin<->jing`, `xin<->xing`, `ming<->min`,
            // `wang<->wan`, `peng<->pen`, etc.) expand across every syllable!
            let mut matched_other = false;
            for init in ["b", "p", "m", "d", "t", "g", "k", "j", "q", "x", "y", "w"] {
                if let Some(rest) = syllable.strip_prefix(init) {
                    initial_variants.push((init, rest, 0));
                    matched_other = true;
                    break;
                }
            }
            if !matched_other {
                initial_variants.push(("", syllable, 0));
            }
        }

        for (init, rest, init_edits) in initial_variants {
            let final_variants = self.expand_final(rest);
            for (fin, fin_edits) in final_variants {
                let total_edits = init_edits + fin_edits;
                if total_edits == 0 {
                    continue;
                }
                let mut candidate = String::with_capacity(init.len() + fin.len());
                candidate.push_str(init);
                candidate.push_str(fin);
                if let Some(id) = lookup_syllable(&candidate)
                    && !out.iter().any(|(existing, _)| *existing == id)
                {
                    #[allow(clippy::cast_precision_loss)]
                    let penalty = (total_edits as f32) * self.fuzzy_penalty;
                    out.push((id, penalty));
                }
            }
        }
        out
    }

    fn expand_final<'a>(&self, rest: &'a str) -> SmallVec<[(&'a str, u32); 2]> {
        let mut out = SmallVec::new();
        out.push((rest, 0));

        match rest {
            "iang" if self.ian_iang => out.push(("ian", 1)),
            "ian" if self.ian_iang => out.push(("iang", 1)),
            "uang" if self.uan_uang => out.push(("uan", 1)),
            "uan" if self.uan_uang => out.push(("uang", 1)),
            "ang" if self.an_ang => out.push(("an", 1)),
            "an" if self.an_ang => out.push(("ang", 1)),
            "eng" if self.en_eng => out.push(("en", 1)),
            "en" if self.en_eng => out.push(("eng", 1)),
            "ing" if self.in_ing => out.push(("in", 1)),
            "in" if self.in_ing => out.push(("ing", 1)),
            "uang" if self.an_ang => out.push(("uan", 1)),
            "uan" if self.an_ang => out.push(("uang", 1)),
            "iang" if self.an_ang => out.push(("ian", 1)),
            "ian" if self.an_ang => out.push(("iang", 1)),
            _ => {}
        }
        out
    }
}

/// Classification of a raw token or span for Chinese-English code-switching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeSwitchSignal {
    /// Contains uppercase mid-word, digits, underscores, slashes, or dots (`useEffect`, `v0.1`).
    HardRawCode,
    /// Contains English consonant clusters impossible in Pinyin (`script`, `chrome`, `pdf`).
    PhonotacticEnglish,
    /// Ambiguous lowercase ASCII that can be Pinyin or English (`git`, `api`, `macos`, `nihao`).
    AmbiguousAscii,
}

/// Inspects a raw ASCII slice and classifies its code-switching signal.
///
/// # Examples
///
/// ```
/// use chime_phonetics::{CodeSwitchSignal, classify_code_switch};
///
/// assert_eq!(classify_code_switch("useEffect"), CodeSwitchSignal::HardRawCode);
/// assert_eq!(classify_code_switch("chrome"), CodeSwitchSignal::PhonotacticEnglish);
/// assert_eq!(classify_code_switch("nihao"), CodeSwitchSignal::AmbiguousAscii);
/// ```
#[must_use]
pub fn classify_code_switch(span: &str) -> CodeSwitchSignal {
    if span.is_empty() {
        return CodeSwitchSignal::AmbiguousAscii;
    }

    let has_non_alpha = span.bytes().any(|b| !b.is_ascii_alphabetic() && b != b'\'');
    let has_internal_upper = span.bytes().skip(1).any(|b| b.is_ascii_uppercase());
    let all_upper_acronym = span.len() >= 2 && span.bytes().all(|b| b.is_ascii_uppercase());
    if has_non_alpha || has_internal_upper || all_upper_acronym {
        return CodeSwitchSignal::HardRawCode;
    }

    let lower = span.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let has_english_v = bytes
        .iter()
        .enumerate()
        .any(|(i, &b)| b == b'v' && (i == 0 || !matches!(bytes[i - 1], b'l' | b'n')));
    let impossible_clusters = [
        "str", "scr", "spr", "spl", "chr", "thr", "th", "ph", "ck", "gh", "wh", "wr", "kn", "bl",
        "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sc", "sk", "sl", "sm", "sp",
        "st", "sw", "tr", "tw", "ct", "ft", "lt", "mp", "nd", "nk", "nt", "pt", "rd", "rk", "rl",
        "rm", "rn", "rp", "rt", "rv", "pdf", "sql", "http", "html",
    ];
    if (has_english_v || impossible_clusters.iter().any(|pat| lower.contains(pat)))
        && !can_segment_exact(&lower)
    {
        return CodeSwitchSignal::PhonotacticEnglish;
    }

    CodeSwitchSignal::AmbiguousAscii
}

/// Checks whether `input` can be completely segmented into exact canonical Pinyin syllables.
#[must_use]
pub fn can_segment_exact(input: &str) -> bool {
    if input.is_empty() {
        return true;
    }
    let bytes = input.as_bytes();
    let n = bytes.len();
    let mut reachable = vec![false; n + 1];
    reachable[0] = true;

    for i in 0..n {
        if !reachable[i] {
            continue;
        }
        if bytes[i] == b'\'' {
            reachable[i + 1] = true;
            continue;
        }
        let max_len = (n - i).min(6);
        for len in 1..=max_len {
            let sub = &input[i..i + len];
            if is_valid_syllable(sub) {
                reachable[i + len] = true;
            }
        }
    }
    reachable[n]
}

/// Extracts the initial key (1-char or `zh`/`ch`/`sh`) of a canonical syllable.
#[must_use]
pub fn extract_initial_head(syllable: &str) -> &str {
    for retro in ["zh", "ch", "sh"] {
        if syllable.starts_with(retro) {
            return retro;
        }
    }
    if syllable.is_empty() {
        ""
    } else {
        &syllable[..1]
    }
}

/// Formats a raw preedit buffer into human-friendly apostrophe-separated syllables
/// (`jin'tian'wo'men`).
#[must_use]
pub fn format_segmented_preedit(raw: &str, scheme: ShuangpinScheme) -> String {
    if raw.is_empty() || classify_code_switch(raw) != CodeSwitchSignal::AmbiguousAscii {
        return raw.to_string();
    }
    let lower = raw.to_ascii_lowercase();
    if scheme != ShuangpinScheme::Full {
        let mut parts = Vec::new();
        let mut i = 0usize;
        while i < raw.len() {
            let end = (i + 2).min(raw.len());
            parts.push(&raw[i..end]);
            i = end;
        }
        return parts.join("'");
    }

    let n = lower.len();
    let mut parts = Vec::new();
    let mut pos = 0usize;
    while pos < n {
        if lower.as_bytes()[pos] == b'\'' {
            pos += 1;
            continue;
        }
        let max_len = (n - pos).min(6);
        let mut chosen_len = 0usize;
        for len in (1..=max_len).rev() {
            let sub = &lower[pos..pos + len];
            if !sub.contains('\'') && is_valid_syllable(sub) {
                // Avoid greedy over-bite (`fangan` -> prefer `fang'an` over `fan'gan` only when
                // remainder valid; if `pos + len < n` and remainder starts with vowel like `n`/`g`
                // where `len-1` leaves valid vowel, check!).
                if pos + len < n
                    && len >= 3
                    && can_segment_exact(&lower[pos + len - 1..])
                    && !can_segment_exact(&lower[pos + len..])
                {
                    continue;
                }
                chosen_len = len;
                break;
            }
        }
        if chosen_len == 0 {
            for len in (2..=max_len).rev() {
                let sub = &lower[pos..pos + len];
                if !sub.contains('\'') && !generate_typo_variants(sub).is_empty() {
                    chosen_len = len;
                    break;
                }
            }
        }
        if chosen_len == 0 {
            chosen_len = 1;
        }
        parts.push(&raw[pos..pos + chosen_len]);
        pos += chosen_len;
    }
    parts.join("'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_expand_fuzzy_pinyin_without_invalid_syllables() {
        let cfg = FuzzyConfig {
            uan_uang: true,
            ..Default::default()
        };
        let expanded = cfg.expand("shuang");
        let names: Vec<&str> = expanded
            .iter()
            .map(|(id, _)| canonical_syllable(*id))
            .collect();
        // "suang" is not a valid Mandarin syllable and must be filtered out;
        // "shuan" and "suan" are valid.
        assert!(names.contains(&"shuan"));
        assert!(names.contains(&"suan"));
        assert!(!names.contains(&"suang"));
    }

    #[test]
    fn test_should_classify_code_switching_signals() {
        assert_eq!(
            classify_code_switch("useEffect"),
            CodeSwitchSignal::HardRawCode
        );
        assert_eq!(classify_code_switch("PR"), CodeSwitchSignal::HardRawCode);
        assert_eq!(
            classify_code_switch("v0.2.1"),
            CodeSwitchSignal::HardRawCode
        );
        assert_eq!(
            classify_code_switch("typescript"),
            CodeSwitchSignal::PhonotacticEnglish
        );
        assert_eq!(
            classify_code_switch("zhongguo"),
            CodeSwitchSignal::AmbiguousAscii
        );
    }

    #[test]
    fn test_should_verify_exact_pinyin_segmentation() {
        assert!(can_segment_exact("zhongguoren"));
        assert!(can_segment_exact("xi'an"));
        assert!(!can_segment_exact("shaung"));
        assert!(!can_segment_exact("chrome"));
    }
}
