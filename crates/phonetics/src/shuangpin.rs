//! Shuangpin (Double Pinyin) 2-key-per-syllable schemes: Full, Xiaohe, Ziranma, Microsoft.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::syllables::{SyllableId, lookup_syllable};

/// Supported Pinyin keyboard schemes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShuangpinScheme {
    /// Standard full Pinyin (`shuang` = 6 keys).
    #[default]
    Full,
    /// Xiaohe Shuangpin (小鹤双拼: `ul` -> `shuang`).
    Xiaohe,
    /// Ziranma Shuangpin (自然码: `ud` -> `shuang`).
    Ziranma,
    /// Microsoft Shuangpin (微软双拼: `ud` -> `shuang`).
    Microsoft,
}

/// Expands a 2-letter ASCII key pair under the given Shuangpin scheme into valid [`SyllableId`]s.
#[must_use]
pub fn expand_shuangpin_pair(scheme: ShuangpinScheme, c1: u8, c2: u8) -> SmallVec<[SyllableId; 2]> {
    let mut out = SmallVec::new();
    if scheme == ShuangpinScheme::Full
        || !c1.is_ascii_lowercase()
        || !(c2.is_ascii_lowercase() || (scheme == ShuangpinScheme::Microsoft && c2 == b';'))
    {
        return out;
    }

    // Zero-initial two-key vowel rules shared across Xiaohe & Ziranma (e.g. `aa`->`a`,
    // `ah`->`ang`).
    if let Some(zero_syl) = zero_initial_pair(scheme, c1, c2)
        && let Some(id) = lookup_syllable(zero_syl)
    {
        out.push(id);
        return out;
    }

    let initial = map_initial(scheme, c1);
    if initial.is_empty() {
        return out;
    }

    for fin in map_finals(scheme, c2) {
        let composed = compose_initial_final(initial, fin);
        if let Some(id) = lookup_syllable(&composed)
            && !out.contains(&id)
        {
            out.push(id);
        }
    }
    out
}

/// Positions where a one-key syllable participates in a parse with the fewest omitted finals.
/// Keep *all* tied segmentations for language-model scoring. A valid sequence of complete
/// pairs therefore never sprouts abbreviation candidates. Apostrophes are hard boundaries.
#[must_use]
pub fn shuangpin_initial_mask(scheme: ShuangpinScheme, keys: &[u8]) -> Vec<bool> {
    let n = keys.len();
    let mut singles = vec![false; n];
    if scheme == ShuangpinScheme::Full || n == 0 {
        return singles;
    }
    let pairs: Vec<bool> = keys
        .windows(2)
        .map(|p| !expand_shuangpin_pair(scheme, p[0], p[1]).is_empty())
        .collect();
    let initial = |i: usize| {
        !map_initial(scheme, keys[i]).is_empty()
            || (i + 1 == n && !expand_shuangpin_initial(scheme, keys[i]).is_empty())
    };
    // Minimum number of missing finals required to finish each suffix.
    let mut cost = vec![usize::MAX; n + 1];
    cost[n] = 0;
    for i in (0..n).rev() {
        if keys[i] == b'\'' {
            cost[i] = cost[i + 1];
            continue;
        }
        if initial(i) {
            cost[i] = cost[i + 1].saturating_add(1);
        }
        if pairs.get(i) == Some(&true) {
            cost[i] = cost[i].min(cost[i + 2]);
        }
    }
    if cost[0] == usize::MAX {
        // The lattice may start on the English track. Preserve trailing-initial completion.
        singles[n - 1] = initial(n - 1);
        return singles;
    }
    let mut reachable = vec![false; n + 1];
    reachable[0] = true;
    for i in 0..n {
        if !reachable[i] {
            continue;
        }
        if keys[i] == b'\'' {
            reachable[i + 1] = true;
            continue;
        }
        if pairs.get(i) == Some(&true) && cost[i] == cost[i + 2] {
            reachable[i + 2] = true;
        }
        if initial(i) && cost[i] == cost[i + 1].saturating_add(1) {
            singles[i] = true;
            reachable[i + 1] = true;
        }
    }
    singles
}

/// Prefer a still-unfinished trailing pair to reinterpreting an earlier complete pair.
#[must_use]
pub const fn initial_log_cost(end: usize, input_len: usize) -> f32 {
    if end == input_len { -1.35 } else { -2.7 }
}

/// Possible syllables when the second key of a pair is omitted. Use the
/// layout itself so retroflex initials, zero initials and Microsoft's `;` agree with exact input.
#[must_use]
pub fn expand_shuangpin_initial(scheme: ShuangpinScheme, first: u8) -> SmallVec<[SyllableId; 32]> {
    let mut ids = SmallVec::new();
    for &second in b"abcdefghijklmnopqrstuvwxyz;" {
        for id in expand_shuangpin_pair(scheme, first, second) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

fn zero_initial_pair(scheme: ShuangpinScheme, c1: u8, c2: u8) -> Option<&'static str> {
    match scheme {
        ShuangpinScheme::Xiaohe | ShuangpinScheme::Ziranma => match (c1, c2) {
            (b'a', b'a') => Some("a"),
            (b'a', b'i') => Some("ai"),
            (b'a', b'n') => Some("an"),
            (b'a', b'h') => Some("ang"),
            (b'a', b'o') => Some("ao"),
            (b'e', b'e') => Some("e"),
            (b'e', b'i') => Some("ei"),
            (b'e', b'n') => Some("en"),
            (b'e', b'g') => Some("eng"),
            (b'e', b'r') => Some("er"),
            (b'o', b'o') => Some("o"),
            (b'o', b'u') => Some("ou"),
            _ => None,
        },
        ShuangpinScheme::Microsoft => match (c1, c2) {
            (b'o', b'a') => Some("a"),
            (b'o', b'l') => Some("ai"),
            (b'o', b'j') => Some("an"),
            (b'o', b'h') => Some("ang"),
            (b'o', b'k') => Some("ao"),
            (b'o', b'e') => Some("e"),
            (b'o', b'z') => Some("ei"),
            (b'o', b'f') => Some("en"),
            (b'o', b'g') => Some("eng"),
            (b'o', b'r') => Some("er"),
            (b'o', b'o') => Some("o"),
            (b'o', b'b') => Some("ou"),
            _ => None,
        },
        ShuangpinScheme::Full => None,
    }
}

fn map_initial(scheme: ShuangpinScheme, key: u8) -> &'static str {
    match (scheme, key) {
        (_, b'u') => "sh",
        (_, b'i') => "ch",
        (_, b'v') => "zh",
        (_, b'b') => "b",
        (_, b'p') => "p",
        (_, b'm') => "m",
        (_, b'f') => "f",
        (_, b'd') => "d",
        (_, b't') => "t",
        (_, b'n') => "n",
        (_, b'l') => "l",
        (_, b'g') => "g",
        (_, b'k') => "k",
        (_, b'h') => "h",
        (_, b'j') => "j",
        (_, b'q') => "q",
        (_, b'x') => "x",
        (_, b'r') => "r",
        (_, b'z') => "z",
        (_, b'c') => "c",
        (_, b's') => "s",
        (_, b'y') => "y",
        (_, b'w') => "w",
        _ => "",
    }
}

fn map_finals(scheme: ShuangpinScheme, key: u8) -> &'static [&'static str] {
    match scheme {
        ShuangpinScheme::Xiaohe => match key {
            b'a' => &["a"],
            b'o' => &["o", "uo"],
            b'e' => &["e"],
            b'i' => &["i"],
            b'u' => &["u"],
            b'v' => &["v", "ui"],
            b'b' => &["in"],
            b'c' => &["ao"],
            b'd' => &["ai"],
            b'f' => &["en"],
            b'g' => &["eng"],
            b'h' => &["ang"],
            b'j' => &["an"],
            b'k' => &["ing", "uai"],
            b'l' => &["iang", "uang"],
            b'm' => &["ian"],
            b'n' => &["iao"],
            b'p' => &["ie"],
            b'q' => &["iu"],
            b'r' => &["uan"],
            b's' => &["ong", "iong"],
            b't' => &["ue", "ve"],
            b'w' => &["ei"],
            b'x' => &["ia", "ua"],
            b'y' => &["un"],
            b'z' => &["ou"],
            _ => &[],
        },
        ShuangpinScheme::Ziranma => match key {
            b'a' => &["a"],
            b'o' => &["o", "uo"],
            b'e' => &["e"],
            b'i' => &["i"],
            b'u' => &["u"],
            b'v' => &["v", "ui"],
            b'b' => &["ou"],
            b'c' => &["iao"],
            b'd' => &["uang", "iang"],
            b'f' => &["en"],
            b'g' => &["eng"],
            b'h' => &["ang"],
            b'j' => &["an"],
            b'k' => &["ao"],
            b'l' => &["ai"],
            b'm' => &["ian"],
            b'n' => &["in"],
            b'p' => &["un"],
            b'q' => &["iu"],
            b'r' => &["uan"],
            b's' => &["ong", "iong"],
            b't' => &["ue", "ve"],
            b'w' => &["ia", "ua"],
            b'x' => &["ie"],
            b'y' => &["ing", "uai"],
            b'z' => &["ei"],
            _ => &[],
        },
        ShuangpinScheme::Microsoft => match key {
            b'a' => &["a"],
            b'o' => &["o", "uo"],
            b'e' => &["e"],
            b'i' => &["i"],
            b'u' => &["u"],
            b'v' => &["ui", "ue"],
            b'y' => &["v", "uai"],
            b'b' => &["ou"],
            b'c' => &["iao"],
            b'd' => &["uang", "iang"],
            b'f' => &["en"],
            b'g' => &["eng"],
            b'h' => &["ang"],
            b'j' => &["an"],
            b'k' => &["ao"],
            b'l' => &["ai"],
            b'm' => &["ian"],
            b'n' => &["in"],
            b'p' => &["un"],
            b'q' => &["iu"],
            b'r' => &["uan"],
            b's' => &["ong", "iong"],
            b't' => &["ve"],
            b'w' => &["ia", "ua"],
            b'x' => &["ie"],
            b';' => &["ing"],
            b'z' => &["ei"],
            _ => &[],
        },
        ShuangpinScheme::Full => &[],
    }
}

fn compose_initial_final(initial: &str, fin: &str) -> String {
    // Handle `j`/`q`/`x`/`y` + `v`/`ve` orthography (`ju`, `jue`, `yu`, `yue`).
    if matches!(initial, "j" | "q" | "x" | "y") {
        match fin {
            "v" => return format!("{initial}u"),
            "ve" => return format!("{initial}ue"),
            _ => {}
        }
    }
    let mut s = String::with_capacity(initial.len() + fin.len());
    s.push_str(initial);
    s.push_str(fin);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syllables::canonical_syllable;

    #[test]
    fn test_should_decode_xiaohe_shuangpin_pairs() {
        // 小鹤双拼: ul -> shuang, vs -> zhong, go -> guo, ju -> ju / jv->ju
        let ul = expand_shuangpin_pair(ShuangpinScheme::Xiaohe, b'u', b'l');
        assert_eq!(ul.len(), 1);
        assert_eq!(canonical_syllable(ul[0]), "shuang");

        let vs = expand_shuangpin_pair(ShuangpinScheme::Xiaohe, b'v', b's');
        assert_eq!(canonical_syllable(vs[0]), "zhong");

        let xt = expand_shuangpin_pair(ShuangpinScheme::Xiaohe, b'x', b't');
        assert_eq!(canonical_syllable(xt[0]), "xue");
    }
}
