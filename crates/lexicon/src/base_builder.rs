//! Permissive large-scale base lexicon generator (`MIT` `pinyin` + YAML heteronym overrides
//! + `MIT` `jieba-rs` corpus frequencies + system English vocabulary).

use std::{collections::HashMap, path::Path};

use pinyin::{ToPinyin, ToPinyinMulti};

use crate::{
    Lexicon,
    generated::{MODERN_DAILY_HETERONYM_CHARS, POLYPHONE_WORD_OVERRIDES},
};

/// Annotates a pure-CJK word into space-separated toneless Pinyin using overrides + `pinyin` crate.
#[must_use]
pub fn annotate_chinese_word<S: std::hash::BuildHasher>(
    word: &str,
    overrides: &HashMap<&str, &str, S>,
) -> Option<String> {
    if let Some(&ov) = overrides.get(word) {
        return Some(ov.to_string());
    }
    let mut syllables = Vec::with_capacity(word.chars().count());
    for ch in word.chars() {
        let py = ch.to_pinyin()?;
        syllables.push(py.plain());
    }
    if syllables.is_empty() {
        None
    } else {
        Some(syllables.join(" "))
    }
}

/// Populates all common Unicode CJK characters (including heteronym readings) so no character is
/// ever unreachable.
pub fn populate_standard_cjk_characters(lex: &mut Lexicon) -> usize {
    let mut added = 0usize;
    for code in 0x4E00u32..=0x9FA5u32 {
        let Some(ch) = char::from_u32(code) else {
            continue;
        };
        let Some(multi) = ch.to_pinyin_multi() else {
            continue;
        };
        let surface = ch.to_string();
        for (idx, py) in multi.into_iter().take(3).enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let base_lp = -9.4 - (idx as f32) * 0.9;
            if lex.insert_pinyin_word(&surface, py.plain(), base_lp, false) {
                added += 1;
            }
        }
    }
    for &(ch_surf, py, lp) in MODERN_DAILY_HETERONYM_CHARS {
        let _ = lex.insert_pinyin_word(ch_surf, py, lp, false);
    }
    added
}

/// Builds a comprehensive, 100% permissive base [`Lexicon`] (~150,000+ entries) from:
/// 1. Declarative closed-class grammar & polyphone YAML tables (`crates/lexicon/data/*.yaml`)
/// 2. All 20,902 CJK characters with heteronym readings (`pinyin`, MIT)
/// 3. Modern Chinese multi-character vocabulary & corpus frequencies (`jieba_dict_text`, MIT)
/// 4. Optional English wordlist (`/usr/share/dict/words`)
#[must_use]
pub fn build_comprehensive_lexicon(
    jieba_dict_text: &str,
    english_words_path: Option<&Path>,
    min_word_freq: u64,
) -> Lexicon {
    let mut lex = Lexicon::with_core_grammar();
    let overrides: HashMap<&str, &str> = POLYPHONE_WORD_OVERRIDES.iter().copied().collect();

    // 1. Insert curated polyphone words from `data/polyphones.yaml`.
    for &(word, py) in POLYPHONE_WORD_OVERRIDES {
        lex.insert_pinyin_word(word, py, -3.75, false);
    }

    // 2. Insert all 20,902 BMP CJK characters + modern heteronyms.
    let _ = populate_standard_cjk_characters(&mut lex);

    // 3. Parse Jieba corpus dictionary (`word freq [tag]`) and compute smoothed empirical log-prob.
    let mut rows: Vec<(&str, u64)> = Vec::new();
    let mut total_freq: f64 = 0.0;
    for raw_line in jieba_dict_text.lines() {
        let mut parts = raw_line.split_whitespace();
        let Some(word) = parts.next() else { continue };
        let Some(freq_str) = parts.next() else {
            continue;
        };
        let Ok(freq) = freq_str.parse::<u64>() else {
            continue;
        };
        let char_len = word.chars().count();
        if !(1..=12).contains(&char_len) {
            continue;
        }
        if char_len >= 2 && freq < min_word_freq {
            continue;
        }
        if !word.chars().all(|c| ('\u{4E00}'..='\u{9FA5}').contains(&c)) {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            total_freq += freq as f64;
        }
        rows.push((word, freq));
    }

    let denom = total_freq.max(1.0).ln();
    let ln10 = 10.0_f32.ln();
    for (word, freq) in rows {
        let Some(py) = annotate_chinese_word(word, &overrides) else {
            continue;
        };
        // Convert `jieba` empirical probability into the exact same base-10 Zipf scale (`log10 P +
        // 9.0`) used by `calibrate_modern_lexicon.py`, with a `-0.50` Zipf discount on 1990s
        // newspaper frequencies so calibrated modern spoken/web + Qwen3 priors in `multi_lexicon`
        // govern whenever present, while rare `jieba` entries remain available on a consistent
        // scale!
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let raw_ln = ((freq.max(1) as f64).ln() - denom) as f32;
        let zipf_jb = (raw_ln / ln10) + 9.0 - 0.50;
        let scaled_lp = if word.chars().count() == 1 {
            (-6.30 + (zipf_jb - 2.25) * 0.82).clamp(-6.50, -2.55)
        } else {
            (-5.68 + (zipf_jb - 2.65) * 0.85).clamp(-6.20, -2.65)
        };
        lex.insert_pinyin_word(word, &py, scaled_lp, false);
    }

    // 4. Optional English dictionary words (common lowercase words length 5..=14) as rare-word
    // validity floor so uncommon valid words are not falsely auto-corrected in `En` mode.
    if let Some(eng_path) = english_words_path
        && let Ok(eng_text) = std::fs::read_to_string(eng_path)
    {
        for line in eng_text.lines() {
            let w = line.trim();
            if (5..=14).contains(&w.len())
                && w.bytes().all(|b| b.is_ascii_lowercase())
                && lex.lookup_english(&w.to_ascii_lowercase()).is_none()
            {
                lex.insert_english_term(w, -6.85, false);
            }
        }
    }

    lex
}
