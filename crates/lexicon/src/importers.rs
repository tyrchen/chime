//! Pure-Rust third-party dictionary importers:
//! - Rime `.dict.yaml`
//! - Tab-separated `.tsv`
//! - Sogou `.scel` binary cell dictionary
//! - Baidu `.bdict` binary dictionary
//! - `macOS` System Text Replacements (`~/Library/KeyboardServices/TextReplacements.db`)

use std::{collections::HashMap, path::Path};

use encoding_rs::UTF_16LE;
use rusqlite::{Connection, OpenFlags};

use crate::{Lexicon, LexiconError};

/// Summary statistics returned after importing an external dictionary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImportStats {
    /// Total non-comment entry rows encountered.
    pub total_rows: usize,
    /// Number of valid Mandarin or English entries inserted into the [`Lexicon`].
    pub imported_entries: usize,
    /// Rows skipped due to malformed syntax or non-standard readings.
    pub skipped_rows: usize,
}

/// Parses a Rime `.dict.yaml` text dictionary into `lexicon`.
#[must_use]
pub fn parse_rime_dict_yaml(content: &str, lexicon: &mut Lexicon) -> ImportStats {
    let mut in_header = false;
    let mut stats = ImportStats::default();

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "---" {
            in_header = true;
            continue;
        }
        if in_header {
            if line == "..." {
                in_header = false;
            }
            continue;
        }
        import_tab_row(line, lexicon, &mut stats);
    }
    stats
}

/// Parses a tab-separated `word\tpinyin[\tweight]` file into `lexicon`.
#[must_use]
pub fn parse_tsv_dict(content: &str, lexicon: &mut Lexicon) -> ImportStats {
    let mut stats = ImportStats::default();
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        import_tab_row(line, lexicon, &mut stats);
    }
    stats
}

fn import_tab_row(line: &str, lexicon: &mut Lexicon, stats: &mut ImportStats) {
    stats.total_rows += 1;
    let cols: Vec<&str> = line.split('\t').map(str::trim).collect();
    if let ["EN", surface, lp_str, tech_str] = cols.as_slice()
        && surface.is_ascii()
        && surface.len() >= 2
        && let Ok(lp) = lp_str.parse::<f32>()
    {
        lexicon.upsert_calibrated_english(surface, lp, *tech_str == "1");
        stats.imported_entries += 1;
        return;
    }

    let mut parts = cols.into_iter();
    let Some(word) = parts.next() else {
        stats.skipped_rows += 1;
        return;
    };
    let Some(reading_or_weight) = parts.next() else {
        stats.skipped_rows += 1;
        return;
    };

    let weight_raw = parts
        .next()
        .and_then(|w| w.trim_end_matches('%').parse::<f32>().ok())
        .unwrap_or(100.0);
    let log_prob = weight_to_log_prob(weight_raw);

    if lexicon.insert_pinyin_word(word, reading_or_weight, log_prob, false) {
        stats.imported_entries += 1;
    } else if word.is_ascii() && word.len() >= 2 {
        lexicon.insert_english_term(word, log_prob, false);
        stats.imported_entries += 1;
    } else {
        stats.skipped_rows += 1;
    }
}

/// Parses a binary Sogou `.scel` cell dictionary slice into `lexicon`.
///
/// # Errors
///
/// Returns [`LexiconError::InvalidScel`] if `bytes` does not carry the `.scel` magic signature
/// or is shorter than the `0x2628` word-table offset.
pub fn parse_scel_bytes(bytes: &[u8], lexicon: &mut Lexicon) -> Result<ImportStats, LexiconError> {
    const SCEL_MAGIC: &[u8; 4] = b"\x40\x15\x00\x00";
    const PINYIN_OFFSET: usize = 0x1540;
    const WORD_OFFSET_STD: usize = 0x2628;
    const WORD_OFFSET_ALT: usize = 0x26c4;

    if bytes.len() < WORD_OFFSET_STD {
        return Err(LexiconError::InvalidScel(
            "buffer shorter than 0x2628 header",
        ));
    }
    if &bytes[0..4] != SCEL_MAGIC {
        return Err(LexiconError::InvalidScel(
            "missing 0x40 0x15 0x00 0x00 magic",
        ));
    }

    let word_start = if bytes.get(4) == Some(&0x44) {
        WORD_OFFSET_STD
    } else if bytes.get(4) == Some(&0x45) && bytes.len() >= WORD_OFFSET_ALT {
        WORD_OFFSET_ALT
    } else {
        WORD_OFFSET_STD
    };

    let mut pinyin_map: HashMap<u16, String> = HashMap::new();
    let mut cursor = PINYIN_OFFSET + 4;
    while cursor + 4 <= word_start {
        let idx = read_u16_le(bytes, cursor);
        let byte_len = usize::from(read_u16_le(bytes, cursor + 2));
        cursor += 4;
        if byte_len == 0 || cursor + byte_len > word_start {
            break;
        }
        let syl = decode_utf16le(&bytes[cursor..cursor + byte_len]);
        pinyin_map.insert(idx, syl);
        cursor += byte_len;
    }

    let mut stats = ImportStats::default();
    cursor = word_start;
    while cursor + 4 <= bytes.len() {
        let same_count = usize::from(read_u16_le(bytes, cursor));
        let py_bytes_len = usize::from(read_u16_le(bytes, cursor + 2));
        cursor += 4;

        if same_count == 0 || py_bytes_len == 0 || cursor + py_bytes_len > bytes.len() {
            break;
        }

        let mut syllables = Vec::with_capacity(py_bytes_len / 2);
        let py_end = cursor + py_bytes_len;
        while cursor + 2 <= py_end {
            let py_idx = read_u16_le(bytes, cursor);
            if let Some(s) = pinyin_map.get(&py_idx) {
                syllables.push(s.as_str());
            }
            cursor += 2;
        }
        let joined_pinyin = syllables.join(" ");

        for _ in 0..same_count {
            if cursor + 2 > bytes.len() {
                break;
            }
            let word_byte_len = usize::from(read_u16_le(bytes, cursor));
            cursor += 2;
            if cursor + word_byte_len > bytes.len() {
                break;
            }
            let word = decode_utf16le(&bytes[cursor..cursor + word_byte_len]);
            cursor += word_byte_len;

            if cursor + 2 > bytes.len() {
                break;
            }
            let ext_len = usize::from(read_u16_le(bytes, cursor));
            cursor += 2;
            let freq = if ext_len >= 2 && cursor + 2 <= bytes.len() {
                f32::from(read_u16_le(bytes, cursor))
            } else {
                50.0
            };
            if cursor + ext_len > bytes.len() {
                break;
            }
            cursor += ext_len;

            stats.total_rows += 1;
            if lexicon.insert_pinyin_word(word, &joined_pinyin, weight_to_log_prob(freq), false) {
                stats.imported_entries += 1;
            } else {
                stats.skipped_rows += 1;
            }
        }
    }

    Ok(stats)
}

/// Parses a Baidu Pinyin `.bdict` binary dictionary (`bipt` magic at `0x00`, word table at
/// `0x350`).
///
/// # Errors
///
/// Returns [`LexiconError::InvalidBdict`] if the header signature or offset is invalid.
pub fn parse_bdict_bytes(bytes: &[u8], lexicon: &mut Lexicon) -> Result<ImportStats, LexiconError> {
    const BDICT_HEADER_LEN: usize = 0x350;
    const INITIALS: &[&str] = &[
        "c", "d", "b", "f", "g", "h", "ch", "j", "k", "l", "m", "n", "", "p", "q", "r", "s", "t",
        "sh", "zh", "w", "x", "y", "z",
    ];
    const FINALS: &[&str] = &[
        "uang", "iang", "iong", "ang", "eng", "ian", "iao", "ing", "ong", "uai", "uan", "ai", "an",
        "ao", "ei", "en", "er", "ua", "ie", "in", "iu", "ou", "ia", "ue", "ui", "un", "uo", "a",
        "e", "i", "o", "u", "v",
    ];

    if bytes.len() < BDICT_HEADER_LEN || &bytes[0..4] != b"bipt" {
        return Err(LexiconError::InvalidBdict(
            "missing bipt magic or header < 0x350",
        ));
    }

    let end_offset = (read_u32_le(bytes, 0x70) as usize).min(bytes.len());
    let mut cursor = BDICT_HEADER_LEN;
    let mut stats = ImportStats::default();

    while cursor + 4 <= end_offset {
        let syl_len = usize::from(read_u16_le(bytes, cursor));
        cursor += 2;
        if syl_len == 0 || syl_len > 32 || cursor + syl_len * 2 > end_offset {
            break;
        }
        let mut syllables = Vec::with_capacity(syl_len);
        for _ in 0..syl_len {
            let sm = usize::from(bytes[cursor]);
            let ym = usize::from(bytes[cursor + 1]);
            cursor += 2;
            if let (Some(&init), Some(&fin)) = (INITIALS.get(sm), FINALS.get(ym)) {
                syllables.push(format!("{init}{fin}"));
            }
        }
        let byte_len = syl_len * 2;
        if cursor + byte_len > end_offset {
            break;
        }
        let word = decode_utf16le(&bytes[cursor..cursor + byte_len]);
        cursor += byte_len;

        stats.total_rows += 1;
        if syllables.len() == syl_len
            && lexicon.insert_pinyin_word(word, &syllables.join(" "), -4.1, false)
        {
            stats.imported_entries += 1;
        } else {
            stats.skipped_rows += 1;
        }
    }

    Ok(stats)
}

/// Imports user shortcuts from `macOS` `TextReplacements.db` (`ZTEXTREPLACEMENTENTRY` `CoreData`
/// table).
///
/// # Errors
///
/// Returns [`LexiconError::Sqlite`] if the `SQLite` database cannot be opened or queried.
pub fn import_macos_text_replacements(
    db_path: &Path,
    lexicon: &mut Lexicon,
) -> Result<ImportStats, LexiconError> {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut stmt = conn.prepare(
        "SELECT ZSHORTCUT, ZPHRASE FROM ZTEXTREPLACEMENTENTRY WHERE ZWASDELETED = 0 AND ZSHORTCUT \
         IS NOT NULL AND ZPHRASE IS NOT NULL",
    )?;
    let mut rows = stmt.query([])?;
    let mut stats = ImportStats::default();

    while let Some(row) = rows.next()? {
        let shortcut: String = row.get(0)?;
        let phrase: String = row.get(1)?;
        stats.total_rows += 1;
        let clean_key = shortcut.trim();
        let clean_phrase = phrase.trim();
        if clean_key.len() < 2 || clean_phrase.is_empty() {
            stats.skipped_rows += 1;
            continue;
        }
        if lexicon.insert_pinyin_word(clean_phrase, clean_key, -3.0, false) {
            stats.imported_entries += 1;
        } else if clean_key.bytes().all(|b| b.is_ascii_alphabetic()) {
            // Register arbitrary ASCII shortcut into skeleton index so typing `omw` -> `On my way!`
            // works!
            lexicon.insert_custom_shortcut(clean_key, clean_phrase, -2.9);
            stats.imported_entries += 1;
        } else {
            stats.skipped_rows += 1;
        }
    }
    Ok(stats)
}

fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    let b0 = bytes.get(offset).copied().unwrap_or(0);
    let b1 = bytes.get(offset + 1).copied().unwrap_or(0);
    u16::from_le_bytes([b0, b1])
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    let b0 = bytes.get(offset).copied().unwrap_or(0);
    let b1 = bytes.get(offset + 1).copied().unwrap_or(0);
    let b2 = bytes.get(offset + 2).copied().unwrap_or(0);
    let b3 = bytes.get(offset + 3).copied().unwrap_or(0);
    u32::from_le_bytes([b0, b1, b2, b3])
}

fn decode_utf16le(slice: &[u8]) -> String {
    let (cow, _, _) = UTF_16LE.decode(slice);
    cow.trim_matches('\0').to_string()
}

fn weight_to_log_prob(weight: f32) -> f32 {
    let clamped = weight.max(1.0);
    (-6.2 + 0.28 * clamped.ln()).clamp(-6.5, -2.5)
}

#[cfg(test)]
mod tests {
    use chime_phonetics::lookup_syllable;

    use super::*;

    #[test]
    fn test_should_import_rime_dict_yaml_entries() {
        let yaml = "\
# Rime dictionary sample
---
name: sample
version: \"2026.11\"
sort: by_weight
...
雾凇拼音\twu song pin yin\t1200
量子计算\tliang zi ji suan\t850
";
        let mut lex = Lexicon::new();
        let stats = parse_rime_dict_yaml(yaml, &mut lex);
        assert_eq!(stats.imported_entries, 2);

        let ids = [
            lookup_syllable("wu").unwrap_or(chime_phonetics::SyllableId(0)),
            lookup_syllable("song").unwrap_or(chime_phonetics::SyllableId(0)),
            lookup_syllable("pin").unwrap_or(chime_phonetics::SyllableId(0)),
            lookup_syllable("yin").unwrap_or(chime_phonetics::SyllableId(0)),
        ];
        assert_eq!(lex.lookup_exact(&ids)[0].surface, "雾凇拼音");
    }

    #[test]
    fn test_should_parse_synthetic_scel_binary_payload() {
        let mut buf = vec![0u8; 0x2628];
        buf[0..4].copy_from_slice(b"\x40\x15\x00\x00");
        buf[4] = 0x44;

        let mut cur = 0x1544;
        for (idx, py) in [(0u16, "ni"), (1u16, "hao")] {
            let utf16: Vec<u8> = py.encode_utf16().flat_map(u16::to_le_bytes).collect();
            buf[cur..cur + 2].copy_from_slice(&idx.to_le_bytes());
            buf[cur + 2..cur + 4]
                .copy_from_slice(&u16::try_from(utf16.len()).unwrap_or(0).to_le_bytes());
            cur += 4;
            buf[cur..cur + utf16.len()].copy_from_slice(&utf16);
            cur += utf16.len();
        }

        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&4u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());

        let word_utf16: Vec<u8> = "你好".encode_utf16().flat_map(u16::to_le_bytes).collect();
        buf.extend_from_slice(&u16::try_from(word_utf16.len()).unwrap_or(0).to_le_bytes());
        buf.extend_from_slice(&word_utf16);
        buf.extend_from_slice(&10u16.to_le_bytes());
        buf.extend_from_slice(&500u16.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);

        let mut lex = Lexicon::new();
        let stats = parse_scel_bytes(&buf, &mut lex).unwrap_or_default();
        assert_eq!(stats.imported_entries, 1);
    }

    #[test]
    fn test_should_import_synthetic_macos_text_replacements_sqlite() {
        let dir = tempfile::tempdir().unwrap_or_else(|_| std::process::abort());
        let db_path = dir.path().join("TextReplacements.db");
        let conn = Connection::open(&db_path).unwrap_or_else(|_| std::process::abort());
        conn.execute_batch(
            "CREATE TABLE ZTEXTREPLACEMENTENTRY (
                Z_PK INTEGER PRIMARY KEY,
                ZWASDELETED INTEGER,
                ZSHORTCUT VARCHAR,
                ZPHRASE VARCHAR
            );
            INSERT INTO ZTEXTREPLACEMENTENTRY VALUES (1, 0, 'dz', '北京市海淀区');",
        )
        .unwrap_or_else(|_| std::process::abort());
        drop(conn);

        let mut lex = Lexicon::new();
        let stats = import_macos_text_replacements(&db_path, &mut lex).unwrap_or_default();
        assert_eq!(stats.imported_entries, 1);
        assert_eq!(lex.lookup_skeleton("dz")[0].surface, "北京市海淀区");
    }
}
