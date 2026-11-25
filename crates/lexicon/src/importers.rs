//! Pure-Rust third-party dictionary importers: Rime `.dict.yaml`, TSV, and Sogou `.scel`.

use std::collections::HashMap;

use encoding_rs::UTF_16LE;

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
    let mut parts = line.split('\t');
    let Some(word) = parts.next().map(str::trim) else {
        stats.skipped_rows += 1;
        return;
    };
    let Some(reading_or_weight) = parts.next().map(str::trim) else {
        stats.skipped_rows += 1;
        return;
    };

    let weight_raw = parts
        .next()
        .and_then(|w| w.trim().trim_end_matches('%').parse::<f32>().ok())
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

    // 1. Parse global UTF-16LE Pinyin index table starting at 0x1540 (+ 4-byte count header).
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

    // 2. Parse homophone word groups starting at `word_start`.
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

fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    let b0 = bytes.get(offset).copied().unwrap_or(0);
    let b1 = bytes.get(offset + 1).copied().unwrap_or(0);
    u16::from_le_bytes([b0, b1])
}

fn decode_utf16le(slice: &[u8]) -> String {
    let (cow, _, _) = UTF_16LE.decode(slice);
    cow.trim_matches('\0').to_string()
}

fn weight_to_log_prob(weight: f32) -> f32 {
    let clamped = weight.max(1.0);
    // Map [1, 100_000] smoothly into [-6.2, -3.0].
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

        // Write Pinyin table at 0x1540 + 4: idx 0 -> "ni", idx 1 -> "hao"
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

        // Append 1 homophone group at 0x2628: same=1, py_bytes_len=4 ([0, 1]), word="你好",
        // ext_len=10
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
}
