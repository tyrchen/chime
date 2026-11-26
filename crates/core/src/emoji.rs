//! Zero-latency offline Unicode Emoji & Kaomoji (颜文字) candidate suggestions.
//!
//! Design invariants:
//! 1. **Never steals `#1`**: normal Chinese / English candidates always retain the top slot so
//!    `Space` never accidentally commits an emoji.
//! 2. **Scheme-agnostic**: triggers off decoded Chinese surface text, so Full Pinyin, all three
//!    Shuangpin schemes (Xiaohe / Ziranma / Microsoft), fuzzy Pinyin, abbreviations, and typo
//!    corrections work automatically.
//! 3. **IDE-aware**: automatically quiet inside developer apps (terminals / IDEs) unless the user
//!    explicitly types a browse keyword (`emoji`, `biaoqing`, `yanwenzi`, `kaomoji`).
//! 4. **Context-clean**: callers check `MatchKind::Emoji` to avoid polluting neural LM context,
//!    negative-feedback penalties, or `SQLite` word habits.

use std::{collections::HashMap, sync::OnceLock};

use chime_decoder::{Candidate, MatchKind};

const EMOJI_KAOMOJI_TSV: &str = include_str!("../data/emoji_kaomoji.tsv");

/// Controls how Emoji & Kaomoji candidates are injected for the current session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct EmojiSuggestionOptions {
    /// Enable Unicode Emoji candidate suggestions.
    pub emoji_enabled: bool,
    /// Enable Kaomoji (颜文字) candidate suggestions.
    pub kaomoji_enabled: bool,
    /// Suppress automatic word-triggered suggestions inside terminals & IDEs.
    pub suppress_in_ide: bool,
    /// True when the active host app is a terminal or IDE (`UserStore::is_developer_app`).
    pub is_developer_app: bool,
}

#[derive(Debug, Default)]
struct EmojiIndex {
    word_emojis: HashMap<&'static str, Vec<&'static str>>,
    word_kaomojis: HashMap<&'static str, Vec<&'static str>>,
    ascii_emojis: HashMap<&'static str, Vec<&'static str>>,
    ascii_kaomojis: HashMap<&'static str, Vec<&'static str>>,
}

fn push_unique<'a>(vec: &mut Vec<&'a str>, item: &'a str) {
    if !item.is_empty() && !vec.contains(&item) {
        vec.push(item);
    }
}

fn is_explicit_browse_keyword(lower_raw: &str) -> bool {
    matches!(
        lower_raw,
        "emoji" | "biaoqing" | "bq" | "yanwenzi" | "kaomoji"
    )
}

fn global_index() -> &'static EmojiIndex {
    static INDEX: OnceLock<EmojiIndex> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut idx = EmojiIndex::default();
        for raw_line in EMOJI_KAOMOJI_TSV.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split('\t');
            let (Some(kind), Some(triggers_col), Some(items_col)) =
                (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            let target_map = match kind.trim() {
                "E" => &mut idx.word_emojis,
                "K" => &mut idx.word_kaomojis,
                "AE" => &mut idx.ascii_emojis,
                "AK" => &mut idx.ascii_kaomojis,
                _ => continue,
            };
            for trig in triggers_col
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                let entry = target_map.entry(trig).or_default();
                for sym in items_col
                    .split("||")
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    push_unique(entry, sym);
                }
            }
        }
        idx
    })
}

/// Removes any previously injected `MatchKind::Emoji` candidates in place.
pub fn strip_emoji_candidates(candidates: &mut Vec<Candidate>) {
    candidates.retain(|c| c.kind != MatchKind::Emoji);
}

/// Idempotently injects relevant Unicode Emoji and/or Kaomoji candidates into `candidates` without
/// ever displacing `#1`.
pub fn inject_emoji_and_kaomoji(
    candidates: &mut Vec<Candidate>,
    raw_input: &str,
    opts: EmojiSuggestionOptions,
) {
    strip_emoji_candidates(candidates);

    if (!opts.emoji_enabled && !opts.kaomoji_enabled)
        || raw_input.is_empty()
        || candidates.is_empty()
    {
        return;
    }

    let lower_raw = raw_input.to_ascii_lowercase();
    let is_browse = is_explicit_browse_keyword(&lower_raw);

    if opts.suppress_in_ide && opts.is_developer_app && !is_browse {
        return;
    }

    let idx = global_index();
    let raw_len = raw_input.len();

    let full_span_count = candidates
        .iter()
        .take_while(|c| {
            c.consumed_bytes == raw_len && !(c.kind == MatchKind::CodeSwitch && c.text == raw_input)
        })
        .count();

    let mut primary: Vec<&'static str> = Vec::new();
    let mut overflow: Vec<&'static str> = Vec::new();

    if is_browse {
        if opts.emoji_enabled
            && let Some(list) = idx.ascii_emojis.get(lower_raw.as_str())
        {
            for &sym in list {
                push_unique(&mut primary, sym);
            }
        }
        if opts.kaomoji_enabled
            && let Some(list) = idx.ascii_kaomojis.get(lower_raw.as_str())
        {
            for &sym in list {
                push_unique(&mut primary, sym);
            }
        }
        if primary.is_empty() {
            return;
        }
        let insert_at = 1.min(candidates.len());
        let anchor_score = candidates
            .get(insert_at.saturating_sub(1))
            .map_or(0.0, |c| c.score);
        let built = build_candidates(&primary, raw_len, anchor_score);
        candidates.splice(insert_at..insert_at, built);
        return;
    }

    let is_single_char_top = full_span_count >= 1 && candidates[0].text.chars().count() == 1;

    if is_single_char_top {
        // For single-syllable characters (e.g. `zan` -> `赞`), keep #1..#3 pure Hanzi and surface
        // at most 1 high-signal Emoji at slot #4 (index 3) with zero kaomoji clutter.
        if opts.emoji_enabled
            && let Some(list) = idx.word_emojis.get(candidates[0].text.as_str())
            && let Some(&first) = list.first()
        {
            primary.push(first);
        }
        if primary.is_empty() {
            return;
        }
        let insert_at = full_span_count.clamp(1, 3).min(candidates.len());
        let anchor_score = candidates[insert_at.saturating_sub(1)].score;
        let built = build_candidates(&primary, raw_len, anchor_score);
        candidates.splice(insert_at..insert_at, built);
        return;
    }

    let (matched_emojis, matched_kaomojis) =
        collect_word_and_ascii_matches(idx, candidates, full_span_count, &lower_raw, opts);
    if matched_emojis.is_empty() && matched_kaomojis.is_empty() {
        return;
    }

    split_primary_and_overflow(
        &matched_emojis,
        &matched_kaomojis,
        &mut primary,
        &mut overflow,
    );

    // Never steal #1! Place after top 2 full-span words (or right after #1 if only 1 full-span
    // word).
    let insert_at = if full_span_count == 0 {
        1.min(candidates.len())
    } else {
        full_span_count.clamp(1, 2).min(candidates.len())
    };
    let anchor_score = candidates[insert_at.saturating_sub(1)].score;
    let primary_cands = build_candidates(&primary, raw_len, anchor_score);
    let primary_len = primary_cands.len();
    candidates.splice(insert_at..insert_at, primary_cands);

    if !overflow.is_empty() {
        let overflow_at = (full_span_count + primary_len).min(candidates.len());
        let tail_score = candidates[overflow_at.saturating_sub(1)].score - 0.1;
        let overflow_cands = build_candidates(&overflow, raw_len, tail_score);
        candidates.splice(overflow_at..overflow_at, overflow_cands);
    }
}

fn collect_word_and_ascii_matches(
    idx: &EmojiIndex,
    candidates: &[Candidate],
    full_span_count: usize,
    lower_raw: &str,
    opts: EmojiSuggestionOptions,
) -> (Vec<&'static str>, Vec<&'static str>) {
    let mut matched_emojis: Vec<&'static str> = Vec::new();
    let mut matched_kaomojis: Vec<&'static str> = Vec::new();

    for cand in candidates.iter().take(full_span_count.min(2)) {
        if cand.text.chars().count() < 2 {
            continue;
        }
        if opts.emoji_enabled
            && matched_emojis.is_empty()
            && let Some(list) = idx.word_emojis.get(cand.text.as_str())
        {
            for &sym in list {
                push_unique(&mut matched_emojis, sym);
            }
        }
        if opts.kaomoji_enabled
            && matched_kaomojis.is_empty()
            && let Some(list) = idx.word_kaomojis.get(cand.text.as_str())
        {
            for &sym in list {
                push_unique(&mut matched_kaomojis, sym);
            }
        }
    }

    if matched_emojis.is_empty()
        && opts.emoji_enabled
        && let Some(list) = idx.ascii_emojis.get(lower_raw)
    {
        for &sym in list {
            push_unique(&mut matched_emojis, sym);
        }
    }
    if matched_kaomojis.is_empty()
        && opts.kaomoji_enabled
        && let Some(list) = idx.ascii_kaomojis.get(lower_raw)
    {
        for &sym in list {
            push_unique(&mut matched_kaomojis, sym);
        }
    }

    (matched_emojis, matched_kaomojis)
}

fn split_primary_and_overflow(
    matched_emojis: &[&'static str],
    matched_kaomojis: &[&'static str],
    primary: &mut Vec<&'static str>,
    overflow: &mut Vec<&'static str>,
) {
    match (matched_emojis.is_empty(), matched_kaomojis.is_empty()) {
        (false, false) => {
            push_unique(primary, matched_emojis[0]);
            push_unique(primary, matched_kaomojis[0]);
            for &e in matched_emojis.iter().skip(1) {
                push_unique(overflow, e);
            }
            for &k in matched_kaomojis.iter().skip(1) {
                push_unique(overflow, k);
            }
        }
        (false, true) => {
            for &e in matched_emojis.iter().take(2) {
                push_unique(primary, e);
            }
            for &e in matched_emojis.iter().skip(2) {
                push_unique(overflow, e);
            }
        }
        (true, false) => {
            for &k in matched_kaomojis.iter().take(2) {
                push_unique(primary, k);
            }
            for &k in matched_kaomojis.iter().skip(2) {
                push_unique(overflow, k);
            }
        }
        (true, true) => {}
    }
    overflow.truncate(4);
}

fn build_candidates(
    symbols: &[&'static str],
    consumed_bytes: usize,
    anchor_score: f32,
) -> Vec<Candidate> {
    symbols
        .iter()
        .enumerate()
        .map(|(i, &sym)| {
            #[allow(clippy::cast_precision_loss)]
            let offset = (i + 1) as f32 * 0.001;
            Candidate {
                text: sym.to_string(),
                score: anchor_score - offset,
                kind: MatchKind::Emoji,
                consumed_bytes,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_exact(text: &str, consumed: usize, score: f32) -> Candidate {
        Candidate {
            text: text.to_string(),
            score,
            kind: MatchKind::Exact,
            consumed_bytes: consumed,
        }
    }

    #[test]
    fn test_should_inject_emoji_and_kaomoji_without_stealing_first_slot() {
        let mut cands = vec![
            mk_exact("开心", 6, -2.0),
            mk_exact("开新", 6, -4.5),
            mk_exact("开", 3, -5.0),
        ];
        inject_emoji_and_kaomoji(
            &mut cands,
            "kaixin",
            EmojiSuggestionOptions {
                emoji_enabled: true,
                kaomoji_enabled: true,
                suppress_in_ide: false,
                is_developer_app: false,
            },
        );
        assert_eq!(cands[0].text, "开心");
        assert_eq!(cands[1].text, "开新");
        assert_eq!(cands[2].kind, MatchKind::Emoji);
        assert_eq!(cands[2].text, "😄");
        assert_eq!(cands[3].kind, MatchKind::Emoji);
        assert_eq!(cands[3].text, "ヽ(✿ﾟ▽ﾟ)ノ");
        assert_eq!(cands[2].consumed_bytes, 6);

        // Idempotent re-injection produces identical results
        let snapshot = cands.clone();
        inject_emoji_and_kaomoji(
            &mut cands,
            "kaixin",
            EmojiSuggestionOptions {
                emoji_enabled: true,
                kaomoji_enabled: true,
                suppress_in_ide: false,
                is_developer_app: false,
            },
        );
        assert_eq!(cands, snapshot);
    }

    #[test]
    fn test_should_respect_configurable_ide_suppression_policy() {
        let mut normal = vec![mk_exact("笑哭", 6, -2.0)];
        inject_emoji_and_kaomoji(
            &mut normal,
            "xiaoku",
            EmojiSuggestionOptions {
                emoji_enabled: true,
                kaomoji_enabled: true,
                suppress_in_ide: false,
                is_developer_app: true,
            },
        );
        assert_eq!(normal[0].text, "笑哭");
        assert!(normal.iter().skip(1).any(|c| c.text == "😂"));

        inject_emoji_and_kaomoji(
            &mut normal,
            "xiaoku",
            EmojiSuggestionOptions {
                emoji_enabled: true,
                kaomoji_enabled: true,
                suppress_in_ide: true,
                is_developer_app: true,
            },
        );
        assert!(normal.iter().all(|c| c.kind != MatchKind::Emoji));

        let mut explicit = vec![mk_exact("颜文字", 8, -2.0)];
        inject_emoji_and_kaomoji(
            &mut explicit,
            "yanwenzi",
            EmojiSuggestionOptions {
                emoji_enabled: true,
                kaomoji_enabled: true,
                suppress_in_ide: true,
                is_developer_app: true,
            },
        );
        assert_eq!(explicit[0].text, "颜文字");
        assert!(explicit.iter().skip(1).any(|c| c.kind == MatchKind::Emoji));
    }

    #[test]
    fn test_should_keep_top_three_single_chars_untouched() {
        let mut cands = vec![
            mk_exact("赞", 3, -1.0),
            mk_exact("咱", 3, -2.0),
            mk_exact("暂", 3, -3.0),
            mk_exact("攒", 3, -4.0),
        ];
        inject_emoji_and_kaomoji(
            &mut cands,
            "zan",
            EmojiSuggestionOptions {
                emoji_enabled: true,
                kaomoji_enabled: true,
                suppress_in_ide: false,
                is_developer_app: false,
            },
        );
        assert_eq!(cands[0].text, "赞");
        assert_eq!(cands[1].text, "咱");
        assert_eq!(cands[2].text, "暂");
        assert_eq!(cands[3].text, "👍");
        assert_eq!(cands[3].kind, MatchKind::Emoji);
        assert_eq!(cands[4].text, "攒");
    }
}
