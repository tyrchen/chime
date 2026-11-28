//! Final-transcript suggestions. Query the active dictionary; never rewrite the ASR buffer.
use std::sync::LazyLock;

use chime_phonetics::{SyllableId, lookup_syllable};
use pinyin::{ToPinyin, ToPinyinMulti};

use crate::{Lexicon, generated::COUPLET_CHAINS};

fn separator(c: char) -> bool {
    c.is_whitespace() || "，。！？、；：,.!?;:".contains(c)
}

struct Verse {
    text: String,
    chars: Vec<char>,
    phones: Vec<&'static str>,
}

// The same canonical surfaces, readings and separators as keyboard couplet completion.
static VERSES: LazyLock<Vec<Verse>> = LazyLock::new(|| {
    COUPLET_CHAINS
        .iter()
        .map(|&(a, pa, join, b, pb)| Verse {
            text: terminated(format!("{a}{join}{b}")),
            chars: a
                .chars()
                .chain(b.chars())
                .filter(|c| !separator(*c))
                .collect(),
            phones: pa.split_whitespace().chain(pb.split_whitespace()).collect(),
        })
        .collect()
});

fn terminated(mut s: String) -> String {
    if !s.ends_with(['。', '！', '？', '!', '?', '.']) {
        s.push('。');
    }
    s
}

#[derive(Default)]
struct Best {
    score: Option<usize>,
    text: String,
    ambiguous: bool,
}

impl Best {
    fn offer(&mut self, score: usize, text: &str) {
        if self.score.is_none_or(|s| score > s) {
            self.score = Some(score);
            text.clone_into(&mut self.text);
            self.ambiguous = false;
        } else if self.score == Some(score) && self.text != text {
            self.ambiguous = true;
        }
    }
    fn result(self, raw: &str) -> Option<String> {
        (self.score.is_some() && !self.ambiguous && self.text != raw).then_some(self.text)
    }
}

fn verse_score(heard: &[char], target: &[char], phones: &[&str]) -> Option<usize> {
    let n = heard.len();
    if n < 8 || n != target.len() || n != phones.len() {
        return None;
    }
    let mut same = 0;
    let mut misses = 0;
    for ((&ch, &want), py) in heard.iter().zip(target).zip(phones) {
        if ch == want {
            same += 1;
        } else if !ch.to_pinyin_multi()?.into_iter().any(|p| p.plain() == *py) {
            misses += 1;
        }
    }
    // At least half the characters match and at most 3 out of 14 syllables disagree.
    // No insertion/deletion or prefix-only completion.
    (same * 2 >= n && misses * 14 <= n * 3).then_some(n + same - misses)
}

impl Lexicon {
    /// A single high-confidence alternative to the current hold only. Bounded exact index probes
    /// keep this independent of the number of entries in the compiled dictionary.
    #[must_use]
    pub fn voice_suggestion(&self, raw: &str) -> Option<String> {
        if raw.len() > 512 || raw.chars().take(81).count() > 80 {
            return None;
        }
        self.verse_match(raw)
            .map_or_else(|| self.term_suggestion(raw), |best| best.result(raw))
    }

    /// Whole-verse lookup, without the legacy term fallback. Used before neural term review.
    #[must_use]
    pub fn voice_verse_suggestion(&self, raw: &str) -> Option<String> {
        self.verse_match(raw)?.result(raw)
    }

    fn verse_match(&self, raw: &str) -> Option<Best> {
        if raw.len() > 512 || raw.chars().count() > 80 {
            return None;
        }
        let heard: Vec<_> = raw.chars().filter(|&c| !separator(c)).collect();
        let mut best = Best::default();
        if (8..=40).contains(&heard.len()) && heard.iter().all(|c| c.to_pinyin().is_some()) {
            for v in &*VERSES {
                if let Some(score) = verse_score(&heard, &v.chars, &v.phones) {
                    best.offer(score, &v.text);
                }
            }
            // Mounted packs store a hash of the trigger, not a reverse phonetic index of the
            // continuation. Recover candidate triggers via their exact reading, then verify
            // the *whole* stored continuation (never complete an unspoken second clause).
            let mut ids = Vec::new();
            for (i, &c) in heard.iter().take(16).enumerate() {
                let Some(id) = c.to_pinyin().and_then(|p| lookup_syllable(p.plain())) else {
                    break;
                };
                ids.push(id);
                if i < 2 || i + 3 >= heard.len() {
                    continue;
                }
                for entry in self.lookup_exact_with_limit(&ids, 8) {
                    if let Some(next) = self.lookup_continuation_raw(&entry.surface) {
                        let text = terminated(format!("{}{next}", entry.surface));
                        let chars: Vec<_> = text.chars().filter(|&c| !separator(c)).collect();
                        if chars.len() != heard.len() {
                            continue;
                        }
                        let phones: Vec<_> = chars
                            .iter()
                            .filter_map(|c| c.to_pinyin().map(pinyin::Pinyin::plain))
                            .collect();
                        if let Some(score) = verse_score(&heard, &chars, &phones) {
                            best.offer(score, &text);
                        }
                    }
                }
            }
        }
        best.score.map(|_| best)
    }

    fn term_suggestion(&self, raw: &str) -> Option<String> {
        let mut best = Best::default();
        self.visit_term_candidates(raw, 4, |length, text| {
            best.offer(length, &text);
            true
        });
        best.result(raw)
    }

    /// Small batch of full transcripts, original first. Each alternative changes only one
    /// character in a locally attested 3+ syllable word. Abstain on overly broad matches.
    #[must_use]
    pub fn voice_pinyin_candidates(&self, raw: &str) -> Vec<String> {
        if raw.len() > 256 || !(4..=64).contains(&raw.chars().count()) {
            return Vec::new();
        }
        let mut batch = vec![raw.to_owned()];
        self.visit_term_candidates(raw, 3, |_, text| {
            if !batch.contains(&text) {
                batch.push(text);
            }
            batch.len() <= 8
        });
        if (2..=8).contains(&batch.len()) {
            batch
        } else {
            Vec::new()
        }
    }

    fn visit_term_candidates(
        &self,
        raw: &str,
        min_len: usize,
        mut visit: impl FnMut(usize, String) -> bool,
    ) {
        let chars: Vec<_> = raw
            .char_indices()
            .map(|(at, c)| {
                (
                    at,
                    c,
                    c.to_pinyin().and_then(|p| lookup_syllable(p.plain())),
                )
            })
            .collect();
        for start in 0..chars.len() {
            let mut ids: Vec<SyllableId> = Vec::new();
            for &(at, c, id) in chars[start..].iter().take(16) {
                let Some(id) = id else { break };
                ids.push(id);
                if ids.len() < min_len {
                    continue;
                }
                let end = at + c.len_utf8();
                let span = &raw[chars[start].0..end];
                let entries = self.lookup_exact_with_limit(&ids, 12);
                if entries.iter().any(|e| e.surface == span) {
                    continue;
                }
                for e in entries {
                    if e.surface.chars().count() != ids.len() {
                        continue;
                    }
                    // Never propose a change to a digit or negation.
                    let changes: Vec<_> = span
                        .chars()
                        .zip(e.surface.chars())
                        .filter(|(a, b)| a != b)
                        .collect();
                    // An imported entry can carry arbitrary surface text. Check the changed
                    // character really has the same reading, not just the entry's index key.
                    if changes.len() != 1
                        || !changes[0].1.to_pinyin_multi().is_some_and(|readings| {
                            readings.into_iter().any(|p| {
                                changes[0]
                                    .0
                                    .to_pinyin()
                                    .is_some_and(|h| h.plain() == p.plain())
                            })
                        })
                        || changes.iter().any(|(a, b)| {
                            [*a, *b].iter().any(|c| {
                                "不没无未非勿莫别零〇一二两三四五六七八九十百千万亿".contains(*c)
                            })
                        })
                    {
                        continue;
                    }
                    if !visit(
                        ids.len(),
                        format!("{}{}{}", &raw[..chars[start].0], e.surface, &raw[end..]),
                    ) {
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiled::{CompiledDict, CompiledDictBuilder};
    #[test]
    fn cited_asr_errors_offer_the_complete_canonical_couplet() {
        let lex = Lexicon::with_core_grammar();
        assert_quote_suggestions(&lex);
    }

    #[test]
    #[ignore = "requires CHIME_TEST_CORE_DICT pointing at the packaged core.imedict"]
    fn packaged_dictionary() -> Result<(), Box<dyn std::error::Error>> {
        let mut lex = Lexicon::with_core_grammar();
        lex.mount_compiled_mmap(std::path::Path::new(&std::env::var(
            "CHIME_TEST_CORE_DICT",
        )?))?;
        assert_quote_suggestions(&lex);
        Ok(())
    }

    fn assert_quote_suggestions(lex: &Lexicon) {
        let verse = "曾经沧海难为水，除却巫山不是云。";
        for raw in [
            "陈金沧海南尾水除窃巫山，不是云。",
            "曾经沧海，南尾水储却无山不是云。",
        ] {
            assert_eq!(lex.voice_suggestion(raw).as_deref(), Some(verse));
        }
        for raw in [
            verse,
            "无山",
            "这里无山也无水",
            "曾经沧海难为水",
            "曾经沧海难为水下班还要买菜呢",
            "曾经沧海难为水123除却巫山不是云",
        ] {
            assert_eq!(lex.voice_suggestion(raw), None, "{raw}");
        }
    }
    #[test]
    fn bounded_pinyin_edits_preserve_original_and_protected_tokens() {
        let mut lex = Lexicon::new();
        for (word, py) in [
            ("数据库", "shu ju ku"),
            ("数X库", "shu ju ku"),
            ("无线上网", "wu xian shang wang"),
            ("一心一意", "yi xin yi yi"),
        ] {
            lex.insert_pinyin_word(word, py, -2., false);
        }
        assert_eq!(
            lex.voice_pinyin_candidates("API 3.14 的数剧库。"),
            ["API 3.14 的数剧库。", "API 3.14 的数据库。"]
        );
        for raw in [
            "API 3.14 的数据库。",
            "我用吴线上网。",
            "我已心一意。",
            "无山",
            &"数剧库".repeat(30),
        ] {
            assert!(lex.voice_pinyin_candidates(raw).is_empty(), "{raw}");
        }
    }

    #[test]
    fn suggestions_follow_mounted_layers_and_keep_surrounding_text()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut pack = Lexicon::new();
        pack.insert_pinyin_word("星河观测站", "xing he guan ce zhan", -2.0, false);
        pack.insert_continuation("星河观测站", "，月落照江南");
        let mut lex = Lexicon::new();
        let raw = "请联系星河关测站。";
        assert_eq!(lex.voice_suggestion(raw), None);
        lex.mount_compiled_dict(CompiledDict::from_bytes(
            CompiledDictBuilder::from_lexicon(&pack).build_bytes(),
        )?);
        assert_eq!(
            lex.voice_suggestion(raw).as_deref(),
            Some("请联系星河观测站。")
        );
        assert_eq!(
            lex.voice_suggestion("星河关测站月落照江南").as_deref(),
            Some("星河观测站，月落照江南。")
        );
        lex.insert_pinyin_word("星河关侧站", "xing he guan ce zhan", -2.0, false);
        assert_eq!(lex.voice_suggestion(raw), None);
        Ok(())
    }
}
