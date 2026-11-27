//! Multi-tiered lexicon storage: zero-copy `mmap` `.imedict` layers, dynamic overlay,
//! canonical English casing, third-party dictionary importers, and `SQLite` WAL user store.

pub mod base_builder;
pub mod builtin;
pub mod compiled;
pub(crate) mod generated {
    include!(concat!(env!("OUT_DIR"), "/generated_lexicon_data.rs"));
}
pub mod importers;
pub mod poetry;
pub mod user_store;
mod voice_match;

use std::{collections::HashMap, path::Path};

use chime_phonetics::{SyllableId, canonical_syllable, extract_initial_head, lookup_syllable};
use smallvec::SmallVec;
use thiserror::Error;

pub use crate::{
    base_builder::{annotate_chinese_word, build_comprehensive_lexicon},
    compiled::{CompiledDict, CompiledDictBuilder},
    importers::{
        ImportStats, import_macos_text_replacements, parse_bdict_bytes, parse_rime_dict_yaml,
        parse_scel_bytes, parse_tsv_dict,
    },
    user_store::{TypingSample, UserHabit, UserStore},
};

/// Errors raised during lexicon loading, `mmap` validation, or `SQLite` persistence.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LexiconError {
    /// Binary `.imedict` image failed header, bounds, or UTF-8 validation.
    #[error("invalid compiled .imedict dictionary: {0}")]
    InvalidCompiledDict(&'static str),
    /// Binary `.scel` file is truncated or has invalid magic header.
    #[error("invalid Sogou .scel dictionary format: {0}")]
    InvalidScel(&'static str),
    /// Binary `.bdict` file is truncated or has invalid magic header.
    #[error("invalid Baidu .bdict dictionary format: {0}")]
    InvalidBdict(&'static str),
    /// `SQLite` error in `UserStore` or `TextReplacements.db` import.
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// I/O error while reading or saving dictionary files.
    #[error("dictionary I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A single lexical entry with log-probability weight.
#[derive(Debug, Clone, PartialEq)]
pub struct WordEntry {
    /// Target surface text (Chinese characters, English word, or mixed phrase).
    pub surface: String,
    /// Smoothed natural log-probability (closer to `0.0` is more frequent, e.g. `-3.2`).
    pub log_prob: f32,
    /// True if this entry is a technical or developer-oriented term.
    pub is_tech_term: bool,
}

/// Kind of orthographic or motor-slip edit used to recover a misspelled English word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnglishEditKind {
    /// Adjacent character transposition (`teh` -> `the`, `reivew` -> `review`, `recieve` ->
    /// `receive`).
    AdjacentSwap,
    /// Accidental repeated character deletion (`untill` -> `until`, `commmit` -> `commit`).
    DuplicateDelete,
    /// Single extra interior/trailing character deletion (`truely` -> `truly`, `arguement` ->
    /// `argument`).
    SingleDelete,
    /// Physical QWERTY neighbor key or interior vowel confusion (`definately` -> `definitely`,
    /// `seperate` -> `separate`, `githib` -> `GitHub`).
    NeighborOrVowelSub,
    /// Single missing interior character insertion (`enviroment` -> `environment`, `kubernets` ->
    /// `Kubernetes`, `occured` -> `occurred`).
    InteriorInsert,
    /// Doubled-consonant position shift (`tommorow` -> `tomorrow`).
    GeminateShift,
}

/// Layered lexicon combining zero-copy `mmap` `.imedict` packs with a mutable dynamic overlay.
#[derive(Debug, Default)]
pub struct Lexicon {
    pub(crate) compiled_layers: Vec<CompiledDict>,
    pub(crate) exact_map: HashMap<SmallVec<[SyllableId; 4]>, Vec<WordEntry>>,
    pub(crate) skeleton_map: HashMap<String, Vec<WordEntry>>,
    pub(crate) english_casing: HashMap<String, WordEntry>,
    pub(crate) bigram_boosts: HashMap<(String, String), f32>,
    pub(crate) trigram_boosts: HashMap<(String, String, String), f32>,
    pub(crate) continuations: HashMap<String, String>,
}

impl Lexicon {
    /// Creates an empty lexicon.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a lexicon preloaded with declarative closed-class grammar, canonical English
    /// proper-casing, and classical couplets (without unit-test-only seed words).
    #[must_use]
    pub fn with_core_grammar() -> Self {
        let mut lex = Self::new();
        builtin::populate_core_grammar_and_casing(&mut lex);
        lex
    }

    /// Creates a lexicon preloaded with core grammar plus minimal unit-test seed words for
    /// standalone tests running without `core.imedict`.
    #[must_use]
    pub fn with_builtin_seed() -> Self {
        let mut lex = Self::new();
        builtin::populate_seed_lexicon(&mut lex);
        lex
    }

    /// Mounts a zero-copy `mmap` `.imedict` compiled dictionary layer at highest static priority.
    ///
    /// # Errors
    ///
    /// Returns [`LexiconError`] if the file cannot be mapped or fails validation.
    pub fn mount_compiled_mmap(&mut self, path: &Path) -> Result<(), LexiconError> {
        let dict = CompiledDict::open_mmap(path)?;
        self.compiled_layers.insert(0, dict);
        Ok(())
    }

    /// Mounts an already-validated [`CompiledDict`].
    pub fn mount_compiled_dict(&mut self, dict: CompiledDict) {
        self.compiled_layers.insert(0, dict);
    }

    /// Inserts a Mandarin word entry under its normalized space-separated Pinyin reading.
    pub fn insert_pinyin_word(
        &mut self,
        surface: impl Into<String>,
        pinyin_spaced: &str,
        log_prob: f32,
        is_tech_term: bool,
    ) -> bool {
        let mut ids: SmallVec<[SyllableId; 4]> = SmallVec::new();
        let mut skeleton = String::new();

        for raw_syl in pinyin_spaced.split_whitespace() {
            let norm = chime_phonetics::syllables::normalize_pinyin_syllable(raw_syl);
            let Some(id) = lookup_syllable(&norm) else {
                return false;
            };
            ids.push(id);
            let canonical = canonical_syllable(id);
            skeleton.push_str(extract_initial_head(canonical));
        }

        if ids.is_empty() {
            return false;
        }

        let entry = WordEntry {
            surface: surface.into(),
            log_prob,
            is_tech_term,
        };

        if ids.len() >= 2 {
            let compact_initials: String = ids
                .iter()
                .filter_map(|id| canonical_syllable(*id).chars().next())
                .collect();
            Self::upsert_sorted(
                self.skeleton_map.entry(compact_initials).or_default(),
                &entry,
            );
            if skeleton != entry.surface {
                Self::upsert_sorted(self.skeleton_map.entry(skeleton).or_default(), &entry);
            }
        }

        let bucket = self.exact_map.entry(ids).or_default();
        Self::upsert_sorted(bucket, &entry);
        true
    }

    /// Inserts an arbitrary ASCII shortcut (`dz` -> `北京市海淀区`) into the skeleton index.
    pub fn insert_custom_shortcut(
        &mut self,
        shortcut_ascii: &str,
        surface: impl Into<String>,
        log_prob: f32,
    ) {
        let key = shortcut_ascii.to_ascii_lowercase();
        let entry = WordEntry {
            surface: surface.into(),
            log_prob,
            is_tech_term: false,
        };
        Self::upsert_sorted(self.skeleton_map.entry(key).or_default(), &entry);
    }

    /// Inserts a canonical English / code-switching word (`macos` -> `macOS`).
    pub fn insert_english_term(
        &mut self,
        canonical_surface: impl Into<String>,
        log_prob: f32,
        is_tech_term: bool,
    ) {
        let surface = canonical_surface.into();
        let lower = surface.to_ascii_lowercase();
        self.english_casing.insert(
            lower,
            WordEntry {
                surface,
                log_prob,
                is_tech_term,
            },
        );
    }

    /// Inserts or upgrades an empirical English word entry while preserving any curated proper
    /// casing (`GitHub`, `macOS`, `Kubernetes`) or `is_tech_term` flag already present.
    pub fn upsert_calibrated_english(&mut self, surface: &str, log_prob: f32, is_tech_term: bool) {
        let lower = surface.to_ascii_lowercase();
        if let Some(existing) = self.english_casing.get_mut(&lower) {
            let existing_is_curated = existing.is_tech_term || existing.surface != lower;
            if existing_is_curated {
                existing.log_prob = existing.log_prob.max(log_prob);
                existing.is_tech_term |= is_tech_term;
                return;
            }
        }
        self.english_casing.insert(
            lower,
            WordEntry {
                surface: surface.to_string(),
                log_prob,
                is_tech_term,
            },
        );
    }

    /// Inserts a contextual bigram transition bonus `log P(next | prev)` boost.
    pub fn insert_bigram_boost(
        &mut self,
        left_word: impl Into<String>,
        right_word: impl Into<String>,
        bonus: f32,
    ) {
        self.bigram_boosts
            .insert((left_word.into(), right_word.into()), bonus);
    }

    /// Inserts a contextual trigram transition bonus `log P(w3 | w1, w2)` boost.
    pub fn insert_trigram_boost(
        &mut self,
        w1: impl Into<String>,
        w2: impl Into<String>,
        w3: impl Into<String>,
        bonus: f32,
    ) {
        self.trigram_boosts
            .insert((w1.into(), w2.into(), w3.into()), bonus);
    }

    /// Loads newline-delimited N-gram TSV rows (`2\tprev\tnext\tbonus` or `3\tw1\tw2\tw3\tbonus`).
    pub fn load_ngram_tsv(&mut self, tsv_content: &str) -> usize {
        let mut added = 0usize;
        for raw_line in tsv_content.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            match parts.as_slice() {
                ["2", w1, w2, score_str] => {
                    if let Ok(bonus) = score_str.parse::<f32>() {
                        self.insert_bigram_boost(*w1, *w2, bonus);
                        added += 1;
                    }
                }
                ["3", w1, w2, w3, score_str] => {
                    if let Ok(bonus) = score_str.parse::<f32>() {
                        self.insert_trigram_boost(*w1, *w2, *w3, bonus);
                        added += 1;
                    }
                }
                _ => {}
            }
        }
        added
    }

    /// Inserts a next-clause continuation (`天生我材必有用` -> `，千金散尽还复来`) for `Tab`
    /// auto-completion.
    pub fn insert_continuation(
        &mut self,
        trigger_clause: impl Into<String>,
        next_clause: impl Into<String>,
    ) {
        self.continuations
            .insert(trigger_clause.into(), next_clause.into());
    }

    /// Loads newline-delimited continuation TSV rows (`trigger_clause\tnext_clause`).
    pub fn load_continuations_tsv(&mut self, tsv_content: &str) -> usize {
        let mut added = 0usize;
        for raw_line in tsv_content.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((trig, next_c)) = line.split_once('\t') {
                let trig = trig.trim();
                let next_c = next_c.trim();
                if !trig.is_empty() && !next_c.is_empty() {
                    self.insert_continuation(trig, next_c);
                    added += 1;
                }
            }
        }
        added
    }

    fn lookup_continuation_raw(&self, trigger: &str) -> Option<&str> {
        if let Some(cont) = self.continuations.get(trigger) {
            return Some(cont.as_str());
        }
        for layer in &self.compiled_layers {
            if let Some(cont) = layer.lookup_continuation(trigger) {
                return Some(cont);
            }
        }
        None
    }

    /// Looks up a classical couplet or proverb next-clause continuation matching the tail of
    /// `left_context` across the mutable overlay and all mounted `mmap` `.imedict` layers.
    #[must_use]
    pub fn next_clause_completion(&self, left_context: &str) -> Option<String> {
        let trimmed = left_context.trim_end_matches(|c: char| {
            c.is_whitespace()
                || matches!(
                    c,
                    '，' | '。' | '！' | '？' | '；' | '：' | ',' | '.' | '!' | '?' | ';'
                )
        });
        if trimmed.is_empty() {
            return None;
        }
        let ended_with_punct = left_context.chars().last().is_some_and(|c| {
            matches!(
                c,
                '，' | '。' | '！' | '？' | '；' | '：' | ',' | '.' | '!' | '?' | ';'
            )
        });
        let format_cont = |cont: &str| -> String {
            if ended_with_punct {
                cont.trim_start_matches(|c: char| {
                    matches!(c, '，' | '。' | '；' | '：' | ',' | '.' | ';')
                })
                .to_string()
            } else {
                cont.to_string()
            }
        };

        let chars: Vec<char> = trimmed.chars().collect();
        let max_tail = chars.len().min(20);
        for span in (3..=max_tail).rev() {
            let suffix: String = chars[chars.len() - span..].iter().collect();
            if let Some(cont) = self.lookup_continuation_raw(&suffix) {
                return Some(format_cont(cont));
            }
        }
        None
    }

    /// Queries both the mutable overlay and all mounted `mmap` `.imedict` layers for `syllables`.
    #[must_use]
    pub fn lookup_exact(&self, syllables: &[SyllableId]) -> Vec<WordEntry> {
        let limit = if syllables.len() == 1 { 36 } else { 10 };
        self.lookup_exact_with_limit(syllables, limit)
    }

    /// Queries exact `syllables` returning up to `limit` deduplicated entries across layers.
    #[must_use]
    pub fn lookup_exact_with_limit(
        &self,
        syllables: &[SyllableId],
        limit: usize,
    ) -> Vec<WordEntry> {
        let overlay = self.exact_map.get(syllables);
        // Fast path: single compiled `mmap` layer with no dynamic overlay entry (already sorted &
        // deduplicated inside `.imedict`, avoiding O(K^2) `upsert_sorted` and double `Vec` alloc!).
        if overlay.is_none() && self.compiled_layers.len() == 1 {
            let mut out = Vec::with_capacity(limit.min(12));
            self.compiled_layers[0].lookup_exact_into(syllables, limit, &mut out);
            return out;
        }
        let mut raw = Vec::new();
        if let Some(ov) = overlay {
            raw.extend(ov.iter().take(limit).cloned());
        }
        for layer in &self.compiled_layers {
            layer.lookup_exact_into(syllables, limit, &mut raw);
        }
        let mut out = Vec::with_capacity(raw.len());
        for entry in raw {
            Self::upsert_sorted(&mut out, &entry);
        }
        out.truncate(limit);
        out
    }

    /// Returns true if any word or verse strictly longer than `prefix` starts with `prefix`.
    #[must_use]
    pub fn has_longer_pinyin_prefix(&self, prefix: &[SyllableId]) -> bool {
        if prefix.is_empty() {
            return false;
        }
        if self
            .compiled_layers
            .iter()
            .any(|l| l.has_longer_pinyin_prefix(prefix))
        {
            return true;
        }
        self.exact_map
            .keys()
            .any(|k| k.len() > prefix.len() && k.starts_with(prefix))
    }

    /// Finds top Pinyin prefix completions (`query.len() >= 3`, e.g. `tianshengwocai` ->
    /// `天生我材必有用`).
    #[must_use]
    pub fn complete_pinyin_prefix(&self, query: &[SyllableId], limit: usize) -> Vec<WordEntry> {
        if query.len() < 3 {
            return Vec::new();
        }
        let mut raw: Vec<WordEntry> = self
            .exact_map
            .iter()
            .filter(|(k, _)| k.len() > query.len() && k.starts_with(query))
            .flat_map(|(_, v)| v.iter().take(2).cloned())
            .collect();
        for layer in &self.compiled_layers {
            layer.complete_pinyin_prefix_into(query, limit, &mut raw);
        }
        let mut out = Vec::with_capacity(raw.len());
        for entry in raw {
            Self::upsert_sorted(&mut out, &entry);
        }
        out.truncate(limit);
        out
    }

    /// Finds top high-frequency 2-syllable words starting with `first_syl` (`gen` -> `根据`,
    /// `根本`).
    #[must_use]
    pub fn complete_2syl_from_single(&self, first_syl: SyllableId, limit: usize) -> Vec<WordEntry> {
        let mut raw: Vec<WordEntry> = self
            .exact_map
            .iter()
            .filter(|(k, _)| k.len() == 2 && k[0] == first_syl)
            .flat_map(|(_, v)| v.iter().filter(|w| w.log_prob >= -4.25).take(2).cloned())
            .collect();
        for layer in &self.compiled_layers {
            layer.complete_2syl_from_single_into(first_syl, limit, &mut raw);
        }
        let mut out = Vec::with_capacity(raw.len());
        for entry in raw {
            Self::upsert_sorted(&mut out, &entry);
        }
        out.truncate(limit);
        out
    }

    /// Queries both the mutable overlay and all mounted `mmap` layers for an initial skeleton.
    #[must_use]
    pub fn lookup_skeleton(&self, skeleton: &str) -> Vec<WordEntry> {
        let mut raw = Vec::new();
        if let Some(overlay) = self.skeleton_map.get(skeleton) {
            raw.extend(overlay.iter().take(5).cloned());
        }
        for layer in &self.compiled_layers {
            layer.lookup_skeleton_into(skeleton, 5, &mut raw);
        }
        let mut out = Vec::with_capacity(raw.len());
        for entry in raw {
            Self::upsert_sorted(&mut out, &entry);
        }
        out
    }

    /// Looks up a canonical English term across overlay and mounted `mmap` layers.
    #[must_use]
    pub fn lookup_english(&self, lower_ascii: &str) -> Option<WordEntry> {
        let mut best: Option<WordEntry> = self.english_casing.get(lower_ascii).cloned();
        for layer in &self.compiled_layers {
            if let Some(entry) = layer.lookup_english(lower_ascii) {
                match &mut best {
                    Some(cur) => {
                        if entry.log_prob > cur.log_prob {
                            let keep_curated = cur.is_tech_term || cur.surface != lower_ascii;
                            if keep_curated && !entry.is_tech_term && entry.surface == lower_ascii {
                                cur.log_prob = entry.log_prob;
                            } else {
                                *cur = entry;
                            }
                        }
                    }
                    None => best = Some(entry),
                }
            }
        }
        best
    }

    /// Returns true if `lower_ascii` is already a known valid English word or developer token.
    #[must_use]
    pub fn is_known_valid_english(&self, lower_ascii: &str) -> bool {
        self.lookup_english(lower_ascii).is_some_and(|e| {
            e.is_tech_term || e.log_prob >= -6.10 || (lower_ascii.len() >= 5 && e.log_prob >= -6.90)
        })
    }

    /// Recovers high-confidence 1-edit English spelling corrections (`teh` -> `the`, `reivew` ->
    /// `review`, `definately` -> `definitely`, `kubernets` -> `Kubernetes`, `tommorow` ->
    /// `tomorrow`) over zero-copy `mmap` and overlay lexicons using a stack-only probe buffer.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn correct_english_word(
        &self,
        lower_query: &str,
        max_results: usize,
    ) -> Vec<(WordEntry, f32, EnglishEditKind)> {
        let bytes = lower_query.as_bytes();
        let len = bytes.len();
        if !(3..=18).contains(&len) || max_results == 0 || !bytes.iter().all(u8::is_ascii_lowercase)
        {
            return Vec::new();
        }

        let mut hits: Vec<(WordEntry, f32, EnglishEditKind)> = Vec::new();
        let mut record_probe =
            |probe_bytes: &[u8], penalty: f32, kind: EnglishEditKind, min_lp: f32| {
                let Ok(cand_str) = std::str::from_utf8(probe_bytes) else {
                    return;
                };
                if cand_str == lower_query {
                    return;
                }
                let Some(entry) = self.lookup_english(cand_str) else {
                    return;
                };
                let eff_min = if entry.is_tech_term {
                    min_lp - 0.25
                } else {
                    min_lp
                };
                if entry.log_prob < eff_min {
                    return;
                }
                let ranked_score = entry.log_prob - penalty;
                if let Some(existing) = hits.iter_mut().find(|(e, _, _)| e.surface == entry.surface)
                {
                    if ranked_score > existing.1 {
                        existing.1 = ranked_score;
                        existing.2 = kind;
                    }
                } else {
                    hits.push((entry, ranked_score, kind));
                }
            };

        let mut buf = [0u8; 24];

        // 1. Adjacent character transposition (`teh` -> `the`, `reivew` -> `review`, `recieve` ->
        //    `receive`, `githbu` -> `GitHub`, `chatgtp` -> `ChatGPT`).
        buf[..len].copy_from_slice(bytes);
        let swap_min_lp = if len == 3 {
            -2.95
        } else if len == 4 {
            -4.10
        } else {
            -4.75
        };
        for i in 0..(len - 1) {
            if buf[i] == buf[i + 1] {
                continue;
            }
            buf.swap(i, i + 1);
            record_probe(
                &buf[..len],
                0.95,
                EnglishEditKind::AdjacentSwap,
                swap_min_lp,
            );
            buf.swap(i, i + 1);
        }

        if len == 3 {
            hits.sort_by(|a, b| b.1.total_cmp(&a.1));
            hits.truncate(max_results);
            return hits;
        }

        // 2. Duplicate or single extra letter deletion (`untill` -> `until`, `commmit` -> `commit`,
        //    `neccessary` -> `necessary`, `truely` -> `truly`, `arguement` -> `argument`).
        for i in 0..len {
            let is_dup =
                (i + 1 < len && bytes[i] == bytes[i + 1]) || (i > 0 && bytes[i] == bytes[i - 1]);
            if len == 4 && !is_dup {
                continue;
            }
            buf[..i].copy_from_slice(&bytes[..i]);
            buf[i..(len - 1)].copy_from_slice(&bytes[(i + 1)..len]);
            let (pen, kind, min_lp) = if is_dup {
                (1.02, EnglishEditKind::DuplicateDelete, -4.50)
            } else {
                (1.32, EnglishEditKind::SingleDelete, -4.25)
            };
            record_probe(&buf[..(len - 1)], pen, kind, min_lp);
        }

        // 3. Physical QWERTY neighbor substitution (`githib` -> `GitHub`) + interior vowel
        //    confusion (`definately` -> `definitely`, `seperate` -> `separate`, `independant` ->
        //    `independent`).
        buf[..len].copy_from_slice(bytes);
        for i in 0..len {
            let orig = bytes[i];
            let orig_is_vowel = matches!(orig, b'a' | b'e' | b'i' | b'o' | b'u');
            for cand_ch in b'a'..=b'z' {
                if cand_ch == orig {
                    continue;
                }
                let is_interior_vowel_swap = len >= 6
                    && i > 0
                    && i + 1 < len
                    && orig_is_vowel
                    && matches!(cand_ch, b'a' | b'e' | b'i' | b'o' | b'u');
                let pen_and_floor = if is_interior_vowel_swap {
                    Some((1.15, -4.55))
                } else {
                    chime_phonetics::typo::keyboard_substitution_cost(orig, cand_ch).map(|c| {
                        let min_lp = if len == 4 { -3.35 } else { -4.40 };
                        (1.12 + 0.32 * c, min_lp)
                    })
                };
                if let Some((pen, min_lp)) = pen_and_floor {
                    buf[i] = cand_ch;
                    record_probe(
                        &buf[..len],
                        pen,
                        EnglishEditKind::NeighborOrVowelSub,
                        min_lp,
                    );
                    buf[i] = orig;
                }
            }
        }

        // 4. Single missing interior letter insertion (`woud` -> `would`, `enviroment` ->
        //    `environment`, `kubernets` -> `Kubernetes`, `occured` -> `occurred`, `imortant` ->
        //    `important`). Never inserts at index `0` or trailing `len` (prefixes are Completions).
        for i in 1..len {
            buf[..i].copy_from_slice(&bytes[..i]);
            buf[(i + 1)..=len].copy_from_slice(&bytes[i..len]);
            for ins in b'a'..=b'z' {
                buf[i] = ins;
                let doubles_neighbor = bytes[i - 1] == ins || bytes[i] == ins;
                let (pen, min_lp) = if len == 4 {
                    (1.25, -2.85)
                } else if doubles_neighbor {
                    (1.08, -4.60)
                } else {
                    (1.30, -4.45)
                };
                record_probe(&buf[..=len], pen, EnglishEditKind::InteriorInsert, min_lp);
            }
        }

        // 5. Geminate consonant shift (`tommorow` -> `tomorrow`).
        if len >= 6 {
            for i in 0..(len - 1) {
                if bytes[i] == bytes[i + 1] && !matches!(bytes[i], b'a' | b'e' | b'i' | b'o' | b'u')
                {
                    let mut undoubled = [0u8; 24];
                    undoubled[..i].copy_from_slice(&bytes[..i]);
                    undoubled[i..(len - 1)].copy_from_slice(&bytes[(i + 1)..len]);
                    let u_len = len - 1;
                    let j_start = i.saturating_sub(2).max(1);
                    let j_end = (i + 3).min(u_len - 1);
                    for j in j_start..j_end {
                        if j == i || matches!(undoubled[j], b'a' | b'e' | b'i' | b'o' | b'u') {
                            continue;
                        }
                        buf[..=j].copy_from_slice(&undoubled[..=j]);
                        buf[j + 1] = undoubled[j];
                        buf[(j + 2)..=u_len].copy_from_slice(&undoubled[(j + 1)..u_len]);
                        record_probe(&buf[..=u_len], 1.28, EnglishEditKind::GeminateShift, -4.35);
                    }
                }
            }
        }

        hits.sort_by(|a, b| b.1.total_cmp(&a.1));
        hits.truncate(max_results);
        hits
    }

    /// Finds top prefix completions for English words of length >= 3 (`Mac` -> `MacBook`,
    /// `chatg` -> `ChatGPT`).
    #[must_use]
    pub fn complete_english_prefix(&self, lower_prefix: &str, limit: usize) -> Vec<WordEntry> {
        if lower_prefix.len() < 3 {
            return Vec::new();
        }
        let mut out: Vec<WordEntry> = self
            .english_casing
            .iter()
            .filter(|(k, _)| k.len() > lower_prefix.len() && k.starts_with(lower_prefix))
            .map(|(_, v)| v.clone())
            .collect();
        for layer in &self.compiled_layers {
            layer.complete_english_prefix_into(lower_prefix, limit, &mut out);
        }
        out.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
        let mut deduped = Vec::with_capacity(limit);
        for item in out {
            if !deduped
                .iter()
                .any(|x: &WordEntry| x.surface == item.surface)
            {
                deduped.push(item);
                if deduped.len() >= limit {
                    break;
                }
            }
        }
        deduped
    }

    /// Returns the strongest contextual bigram transition bonus across overlay and `mmap` layers.
    #[must_use]
    pub fn bigram_bonus(&self, prev: &str, next: &str) -> f32 {
        if let Some(&b) = self
            .bigram_boosts
            .get(&(prev.to_string(), next.to_string()))
        {
            return b;
        }
        for layer in &self.compiled_layers {
            let b = layer.bigram_bonus(prev, next);
            if b.abs() > f32::EPSILON {
                return b;
            }
        }
        0.0
    }

    /// Returns the strongest contextual trigram transition bonus across overlay and `mmap` layers.
    #[must_use]
    pub fn trigram_bonus(&self, prev2: &str, prev1: &str, next: &str) -> f32 {
        if let Some(&b) =
            self.trigram_boosts
                .get(&(prev2.to_string(), prev1.to_string(), next.to_string()))
        {
            return b;
        }
        for layer in &self.compiled_layers {
            let b = layer.trigram_bonus(prev2, prev1, next);
            if b.abs() > f32::EPSILON {
                return b;
            }
        }
        0.0
    }

    /// Returns total distinct Pinyin keys across overlay and mounted compiled layers.
    #[must_use]
    pub fn pinyin_key_count(&self) -> usize {
        self.exact_map.len()
            + self
                .compiled_layers
                .iter()
                .map(CompiledDict::pinyin_group_count)
                .sum::<usize>()
    }

    /// Returns total word records across mounted compiled layers plus overlay keys.
    #[must_use]
    pub fn total_word_entries(&self) -> usize {
        let overlay_words: usize = self.exact_map.values().map(Vec::len).sum();
        let compiled_words: usize = self
            .compiled_layers
            .iter()
            .map(CompiledDict::word_record_count)
            .sum();
        overlay_words + compiled_words
    }

    /// Returns total `Tab` continuation rules across overlay and mounted compiled layers.
    #[must_use]
    pub fn continuation_count(&self) -> usize {
        self.continuations.len()
            + self
                .compiled_layers
                .iter()
                .map(CompiledDict::continuation_count)
                .sum::<usize>()
    }

    fn upsert_sorted(bucket: &mut Vec<WordEntry>, entry: &WordEntry) {
        if let Some(existing) = bucket.iter_mut().find(|e| e.surface == entry.surface) {
            if entry.log_prob > existing.log_prob {
                existing.log_prob = entry.log_prob;
            }
            existing.is_tech_term |= entry.is_tech_term;
        } else {
            bucket.push(entry.clone());
        }
        bucket.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
    }
}
