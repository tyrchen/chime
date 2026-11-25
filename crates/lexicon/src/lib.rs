//! Multi-tiered lexicon storage: zero-copy `mmap` `.imedict` layers, dynamic overlay,
//! canonical English casing, third-party dictionary importers, and `SQLite` WAL user store.

pub mod base_builder;
pub mod builtin;
pub mod compiled;
pub mod importers;
pub mod poetry;
pub mod user_store;

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

    /// Creates a lexicon preloaded with the built-in seed Mandarin + English corpus.
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

    /// Queries exact `syllables` returning up to `limit` entries per layer.
    #[must_use]
    pub fn lookup_exact_with_limit(
        &self,
        syllables: &[SyllableId],
        limit: usize,
    ) -> Vec<WordEntry> {
        let mut out = Vec::new();
        if let Some(overlay) = self.exact_map.get(syllables) {
            out.extend(overlay.iter().take(limit).cloned());
        }
        for layer in &self.compiled_layers {
            layer.lookup_exact_into(syllables, limit, &mut out);
        }
        out.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
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
        let mut out: Vec<WordEntry> = self
            .exact_map
            .iter()
            .filter(|(k, _)| k.len() > query.len() && k.starts_with(query))
            .flat_map(|(_, v)| v.iter().take(2).cloned())
            .collect();
        for layer in &self.compiled_layers {
            layer.complete_pinyin_prefix_into(query, limit, &mut out);
        }
        out.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
        out.truncate(limit);
        out
    }

    /// Finds top high-frequency 2-syllable words starting with `first_syl` (`gen` -> `根据`,
    /// `根本`).
    #[must_use]
    pub fn complete_2syl_from_single(&self, first_syl: SyllableId, limit: usize) -> Vec<WordEntry> {
        let mut out: Vec<WordEntry> = self
            .exact_map
            .iter()
            .filter(|(k, _)| k.len() == 2 && k[0] == first_syl)
            .flat_map(|(_, v)| v.iter().filter(|w| w.log_prob >= -4.25).take(2).cloned())
            .collect();
        for layer in &self.compiled_layers {
            layer.complete_2syl_from_single_into(first_syl, limit, &mut out);
        }
        out.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
        out.truncate(limit);
        out
    }

    /// Queries both the mutable overlay and all mounted `mmap` layers for an initial skeleton.
    #[must_use]
    pub fn lookup_skeleton(&self, skeleton: &str) -> Vec<WordEntry> {
        let mut out = Vec::new();
        if let Some(overlay) = self.skeleton_map.get(skeleton) {
            out.extend(overlay.iter().take(5).cloned());
        }
        for layer in &self.compiled_layers {
            layer.lookup_skeleton_into(skeleton, 5, &mut out);
        }
        out.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
        out
    }

    /// Looks up a canonical English term across overlay and mounted `mmap` layers.
    #[must_use]
    pub fn lookup_english(&self, lower_ascii: &str) -> Option<WordEntry> {
        if let Some(entry) = self.english_casing.get(lower_ascii) {
            return Some(entry.clone());
        }
        for layer in &self.compiled_layers {
            if let Some(entry) = layer.lookup_english(lower_ascii) {
                return Some(entry);
            }
        }
        None
    }

    /// Finds top prefix completions for English words of length >= 4 (`chatg` -> `ChatGPT`).
    #[must_use]
    pub fn complete_english_prefix(&self, lower_prefix: &str, limit: usize) -> Vec<WordEntry> {
        if lower_prefix.len() < 4 {
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
        out.truncate(limit);
        out
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
