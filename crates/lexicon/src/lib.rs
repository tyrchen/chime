//! Multi-tiered lexicon storage, canonical English casing, third-party dictionary
//! importers (`.scel`, Rime `.dict.yaml`, TSV), and online user adaptation.

#![forbid(unsafe_code)]

pub mod builtin;
pub mod importers;
pub mod user_store;

use std::collections::HashMap;

use chime_phonetics::{SyllableId, canonical_syllable, extract_initial_head, lookup_syllable};
use smallvec::SmallVec;
use thiserror::Error;

pub use crate::{
    importers::{ImportStats, parse_rime_dict_yaml, parse_scel_bytes, parse_tsv_dict},
    user_store::UserStore,
};

/// Errors raised during lexicon loading or third-party dictionary import.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LexiconError {
    /// Binary `.scel` file is truncated or has invalid magic header.
    #[error("invalid Sogou .scel dictionary format: {0}")]
    InvalidScel(&'static str),
    /// I/O error while reading or saving dictionary files.
    #[error("dictionary I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A single lexical entry with log-probability weight.
#[derive(Debug, Clone, PartialEq)]
pub struct WordEntry {
    /// Target surface text (Chinese characters, English word, or mixed phrase).
    pub surface: String,
    /// Smoothed natural log-probability (closer to `0.0` is more frequent, e.g. `-4.2`).
    pub log_prob: f32,
    /// True if this entry is a technical or developer-oriented term.
    pub is_tech_term: bool,
}

/// Fast multi-index lexicon supporting exact syllable chains, initial skeletons, and English
/// casing.
#[derive(Debug, Clone, Default)]
pub struct Lexicon {
    exact_map: HashMap<SmallVec<[SyllableId; 4]>, Vec<WordEntry>>,
    skeleton_map: HashMap<String, Vec<WordEntry>>,
    english_casing: HashMap<String, WordEntry>,
    bigram_boosts: HashMap<(String, String), f32>,
}

impl Lexicon {
    /// Creates an empty lexicon.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a lexicon preloaded with the built-in permissive seed Mandarin + English corpus.
    #[must_use]
    pub fn with_builtin_seed() -> Self {
        let mut lex = Self::new();
        builtin::populate_seed_lexicon(&mut lex);
        lex
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

        let surface_str = surface.into();
        let entry = WordEntry {
            surface: surface_str,
            log_prob,
            is_tech_term,
        };

        // Insert into multi-syllable skeleton index for super-abbreviation (`zgr` -> `中国人`).
        if ids.len() >= 2 {
            // Also index single-char first letters (`zh` -> `z`) when retroflex differs.
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

    /// Looks up entries matching an exact sequence of [`SyllableId`]s.
    #[must_use]
    pub fn lookup_exact(&self, syllables: &[SyllableId]) -> &[WordEntry] {
        self.exact_map.get(syllables).map_or(&[], Vec::as_slice)
    }

    /// Looks up multi-character words matching an initial skeleton (`zgr`, `wsm`, `rgzn`).
    #[must_use]
    pub fn lookup_skeleton(&self, skeleton: &str) -> &[WordEntry] {
        self.skeleton_map.get(skeleton).map_or(&[], Vec::as_slice)
    }

    /// Looks up a canonical English term by lowercase ASCII key, or prefix-completes long tokens.
    #[must_use]
    pub fn lookup_english(&self, lower_ascii: &str) -> Option<&WordEntry> {
        self.english_casing.get(lower_ascii)
    }

    /// Finds top prefix completions for English words of length >= 5 (`kubern` -> `Kubernetes`).
    #[must_use]
    pub fn complete_english_prefix(&self, lower_prefix: &str, limit: usize) -> Vec<&WordEntry> {
        if lower_prefix.len() < 5 {
            return Vec::new();
        }
        let mut matches: Vec<&WordEntry> = self
            .english_casing
            .iter()
            .filter(|(k, _)| k.len() > lower_prefix.len() && k.starts_with(lower_prefix))
            .map(|(_, v)| v)
            .collect();
        matches.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
        matches.truncate(limit);
        matches
    }

    /// Returns any contextual bigram transition bonus for `(prev, next)`.
    #[must_use]
    pub fn bigram_bonus(&self, prev: &str, next: &str) -> f32 {
        self.bigram_boosts
            .get(&(prev.to_string(), next.to_string()))
            .copied()
            .unwrap_or(0.0)
    }

    /// Returns the total number of distinct Pinyin reading keys indexed.
    #[must_use]
    pub fn pinyin_key_count(&self) -> usize {
        self.exact_map.len()
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
