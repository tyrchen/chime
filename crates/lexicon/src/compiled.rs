//! Zero-copy `CHIMED02` binary dictionary compiler and `memmap2` reader supporting
//! exact syllables, initial skeletons, canonical English, Bigrams, and Trigrams.

use std::{cmp::Ordering, collections::BTreeMap, fs::File, io::Write, path::Path};

use chime_phonetics::SyllableId;
use memmap2::Mmap;
use smallvec::SmallVec;

use crate::{LexiconError, WordEntry};

const MAGIC: &[u8; 8] = b"CHIMED02";
const HEADER_LEN: usize = 48;

#[derive(Debug)]
enum Backing {
    Owned(Vec<u8>),
    Mapped(Mmap),
}

impl AsRef<[u8]> for Backing {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v.as_slice(),
            Self::Mapped(m) => m.as_ref(),
        }
    }
}

/// Zero-copy compiled dictionary backed either by `mmap` or an in-memory buffer.
#[derive(Debug)]
pub struct CompiledDict {
    backing: Backing,
    pinyin_group_count: usize,
    skeleton_group_count: usize,
    english_count: usize,
    word_record_count: usize,
    bigram_count: usize,
    trigram_count: usize,
    continuation_count: usize,
    pinyin_groups_off: usize,
    skeleton_groups_off: usize,
    english_off: usize,
    word_records_off: usize,
    bigrams_off: usize,
    trigrams_off: usize,
    continuations_off: usize,
    syl_arena_off: usize,
    str_arena_off: usize,
    str_arena_len: usize,
}

impl CompiledDict {
    /// Opens and validates a `.imedict` file via read-only OS memory mapping (`mmap`).
    ///
    /// # Errors
    ///
    /// Returns [`LexiconError`] if the file cannot be opened, mapped, or fails header/UTF-8
    /// validation.
    pub fn open_mmap(path: &Path) -> Result<Self, LexiconError> {
        let file = File::open(path)?;
        // SAFETY: `file` is opened read-only for an immutable `.imedict` dictionary asset;
        // all offsets and UTF-8 string arenas are bounds-checked before constructing
        // `CompiledDict`.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_backing(Backing::Mapped(mmap))
    }

    /// Parses and validates a `.imedict` image from an owned byte vector.
    ///
    /// # Errors
    ///
    /// Returns [`LexiconError::InvalidCompiledDict`] if the header, section bounds, or UTF-8 arena
    /// is invalid.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, LexiconError> {
        Self::from_backing(Backing::Owned(bytes))
    }

    fn from_backing(backing: Backing) -> Result<Self, LexiconError> {
        let buf = backing.as_ref();
        if buf.len() < HEADER_LEN || (&buf[0..8] != MAGIC && &buf[0..8] != b"CHIMED01") {
            return Err(LexiconError::InvalidCompiledDict(
                "missing CHIMED01/CHIMED02 magic header",
            ));
        }
        let is_v2 = &buf[0..8] == MAGIC;

        let pinyin_group_count = read_u32(buf, 8) as usize;
        let skeleton_group_count = read_u32(buf, 12) as usize;
        let english_count = read_u32(buf, 16) as usize;
        let word_record_count = read_u32(buf, 20) as usize;
        let bigram_count = read_u32(buf, 24) as usize;
        let syl_arena_bytes = read_u32(buf, 28) as usize;
        let str_arena_len = read_u32(buf, 32) as usize;
        let trigram_count = if is_v2 { read_u32(buf, 36) as usize } else { 0 };
        let continuation_count = if is_v2 { read_u32(buf, 40) as usize } else { 0 };

        let pinyin_groups_off = HEADER_LEN;
        let skeleton_groups_off = pinyin_groups_off
            .checked_add(pinyin_group_count.saturating_mul(16))
            .ok_or(LexiconError::InvalidCompiledDict("section overflow"))?;
        let english_off = skeleton_groups_off
            .checked_add(skeleton_group_count.saturating_mul(16))
            .ok_or(LexiconError::InvalidCompiledDict("section overflow"))?;
        let word_records_off = english_off
            .checked_add(english_count.saturating_mul(16))
            .ok_or(LexiconError::InvalidCompiledDict("section overflow"))?;
        let bigrams_off = word_records_off
            .checked_add(word_record_count.saturating_mul(12))
            .ok_or(LexiconError::InvalidCompiledDict("section overflow"))?;
        let trigrams_off = bigrams_off
            .checked_add(bigram_count.saturating_mul(16))
            .ok_or(LexiconError::InvalidCompiledDict("section overflow"))?;
        let continuations_off = trigrams_off
            .checked_add(trigram_count.saturating_mul(16))
            .ok_or(LexiconError::InvalidCompiledDict("section overflow"))?;
        let syl_arena_off = continuations_off
            .checked_add(continuation_count.saturating_mul(16))
            .ok_or(LexiconError::InvalidCompiledDict("section overflow"))?;
        let str_arena_off = syl_arena_off
            .checked_add(syl_arena_bytes)
            .ok_or(LexiconError::InvalidCompiledDict("section overflow"))?;
        let total_expected = str_arena_off
            .checked_add(str_arena_len)
            .ok_or(LexiconError::InvalidCompiledDict("section overflow"))?;

        if buf.len() < total_expected {
            return Err(LexiconError::InvalidCompiledDict(
                "file truncated before end of string arena",
            ));
        }
        std::str::from_utf8(&buf[str_arena_off..str_arena_off + str_arena_len])
            .map_err(|_| LexiconError::InvalidCompiledDict("corrupt UTF-8 in string arena"))?;

        Ok(Self {
            backing,
            pinyin_group_count,
            skeleton_group_count,
            english_count,
            word_record_count,
            bigram_count,
            trigram_count,
            continuation_count,
            pinyin_groups_off,
            skeleton_groups_off,
            english_off,
            word_records_off,
            bigrams_off,
            trigrams_off,
            continuations_off,
            syl_arena_off,
            str_arena_off,
            str_arena_len,
        })
    }

    /// Number of distinct Pinyin readings indexed in this compiled dictionary.
    #[must_use]
    pub const fn pinyin_group_count(&self) -> usize {
        self.pinyin_group_count
    }

    /// Total word records stored in this compiled dictionary.
    #[must_use]
    pub const fn word_record_count(&self) -> usize {
        self.word_record_count
    }

    /// Total English terms stored in this compiled dictionary.
    #[must_use]
    pub const fn english_count(&self) -> usize {
        self.english_count
    }

    /// Total N-gram (Bigram + Trigram) transitions stored in this compiled dictionary.
    #[must_use]
    pub const fn ngram_count(&self) -> usize {
        self.bigram_count + self.trigram_count
    }

    /// Total verse/clause `Tab` continuations stored in this compiled dictionary.
    #[must_use]
    pub const fn continuation_count(&self) -> usize {
        self.continuation_count
    }

    /// Looks up a zero-copy `Tab` continuation string (`trigger_clause -> next_clause`) in
    /// $O(\log C)$ via deterministic 64-bit FNV-1a hash.
    #[must_use]
    pub fn lookup_continuation(&self, trigger: &str) -> Option<&str> {
        if self.continuation_count == 0 || trigger.is_empty() {
            return None;
        }
        let target_hash = hash_str_fnv1a(trigger);
        let buf = self.backing.as_ref();
        let mut lo = 0usize;
        let mut hi = self.continuation_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let off = self.continuations_off + mid * 16;
            let h = read_u64(buf, off);
            match h.cmp(&target_hash) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => {
                    let str_off = read_u32(buf, off + 8) as usize;
                    let str_len = usize::from(read_u16(buf, off + 12));
                    let s = self.get_str(str_off, str_len);
                    return (!s.is_empty()).then_some(s);
                }
            }
        }
        None
    }

    /// Appends up to `limit` matching [`WordEntry`] items for an exact `SyllableId` sequence.
    pub fn lookup_exact_into(&self, query: &[SyllableId], limit: usize, out: &mut Vec<WordEntry>) {
        let buf = self.backing.as_ref();
        let mut lo = 0usize;
        let mut hi = self.pinyin_group_count;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let grp_off = self.pinyin_groups_off + mid * 16;
            let syl_idx = read_u32(buf, grp_off) as usize;
            let syl_len = usize::from(read_u16(buf, grp_off + 4));
            match self.cmp_syllables(syl_idx, syl_len, query) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => {
                    let rec_start = read_u32(buf, grp_off + 8) as usize;
                    let rec_count = (read_u32(buf, grp_off + 12) as usize).min(limit);
                    self.collect_word_records(rec_start, rec_count, out);
                    return;
                }
            }
        }
    }

    /// Appends up to `limit` matching [`WordEntry`] items for an initial skeleton (`zgr`, `wsm`).
    pub fn lookup_skeleton_into(&self, skeleton: &str, limit: usize, out: &mut Vec<WordEntry>) {
        let buf = self.backing.as_ref();
        let mut lo = 0usize;
        let mut hi = self.skeleton_group_count;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let grp_off = self.skeleton_groups_off + mid * 16;
            let key_off = read_u32(buf, grp_off) as usize;
            let key_len = usize::from(read_u16(buf, grp_off + 4));
            let key_str = self.get_str(key_off, key_len);
            match key_str.cmp(skeleton) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => {
                    let rec_start = read_u32(buf, grp_off + 8) as usize;
                    let rec_count = (read_u32(buf, grp_off + 12) as usize).min(limit);
                    self.collect_word_records(rec_start, rec_count, out);
                    return;
                }
            }
        }
    }

    /// Looks up a canonical English term by lowercase ASCII key in $O(\log N)$.
    #[must_use]
    pub fn lookup_english(&self, lower_ascii: &str) -> Option<WordEntry> {
        let buf = self.backing.as_ref();
        let mut lo = 0usize;
        let mut hi = self.english_count;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let rec_off = self.english_off + mid * 16;
            let low_off = read_u32(buf, rec_off) as usize;
            let low_len = usize::from(read_u16(buf, rec_off + 4));
            let key_str = self.get_str(low_off, low_len);
            match key_str.cmp(lower_ascii) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(self.read_english_at(rec_off)),
            }
        }
        None
    }

    /// Returns true in $O(\log N)$ if at least one indexed Pinyin sequence strictly longer than
    /// `prefix` starts with `prefix` (enabling fast 5..=12 syllable verse/idiom lattice edges).
    #[must_use]
    pub fn has_longer_pinyin_prefix(&self, prefix: &[SyllableId]) -> bool {
        if prefix.is_empty() {
            return false;
        }
        let buf = self.backing.as_ref();
        let mut lo = 0usize;
        let mut hi = self.pinyin_group_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let grp_off = self.pinyin_groups_off + mid * 16;
            let syl_idx = read_u32(buf, grp_off) as usize;
            let syl_len = usize::from(read_u16(buf, grp_off + 4));
            if self.cmp_syllables(syl_idx, syl_len, prefix) == Ordering::Less {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        for idx in lo..(lo + 2).min(self.pinyin_group_count) {
            let grp_off = self.pinyin_groups_off + idx * 16;
            let syl_idx = read_u32(buf, grp_off) as usize;
            let syl_len = usize::from(read_u16(buf, grp_off + 4));
            if !self.starts_with_syllables(syl_idx, syl_len, prefix) {
                return false;
            }
            if syl_len > prefix.len() {
                return true;
            }
        }
        false
    }

    /// Collects up to `limit` Pinyin prefix completions (`query.len() >= 3`, e.g. `tianshengwocai`
    /// -> `天生我材必有用`) in $O(\log N)$ via binary search over sorted `pinyin_groups`.
    pub fn complete_pinyin_prefix_into(
        &self,
        query: &[SyllableId],
        limit: usize,
        out: &mut Vec<WordEntry>,
    ) {
        if query.len() < 3 {
            return;
        }
        let buf = self.backing.as_ref();
        let mut lo = 0usize;
        let mut hi = self.pinyin_group_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let grp_off = self.pinyin_groups_off + mid * 16;
            let syl_idx = read_u32(buf, grp_off) as usize;
            let syl_len = usize::from(read_u16(buf, grp_off + 4));
            if self.cmp_syllables(syl_idx, syl_len, query) == Ordering::Less {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let mut matches = Vec::new();
        let mut idx = lo;
        while idx < self.pinyin_group_count && matches.len() < 32 {
            let grp_off = self.pinyin_groups_off + idx * 16;
            let syl_idx = read_u32(buf, grp_off) as usize;
            let syl_len = usize::from(read_u16(buf, grp_off + 4));
            if !self.starts_with_syllables(syl_idx, syl_len, query) {
                break;
            }
            if syl_len > query.len() {
                let rec_start = read_u32(buf, grp_off + 8) as usize;
                let rec_count = (read_u32(buf, grp_off + 12) as usize).min(3);
                self.collect_word_records(rec_start, rec_count, &mut matches);
            }
            idx += 1;
        }
        matches.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
        for item in matches.into_iter().take(limit) {
            if !out.iter().any(|e| e.surface == item.surface) {
                out.push(item);
            }
        }
    }

    /// Collects top high-frequency 2-syllable everyday words starting with `first_syl` (`gen` ->
    /// `根据`, `根本`) in $O(\log N)$ so single-syllable candidate bars offer high-utility words
    /// instead of archaic single characters!
    pub fn complete_2syl_from_single_into(
        &self,
        first_syl: SyllableId,
        limit: usize,
        out: &mut Vec<WordEntry>,
    ) {
        let query = [first_syl];
        let buf = self.backing.as_ref();
        let mut lo = 0usize;
        let mut hi = self.pinyin_group_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let grp_off = self.pinyin_groups_off + mid * 16;
            let syl_idx = read_u32(buf, grp_off) as usize;
            let syl_len = usize::from(read_u16(buf, grp_off + 4));
            if self.cmp_syllables(syl_idx, syl_len, &query) == Ordering::Less {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let mut matches = Vec::new();
        let mut idx = lo;
        let mut scanned = 0usize;
        while idx < self.pinyin_group_count && scanned < 512 {
            let grp_off = self.pinyin_groups_off + idx * 16;
            let syl_idx = read_u32(buf, grp_off) as usize;
            let syl_len = usize::from(read_u16(buf, grp_off + 4));
            if !self.starts_with_syllables(syl_idx, syl_len, &query) {
                break;
            }
            if syl_len == 2 {
                let rec_start = read_u32(buf, grp_off + 8) as usize;
                let rec_count = (read_u32(buf, grp_off + 12) as usize).min(2);
                self.collect_word_records(rec_start, rec_count, &mut matches);
            }
            idx += 1;
            scanned += 1;
        }
        matches.retain(|w| w.log_prob >= -4.25 && w.surface.chars().count() == 2);
        matches.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
        for item in matches.into_iter().take(limit) {
            if !out.iter().any(|e| e.surface == item.surface) {
                out.push(item);
            }
        }
    }

    fn starts_with_syllables(&self, syl_idx: usize, syl_len: usize, prefix: &[SyllableId]) -> bool {
        if syl_len < prefix.len() {
            return false;
        }
        let buf = self.backing.as_ref();
        for (i, &q) in prefix.iter().enumerate() {
            let byte_off = self.syl_arena_off + (syl_idx + i) * 2;
            if SyllableId(read_u16(buf, byte_off)) != q {
                return false;
            }
        }
        true
    }

    /// Collects up to `limit` English prefix completions for `lower_prefix` (length >= 4).
    pub fn complete_english_prefix_into(
        &self,
        lower_prefix: &str,
        limit: usize,
        out: &mut Vec<WordEntry>,
    ) {
        if lower_prefix.len() < 4 {
            return;
        }
        let buf = self.backing.as_ref();
        let mut lo = 0usize;
        let mut hi = self.english_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let rec_off = self.english_off + mid * 16;
            let low_off = read_u32(buf, rec_off) as usize;
            let low_len = usize::from(read_u16(buf, rec_off + 4));
            if self.get_str(low_off, low_len) < lower_prefix {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let mut matches = Vec::new();
        let mut idx = lo;
        while idx < self.english_count && matches.len() < 64 {
            let rec_off = self.english_off + idx * 16;
            let low_off = read_u32(buf, rec_off) as usize;
            let low_len = usize::from(read_u16(buf, rec_off + 4));
            let key_str = self.get_str(low_off, low_len);
            if !key_str.starts_with(lower_prefix) {
                break;
            }
            if key_str.len() > lower_prefix.len() {
                matches.push(self.read_english_at(rec_off));
            }
            idx += 1;
        }
        matches.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
        for item in matches.into_iter().take(limit) {
            if !out.iter().any(|e| e.surface == item.surface) {
                out.push(item);
            }
        }
    }

    /// Looks up a contextual Bigram bonus in $O(\log B)$ via deterministic FNV-1a 64-bit hash.
    #[must_use]
    pub fn bigram_bonus(&self, prev: &str, next: &str) -> f32 {
        if self.bigram_count == 0 || prev.is_empty() || next.is_empty() {
            return 0.0;
        }
        self.lookup_hash_table(
            self.bigrams_off,
            self.bigram_count,
            hash_bigram_pair(prev, next),
        )
    }

    /// Looks up a contextual Trigram bonus in $O(\log T)$ via deterministic FNV-1a 64-bit hash.
    #[must_use]
    pub fn trigram_bonus(&self, prev2: &str, prev1: &str, next: &str) -> f32 {
        if self.trigram_count == 0 || prev2.is_empty() || prev1.is_empty() || next.is_empty() {
            return 0.0;
        }
        self.lookup_hash_table(
            self.trigrams_off,
            self.trigram_count,
            hash_trigram_triple(prev2, prev1, next),
        )
    }

    fn lookup_hash_table(&self, table_off: usize, count: usize, target_hash: u64) -> f32 {
        let buf = self.backing.as_ref();
        let mut lo = 0usize;
        let mut hi = count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let off = table_off + mid * 16;
            let h = read_u64(buf, off);
            match h.cmp(&target_hash) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return read_f32(buf, off + 8),
            }
        }
        0.0
    }

    fn cmp_syllables(&self, syl_idx: usize, syl_len: usize, query: &[SyllableId]) -> Ordering {
        let buf = self.backing.as_ref();
        let min_len = syl_len.min(query.len());
        for (i, &query_item) in query.iter().enumerate().take(min_len) {
            let byte_off = self.syl_arena_off + (syl_idx + i) * 2;
            let stored = SyllableId(read_u16(buf, byte_off));
            match stored.cmp(&query_item) {
                Ordering::Equal => {}
                non_eq => return non_eq,
            }
        }
        syl_len.cmp(&query.len())
    }

    fn collect_word_records(&self, rec_start: usize, rec_count: usize, out: &mut Vec<WordEntry>) {
        let buf = self.backing.as_ref();
        let end = (rec_start + rec_count).min(self.word_record_count);
        for idx in rec_start..end {
            let off = self.word_records_off + idx * 12;
            let str_off = read_u32(buf, off) as usize;
            let str_len = usize::from(read_u16(buf, off + 4));
            let flags = read_u16(buf, off + 6);
            let log_prob = read_f32(buf, off + 8);
            let surface = self.get_str(str_off, str_len);
            if let Some(existing) = out.iter_mut().find(|e| e.surface == surface) {
                if log_prob > existing.log_prob {
                    existing.log_prob = log_prob;
                }
                existing.is_tech_term |= (flags & 1) != 0;
            } else {
                out.push(WordEntry {
                    surface: surface.to_string(),
                    log_prob,
                    is_tech_term: (flags & 1) != 0,
                });
            }
        }
    }

    fn read_english_at(&self, rec_off: usize) -> WordEntry {
        let buf = self.backing.as_ref();
        let flags = read_u16(buf, rec_off + 6);
        let surf_off = read_u32(buf, rec_off + 8) as usize;
        let surf_len = usize::from(read_u16(buf, rec_off + 12));
        let quant_lp = i16::from_le_bytes([buf[rec_off + 14], buf[rec_off + 15]]);
        let log_prob = f32::from(quant_lp) / 1000.0;
        WordEntry {
            surface: self.get_str(surf_off, surf_len).to_string(),
            log_prob,
            is_tech_term: (flags & 1) != 0,
        }
    }

    fn get_str(&self, rel_off: usize, len: usize) -> &str {
        if rel_off.saturating_add(len) > self.str_arena_len {
            return "";
        }
        let start = self.str_arena_off + rel_off;
        std::str::from_utf8(&self.backing.as_ref()[start..start + len]).unwrap_or("")
    }
}

/// Compiler for serializing an in-memory [`crate::Lexicon`] into a `CHIMED02` `.imedict` binary
/// image.
#[derive(Debug, Default)]
pub struct CompiledDictBuilder {
    exact: BTreeMap<SmallVec<[SyllableId; 4]>, Vec<WordEntry>>,
    skeletons: BTreeMap<String, Vec<WordEntry>>,
    english: BTreeMap<String, WordEntry>,
    bigrams: BTreeMap<u64, f32>,
    trigrams: BTreeMap<u64, f32>,
    continuations: BTreeMap<u64, String>,
}

impl CompiledDictBuilder {
    /// Creates a new builder from the contents of `lexicon`.
    #[must_use]
    pub fn from_lexicon(lexicon: &crate::Lexicon) -> Self {
        let mut builder = Self::default();
        for (k, v) in &lexicon.exact_map {
            let mut sorted = v.clone();
            sorted.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
            builder.exact.insert(k.clone(), sorted);
        }
        for (k, v) in &lexicon.skeleton_map {
            let mut sorted = v.clone();
            sorted.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
            sorted.truncate(8);
            builder.skeletons.insert(k.clone(), sorted);
        }
        for (k, v) in &lexicon.english_casing {
            builder.english.insert(k.clone(), v.clone());
        }
        for ((prev, next), &bonus) in &lexicon.bigram_boosts {
            builder.bigrams.insert(hash_bigram_pair(prev, next), bonus);
        }
        for ((w1, w2, w3), &bonus) in &lexicon.trigram_boosts {
            builder
                .trigrams
                .insert(hash_trigram_triple(w1, w2, w3), bonus);
        }
        for (trig, cont) in &lexicon.continuations {
            if !trig.is_empty() && !cont.is_empty() {
                builder
                    .continuations
                    .insert(hash_str_fnv1a(trig), cont.clone());
            }
        }
        builder
    }

    /// Serializes the compiled dictionary into a `CHIMED02` byte vector.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn build_bytes(&self) -> Vec<u8> {
        let mut str_arena = Vec::new();
        let mut str_dedup: BTreeMap<String, (u32, u16)> = BTreeMap::new();
        let mut intern_str = |s: &str, arena: &mut Vec<u8>| -> (u32, u16) {
            if let Some(&loc) = str_dedup.get(s) {
                return loc;
            }
            let off = u32::try_from(arena.len()).unwrap_or(0);
            let capped_len = s.len().min(usize::from(u16::MAX));
            arena.extend_from_slice(&s.as_bytes()[..capped_len]);
            let len = u16::try_from(capped_len).unwrap_or(0);
            str_dedup.insert(s.to_string(), (off, len));
            (off, len)
        };

        let mut syl_arena: Vec<u8> = Vec::new();
        let mut word_records: Vec<[u8; 12]> = Vec::new();
        let mut pinyin_groups: Vec<[u8; 16]> = Vec::with_capacity(self.exact.len());

        for (syls, entries) in &self.exact {
            let syl_idx = u32::try_from(syl_arena.len() / 2).unwrap_or(0);
            let syl_len = u16::try_from(syls.len()).unwrap_or(0);
            for s in syls {
                syl_arena.extend_from_slice(&s.0.to_le_bytes());
            }
            let rec_start = u32::try_from(word_records.len()).unwrap_or(0);
            for e in entries {
                let (s_off, s_len) = intern_str(&e.surface, &mut str_arena);
                let flags: u16 = u16::from(e.is_tech_term);
                let mut rec = [0u8; 12];
                rec[0..4].copy_from_slice(&s_off.to_le_bytes());
                rec[4..6].copy_from_slice(&s_len.to_le_bytes());
                rec[6..8].copy_from_slice(&flags.to_le_bytes());
                rec[8..12].copy_from_slice(&e.log_prob.to_le_bytes());
                word_records.push(rec);
            }
            let rec_count = u32::try_from(entries.len()).unwrap_or(0);
            let mut grp = [0u8; 16];
            grp[0..4].copy_from_slice(&syl_idx.to_le_bytes());
            grp[4..6].copy_from_slice(&syl_len.to_le_bytes());
            grp[8..12].copy_from_slice(&rec_start.to_le_bytes());
            grp[12..16].copy_from_slice(&rec_count.to_le_bytes());
            pinyin_groups.push(grp);
        }

        let mut skeleton_groups: Vec<[u8; 16]> = Vec::with_capacity(self.skeletons.len());
        for (skel, entries) in &self.skeletons {
            let (k_off, k_len) = intern_str(skel, &mut str_arena);
            let rec_start = u32::try_from(word_records.len()).unwrap_or(0);
            for e in entries {
                let (s_off, s_len) = intern_str(&e.surface, &mut str_arena);
                let flags: u16 = u16::from(e.is_tech_term);
                let mut rec = [0u8; 12];
                rec[0..4].copy_from_slice(&s_off.to_le_bytes());
                rec[4..6].copy_from_slice(&s_len.to_le_bytes());
                rec[6..8].copy_from_slice(&flags.to_le_bytes());
                rec[8..12].copy_from_slice(&e.log_prob.to_le_bytes());
                word_records.push(rec);
            }
            let rec_count = u32::try_from(entries.len()).unwrap_or(0);
            let mut grp = [0u8; 16];
            grp[0..4].copy_from_slice(&k_off.to_le_bytes());
            grp[4..6].copy_from_slice(&k_len.to_le_bytes());
            grp[8..12].copy_from_slice(&rec_start.to_le_bytes());
            grp[12..16].copy_from_slice(&rec_count.to_le_bytes());
            skeleton_groups.push(grp);
        }

        let mut english_records: Vec<[u8; 16]> = Vec::with_capacity(self.english.len());
        for (lower, entry) in &self.english {
            let (l_off, l_len) = intern_str(lower, &mut str_arena);
            let (s_off, s_len) = intern_str(&entry.surface, &mut str_arena);
            let flags: u16 = u16::from(entry.is_tech_term);
            #[allow(clippy::cast_possible_truncation)]
            let quant_lp = (entry.log_prob * 1000.0).clamp(-32_000.0, 32_000.0) as i16;
            let mut rec = [0u8; 16];
            rec[0..4].copy_from_slice(&l_off.to_le_bytes());
            rec[4..6].copy_from_slice(&l_len.to_le_bytes());
            rec[6..8].copy_from_slice(&flags.to_le_bytes());
            rec[8..12].copy_from_slice(&s_off.to_le_bytes());
            rec[12..14].copy_from_slice(&s_len.to_le_bytes());
            rec[14..16].copy_from_slice(&quant_lp.to_le_bytes());
            english_records.push(rec);
        }

        let bigram_records = encode_hash_bonus_table(&self.bigrams);
        let trigram_records = encode_hash_bonus_table(&self.trigrams);
        let mut continuation_records: Vec<[u8; 16]> = Vec::with_capacity(self.continuations.len());
        for (&trig_hash, cont_str) in &self.continuations {
            let (c_off, c_len) = intern_str(cont_str, &mut str_arena);
            let mut rec = [0u8; 16];
            rec[0..8].copy_from_slice(&trig_hash.to_le_bytes());
            rec[8..12].copy_from_slice(&c_off.to_le_bytes());
            rec[12..14].copy_from_slice(&c_len.to_le_bytes());
            continuation_records.push(rec);
        }

        let mut out = Vec::with_capacity(
            HEADER_LEN
                + pinyin_groups.len() * 16
                + skeleton_groups.len() * 16
                + english_records.len() * 16
                + word_records.len() * 12
                + bigram_records.len() * 16
                + trigram_records.len() * 16
                + continuation_records.len() * 16
                + syl_arena.len()
                + str_arena.len(),
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(
            &u32::try_from(pinyin_groups.len())
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &u32::try_from(skeleton_groups.len())
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &u32::try_from(english_records.len())
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(&u32::try_from(word_records.len()).unwrap_or(0).to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(bigram_records.len())
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(&u32::try_from(syl_arena.len()).unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&u32::try_from(str_arena.len()).unwrap_or(0).to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(trigram_records.len())
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &u32::try_from(continuation_records.len())
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(&[0u8; 4]);

        for g in &pinyin_groups {
            out.extend_from_slice(g);
        }
        for g in &skeleton_groups {
            out.extend_from_slice(g);
        }
        for r in &english_records {
            out.extend_from_slice(r);
        }
        for r in &word_records {
            out.extend_from_slice(r);
        }
        for r in &bigram_records {
            out.extend_from_slice(r);
        }
        for r in &trigram_records {
            out.extend_from_slice(r);
        }
        for r in &continuation_records {
            out.extend_from_slice(r);
        }
        out.extend_from_slice(&syl_arena);
        out.extend_from_slice(&str_arena);
        out
    }

    /// Writes the compiled `.imedict` image directly to `path`.
    ///
    /// # Errors
    ///
    /// Returns [`LexiconError::Io`] on filesystem write failure.
    pub fn write_to_file(&self, path: &Path) -> Result<(), LexiconError> {
        let bytes = self.build_bytes();
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        file.flush()?;
        Ok(())
    }
}

fn encode_hash_bonus_table(map: &BTreeMap<u64, f32>) -> Vec<[u8; 16]> {
    let mut records = Vec::with_capacity(map.len());
    for (&hash, &bonus) in map {
        let mut rec = [0u8; 16];
        rec[0..8].copy_from_slice(&hash.to_le_bytes());
        rec[8..12].copy_from_slice(&bonus.to_le_bytes());
        records.push(rec);
    }
    records
}

fn hash_str_fnv1a(text: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for &b in text.as_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn hash_bigram_pair(prev: &str, next: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for &b in prev.as_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h ^= 0xff;
    h = h.wrapping_mul(FNV_PRIME);
    for &b in next.as_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn hash_trigram_triple(w1: &str, w2: &str, w3: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for part in [w1, w2, w3] {
        for &b in part.as_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
        h ^= 0xfe;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn read_u16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn read_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

fn read_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[cfg(test)]
mod tests {
    use chime_phonetics::lookup_syllable;

    use super::*;
    use crate::Lexicon;

    #[test]
    fn test_should_roundtrip_compiled_imedict_v2_with_bigrams_and_trigrams() {
        let mut seed = Lexicon::with_builtin_seed();
        seed.insert_bigram_boost("端侧", "量化", 1.8);
        seed.insert_trigram_boost("保障", "公民", "权利", 2.4);
        let builder = CompiledDictBuilder::from_lexicon(&seed);
        let dir = tempfile::tempdir().unwrap_or_else(|_| std::process::abort());
        let path = dir.path().join("test.imedict");
        builder
            .write_to_file(&path)
            .unwrap_or_else(|_| std::process::abort());

        let mapped = CompiledDict::open_mmap(&path).unwrap_or_else(|_| std::process::abort());
        let mut out = Vec::new();
        let ni = lookup_syllable("ni").unwrap_or(SyllableId(0));
        let hao = lookup_syllable("hao").unwrap_or(SyllableId(0));
        mapped.lookup_exact_into(&[ni, hao], 5, &mut out);
        assert_eq!(out.first().map(|w| w.surface.as_str()), Some("你好"));
        assert!(mapped.bigram_bonus("端侧", "量化") > 1.0);
        assert!(mapped.trigram_bonus("保障", "公民", "权利") > 2.0);
    }
}
