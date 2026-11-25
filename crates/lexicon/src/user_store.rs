//! Adaptive user lexicon backed by an in-memory fast cache (`Cedar` prefix trie + hash maps)
//! and durable `SQLite` (`WAL` mode) persistence for habits, per-App affinity, and negative
//! feedback.

use std::{collections::HashMap, path::Path};

use cedarwood::Cedar;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::LexiconError;

/// Per-word dynamic statistics learned from user commits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserHabit {
    /// Decayed selection weight.
    pub weight: f32,
    /// Last commit timestamp in seconds.
    pub last_used_epoch_secs: u64,
    /// Total lifetime commits.
    pub commit_count: u64,
}

/// Recorded local training sample for optional Colab `LoRA` personalization export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypingSample {
    /// Preceding context before cursor (up to 64 chars).
    pub left_context: String,
    /// Raw ASCII keys typed by user.
    pub raw_keys: String,
    /// Final surface text committed by user.
    pub committed_surface: String,
    /// Active `macOS` host app bundle ID.
    pub app_bundle_id: String,
}

/// Online user adaptation store with zero-blocking RAM reads and `SQLite` WAL write-through.
pub struct UserStore {
    /// `(normalized_reading, surface) -> UserHabit`.
    habits: HashMap<(String, String), UserHabit>,
    /// Fast dynamic prefix Double-Array Trie (`cedarwood`) of custom user readings.
    custom_words: HashMap<String, Vec<(String, f32)>>,
    reading_trie: Cedar,
    /// Backspace negative feedback: `(normalized_reading, rejected_surface, preferred_surface) ->
    /// count`.
    negative_overrides: HashMap<(String, String, String), u32>,
    /// Per-App word affinity: `(app_bundle_id, surface) -> count`.
    app_affinity: HashMap<(String, String), u32>,
    /// Online user contextual transitions: `prev_surface -> Vec<(next_surface, count,
    /// last_used_secs)>`.
    user_bigrams: HashMap<String, Vec<(String, u32, u64)>>,
    /// Optional durable `SQLite` WAL connection.
    conn: Option<Connection>,
}

impl std::fmt::Debug for UserStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserStore")
            .field("habits_len", &self.habits.len())
            .field("negative_overrides_len", &self.negative_overrides.len())
            .field("app_affinity_len", &self.app_affinity.len())
            .field("persistent", &self.conn.is_some())
            .finish_non_exhaustive()
    }
}

impl Default for UserStore {
    fn default() -> Self {
        Self::new()
    }
}

impl UserStore {
    /// Creates an in-memory user adaptation store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            habits: HashMap::new(),
            custom_words: HashMap::new(),
            reading_trie: Cedar::new(),
            negative_overrides: HashMap::new(),
            app_affinity: HashMap::new(),
            user_bigrams: HashMap::new(),
            conn: None,
        }
    }

    /// Opens or creates a persistent `SQLite` `WAL` user database at `db_path` and warms the RAM
    /// cache.
    ///
    /// # Errors
    ///
    /// Returns [`LexiconError::Sqlite`] if initialization or schema migration fails.
    #[allow(clippy::too_many_lines)]
    pub fn open_sqlite(db_path: &Path) -> Result<Self, LexiconError> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS user_habits (
                 reading TEXT NOT NULL,
                 surface TEXT NOT NULL,
                 weight REAL NOT NULL,
                 last_used_secs INTEGER NOT NULL,
                 commit_count INTEGER NOT NULL DEFAULT 1,
                 PRIMARY KEY (reading, surface)
             );
             CREATE TABLE IF NOT EXISTS negative_feedback (
                 reading TEXT NOT NULL,
                 rejected_surface TEXT NOT NULL,
                 preferred_surface TEXT NOT NULL,
                 count INTEGER NOT NULL,
                 PRIMARY KEY (reading, rejected_surface, preferred_surface)
             );
             CREATE TABLE IF NOT EXISTS app_affinity (
                 app_bundle_id TEXT NOT NULL,
                 surface TEXT NOT NULL,
                 count INTEGER NOT NULL,
                 PRIMARY KEY (app_bundle_id, surface)
             );
             CREATE TABLE IF NOT EXISTS user_bigrams (
                 prev_surface TEXT NOT NULL,
                 next_surface TEXT NOT NULL,
                 count INTEGER NOT NULL DEFAULT 1,
                 last_used_secs INTEGER NOT NULL DEFAULT 0,
                 PRIMARY KEY (prev_surface, next_surface)
             );
             CREATE TABLE IF NOT EXISTS typing_samples (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 left_context TEXT NOT NULL,
                 raw_keys TEXT NOT NULL,
                 committed_surface TEXT NOT NULL,
                 app_bundle_id TEXT NOT NULL,
                 created_secs INTEGER NOT NULL
             );",
        )?;

        let mut store = Self::new();
        {
            let mut stmt = conn.prepare(
                "SELECT reading, surface, weight, last_used_secs, commit_count FROM user_habits",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let reading: String = row.get(0)?;
                let surface: String = row.get(1)?;
                let weight: f32 = row.get(2)?;
                let last_used: i64 = row.get(3)?;
                let count: i64 = row.get(4)?;
                let habit = UserHabit {
                    weight,
                    last_used_epoch_secs: u64::try_from(last_used).unwrap_or(0),
                    commit_count: u64::try_from(count).unwrap_or(1),
                };
                store.index_custom_phrase(&reading, &surface, weight);
                store.habits.insert((reading, surface), habit);
            }
        }
        {
            let mut stmt = conn.prepare(
                "SELECT reading, rejected_surface, preferred_surface, count FROM negative_feedback",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let r: String = row.get(0)?;
                let rej: String = row.get(1)?;
                let pref: String = row.get(2)?;
                let c: u32 = row.get(3)?;
                store.negative_overrides.insert((r, rej, pref), c);
            }
        }
        {
            let mut stmt =
                conn.prepare("SELECT app_bundle_id, surface, count FROM app_affinity")?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let app: String = row.get(0)?;
                let surf: String = row.get(1)?;
                let c: u32 = row.get(2)?;
                store.app_affinity.insert((app, surf), c);
            }
        }
        {
            let mut stmt = conn.prepare(
                "SELECT prev_surface, next_surface, count, last_used_secs FROM user_bigrams",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let prev: String = row.get(0)?;
                let next: String = row.get(1)?;
                let c: u32 = row.get(2)?;
                let ts: i64 = row.get(3)?;
                let list = store.user_bigrams.entry(prev).or_default();
                list.push((next, c, u64::try_from(ts).unwrap_or(0)));
                list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
            }
        }

        store.conn = Some(conn);
        Ok(store)
    }

    /// Records a positive commit (including custom user-composed phrases!) and persists to `SQLite`
    /// WAL.
    pub fn record_commit(
        &mut self,
        reading: &str,
        surface: &str,
        app_bundle_id: &str,
        now_secs: u64,
    ) {
        if reading.is_empty() || surface.is_empty() {
            return;
        }
        let key = (reading.to_string(), surface.to_string());
        let entry = self.habits.entry(key).or_insert(UserHabit {
            weight: 0.0,
            last_used_epoch_secs: now_secs,
            commit_count: 0,
        });
        let decayed = Self::decay_weight(entry.weight, entry.last_used_epoch_secs, now_secs);
        entry.weight = decayed + 1.0;
        entry.last_used_epoch_secs = now_secs;
        entry.commit_count = entry.commit_count.saturating_add(1);
        let new_weight = entry.weight;
        let new_count = entry.commit_count;

        self.index_custom_phrase(reading, surface, new_weight);

        let app_count = if app_bundle_id.is_empty() {
            0
        } else {
            let c = self
                .app_affinity
                .entry((app_bundle_id.to_string(), surface.to_string()))
                .or_insert(0);
            *c = c.saturating_add(1);
            *c
        };

        if let Some(conn) = &self.conn {
            let _ = conn.execute(
                "INSERT INTO user_habits (reading, surface, weight, last_used_secs, commit_count)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(reading, surface) DO UPDATE SET
                   weight = excluded.weight,
                   last_used_secs = excluded.last_used_secs,
                   commit_count = excluded.commit_count",
                params![
                    reading,
                    surface,
                    new_weight,
                    i64::try_from(now_secs).unwrap_or(0),
                    i64::try_from(new_count).unwrap_or(1)
                ],
            );
            if app_count > 0 {
                let _ = conn.execute(
                    "INSERT INTO app_affinity (app_bundle_id, surface, count)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(app_bundle_id, surface) DO UPDATE SET count = excluded.count",
                    params![app_bundle_id, surface, app_count],
                );
            }
        }
    }

    /// Reverts an accidental auto-commit when the user immediately presses Backspace after Space,
    /// preventing mis-segmented Top-1 commits (`过成中`, `最和是`) from polluting `user_habits`!
    pub fn revert_commit(&mut self, reading: &str, surface: &str) {
        if reading.is_empty() || surface.is_empty() {
            return;
        }
        let key = (reading.to_string(), surface.to_string());
        let mut should_remove = false;
        let mut updated_weight = 0.0_f32;
        let mut updated_count = 0_u64;

        if let Some(entry) = self.habits.get_mut(&key) {
            entry.commit_count = entry.commit_count.saturating_sub(1);
            entry.weight = (entry.weight - 1.0).max(0.0);
            if entry.commit_count == 0 || entry.weight <= 0.05 {
                should_remove = true;
            } else {
                updated_weight = entry.weight;
                updated_count = entry.commit_count;
            }
        } else {
            return;
        }

        if should_remove {
            self.habits.remove(&key);
            if let Some(list) = self.custom_words.get_mut(reading) {
                list.retain(|(s, _)| s != surface);
            }
            if let Some(conn) = &self.conn {
                let _ = conn.execute(
                    "DELETE FROM user_habits WHERE reading = ?1 AND surface = ?2",
                    params![reading, surface],
                );
            }
        } else {
            self.index_custom_phrase(reading, surface, updated_weight);
            if let Some(conn) = &self.conn {
                let _ = conn.execute(
                    "UPDATE user_habits SET weight = ?3, commit_count = ?4
                     WHERE reading = ?1 AND surface = ?2",
                    params![
                        reading,
                        surface,
                        updated_weight,
                        i64::try_from(updated_count).unwrap_or(1)
                    ],
                );
            }
        }
    }

    /// Records that the user rejected `rejected` and chose `preferred` for `reading`.
    pub fn record_negative_feedback(&mut self, reading: &str, rejected: &str, preferred: &str) {
        if rejected == preferred || reading.is_empty() {
            return;
        }
        let key = (
            reading.to_string(),
            rejected.to_string(),
            preferred.to_string(),
        );
        let count = self.negative_overrides.entry(key).or_insert(0);
        *count = count.saturating_add(1);
        let updated = *count;

        if let Some(conn) = &self.conn {
            let _ = conn.execute(
                "INSERT INTO negative_feedback (reading, rejected_surface, preferred_surface, \
                 count)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(reading, rejected_surface, preferred_surface)
                 DO UPDATE SET count = excluded.count",
                params![reading, rejected, preferred, updated],
            );
        }
    }

    /// Appends an opt-in local typing sample (never called during `secure_input`) for Colab
    /// training.
    pub fn record_typing_sample(&self, sample: &TypingSample, now_secs: u64) {
        if let Some(conn) = &self.conn {
            let _ = conn.execute(
                "INSERT INTO typing_samples (left_context, raw_keys, committed_surface, \
                 app_bundle_id, created_secs)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    sample.left_context,
                    sample.raw_keys,
                    sample.committed_surface,
                    sample.app_bundle_id,
                    i64::try_from(now_secs).unwrap_or(0)
                ],
            );
        }
    }

    /// Exports all locally stored typing samples for Colab `LoRA` training.
    ///
    /// # Errors
    ///
    /// Returns [`LexiconError::Sqlite`] if querying `typing_samples` fails.
    pub fn export_typing_samples(&self) -> Result<Vec<TypingSample>, LexiconError> {
        let Some(conn) = &self.conn else {
            return Ok(Vec::new());
        };
        let mut stmt = conn.prepare(
            "SELECT left_context, raw_keys, committed_surface, app_bundle_id
             FROM typing_samples ORDER BY id ASC",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(TypingSample {
                left_context: row.get(0)?,
                raw_keys: row.get(1)?,
                committed_surface: row.get(2)?,
                app_bundle_id: row.get(3)?,
            });
        }
        Ok(out)
    }

    /// Looks up custom user-coined phrases matching exact `reading` via the `Cedar` trie.
    #[must_use]
    pub fn lookup_custom_phrases(&self, reading: &str) -> &[(String, f32)] {
        if self.reading_trie.exact_match_search(reading).is_some()
            && let Some(list) = self.custom_words.get(reading)
        {
            return list.as_slice();
        }
        &[]
    }

    /// Records an online user transition `(prev_surface -> next_surface)` into RAM and `SQLite`
    /// WAL so personal collocations boost future decoding AND trigger `Tab` ghost completions!
    pub fn record_bigram(&mut self, prev_surface: &str, next_surface: &str, now_secs: u64) {
        let prev = prev_surface.trim();
        let next = next_surface.trim();
        if prev.is_empty() || next.is_empty() || prev.chars().count() > 16 {
            return;
        }
        let list = self.user_bigrams.entry(prev.to_string()).or_default();
        let updated_count = if let Some(existing) = list.iter_mut().find(|(n, _, _)| n == next) {
            existing.1 = existing.1.saturating_add(1);
            existing.2 = now_secs;
            existing.1
        } else {
            list.push((next.to_string(), 1, now_secs));
            1
        };
        list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));

        if let Some(conn) = &self.conn {
            let _ = conn.execute(
                "INSERT INTO user_bigrams (prev_surface, next_surface, count, last_used_secs)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(prev_surface, next_surface) DO UPDATE SET
                   count = excluded.count,
                   last_used_secs = excluded.last_used_secs",
                params![
                    prev,
                    next,
                    updated_count,
                    i64::try_from(now_secs).unwrap_or(0)
                ],
            );
        }
    }

    /// Returns a contextual transition score boost learned from the user's own typing history.
    #[must_use]
    pub fn bigram_boost(&self, prev_surface: &str, next_surface: &str) -> f32 {
        let Some(list) = self.user_bigrams.get(prev_surface) else {
            return 0.0;
        };
        if let Some(&(_, count, _)) = list.iter().find(|(n, _, _)| n == next_surface) {
            #[allow(clippy::cast_precision_loss)]
            return 1.15 + (count as f32).ln() * 0.95;
        }
        0.0
    }

    /// Predicts a habitual next phrase for `Tab` ghost auto-completion once the user has typed
    /// `(prev -> next)` at least twice!
    #[must_use]
    pub fn predict_next_phrase(&self, left_context: &str) -> Option<String> {
        let trimmed = left_context.trim_end_matches(|c: char| {
            c.is_whitespace() || matches!(c, '，' | '。' | '！' | '？' | ',' | '.')
        });
        if trimmed.is_empty() {
            return None;
        }
        let chars: Vec<char> = trimmed.chars().collect();
        let max_tail = chars.len().min(8);
        for span in (2..=max_tail).rev() {
            let suffix: String = chars[chars.len() - span..].iter().collect();
            if let Some(list) = self.user_bigrams.get(&suffix)
                && let Some((next, count, _)) = list.first()
                && *count >= 2
                && next.chars().count() >= 2
            {
                return Some(next.clone());
            }
        }
        None
    }

    /// Computes the dynamic score delta for `(reading, surface)` in `app_bundle_id` at `now_secs`.
    #[must_use]
    pub fn score_boost(
        &self,
        reading: &str,
        surface: &str,
        app_bundle_id: &str,
        now_secs: u64,
    ) -> f32 {
        let mut delta = 0.0;
        if let Some(habit) = self.habits.get(&(reading.to_string(), surface.to_string())) {
            let w = Self::decay_weight(habit.weight, habit.last_used_epoch_secs, now_secs);
            if w > 0.05 {
                // Strong immediate promotion so picking a word once lifts it straight to #1 next
                // time!
                delta += 1.75 + (1.0 + w).ln() * 1.35;
            }
        }

        if !app_bundle_id.is_empty()
            && let Some(&app_cnt) = self
                .app_affinity
                .get(&(app_bundle_id.to_string(), surface.to_string()))
        {
            #[allow(clippy::cast_precision_loss)]
            let app_bonus = (1.0 + app_cnt as f32).ln() * 0.45;
            delta += app_bonus;
        }

        for ((r, rej, pref), &count) in &self.negative_overrides {
            if r == reading {
                #[allow(clippy::cast_precision_loss)]
                let strength = 1.45 + (count.min(5) as f32) * 0.95;
                if surface == pref {
                    delta += strength;
                } else if surface == rej {
                    delta -= strength;
                }
            }
        }
        delta
    }

    /// Returns the number of learned `(reading, surface)` habits in RAM.
    #[must_use]
    pub fn habit_count(&self) -> usize {
        self.habits.len()
    }

    /// Returns whether `app_bundle_id` belongs to a developer / technical host application.
    #[must_use]
    pub fn is_developer_app(app_bundle_id: &str) -> bool {
        let lower = app_bundle_id.to_ascii_lowercase();
        [
            "vscode",
            "ghostty",
            "terminal",
            "iterm",
            "xcode",
            "jetbrains",
            "cursor",
            "zed",
            "warp",
            "alacritty",
            "kitty",
        ]
        .iter()
        .any(|pat| lower.contains(pat))
    }

    fn index_custom_phrase(&mut self, reading: &str, surface: &str, weight: f32) {
        self.reading_trie.update(reading, 1);
        let list = self.custom_words.entry(reading.to_string()).or_default();
        if let Some(existing) = list.iter_mut().find(|(s, _)| s == surface) {
            existing.1 = weight;
        } else {
            list.push((surface.to_string(), weight));
        }
        list.sort_by(|a, b| b.1.total_cmp(&a.1));
    }

    fn decay_weight(weight: f32, last_secs: u64, now_secs: u64) -> f32 {
        let elapsed_days =
            f64::from(u32::try_from(now_secs.saturating_sub(last_secs)).unwrap_or(u32::MAX))
                / 86_400.0;
        #[allow(clippy::cast_possible_truncation)]
        let factor = (-0.0495 * elapsed_days).exp() as f32;
        weight * factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_persist_habits_and_negative_feedback_in_sqlite_wal() {
        let dir = tempfile::tempdir().unwrap_or_else(|_| std::process::abort());
        let db_path = dir.path().join("user_store.db");

        {
            let mut store =
                UserStore::open_sqlite(&db_path).unwrap_or_else(|_| std::process::abort());
            store.record_commit("wsm", "为什么呢", "com.tencent.xinWeChat", 1_000);
            store.record_negative_feedback("shishi", "事实", "试试");
            store.record_negative_feedback("shishi", "事实", "试试");
        }

        let reloaded = UserStore::open_sqlite(&db_path).unwrap_or_else(|_| std::process::abort());
        assert_eq!(reloaded.habit_count(), 1);
        assert_eq!(reloaded.lookup_custom_phrases("wsm")[0].0, "为什么呢");

        let preferred_boost =
            reloaded.score_boost("shishi", "试试", "com.tencent.xinWeChat", 1_050);
        let rejected_boost = reloaded.score_boost("shishi", "事实", "com.tencent.xinWeChat", 1_050);
        assert!(preferred_boost - rejected_boost > 3.0);
    }
}
