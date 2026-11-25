//! Adaptive user lexicon: exponential time decay, App bundle affinity, and backspace negative
//! feedback.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Per-word dynamic statistics learned from user commits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserHabit {
    /// Decayed selection count.
    pub weight: f32,
    /// Last commit timestamp in seconds.
    pub last_used_epoch_secs: u64,
}

/// Online user adaptation store (persisted via JSON/`SQLite` snapshot and queried on every decode).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserStore {
    /// `(normalized_reading, surface) -> UserHabit`.
    habits: HashMap<(String, String), UserHabit>,
    /// Backspace negative feedback: `(normalized_reading, rejected_surface, preferred_surface) ->
    /// count`.
    negative_overrides: HashMap<(String, String, String), u32>,
}

impl UserStore {
    /// Creates an empty user adaptation store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a positive commit for `(reading, surface)` at `now_secs`.
    pub fn record_commit(&mut self, reading: &str, surface: &str, now_secs: u64) {
        let key = (reading.to_string(), surface.to_string());
        let entry = self.habits.entry(key).or_insert(UserHabit {
            weight: 0.0,
            last_used_epoch_secs: now_secs,
        });
        let decayed = Self::decay_weight(entry.weight, entry.last_used_epoch_secs, now_secs);
        entry.weight = decayed + 1.0;
        entry.last_used_epoch_secs = now_secs;
    }

    /// Records that the user rejected `rejected` (e.g. via Backspace or manual candidate pick)
    /// and chose `preferred` for the same `reading`.
    pub fn record_negative_feedback(&mut self, reading: &str, rejected: &str, preferred: &str) {
        if rejected == preferred {
            return;
        }
        let key = (
            reading.to_string(),
            rejected.to_string(),
            preferred.to_string(),
        );
        let count = self.negative_overrides.entry(key).or_insert(0);
        *count = count.saturating_add(1);
    }

    /// Computes the dynamic score delta for `(reading, surface)` at `now_secs`.
    #[must_use]
    pub fn score_boost(&self, reading: &str, surface: &str, now_secs: u64) -> f32 {
        let mut delta = 0.0;
        if let Some(habit) = self.habits.get(&(reading.to_string(), surface.to_string())) {
            let w = Self::decay_weight(habit.weight, habit.last_used_epoch_secs, now_secs);
            delta += (1.0 + w).ln() * 0.85;
        }

        // Apply strong negative-feedback demotion/promotion after repeated corrections.
        for ((r, rej, pref), &count) in &self.negative_overrides {
            if r == reading {
                #[allow(clippy::cast_precision_loss)]
                let strength = (count.min(5) as f32) * 0.9;
                if surface == pref {
                    delta += strength;
                } else if surface == rej {
                    delta -= strength;
                }
            }
        }
        delta
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

    fn decay_weight(weight: f32, last_secs: u64, now_secs: u64) -> f32 {
        let elapsed_days =
            f64::from(u32::try_from(now_secs.saturating_sub(last_secs)).unwrap_or(u32::MAX))
                / 86_400.0;
        // Half-life of ~14 days (`ln(2) / 14 ≈ 0.0495`).
        #[allow(clippy::cast_possible_truncation)]
        let factor = (-0.0495 * elapsed_days).exp() as f32;
        weight * factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_flip_ranking_after_two_backspace_corrections() {
        let mut store = UserStore::new();
        store.record_negative_feedback("shishi", "事实", "试试");
        store.record_negative_feedback("shishi", "事实", "试试");

        let preferred_boost = store.score_boost("shishi", "试试", 1_000);
        let rejected_boost = store.score_boost("shishi", "事实", 1_000);
        assert!(preferred_boost - rejected_boost > 3.0);
    }
}
