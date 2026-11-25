//! Flat, strongly-typed TOML configuration for Chime IME.

use std::path::Path;

use chime_phonetics::{FuzzyConfig, ShuangpinScheme};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors raised when reading, writing, or parsing `config.toml`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// Filesystem I/O error.
    #[error("config I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid TOML syntax or field value.
    #[error("failed to parse Chime TOML config: {0}")]
    TomlDe(#[from] toml::de::Error),
    /// Failed to serialize TOML config.
    #[error("failed to serialize Chime TOML config: {0}")]
    TomlSer(#[from] toml::ser::Error),
}

/// Visual orientation of the floating candidate bar.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CandidateLayout {
    /// Single-row horizontal candidate bar.
    #[default]
    Horizontal,
    /// Vertical stacked candidate list.
    Vertical,
}

/// Modifier shortcut used to toggle inline Chinese/English mode inside Chime (aligned with Sogou &
/// Doubao).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnglishToggleKey {
    /// Either Left Shift (`0x38`) or Right Shift (`0x3C`) clean tap toggles mode (Sogou/Doubao
    /// default).
    #[default]
    BothShift,
    /// Only Left Shift (`0x38`) clean tap toggles mode.
    LeftShift,
    /// Only Right Shift (`0x3C`) clean tap toggles mode.
    RightShift,
    /// `Caps Lock` (`0x39`) toggles inline Chinese/English mode.
    CapsLock,
    /// Disable internal mode-switch key (rely on macOS input source switcher only).
    None,
}

/// Action performed when pressing `Shift` while actively composing Pinyin preedit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShiftCommitMode {
    /// Commit raw ASCII letters and switch into inline English mode (Sogou/Doubao default).
    #[default]
    CommitAndToggle,
    /// Commit raw ASCII letters and remain in Chinese mode (Rime/Sogou power-user mode).
    CommitKeepChinese,
}

/// Root user configuration stored at `~/Library/Application Support/Chime/config.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct ChimeConfig {
    /// Active keyboard scheme (`full`, `xiaohe`, `ziranma`, `microsoft`).
    #[serde(default)]
    pub scheme: ShuangpinScheme,
    /// Fuzzy Pinyin rules.
    #[serde(default)]
    pub fuzzy: FuzzyConfig,
    /// Automatically insert half-width spaces at Chinese-English and Chinese-number boundaries.
    #[serde(default = "default_true")]
    pub pangu_spacing: bool,
    /// Convert ASCII punctuation into full-width Chinese punctuation (with smart `3.14` / `12:30`
    /// numeric exemption).
    #[serde(default = "default_true")]
    pub chinese_punctuation: bool,
    /// Internal Chinese/English toggle shortcut (`both_shift` default).
    #[serde(default)]
    pub english_toggle_key: EnglishToggleKey,
    /// Preedit behavior when tapping `Shift` during active Pinyin input.
    #[serde(default)]
    pub shift_commit_mode: ShiftCommitMode,
    /// Treat words starting with an uppercase letter (`Hello`, `Mac`) as direct English input.
    #[serde(default = "default_true")]
    pub uppercase_triggers_english: bool,
    /// Automatically correct clear English spelling & motor-slip typos in both Chinese/mixed mode
    /// and inline `En` mode (with 1-key `Backspace` undo).
    #[serde(default = "default_true")]
    pub english_auto_correct: bool,
    /// Show a brief floating `[中]` / `[英]` HUD badge near the cursor when switching mode.
    #[serde(default = "default_true")]
    pub show_status_toast: bool,
    /// Enable `-` / `=` candidate page turning.
    #[serde(default = "default_true")]
    pub minus_equal_paging: bool,
    /// Enable `[` / `]` candidate page turning.
    #[serde(default)]
    pub bracket_paging: bool,
    /// Enable `,` / `.` candidate page turning when multiple pages exist.
    #[serde(default)]
    pub comma_period_paging: bool,
    /// Enable `;` / `'` quick selection of 2nd and 3rd candidates.
    #[serde(default)]
    pub semicolon_apostrophe_select: bool,
    /// Opt-in local recording of typing pairs in `user_store.db` for personal Colab `LoRA` export.
    #[serde(default)]
    pub record_training_samples: bool,
    /// Anti-jitter freeze window in milliseconds for asynchronous Stage-2 neural reranking.
    #[serde(default = "default_anti_jitter_ms")]
    pub anti_jitter_ms: u64,
    /// Number of candidates displayed per page (`3..=9`).
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    /// Candidate font size in points (`13..=20`, default `15`).
    #[serde(default = "default_candidate_font_size")]
    pub candidate_font_size: u32,
    /// Candidate window layout orientation.
    #[serde(default)]
    pub candidate_layout: CandidateLayout,
    /// Automatically check static `releases-manifest.json` once every 24h for app/pack updates
    /// (read-only HTTPS GET, zero telemetry or device identifiers).
    #[serde(default = "default_true")]
    pub auto_check_updates: bool,
}

const fn default_true() -> bool {
    true
}

const fn default_anti_jitter_ms() -> u64 {
    60
}

const fn default_page_size() -> usize {
    5
}

const fn default_candidate_font_size() -> u32 {
    15
}

impl Default for ChimeConfig {
    fn default() -> Self {
        Self {
            scheme: ShuangpinScheme::Full,
            fuzzy: FuzzyConfig::default(),
            pangu_spacing: true,
            chinese_punctuation: true,
            english_toggle_key: EnglishToggleKey::BothShift,
            shift_commit_mode: ShiftCommitMode::CommitAndToggle,
            uppercase_triggers_english: true,
            english_auto_correct: true,
            show_status_toast: true,
            minus_equal_paging: true,
            bracket_paging: false,
            comma_period_paging: false,
            semicolon_apostrophe_select: false,
            record_training_samples: false,
            anti_jitter_ms: 60,
            page_size: 5,
            candidate_font_size: 15,
            candidate_layout: CandidateLayout::Horizontal,
            auto_check_updates: true,
        }
    }
}

impl ChimeConfig {
    /// Parses configuration from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::TomlDe`] if the string is not valid TOML.
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        let mut cfg: Self = toml::from_str(raw)?;
        cfg.page_size = cfg.page_size.clamp(3, 9);
        cfg.candidate_font_size = cfg.candidate_font_size.clamp(13, 20);
        Ok(cfg)
    }

    /// Loads `config.toml` from `path`, or writes and returns [`Self::default()`] if absent.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if reading or parsing an existing file fails.
    pub fn load_or_init(path: &Path) -> Result<Self, ConfigError> {
        if path.exists() {
            let raw = std::fs::read_to_string(path)?;
            Self::from_toml_str(&raw)
        } else {
            let cfg = Self::default();
            cfg.save(path)?;
            Ok(cfg)
        }
    }

    /// Saves this configuration atomically as human-readable TOML to `path`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if serialization or file write fails.
    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut clean = self.clone();
        clean.page_size = clean.page_size.clamp(3, 9);
        clean.candidate_font_size = clean.candidate_font_size.clamp(13, 20);
        let encoded = toml::to_string_pretty(&clean)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }
}
