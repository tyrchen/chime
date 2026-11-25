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
    /// Automatically insert half-width spaces between Chinese and English/numbers in prose apps.
    #[serde(default = "default_true")]
    pub pangu_spacing: bool,
    /// Convert ASCII punctuation into full-width Chinese punctuation (with smart numeric `.`
    /// exemption).
    #[serde(default = "default_true")]
    pub chinese_punctuation: bool,
    /// Opt-in local recording of typing pairs in `user_store.db` for personal Colab `LoRA` export.
    #[serde(default)]
    pub record_training_samples: bool,
    /// Anti-jitter freeze window in milliseconds for asynchronous Stage-2 neural reranking.
    #[serde(default = "default_anti_jitter_ms")]
    pub anti_jitter_ms: u64,
    /// Number of candidates displayed per page (`5..=9`).
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    /// Candidate window layout orientation.
    #[serde(default)]
    pub candidate_layout: CandidateLayout,
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

impl Default for ChimeConfig {
    fn default() -> Self {
        Self {
            scheme: ShuangpinScheme::Full,
            fuzzy: FuzzyConfig::default(),
            pangu_spacing: true,
            chinese_punctuation: true,
            record_training_samples: false,
            anti_jitter_ms: 60,
            page_size: 5,
            candidate_layout: CandidateLayout::Horizontal,
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
        let encoded = toml::to_string_pretty(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }
}
