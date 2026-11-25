//! Flat, strongly-typed TOML configuration for Chime IME (no YAML patch complexity).

use chime_phonetics::{FuzzyConfig, ShuangpinScheme};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors raised when parsing `config.toml`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// Invalid TOML syntax or field value.
    #[error("failed to parse Chime TOML config: {0}")]
    Toml(#[from] toml::de::Error),
}

/// Root user configuration loaded from `~/Library/Application Support/Chime/config.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// Anti-jitter freeze window in milliseconds for asynchronous Stage-2 neural reranking.
    #[serde(default = "default_anti_jitter_ms")]
    pub anti_jitter_ms: u64,
    /// Number of candidates displayed per page (5..=9).
    #[serde(default = "default_page_size")]
    pub page_size: usize,
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
            anti_jitter_ms: 60,
            page_size: 5,
        }
    }
}

impl ChimeConfig {
    /// Parses configuration from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Toml`] if the string is not valid TOML.
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        Ok(toml::from_str(raw)?)
    }
}
