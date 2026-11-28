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

/// Modifier key that is held down to dictate (push-to-talk).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceHotkey {
    /// Right `Option` (`0x3D`).
    #[default]
    RightOption,
    /// Right `Command` (`0x36`).
    RightCommand,
    /// Right `Control` (`0x3E`).
    RightControl,
    /// `Fn` / Globe (`0x3F`).
    Fn,
}

/// Source of speech recognition for hold-to-talk dictation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceBackend {
    /// Offline Streaming Paraformer installed on demand. Accept pre-model-pack configs.
    #[default]
    #[serde(alias = "bundled")]
    Paraformer,
    /// User-configured OpenAI-compatible speech service.
    Http,
}

/// Hold-to-talk dictation (`[voice]` table).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[allow(clippy::struct_excessive_bools)] // Independent, user-facing switches in config.toml.
pub struct VoiceConfig {
    /// Recognition backend; the default does not send audio over the network.
    pub backend: VoiceBackend,
    /// Master switch; microphone use is opt-in.
    pub enabled: bool,
    /// Modifier key held to dictate.
    pub hotkey: VoiceHotkey,
    /// OpenAI-compatible transcription URL (whisper.cpp `/inference`, speaches / vLLM / `LocalAI`
    /// `/v1/audio/transcriptions`, …).
    pub endpoint: String,
    /// `model` form field sent to the server.
    pub model: String,
    /// Language hint (`zh`, `en`, …); empty means auto-detect.
    pub language: String,
    /// Optional decoding prompt (vocabulary / style hint).
    pub prompt: String,
    /// Restore punctuation on the final transcript locally.
    pub auto_punctuation: bool,
    /// Opt-in removal of redundant spoken phrases from the final transcript.
    pub tidy_dictation: bool,
    /// Offer a dictionary correction on strong matches; never apply it automatically.
    pub lexicon_suggestions: bool,
    /// Opt-in: score local homophone proposals against the final transcript with the keyboard LM.
    pub pinyin_rerank: bool,
    /// Optional local `SenseVoice-Small` INT8 directory for final-only Mandarin recognition.
    pub sense_final_dir: String,
    /// Show recognized text live while speaking.
    pub live_preview: bool,
    /// Minimum gap between live-preview requests (`200..=5000` ms).
    pub partial_interval_ms: u64,
    /// Per-request timeout (`2000..=120000` ms).
    pub request_timeout_ms: u64,
    /// Recording stops automatically after this many seconds (`10..=600`).
    pub max_record_secs: u64,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: VoiceBackend::Paraformer,
            hotkey: VoiceHotkey::RightOption,
            endpoint: "http://127.0.0.1:8000/v1/audio/transcriptions".to_string(),
            model: "whisper-1".to_string(),
            language: "zh".to_string(),
            prompt: "以下是普通话的句子，使用简体中文，并带有标点符号。".to_string(),
            auto_punctuation: true,
            tidy_dictation: true,
            lexicon_suggestions: true,
            pinyin_rerank: true,
            sense_final_dir: String::new(),
            live_preview: true,
            partial_interval_ms: 700,
            request_timeout_ms: 20_000,
            max_record_secs: 180,
        }
    }
}

impl VoiceConfig {
    fn clamp(&mut self) {
        self.partial_interval_ms = self.partial_interval_ms.clamp(200, 5_000);
        self.request_timeout_ms = self.request_timeout_ms.clamp(2_000, 120_000);
        self.max_record_secs = self.max_record_secs.clamp(10, 600);
    }
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
    /// Surface offline Unicode Emoji suggestions in the candidate bar.
    #[serde(default = "default_true")]
    pub emoji_suggestions: bool,
    /// Surface offline Kaomoji (颜文字) suggestions in the candidate bar.
    #[serde(default = "default_true")]
    pub kaomoji_suggestions: bool,
    /// Suppress automatic word-triggered Emoji/Kaomoji suggestions inside terminals & IDEs
    /// (explicit browse keywords such as `emoji` / `yanwenzi` still work).
    #[serde(default)]
    pub suppress_emoji_in_ide: bool,
    /// Hold-to-talk voice dictation (`[voice]` table).
    #[serde(default)]
    pub voice: VoiceConfig,
    /// Enable optional confidence-gated Two-Tier local MLX reranking (`Qwen3-1.7B/4B-Base` 4-bit).
    #[serde(default)]
    pub tier2_enabled: bool,
    /// Optional local directory containing an MLX 4-bit Qwen3-Base bundle (`config.json`,
    /// `vocab.json`, `merges.txt`, `model.safetensors`). Empty defaults to
    /// `~/Library/Application Support/Chime/models/tier2`.
    #[serde(default)]
    pub tier2_model_dir: String,
    /// Escalate to Tier-2 only when Tier-1 `score[0] - score[1] < tier2_margin_threshold`.
    #[serde(default = "default_tier2_margin")]
    pub tier2_margin_threshold: f32,
    /// Interpolation weight for Tier-2 LM log-probability (`0.65` => `0.35 * T1 + 0.65 * T2`).
    #[serde(default = "default_tier2_alpha")]
    pub tier2_alpha: f32,
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

const fn default_tier2_margin() -> f32 {
    3.0
}

const fn default_tier2_alpha() -> f32 {
    0.65
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
            emoji_suggestions: true,
            kaomoji_suggestions: true,
            suppress_emoji_in_ide: false,
            voice: VoiceConfig::default(),
            tier2_enabled: false,
            tier2_model_dir: String::new(),
            tier2_margin_threshold: 3.0,
            tier2_alpha: 0.65,
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
        cfg.voice.clamp();
        cfg.tier2_margin_threshold = cfg.tier2_margin_threshold.clamp(0.5, 10.0);
        cfg.tier2_alpha = cfg.tier2_alpha.clamp(0.0, 1.0);
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
        clean.voice.clamp();
        clean.tier2_margin_threshold = clean.tier2_margin_threshold.clamp(0.5, 10.0);
        clean.tier2_alpha = clean.tier2_alpha.clamp(0.0, 1.0);
        let encoded = toml::to_string_pretty(&clean)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_config_without_voice_table_gets_defaults() {
        let cfg = ChimeConfig::from_toml_str("scheme = \"full\"\npage_size = 5\n")
            .unwrap_or_else(|_| std::process::abort());
        assert_eq!(cfg.voice, VoiceConfig::default());
        assert!(!cfg.voice.enabled);
        assert!(cfg.voice.auto_punctuation);
        // The settings UI exposes these as one switch, so defaults must agree.
        assert!(
            cfg.voice.tidy_dictation && cfg.voice.lexicon_suggestions && cfg.voice.pinyin_rerank
        );
        assert_eq!(cfg.voice.backend, VoiceBackend::Paraformer);
        let legacy = ChimeConfig::from_toml_str("[voice]\nbackend = \"bundled\"\n")
            .unwrap_or_else(|_| std::process::abort());
        assert_eq!(legacy.voice.backend, VoiceBackend::Paraformer);
    }

    #[test]
    fn voice_table_round_trips_and_clamps() {
        let raw = "[voice]\nenabled = true\nhotkey = \"right_command\"\nbackend = \"http\"\nendpoint = \"http://10.0.0.5:9000/inference\"\npartial_interval_ms = 10\n";
        let cfg = ChimeConfig::from_toml_str(raw).unwrap_or_else(|_| std::process::abort());
        assert!(cfg.voice.enabled);
        assert_eq!(cfg.voice.backend, VoiceBackend::Http);
        assert_eq!(cfg.voice.hotkey, VoiceHotkey::RightCommand);
        assert_eq!(cfg.voice.endpoint, "http://10.0.0.5:9000/inference");
        assert_eq!(cfg.voice.partial_interval_ms, 200);
        assert_eq!(cfg.voice.language, "zh");

        let dir = tempfile::tempdir().unwrap_or_else(|_| std::process::abort());
        let path = dir.path().join("config.toml");
        cfg.save(&path).unwrap_or_else(|_| std::process::abort());
        let back = ChimeConfig::load_or_init(&path).unwrap_or_else(|_| std::process::abort());
        assert_eq!(back, cfg);
    }

    #[test]
    fn test_tier2_config_defaults_and_clamping() -> Result<(), ConfigError> {
        let defaults = ChimeConfig::from_toml_str("")?;
        assert!(!defaults.tier2_enabled);
        assert_eq!(defaults.tier2_model_dir, "");
        assert!((defaults.tier2_margin_threshold - 3.0).abs() < 1e-6);
        assert!((defaults.tier2_alpha - 0.65).abs() < 1e-6);

        let clamped = ChimeConfig::from_toml_str(
            r#"
tier2_enabled = true
tier2_model_dir = "~/models/qwen3-4b"
tier2_margin_threshold = 99.0
tier2_alpha = -0.5
"#,
        )?;
        assert!(clamped.tier2_enabled);
        assert_eq!(clamped.tier2_model_dir, "~/models/qwen3-4b");
        assert!((clamped.tier2_margin_threshold - 10.0).abs() < 1e-6);
        assert!((clamped.tier2_alpha - 0.0).abs() < 1e-6);
        Ok(())
    }
}
