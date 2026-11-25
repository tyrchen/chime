//! Chime IME core session state machine, bounded LRU session pool, smart Chinese punctuation,
//! step-by-step partial selection, candidate paging, `Tab` typo rollback, and Stage-2 async
//! reranking.

#![forbid(unsafe_code)]

pub mod config;
pub mod personalize;
pub mod punctuation;
pub mod sanitize;

use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use chime_decoder::{
    Candidate, DecodeContext, LatticeDecoder, MatchKind, apply_anti_jitter_freeze_window,
    promote_top_completion_into_bar,
};
use chime_lexicon::{
    CompiledDictBuilder, EnglishEditKind, Lexicon, TypingSample, UserStore,
    import_macos_text_replacements, parse_bdict_bytes, parse_rime_dict_yaml, parse_scel_bytes,
    parse_tsv_dict,
};
use chime_neural::{AsyncRerankHandle, LoraMountSpec, RerankJob};
use chime_phonetics::{ShuangpinScheme, format_segmented_preedit};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::{
    config::{CandidateLayout, ChimeConfig, ConfigError, EnglishToggleKey, ShiftCommitMode},
    personalize::{
        ImePackManifest, InstalledImePack, PackBuildOptions, PersonalizeError, TrainingRecord,
        build_colab_training_jsonl, build_imepack_dir, build_imepack_v2, discover_installed_packs,
        install_imepack_dir,
    },
    punctuation::PunctuationState,
    sanitize::{SanitizeReport, sanitize_corpus_text},
};

/// Maximum concurrent sessions retained in Rust regardless of leaked `IMKInputController` shells.
pub const MAX_ACTIVE_SESSIONS: usize = 32;

/// Errors raised when importing external dictionaries/packs or reloading `ChimeEngine`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EngineError {
    /// Configuration error.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// Lexicon / `SQLite` / `mmap` error.
    #[error(transparent)]
    Lexicon(#[from] chime_lexicon::LexiconError),
    /// Neural model load error.
    #[error(transparent)]
    Neural(#[from] chime_neural::NeuralError),
    /// Personalization pack error.
    #[error(transparent)]
    Personalize(#[from] PersonalizeError),
    /// I/O error.
    #[error("engine I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Key event passed across the FFI boundary from `macOS` `InputMethodKit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyEvent {
    /// Printable ASCII byte (`0` when non-ASCII special key such as Arrow/PageUp).
    pub ascii_char: u8,
    /// `macOS` virtual key code (`0x24` Return, `0x30` Tab, `0x31` Space, `0x33` Delete,
    /// `0x35` Escape, `0x7B..0x7E` Arrows, `0x74`/`0x79` PageUp/Down).
    pub key_code: u16,
    /// True if Shift is held.
    pub shift: bool,
    /// True if Command, Control, or Option is held.
    pub has_command_or_ctrl: bool,
    /// True if `macOS` `IsSecureEventInputEnabled()` is active (password / `sudo` field).
    pub secure_input: bool,
}

/// Snapshot returned synchronously (<2ms) to the Swift UI on every keystroke or navigation action.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct EngineResponse {
    /// True if the IME consumed the keystroke; false if `macOS` should pass it to the client app.
    pub handled: bool,
    /// Text to commit immediately to `NSTextInputClient.insertText`.
    pub committed_text: String,
    /// Formatted inline marked preedit text (`今天 wo'men` or `ni'hao`).
    pub preedit_text: String,
    /// Current page of ranked candidates for the non-activating `NSPanel`.
    pub candidates: Vec<Candidate>,
    /// Highlighted candidate index (`0..candidates.len()`) within the current page.
    pub highlighted_index: usize,
    /// Current 0-based page index.
    pub page_index: usize,
    /// Total number of candidate pages available.
    pub total_pages: usize,
    /// True when inline English pass-through mode is toggled on.
    pub inline_english_mode: bool,
    /// True when Chinese full-width smart punctuation (`。，`) is active (`false` = ASCII `.,`).
    pub chinese_punctuation: bool,
    /// True when the candidate panel is expanded into the `5x5` 2D grid (`↓` key).
    pub grid_expanded: bool,
    /// Next-clause contextual auto-completion preview (`⇥ ，千金散尽还复来`) ready for `Tab`.
    pub ghost_text: String,
    /// Candidate window layout orientation (`horizontal` or `vertical`).
    pub candidate_layout: CandidateLayout,
    /// Candidate font size in points (`13..=20`).
    pub candidate_font_size: u32,
    /// Whether to display a brief HUD badge near the cursor on mode/punctuation switch.
    pub show_status_toast: bool,
    /// Number of characters immediately preceding the cursor to replace when committing an inline
    /// `En`-mode spelling correction or `Backspace` undo (`0` for normal insertion).
    #[serde(default)]
    pub replace_prev_chars: usize,
    /// Optional custom status toast text (e.g. `"teh → the"` or `"已撤回 → teh"`).
    #[serde(default)]
    pub toast_override: String,
    /// Monotonic keystroke generation ID for anti-jitter async rerank validation.
    pub generation_id: u64,
}

/// Summary of a single mounted `.imepack` cartridge reported to the `macOS` Preferences window.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledPackSummary {
    /// Canonical cartridge slug (`"classics-poetry"`, `"tech-developer"`, …).
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Installed cartridge version string.
    pub version: String,
    /// True if this cartridge mounts a `LoRA` adapter onto Metal GPU.
    pub has_lora: bool,
}

/// Runtime status summary displayed in the `macOS` Preferences window.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineStatus {
    /// Active keyboard scheme name.
    pub scheme: ShuangpinScheme,
    /// Total indexed Pinyin reading groups across static + user layers.
    pub pinyin_keys: usize,
    /// Total word entries across static + user layers.
    pub word_entries: usize,
    /// Learned user habits in `SQLite` WAL store.
    pub learned_habits: usize,
    /// True if a Stage-2 `Safetensors` neural reranker is active.
    pub neural_loaded: bool,
    /// Number of installed `.imepack` cartridges mounted from `packs/`.
    #[serde(default)]
    pub installed_packs: usize,
    /// Number of active `Multi-LoRA` sidecar adapters mounted on Metal GPU.
    #[serde(default)]
    pub active_loras: usize,
    /// Detailed metadata for each mounted `.imepack` cartridge.
    #[serde(default)]
    pub pack_details: Vec<InstalledPackSummary>,
}

#[derive(Debug, Clone)]
struct EnAutoCorrectUndo {
    original_word: String,
    emitted_chars: usize,
}

#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
struct InputSession {
    id: u64,
    app_bundle_id: String,
    inline_english_mode: bool,
    /// Already-selected prefix words during step-by-step sentence composition.
    partial_prefix: String,
    /// Full original raw keys accumulated for the current sentence (for habit learning).
    full_sentence_keys: String,
    /// Remaining unconsumed raw ASCII keys.
    raw_buffer: String,
    left_context: String,
    last_committed_char: Option<char>,
    punctuation: PunctuationState,
    all_candidates: Vec<Candidate>,
    selected_global_idx: usize,
    grid_expanded: bool,
    ghost_suggestion: Option<String>,
    ghost_dismissed: bool,
    last_committed_word: String,
    typo_rollback_prev_idx: Option<usize>,
    last_auto_committed: Option<(String, String)>,
    en_word_buffer: String,
    en_token_tainted: bool,
    en_last_autocorrect: Option<EnAutoCorrectUndo>,
    en_suppressed_words: Vec<String>,
    generation_id: u64,
}

impl InputSession {
    fn new(id: u64, app_bundle_id: String) -> Self {
        Self {
            id,
            app_bundle_id,
            inline_english_mode: false,
            partial_prefix: String::new(),
            full_sentence_keys: String::new(),
            raw_buffer: String::new(),
            left_context: String::new(),
            last_committed_char: None,
            punctuation: PunctuationState::default(),
            all_candidates: Vec::new(),
            selected_global_idx: 0,
            grid_expanded: false,
            ghost_suggestion: None,
            ghost_dismissed: false,
            last_committed_word: String::new(),
            typo_rollback_prev_idx: None,
            last_auto_committed: None,
            en_word_buffer: String::new(),
            en_token_tainted: false,
            en_last_autocorrect: None,
            en_suppressed_words: Vec::new(),
            generation_id: 0,
        }
    }

    fn clear_en_tracking(&mut self) {
        self.en_word_buffer.clear();
        self.en_token_tainted = false;
        self.en_last_autocorrect = None;
    }

    fn pop_left_context_chars(&mut self, count: usize) {
        for _ in 0..count {
            self.left_context.pop();
        }
        self.last_committed_char = self.left_context.chars().last();
    }

    fn is_composing(&self) -> bool {
        !self.raw_buffer.is_empty() || !self.partial_prefix.is_empty()
    }

    fn clear_composing(&mut self) {
        self.partial_prefix.clear();
        self.full_sentence_keys.clear();
        self.raw_buffer.clear();
        self.all_candidates.clear();
        self.selected_global_idx = 0;
        self.grid_expanded = false;
        self.typo_rollback_prev_idx = None;
    }

    fn record_emitted_text(&mut self, emitted: &str) {
        if let Some(last_ch) = emitted.chars().last() {
            self.last_committed_char = Some(last_ch);
        }
        self.left_context.push_str(emitted);
        if self.left_context.chars().count() > 128 {
            self.left_context = self
                .left_context
                .chars()
                .rev()
                .take(128)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
        }
    }
}

/// Core IME engine managing layered `mmap` lexicons, `SQLite` WAL user store, Stage-2 worker, and
/// LRU sessions.
#[derive(Debug)]
pub struct ChimeEngine {
    app_support_dir: Option<PathBuf>,
    bundle_resources_dir: Option<PathBuf>,
    config: ChimeConfig,
    lexicon: Lexicon,
    user_store: UserStore,
    decoder: LatticeDecoder,
    async_reranker: Option<AsyncRerankHandle>,
    installed_packs_count: usize,
    installed_packs_meta: Vec<InstalledPackSummary>,
    inline_english_mode: bool,
    sessions: VecDeque<InputSession>,
    next_session_id: u64,
}

impl Default for ChimeEngine {
    fn default() -> Self {
        Self::new(ChimeConfig::default())
    }
}

impl ChimeEngine {
    /// Initializes an in-memory engine with built-in seed vocabulary (ideal for unit tests &
    /// stateless CLI).
    #[must_use]
    pub fn new(config: ChimeConfig) -> Self {
        Self {
            app_support_dir: None,
            bundle_resources_dir: None,
            config,
            lexicon: Lexicon::with_builtin_seed(),
            user_store: UserStore::new(),
            decoder: LatticeDecoder::new(),
            async_reranker: None,
            installed_packs_count: 0,
            installed_packs_meta: Vec::new(),
            inline_english_mode: false,
            sessions: VecDeque::with_capacity(MAX_ACTIVE_SESSIONS),
            next_session_id: 1,
        }
    }

    /// Initializes a full production engine backed by `app_support_dir` (`config.toml`,
    /// `user_store.db`, `dicts/*.imedict`, `packs/*.imepack`, `neural/`) and optional
    /// `bundle_resources_dir/core.imedict`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] if creating `app_support_dir` fails. Individual corrupt optional
    /// assets degrade gracefully without crashing the IME.
    pub fn with_data_dir(
        app_support_dir: &Path,
        bundle_resources_dir: Option<&Path>,
    ) -> Result<Self, EngineError> {
        std::fs::create_dir_all(app_support_dir)?;
        std::fs::create_dir_all(app_support_dir.join("dicts"))?;
        std::fs::create_dir_all(app_support_dir.join("packs"))?;

        let config =
            ChimeConfig::load_or_init(&app_support_dir.join("config.toml")).unwrap_or_default();
        let user_store =
            UserStore::open_sqlite(&app_support_dir.join("user_store.db")).unwrap_or_default();

        let mut engine = Self {
            app_support_dir: Some(app_support_dir.to_path_buf()),
            bundle_resources_dir: bundle_resources_dir.map(Path::to_path_buf),
            config,
            lexicon: Lexicon::with_builtin_seed(),
            user_store,
            decoder: LatticeDecoder::new(),
            async_reranker: None,
            installed_packs_count: 0,
            installed_packs_meta: Vec::new(),
            inline_english_mode: false,
            sessions: VecDeque::with_capacity(MAX_ACTIVE_SESSIONS),
            next_session_id: 1,
        };
        engine.reload_assets();
        Ok(engine)
    }

    /// Hot-reloads `config.toml`, mounted `.imedict` dictionaries, `.imepack` cartridges,
    /// `macOS` text replacements, and Stage-2 neural model + `Multi-LoRA` sidecar adapters!
    pub fn reload_assets(&mut self) {
        if let Some(dir) = &self.app_support_dir
            && let Ok(cfg) = ChimeConfig::load_or_init(&dir.join("config.toml"))
        {
            self.config = cfg;
        }

        let core_dict_path = self
            .bundle_resources_dir
            .as_ref()
            .map(|r| r.join("core.imedict"))
            .filter(|p| p.exists());
        let mut lex = if core_dict_path.is_some() {
            Lexicon::with_core_grammar()
        } else {
            Lexicon::with_builtin_seed()
        };
        if let Some(core_path) = core_dict_path {
            let _ = lex.mount_compiled_mmap(&core_path);
        }
        let mut lora_specs: Vec<LoraMountSpec> = Vec::new();
        self.installed_packs_count = 0;
        self.installed_packs_meta.clear();

        if let Some(dir) = &self.app_support_dir {
            let dicts_dir = dir.join("dicts");
            if let Ok(entries) = std::fs::read_dir(&dicts_dir) {
                let mut paths: Vec<PathBuf> = entries
                    .filter_map(Result::ok)
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("imedict"))
                    .collect();
                paths.sort();
                for path in paths {
                    let _ = lex.mount_compiled_mmap(&path);
                }
            }

            let packs = discover_installed_packs(dir);
            self.installed_packs_count = packs.len();
            for pack in packs {
                let eff_id = pack.manifest.effective_id();
                self.installed_packs_meta.push(InstalledPackSummary {
                    id: eff_id.clone(),
                    name: pack.manifest.name.clone(),
                    version: pack.manifest.version.clone(),
                    has_lora: pack.lora_path.is_some(),
                });
                if let Some(dp) = pack.dict_path {
                    let _ = lex.mount_compiled_mmap(&dp);
                }
                if let Some(lp) = pack.lora_path {
                    lora_specs.push(LoraMountSpec {
                        id: eff_id,
                        path: lp,
                        scale: pack.manifest.lora_scale,
                        routing: pack.manifest.routing,
                    });
                }
            }

            // Opportunistically import macOS System Text Replacements if readable.
            if let Some(home) = std::env::var_os("HOME") {
                let tr_db =
                    PathBuf::from(home).join("Library/KeyboardServices/TextReplacements.db");
                if tr_db.exists() {
                    let _ = import_macos_text_replacements(&tr_db, &mut lex);
                }
            }

            let user_neural_dir = dir.join("neural");
            let bundle_neural_dir = self.bundle_resources_dir.as_ref().map(|r| r.join("neural"));
            let chosen_neural = if user_neural_dir.join("model.safetensors").exists() {
                Some(user_neural_dir)
            } else {
                bundle_neural_dir.filter(|b| b.join("model.safetensors").exists())
            };
            if let Some(n_dir) = chosen_neural
                && let Ok(worker) = AsyncRerankHandle::spawn_from_dir_with_loras(&n_dir, lora_specs)
            {
                self.decoder
                    .set_reranker(Some(Box::new(MidpointPrefixAdapter {
                        worker: worker.clone(),
                    })));
                self.async_reranker = Some(worker);
            } else {
                self.decoder.set_reranker(None);
            }
        }
        self.lexicon = lex;
    }

    /// Updates the runtime configuration and persists it if `app_support_dir` is configured.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Config`] if saving `config.toml` fails.
    pub fn update_config(&mut self, config: ChimeConfig) -> Result<(), EngineError> {
        if let Some(dir) = &self.app_support_dir {
            config.save(&dir.join("config.toml"))?;
        }
        self.config = config;
        Ok(())
    }

    /// Borrows the current [`ChimeConfig`].
    #[must_use]
    pub const fn config(&self) -> &ChimeConfig {
        &self.config
    }

    /// Returns a live status snapshot for UI diagnostics.
    #[must_use]
    pub fn status(&self) -> EngineStatus {
        EngineStatus {
            scheme: self.config.scheme,
            pinyin_keys: self.lexicon.pinyin_key_count(),
            word_entries: self.lexicon.total_word_entries(),
            learned_habits: self.user_store.habit_count(),
            neural_loaded: self.async_reranker.is_some(),
            installed_packs: self.installed_packs_count,
            active_loras: self
                .async_reranker
                .as_ref()
                .map_or(0, AsyncRerankHandle::lora_count),
            pack_details: self.installed_packs_meta.clone(),
        }
    }

    /// Removes an installed `.imepack` cartridge directory by slug (`packs/<slug>.imepack`) and
    /// hot-reloads active lexicons and `LoRA` adapters.
    pub fn remove_installed_pack(&mut self, pack_id: &str) -> bool {
        let slug = pack_id.trim();
        if slug.is_empty()
            || !slug
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return false;
        }
        let Some(dir) = &self.app_support_dir else {
            return false;
        };
        let target = dir.join("packs").join(format!("{slug}.imepack"));
        if !target.exists() {
            return false;
        }
        if std::fs::remove_dir_all(&target).is_ok() {
            self.reload_assets();
            true
        } else {
            false
        }
    }

    /// Imports a user-selected file (`.scel`, `.bdict`, `.dict.yaml`, `.tsv`, `.imedict`,
    /// `.imepack` dir, or `.imepack.zip` archive), compiles/extracts it into `app_support_dir`, and
    /// hot-mounts immediately!
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] if parsing or writing the imported asset fails.
    pub fn import_external_asset(&mut self, path: &Path) -> Result<usize, EngineError> {
        let support_dir = self
            .app_support_dir
            .clone()
            .unwrap_or_else(std::env::temp_dir);
        let dicts_dir = support_dir.join("dicts");
        std::fs::create_dir_all(&dicts_dir)?;

        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if (path.is_dir() && path.join("manifest.json").exists()) || ext == "zip" {
            let manifest = install_imepack_dir(path, &support_dir)?;
            self.reload_assets();
            return Ok(usize::from(manifest.has_dictionary)
                + usize::from(manifest.has_lora)
                + usize::from(manifest.has_safetensors));
        }
        if ext == "imedict" {
            let stem = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("custom.imedict");
            let dst = dicts_dir.join(stem);
            if path != dst {
                std::fs::copy(path, &dst)?;
            }
            self.lexicon.mount_compiled_mmap(&dst)?;
            return Ok(1);
        }

        let mut imported_lex = Lexicon::new();
        let stats = if ext == "scel" {
            parse_scel_bytes(&std::fs::read(path)?, &mut imported_lex)?
        } else if ext == "bdict" {
            parse_bdict_bytes(&std::fs::read(path)?, &mut imported_lex)?
        } else {
            let text = std::fs::read_to_string(path)?;
            if path.to_string_lossy().ends_with(".dict.yaml") {
                parse_rime_dict_yaml(&text, &mut imported_lex)
            } else {
                parse_tsv_dict(&text, &mut imported_lex)
            }
        };

        if stats.imported_entries > 0 {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("imported")
                .trim_end_matches(".dict");
            let out_path = dicts_dir.join(format!("{stem}.imedict"));
            let builder = CompiledDictBuilder::from_lexicon(&imported_lex);
            builder.write_to_file(&out_path)?;
            self.lexicon.mount_compiled_mmap(&out_path)?;
        }
        Ok(stats.imported_entries)
    }

    /// Borrows the underlying [`Lexicon`] mutably.
    pub fn lexicon_mut(&mut self) -> &mut Lexicon {
        &mut self.lexicon
    }

    /// Borrows the underlying [`Lexicon`].
    #[must_use]
    pub const fn lexicon(&self) -> &Lexicon {
        &self.lexicon
    }

    /// Borrows the underlying [`UserStore`].
    #[must_use]
    pub const fn user_store(&self) -> &UserStore {
        &self.user_store
    }

    /// Clears all learned personal word habits and negative feedback overrides from RAM and
    /// `SQLite`.
    pub fn clear_user_habits(&mut self) {
        self.user_store.clear_all();
    }

    /// Allocates a lightweight `session_id`, evicting the oldest session if over
    /// `MAX_ACTIVE_SESSIONS`.
    pub fn create_session(&mut self, app_bundle_id: impl Into<String>) -> u64 {
        let id = self.next_session_id;
        self.next_session_id = self.next_session_id.saturating_add(1);
        if self.sessions.len() >= MAX_ACTIVE_SESSIONS {
            self.sessions.pop_front();
        }
        let mut sess = InputSession::new(id, app_bundle_id.into());
        sess.inline_english_mode = self.inline_english_mode;
        self.sessions.push_back(sess);
        id
    }

    /// Destroys a session when `deactivateServer` fires on `macOS`.
    pub fn destroy_session(&mut self, session_id: u64) {
        self.sessions.retain(|s| s.id != session_id);
    }

    /// Resets any active preedit buffer in `session_id`.
    pub fn reset_session(&mut self, session_id: u64) {
        if let Some(sess) = self.sessions.iter_mut().find(|s| s.id == session_id) {
            sess.clear_composing();
            sess.clear_en_tracking();
            sess.ghost_suggestion = None;
        }
    }

    /// Handles `IMKInputController.commitComposition(_:)`: clears active preedit ONLY when
    /// composing, preserving any post-commit `ghost_suggestion` (`⇥ 联想`) so `一路 Tab` works!
    pub fn commit_composition_if_active(&mut self, session_id: u64) -> EngineResponse {
        let Some(idx) = self.touch_session(session_id) else {
            return EngineResponse::default();
        };
        if self.sessions[idx].is_composing() {
            self.sessions[idx].clear_composing();
        }
        self.build_snapshot(idx, false, String::new())
    }

    /// Synchronizes the session's sliding context window (`left_context`, up to 128 chars) with
    /// real surrounding document text observed via `IMKTextInput.attributedSubstring` when idle!
    pub fn sync_surrounding_context(&mut self, session_id: u64, surrounding_left: &str) {
        if surrounding_left.is_empty() {
            return;
        }
        if let Some(sess) = self.sessions.iter_mut().find(|s| s.id == session_id)
            && !sess.is_composing()
        {
            let chars: Vec<char> = surrounding_left.chars().collect();
            let start = chars.len().saturating_sub(128);
            sess.left_context = chars[start..].iter().collect();
            sess.last_committed_char = sess.left_context.chars().last();
        }
    }

    /// Toggles Chinese full-width (`。，`) vs ASCII (`.,`) punctuation, persisting to
    /// `config.toml`.
    pub fn toggle_punctuation(&mut self, session_id: u64) -> EngineResponse {
        self.config.chinese_punctuation = !self.config.chinese_punctuation;
        if let Some(dir) = &self.app_support_dir {
            let _ = self.config.save(&dir.join("config.toml"));
        }
        let Some(idx) = self.touch_session(session_id) else {
            return EngineResponse {
                handled: true,
                chinese_punctuation: self.config.chinese_punctuation,
                candidate_layout: self.config.candidate_layout,
                candidate_font_size: self.config.candidate_font_size,
                show_status_toast: self.config.show_status_toast,
                ..EngineResponse::default()
            };
        };
        self.build_snapshot(idx, true, String::new())
    }

    /// Toggles automatic half-width Pangu spacing around English & numbers, persisting to
    /// `config.toml`.
    pub fn toggle_pangu_spacing(&mut self, session_id: u64) -> EngineResponse {
        self.config.pangu_spacing = !self.config.pangu_spacing;
        if let Some(dir) = &self.app_support_dir {
            let _ = self.config.save(&dir.join("config.toml"));
        }
        let Some(idx) = self.touch_session(session_id) else {
            return EngineResponse {
                handled: true,
                chinese_punctuation: self.config.chinese_punctuation,
                candidate_layout: self.config.candidate_layout,
                candidate_font_size: self.config.candidate_font_size,
                show_status_toast: self.config.show_status_toast,
                ..EngineResponse::default()
            };
        };
        self.build_snapshot(idx, true, String::new())
    }

    /// Toggles the Doubao-style `5x5` expanded candidate grid (`grid_expanded`).
    pub fn toggle_grid_expanded(&mut self, session_id: u64) -> EngineResponse {
        let Some(idx) = self.touch_session(session_id) else {
            return EngineResponse::default();
        };
        if !self.sessions[idx].all_candidates.is_empty() {
            self.sessions[idx].grid_expanded = !self.sessions[idx].grid_expanded;
        }
        self.build_snapshot(idx, true, String::new())
    }

    /// Toggles inline English mode on a clean modifier tap (or commits raw ASCII while staying in
    /// Chinese mode when `shift_commit_mode == CommitKeepChinese` during active preedit).
    pub fn toggle_inline_english(&mut self, session_id: u64) -> EngineResponse {
        let pangu = self.config.pangu_spacing;
        let shift_mode = self.config.shift_commit_mode;
        let Some(idx) = self.touch_session(session_id) else {
            self.inline_english_mode = !self.inline_english_mode;
            for s in &mut self.sessions {
                s.inline_english_mode = self.inline_english_mode;
            }
            return EngineResponse {
                handled: true,
                inline_english_mode: self.inline_english_mode,
                chinese_punctuation: self.config.chinese_punctuation,
                candidate_layout: self.config.candidate_layout,
                candidate_font_size: self.config.candidate_font_size,
                show_status_toast: self.config.show_status_toast,
                ..EngineResponse::default()
            };
        };
        let was_composing = self.sessions[idx].is_composing();
        if !(was_composing && shift_mode == ShiftCommitMode::CommitKeepChinese) {
            self.inline_english_mode = !self.inline_english_mode;
            for s in &mut self.sessions {
                s.inline_english_mode = self.inline_english_mode;
            }
        }
        let sess = &mut self.sessions[idx];
        sess.ghost_suggestion = None;
        sess.clear_en_tracking();
        let mut raw_commit = std::mem::take(&mut sess.partial_prefix);
        raw_commit.push_str(&std::mem::take(&mut sess.raw_buffer));
        sess.clear_composing();
        let committed = apply_boundary_pangu_spacing(&sess.left_context, &raw_commit, pangu);
        if !committed.is_empty() {
            sess.record_emitted_text(&committed);
        }
        let gen_id = sess.generation_id;
        EngineResponse {
            handled: true,
            committed_text: committed,
            inline_english_mode: self.inline_english_mode,
            chinese_punctuation: self.config.chinese_punctuation,
            candidate_layout: self.config.candidate_layout,
            candidate_font_size: self.config.candidate_font_size,
            show_status_toast: self.config.show_status_toast
                && !(was_composing && shift_mode == ShiftCommitMode::CommitKeepChinese),
            generation_id: gen_id,
            ..EngineResponse::default()
        }
    }

    /// Polls completed asynchronous Stage-2 neural reranks and applies the 60ms Anti-Jitter Freeze
    /// Window.
    pub fn poll_async_rerank(&mut self, session_id: u64) -> Option<EngineResponse> {
        let worker = self.async_reranker.as_ref()?;
        let results = worker.drain_results();
        if results.is_empty() {
            return None;
        }

        let mut updated_for_target = false;
        let freeze_ms = self.config.anti_jitter_ms;
        for res in results {
            let Some(sess) = self.sessions.iter_mut().find(|s| s.id == res.session_id) else {
                continue;
            };
            // Drop stale reranks from earlier keystrokes!
            if sess.generation_id != res.generation_id || sess.all_candidates.is_empty() {
                continue;
            }
            let displayed = sess.all_candidates.clone();
            let mut rescored = displayed.clone();
            let best_non_comp = displayed
                .iter()
                .find(|c| {
                    c.consumed_bytes == sess.raw_buffer.len() && c.kind != MatchKind::Completion
                })
                .map_or(f32::NEG_INFINITY, |c| c.score);
            for (cand, bonus) in rescored
                .iter_mut()
                .filter(|c| {
                    c.consumed_bytes == sess.raw_buffer.len()
                        && c.kind != MatchKind::Completion
                        && c.score >= best_non_comp - 5.4
                })
                .take(10)
                .zip(res.bonus_scores)
            {
                cand.score += bonus;
            }
            let (mut full_part, partial_part): (Vec<_>, Vec<_>) = rescored
                .into_iter()
                .partition(|c| c.consumed_bytes == sess.raw_buffer.len());
            full_part.sort_by(|a, b| b.score.total_cmp(&a.score));
            promote_top_completion_into_bar(&mut full_part);
            full_part.extend(partial_part);

            sess.all_candidates =
                apply_anti_jitter_freeze_window(&displayed, full_part, res.elapsed_ms, freeze_ms);
            if res.session_id == session_id {
                updated_for_target = true;
            }
        }

        if updated_for_target {
            let pos = self.sessions.iter().position(|s| s.id == session_id)?;
            Some(self.build_snapshot(pos, true, String::new()))
        } else {
            None
        }
    }

    /// Processes a single key event synchronously on the hot path.
    #[allow(clippy::too_many_lines)]
    pub fn process_key(
        &mut self,
        session_id: u64,
        event: KeyEvent,
        now_epoch_secs: u64,
    ) -> EngineResponse {
        // Opportunistically drain any finished Stage-2 rerank before handling the new key!
        let _ = self.poll_async_rerank(session_id);

        if event.secure_input || event.has_command_or_ctrl {
            self.reset_session(session_id);
            return EngineResponse::default();
        }

        let Some(idx) = self.touch_session(session_id) else {
            return EngineResponse::default();
        };

        self.sessions[idx].inline_english_mode = self.inline_english_mode;
        if self.inline_english_mode {
            return self.handle_inline_english_key(idx, event);
        }

        let ch = event.ascii_char;
        let is_composing = self.sessions[idx].is_composing();

        // Handle Copilot-style `一路 Tab` contextual ghost completion when idle (even if a comma
        // or space was typed right after the trigger clause, e.g. `月有悲欢离合，` -> `Tab` ->
        // `人有旦夕祸福`)!
        if !is_composing {
            if event.key_code == 0x30 && !event.shift && !self.sessions[idx].ghost_dismissed {
                let resolved_ghost = self.sessions[idx]
                    .ghost_suggestion
                    .take()
                    .or_else(|| self.compute_ghost_for_session(idx));
                if let Some(ghost) = resolved_ghost {
                    let prev_w = self.sessions[idx].last_committed_word.clone();
                    if !prev_w.is_empty() {
                        self.user_store
                            .record_bigram(&prev_w, &ghost, now_epoch_secs);
                    }
                    self.sessions[idx].last_committed_word.clone_from(&ghost);
                    self.sessions[idx].record_emitted_text(&ghost);
                    self.sessions[idx].ghost_suggestion = self.compute_ghost_for_session(idx);
                    return self.build_snapshot(idx, true, ghost);
                }
            }
            if event.key_code == 0x35 && self.sessions[idx].ghost_suggestion.is_some() {
                // Dismiss ghost suggestion on Escape without passing Esc to host app!
                self.sessions[idx].ghost_suggestion = None;
                self.sessions[idx].ghost_dismissed = true;
                return self.build_snapshot(idx, true, String::new());
            }
            // Idle Backspace (`0x33`): keep sliding `left_context` accurate in terminals AND
            // automatically revert any just-auto-committed Top-1 phrase that the user immediately
            // deleted!
            if event.key_code == 0x33 {
                self.sessions[idx].left_context.pop();
                self.sessions[idx].last_committed_char =
                    self.sessions[idx].left_context.chars().last();
                if let Some((prev_keys, prev_surf)) = self.sessions[idx].last_auto_committed.take()
                {
                    self.user_store.revert_commit(&prev_keys, &prev_surf);
                }
                self.sessions[idx].ghost_suggestion = None;
                return EngineResponse {
                    handled: false,
                    chinese_punctuation: self.config.chinese_punctuation,
                    candidate_layout: self.config.candidate_layout,
                    ..EngineResponse::default()
                };
            }
            // Idle Return (`0x24`): record newline boundary in sliding context window.
            if event.key_code == 0x24 {
                self.sessions[idx].record_emitted_text("\n");
                self.sessions[idx].last_auto_committed = None;
                self.sessions[idx].ghost_suggestion = None;
            } else {
                self.sessions[idx].ghost_suggestion = None;
                self.sessions[idx].ghost_dismissed = false;
            }
        }

        if is_composing
            && let Some(nav_resp) = self.handle_composing_control_key(idx, event, now_epoch_secs)
        {
            return nav_resp;
        }

        // Microsoft Shuangpin uses `;` as the `ing` final key when already composing or after an
        // initial!
        let is_ms_ing_semicolon = self.config.scheme == ShuangpinScheme::Microsoft
            && ch == b';'
            && !self.sessions[idx].raw_buffer.is_empty();

        let starts_composing = ch.is_ascii_alphabetic();
        let continues_composing = !self.sessions[idx].raw_buffer.is_empty()
            && (ch.is_ascii_alphanumeric()
                || is_ms_ing_semicolon
                || matches!(ch, b'\'' | b'_' | b'.' | b'-' | b'/'));

        if starts_composing || continues_composing {
            let c = char::from(ch);
            self.sessions[idx].raw_buffer.push(c);
            self.sessions[idx].full_sentence_keys.push(c);
            return self.recompute_session(idx, now_epoch_secs);
        }

        // Smart Chinese full-width punctuation (auto-commits active candidate if composing!).
        if self.config.chinese_punctuation && ch != 0 {
            if is_composing && PunctuationState::triggers_preedit_commit(ch) {
                let pick = self.sessions[idx].selected_global_idx;
                let mut resp = self.commit_candidate_global(idx, pick, now_epoch_secs);
                // If partial commit left remaining chars, flush top-1 of remainder too.
                if !self.sessions[idx].raw_buffer.is_empty() {
                    let tail = self.commit_candidate_global(idx, 0, now_epoch_secs);
                    resp.committed_text.push_str(&tail.committed_text);
                }
                let prev_ch = self.sessions[idx].last_committed_char;
                if let Some(mapped) = self.sessions[idx].punctuation.map_punctuation(ch, prev_ch) {
                    resp.committed_text.push_str(mapped);
                    self.sessions[idx].record_emitted_text(mapped);
                    self.sessions[idx].ghost_suggestion = self.compute_ghost_for_session(idx);
                    resp.ghost_text = self.sessions[idx]
                        .ghost_suggestion
                        .clone()
                        .unwrap_or_default();
                }
                resp.preedit_text.clear();
                resp.candidates.clear();
                return resp;
            } else if !is_composing {
                let prev_ch = self.sessions[idx].last_committed_char;
                if let Some(mapped) = self.sessions[idx].punctuation.map_punctuation(ch, prev_ch) {
                    self.sessions[idx].record_emitted_text(mapped);
                    let next_ghost = self.compute_ghost_for_session(idx);
                    self.sessions[idx].ghost_suggestion.clone_from(&next_ghost);
                    return EngineResponse {
                        handled: true,
                        committed_text: mapped.to_string(),
                        chinese_punctuation: self.config.chinese_punctuation,
                        ghost_text: next_ghost.unwrap_or_default(),
                        candidate_layout: self.config.candidate_layout,
                        ..EngineResponse::default()
                    };
                }
            }
        }

        // Symmetric Pangu auto-spacing before standalone ASCII digits typed right after CJK Hanzi
        // (`我有` + `100` + `个` -> `我有 100 个` when `pangu_spacing == true`, `我有100个` when
        // false).
        if !is_composing
            && self.config.pangu_spacing
            && ch.is_ascii_digit()
            && self.sessions[idx]
                .left_context
                .chars()
                .last()
                .is_some_and(is_cjk_hanzi)
        {
            let spaced = format!(" {}", char::from(ch));
            self.sessions[idx].record_emitted_text(&spaced);
            return EngineResponse {
                handled: true,
                committed_text: spaced,
                chinese_punctuation: self.config.chinese_punctuation,
                candidate_layout: self.config.candidate_layout,
                candidate_font_size: self.config.candidate_font_size,
                show_status_toast: self.config.show_status_toast,
                ..EngineResponse::default()
            };
        }

        if !is_composing && (ch.is_ascii_graphic() || ch == b' ') {
            let s = char::from(ch).to_string();
            self.sessions[idx].record_emitted_text(&s);
        }

        EngineResponse {
            chinese_punctuation: self.config.chinese_punctuation,
            candidate_layout: self.config.candidate_layout,
            candidate_font_size: self.config.candidate_font_size,
            show_status_toast: self.config.show_status_toast,
            ..EngineResponse::default()
        }
    }

    /// Selects candidate `page_relative_index` (`0..eff_page_size`) on the active page (e.g. mouse
    /// click).
    pub fn select_candidate(
        &mut self,
        session_id: u64,
        page_relative_index: usize,
        now_epoch_secs: u64,
    ) -> EngineResponse {
        let Some(idx) = self.touch_session(session_id) else {
            return EngineResponse::default();
        };
        if !self.sessions[idx].is_composing()
            && let Some(ghost) = self.sessions[idx].ghost_suggestion.take()
        {
            self.sessions[idx].record_emitted_text(&ghost);
            self.sessions[idx].ghost_suggestion = self.compute_ghost_for_session(idx);
            return self.build_snapshot(idx, true, ghost);
        }
        let page_size = self.effective_page_size(idx);
        let page_start = (self.sessions[idx].selected_global_idx / page_size) * page_size;
        let global_idx = page_start + page_relative_index;
        self.commit_candidate_global(idx, global_idx, now_epoch_secs)
    }

    /// One-shot stateless decode helper used by CLI and offline benchmark harnesses.
    #[must_use]
    pub fn decode_stateless(
        &self,
        raw_input: &str,
        left_context: &str,
        app_bundle_id: &str,
    ) -> Vec<Candidate> {
        let ctx = DecodeContext {
            left_context,
            app_bundle_id,
            scheme: self.config.scheme,
            fuzzy: self.config.fuzzy,
            pangu_spacing: self.config.pangu_spacing,
            uppercase_triggers_english: self.config.uppercase_triggers_english,
            english_auto_correct: self.config.english_auto_correct,
            now_epoch_secs: 0,
        };
        self.decoder
            .decode(raw_input, &self.lexicon, &self.user_store, &ctx)
    }

    #[allow(clippy::too_many_lines)]
    fn handle_composing_control_key(
        &mut self,
        idx: usize,
        event: KeyEvent,
        now_epoch_secs: u64,
    ) -> Option<EngineResponse> {
        let ch = event.ascii_char;
        let total = self.sessions[idx].all_candidates.len();
        let page_size = self.effective_page_size(idx);
        let grid = self.sessions[idx].grid_expanded;

        match event.key_code {
            // Space -> commit highlighted candidate
            0x31 => {
                let sel = self.sessions[idx].selected_global_idx;
                return Some(self.commit_candidate_global(idx, sel, now_epoch_secs));
            }
            // Return -> commit raw ASCII directly (with boundary Pangu spacing if applicable)
            0x24 => {
                let mut raw_out = std::mem::take(&mut self.sessions[idx].partial_prefix);
                raw_out.push_str(&std::mem::take(&mut self.sessions[idx].raw_buffer));
                self.sessions[idx].clear_composing();
                let out = apply_boundary_pangu_spacing(
                    &self.sessions[idx].left_context,
                    &raw_out,
                    self.config.pangu_spacing,
                );
                self.sessions[idx].record_emitted_text(&out);
                return Some(EngineResponse {
                    handled: true,
                    committed_text: out,
                    chinese_punctuation: self.config.chinese_punctuation,
                    candidate_layout: self.config.candidate_layout,
                    ..EngineResponse::default()
                });
            }
            // Backspace -> undo partial prefix if raw_buffer is empty, else pop 1 raw char
            0x33 => {
                if self.sessions[idx].raw_buffer.is_empty() {
                    self.sessions[idx].partial_prefix.pop();
                } else {
                    self.sessions[idx].raw_buffer.pop();
                    self.sessions[idx].full_sentence_keys.pop();
                }
                return Some(self.recompute_session(idx, now_epoch_secs));
            }
            // Escape -> collapse 5x5 grid first if expanded, otherwise cancel composition
            0x35 => {
                if grid {
                    self.sessions[idx].grid_expanded = false;
                    return Some(self.build_snapshot(idx, true, String::new()));
                }
                self.sessions[idx].clear_composing();
                return Some(EngineResponse {
                    handled: true,
                    chinese_punctuation: self.config.chinese_punctuation,
                    candidate_layout: self.config.candidate_layout,
                    ..EngineResponse::default()
                });
            }
            // Tab ->
            // 1) Accept prefix verse/idiom/English Completion (`tianshengwocai` ->
            //    `天生我材必有用`) immediately;
            // 2) Or if highlighted candidate itself triggers a couplet/habit continuation
            //    (`yueyoubeihuanlihe` -> `月有悲欢离合` -> `，人有旦夕祸福`), commit & chain on
            //    Tab!
            // 3) Otherwise Typo Rollback toggle or cycle candidate!
            0x30 if total >= 1 => {
                if !event.shift {
                    if let Some(comp_idx) = self.sessions[idx]
                        .all_candidates
                        .iter()
                        .position(|c| c.kind == MatchKind::Completion)
                    {
                        return Some(self.commit_candidate_global(idx, comp_idx, now_epoch_secs));
                    }
                    let cur = self.sessions[idx].selected_global_idx.min(total - 1);
                    let cur_text = &self.sessions[idx].all_candidates[cur].text;
                    if self.lexicon.next_clause_completion(cur_text).is_some()
                        || self.user_store.predict_next_phrase(cur_text).is_some()
                    {
                        return Some(self.commit_candidate_global(idx, cur, now_epoch_secs));
                    }
                }
                if total > 1 {
                    let cur = self.sessions[idx].selected_global_idx;
                    if let Some(prev) = self.sessions[idx].typo_rollback_prev_idx.take() {
                        self.sessions[idx].selected_global_idx = prev.min(total - 1);
                    } else if self.sessions[idx].all_candidates[cur].kind
                        == MatchKind::TypoCorrected
                    {
                        if let Some(exact_idx) =
                            self.sessions[idx].all_candidates.iter().position(|c| {
                                matches!(c.kind, MatchKind::Exact | MatchKind::CodeSwitch)
                            })
                        {
                            self.sessions[idx].typo_rollback_prev_idx = Some(cur);
                            self.sessions[idx].selected_global_idx = exact_idx;
                        } else {
                            self.sessions[idx].selected_global_idx = (cur + 1) % total;
                        }
                    } else {
                        self.sessions[idx].selected_global_idx = (cur + 1) % total;
                    }
                    return Some(self.build_snapshot(idx, true, String::new()));
                }
            }
            // Down Arrow (0x7D):
            // - In compact 1x5 bar: expands into Doubao-style 5x5 Grid when > page_size candidates!
            // - In 5x5 Grid: moves down one row (+5)!
            0x7D if total > 0 => {
                if !grid && total > page_size {
                    self.sessions[idx].grid_expanded = true;
                } else if grid {
                    let next = self.sessions[idx].selected_global_idx + 5;
                    self.sessions[idx].selected_global_idx =
                        if next < total { next } else { total - 1 };
                } else {
                    self.sessions[idx].selected_global_idx =
                        (self.sessions[idx].selected_global_idx + 1) % total;
                }
                return Some(self.build_snapshot(idx, true, String::new()));
            }
            // Up Arrow (0x7E):
            // - In 5x5 Grid: moves up one row (-5), or collapses back to 1x5 bar from row 0!
            // - In compact 1x5 bar: previous candidate.
            0x7E if total > 0 => {
                let cur = self.sessions[idx].selected_global_idx;
                if grid {
                    if cur < 5 {
                        self.sessions[idx].grid_expanded = false;
                    } else {
                        self.sessions[idx].selected_global_idx = cur - 5;
                    }
                } else {
                    self.sessions[idx].selected_global_idx =
                        if cur == 0 { total - 1 } else { cur - 1 };
                }
                return Some(self.build_snapshot(idx, true, String::new()));
            }
            // Right Arrow (0x7C) -> next candidate
            0x7C if total > 0 => {
                self.sessions[idx].selected_global_idx =
                    (self.sessions[idx].selected_global_idx + 1) % total;
                return Some(self.build_snapshot(idx, true, String::new()));
            }
            // Left Arrow (0x7B) -> previous candidate
            0x7B if total > 0 => {
                let cur = self.sessions[idx].selected_global_idx;
                self.sessions[idx].selected_global_idx = if cur == 0 { total - 1 } else { cur - 1 };
                return Some(self.build_snapshot(idx, true, String::new()));
            }
            // PageDown (0x79) -> next page
            0x79 if total > page_size => {
                let next_page =
                    ((self.sessions[idx].selected_global_idx / page_size) + 1) * page_size;
                self.sessions[idx].selected_global_idx =
                    if next_page < total { next_page } else { 0 };
                return Some(self.build_snapshot(idx, true, String::new()));
            }
            // PageUp (0x74) -> prev page
            0x74 if total > page_size => {
                let cur_page = self.sessions[idx].selected_global_idx / page_size;
                self.sessions[idx].selected_global_idx = cur_page.saturating_sub(1) * page_size;
                return Some(self.build_snapshot(idx, true, String::new()));
            }
            _ => {}
        }

        // Configurable candidate page-turning shortcuts (`-`/`=`, `[`/`]`, `,`/`.`).
        let is_next_page_key = (self.config.minus_equal_paging && ch == b'=')
            || (self.config.bracket_paging && ch == b']')
            || (self.config.comma_period_paging && ch == b'.');
        let is_prev_page_key = (self.config.minus_equal_paging && ch == b'-')
            || (self.config.bracket_paging && ch == b'[')
            || (self.config.comma_period_paging && ch == b',');
        if is_next_page_key && total > page_size {
            let next_page = ((self.sessions[idx].selected_global_idx / page_size) + 1) * page_size;
            self.sessions[idx].selected_global_idx = if next_page < total { next_page } else { 0 };
            return Some(self.build_snapshot(idx, true, String::new()));
        }
        if is_prev_page_key && total > page_size {
            let cur_page = self.sessions[idx].selected_global_idx / page_size;
            self.sessions[idx].selected_global_idx = cur_page.saturating_sub(1) * page_size;
            return Some(self.build_snapshot(idx, true, String::new()));
        }

        // Optional `;` / `'` quick selection of 2nd and 3rd candidates (Sogou/Doubao/Rime style).
        if self.config.semicolon_apostrophe_select && !grid {
            let page_start = (self.sessions[idx].selected_global_idx / page_size) * page_size;
            if ch == b';'
                && self.config.scheme != ShuangpinScheme::Microsoft
                && page_size >= 2
                && page_start + 1 < total
            {
                return Some(self.commit_candidate_global(idx, page_start + 1, now_epoch_secs));
            }
            if ch == b'\'' && page_size >= 3 && page_start + 2 < total {
                return Some(self.commit_candidate_global(idx, page_start + 2, now_epoch_secs));
            }
        }

        // Digit selection:
        // - In 5x5 grid: `1..=5` selects column `0..5` on the currently highlighted row!
        // - In compact bar: `1..=bar_size` selects on the current bar page!
        if (b'1'..=b'9').contains(&ch) {
            let rel = usize::from(ch - b'1');
            if grid {
                if rel < 5 {
                    let row_start = (self.sessions[idx].selected_global_idx / 5) * 5;
                    let global_idx = row_start + rel;
                    if global_idx < total {
                        return Some(self.commit_candidate_global(idx, global_idx, now_epoch_secs));
                    }
                }
            } else if rel < page_size {
                let page_start = (self.sessions[idx].selected_global_idx / page_size) * page_size;
                let global_idx = page_start + rel;
                if global_idx < total {
                    return Some(self.commit_candidate_global(idx, global_idx, now_epoch_secs));
                }
            }
        }

        // Period `.` when composing a pure Chinese phrase commits candidate #1 + `。`!
        if self.config.chinese_punctuation
            && ch == b'.'
            && !self.sessions[idx]
                .raw_buffer
                .bytes()
                .any(|b| b.is_ascii_digit())
            && self.sessions[idx]
                .all_candidates
                .first()
                .is_some_and(|c| c.kind != MatchKind::CodeSwitch)
        {
            let sel = self.sessions[idx].selected_global_idx;
            let mut resp = self.commit_candidate_global(idx, sel, now_epoch_secs);
            if !self.sessions[idx].raw_buffer.is_empty() {
                let tail = self.commit_candidate_global(idx, 0, now_epoch_secs);
                resp.committed_text.push_str(&tail.committed_text);
            }
            resp.committed_text.push('。');
            self.sessions[idx].record_emitted_text("。");
            return Some(resp);
        }

        None
    }

    fn recompute_session(&mut self, idx: usize, now_epoch_secs: u64) -> EngineResponse {
        self.sessions[idx].generation_id = self.sessions[idx].generation_id.saturating_add(1);
        self.sessions[idx].selected_global_idx = 0;
        self.sessions[idx].grid_expanded = false;
        self.sessions[idx].ghost_suggestion = None;
        self.sessions[idx].ghost_dismissed = false;
        self.sessions[idx].typo_rollback_prev_idx = None;

        if self.sessions[idx].raw_buffer.is_empty() {
            if !self.sessions[idx].partial_prefix.is_empty() {
                let raw_committed = std::mem::take(&mut self.sessions[idx].partial_prefix);
                self.sessions[idx].clear_composing();
                let committed = apply_boundary_pangu_spacing(
                    &self.sessions[idx].left_context,
                    &raw_committed,
                    self.config.pangu_spacing,
                );
                self.sessions[idx].record_emitted_text(&committed);
                self.sessions[idx].ghost_suggestion = self.compute_ghost_for_session(idx);
                return self.build_snapshot(idx, true, committed);
            }
            self.sessions[idx].all_candidates.clear();
            return self.build_snapshot(idx, true, String::new());
        }

        let raw_clone = self.sessions[idx].raw_buffer.clone();
        let mut effective_left = self.sessions[idx].left_context.clone();
        effective_left.push_str(&self.sessions[idx].partial_prefix);
        let app_clone = self.sessions[idx].app_bundle_id.clone();

        let ctx = DecodeContext {
            left_context: &effective_left,
            app_bundle_id: &app_clone,
            scheme: self.config.scheme,
            fuzzy: self.config.fuzzy,
            pangu_spacing: self.config.pangu_spacing,
            uppercase_triggers_english: self.config.uppercase_triggers_english,
            english_auto_correct: self.config.english_auto_correct,
            now_epoch_secs,
        };
        let mut cands = self
            .decoder
            .decode(&raw_clone, &self.lexicon, &self.user_store, &ctx);

        if let Some(worker) = &self.async_reranker {
            cands = Self::rerank_candidates_with_budget(
                worker,
                self.sessions[idx].id,
                self.sessions[idx].generation_id,
                app_clone,
                effective_left,
                &raw_clone,
                cands,
            );
        }
        self.sessions[idx].all_candidates = cands;

        self.build_snapshot(idx, true, String::new())
    }

    #[allow(clippy::too_many_arguments)]
    fn rerank_candidates_with_budget(
        worker: &AsyncRerankHandle,
        session_id: u64,
        generation_id: u64,
        app_bundle_id: String,
        effective_left: String,
        raw_clone: &str,
        mut cands: Vec<Candidate>,
    ) -> Vec<Candidate> {
        let best_non_comp = cands
            .iter()
            .find(|c| c.consumed_bytes == raw_clone.len() && c.kind != MatchKind::Completion)
            .map_or(f32::NEG_INFINITY, |c| c.score);
        let full_texts: Vec<String> = cands
            .iter()
            .filter(|c| {
                c.consumed_bytes == raw_clone.len()
                    && c.kind != MatchKind::Completion
                    && c.score >= best_non_comp - 5.4
            })
            .take(10)
            .map(|c| c.text.clone())
            .collect();
        if full_texts.len() < 2 {
            return cands;
        }
        let has_fem_cue = effective_left.chars().any(|c| {
            matches!(
                c,
                '她' | '女'
                    | '妈'
                    | '姐'
                    | '妹'
                    | '姑'
                    | '娘'
                    | '妻'
                    | '婆'
                    | '姨'
                    | '嫂'
                    | '妇'
                    | '母'
                    | '护'
                    | '闺'
                    | '它'
                    | '猫'
                    | '狗'
            )
        });
        let job = RerankJob {
            session_id,
            generation_id,
            app_bundle_id,
            left_context: effective_left,
            candidate_texts: full_texts,
            dispatched_at: Instant::now(),
        };
        let Some(bonuses) = worker.submit_with_budget(job, Duration::from_millis(12)) else {
            return cands;
        };
        let mut env_after = f32::NEG_INFINITY;
        for (cand, bonus) in cands
            .iter_mut()
            .filter(|c| {
                c.consumed_bytes == raw_clone.len()
                    && c.kind != MatchKind::Completion
                    && c.score >= best_non_comp - 5.4
            })
            .take(10)
            .zip(bonuses)
        {
            cand.score += bonus;
            env_after = env_after.max(cand.score);
        }
        let drift = env_after - best_non_comp;
        if drift.is_finite() {
            for cand in &mut cands {
                if cand.consumed_bytes == raw_clone.len() && cand.kind != MatchKind::Completion {
                    cand.score -= drift;
                }
            }
        }
        let (mut full_part, partial_part): (Vec<_>, Vec<_>) = cands
            .into_iter()
            .partition(|c| c.consumed_bytes == raw_clone.len());
        full_part.sort_by(|a, b| b.score.total_cmp(&a.score));
        if !has_fem_cue {
            chime_decoder::apply_unmarked_pronoun_symmetry(&mut full_part, raw_clone.len());
        }
        promote_top_completion_into_bar(&mut full_part);
        full_part.extend(partial_part);
        full_part
    }

    fn commit_candidate_global(
        &mut self,
        idx: usize,
        global_idx: usize,
        now_epoch_secs: u64,
    ) -> EngineResponse {
        let Some(chosen) = self.sessions[idx].all_candidates.get(global_idx).cloned() else {
            return EngineResponse::default();
        };

        // Step-by-step partial prefix selection (`consumed_bytes < raw_buffer.len()`)!
        let consume = chosen
            .consumed_bytes
            .min(self.sessions[idx].raw_buffer.len());
        if consume > 0 && consume < self.sessions[idx].raw_buffer.len() {
            let prefix_keys = self.sessions[idx].raw_buffer[..consume].to_ascii_lowercase();
            let app_id = self.sessions[idx].app_bundle_id.clone();
            self.user_store
                .record_commit(&prefix_keys, &chosen.text, &app_id, now_epoch_secs);
            self.sessions[idx].partial_prefix.push_str(&chosen.text);
            self.sessions[idx].raw_buffer.drain(..consume);
            return self.recompute_session(idx, now_epoch_secs);
        }

        // Full completion of the sentence!
        let tail_reading = std::mem::take(&mut self.sessions[idx].raw_buffer);
        let full_keys = std::mem::take(&mut self.sessions[idx].full_sentence_keys);
        let mut final_text = std::mem::take(&mut self.sessions[idx].partial_prefix);
        final_text.push_str(&chosen.text);
        let app_id = self.sessions[idx].app_bundle_id.clone();

        if global_idx > 0
            && let Some(top_rejected) = self.sessions[idx].all_candidates.first()
            && top_rejected.consumed_bytes == chosen.consumed_bytes
        {
            self.user_store.record_negative_feedback(
                &tail_reading,
                &top_rejected.text,
                &chosen.text,
            );
        } else if let Some((prev_reading, prev_word)) = &self.sessions[idx].last_auto_committed
            && *prev_reading == full_keys
            && *prev_word != final_text
        {
            self.user_store
                .record_negative_feedback(&full_keys, prev_word, &final_text);
        }

        let learn_key = if full_keys.is_empty() {
            &tail_reading
        } else {
            &full_keys
        };
        self.user_store
            .record_commit(learn_key, &final_text, &app_id, now_epoch_secs);

        // Online contextual bigram learning (`prev_word -> final_text`)!
        let prev_word = self.sessions[idx].last_committed_word.clone();
        if !prev_word.is_empty() {
            self.user_store
                .record_bigram(&prev_word, &final_text, now_epoch_secs);
        }
        self.sessions[idx]
            .last_committed_word
            .clone_from(&final_text);

        if self.config.record_training_samples {
            self.user_store.record_typing_sample(
                &TypingSample {
                    left_context: self.sessions[idx].left_context.clone(),
                    raw_keys: learn_key.clone(),
                    committed_surface: final_text.clone(),
                    app_bundle_id: app_id,
                },
                now_epoch_secs,
            );
        }

        let spaced_output = apply_boundary_pangu_spacing(
            &self.sessions[idx].left_context,
            &final_text,
            self.config.pangu_spacing,
        );
        self.sessions[idx].last_auto_committed = Some((learn_key.clone(), final_text));
        self.sessions[idx].record_emitted_text(&spaced_output);
        self.sessions[idx].clear_composing();
        self.sessions[idx].ghost_suggestion = self.compute_ghost_for_session(idx);

        self.build_snapshot(idx, true, spaced_output)
    }

    fn effective_page_size(&self, idx: usize) -> usize {
        if self.sessions[idx].grid_expanded {
            25
        } else {
            self.config.page_size.max(1)
        }
    }

    fn compute_ghost_for_session(&self, idx: usize) -> Option<String> {
        let left = &self.sessions[idx].left_context;
        if let Some(couplet_next) = self.lexicon.next_clause_completion(left) {
            return Some(couplet_next);
        }
        self.user_store.predict_next_phrase(left)
    }

    fn build_snapshot(&self, idx: usize, handled: bool, committed_text: String) -> EngineResponse {
        let sess = &self.sessions[idx];
        let page_size = self.effective_page_size(idx);
        let total = sess.all_candidates.len();
        let total_pages = if total == 0 {
            0
        } else {
            total.div_ceil(page_size)
        };
        let sel = sess.selected_global_idx.min(total.saturating_sub(1));
        let page_index = if total == 0 { 0 } else { sel / page_size };
        let page_start = page_index * page_size;
        let page_end = (page_start + page_size).min(total);
        let page_slice = if total == 0 {
            Vec::new()
        } else {
            sess.all_candidates[page_start..page_end].to_vec()
        };
        let highlighted_index = sel.saturating_sub(page_start);

        let segmented_raw = format_segmented_preedit(&sess.raw_buffer, self.config.scheme);
        let preedit_text = if sess.partial_prefix.is_empty() {
            segmented_raw
        } else if segmented_raw.is_empty() {
            sess.partial_prefix.clone()
        } else {
            format!("{} {}", sess.partial_prefix, segmented_raw)
        };

        EngineResponse {
            handled,
            committed_text,
            preedit_text,
            candidates: page_slice,
            highlighted_index,
            page_index,
            total_pages,
            inline_english_mode: sess.inline_english_mode,
            chinese_punctuation: self.config.chinese_punctuation,
            grid_expanded: sess.grid_expanded,
            ghost_text: sess.ghost_suggestion.clone().unwrap_or_default(),
            candidate_layout: self.config.candidate_layout,
            candidate_font_size: self.config.candidate_font_size,
            show_status_toast: self.config.show_status_toast,
            replace_prev_chars: 0,
            toast_override: String::new(),
            generation_id: sess.generation_id,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn handle_inline_english_key(&mut self, idx: usize, event: KeyEvent) -> EngineResponse {
        let ch = event.ascii_char;
        let base_unhandled = EngineResponse {
            handled: false,
            inline_english_mode: true,
            chinese_punctuation: self.config.chinese_punctuation,
            candidate_layout: self.config.candidate_layout,
            candidate_font_size: self.config.candidate_font_size,
            show_status_toast: self.config.show_status_toast,
            ..EngineResponse::default()
        };

        // 1. Single-key `Backspace` (`0x33`) instant undo of the just-applied auto-correction!
        if event.key_code == 0x33 {
            if let Some(undo) = self.sessions[idx].en_last_autocorrect.take() {
                let lower_orig = undo.original_word.to_ascii_lowercase();
                if !self.sessions[idx].en_suppressed_words.contains(&lower_orig) {
                    if self.sessions[idx].en_suppressed_words.len() >= 64 {
                        self.sessions[idx].en_suppressed_words.remove(0);
                    }
                    self.sessions[idx].en_suppressed_words.push(lower_orig);
                }
                self.sessions[idx].pop_left_context_chars(undo.emitted_chars);
                self.sessions[idx].record_emitted_text(&undo.original_word);
                self.sessions[idx]
                    .en_word_buffer
                    .clone_from(&undo.original_word);
                self.sessions[idx].en_token_tainted = false;
                return EngineResponse {
                    handled: true,
                    committed_text: undo.original_word.clone(),
                    replace_prev_chars: undo.emitted_chars,
                    toast_override: format!("已撤回 → {}", undo.original_word),
                    ..base_unhandled
                };
            }
            self.sessions[idx].en_word_buffer.pop();
            if self.sessions[idx].en_word_buffer.is_empty() {
                self.sessions[idx].en_token_tainted = false;
            }
            self.sessions[idx].pop_left_context_chars(1);
            return base_unhandled;
        }

        // 2. Navigation / special keys clear token tracking.
        if ch == 0 {
            self.sessions[idx].clear_en_tracking();
            if event.key_code == 0x24 {
                self.sessions[idx].record_emitted_text("\n");
            }
            return base_unhandled;
        }

        // 3. Alphabetic characters accumulate silently into `en_word_buffer` (`handled = false`).
        if ch.is_ascii_alphabetic() {
            self.sessions[idx].en_last_autocorrect = None;
            let prev_ch = self.sessions[idx].left_context.chars().last();
            if self.sessions[idx].en_word_buffer.is_empty()
                && !self.sessions[idx].en_token_tainted
                && prev_ch.is_some_and(|c| {
                    c.is_ascii_digit()
                        || matches!(
                            c,
                            '/' | '.' | '@' | ':' | '_' | '-' | '~' | '$' | '#' | '\\'
                        )
                })
            {
                self.sessions[idx].en_token_tainted = true;
            }
            if !self.sessions[idx].en_token_tainted {
                if self.sessions[idx].en_word_buffer.len() < 24 {
                    self.sessions[idx].en_word_buffer.push(char::from(ch));
                } else {
                    self.sessions[idx].en_word_buffer.clear();
                    self.sessions[idx].en_token_tainted = true;
                }
            }
            if self.config.pangu_spacing && prev_ch.is_some_and(is_cjk_hanzi) {
                let spaced = format!(" {}", char::from(ch));
                self.sessions[idx].record_emitted_text(&spaced);
                return EngineResponse {
                    handled: true,
                    committed_text: spaced,
                    ..base_unhandled
                };
            }
            self.sessions[idx].record_emitted_text(&char::from(ch).to_string());
            return base_unhandled;
        }

        // 4. Word-boundary delimiters (`Space` or sentence punctuation) trigger spelling check.
        if matches!(ch, b' ' | b',' | b'.' | b'!' | b'?' | b';' | b':') {
            self.sessions[idx].en_last_autocorrect = None;
            let typed_word = std::mem::take(&mut self.sessions[idx].en_word_buffer);
            let was_tainted = std::mem::replace(&mut self.sessions[idx].en_token_tainted, false);
            if !was_tainted
                && let Some(auto_resp) =
                    self.try_autocorrect_inline_english(idx, &typed_word, char::from(ch))
            {
                return auto_resp;
            }
            self.sessions[idx].record_emitted_text(&char::from(ch).to_string());
            return base_unhandled;
        }

        // 5. Digits, underscores, slashes, hyphens, apostrophes (`don't`) taint the current token.
        self.sessions[idx].en_last_autocorrect = None;
        self.sessions[idx].en_word_buffer.clear();
        self.sessions[idx].en_token_tainted = true;
        if self.config.pangu_spacing
            && ch.is_ascii_digit()
            && self.sessions[idx]
                .left_context
                .chars()
                .last()
                .is_some_and(is_cjk_hanzi)
        {
            let spaced = format!(" {}", char::from(ch));
            self.sessions[idx].record_emitted_text(&spaced);
            return EngineResponse {
                handled: true,
                committed_text: spaced,
                ..base_unhandled
            };
        }
        if ch.is_ascii_graphic() {
            self.sessions[idx].record_emitted_text(&char::from(ch).to_string());
        }
        base_unhandled
    }

    fn try_autocorrect_inline_english(
        &mut self,
        idx: usize,
        typed_word: &str,
        delim: char,
    ) -> Option<EngineResponse> {
        if !self.config.english_auto_correct || !(3..=18).contains(&typed_word.len()) {
            return None;
        }
        let app_id = &self.sessions[idx].app_bundle_id;
        if is_terminal_app(app_id) {
            return None;
        }
        let is_ide = UserStore::is_developer_app(app_id);
        if is_ide && typed_word.len() < 5 {
            return None;
        }
        // Exempt camelCase (`useEffect`, `iPhone`) and ALLCAPS (`HTTP`, `NASA`, `PR`).
        if typed_word[1..].bytes().any(|b| b.is_ascii_uppercase())
            || (typed_word.len() >= 2 && typed_word.bytes().all(|b| b.is_ascii_uppercase()))
        {
            return None;
        }

        let lower = typed_word.to_ascii_lowercase();
        if self.sessions[idx].en_suppressed_words.contains(&lower)
            || self.lexicon.is_known_valid_english(&lower)
        {
            return None;
        }
        if self
            .lexicon
            .complete_english_prefix(&lower, 1)
            .into_iter()
            .any(|c| c.is_tech_term && c.surface.len() <= lower.len() + 2)
        {
            return None;
        }

        let (best, _, edit_kind) = self
            .lexicon
            .correct_english_word(&lower, 1)
            .into_iter()
            .next()?;
        if is_ide
            && (best.log_prob < -3.90
                || (edit_kind == EnglishEditKind::NeighborOrVowelSub && typed_word.len() < 6))
        {
            return None;
        }

        let best_lower = best.surface.to_ascii_lowercase();
        let corrected = if delim == '.' && typed_word == lower {
            best_lower
        } else if best.surface != best_lower {
            best.surface
        } else if typed_word.as_bytes()[0].is_ascii_uppercase() {
            let mut chars = best.surface.chars();
            let first = chars.next()?.to_ascii_uppercase();
            format!("{first}{}", chars.as_str())
        } else {
            best.surface
        };
        if corrected == typed_word {
            return None;
        }

        let replace_len = typed_word.chars().count();
        self.sessions[idx].pop_left_context_chars(replace_len);
        let committed = format!("{corrected}{delim}");
        let emitted_chars = committed.chars().count();
        self.sessions[idx].record_emitted_text(&committed);
        self.sessions[idx].en_last_autocorrect = Some(EnAutoCorrectUndo {
            original_word: typed_word.to_string(),
            emitted_chars,
        });

        Some(EngineResponse {
            handled: true,
            committed_text: committed,
            inline_english_mode: true,
            chinese_punctuation: self.config.chinese_punctuation,
            candidate_layout: self.config.candidate_layout,
            candidate_font_size: self.config.candidate_font_size,
            show_status_toast: self.config.show_status_toast,
            replace_prev_chars: replace_len,
            toast_override: format!("{typed_word} → {corrected}"),
            generation_id: self.sessions[idx].generation_id,
            ..EngineResponse::default()
        })
    }

    fn touch_session(&mut self, session_id: u64) -> Option<usize> {
        let pos = self.sessions.iter().position(|s| s.id == session_id)?;
        if pos + 1 != self.sessions.len()
            && let Some(sess) = self.sessions.remove(pos)
        {
            self.sessions.push_back(sess);
            return Some(self.sessions.len() - 1);
        }
        Some(pos)
    }
}

fn is_cjk_hanzi(c: char) -> bool {
    ('\u{4E00}'..='\u{9FA5}').contains(&c)
}

fn is_terminal_app(app_bundle_id: &str) -> bool {
    let lower = app_bundle_id.to_ascii_lowercase();
    [
        "ghostty",
        "terminal",
        "iterm",
        "warp",
        "alacritty",
        "kitty",
        "wezterm",
        "hyper",
    ]
    .iter()
    .any(|pat| lower.contains(pat))
}

#[derive(Debug)]
struct MidpointPrefixAdapter {
    worker: AsyncRerankHandle,
}

impl chime_decoder::NeuralReranker for MidpointPrefixAdapter {
    fn score_batch(
        &self,
        _left_context: &str,
        _app_bundle_id: &str,
        candidates: &[Candidate],
    ) -> Vec<f32> {
        vec![0.0; candidates.len()]
    }

    fn score_open_prefix_batch(
        &self,
        left_context: &str,
        app_bundle_id: &str,
        candidates: &[Candidate],
    ) -> Vec<f32> {
        let texts: Vec<String> = candidates.iter().map(|c| c.text.clone()).collect();
        self.worker.score_sync_mode_with_timeout(
            left_context,
            app_bundle_id,
            &texts,
            true,
            Some(Duration::from_millis(5)),
        )
    }
}

fn apply_boundary_pangu_spacing(left_context: &str, committed: &str, enabled: bool) -> String {
    if !enabled || left_context.is_empty() || committed.is_empty() {
        return committed.to_string();
    }
    let Some(prev_ch) = left_context.chars().last() else {
        return committed.to_string();
    };
    let Some(first_ch) = committed.chars().next() else {
        return committed.to_string();
    };
    let needs_space = (is_cjk_hanzi(prev_ch) && first_ch.is_ascii_alphanumeric())
        || (prev_ch.is_ascii_alphanumeric() && is_cjk_hanzi(first_ch));
    if needs_space {
        let mut out = String::with_capacity(committed.len() + 1);
        out.push(' ');
        out.push_str(committed);
        out
    } else {
        committed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(ascii_char: u8, key_code: u16) -> KeyEvent {
        KeyEvent {
            ascii_char,
            key_code,
            shift: false,
            has_command_or_ctrl: false,
            secure_input: false,
        }
    }

    #[test]
    fn test_should_auto_commit_candidate_and_fullwidth_punctuation_on_comma() {
        let mut engine = ChimeEngine::default();
        let sid = engine.create_session("md.obsidian");
        for b in *b"nihao" {
            let _ = engine.process_key(sid, key(b, 0), 100);
        }
        let resp = engine.process_key(sid, key(b',', 0x2B), 100);
        assert_eq!(resp.committed_text, "你好，");
        assert_eq!(resp.preedit_text, "");
    }

    #[test]
    fn test_should_rollback_typo_correction_on_tab_and_support_step_by_step_prefix() {
        let mut engine = ChimeEngine::default();
        let sid = engine.create_session("md.obsidian");
        let mut last = EngineResponse::default();
        for b in *b"shaung" {
            last = engine.process_key(sid, key(b, 0), 100);
        }
        assert_eq!(last.candidates[0].text, "双");
        assert_eq!(last.candidates[0].kind, MatchKind::TypoCorrected);

        // Press Tab (0x30) -> jumps highlight to un-corrected raw `shaung`!
        let rolled = engine.process_key(sid, key(0, 0x30), 100);
        assert_eq!(rolled.candidates[rolled.highlighted_index].text, "shaung");

        // Reset and test step-by-step partial prefix commit on `jintianwomen`!
        engine.reset_session(sid);
        for b in *b"jintianwomen" {
            last = engine.process_key(sid, key(b, 0), 100);
        }
        assert_eq!(last.preedit_text, "jin'tian'wo'men");
        // Find partial candidate "今天" and select it:
        if let Some(pos) = last.candidates.iter().position(|c| c.text == "今天") {
            let step1 = engine.select_candidate(sid, pos, 100);
            assert_eq!(step1.committed_text, "");
            assert_eq!(step1.preedit_text, "今天 wo'men");
            // Now press Space (0x31) to finish remaining `women` -> commits `"今天我们"`!
            let step2 = engine.process_key(sid, key(b' ', 0x31), 100);
            assert_eq!(step2.committed_text, "今天我们");
        }
    }

    #[test]
    fn test_should_expand_5x5_grid_pick_single_chars_and_learn_online_habits() {
        let mut engine = ChimeEngine::default();
        let sid = engine.create_session("com.mitchellh.ghostty");

        // 1. Cross-commit Pangu spacing (`调整` + `chatgpt` -> `调整 ChatGPT`)
        for b in *b"tiaozheng" {
            let _ = engine.process_key(sid, key(b, 0), 200);
        }
        let c1 = engine.process_key(sid, key(b' ', 0x31), 200);
        assert_eq!(c1.committed_text, "调整");

        for b in *b"chatgpt" {
            let _ = engine.process_key(sid, key(b, 0), 200);
        }
        let c2 = engine.process_key(sid, key(b' ', 0x31), 200);
        assert_eq!(c2.committed_text, " ChatGPT");

        // 2. Press Down Arrow (`0x7D`) on `shi` to expand `5x5` grid (`candidates.len() > 5`)
        for b in *b"shi" {
            let _ = engine.process_key(sid, key(b, 0), 200);
        }
        let expanded = engine.process_key(sid, key(0, 0x7D), 200);
        assert!(expanded.grid_expanded);
        assert!(expanded.candidates.len() > 5);
        let _ = engine.process_key(sid, key(0, 0x35), 200); // collapse grid
        let _ = engine.process_key(sid, key(0, 0x35), 200); // cancel composition

        // 3. Punctuation toggle (`chinese_punctuation = true -> false -> true`)
        let p_off = engine.toggle_punctuation(sid);
        assert!(!p_off.chinese_punctuation);
        let p_on = engine.toggle_punctuation(sid);
        assert!(p_on.chinese_punctuation);

        // 4. `一路 Tab` poetry completion (`tianshengwocai` -> `天生我材必有用` ->
        //    `，千金散尽还复来`)
        engine.reset_session(sid);
        for b in *b"tianshengwocai" {
            let _ = engine.process_key(sid, key(b, 0), 200);
        }
        let t1 = engine.process_key(sid, key(0, 0x30), 200);
        assert!(t1.committed_text.contains("必有用"));
        assert_eq!(t1.ghost_text, "，千金散尽还复来");
        let t2 = engine.process_key(sid, key(0, 0x30), 200);
        assert_eq!(t2.committed_text, "，千金散尽还复来");
        assert_eq!(t2.ghost_text, "。烹羊宰牛且为乐");

        // 5. Verify `yueyoubeihuanlihe` -> Space -> `commit_composition_if_active` preserves
        //    `ghost_text = "，人有旦夕祸福"` and `Tab` commits it!
        engine.reset_session(sid);
        for b in *b"yueyoubeihuanlihe" {
            let _ = engine.process_key(sid, key(b, 0), 200);
        }
        let c_moon = engine.process_key(sid, key(b' ', 0x31), 200);
        assert_eq!(c_moon.committed_text, "月有悲欢离合");
        assert_eq!(c_moon.ghost_text, "，人有旦夕祸福");
        // Simulate macOS TSM calling `commitComposition` right after `setMarkedText("")`:
        let after_tsm = engine.commit_composition_if_active(sid);
        assert_eq!(after_tsm.ghost_text, "，人有旦夕祸福");
        let c_next = engine.process_key(sid, key(0, 0x30), 200);
        assert_eq!(c_next.committed_text, "，人有旦夕祸福");

        // 6. Verify single-char habit (`cheng -> 成`) does NOT poison `guochengzhong` (`过程中`),
        //    and idle Backspace (`0x33`) reverts mistaken auto-commits & pops `left_context`!
        for b in *b"cheng" {
            let _ = engine.process_key(sid, key(b, 0), 200);
        }
        let _ = engine.process_key(sid, key(b' ', 0x31), 200);
        let mut gc = EngineResponse::default();
        for b in *b"guochengzhong" {
            gc = engine.process_key(sid, key(b, 0), 200);
        }
        assert_eq!(gc.candidates[0].text, "过程中");
    }

    #[test]
    fn test_should_support_sogou_doubao_usability_settings_pangu_numbers_shift_and_paging() {
        let mut engine = ChimeEngine::default();
        let sid = engine.create_session("md.obsidian");

        // 1. Default `pangu_spacing = true`: symmetric spacing around numbers (`我有 100 个`)
        for b in *b"woyou" {
            let _ = engine.process_key(sid, key(b, 0), 300);
        }
        let r_woyou = engine.process_key(sid, key(b' ', 0x31), 300);
        assert_eq!(r_woyou.committed_text, "我有");

        let r_d1 = engine.process_key(sid, key(b'1', 0x12), 300);
        assert!(r_d1.handled);
        assert_eq!(r_d1.committed_text, " 1");
        let r_d0 = engine.process_key(sid, key(b'0', 0x1D), 300);
        assert!(!r_d0.handled); // subsequent digits pass through natively
        let _ = engine.process_key(sid, key(b'0', 0x1D), 300);

        for b in *b"ge" {
            let _ = engine.process_key(sid, key(b, 0), 300);
        }
        let r_ge = engine.process_key(sid, key(b' ', 0x31), 300);
        assert_eq!(r_ge.committed_text, " 个");

        // 2. Toggle `pangu_spacing = false`: zero spaces inserted around numbers or English!
        let _ = engine.toggle_pangu_spacing(sid);
        assert!(!engine.config().pangu_spacing);
        engine.reset_session(sid);
        for b in *b"woyou" {
            let _ = engine.process_key(sid, key(b, 0), 300);
        }
        let _ = engine.process_key(sid, key(b' ', 0x31), 300);
        let r_no_space = engine.process_key(sid, key(b'5', 0x17), 300);
        assert!(!r_no_space.handled);
        assert_eq!(r_no_space.committed_text, "");
        for b in *b"ge" {
            let _ = engine.process_key(sid, key(b, 0), 300);
        }
        let r_ge_nospace = engine.process_key(sid, key(b' ', 0x31), 300);
        assert_eq!(r_ge_nospace.committed_text, "个");

        // Re-enable `pangu_spacing = true` and verify inline English mode (`Shift`) spacing!
        let _ = engine.toggle_pangu_spacing(sid);
        for b in *b"shiyong" {
            let _ = engine.process_key(sid, key(b, 0), 300);
        }
        let _ = engine.process_key(sid, key(b' ', 0x31), 300); // commits `使用` (or ` 使用`)
        let to_eng = engine.toggle_inline_english(sid);
        assert!(to_eng.inline_english_mode);
        let first_eng = engine.process_key(sid, key(b'R', 0x0F), 300);
        assert!(first_eng.handled);
        assert_eq!(first_eng.committed_text, " R");
        let _ = engine.process_key(sid, key(b'u', 0), 300);
        let _ = engine.process_key(sid, key(b's', 0), 300);
        let _ = engine.process_key(sid, key(b't', 0), 300);
        let back_zh = engine.toggle_inline_english(sid);
        assert!(!back_zh.inline_english_mode);

        // 3. Leading uppercase direct English (`Hello`, `Mac` -> `Mac` + `MacBook`)
        let mut up_resp = EngineResponse::default();
        for b in *b"Mac" {
            up_resp = engine.process_key(sid, key(b, 0), 300);
        }
        assert_eq!(up_resp.candidates[0].text, "Mac");
        assert!(up_resp.candidates.iter().any(|c| c.text == "MacBook"));
        let committed_mac = engine.process_key(sid, key(b' ', 0x31), 300);
        assert!(committed_mac.committed_text.ends_with("Mac"));

        // 4. `ShiftCommitMode::CommitKeepChinese` commits raw ASCII while staying in Chinese mode!
        let mut cfg = engine.config().clone();
        cfg.shift_commit_mode = ShiftCommitMode::CommitKeepChinese;
        cfg.bracket_paging = true;
        cfg.semicolon_apostrophe_select = true;
        engine
            .update_config(cfg)
            .unwrap_or_else(|_| std::process::abort());
        for b in *b"dev" {
            let _ = engine.process_key(sid, key(b, 0), 300);
        }
        let keep_zh = engine.toggle_inline_english(sid);
        assert!(!keep_zh.inline_english_mode);
        assert!(keep_zh.committed_text.ends_with("dev"));

        // 5. Smart numeric colon exemption (`12:30` -> `:` stays half-width!)
        let _ = engine.process_key(sid, key(b'1', 0x12), 300);
        let _ = engine.process_key(sid, key(b'2', 0x13), 300);
        let colon_resp = engine.process_key(sid, key(b':', 0x29), 300);
        assert_eq!(colon_resp.committed_text, ":");

        // 6. Bracket paging (`]`) and `;` 2nd-candidate quick selection + `clear_user_habits`
        for b in *b"shi" {
            let _ = engine.process_key(sid, key(b, 0), 300);
        }
        let paged = engine.process_key(sid, key(b']', 0x1E), 300);
        assert_eq!(paged.page_index, 1);
        let second_on_p2 = paged.candidates[1].text.clone();
        let quick_sel = engine.process_key(sid, key(b';', 0x29), 300);
        assert!(quick_sel.committed_text.ends_with(&second_on_p2));

        assert!(engine.status().learned_habits > 0);
        engine.clear_user_habits();
        assert_eq!(engine.status().learned_habits, 0);
        assert!(engine.config().auto_check_updates);
    }

    #[test]
    fn test_should_report_and_remove_installed_imepack_safely() {
        let support_dir = tempfile::tempdir().unwrap_or_else(|_| std::process::abort());
        let pack_build_dir = tempfile::tempdir().unwrap_or_else(|_| std::process::abort());
        let pack_src = pack_build_dir.path().join("demo-pack.imepack");
        let _ = crate::personalize::build_imepack_v2(
            &crate::personalize::PackBuildOptions {
                id: "demo-pack",
                name: "Demo Pack",
                description: "Unit test pack",
                version: "0.9.0",
                routing: chime_neural::LoraRouting::Developer,
                lora_scale: 1.0,
                lexicon_tsv: None,
                continuations_tsv: None,
                ngrams_tsv: None,
                lora_safetensors: None,
                neural_dir: None,
                private_pack: false,
            },
            &pack_src,
        )
        .unwrap_or_else(|_| std::process::abort());

        let mut engine = ChimeEngine::with_data_dir(support_dir.path(), None)
            .unwrap_or_else(|_| std::process::abort());
        let _ = engine
            .import_external_asset(&pack_src)
            .unwrap_or_else(|_| std::process::abort());

        let st = engine.status();
        assert_eq!(st.installed_packs, 1);
        assert_eq!(st.pack_details.len(), 1);
        assert_eq!(st.pack_details[0].id, "demo-pack");
        assert_eq!(st.pack_details[0].version, "0.9.0");

        // Reject path-traversal attempts cleanly
        assert!(!engine.remove_installed_pack("../etc"));
        assert!(engine.remove_installed_pack("demo-pack"));
        assert_eq!(engine.status().installed_packs, 0);
        assert_eq!(engine.status().pack_details.len(), 0);
    }

    #[test]
    fn test_should_auto_correct_inline_english_preserve_case_exempt_terminals_and_undo_on_backspace()
     {
        let mut engine = ChimeEngine::default();
        let sid = engine.create_session("md.obsidian");
        let toggled = engine.toggle_inline_english(sid);
        assert!(toggled.inline_english_mode);

        // 1. Capitalized `Teh ` -> `The ` (`replace_prev_chars == 3`, `toast_override == "Teh →
        //    The"`)
        for b in *b"Teh" {
            let r = engine.process_key(sid, key(b, 0), 500);
            assert!(!r.handled);
        }
        let r_space = engine.process_key(sid, key(b' ', 0x31), 500);
        assert!(r_space.handled);
        assert_eq!(r_space.replace_prev_chars, 3);
        assert_eq!(r_space.committed_text, "The ");
        assert_eq!(r_space.toast_override, "Teh → The");

        // 2. Press single `Backspace` (`0x33`) -> instant undo back to `"Teh"` AND suppresses
        //    future auto-correct on `"teh"`!
        let r_undo = engine.process_key(sid, key(0, 0x33), 500);
        assert!(r_undo.handled);
        assert_eq!(r_undo.replace_prev_chars, 4);
        assert_eq!(r_undo.committed_text, "Teh");
        assert!(r_undo.toast_override.contains("Teh"));

        // Pressing Space right after undo keeps `"Teh "` without re-correcting!
        let r_retype_space = engine.process_key(sid, key(b' ', 0x31), 500);
        assert!(!r_retype_space.handled);

        // 3. Sentence punctuation boundary (`recieve.` -> `receive.`) & camelCase exemption
        //    (`useEfect ` -> untouched)
        for b in *b"recieve" {
            let _ = engine.process_key(sid, key(b, 0), 500);
        }
        let r_dot = engine.process_key(sid, key(b'.', 0x2F), 500);
        assert!(r_dot.handled);
        assert_eq!(r_dot.replace_prev_chars, 7);
        assert_eq!(r_dot.committed_text, "receive.");

        let _ = engine.process_key(sid, key(b' ', 0x31), 500);
        for b in *b"useEfect" {
            let _ = engine.process_key(sid, key(b, 0), 500);
        }
        let r_camel = engine.process_key(sid, key(b' ', 0x31), 500);
        assert!(!r_camel.handled);

        // 4. Terminal apps (`com.mitchellh.ghostty`) are strictly exempted in `En` mode!
        let term_sid = engine.create_session("com.mitchellh.ghostty");
        for b in *b"definately" {
            let _ = engine.process_key(term_sid, key(b, 0), 500);
        }
        let r_term = engine.process_key(term_sid, key(b' ', 0x31), 500);
        assert!(!r_term.handled);
    }
}
