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
    sync::Arc,
    time::Instant,
};

use chime_decoder::{
    Candidate, DecodeContext, LatticeDecoder, MatchKind, apply_anti_jitter_freeze_window,
};
use chime_lexicon::{
    CompiledDictBuilder, Lexicon, TypingSample, UserStore, import_macos_text_replacements,
    parse_bdict_bytes, parse_rime_dict_yaml, parse_scel_bytes, parse_tsv_dict,
};
use chime_neural::{AsyncRerankHandle, RerankJob, SafetensorsReranker};
use chime_phonetics::{ShuangpinScheme, format_segmented_preedit};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::{
    config::{CandidateLayout, ChimeConfig, ConfigError},
    personalize::{
        ImePackManifest, PersonalizeError, TrainingRecord, build_colab_training_jsonl,
        build_imepack_dir, install_imepack_dir,
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
    /// Candidate window layout orientation (`horizontal` or `vertical`).
    pub candidate_layout: CandidateLayout,
    /// Monotonic keystroke generation ID for anti-jitter async rerank validation.
    pub generation_id: u64,
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
}

#[derive(Debug, Clone)]
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
    typo_rollback_prev_idx: Option<usize>,
    last_auto_committed: Option<(String, String)>,
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
            typo_rollback_prev_idx: None,
            last_auto_committed: None,
            generation_id: 0,
        }
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
        self.typo_rollback_prev_idx = None;
    }

    fn record_emitted_text(&mut self, emitted: &str) {
        if let Some(last_ch) = emitted.chars().last() {
            self.last_committed_char = Some(last_ch);
        }
        self.left_context.push_str(emitted);
        if self.left_context.chars().count() > 64 {
            self.left_context = self
                .left_context
                .chars()
                .rev()
                .take(64)
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
            sessions: VecDeque::with_capacity(MAX_ACTIVE_SESSIONS),
            next_session_id: 1,
        }
    }

    /// Initializes a full production engine backed by `app_support_dir` (`config.toml`,
    /// `user_store.db`, `dicts/*.imedict`, `neural/`) and optional
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
            sessions: VecDeque::with_capacity(MAX_ACTIVE_SESSIONS),
            next_session_id: 1,
        };
        engine.reload_assets();
        Ok(engine)
    }

    /// Hot-reloads `config.toml`, mounted `.imedict` dictionaries, `macOS` text replacements, and
    /// Stage-2 neural model.
    pub fn reload_assets(&mut self) {
        if let Some(dir) = &self.app_support_dir
            && let Ok(cfg) = ChimeConfig::load_or_init(&dir.join("config.toml"))
        {
            self.config = cfg;
        }

        let mut lex = Lexicon::with_builtin_seed();
        if let Some(res_dir) = &self.bundle_resources_dir {
            let core_path = res_dir.join("core.imedict");
            if core_path.exists() {
                let _ = lex.mount_compiled_mmap(&core_path);
            }
        }
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

            //Opportunistically import macOS System Text Replacements if readable.
            if let Some(home) = std::env::var_os("HOME") {
                let tr_db =
                    PathBuf::from(home).join("Library/KeyboardServices/TextReplacements.db");
                if tr_db.exists() {
                    let _ = import_macos_text_replacements(&tr_db, &mut lex);
                }
            }

            let neural_dir = dir.join("neural");
            if neural_dir.join("model.safetensors").exists()
                && let Ok(reranker) = SafetensorsReranker::load_dir(&neural_dir)
            {
                self.async_reranker = Some(AsyncRerankHandle::spawn(Arc::new(reranker)));
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
        }
    }

    /// Imports a user-selected file (`.scel`, `.bdict`, `.dict.yaml`, `.tsv`, `.imedict`, or
    /// `.imepack` dir), compiles it into `app_support_dir/dicts/` when applicable, and hot-mounts
    /// immediately!
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

        if path.is_dir() && path.join("manifest.json").exists() {
            let manifest = install_imepack_dir(path, &support_dir)?;
            self.reload_assets();
            return Ok(usize::from(manifest.has_dictionary) + usize::from(manifest.has_safetensors));
        }

        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
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

    /// Allocates a lightweight `session_id`, evicting the oldest session if over
    /// `MAX_ACTIVE_SESSIONS`.
    pub fn create_session(&mut self, app_bundle_id: impl Into<String>) -> u64 {
        let id = self.next_session_id;
        self.next_session_id = self.next_session_id.saturating_add(1);
        if self.sessions.len() >= MAX_ACTIVE_SESSIONS {
            self.sessions.pop_front();
        }
        self.sessions
            .push_back(InputSession::new(id, app_bundle_id.into()));
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
        }
    }

    /// Toggles inline English mode on clean `< 250ms` single Left-Shift tap, committing any raw
    /// preedit.
    pub fn toggle_inline_english(&mut self, session_id: u64) -> EngineResponse {
        let Some(idx) = self.touch_session(session_id) else {
            return EngineResponse::default();
        };
        let sess = &mut self.sessions[idx];
        sess.inline_english_mode = !sess.inline_english_mode;
        let mut committed = std::mem::take(&mut sess.partial_prefix);
        committed.push_str(&std::mem::take(&mut sess.raw_buffer));
        sess.clear_composing();
        if !committed.is_empty() {
            sess.record_emitted_text(&committed);
        }
        EngineResponse {
            handled: true,
            committed_text: committed,
            inline_english_mode: sess.inline_english_mode,
            candidate_layout: self.config.candidate_layout,
            generation_id: sess.generation_id,
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
            for (cand, bonus) in rescored.iter_mut().zip(res.bonus_scores) {
                // Only rescore full-span candidates; keep partial prefix tails in order below.
                if cand.consumed_bytes == sess.raw_buffer.len() {
                    cand.score += bonus;
                }
            }
            let (mut full_part, partial_part): (Vec<_>, Vec<_>) = rescored
                .into_iter()
                .partition(|c| c.consumed_bytes == sess.raw_buffer.len());
            full_part.sort_by(|a, b| b.score.total_cmp(&a.score));
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

        if self.sessions[idx].inline_english_mode {
            return EngineResponse {
                handled: false,
                inline_english_mode: true,
                candidate_layout: self.config.candidate_layout,
                ..EngineResponse::default()
            };
        }

        let ch = event.ascii_char;
        let is_composing = self.sessions[idx].is_composing();

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
                }
                resp.preedit_text.clear();
                resp.candidates.clear();
                return resp;
            } else if !is_composing {
                let prev_ch = self.sessions[idx].last_committed_char;
                if let Some(mapped) = self.sessions[idx].punctuation.map_punctuation(ch, prev_ch) {
                    self.sessions[idx].record_emitted_text(mapped);
                    return EngineResponse {
                        handled: true,
                        committed_text: mapped.to_string(),
                        candidate_layout: self.config.candidate_layout,
                        ..EngineResponse::default()
                    };
                }
            }
        }

        if !is_composing && ch.is_ascii_graphic() {
            self.sessions[idx].last_committed_char = Some(char::from(ch));
        }

        EngineResponse::default()
    }

    /// Selects candidate `page_relative_index` (`0..page_size`) on the active page (e.g. mouse
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
        let page_size = self.config.page_size.max(1);
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
        let page_size = self.config.page_size.max(1);

        match event.key_code {
            // Space -> commit highlighted candidate
            0x31 => {
                let sel = self.sessions[idx].selected_global_idx;
                return Some(self.commit_candidate_global(idx, sel, now_epoch_secs));
            }
            // Return -> commit raw ASCII directly
            0x24 => {
                let mut out = std::mem::take(&mut self.sessions[idx].partial_prefix);
                out.push_str(&std::mem::take(&mut self.sessions[idx].raw_buffer));
                self.sessions[idx].clear_composing();
                self.sessions[idx].record_emitted_text(&out);
                return Some(EngineResponse {
                    handled: true,
                    committed_text: out,
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
            // Escape -> cancel composition
            0x35 => {
                self.sessions[idx].clear_composing();
                return Some(EngineResponse {
                    handled: true,
                    candidate_layout: self.config.candidate_layout,
                    ..EngineResponse::default()
                });
            }
            // Tab -> Typo Rollback toggle (or cycle candidate)
            0x30 if total > 1 => {
                let cur = self.sessions[idx].selected_global_idx;
                if let Some(prev) = self.sessions[idx].typo_rollback_prev_idx.take() {
                    self.sessions[idx].selected_global_idx = prev.min(total - 1);
                } else if self.sessions[idx].all_candidates[cur].kind == MatchKind::TypoCorrected {
                    if let Some(exact_idx) = self.sessions[idx]
                        .all_candidates
                        .iter()
                        .position(|c| matches!(c.kind, MatchKind::Exact | MatchKind::CodeSwitch))
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
            // Down (0x7D) or Right (0x7C) -> next candidate
            0x7D | 0x7C if total > 0 => {
                self.sessions[idx].selected_global_idx =
                    (self.sessions[idx].selected_global_idx + 1) % total;
                return Some(self.build_snapshot(idx, true, String::new()));
            }
            // Up (0x7E) or Left (0x7B) -> previous candidate
            0x7E | 0x7B if total > 0 => {
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

        // `-` / `=` paging keys when multiple candidates exist
        if ch == b'=' && total > page_size {
            let next_page = ((self.sessions[idx].selected_global_idx / page_size) + 1) * page_size;
            self.sessions[idx].selected_global_idx = if next_page < total { next_page } else { 0 };
            return Some(self.build_snapshot(idx, true, String::new()));
        }
        if ch == b'-' && total > page_size {
            let cur_page = self.sessions[idx].selected_global_idx / page_size;
            self.sessions[idx].selected_global_idx = cur_page.saturating_sub(1) * page_size;
            return Some(self.build_snapshot(idx, true, String::new()));
        }

        // Digit 1..=page_size -> select candidate on the current page!
        if (b'1'..=b'9').contains(&ch) {
            let rel = usize::from(ch - b'1');
            if rel < page_size {
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
        self.sessions[idx].typo_rollback_prev_idx = None;

        if self.sessions[idx].raw_buffer.is_empty() {
            if !self.sessions[idx].partial_prefix.is_empty() {
                let committed = std::mem::take(&mut self.sessions[idx].partial_prefix);
                self.sessions[idx].clear_composing();
                self.sessions[idx].record_emitted_text(&committed);
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
            now_epoch_secs,
        };
        let cands = self
            .decoder
            .decode(&raw_clone, &self.lexicon, &self.user_store, &ctx);
        self.sessions[idx].all_candidates.clone_from(&cands);

        // Dispatch non-blocking Stage-2 neural rescoring job if loaded!
        if let Some(worker) = &self.async_reranker {
            let full_texts: Vec<String> = cands
                .iter()
                .filter(|c| c.consumed_bytes == raw_clone.len())
                .map(|c| c.text.clone())
                .collect();
            if full_texts.len() >= 2 {
                let _ = worker.submit(RerankJob {
                    session_id: self.sessions[idx].id,
                    generation_id: self.sessions[idx].generation_id,
                    left_context: effective_left,
                    candidate_texts: full_texts,
                    dispatched_at: Instant::now(),
                });
            }
        }

        self.build_snapshot(idx, true, String::new())
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

        self.sessions[idx].last_auto_committed = Some((learn_key.clone(), final_text.clone()));
        self.sessions[idx].record_emitted_text(&final_text);
        self.sessions[idx].clear_composing();

        self.build_snapshot(idx, true, final_text)
    }

    fn build_snapshot(&self, idx: usize, handled: bool, committed_text: String) -> EngineResponse {
        let sess = &self.sessions[idx];
        let page_size = self.config.page_size.max(1);
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
            candidate_layout: self.config.candidate_layout,
            generation_id: sess.generation_id,
        }
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
}
