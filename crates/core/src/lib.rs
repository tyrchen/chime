//! Chime IME core session state machine, bounded LRU session pool (preventing `macOS`
//! `IMKServer` controller memory leaks), TOML configuration, and Colab corpus sanitizer.

#![forbid(unsafe_code)]

pub mod config;
pub mod sanitize;

use std::collections::VecDeque;

use chime_decoder::{Candidate, DecodeContext, LatticeDecoder};
use chime_lexicon::{Lexicon, UserStore};
use serde::{Deserialize, Serialize};

pub use crate::{
    config::ChimeConfig,
    sanitize::{SanitizeReport, sanitize_corpus_text},
};

/// Maximum concurrent sessions retained in Rust regardless of leaked `IMKInputController` shells.
pub const MAX_ACTIVE_SESSIONS: usize = 32;

/// Key event passed across the FFI boundary from `macOS` `InputMethodKit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyEvent {
    /// Printable ASCII byte (`0` when special key such as Space, Return, Backspace, Escape).
    pub ascii_char: u8,
    /// `macOS` virtual key code (`kVK_Return = 0x24`, `kVK_Tab = 0x30`, `kVK_Space = 0x31`,
    /// `kVK_Delete = 0x33`, `kVK_Escape = 0x35`).
    pub key_code: u16,
    /// True if Shift is held.
    pub shift: bool,
    /// True if Command, Control, or Option is held.
    pub has_command_or_ctrl: bool,
    /// True if `macOS` `IsSecureEventInputEnabled()` is active (password / `sudo` field).
    pub secure_input: bool,
}

/// Snapshot returned synchronously (<2ms) to the Swift UI on every keystroke.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EngineResponse {
    /// True if the IME consumed the keystroke; false if `macOS` should pass it to the client app.
    pub handled: bool,
    /// Text to commit immediately to `NSTextInputClient.insertText`.
    pub committed_text: String,
    /// Inline marked preedit text to display at the cursor.
    pub preedit_text: String,
    /// Current page of ranked candidates for the non-activating `NSPanel`.
    pub candidates: Vec<Candidate>,
    /// Monotonic keystroke generation ID for anti-jitter async rerank validation.
    pub generation_id: u64,
}

#[derive(Debug, Clone)]
struct InputSession {
    id: u64,
    app_bundle_id: String,
    inline_english_mode: bool,
    raw_buffer: String,
    left_context: String,
    candidates: Vec<Candidate>,
    last_auto_committed: Option<(String, String)>,
    generation_id: u64,
}

impl InputSession {
    fn new(id: u64, app_bundle_id: String) -> Self {
        Self {
            id,
            app_bundle_id,
            inline_english_mode: false,
            raw_buffer: String::new(),
            left_context: String::new(),
            candidates: Vec::new(),
            last_auto_committed: None,
            generation_id: 0,
        }
    }

    fn clear_composing(&mut self) {
        self.raw_buffer.clear();
        self.candidates.clear();
    }
}

/// Thread-confined or mutex-wrapped core IME engine managing lexicon, decoder, and LRU sessions.
#[derive(Debug)]
pub struct ChimeEngine {
    config: ChimeConfig,
    lexicon: Lexicon,
    user_store: UserStore,
    decoder: LatticeDecoder,
    sessions: VecDeque<InputSession>,
    next_session_id: u64,
}

impl Default for ChimeEngine {
    fn default() -> Self {
        Self::new(ChimeConfig::default())
    }
}

impl ChimeEngine {
    /// Initializes a new engine with the built-in seed lexicon and configuration.
    #[must_use]
    pub fn new(config: ChimeConfig) -> Self {
        Self {
            config,
            lexicon: Lexicon::with_builtin_seed(),
            user_store: UserStore::new(),
            decoder: LatticeDecoder::new(),
            sessions: VecDeque::with_capacity(MAX_ACTIVE_SESSIONS),
            next_session_id: 1,
        }
    }

    /// Borrows the underlying [`Lexicon`] mutably (e.g. for importing `.scel` or Rime dicts).
    pub fn lexicon_mut(&mut self) -> &mut Lexicon {
        &mut self.lexicon
    }

    /// Borrows the underlying [`Lexicon`].
    #[must_use]
    pub const fn lexicon(&self) -> &Lexicon {
        &self.lexicon
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
        let committed = std::mem::take(&mut sess.raw_buffer);
        sess.candidates.clear();
        EngineResponse {
            handled: !committed.is_empty(),
            committed_text: committed,
            preedit_text: String::new(),
            candidates: Vec::new(),
            generation_id: sess.generation_id,
        }
    }

    /// Processes a single key event synchronously on the hot path.
    pub fn process_key(
        &mut self,
        session_id: u64,
        event: KeyEvent,
        now_epoch_secs: u64,
    ) -> EngineResponse {
        // Hard bypass for password fields (`IsSecureEventInputEnabled`) or Cmd/Ctrl shortcuts.
        if event.secure_input || event.has_command_or_ctrl {
            self.reset_session(session_id);
            return EngineResponse::default();
        }

        let Some(idx) = self.touch_session(session_id) else {
            return EngineResponse::default();
        };

        if self.sessions[idx].inline_english_mode {
            return EngineResponse::default();
        }

        let ch = event.ascii_char;
        let is_composing = !self.sessions[idx].raw_buffer.is_empty();

        // Handle special keys when composing:
        // Space (0x31) -> commit #1 candidate; Return (0x24) -> commit raw ASCII;
        // Backspace (0x33) -> pop 1 char; Escape (0x35) -> cancel preedit; Digit 1..9 -> pick Nth.
        if is_composing {
            match event.key_code {
                0x31 => return self.commit_candidate_at(idx, 0, now_epoch_secs),
                0x24 => {
                    let raw = std::mem::take(&mut self.sessions[idx].raw_buffer);
                    self.sessions[idx].candidates.clear();
                    return EngineResponse {
                        handled: true,
                        committed_text: raw,
                        ..EngineResponse::default()
                    };
                }
                0x33 => {
                    self.sessions[idx].raw_buffer.pop();
                    return self.recompute_session(idx, now_epoch_secs);
                }
                0x35 => {
                    self.sessions[idx].clear_composing();
                    return EngineResponse {
                        handled: true,
                        ..EngineResponse::default()
                    };
                }
                _ => {}
            }

            if (b'1'..=b'9').contains(&ch) {
                let pick = usize::from(ch - b'1');
                if pick < self.sessions[idx].candidates.len() {
                    return self.commit_candidate_at(idx, pick, now_epoch_secs);
                }
            }
        }

        // Start composing on ASCII letters; once composing, also allow `'`, `.`, `_`, `-`, digits
        // for code-switching.
        let starts_composing = ch.is_ascii_alphabetic();
        let continues_composing = is_composing
            && (ch.is_ascii_alphanumeric() || matches!(ch, b'\'' | b'_' | b'.' | b'-' | b'/'));

        if starts_composing || continues_composing {
            self.sessions[idx].raw_buffer.push(char::from(ch));
            return self.recompute_session(idx, now_epoch_secs);
        }

        EngineResponse::default()
    }

    /// Explicitly selects candidate `index` (0-based) from the UI panel.
    pub fn select_candidate(
        &mut self,
        session_id: u64,
        index: usize,
        now_epoch_secs: u64,
    ) -> EngineResponse {
        let Some(idx) = self.touch_session(session_id) else {
            return EngineResponse::default();
        };
        self.commit_candidate_at(idx, index, now_epoch_secs)
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

    fn recompute_session(&mut self, idx: usize, now_epoch_secs: u64) -> EngineResponse {
        self.sessions[idx].generation_id = self.sessions[idx].generation_id.saturating_add(1);
        let gen_id = self.sessions[idx].generation_id;
        if self.sessions[idx].raw_buffer.is_empty() {
            self.sessions[idx].candidates.clear();
            return EngineResponse {
                handled: true,
                generation_id: gen_id,
                ..EngineResponse::default()
            };
        }

        let raw_clone = self.sessions[idx].raw_buffer.clone();
        let left_clone = self.sessions[idx].left_context.clone();
        let app_clone = self.sessions[idx].app_bundle_id.clone();

        let ctx = DecodeContext {
            left_context: &left_clone,
            app_bundle_id: &app_clone,
            scheme: self.config.scheme,
            fuzzy: self.config.fuzzy,
            pangu_spacing: self.config.pangu_spacing,
            now_epoch_secs,
        };
        let cands = self
            .decoder
            .decode(&raw_clone, &self.lexicon, &self.user_store, &ctx);
        self.sessions[idx].candidates.clone_from(&cands);

        EngineResponse {
            handled: true,
            committed_text: String::new(),
            preedit_text: raw_clone,
            candidates: cands,
            generation_id: gen_id,
        }
    }

    fn commit_candidate_at(
        &mut self,
        idx: usize,
        cand_idx: usize,
        now_epoch_secs: u64,
    ) -> EngineResponse {
        let Some(chosen) = self.sessions[idx].candidates.get(cand_idx).cloned() else {
            return EngineResponse::default();
        };
        let reading = std::mem::take(&mut self.sessions[idx].raw_buffer);

        // If user manually chose candidate > 0, or re-typed the same reading right after an
        // auto-commit, record negative feedback against candidate #0 so the engine adapts rapidly!
        if cand_idx > 0
            && let Some(top_rejected) = self.sessions[idx].candidates.first()
        {
            self.user_store
                .record_negative_feedback(&reading, &top_rejected.text, &chosen.text);
        } else if let Some((prev_reading, prev_word)) = &self.sessions[idx].last_auto_committed
            && *prev_reading == reading
            && *prev_word != chosen.text
        {
            self.user_store
                .record_negative_feedback(&reading, prev_word, &chosen.text);
        }

        self.user_store
            .record_commit(&reading, &chosen.text, now_epoch_secs);
        self.sessions[idx].last_auto_committed = Some((reading, chosen.text.clone()));
        self.sessions[idx].left_context.push_str(&chosen.text);
        if self.sessions[idx].left_context.chars().count() > 64 {
            let trimmed: String = self.sessions[idx]
                .left_context
                .chars()
                .rev()
                .take(64)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            self.sessions[idx].left_context = trimmed;
        }
        self.sessions[idx].candidates.clear();

        EngineResponse {
            handled: true,
            committed_text: chosen.text,
            preedit_text: String::new(),
            candidates: Vec::new(),
            generation_id: self.sessions[idx].generation_id,
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

    #[test]
    fn test_should_bound_lru_session_pool_against_controller_leaks() {
        let mut engine = ChimeEngine::default();
        let first_id = engine.create_session("com.apple.TextEdit");
        for _ in 0..MAX_ACTIVE_SESSIONS {
            let _ = engine.create_session("com.google.Chrome");
        }
        // `first_id` must have been evicted cleanly; pool never exceeds MAX_ACTIVE_SESSIONS.
        let resp = engine.process_key(
            first_id,
            KeyEvent {
                ascii_char: b'n',
                key_code: 0x2D,
                shift: false,
                has_command_or_ctrl: false,
                secure_input: false,
            },
            100,
        );
        assert!(!resp.handled);
    }

    #[test]
    fn test_should_bypass_secure_password_input_completely() {
        let mut engine = ChimeEngine::default();
        let sid = engine.create_session("com.mitchellh.ghostty");
        let resp = engine.process_key(
            sid,
            KeyEvent {
                ascii_char: b's',
                key_code: 1,
                shift: false,
                has_command_or_ctrl: false,
                secure_input: true,
            },
            100,
        );
        assert!(!resp.handled);
        assert_eq!(resp.preedit_text, "");
    }
}
