//! Hold-to-talk voice dictation for Chime.
//!
//! Audio arrives from the macOS front-end as 16 kHz mono `f32` PCM. A lightweight energy VAD
//! ([`Segmenter`]) cuts the stream into utterances at natural pauses, and a background worker sends
//! each utterance to a self-hosted speech-recognition server ([`HttpTranscriber`], speaking the
//! OpenAI-compatible `POST /v1/audio/transcriptions` multipart protocol used by whisper.cpp
//! `server`, faster-whisper / speaches, vLLM, `LocalAI`, …). While the user is still talking, the
//! worker also re-transcribes the open utterance every `partial_interval_ms` so the UI can show a
//! live preview.

#![forbid(unsafe_code)]

mod client;
mod session;
mod streaming;
mod text;
mod vad;
mod wav;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::{
    client::{HttpTranscriber, Transcriber, build_multipart_body, parse_transcript},
    session::{VoiceSession, VoiceSnapshot, VoiceState},
    streaming::StreamingSession,
    text::{join_segments, normalize_transcript},
    vad::{Segmenter, VadParams},
    wav::encode_wav_pcm16,
};

/// Errors raised while talking to the speech-recognition server.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum VoiceError {
    /// Settings are unusable (for example a malformed endpoint URL).
    #[error("invalid voice settings: {0}")]
    InvalidSettings(String),
    /// Connection, DNS, TLS, or timeout failure.
    #[error("cannot reach speech server: {0}")]
    Transport(String),
    /// The server answered with a non-2xx status.
    #[error("speech server returned HTTP {status}: {body}")]
    Status {
        /// HTTP status code.
        status: u16,
        /// First bytes of the response body.
        body: String,
    },
    /// The server answered 2xx but the body could not be understood.
    #[error("unrecognized speech server response: {0}")]
    Response(String),
}

/// Runtime options for one dictation session.
///
/// The macOS front-end builds this from `[voice]` in `config.toml` (plus the global
/// `chinese_punctuation` switch) and passes it as JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct VoiceSettings {
    /// Full transcription URL, e.g. `http://127.0.0.1:8000/v1/audio/transcriptions`.
    pub endpoint: String,
    /// `model` form field (ignored by whisper.cpp, required by most OpenAI-compatible servers).
    pub model: String,
    /// ISO-639-1 language hint (`zh`, `en`, …). Empty lets the server auto-detect.
    pub language: String,
    /// Optional decoding prompt (Whisper `initial_prompt`), useful to bias towards Simplified
    /// Chinese with punctuation or towards domain vocabulary.
    pub prompt: String,
    /// Re-transcribe the open utterance periodically to show live text while speaking.
    pub live_preview: bool,
    /// Minimum gap between two live-preview requests.
    pub partial_interval_ms: u64,
    /// Per-request timeout.
    pub request_timeout_ms: u64,
    /// Recording stops automatically after this many seconds.
    pub max_record_secs: u64,
    /// Convert ASCII `,.?!:;` next to Chinese characters into full-width punctuation.
    pub chinese_punctuation: bool,
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8000/v1/audio/transcriptions".to_string(),
            model: "whisper-1".to_string(),
            language: "zh".to_string(),
            prompt: String::new(),
            live_preview: true,
            partial_interval_ms: 700,
            request_timeout_ms: 20_000,
            max_record_secs: 180,
            chinese_punctuation: true,
        }
    }
}

impl VoiceSettings {
    /// Checks that the endpoint looks like an `http(s)://` URL.
    ///
    /// # Errors
    ///
    /// Returns [`VoiceError::InvalidSettings`] when the endpoint is empty or not HTTP(S).
    pub fn validate(&self) -> Result<(), VoiceError> {
        let ep = self.endpoint.trim();
        let rest = ep
            .strip_prefix("http://")
            .or_else(|| ep.strip_prefix("https://"));
        match rest {
            Some(host) if !host.is_empty() && !host.starts_with('/') => Ok(()),
            _ => Err(VoiceError::InvalidSettings(format!(
                "endpoint must be an http:// or https:// URL, got `{ep}`"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_endpoint_scheme() {
        let mut s = VoiceSettings::default();
        assert!(s.validate().is_ok());
        s.endpoint = "https://asr.internal:9000/inference".into();
        assert!(s.validate().is_ok());
        for bad in ["", "ftp://x", "http://", "127.0.0.1:8000", "http:///path"] {
            s.endpoint = bad.into();
            assert!(s.validate().is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn settings_json_ignores_unknown_and_fills_defaults() {
        let parsed: VoiceSettings =
            serde_json::from_str(r#"{"endpoint":"http://h/x","hotkey":"right_option"}"#)
                .unwrap_or_else(|_| std::process::abort());
        assert_eq!(parsed.endpoint, "http://h/x");
        assert_eq!(parsed.language, "zh");
        assert!(parsed.live_preview);
    }
}
