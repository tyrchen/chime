//! C-ABI for hold-to-talk voice dictation (`chime_voice_*`).
//!
//! The voice handle is independent of [`crate::ChimeEngineHandle`]: audio is pushed from the
//! `AVAudioEngine` tap thread while the UI polls snapshots from the main thread, so the handle is
//! internally synchronized and every function may be called from any thread.

use std::{
    ffi::{CStr, CString, c_char},
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::{Duration, Instant},
};

use chime_voice::{
    HttpTranscriber, StreamingSession, Transcriber, VoiceSession, VoiceSettings, VoiceSnapshot,
};
use serde::Serialize;

#[derive(Debug)]
enum Session {
    Http(VoiceSession),
    Bundled(StreamingSession),
}

impl Session {
    fn snapshot(&self) -> VoiceSnapshot {
        match self {
            Self::Http(s) => s.snapshot(),
            Self::Bundled(s) => s.snapshot(),
        }
    }
    fn push_audio(&mut self, pcm: &[f32]) {
        match self {
            Self::Http(s) => s.push_audio(pcm),
            Self::Bundled(s) => s.push_audio(pcm),
        }
    }
    fn stop(&mut self) {
        match self {
            Self::Http(s) => s.stop(),
            Self::Bundled(s) => s.stop(),
        }
    }
}

/// Opaque, thread-safe voice dictation handle owned by Swift's `ChimeVoiceController`.
#[derive(Debug, Default)]
pub struct ChimeVoiceHandle {
    session: Mutex<Option<Session>>,
}

impl ChimeVoiceHandle {
    fn lock(&self) -> MutexGuard<'_, Option<Session>> {
        self.session.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn to_c_string(s: &str) -> *mut c_char {
    CString::new(s.replace('\0', "")).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// # Safety
///
/// `ptr` must be null or a valid NUL-terminated C string.
unsafe fn opt_str(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: Guaranteed by caller.
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    if s.is_empty() { None } else { Some(s) }
}

fn parse_settings(json: Option<String>) -> Result<VoiceSettings, String> {
    let raw = json.unwrap_or_else(|| "{}".to_string());
    let settings: VoiceSettings =
        serde_json::from_str(&raw).map_err(|e| format!("invalid voice settings JSON: {e}"))?;
    settings.validate().map_err(|e| e.to_string())?;
    Ok(settings)
}

/// Allocates a voice dictation handle.
#[unsafe(no_mangle)]
pub extern "C" fn chime_voice_new() -> *mut ChimeVoiceHandle {
    Box::into_raw(Box::new(ChimeVoiceHandle::default()))
}

/// Frees a handle from [`chime_voice_new`], cancelling any running session.
///
/// # Safety
///
/// `voice` must be null or an un-freed pointer returned by [`chime_voice_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_voice_free(voice: *mut ChimeVoiceHandle) {
    if !voice.is_null() {
        // SAFETY: Allocated via `Box::into_raw`.
        unsafe { drop(Box::from_raw(voice)) };
    }
}

/// Starts a new dictation session (cancelling any previous one).
///
/// `settings_json` is a [`VoiceSettings`] JSON object; `api_key` may be null. Audio pushed later
/// must be mono `f32` at `sample_rate` Hz. Returns null on success, or an error message that must
/// be freed with [`crate::chime_string_free`].
///
/// # Safety
///
/// `voice` must be valid; string arguments must be null or valid NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_voice_start(
    voice: *mut ChimeVoiceHandle,
    settings_json: *const c_char,
    api_key: *const c_char,
    sample_rate: u32,
) -> *mut c_char {
    if voice.is_null() {
        return to_c_string("voice handle is null");
    }
    // SAFETY: Caller contract.
    let (json, key) = unsafe { (opt_str(settings_json), opt_str(api_key)) };
    let settings = match parse_settings(json) {
        Ok(s) => s,
        Err(e) => return to_c_string(&e),
    };
    let transcriber = match HttpTranscriber::new(&settings, key) {
        Ok(t) => t,
        Err(e) => return to_c_string(&e.to_string()),
    };
    // SAFETY: `voice` is valid; the handle is internally synchronized.
    let handle = unsafe { &*voice };
    let mut slot = handle.lock();
    // Dropping the previous session cancels it.
    *slot = Some(Session::Http(VoiceSession::start(
        settings,
        Arc::new(transcriber),
        sample_rate,
    )));
    std::ptr::null_mut()
}

/// Starts a local 16 kHz stream using separately installed weights without reading HTTP settings or
/// credentials. A non-empty `sense_dir` adds an optional whole-utterance `SenseVoice` final;
/// `punctuation` keeps its marks and `layout` adds paragraphs and digit spacing.
///
/// # Safety
/// `voice` must be valid; `contents` and `model_dir` must be valid NUL-terminated UTF-8, and
/// `sense_dir` null or NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_voice_start_local(
    voice: *mut ChimeVoiceHandle,
    contents: *const c_char,
    model_dir: *const c_char,
    sense_dir: *const c_char,
    punctuation: bool,
    layout: bool,
    preview: bool,
    seconds: u64,
) -> *mut c_char {
    let sense = unsafe { opt_str(sense_dir) };
    let (Some(handle), Some(path)) = (unsafe { voice.as_ref() }, unsafe { opt_str(contents) })
    else {
        return to_c_string("missing voice handle or app bundle");
    };
    let Some(dir) = (unsafe { opt_str(model_dir) }) else {
        return to_c_string("missing local model directory");
    };
    match StreamingSession::start_with_final(
        std::path::Path::new(&path),
        std::path::Path::new(&dir),
        preview,
        seconds,
        sense
            .as_deref()
            .map(|dir| (std::path::Path::new(dir), punctuation, layout)),
    ) {
        Ok(s) => {
            *handle.lock() = Some(Session::Bundled(s));
            std::ptr::null_mut()
        }
        Err(e) => to_c_string(&e.to_string()),
    }
}

/// Appends `len` mono `f32` samples to the running session (no-op without one).
///
/// # Safety
///
/// `voice` must be valid; `samples` must point to `len` readable `f32` values (or be null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_voice_push_pcm(
    voice: *mut ChimeVoiceHandle,
    samples: *const f32,
    len: usize,
) {
    if voice.is_null() || samples.is_null() || len == 0 {
        return;
    }
    // SAFETY: Caller guarantees `len` readable samples.
    let pcm = unsafe { std::slice::from_raw_parts(samples, len) };
    // SAFETY: `voice` is valid; the handle is internally synchronized.
    let handle = unsafe { &*voice };
    if let Some(sess) = handle.lock().as_mut() {
        sess.push_audio(pcm);
    }
}

/// Stops recording; recognition of buffered audio continues until the snapshot reports `done`.
///
/// # Safety
///
/// `voice` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_voice_stop(voice: *mut ChimeVoiceHandle) {
    if voice.is_null() {
        return;
    }
    // SAFETY: `voice` is valid.
    let handle = unsafe { &*voice };
    if let Some(sess) = handle.lock().as_mut() {
        sess.stop();
    }
}

/// Cancels and discards the running session.
///
/// # Safety
///
/// `voice` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_voice_cancel(voice: *mut ChimeVoiceHandle) {
    if voice.is_null() {
        return;
    }
    // SAFETY: `voice` is valid.
    let handle = unsafe { &*voice };
    drop(handle.lock().take());
}

/// Releases a finished session after its text has been committed.
///
/// # Safety
///
/// `voice` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_voice_end(voice: *mut ChimeVoiceHandle) {
    if voice.is_null() {
        return;
    }
    // SAFETY: `voice` is valid.
    let handle = unsafe { &*voice };
    drop(handle.lock().take());
}

/// Returns the running session's [`chime_voice::VoiceSnapshot`] as JSON, or null without one.
/// Free with [`crate::chime_string_free`].
///
/// # Safety
///
/// `voice` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_voice_snapshot(voice: *mut ChimeVoiceHandle) -> *mut c_char {
    if voice.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `voice` is valid.
    let handle = unsafe { &*voice };
    let snap = handle.lock().as_ref().map(Session::snapshot);
    snap.map_or(std::ptr::null_mut(), |s| {
        to_c_string(&serde_json::to_string(&s).unwrap_or_else(|_| "{}".into()))
    })
}

/// Format an ASR final without accessing the IMK client or network. Blocking; use off main.
///
/// # Safety
/// `contents` and `text` must be valid NUL-terminated UTF-8. Free the result with
/// `chime_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_voice_format_final(
    contents: *const c_char,
    text: *const c_char,
    punctuation: bool,
    tidy: bool,
) -> *mut c_char {
    let (Some(path), Some(raw)) = (unsafe { opt_str(contents) }, unsafe { opt_str(text) }) else {
        return std::ptr::null_mut();
    };
    to_c_string(&chime_voice::format_final(
        std::path::Path::new(&path),
        &raw,
        punctuation,
        tidy,
    ))
}

#[derive(Serialize)]
struct EndpointProbe {
    ok: bool,
    text: String,
    latency_ms: u64,
    error: Option<String>,
}

/// Sends 0.6 s of silence to the configured endpoint to check reachability, authentication and
/// model name. **Blocking** (up to the request timeout): call it off the main thread. Returns JSON
/// `{ok, text, latency_ms, error}`; free with [`crate::chime_string_free`].
///
/// # Safety
///
/// String arguments must be null or valid NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_voice_test_endpoint(
    settings_json: *const c_char,
    api_key: *const c_char,
) -> *mut c_char {
    // SAFETY: Caller contract.
    let (json, key) = unsafe { (opt_str(settings_json), opt_str(api_key)) };
    let started = Instant::now();
    let outcome = parse_settings(json).and_then(|s| {
        let client = HttpTranscriber::new(&s, key).map_err(|e| e.to_string())?;
        client
            .transcribe(&vec![0.0_f32; 9_600], 16_000)
            .map_err(|e| e.to_string())
    });
    let latency_ms =
        u64::try_from(started.elapsed().min(Duration::from_secs(3600)).as_millis()).unwrap_or(0);
    let probe = match outcome {
        Ok(text) => EndpointProbe {
            ok: true,
            text,
            latency_ms,
            error: None,
        },
        Err(e) => EndpointProbe {
            ok: false,
            text: String::new(),
            latency_ms,
            error: Some(e),
        },
    };
    to_c_string(&serde_json::to_string(&probe).unwrap_or_else(|_| "{}".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_rejects_bad_settings_and_snapshot_is_null_without_session() {
        let h = chime_voice_new();
        let bad = CString::new(r#"{"endpoint":"not-a-url"}"#).unwrap_or_default();
        // SAFETY: Valid handle and strings.
        unsafe {
            let err = chime_voice_start(h, bad.as_ptr(), std::ptr::null(), 16_000);
            assert!(!err.is_null());
            crate::chime_string_free(err);
            assert!(chime_voice_snapshot(h).is_null());

            let ok = CString::new(r#"{"endpoint":"http://127.0.0.1:9/x","live_preview":false}"#)
                .unwrap_or_default();
            let err = chime_voice_start(h, ok.as_ptr(), std::ptr::null(), 16_000);
            assert!(err.is_null());
            let silence = [0.0_f32; 1600];
            chime_voice_push_pcm(h, silence.as_ptr(), silence.len());
            let snap = chime_voice_snapshot(h);
            assert!(!snap.is_null());
            let json = CStr::from_ptr(snap).to_string_lossy().into_owned();
            assert!(json.contains("\"state\":\"recording\""), "{json}");
            crate::chime_string_free(snap);
            chime_voice_cancel(h);
            assert!(chime_voice_snapshot(h).is_null());
            chime_voice_free(h);
        }
    }
}
