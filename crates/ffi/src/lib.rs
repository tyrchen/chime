//! Narrow C-ABI hourglass waist bridging `chime-core` to Swift 6 (`@MainActor` `AppKit`/IMK).

use std::{
    ffi::{CString, c_char},
    time::{SystemTime, UNIX_EPOCH},
};

use chime_core::{ChimeConfig, ChimeEngine, EngineResponse, KeyEvent};

/// Opaque handle owned by Swift's `@MainActor final class ChimeEngineBridge`.
#[derive(Debug)]
pub struct ChimeEngineHandle {
    inner: ChimeEngine,
}

fn current_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn encode_response_json(resp: &EngineResponse) -> *mut c_char {
    let json = serde_json::to_string(resp).unwrap_or_else(|_| "{}".to_string());
    CString::new(json).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// Allocates a new [`ChimeEngineHandle`] configured with default settings.
#[unsafe(no_mangle)]
pub extern "C" fn chime_engine_new() -> *mut ChimeEngineHandle {
    let handle = Box::new(ChimeEngineHandle {
        inner: ChimeEngine::new(ChimeConfig::default()),
    });
    Box::into_raw(handle)
}

/// Frees a [`ChimeEngineHandle`] previously created by [`chime_engine_new`].
///
/// # Safety
///
/// `engine` must be either null or a valid pointer returned by [`chime_engine_new`] that has
/// not yet been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_engine_free(engine: *mut ChimeEngineHandle) {
    if !engine.is_null() {
        // SAFETY: Caller guarantees `engine` was allocated via `Box::into_raw` in
        // `chime_engine_new`.
        unsafe {
            drop(Box::from_raw(engine));
        }
    }
}

/// Creates a lightweight session for `app_bundle_id` (UTF-8 C string) and returns its `session_id`.
///
/// # Safety
///
/// `engine` must be a valid pointer from [`chime_engine_new`]; `app_bundle_id` may be null or a
/// valid NUL-terminated UTF-8 C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_session_create(
    engine: *mut ChimeEngineHandle,
    app_bundle_id: *const c_char,
) -> u64 {
    if engine.is_null() {
        return 0;
    }
    let bundle = if app_bundle_id.is_null() {
        String::new()
    } else {
        // SAFETY: Caller passes a valid NUL-terminated string pointer when non-null.
        unsafe { std::ffi::CStr::from_ptr(app_bundle_id) }
            .to_string_lossy()
            .into_owned()
    };
    // SAFETY: `engine` is non-null and confined to `@MainActor` on the Swift side.
    let eng = unsafe { &mut *engine };
    eng.inner.create_session(bundle)
}

/// Destroys `session_id` when an `IMKInputController` deactivates.
///
/// # Safety
///
/// `engine` must be null or a valid pointer from [`chime_engine_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_session_destroy(engine: *mut ChimeEngineHandle, session_id: u64) {
    if !engine.is_null() {
        // SAFETY: `engine` is valid and non-null.
        let eng = unsafe { &mut *engine };
        eng.inner.destroy_session(session_id);
    }
}

/// Resets active preedit state in `session_id`.
///
/// # Safety
///
/// `engine` must be null or a valid pointer from [`chime_engine_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_reset_session(engine: *mut ChimeEngineHandle, session_id: u64) {
    if !engine.is_null() {
        // SAFETY: `engine` is valid and non-null.
        let eng = unsafe { &mut *engine };
        eng.inner.reset_session(session_id);
    }
}

/// Toggles inline English mode on a clean `< 250ms` single Left-Shift tap, returning a JSON
/// snapshot.
///
/// # Safety
///
/// `engine` must be a valid pointer from [`chime_engine_new`]. Returned string must be freed via
/// [`chime_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_toggle_english(
    engine: *mut ChimeEngineHandle,
    session_id: u64,
) -> *mut c_char {
    if engine.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `engine` is valid and non-null.
    let eng = unsafe { &mut *engine };
    let resp = eng.inner.toggle_inline_english(session_id);
    encode_response_json(&resp)
}

/// Processes a single key event synchronously and returns a newly allocated JSON snapshot string.
///
/// # Safety
///
/// `engine` must be a valid pointer from [`chime_engine_new`]. Returned string must be freed via
/// [`chime_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_process_key(
    engine: *mut ChimeEngineHandle,
    session_id: u64,
    ascii_char: u8,
    key_code: u16,
    shift: bool,
    has_command_or_ctrl: bool,
    secure_input: bool,
) -> *mut c_char {
    if engine.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `engine` is valid and non-null.
    let eng = unsafe { &mut *engine };
    let resp = eng.inner.process_key(
        session_id,
        KeyEvent {
            ascii_char,
            key_code,
            shift,
            has_command_or_ctrl,
            secure_input,
        },
        current_epoch_secs(),
    );
    encode_response_json(&resp)
}

/// Selects candidate `index` (0-based) and returns a newly allocated JSON snapshot string.
///
/// # Safety
///
/// `engine` must be a valid pointer from [`chime_engine_new`]. Returned string must be freed via
/// [`chime_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_select_candidate(
    engine: *mut ChimeEngineHandle,
    session_id: u64,
    index: usize,
) -> *mut c_char {
    if engine.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `engine` is valid and non-null.
    let eng = unsafe { &mut *engine };
    let resp = eng
        .inner
        .select_candidate(session_id, index, current_epoch_secs());
    encode_response_json(&resp)
}

/// Frees a C string returned by [`chime_process_key`], [`chime_select_candidate`], or
/// [`chime_toggle_english`].
///
/// # Safety
///
/// `ptr` must be null or a pointer previously returned by one of the `chime_*` functions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        // SAFETY: `ptr` was allocated via `CString::into_raw`.
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}
