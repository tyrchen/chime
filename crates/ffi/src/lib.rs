//! Narrow C-ABI hourglass waist bridging `chime-core` to Swift 6 (`@MainActor` `AppKit`/IMK).

use std::{
    ffi::{CStr, CString, c_char},
    path::PathBuf,
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

fn encode_json<T: serde::Serialize>(val: &T) -> *mut c_char {
    let json = serde_json::to_string(val).unwrap_or_else(|_| "{}".to_string());
    CString::new(json).map_or(std::ptr::null_mut(), CString::into_raw)
}

unsafe fn opt_cstr_to_path(ptr: *const c_char) -> Option<PathBuf> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: Caller passes either null or a valid NUL-terminated UTF-8 string.
    let s = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s.into_owned()))
    }
}

/// Allocates an in-memory [`ChimeEngineHandle`] configured with default settings.
#[unsafe(no_mangle)]
pub extern "C" fn chime_engine_new() -> *mut ChimeEngineHandle {
    Box::into_raw(Box::new(ChimeEngineHandle {
        inner: ChimeEngine::new(ChimeConfig::default()),
    }))
}

/// Allocates a persistent [`ChimeEngineHandle`] backed by `app_support_dir` (`config.toml`,
/// `user_store.db`, `dicts/`, `neural/`) and optional `bundle_resources_dir/core.imedict`.
///
/// # Safety
///
/// Both pointers must be either null or valid NUL-terminated UTF-8 C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_engine_new_with_dirs(
    app_support_dir: *const c_char,
    bundle_resources_dir: *const c_char,
) -> *mut ChimeEngineHandle {
    // SAFETY: Validated by `opt_cstr_to_path` precondition.
    let support = unsafe { opt_cstr_to_path(app_support_dir) };
    // SAFETY: Validated by `opt_cstr_to_path` precondition.
    let res = unsafe { opt_cstr_to_path(bundle_resources_dir) };

    let inner = if let Some(dir) = support {
        ChimeEngine::with_data_dir(&dir, res.as_deref()).unwrap_or_default()
    } else {
        ChimeEngine::default()
    };
    Box::into_raw(Box::new(ChimeEngineHandle { inner }))
}

/// Frees a [`ChimeEngineHandle`] previously created by [`chime_engine_new`] or
/// [`chime_engine_new_with_dirs`].
///
/// # Safety
///
/// `engine` must be null or a valid un-freed pointer returned by `chime_engine_new*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_engine_free(engine: *mut ChimeEngineHandle) {
    if !engine.is_null() {
        // SAFETY: Allocated via `Box::into_raw`.
        unsafe {
            drop(Box::from_raw(engine));
        }
    }
}

/// Creates a lightweight session for `app_bundle_id` and returns its `session_id`.
///
/// # Safety
///
/// `engine` must be valid; `app_bundle_id` may be null or a valid NUL-terminated UTF-8 C string.
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
        // SAFETY: Valid NUL-terminated string when non-null.
        unsafe { CStr::from_ptr(app_bundle_id) }
            .to_string_lossy()
            .into_owned()
    };
    // SAFETY: `engine` is non-null and isolated to `@MainActor`.
    let eng = unsafe { &mut *engine };
    eng.inner.create_session(bundle)
}

/// Destroys `session_id` when an `IMKInputController` deactivates.
///
/// # Safety
///
/// `engine` must be null or a valid pointer.
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
/// `engine` must be null or a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_reset_session(engine: *mut ChimeEngineHandle, session_id: u64) {
    if !engine.is_null() {
        // SAFETY: `engine` is valid and non-null.
        let eng = unsafe { &mut *engine };
        eng.inner.reset_session(session_id);
    }
}

/// Toggles inline English mode on a clean `< 250ms` single Left-Shift tap, returning JSON.
///
/// # Safety
///
/// `engine` must be valid. Returned string must be freed via [`chime_string_free`].
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
    let resp: EngineResponse = eng.inner.toggle_inline_english(session_id);
    encode_json(&resp)
}

/// Toggles Chinese full-width (`。，`) vs ASCII (`.,`) punctuation, persisting to `config.toml`.
///
/// # Safety
///
/// `engine` must be valid. Returned string must be freed via [`chime_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_toggle_punctuation(
    engine: *mut ChimeEngineHandle,
    session_id: u64,
) -> *mut c_char {
    if engine.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `engine` is valid and non-null.
    let eng = unsafe { &mut *engine };
    let resp: EngineResponse = eng.inner.toggle_punctuation(session_id);
    encode_json(&resp)
}

/// Toggles the Doubao-style `5x5` expanded candidate grid (`grid_expanded`).
///
/// # Safety
///
/// `engine` must be valid. Returned string must be freed via [`chime_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_toggle_grid(
    engine: *mut ChimeEngineHandle,
    session_id: u64,
) -> *mut c_char {
    if engine.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `engine` is valid and non-null.
    let eng = unsafe { &mut *engine };
    let resp: EngineResponse = eng.inner.toggle_grid_expanded(session_id);
    encode_json(&resp)
}

/// Processes a single key event synchronously and returns a newly allocated JSON snapshot string.
///
/// # Safety
///
/// `engine` must be valid. Returned string must be freed via [`chime_string_free`].
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
    encode_json(&resp)
}

/// Selects candidate `index` (0-based on active page) and returns a newly allocated JSON snapshot.
///
/// # Safety
///
/// `engine` must be valid. Returned string must be freed via [`chime_string_free`].
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
    encode_json(&resp)
}

/// Polls completed asynchronous Stage-2 neural reranks. Returns null if no update is ready,
/// or a JSON `EngineResponse` string if `session_id`'s candidates were updated.
///
/// # Safety
///
/// `engine` must be valid. Non-null return must be freed via [`chime_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_poll_async_rerank(
    engine: *mut ChimeEngineHandle,
    session_id: u64,
) -> *mut c_char {
    if engine.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `engine` is valid and non-null.
    let eng = unsafe { &mut *engine };
    eng.inner
        .poll_async_rerank(session_id)
        .map_or(std::ptr::null_mut(), |resp| encode_json(&resp))
}

/// Imports an external dictionary or `.imepack` path and hot-mounts it. Returns imported count, or
/// `-1` on error.
///
/// # Safety
///
/// `engine` and `asset_path` must be valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_import_asset(
    engine: *mut ChimeEngineHandle,
    asset_path: *const c_char,
) -> i64 {
    if engine.is_null() {
        return -1;
    }
    // SAFETY: Validated by `opt_cstr_to_path` precondition.
    let Some(path) = (unsafe { opt_cstr_to_path(asset_path) }) else {
        return -1;
    };
    // SAFETY: `engine` is valid and non-null.
    let eng = unsafe { &mut *engine };
    eng.inner
        .import_external_asset(&path)
        .ok()
        .and_then(|n| i64::try_from(n).ok())
        .unwrap_or(-1)
}

/// Reloads `config.toml`, dictionaries, and neural models from disk.
///
/// # Safety
///
/// `engine` must be null or a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_reload_assets(engine: *mut ChimeEngineHandle) {
    if !engine.is_null() {
        // SAFETY: `engine` is valid and non-null.
        let eng = unsafe { &mut *engine };
        eng.inner.reload_assets();
    }
}

/// Serializes the active [`ChimeConfig`] as JSON for the `SwiftUI` Settings window.
///
/// # Safety
///
/// `engine` must be valid. Returned string must be freed via [`chime_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_get_config_json(engine: *mut ChimeEngineHandle) -> *mut c_char {
    if engine.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `engine` is valid and non-null.
    let eng = unsafe { &*engine };
    encode_json(eng.inner.config())
}

/// Updates and saves [`ChimeConfig`] from a JSON string emitted by the `SwiftUI` Settings window.
///
/// # Safety
///
/// `engine` and `config_json` must be valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_set_config_json(
    engine: *mut ChimeEngineHandle,
    config_json: *const c_char,
) -> bool {
    if engine.is_null() || config_json.is_null() {
        return false;
    }
    // SAFETY: `config_json` is a valid NUL-terminated UTF-8 string.
    let raw = unsafe { CStr::from_ptr(config_json) }.to_string_lossy();
    let Ok(new_cfg) = serde_json::from_str::<ChimeConfig>(&raw) else {
        return false;
    };
    // SAFETY: `engine` is valid and non-null.
    let eng = unsafe { &mut *engine };
    eng.inner.update_config(new_cfg).is_ok()
}

/// Returns live [`chime_core::EngineStatus`] diagnostics as JSON.
///
/// # Safety
///
/// `engine` must be valid. Returned string must be freed via [`chime_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_get_status_json(engine: *mut ChimeEngineHandle) -> *mut c_char {
    if engine.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `engine` is valid and non-null.
    let eng = unsafe { &*engine };
    encode_json(&eng.inner.status())
}

/// Frees a C string returned by any `chime_*` FFI function.
///
/// # Safety
///
/// `ptr` must be null or a pointer previously returned by `chime_*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chime_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        // SAFETY: `ptr` was allocated via `CString::into_raw`.
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}
