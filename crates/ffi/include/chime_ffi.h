#ifndef CHIME_FFI_H
#define CHIME_FFI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ChimeEngineHandle ChimeEngineHandle;

ChimeEngineHandle *chime_engine_new(void);
ChimeEngineHandle *chime_engine_new_with_dirs(
    const char *app_support_dir,
    const char *bundle_resources_dir
);
void chime_engine_free(ChimeEngineHandle *engine);

uint64_t chime_session_create(ChimeEngineHandle *engine, const char *app_bundle_id);
void chime_session_destroy(ChimeEngineHandle *engine, uint64_t session_id);
void chime_reset_session(ChimeEngineHandle *engine, uint64_t session_id);
char *chime_commit_composition(ChimeEngineHandle *engine, uint64_t session_id);
void chime_sync_context(
    ChimeEngineHandle *engine,
    uint64_t session_id,
    const char *surrounding_left
);

char *chime_toggle_english(ChimeEngineHandle *engine, uint64_t session_id);
char *chime_toggle_punctuation(ChimeEngineHandle *engine, uint64_t session_id);
char *chime_toggle_pangu_spacing(ChimeEngineHandle *engine, uint64_t session_id);
char *chime_toggle_grid(ChimeEngineHandle *engine, uint64_t session_id);
char *chime_process_key(
    ChimeEngineHandle *engine,
    uint64_t session_id,
    uint8_t ascii_char,
    uint16_t key_code,
    bool shift,
    bool has_command_or_ctrl,
    bool secure_input
);
char *chime_select_candidate(ChimeEngineHandle *engine, uint64_t session_id, size_t index);
char *chime_poll_async_rerank(ChimeEngineHandle *engine, uint64_t session_id);

int64_t chime_import_asset(ChimeEngineHandle *engine, const char *asset_path);
bool chime_remove_pack(ChimeEngineHandle *engine, const char *pack_id);
void chime_reload_assets(ChimeEngineHandle *engine);
void chime_clear_user_habits(ChimeEngineHandle *engine);
char *chime_get_config_json(ChimeEngineHandle *engine);
bool chime_set_config_json(ChimeEngineHandle *engine, const char *config_json);
char *chime_get_status_json(ChimeEngineHandle *engine);

void chime_string_free(char *ptr);

/* Hold-to-talk voice dictation. Thread-safe: audio may be pushed from the AVAudioEngine tap. */
typedef struct ChimeVoiceHandle ChimeVoiceHandle;

ChimeVoiceHandle *chime_voice_new(void);
void chime_voice_free(ChimeVoiceHandle *voice);
/* Returns NULL on success, else an error message to free with chime_string_free. */
char *chime_voice_start(
    ChimeVoiceHandle *voice,
    const char *settings_json,
    const char *api_key,
    uint32_t sample_rate
);
/* Offline streaming backend: app Contents path, fixed 16 kHz mono f32 input. */
char *chime_voice_start_bundled(ChimeVoiceHandle *voice, const char *contents, bool preview, uint64_t seconds);
void chime_voice_push_pcm(ChimeVoiceHandle *voice, const float *samples, size_t len);
void chime_voice_stop(ChimeVoiceHandle *voice);
void chime_voice_cancel(ChimeVoiceHandle *voice);
void chime_voice_end(ChimeVoiceHandle *voice);
/* JSON snapshot of the running session, or NULL when idle. */
char *chime_voice_snapshot(ChimeVoiceHandle *voice);
/* Blocking connectivity probe; call off the main thread. Returns JSON. */
char *chime_voice_test_endpoint(const char *settings_json, const char *api_key);

#ifdef __cplusplus
}
#endif

#endif /* CHIME_FFI_H */
