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

#ifdef __cplusplus
}
#endif

#endif /* CHIME_FFI_H */
