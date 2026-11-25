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
void chime_engine_free(ChimeEngineHandle *engine);

uint64_t chime_session_create(ChimeEngineHandle *engine, const char *app_bundle_id);
void chime_session_destroy(ChimeEngineHandle *engine, uint64_t session_id);
void chime_reset_session(ChimeEngineHandle *engine, uint64_t session_id);

char *chime_toggle_english(ChimeEngineHandle *engine, uint64_t session_id);
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
void chime_string_free(char *ptr);

#ifdef __cplusplus
}
#endif

#endif /* CHIME_FFI_H */
