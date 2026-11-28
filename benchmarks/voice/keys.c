// Headless keyboard contention probe. Never opens IMK or uses the real user store.
#include "chime_ffi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
int main(int argc, char **argv) {
    if (argc != 3) return 1;
    ChimeEngineHandle *h = chime_engine_new_with_dirs(argv[1], argv[2]);
    char *status = chime_get_status_json(h);
    if (!status || !strstr(status, "\"neural_loaded\":true")) return 2;
    chime_string_free(status);
    uint64_t s = chime_session_create(h, "chime.benchmark");
    const char *phrases[] = {"jintiantianqihenhao", "womengongzuo", "xiawukaihuitongzhi"};
    setlinebuf(stdout);
    for (unsigned i = 0;; ++i) {
        chime_reset_session(h, s);
        for (const char *p = phrases[i%3]; *p; ++p) {
            double start = now();
            char *r = chime_process_key(h, s, *p, 0, false, false, false);
            double end = now();
            chime_string_free(r);
            usleep(60000);
            r = chime_poll_async_rerank(h, s);
            printf("%.6f,%.3f,%d\n", end, (end-start)*1000, r != NULL);
            chime_string_free(r);
        }
    }
}
