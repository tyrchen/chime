/* Chime's per-dictation helper. No microphone, network, Python or files other
 * than the model in the working directory. Both pipe endpoints are arm64 LE.
 * Request: uint32 sample count + float32 PCM. Zero means flush/final.
 * Reply: uint32 byte count + cumulative UTF-8 text (including empty text). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sherpa-onnx/c-api/c-api.h"

/* Text-only second pass. Keep transcripts off argv, disk, and stderr. EOF ends
 * the request. Model failure is isolated from ASR; the host keeps its original. */
static int punctuate(void) {
    char text[32769];
    size_t n = fread(text, 1, sizeof(text), stdin);
    if (!n || n == sizeof(text) || ferror(stdin)) return 2;
    text[n] = '\0';
    SherpaOnnxOfflinePunctuationConfig cfg = {0};
    cfg.model.ct_transformer = "punct.int8.onnx";
    cfg.model.num_threads = 2;
    cfg.model.provider = "cpu";
    const SherpaOnnxOfflinePunctuation *p = SherpaOnnxCreateOfflinePunctuation(&cfg);
    if (!p) return 1;
    const char *result = SherpaOfflinePunctuationAddPunct(p, text);
    int ok = result && fputs(result, stdout) >= 0 && fflush(stdout) == 0;
    if (result) SherpaOfflinePunctuationFreeText(result);
    SherpaOnnxDestroyOfflinePunctuation(p);
    return ok ? 0 : 1;
}

/* Optional final pass in an independent process. PCM never leaves anonymous pipes.
 * Do not run unbounded offline attention on minutes of speech. */
static int sense_final(void) {
    uint32_t n = 0;
    if (fread(&n, sizeof(n), 1, stdin) != 1 || n < 16000 || n > 45 * 16000) return 2;
    float *pcm = malloc((size_t)n * sizeof(*pcm));
    if (!pcm || fread(pcm, sizeof(*pcm), n, stdin) != n) { free(pcm); return 2; }
    SherpaOnnxOfflineRecognizerConfig cfg = {0};
    cfg.feat_config.sample_rate = 16000;
    cfg.feat_config.feature_dim = 80;
    cfg.model_config.sense_voice.model = "model.int8.onnx";
    cfg.model_config.sense_voice.language = "zh";
    cfg.model_config.sense_voice.use_itn = 1;
    cfg.model_config.tokens = "tokens.txt";
    cfg.model_config.num_threads = 2;
    cfg.model_config.provider = "cpu";
    cfg.decoding_method = "greedy_search";
    const SherpaOnnxOfflineRecognizer *rec = SherpaOnnxCreateOfflineRecognizer(&cfg);
    const SherpaOnnxOfflineStream *s = rec ? SherpaOnnxCreateOfflineStream(rec) : NULL;
    int ok = 0;
    if (s) {
        SherpaOnnxAcceptWaveformOffline(s, 16000, pcm, (int32_t)n);
        SherpaOnnxDecodeOfflineStream(rec, s);
        const char *json = SherpaOnnxGetOfflineStreamResultAsJson(s);
        ok = json && fputs(json, stdout) >= 0 && fflush(stdout) == 0;
        if (json) SherpaOnnxDestroyOfflineStreamResultJson(json);
        SherpaOnnxDestroyOfflineStream(s);
    }
    if (rec) SherpaOnnxDestroyOfflineRecognizer(rec);
    free(pcm);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--sense-final") == 0) return sense_final();
    if (argc == 2 && strcmp(argv[1], "--punctuate") == 0) return punctuate();
    if (argc != 1) return 2;
    SherpaOnnxOnlineRecognizerConfig config = {0};
    config.feat_config.sample_rate = 16000;
    config.feat_config.feature_dim = 80;
    config.model_config.paraformer.encoder = "encoder.int8.onnx";
    config.model_config.paraformer.decoder = "decoder.int8.onnx";
    config.model_config.tokens = "tokens.txt";
    config.model_config.provider = "cpu";
    config.model_config.num_threads = 2;
    config.decoding_method = "greedy_search";
    const SherpaOnnxOnlineRecognizer *rec = SherpaOnnxCreateOnlineRecognizer(&config);
    if (!rec) return 1;
    const SherpaOnnxOnlineStream *stream = SherpaOnnxCreateOnlineStream(rec);
    if (!stream) return 1;
    float pcm[16000];
    uint32_t n;
    while (fread(&n, sizeof(n), 1, stdin) == 1) {
        if (n > 16000 || fread(pcm, sizeof(float), n, stdin) != n) return 2;
        if (n) {
            SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, pcm, n);
        } else {
            // Supply 0.8s of right context without waiting for more mic input.
            memset(pcm, 0, sizeof(pcm));
            SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, pcm, 12800);
            SherpaOnnxOnlineStreamInputFinished(stream);
        }
        while (SherpaOnnxIsOnlineStreamReady(rec, stream))
            SherpaOnnxDecodeOnlineStream(rec, stream);
        const SherpaOnnxOnlineRecognizerResult *r = SherpaOnnxGetOnlineStreamResult(rec, stream);
        if (!r || !r->text) return 1;
        uint32_t len = (uint32_t)strlen(r->text);
        if (fwrite(&len, sizeof(len), 1, stdout) != 1 ||
            fwrite(r->text, 1, len, stdout) != len || fflush(stdout)) return 2;
        SherpaOnnxDestroyOnlineRecognizerResult(r);
        if (!n) break;
    }
    SherpaOnnxDestroyOnlineStream(stream);
    SherpaOnnxDestroyOnlineRecognizer(rec);
    return 0;
}
