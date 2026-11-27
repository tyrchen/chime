/* Chime's per-dictation helper. No microphone, network, Python or files other
 * than the model in the working directory. Both pipe endpoints are arm64 LE.
 * Request: uint32 sample count + float32 PCM. Zero means flush/final.
 * Reply: uint32 byte count + cumulative UTF-8 text (including empty text). */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "sherpa-onnx/c-api/c-api.h"

int main(void) {
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
