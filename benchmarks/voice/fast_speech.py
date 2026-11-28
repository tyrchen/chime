"""Compare final text on identical local audio, without starting IMK or recording a mic."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import sherpa_onnx as so
from compare import SR, edits, normalize

# Include polarity, paired alternatives and plain everyday speech (not just the quote).
SENTENCES = [
    "曾经有一段真挚的感情摆在我面前，我没有好好珍惜。",
    "我没有同意取消明天的会议，你别发通知。",
    "我们要好好珍惜这次机会，不要再错过了。",
    "不是今天下午，是明天下午，你不用现在过去。",
    "等雨停了我们再出去买菜，冰箱里还有鸡蛋。",
    "你先把桌上的文件收起来，我一会儿回来拿。",
]


def load_audio(path):
    data = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            path,
            "-ar",
            str(SR),
            "-ac",
            "1",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    x = np.frombuffer(data, dtype="<f4")
    if not len(x):
        raise ValueError("empty audio")
    return x


def fixtures(directory):
    path = directory / "synthetic.aiff"
    for voice in ["Tingting", "Eddy (Chinese (China mainland))"]:
        for rate in [195, 285, 375]:
            for ref in SENTENCES:
                subprocess.run(
                    ["say", "-v", voice, "-r", str(rate), "-o", path, ref], check=True
                )
                yield {"voice": voice, "say_rate": rate}, ref, load_audio(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, help="local model directories para/ and sense/")
    p.add_argument(
        "--audio", type=Path, help="use an existing local recording instead of TTS"
    )
    p.add_argument("--reference", help="expected transcript of --audio")
    args = p.parse_args()
    if bool(args.audio) != bool(args.reference):
        p.error("supply --audio and --reference together")
    para, sense = args.root / "para", args.root / "sense"
    online = so.OnlineRecognizer.from_paraformer(
        encoder=str(para / "encoder.int8.onnx"),
        decoder=str(para / "decoder.int8.onnx"),
        tokens=str(para / "tokens.txt"),
        num_threads=2,
        provider="cpu",
    )
    offline = so.OfflineRecognizer.from_sense_voice(
        model=str(sense / "model.int8.onnx"),
        tokens=str(sense / "tokens.txt"),
        language="zh",
        use_itn=True,
        num_threads=2,
        provider="cpu",
    )
    with tempfile.TemporaryDirectory() as temp:
        inputs = (
            [({}, args.reference, load_audio(args.audio))]
            if args.audio
            else fixtures(Path(temp))
        )
        for meta, ref, x in inputs:
            # Same PCM, no wall-clock sleeps: accuracy after flush, not a capture latency test.
            stream = online.create_stream()
            for pos in range(0, len(x), 800):
                stream.accept_waveform(SR, x[pos : pos + 800])
                while online.is_ready(stream):
                    online.decode_stream(stream)
            stream.accept_waveform(SR, np.zeros(12800, dtype=np.float32))
            stream.input_finished()
            while online.is_ready(stream):
                online.decode_stream(stream)
            a = online.get_result(stream)
            t = time.perf_counter()
            batch = offline.create_stream()
            batch.accept_waveform(SR, x)
            offline.decode_stream(batch)
            b = batch.result.text
            print(
                json.dumps(
                    {
                        **meta,
                        "reference": ref,
                        "seconds": round(len(x) / SR, 3),
                        "pcm_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                        "para": a,
                        "sense": b,
                        "sense_ms_loaded": round(1000 * (time.perf_counter() - t)),
                        "chars": len(normalize(ref)),
                        "para_errors": edits(normalize(ref), normalize(a)),
                        "sense_errors": edits(normalize(ref), normalize(b)),
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
