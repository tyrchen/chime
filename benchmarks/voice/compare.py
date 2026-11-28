"""Reproducible macOS CPU backend probe; see README.md. No network at inference time."""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx as so
from opencc import OpenCC

SR = 16000
CC = OpenCC("t2s")
TEXT = {
    "short1": "打开备忘录。",
    "short2": "帮我设置明天早上的闹钟。",
    "short3": "下午三点开会，记得带上电脑。",
    "mix1": "请在 GitHub 上检查 Rust 项目的代码。",
    "mix2": "打开 Safari 浏览器，搜索 Python 教程。",
    "mix3": "这个 API 返回的是 JSON，不是 HTML。",
    "long": "今天早上我去菜市场买菜，发现卖水果的摊位比平时多了好几个。我先挑了几个苹果，又买了一把青菜，准备晚上回家做饭。走到路口的时候开始下小雨，我就站在屋檐下面等了一会儿。等雨停了，我才想起来还没买鸡蛋，于是又沿着原路走回去。回家以后我把东西放进冰箱，给朋友发了一条消息，约他周末一起去公园散步。",
}


def normalize(s):
    return "".join(
        c
        for c in CC.convert(unicodedata.normalize("NFKC", s)).casefold()
        if c.isalnum()
    )


def edits(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def prepare(root):
    for k, text in TEXT.items():
        aiff = root / (k + ".aiff")
        subprocess.run(
            ["say", "-v", "Tingting", "-r", "195", "-o", aiff, text], check=True
        )
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                aiff,
                "-ac",
                "1",
                "-ar",
                str(SR),
                "-c:a",
                "pcm_s16le",
                root / (k + ".wav"),
            ],
            check=True,
        )


def run(args):
    root, name = args.root, args.model
    start = time.monotonic()
    kw = {
        "tokens": str(root / name / "tokens.txt"),
        "num_threads": 2,
        "provider": "cpu",
    }
    if name == "sense":
        rec = so.OfflineRecognizer.from_sense_voice(
            model=str(root / name / "model.int8.onnx"),
            language="zh",
            use_itn=True,
            **kw,
        )
    else:
        rec = so.OnlineRecognizer.from_paraformer(
            encoder=str(root / name / "encoder.int8.onnx"),
            decoder=str(root / name / "decoder.int8.onnx"),
            **kw,
        )
    load = round(1000 * (time.monotonic() - start))

    def offline(x):
        stream = rec.create_stream()
        stream.accept_waveform(SR, x)
        rec.decode_stream(stream)
        return stream.result.text

    for key in args.keys or TEXT:
        with wave.open(str(root / (key + ".wav"))) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2)
            pcm = (
                np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(
                    np.float32
                )
                / 32768
            )
        # A release shortly after the last audible sound, rather than benchmarking TTS trailing silence.
        voiced = np.flatnonzero(np.abs(pcm) > 0.008)
        pcm = pcm[: voiced[-1] + 1 + int(0.12 * SR)]
        events = []
        changes = 0
        prev = ""
        finalized = ""
        segment_start = 0
        last_voice = 0
        speech = 0
        stream = rec.create_stream() if name == "para" else None

        def show(text):
            nonlocal changes, prev
            norm = normalize(text)
            if norm != prev:
                changes += len(prev) - len(os.path.commonprefix([prev, norm]))
                events.append([round((time.monotonic() - t0) * 1000), text])
                prev = norm

        def decode_online(buf):
            stream.accept_waveform(SR, buf)
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            return rec.get_result(stream)

        # 50 ms capture blocks paced by an independent audio clock (includes inference backlog).
        t0 = time.monotonic()
        clock0 = time.clock_gettime(time.CLOCK_MONOTONIC)
        cpu0 = time.process_time()
        next_preview = t0
        for a in range(0, len(pcm), 800):
            b = min(a + 800, len(pcm))
            chunk = pcm[a:b]
            time.sleep(max(0, t0 + b / SR - time.monotonic()))
            if name == "para":
                show(decode_online(chunk))
            else:
                if np.sqrt(np.mean(chunk**2)) > 0.006:
                    last_voice = b
                    speech += len(chunk)
                cut = (
                    b - last_voice >= 0.65 * SR and b - segment_start >= 1.2 * SR
                ) or b - segment_start >= 24 * SR
                if speech >= 0.24 * SR and (cut or time.monotonic() >= next_preview):
                    next_preview = time.monotonic() + 0.7
                    hyp = offline(pcm[segment_start:b])
                    show(finalized + hyp)
                    if cut:
                        finalized += hyp
                        segment_start = b
                        speech = 0
        if name == "para":
            # Flush right context instantly (not 0.8 s of wall-clock waiting).
            stream.accept_waveform(SR, np.zeros(int(0.8 * SR), dtype=np.float32))
            stream.input_finished()
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            final = rec.get_result(stream)
        else:
            final = finalized + (
                offline(pcm[segment_start:]) if speech >= 0.24 * SR else ""
            )
        end = time.monotonic()
        show(final)
        ref = normalize(TEXT[key])
        print(
            json.dumps(
                {
                    "key": key,
                    "model": name,
                    "mode": args.mode,
                    "load_ms": load,
                    "audio_s": round(len(pcm) / SR, 3),
                    "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                    "first_ms": events[0][0] if events else None,
                    "tail_ms": round((end - t0 - len(pcm) / SR) * 1000),
                    "deleted": changes,
                    "errors": edits(prev, ref),
                    "chars": len(ref),
                    "reference": TEXT[key],
                    "final": final,
                    "events": events,
                    "cpu_s": round(time.process_time() - cpu0, 3),
                    "peak_mib": round(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
                    ),
                    "start": clock0,
                    "end": clock0 + (end - t0),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("model", choices=["prepare", "sense", "para"])
    p.add_argument("keys", nargs="*")
    p.add_argument("--mode", default="solo")
    args = p.parse_args()
    if args.model == "prepare":
        prepare(args.root)
    else:
        run(args)
