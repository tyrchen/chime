"""Measure current macOS RSS (not ru_maxrss) after dropping the last recognizer reference."""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx as so

root, model = Path(sys.argv[1]), sys.argv[2]


def rss():
    return round(
        int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]))
        / 1024
    )


def load_decode():
    kw = {
        "num_threads": 2,
        "tokens": str(root / model / "tokens.txt"),
        "provider": "cpu",
    }
    if model == "sense":
        rec = so.OfflineRecognizer.from_sense_voice(
            model=str(root / model / "model.int8.onnx"),
            language="zh",
            use_itn=True,
            **kw,
        )
    else:
        rec = so.OnlineRecognizer.from_paraformer(
            encoder=str(root / model / "encoder.int8.onnx"),
            decoder=str(root / model / "decoder.int8.onnx"),
            **kw,
        )
    s = rec.create_stream()
    s.accept_waveform(16000, np.zeros(32000, dtype=np.float32))
    if model == "para":
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
    else:
        rec.decode_stream(s)
    return rss()


before = rss()
active = load_decode()
gc.collect()
time.sleep(1)
print(
    json.dumps(
        {
            "model": model,
            "before_mib": before,
            "active_mib": active,
            "after_drop_1s_mib": rss(),
        }
    )
)
