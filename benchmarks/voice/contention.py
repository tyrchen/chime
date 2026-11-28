"""Run real Rust/Metal keyboard work in a separate process and compare overlapping periods."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

root, resources = map(Path, sys.argv[1:])
base = Path(__file__).resolve().parent


def clock():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


with tempfile.TemporaryDirectory() as tmp, (root / "keys.csv").open("w") as out:
    p = subprocess.Popen(
        [root / "keys", tmp, resources], stdout=out, env=dict(os.environ, HOME=tmp)
    )
    try:
        deadline = clock() + 45
        while not (root / "keys.csv").stat().st_size:
            if p.poll() is not None or clock() > deadline:
                raise RuntimeError("keyboard probe failed to initialize")
            time.sleep(0.1)
        time.sleep(3)
        start = clock()
        time.sleep(30)
        windows = {"baseline": (start, clock())}
        for name in ("para", "sense"):
            line = subprocess.check_output(
                [
                    sys.executable,
                    base / "compare.py",
                    root,
                    name,
                    "long",
                    "--mode",
                    "concurrent",
                ],
                text=True,
            )
            data = json.loads(line)
            (base / (name + "-concurrent.jsonl")).write_text(line)
            windows[name] = (data["start"], data["end"])
        time.sleep(0.2)
    finally:
        p.terminate()
        p.wait()
keys = np.loadtxt(root / "keys.csv", delimiter=",")
summary = {}
for k, (a, b) in windows.items():
    group = keys[(keys[:, 0] >= a) & (keys[:, 0] <= b)]
    v = group[:, 1]
    summary[k] = {
        "n": len(v),
        "p50_ms": np.median(v),
        "p95_ms": np.percentile(v, 95),
        "p99_ms": np.percentile(v, 99),
        "max_ms": max(v),
        "async_updates": int(sum(group[:, 2])),
        "over_16ms": int(sum(v > 16)),
    }
print(json.dumps(summary, indent=2))
