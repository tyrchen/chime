"""Build-time only: stage the pinned arm64 ASR runtime and INT8 model in Chime.app."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

MODEL = "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en/resolve/8e40c43232a1c5c66c82111efc5820d3accca11b"
FILES = {
    "encoder.int8.onnx": "81a70226a8934e6ed92aa1d4fc486b428b5398e2f2619ed4897b7294cab90e9a",
    "decoder.int8.onnx": "f3cca9f77bb9d93c8fcbfb63ae617b6b1ee96818df3aa3b151c40658fe38594f",
    "tokens.txt": "59aba8873a2ed1e122c25fee421e25f283b63290efbde85c1f01a853d83cb6e6",
}
WHEEL = "https://files.pythonhosted.org/packages/f1/36/0c887684a033f6de0a5e3c0e7a244330fd1fb1a89a9917da70aff4ff6e59/sherpa_onnx_core-2.0.3-py3-none-macosx_11_0_arm64.whl"
WHEEL_SHA = "ca19ab0ae8c78b9a0a987ebd1aa69b73e3b7490733646e831db46260dab85316"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as data:
        for block in iter(lambda: data.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(cache, name, url, sha):
    dest = cache / name
    if dest.is_file() and digest(dest) == sha:
        return dest
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temp:
        path = Path(temp.name)
    try:
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "3", "--output", path, url],
            check=True,
        )
        if digest(path) != sha:
            raise ValueError(f"SHA-256 mismatch: {name}")
        os.replace(path, dest)
    finally:
        path.unlink(missing_ok=True)
    return dest


def bundle(contents, cache):
    src = Path(__file__).resolve().parents[1] / "VoiceHelper"
    model = contents / "Resources/voice"
    libs = contents / "Frameworks"
    helpers = contents / "Helpers"
    for directory in [cache, model, libs, helpers]:
        directory.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.items():
        shutil.copyfile(fetch(cache, name, f"{MODEL}/{name}", sha), model / name)
    shutil.copytree(src / "licenses", model / "licenses", dirs_exist_ok=True)
    with (
        zipfile.ZipFile(fetch(cache, "core.whl", WHEEL, WHEEL_SHA)) as archive,
        tempfile.TemporaryDirectory() as temp,
    ):
        # Take only the native C ABI and its matching header, never install Python into the app.
        archive.extract("sherpa_onnx/include/sherpa-onnx/c-api/c-api.h", temp)
        for name in ["libonnxruntime.dylib", "libsherpa-onnx-c-api.dylib"]:
            with (
                archive.open(f"sherpa_onnx/lib/{name}") as lib,
                (libs / name).open("wb") as out,
            ):
                shutil.copyfileobj(lib, out)
        subprocess.run(
            [
                "xcrun",
                "clang",
                "-O2",
                "-Wall",
                "-Werror",
                "-arch",
                "arm64",
                "-mmacosx-version-min=14.0",
                "-I",
                Path(temp) / "sherpa_onnx/include",
                src / "main.c",
                "-L",
                libs,
                "-lsherpa-onnx-c-api",
                "-Wl,-rpath,@executable_path/../Frameworks",
                "-o",
                helpers / "ChimeVoice",
            ],
            check=True,
        )


if __name__ == "__main__":
    bundle(Path(sys.argv[1]), Path(sys.argv[2]))
