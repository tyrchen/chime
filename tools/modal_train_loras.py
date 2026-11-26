#!/usr/bin/env python3
"""
Modal H100 Multi-LoRA Trainer for the 28-layer `Qwen3-0.6B` (`Qwen3-14B` distilled) base model.

Trains Rank-8 FP16 LoRA adapters (`classics_lora.safetensors` and `tech_lora.safetensors`) directly
on top of `/vol/distill_28l_q14b/student_fp16.safetensors` (~90s total on H100, ~$0.10).

Usage:
    modal run tools/modal_train_loras.py
"""

from __future__ import annotations

import json
import pathlib

import modal

APP_NAME = "chime-train-loras-28l-h100"
VOL_NAME = "chime-train-vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "datasets>=3.2.0",
        "huggingface_hub>=0.28.0",
        "safetensors>=0.5.0",
        "numpy>=2.0.0",
    )
    .add_local_file("tools/colab_train_loras.py", remote_path="/root/lora_lib.py")
    .add_local_file("dist/classics_lora_corpus.txt", remote_path="/root/classics_lora_corpus.txt")
    .add_local_file("dist/tech_lora_corpus.txt", remote_path="/root/tech_lora_corpus.txt")
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    cpu=4.0,
    memory=32768,
    timeout=1200,
    volumes={"/vol": vol},
)
def train_official_loras_28l(steps: int = 360, batch_size: int = 24, rank: int = 8) -> str:
    import os
    import sys

    os.environ["HF_HOME"] = "/vol/hf_cache"
    sys.path.insert(0, "/root")

    import torch
    from safetensors.torch import load_file

    import lora_lib

    vol.reload()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    base_dir = pathlib.Path("/vol/distill_28l_q14b")
    out_dir = pathlib.Path("/vol/loras_28l_q14b")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    vocab_strings = json.loads((base_dir / "vocab.json").read_text(encoding="utf-8"))
    st = load_file(str(base_dir / "student_fp16.safetensors"))
    embed_pruned = st["model.embed_tokens.weight"].contiguous()

    model = lora_lib.Qwen3FrozenBaseWithLora(
        st, embed_pruned, cfg, rank=rank, compute_dtype=torch.bfloat16
    ).to(device)
    cos, sin = lora_lib.build_rope_cache(256, int(cfg["head_dim"]), float(cfg["rope_theta"]), device)
    tok = lora_lib.GreedyCompactTokenizer(vocab_strings)

    # 1. Classics & Poetry LoRA
    c_lines = [
        ln.strip()
        for ln in pathlib.Path("/root/classics_lora_corpus.txt").read_text(encoding="utf-8").splitlines()
        if len(ln.strip()) >= 8
    ]
    c_blocks = lora_lib.tokenize_into_blocks(c_lines, tok, seq_len=96)
    lora_lib.train_one_lora(
        "classics-28l",
        model,
        c_blocks,
        cos,
        sin,
        out_dir / "classics_lora.safetensors",
        steps=steps,
        batch_size=batch_size,
        lr=3.0e-4,
        device=device,
    )

    # 2. Tech & Developer Bilingual LoRA
    t_seed = [
        ln.strip()
        for ln in pathlib.Path("/root/tech_lora_corpus.txt").read_text(encoding="utf-8").splitlines()
        if len(ln.strip()) >= 8
    ]
    t_lines = lora_lib.augment_tech_corpus_from_hf(t_seed, target_chars=1_200_000)
    t_blocks = lora_lib.tokenize_into_blocks(t_lines, tok, seq_len=96)
    lora_lib.train_one_lora(
        "tech-28l",
        model,
        t_blocks,
        cos,
        sin,
        out_dir / "tech_lora.safetensors",
        steps=steps,
        batch_size=batch_size,
        lr=3.0e-4,
        device=device,
    )

    vol.commit()
    return f"DONE 28L LoRAs -> {out_dir}"


@app.local_entrypoint()
def main() -> None:
    res = train_official_loras_28l.remote()
    print(res)
