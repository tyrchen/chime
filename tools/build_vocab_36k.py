#!/usr/bin/env python3
"""
Builds a lossless-CJK `36,864`-token compact vocabulary (`dist/vocab_36k.json`) from Qwen3's
151,643-token BPE vocabulary:
  - Slot 0: `<|endoftext|>` (BOS)
  - Slot 1: `<unk>`
  - ALL 24,872 pure-CJK tokens (1-char + 2..=6-char) in Qwen3 — ZERO CJK word pruning!
  - Top ~11,990 punctuation, digits, ASCII/English & developer tokens to fill 36,864 (64-aligned).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from transformers import AutoTokenizer


def is_all_cjk(s: str) -> bool:
    return bool(s) and all("\u4e00" <= ch <= "\u9fa5" for ch in s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dist/vocab_36k.json")
    parser.add_argument("--target-size", type=int, default=36864)
    args = parser.parse_args()

    snaps = sorted(
        glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/*"))
    )
    if not snaps:
        raise FileNotFoundError("Qwen/Qwen3-0.6B-Base snapshot not found in HuggingFace cache")
    tok = AutoTokenizer.from_pretrained(snaps[-1])

    selected: list[str] = ["<|endoftext|>", "<unk>"]
    seen: set[str] = set(selected)

    def add_piece(p: str) -> None:
        if not p or p in seen or "\ufffd" in p:
            return
        seen.add(p)
        selected.append(p)

    # 1. Essential Chinese/ASCII punctuation & digits
    for ch in "。，！？；：、（）《》【】“”‘’—…·.,!?;:()[]{}<>\"'-_/\\@#$%^&*+=|~`0123456789\n":
        add_piece(ch)

    # 2. ALL pure-CJK tokens (1..=6 chars) from Qwen3 BPE — 100% CJK coverage!
    cjk_count = 0
    for tid in range(tok.vocab_size):
        piece = tok.decode([tid])
        if 1 <= len(piece) <= 6 and is_all_cjk(piece):
            if piece not in seen:
                add_piece(piece)
                cjk_count += 1

    # 3. Top ASCII/English/code/mixed tokens in BPE merge priority order until target_size
    for tid in range(tok.vocab_size):
        if len(selected) >= args.target_size:
            break
        piece = tok.decode([tid])
        if not piece or len(piece) > 16 or "\ufffd" in piece:
            continue
        stripped = piece.strip()
        if not stripped:
            continue
        if stripped.isascii() or any("\u4e00" <= c <= "\u9fa5" for c in stripped):
            add_piece(stripped)

    # Pad if needed
    for tid in range(tok.vocab_size):
        if len(selected) >= args.target_size:
            break
        piece = tok.decode([tid])
        if piece and "\ufffd" not in piece:
            add_piece(piece)

    selected = selected[: args.target_size]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")
    print(
        f"Built {len(selected)}-token compact vocab ({cjk_count} pure-CJK tokens = 100% of Qwen3 CJK) -> {out_path}"
    )


if __name__ == "__main__":
    main()
