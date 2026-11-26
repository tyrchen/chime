#!/usr/bin/env python3
"""
Quantizes a Colab-distilled `student_fp16.safetensors` (`6L` or `8L`, `36,864` vocab) into a
ready-to-run native 4-bit MLX Metal bundle (`model.safetensors`, `config.json`, `vocab.json`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, help="Directory with student_fp16.safetensors, config.json, vocab.json")
    parser.add_argument("--out-dir", required=True, help="Output 4-bit MLX bundle directory")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((in_dir / "config.json").read_text(encoding="utf-8"))
    st = load_file(str(in_dir / "student_fp16.safetensors"))
    num_layers = int(cfg["num_hidden_layers"])

    mlx_quant: dict[str, mx.array] = {}
    embed = mx.array(st["model.embed_tokens.weight"].to(torch.float16).numpy())
    wq, sc, bi = mx.quantize(embed, group_size=args.group_size, bits=args.bits)
    mlx_quant["model.embed_tokens.weight"] = wq
    mlx_quant["model.embed_tokens.scales"] = sc
    mlx_quant["model.embed_tokens.biases"] = bi
    mlx_quant["model.norm.weight"] = mx.array(st["model.norm.weight"].to(torch.float16).numpy())

    for l_idx in range(num_layers):
        p = f"model.layers.{l_idx}"
        for suf in (
            "input_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
        ):
            mlx_quant[f"{p}.{suf}"] = mx.array(st[f"{p}.{suf}"].to(torch.float16).numpy())
        for proj in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            arr = mx.array(st[f"{p}.{proj}.weight"].to(torch.float16).numpy())
            wq, sc, bi = mx.quantize(arr, group_size=args.group_size, bits=args.bits)
            mlx_quant[f"{p}.{proj}.weight"] = wq
            mlx_quant[f"{p}.{proj}.scales"] = sc
            mlx_quant[f"{p}.{proj}.biases"] = bi
            if f"{p}.{proj}.bias" in st:
                mlx_quant[f"{p}.{proj}.bias"] = mx.array(st[f"{p}.{proj}.bias"].to(torch.float16).numpy())

    out_st = out_dir / "model.safetensors"
    mx.save_safetensors(str(out_st), mlx_quant)
    cfg["quantization"] = {"bits": args.bits, "group_size": args.group_size}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "vocab.json").write_text((in_dir / "vocab.json").read_text(encoding="utf-8"), encoding="utf-8")
    size_mb = out_st.stat().st_size / (1024 * 1024)
    print(f"Quantized {num_layers}L / {cfg['vocab_size']}-vocab 4-bit MLX bundle -> {out_dir} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
