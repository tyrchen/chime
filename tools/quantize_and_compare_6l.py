#!/usr/bin/env python3
"""
Converts Colab-distilled `layers_6l_fp16.safetensors` into a native 4-bit MLX bundle (`~64.2 MB`)
and runs a rigorous head-to-head comparison on Apple Silicon Metal against the intact 28-layer
4-bit MLX `Qwen3-0.6B-Base` (`249.9 MB`) model!
"""

import argparse
import glob
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

STRESS_PROBES = [
    ("4/5究竟有", ["多大", "多达", "多打", "躲大"]),
    ("打字的过程中要维护一个上下文窗口，这样才能推荐出来最", ["合适", "何时", "何事", "和是"]),
    ("感觉第一候选字有时候不那么", ["靠谱", "靠铺", "考铺"]),
    ("法庭依法保障公民的", ["权利", "权力", "全力"]),
    ("严禁滥用国家的", ["权力", "权利", "全力"]),
    ("群众向上级如实地", ["反映", "反应"]),
    ("吃海鲜后身体出现了过敏", ["反应", "反映"]),
    ("空气和水是人类生存所", ["必需", "必须"]),
    ("我们每个人都", ["必须", "必需"]),
    ("统计数据时间", ["截至", "截止"]),
    ("报名通道已经", ["截止", "截至"]),
    ("面对复杂多变的国际", ["形势", "形式"]),
    ("丰富多样的艺术表现", ["形式", "形势"]),
    ("为什么不用更新", ["版本", "板本", "半本"]),
    ("现在的排序算法还有很大的提升", ["空间", "空降", "控件"]),
    ("模糊音处理和候选字还有", ["不少", "不稍", "步哨"]),
]


def quantize_colab_6l(layers_fp16_path: Path, vocab_path: Path, out_dir: Path) -> float:
    snap = Path(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/*"))[0])
    orig_cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    vocab_strings = json.loads(vocab_path.read_text(encoding="utf-8"))

    tok = AutoTokenizer.from_pretrained(str(snap))
    bos_id = 151643
    selected_old_ids = [bos_id, bos_id]
    for piece in vocab_strings[2:]:
        ids = tok.encode(piece, add_special_tokens=False)
        selected_old_ids.append(ids[0] if len(ids) == 1 else bos_id)

    teacher_st = load_file(str(snap / "model.safetensors"))
    idx_cpu = torch.tensor(selected_old_ids, dtype=torch.long)
    embed_pruned = teacher_st["model.embed_tokens.weight"].index_select(0, idx_cpu).to(torch.float16).numpy()
    del teacher_st

    student_st = load_file(str(layers_fp16_path))
    mlx_quant: dict[str, mx.array] = {}

    wq, sc, bi = mx.quantize(mx.array(embed_pruned), group_size=64, bits=4)
    mlx_quant["model.embed_tokens.weight"] = wq
    mlx_quant["model.embed_tokens.scales"] = sc
    mlx_quant["model.embed_tokens.biases"] = bi
    mlx_quant["model.norm.weight"] = mx.array(student_st["model.norm.weight"].to(torch.float16).numpy())

    for l_idx in range(6):
        p = f"model.layers.{l_idx}"
        for suf in [
            "input_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
        ]:
            mlx_quant[f"{p}.{suf}"] = mx.array(student_st[f"{p}.{suf}"].to(torch.float16).numpy())
        for proj in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]:
            arr = mx.array(student_st[f"{p}.{proj}.weight"].to(torch.float16).numpy())
            wq, sc, bi = mx.quantize(arr, group_size=64, bits=4)
            mlx_quant[f"{p}.{proj}.weight"] = wq
            mlx_quant[f"{p}.{proj}.scales"] = sc
            mlx_quant[f"{p}.{proj}.biases"] = bi
            if f"{p}.{proj}.bias" in student_st:
                mlx_quant[f"{p}.{proj}.bias"] = mx.array(student_st[f"{p}.{proj}.bias"].to(torch.float16).numpy())

    out_dir.mkdir(parents=True, exist_ok=True)
    st_out = out_dir / "model.safetensors"
    mx.save_safetensors(str(st_out), mlx_quant)
    st_out.chmod(0o644)

    cfg_out = {
        "vocab_size": len(vocab_strings),
        "hidden_size": orig_cfg["hidden_size"],
        "intermediate_size": orig_cfg["intermediate_size"],
        "num_hidden_layers": 6,
        "num_attention_heads": orig_cfg["num_attention_heads"],
        "num_key_value_heads": orig_cfg["num_key_value_heads"],
        "head_dim": orig_cfg["head_dim"],
        "rms_norm_eps": orig_cfg["rms_norm_eps"],
        "rope_theta": orig_cfg["rope_theta"],
        "quantization": {"group_size": 64, "bits": 4},
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
    (out_dir / "vocab.json").write_text(json.dumps(vocab_strings, ensure_ascii=False), encoding="utf-8")
    return st_out.stat().st_size / (1024 * 1024)


def benchmark_mlx_bundle(bundle_dir: Path, label: str):
    cfg = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((bundle_dir / "vocab.json").read_text(encoding="utf-8"))
    p2i = {p: i for i, p in enumerate(vocab)}
    w = mx.load(str(bundle_dir / "model.safetensors"))
    size_mb = (bundle_dir / "model.safetensors").stat().st_size / (1024 * 1024)

    n_layers = cfg["num_hidden_layers"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
    bits = cfg["quantization"]["bits"]
    gs = cfg["quantization"]["group_size"]

    def encode(s: str) -> list[int]:
        out = []
        i = 0
        while i < len(s):
            if s[i].isspace():
                i += 1
                continue
            for span in range(min(6, len(s) - i), 0, -1):
                tid = p2i.get(s[i : i + span])
                if tid is not None:
                    out.append(tid)
                    i += span
                    break
            else:
                i += 1
        return out

    def forward_step(ids: list[int], past_kv=None, offset: int = 0):
        t_len = len(ids)
        idx = mx.array(ids)
        x = mx.dequantize(
            w["model.embed_tokens.weight"][idx],
            w["model.embed_tokens.scales"][idx],
            w["model.embed_tokens.biases"][idx],
            gs,
            bits,
        )[None, :, :]
        new_kv = []
        for l in range(n_layers):
            p = f"model.layers.{l}"
            xn = mx.fast.rms_norm(x, w[f"{p}.input_layernorm.weight"], eps)
            q = mx.quantized_matmul(
                xn, w[f"{p}.self_attn.q_proj.weight"], w[f"{p}.self_attn.q_proj.scales"], w[f"{p}.self_attn.q_proj.biases"], True, gs, bits
            ).reshape(1, t_len, nh, hd)
            k = mx.quantized_matmul(
                xn, w[f"{p}.self_attn.k_proj.weight"], w[f"{p}.self_attn.k_proj.scales"], w[f"{p}.self_attn.k_proj.biases"], True, gs, bits
            ).reshape(1, t_len, nkv, hd)
            v = mx.quantized_matmul(
                xn, w[f"{p}.self_attn.v_proj.weight"], w[f"{p}.self_attn.v_proj.scales"], w[f"{p}.self_attn.v_proj.biases"], True, gs, bits
            ).reshape(1, t_len, nkv, hd)
            q = mx.fast.rms_norm(q, w[f"{p}.self_attn.q_norm.weight"], eps).transpose(0, 2, 1, 3)
            k = mx.fast.rms_norm(k, w[f"{p}.self_attn.k_norm.weight"], eps).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, hd, traditional=False, base=theta, scale=1.0, offset=offset)
            k = mx.fast.rope(k, hd, traditional=False, base=theta, scale=1.0, offset=offset)
            if past_kv is not None:
                k = mx.concatenate([past_kv[l][0], k], axis=2)
                v = mx.concatenate([past_kv[l][1], v], axis=2)
            new_kv.append((k, v))
            mask = "causal" if t_len > 1 else None
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5, mask=mask).transpose(0, 2, 1, 3).reshape(1, t_len, nh * hd)
            x = x + mx.quantized_matmul(
                a, w[f"{p}.self_attn.o_proj.weight"], w[f"{p}.self_attn.o_proj.scales"], w[f"{p}.self_attn.o_proj.biases"], True, gs, bits
            )
            xm = mx.fast.rms_norm(x, w[f"{p}.post_attention_layernorm.weight"], eps)
            gate = nn.silu(
                mx.quantized_matmul(xm, w[f"{p}.mlp.gate_proj.weight"], w[f"{p}.mlp.gate_proj.scales"], w[f"{p}.mlp.gate_proj.biases"], True, gs, bits)
            )
            up = mx.quantized_matmul(xm, w[f"{p}.mlp.up_proj.weight"], w[f"{p}.mlp.up_proj.scales"], w[f"{p}.mlp.up_proj.biases"], True, gs, bits)
            x = x + mx.quantized_matmul(
                gate * up, w[f"{p}.mlp.down_proj.weight"], w[f"{p}.mlp.down_proj.scales"], w[f"{p}.mlp.down_proj.biases"], True, gs, bits
            )
        out = mx.fast.rms_norm(x, w["model.norm.weight"], eps)
        logits = mx.quantized_matmul(
            out, w["model.embed_tokens.weight"], w["model.embed_tokens.scales"], w["model.embed_tokens.biases"], True, gs, bits
        ).astype(mx.float32)
        return nn.log_softmax(logits, axis=-1)[0], new_kv

    wins = 0
    margins = []
    per_probe = []
    for ctx_s, cands in STRESS_PROBES:
        c_ids = encode(ctx_s)
        lps0, kv = forward_step(c_ids)
        mx.eval(lps0, *[t for pair in kv for t in pair])
        scores = []
        for cand in cands:
            tids = encode(cand)
            s = float(lps0[-1, tids[0]])
            if len(tids) > 1:
                lps_c, _ = forward_step(tids[:-1], past_kv=kv, offset=len(c_ids))
                mx.eval(lps_c)
                for i, tid in enumerate(tids[1:]):
                    s += float(lps_c[i, tid])
            scores.append((cand, s))
        target_lp = scores[0][1]
        best_foil_lp = max(x[1] for x in scores[1:])
        margin = target_lp - best_foil_lp
        if margin > 0:
            wins += 1
        margins.append(margin)
        per_probe.append((ctx_s, [(c, round(v, 2)) for c, v in scores], round(margin, 2)))

    # Warm Metal GPU latency benchmark (24-token prefill + 1-token KV step)
    ctx_ids = encode("打字的过程中要维护一个上下文窗口，这样才能推荐出来最")
    for _ in range(5):
        lp0, kv = forward_step(ctx_ids)
        mx.eval(lp0, *[t for pair in kv for t in pair])
    t0 = time.perf_counter()
    for _ in range(30):
        lp0, kv = forward_step(ctx_ids)
        mx.eval(lp0, *[t for pair in kv for t in pair])
    prefill_ms = (time.perf_counter() - t0) * 1000 / 30

    for _ in range(5):
        o, _ = forward_step([239], past_kv=kv, offset=len(ctx_ids))
        mx.eval(o)
    t1 = time.perf_counter()
    for _ in range(40):
        o, _ = forward_step([239], past_kv=kv, offset=len(ctx_ids))
        mx.eval(o)
    step_ms = (time.perf_counter() - t1) * 1000 / 40

    mean_margin = sum(margins) / len(margins)
    print(f"\n==================== {label} ({size_mb:.1f} MB, {n_layers} layers) ====================")
    print(f"Pure Zero-Shot SLM Stress Accuracy: {wins}/{len(STRESS_PROBES)} ({wins * 100 / len(STRESS_PROBES):.1f}%)")
    print(f"Mean Log-Likelihood Margin (Target - Best Foil): {mean_margin:+.2f} nats")
    print(f"Metal GPU 24-Token Prefill Latency: {prefill_ms:.2f} ms | Warm 1-Step KV Latency: {step_ms:.2f} ms")
    for ctx_s, sc, mg in per_probe:
        mark = "PASS" if mg > 0 else "FAIL"
        print(f"  [{mark} {mg:+6.2f}] {ctx_s} -> {sc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers-fp16", required=True)
    parser.add_argument("--base-28l", default="/Users/tchen/projects/mycode/rust/chime/dist/base_neural")
    parser.add_argument("--out-6l", default="/Users/tchen/projects/mycode/rust/chime/dist/colab_6l_neural")
    args = parser.parse_args()

    base_dir = Path(args.base_28l)
    out_6l = Path(args.out_6l)
    mb_6l = quantize_colab_6l(Path(args.layers_fp16), base_dir / "vocab.json", out_6l)
    print(f"Quantized Colab-distilled 6-layer student to 4-bit MLX: {mb_6l:.2f} MB at {out_6l}")

    benchmark_mlx_bundle(base_dir, "Intact 28-Layer Qwen3-0.6B-Base 4-bit MLX")
    benchmark_mlx_bundle(out_6l, "Colab-Distilled 6-Layer Qwen3-Chime 4-bit MLX")


if __name__ == "__main__":
    main()
