#!/usr/bin/env python3
"""
Google Colab NVIDIA L4 GPU Multi-LoRA Trainer for Chime (`v0.6.0`):
Loads the exact frozen 6-layer `Qwen3-6L-Chime` student (`layers_6l_fp16.safetensors` + pruned
`Qwen/Qwen3-0.6B-Base` embeddings + `vocab.json`) and trains clean Rank-8 (`r=8, scale=1.0`)
FP16 `lora.safetensors` sidecar adapters (`~2.06 MB` each) for:
  1. `classics` (`classics_lora.safetensors` — Classical Chinese Poetry, Ci, Qu & Guwen)
  2. `tech`     (`tech_lora.safetensors`     — Bilingual Developer / Systems / AI / Cloud prose)
  3. Optional local private corpus (`personal_lora.safetensors` — strictly ephemeral, local-only)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("weight", weight.float().clone())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(orig_dtype)


def build_rope_cache(max_seq_len: int, head_dim: int, theta: float, device: torch.device):
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = pos * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).view(1, 1, max_seq_len, head_dim)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).view(1, 1, max_seq_len, head_dim)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    hd = x.shape[-1]
    t = x.shape[2]
    c = cos[:, :, :t, :].to(x.dtype)
    s = sin[:, :, :t, :].to(x.dtype)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * c + rotated * s


class FrozenLinearWithLora(nn.Module):
    """Exact mirror of `MlxLinear::forward_with_loras`: `y = x @ W^T + scale * (x @ A^T) @ B^T`."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: int = 8,
        scale: float = 1.0,
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        out_f, in_f = weight.shape
        self.register_buffer("weight", weight.to(compute_dtype).clone())
        if bias is not None:
            self.register_buffer("bias", bias.to(compute_dtype).clone())
        else:
            self.bias = None
        self.rank = rank
        self.scale = scale
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32))
        self.reset_lora()

    def reset_lora(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        low = F.linear(x, self.lora_A.to(x.dtype))
        delta = F.linear(low, self.lora_B.to(x.dtype))
        return base + self.scale * delta


class Qwen3LoraBlock(nn.Module):
    def __init__(
        self,
        raw_st: dict[str, torch.Tensor],
        layer_idx: int,
        cfg: dict,
        rank: int,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        p = f"model.layers.{layer_idx}"
        self.n_h = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.q_out = self.n_h * self.hd
        eps = float(cfg.get("rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(raw_st[f"{p}.input_layernorm.weight"], eps)
        self.post_attention_layernorm = RMSNorm(raw_st[f"{p}.post_attention_layernorm.weight"], eps)
        self.q_norm = RMSNorm(raw_st[f"{p}.self_attn.q_norm.weight"], eps)
        self.k_norm = RMSNorm(raw_st[f"{p}.self_attn.k_norm.weight"], eps)

        def mk(proj: str) -> FrozenLinearWithLora:
            return FrozenLinearWithLora(
                raw_st[f"{p}.{proj}.weight"],
                raw_st.get(f"{p}.{proj}.bias"),
                rank=rank,
                scale=1.0,
                compute_dtype=compute_dtype,
            )

        self.q_proj = mk("self_attn.q_proj")
        self.k_proj = mk("self_attn.k_proj")
        self.v_proj = mk("self_attn.v_proj")
        self.o_proj = mk("self_attn.o_proj")
        self.gate_proj = mk("mlp.gate_proj")
        self.up_proj = mk("mlp.up_proj")
        self.down_proj = mk("mlp.down_proj")

    def reset_lora(self) -> None:
        for mod in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        ):
            mod.reset_lora()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        xn = self.input_layernorm(x)
        q = self.q_norm(self.q_proj(xn).view(b, t, self.n_h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(xn).view(b, t, self.n_kv, self.hd)).transpose(1, 2)
        v = self.v_proj(xn).view(b, t, self.n_kv, self.hd).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        rep = self.n_h // self.n_kv
        k_rep = k.repeat_interleave(rep, dim=1)
        v_rep = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        x = x + self.o_proj(attn.transpose(1, 2).contiguous().view(b, t, self.q_out))

        xm = self.post_attention_layernorm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(xm)) * self.up_proj(xm))
        return x


class Qwen3FrozenBaseWithLora(nn.Module):
    def __init__(
        self,
        student_st: dict[str, torch.Tensor],
        embed_pruned: torch.Tensor,
        cfg: dict,
        rank: int,
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.register_buffer("embed_weight", embed_pruned.to(compute_dtype))
        self.layers = nn.ModuleList(
            [Qwen3LoraBlock(student_st, i, cfg, rank=rank, compute_dtype=compute_dtype) for i in range(6)]
        )
        self.norm = RMSNorm(student_st["model.norm.weight"], float(cfg.get("rms_norm_eps", 1e-6)))

    def reset_lora(self) -> None:
        for blk in self.layers:
            blk.reset_lora()

    def export_lora_fp16(self, out_path: Path) -> float:
        tensors: dict[str, torch.Tensor] = {}
        proj_map = [
            ("self_attn.q_proj", "q_proj"),
            ("self_attn.k_proj", "k_proj"),
            ("self_attn.v_proj", "v_proj"),
            ("self_attn.o_proj", "o_proj"),
            ("mlp.gate_proj", "gate_proj"),
            ("mlp.up_proj", "up_proj"),
            ("mlp.down_proj", "down_proj"),
        ]
        for idx, blk in enumerate(self.layers):
            p = f"model.layers.{idx}"
            for st_name, attr_name in proj_map:
                lin: FrozenLinearWithLora = getattr(blk, attr_name)
                tensors[f"{p}.{st_name}.lora_A"] = lin.lora_A.detach().to(torch.float16).cpu().contiguous()
                tensors[f"{p}.{st_name}.lora_B"] = lin.lora_B.detach().to(torch.float16).cpu().contiguous()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out_path))
        return out_path.stat().st_size / (1024 * 1024)

    def forward_logits(self, ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = F.embedding(ids, self.embed_weight)
        for blk in self.layers:
            x = blk(x, cos, sin)
        u = self.norm(x)
        return F.linear(u, self.embed_weight).float()


class GreedyCompactTokenizer:
    def __init__(self, vocab_strings: list[str]) -> None:
        self.p2i = {p: i for i, p in enumerate(vocab_strings)}
        self.unk = 1

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        n = len(text)
        i = 0
        while i < n:
            if text[i].isspace():
                i += 1
                continue
            max_span = min(6, n - i)
            matched_id = -1
            matched_span = 1
            for span in range(max_span, 0, -1):
                tid = self.p2i.get(text[i : i + span])
                if tid is not None:
                    matched_id = tid
                    matched_span = span
                    break
            if matched_id >= 0:
                out.append(matched_id)
                i += matched_span
            else:
                out.append(self.unk)
                i += 1
        return out


def augment_tech_corpus_from_hf(seed_lines: list[str], target_chars: int = 2_200_000) -> list[str]:
    lines = list(seed_lines[:24])
    chars = sum(len(s) for s in lines)
    tech_keywords = (
        "代码", "编译", "内存", "线程", "并发", "接口", "模型", "推理", "集群", "数据库",
        "函数", "算法", "架构", "服务器", "缓存", "协议", "部署", "容器", "索引", "延迟",
        "Rust", "Python", "Swift", "Linux", "macOS", "GitHub", "API", "SQL", "GPU", "Docker",
    )
    sources = [
        ("opencsg/FineWeb-Edu-Chinese-V2.1", None, ["text"]),
        ("BelleGroup/train_0.5M_CN", None, ["instruction", "output"]),
    ]
    for repo, subset, fields in sources:
        if chars >= target_chars:
            break
        try:
            ds = (
                load_dataset(repo, subset, split="train", streaming=True)
                if subset
                else load_dataset(repo, split="train", streaming=True)
            )
            for row in ds:
                txt = " ".join(str(row.get(f, "")).strip() for f in fields if row.get(f)).strip()
                if len(txt) < 30:
                    continue
                if any(kw in txt for kw in tech_keywords):
                    snippet = txt[:500].replace("\n\n", "\n")
                    lines.append(snippet)
                    chars += len(snippet)
                    if chars >= target_chars:
                        break
        except Exception as exc:
            print(f"  [tech-hf] skipped {repo}: {exc}", flush=True)
    print(f"[tech] corpus ready: {len(lines):,} passages ({chars / 1e6:.2f}M chars)", flush=True)
    return lines


def tokenize_into_blocks(lines: list[str], tok: GreedyCompactTokenizer, seq_len: int = 96) -> torch.Tensor:
    rng = random.Random(2026)
    shuffled = list(lines)
    rng.shuffle(shuffled)
    flat: list[int] = []
    sep = tok.p2i.get("。", 0)
    for ln in shuffled:
        ids = [i for i in tok.encode(ln) if i != tok.unk]
        if len(ids) >= 6:
            flat.extend(ids)
            if flat[-1] != sep:
                flat.append(sep)
    n_blocks = len(flat) // (seq_len + 1)
    trimmed = flat[: n_blocks * (seq_len + 1)]
    return torch.tensor(trimmed, dtype=torch.long).view(n_blocks, seq_len + 1)


def train_one_lora(
    name: str,
    model: Qwen3FrozenBaseWithLora,
    blocks: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out_path: Path,
    steps: int = 450,
    batch_size: int = 32,
    lr: float = 4e-4,
    device: torch.device = torch.device("cuda"),
) -> None:
    model.reset_lora()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    n_total = blocks.shape[0]
    n_val = max(16, min(128, n_total // 15))
    val_blocks = blocks[:n_val].to(device)
    train_blocks = blocks[n_val:] if n_total > n_val + 8 else blocks

    def eval_nll() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, val_blocks.shape[0], batch_size):
                b = val_blocks[start : start + batch_size]
                logits = model.forward_logits(b[:, :-1], cos, sin)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
                losses.append(loss.item())
        model.train()
        return sum(losses) / max(1, len(losses))

    base_nll = eval_nll()
    print(f"\n=== [{name}] Training Rank-8 LoRA ({steps} steps, blocks={n_total}) | base_val_nll={base_nll:.4f} ===", flush=True)
    t0 = time.time()

    for step in range(1, steps + 1):
        # Cosine schedule with 30-step warmup
        if step <= 30:
            cur_lr = lr * (step / 30.0)
        else:
            prog = (step - 30) / max(1, steps - 30)
            cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        idx = torch.randint(0, train_blocks.shape[0], (batch_size,))
        batch = train_blocks[idx].to(device, non_blocking=True)
        logits = model.forward_logits(batch[:, :-1], cos, sin)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % 150 == 0 or step == steps:
            val_nll = eval_nll()
            print(
                f"  [{name}] step {step:4d}/{steps} | train_nll={loss.item():.4f} | "
                f"val_nll={val_nll:.4f} (delta={val_nll - base_nll:+.4f}) | {time.time() - t0:.1f}s",
                flush=True,
            )

    mb = model.export_lora_fp16(out_path)
    print(f"=== [{name}] Saved {out_path} ({mb:.2f} MB) ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers-st", type=Path, default=Path("/content/layers_6l_fp16.safetensors"))
    parser.add_argument(
        "--q4-model-dir",
        type=Path,
        default=None,
        help="Optional 4-bit quantized runtime directory (`dist/base_neural`): dequantizes exact runtime 4-bit weights so LoRA trains directly on the quantized manifold without gamma=1.0 overshoot!",
    )
    parser.add_argument("--vocab", type=Path, default=Path("/content/vocab.json"))
    parser.add_argument("--classics-txt", type=Path, default=Path("/content/classics_lora_corpus.txt"))
    parser.add_argument("--tech-txt", type=Path, default=Path("/content/tech_lora_corpus.txt"))
    parser.add_argument("--private-txt", type=Path, default=Path("/tmp/chime_private_personal/personal_lora_corpus.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("/content/out"))
    parser.add_argument("--private-out-lora", type=Path, default=None)
    parser.add_argument("--only-tech", action="store_true")
    parser.add_argument("--skip-hf-tech", action="store_true")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    args, _ = parser.parse_known_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        compute_dtype = torch.bfloat16
        dev_label = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        compute_dtype = torch.float16
        dev_label = "Apple Silicon Metal (MPS)"
    else:
        device = torch.device("cpu")
        compute_dtype = torch.float32
        dev_label = "CPU"
    print(f"Device: {device} ({dev_label}, dtype={compute_dtype})", flush=True)

    snap = Path(
        snapshot_download(
            "Qwen/Qwen3-0.6B-Base",
            allow_patterns=["config.json", "tokenizer*", "vocab.json", "merges.txt", "model.safetensors"],
        )
    )
    orig_cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    vocab_strings = json.loads(args.vocab.read_text(encoding="utf-8"))

    if args.q4_model_dir is not None and (args.q4_model_dir / "model.safetensors").exists():
        import mlx.core as mx
        import numpy as np

        q4_cfg = json.loads((args.q4_model_dir / "config.json").read_text(encoding="utf-8"))
        q_gs = q4_cfg["quantization"]["group_size"]
        q_bits = q4_cfg["quantization"]["bits"]
        q4_w = mx.load(str(args.q4_model_dir / "model.safetensors"))
        emb_deq = mx.dequantize(
            q4_w["model.embed_tokens.weight"],
            q4_w["model.embed_tokens.scales"],
            q4_w["model.embed_tokens.biases"],
            q_gs,
            q_bits,
        ).astype(mx.float16)
        mx.eval(emb_deq)
        embed_pruned = torch.from_numpy(np.array(emb_deq)).contiguous()
        student_st: dict[str, torch.Tensor] = {
            "model.norm.weight": torch.from_numpy(np.array(q4_w["model.norm.weight"].astype(mx.float32))).contiguous()
        }
        for l_idx in range(6):
            p = f"model.layers.{l_idx}"
            for norm_suf in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ):
                student_st[f"{p}.{norm_suf}"] = torch.from_numpy(
                    np.array(q4_w[f"{p}.{norm_suf}"].astype(mx.float32))
                ).contiguous()
            for proj in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ):
                w_deq = mx.dequantize(
                    q4_w[f"{p}.{proj}.weight"],
                    q4_w[f"{p}.{proj}.scales"],
                    q4_w[f"{p}.{proj}.biases"],
                    q_gs,
                    q_bits,
                ).astype(mx.float16)
                mx.eval(w_deq)
                student_st[f"{p}.{proj}.weight"] = torch.from_numpy(np.array(w_deq)).contiguous()
                if f"{p}.{proj}.bias" in q4_w:
                    student_st[f"{p}.{proj}.bias"] = torch.from_numpy(
                        np.array(q4_w[f"{p}.{proj}.bias"].astype(mx.float16))
                    ).contiguous()
        print(
            f"Loaded exact dequantized 4-bit runtime manifold from {args.q4_model_dir} (group_size={q_gs}, bits={q_bits})",
            flush=True,
        )
    else:
        hf_tok = AutoTokenizer.from_pretrained(str(snap))
        bos_id = 151643
        selected_old_ids = [bos_id, bos_id]
        for piece in vocab_strings[2:]:
            ids = hf_tok.encode(piece, add_special_tokens=False)
            selected_old_ids.append(ids[0] if len(ids) == 1 else bos_id)

        teacher_st = load_file(str(snap / "model.safetensors"))
        embed_pruned = (
            teacher_st["model.embed_tokens.weight"]
            .index_select(0, torch.tensor(selected_old_ids, dtype=torch.long))
            .contiguous()
        )
        del teacher_st
        student_st = load_file(str(args.layers_st))
    model = Qwen3FrozenBaseWithLora(
        student_st, embed_pruned, orig_cfg, rank=args.rank, compute_dtype=compute_dtype
    ).to(device)
    cos, sin = build_rope_cache(256, orig_cfg["head_dim"], float(orig_cfg["rope_theta"]), device)
    compact_tok = GreedyCompactTokenizer(vocab_strings)

    if args.classics_txt.exists() and not args.only_tech:
        c_lines = [ln.strip() for ln in args.classics_txt.read_text(encoding="utf-8").splitlines() if len(ln.strip()) >= 8]
        c_blocks = tokenize_into_blocks(c_lines, compact_tok, seq_len=96)
        train_one_lora(
            "classics",
            model,
            c_blocks,
            cos,
            sin,
            args.out_dir / "classics_lora.safetensors",
            steps=args.steps,
            batch_size=args.batch_size,
            device=device,
        )

    if args.tech_txt.exists():
        t_seed = [ln.strip() for ln in args.tech_txt.read_text(encoding="utf-8").splitlines() if len(ln.strip()) >= 8]
        t_lines = t_seed * 12 if args.skip_hf_tech else augment_tech_corpus_from_hf(t_seed, target_chars=1_200_000)
        t_blocks = tokenize_into_blocks(t_lines, compact_tok, seq_len=96)
        train_one_lora(
            "tech",
            model,
            t_blocks,
            cos,
            sin,
            args.out_dir / "tech_lora.safetensors",
            steps=args.steps,
            batch_size=args.batch_size,
            device=device,
        )

    if args.private_txt.exists() and not args.only_tech:
        p_lines = [ln.strip() for ln in args.private_txt.read_text(encoding="utf-8").splitlines() if len(ln.strip()) >= 12]
        p_blocks = tokenize_into_blocks(p_lines * 4, compact_tok, seq_len=96)
        priv_out = args.private_out_lora or (args.out_dir / "personal_lora.safetensors")
        train_one_lora(
            "private-personal",
            model,
            p_blocks,
            cos,
            sin,
            priv_out,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=3e-4,
            device=device,
        )


if __name__ == "__main__":
    main()
