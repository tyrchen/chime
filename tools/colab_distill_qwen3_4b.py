#!/usr/bin/env python3
"""
Google Colab GPU Stronger-Teacher + Full-CJK-Vocab Unified Distillation:
Distills 36-layer `Qwen/Qwen3-4B-Base` (Apache-2.0) into a compact 6L or 8L `Qwen3-Chime` student
(`36,864` vocab = 100% of all 24,872 Qwen3 CJK tokens + top English/code tokens!) using:
  1. Full-vocab KL divergence (T=1.5) against `Qwen/Qwen3-4B-Base` sliced to `36,864` logits;
  2. Phonetic-Equivalence Homophone-Subspace KL (T=1.0) over `H(pinyin(y))` (e.g. `的/地/得`,
     `在/再`, `做/坐/座`, `权利/权力`, `事/时/是/试`);
  3. Unfrozen tied `embed_tokens` (lower LR) so the 1024-dim student aligns cleanly to the 4B
     teacher's richer 36,864-token probability manifold!

Usage on Google Colab (A100 / L4 recommended):
  1. Upload `dist/vocab_36k.json` and `tools/colab_distill_qwen3_4b.py` to `/content/`.
  2. Run:
     !pip install -q torch transformers datasets safetensors huggingface_hub pypinyin
     !python /content/colab_distill_qwen3_4b.py \
       --vocab /content/vocab_36k.json \
       --teacher Qwen/Qwen3-4B-Base \
       --num-layers 8 \
       --steps 2400 \
       --batch-size 32 \
       --out-dir /content/student_8l_36k
  3. Download `/content/student_8l_36k/student_fp16.safetensors` & `config.json` & `vocab.json`
     and quantize locally to 4-bit MLX with `tools/quantize_and_compare_6l.py`!
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from pypinyin import Style, lazy_pinyin
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

DISAMBIGUATION_PROBES = [
    ("4/5究竟有", ["多大", "多达", "多打", "躲大"]),
    ("打字的过程中要维护一个上下文窗口，这样才能推荐出来最", ["合适", "何时", "何事", "和是"]),
    ("感觉第一候选字有时候不那么", ["靠谱", "靠铺", "考铺"]),
    ("依法保障", ["权利", "权力", "全力"]),
    ("严禁滥用", ["权力", "权利", "全力"]),
    ("我晚上有", ["点事", "电视", "点时"]),
    ("路上注意安全", ["慢点开车", "盲点开车"]),
    ("帮我把客厅的", ["灯关一下", "灯光一下"]),
    ("还是你想", ["得比较周到", "的比较周到"]),
    ("玉桂", ["是", "时", "市"]),
    ("玉", ["桂", "鬼", "归"]),
]


def is_all_cjk(s: str) -> bool:
    return bool(s) and all("\u4e00" <= ch <= "\u9fa5" for ch in s)


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.float().clone())
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


class Qwen3Block(nn.Module):
    def __init__(self, raw_st: dict[str, torch.Tensor], layer_idx: int, cfg: dict) -> None:
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

        self.q_proj = self._make_linear(raw_st, f"{p}.self_attn.q_proj")
        self.k_proj = self._make_linear(raw_st, f"{p}.self_attn.k_proj")
        self.v_proj = self._make_linear(raw_st, f"{p}.self_attn.v_proj")
        self.o_proj = self._make_linear(raw_st, f"{p}.self_attn.o_proj")

        self.gate_proj = self._make_linear(raw_st, f"{p}.mlp.gate_proj")
        self.up_proj = self._make_linear(raw_st, f"{p}.mlp.up_proj")
        self.down_proj = self._make_linear(raw_st, f"{p}.mlp.down_proj")

    @staticmethod
    def _make_linear(raw_st: dict[str, torch.Tensor], prefix: str) -> nn.Linear:
        w = raw_st[f"{prefix}.weight"].float()
        b = raw_st.get(f"{prefix}.bias")
        lin = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
        with torch.no_grad():
            lin.weight.copy_(w)
            if b is not None:
                lin.bias.copy_(b.float())
        return lin

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


class Qwen3Student(nn.Module):
    def __init__(
        self,
        raw_st: dict[str, torch.Tensor],
        embed_pruned: torch.Tensor,
        selected_layers: list[int],
        cfg: dict,
    ) -> None:
        super().__init__()
        self.embed_weight = nn.Parameter(embed_pruned.float().clone())
        self.layers = nn.ModuleList([Qwen3Block(raw_st, li, cfg) for li in selected_layers])
        self.norm = RMSNorm(raw_st["model.norm.weight"], float(cfg.get("rms_norm_eps", 1e-6)))

    def forward(self, new_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = F.embedding(new_ids, self.embed_weight)
        for blk in self.layers:
            x = blk(x, cos, sin)
        return self.norm(x)


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


def build_homophone_groups(vocab_strings: list[str], max_group: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Maps every CJK token id -> padded tensor of its toneless-Pinyin homophone equivalence class."""
    by_py: dict[str, list[int]] = defaultdict(list)
    for tid, piece in enumerate(vocab_strings):
        if tid < 2 or not (1 <= len(piece) <= 4) or not is_all_cjk(piece):
            continue
        py = "".join(lazy_pinyin(piece, style=Style.NORMAL))
        if py and py.isalpha():
            by_py[py].append(tid)

    v = len(vocab_strings)
    group_ids = torch.zeros((v, max_group), dtype=torch.long)
    group_mask = torch.zeros((v, max_group), dtype=torch.bool)
    covered = 0
    for tids in by_py.values():
        if len(tids) < 2:
            continue
        sub = tids[:max_group]
        for tid in sub:
            for j, peer in enumerate(sub):
                group_ids[tid, j] = peer
                group_mask[tid, j] = True
            covered += 1
    print(f"Built phonetic-equivalence homophone table: {covered} CJK tokens across {len(by_py)} readings", flush=True)
    return group_ids, group_mask


def collect_chinese_corpus(target_chars: int = 24_000_000) -> list[str]:
    texts: list[str] = []
    total_chars = 0
    candidate_sources = [
        ("opencsg/FineWeb-Edu-Chinese-V2.1", None, "train", ["text"]),
        ("BAAI/COIG-CQIA", "zhihu", "train", ["instruction", "output"]),
        ("BAAI/COIG-CQIA", "coig_pc", "train", ["instruction", "output"]),
        ("BelleGroup/train_0.5M_CN", None, "train", ["instruction", "output"]),
        ("wikimedia/wikipedia", "20231101.zh", "train", ["text"]),
    ]
    for repo, subset, split, fields in candidate_sources:
        if total_chars >= target_chars:
            break
        try:
            print(f"Streaming corpus from {repo} ({subset or 'default'})...", flush=True)
            ds = (
                load_dataset(repo, subset, split=split, streaming=True)
                if subset
                else load_dataset(repo, split=split, streaming=True)
            )
            source_cap = target_chars // 3
            source_chars = 0
            taken = 0
            for row in ds:
                parts = [str(row.get(f, "")).strip() for f in fields if row.get(f)]
                merged = " ".join(p for p in parts if p)
                if len(merged) < 20:
                    continue
                cjk = sum(1 for c in merged[:300] if "\u4e00" <= c <= "\u9fa5")
                if cjk < 12:
                    continue
                snippet = merged[:600].replace("\n\n", "\n")
                texts.append(snippet)
                total_chars += len(snippet)
                source_chars += len(snippet)
                taken += 1
                if source_chars >= source_cap or total_chars >= target_chars:
                    break
            print(f"  -> collected {taken} passages ({source_chars / 1e6:.2f}M chars)", flush=True)
        except Exception as exc:
            print(f"  -> skipped {repo}: {exc}", flush=True)

    random.Random(42).shuffle(texts)
    print(f"Total corpus: {len(texts)} passages, {total_chars / 1e6:.2f}M chars", flush=True)
    return texts


@torch.no_grad()
def evaluate_probes(student: Qwen3Student, c_tok: GreedyCompactTokenizer, cos: torch.Tensor, sin: torch.Tensor, device: torch.device):
    student.eval()
    passed = 0
    for ctx_s, cands in DISAMBIGUATION_PROBES:
        c_ids = c_tok.encode(ctx_s)
        scores = []
        for cand in cands:
            t_ids = c_tok.encode(cand)
            seq = torch.tensor([c_ids + t_ids], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                u_s = student(seq, cos, sin)
                logits = F.linear(u_s, student.embed_weight).float()
                lps = F.log_softmax(logits[0], dim=-1)
            lp = sum(float(lps[len(c_ids) - 1 + i, tid].item()) for i, tid in enumerate(t_ids))
            scores.append((cand, lp))
        if max(scores, key=lambda x: x[1])[0] == cands[0]:
            passed += 1
    student.train()
    return passed, len(DISAMBIGUATION_PROBES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="/content/vocab_36k.json")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--init-base", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--resume-from", default="", help="Optional student_fp16.safetensors checkpoint to resume from")
    parser.add_argument("--num-layers", type=int, default=8, choices=[6, 8])
    parser.add_argument("--out-dir", default="/content/student_8l_36k_v2")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--corpus-chars", type=int, default=50_000_000)
    parser.add_argument("--lr", type=float, default=2.2e-4)
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vocab_strings: list[str] = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    c_tok = GreedyCompactTokenizer(vocab_strings)
    group_ids_cpu, group_mask_cpu = build_homophone_groups(vocab_strings)
    group_ids_gpu = group_ids_cpu.to(device)
    group_mask_gpu = group_mask_cpu.to(device)

    init_snap = Path(
        snapshot_download(
            args.init_base,
            allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"],
        )
    )
    orig_cfg = json.loads((init_snap / "config.json").read_text(encoding="utf-8"))
    hf_tok = AutoTokenizer.from_pretrained(str(init_snap))

    bos_id = 151643
    selected_old_ids = [bos_id, bos_id]
    for piece in vocab_strings[2:]:
        ids = hf_tok.encode(piece, add_special_tokens=False)
        selected_old_ids.append(ids[0] if len(ids) == 1 else bos_id)
    idx_cpu = torch.tensor(selected_old_ids, dtype=torch.long)
    new_to_old_gpu = idx_cpu.to(device)

    if args.resume_from and Path(args.resume_from).exists():
        print(f"Resuming student weights from {args.resume_from}...", flush=True)
        res_st = load_file(args.resume_from)
        embed_pruned = res_st["model.embed_tokens.weight"].float().contiguous()
        selected_layers = list(range(args.num_layers))
        student = Qwen3Student(res_st, embed_pruned, selected_layers, orig_cfg).to(device)
        del res_st
    else:
        raw_st = load_file(str(init_snap / "model.safetensors"))
        embed_pruned = raw_st["model.embed_tokens.weight"].index_select(0, idx_cpu).float().contiguous()
        selected_layers = [0, 3, 7, 11, 15, 19, 23, 27] if args.num_layers == 8 else [0, 5, 11, 16, 22, 27]
        student = Qwen3Student(raw_st, embed_pruned, selected_layers, orig_cfg).to(device)
        del raw_st

    print(f"Loading strong teacher {args.teacher} in bfloat16...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    # Pre-slice teacher LM head to our exact 36,864-token vocabulary (4x faster logit matmul!)
    teacher_lm_head = teacher.get_output_embeddings().weight.detach().index_select(0, new_to_old_gpu).contiguous()

    # Training-only linear projector (1024 -> teacher_hidden=2560) for dense hidden-state cosine +
    # SmoothL1 alignment; discarded at export (0 bytes inference overhead)!
    t_hidden = int(teacher.config.hidden_size)
    s_hidden = int(orig_cfg["hidden_size"])
    feat_proj = nn.Linear(s_hidden, t_hidden, bias=False).to(device)

    cos_cache, sin_cache = build_rope_cache(512, orig_cfg["head_dim"], float(orig_cfg.get("rope_theta", 1000000.0)), device)

    passages = collect_chinese_corpus(target_chars=args.corpus_chars)
    flat_tokens: list[int] = []
    sep_id = c_tok.p2i.get("。", 0)
    for text in passages:
        ids = [i for i in c_tok.encode(text) if i != 1]
        if len(ids) >= 8:
            flat_tokens.extend(ids)
            flat_tokens.append(sep_id)
    n_seqs = len(flat_tokens) // args.seq_len
    packed = torch.tensor(flat_tokens[: n_seqs * args.seq_len], dtype=torch.long).view(n_seqs, args.seq_len)
    packed = packed[torch.randperm(n_seqs, generator=torch.Generator().manual_seed(1337))]
    val_size = min(256, max(32, n_seqs // 20))
    val_seqs, train_seqs = packed[:val_size], packed[val_size:]
    print(f"Packed {len(train_seqs)} train blocks, {len(val_seqs)} val blocks (seq_len={args.seq_len})", flush=True)

    backbone_params = [p for n, p in student.named_parameters() if n != "embed_weight"]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": [student.embed_weight], "lr": args.lr * 0.25},
            {"params": list(feat_proj.parameters()), "lr": args.lr * 2.0},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    warmup = min(200, args.steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup) if s < warmup else 0.04 + 0.96 * 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, args.steps - warmup))),
    )

    t0 = time.perf_counter()
    cursor = 0
    student.train()
    feat_proj.train()
    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(train_seqs):
            train_seqs = train_seqs[torch.randperm(len(train_seqs))]
            cursor = 0
        batch_new = train_seqs[cursor : cursor + args.batch_size].to(device, non_blocking=True)
        cursor += args.batch_size
        batch_old = new_to_old_gpu[batch_new]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            u_t = teacher.model(input_ids=batch_old, use_cache=False).last_hidden_state
            z_t = F.linear(u_t, teacher_lm_head)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            u_s = student(batch_new, cos_cache, sin_cache)
            z_s = F.linear(u_s, student.embed_weight)

            # 1. Full 36,864-vocab KL divergence (T=1.3)
            z_s_flat = z_s[:, :-1, :].reshape(-1, len(vocab_strings)).float()
            z_t_flat = z_t[:, :-1, :].reshape(-1, len(vocab_strings)).float()
            inv_t = 1.0 / 1.3
            kl_full = (
                F.kl_div(
                    F.log_softmax(z_s_flat * inv_t, dim=-1),
                    F.softmax(z_t_flat * inv_t, dim=-1),
                    reduction="batchmean",
                )
                * (1.3 * 1.3)
            )

            # 2. Phonetic-Equivalence Homophone-Subspace KL (T=1.0) on next-token CJK targets!
            targets = batch_new[:, 1:].reshape(-1)
            g_ids = group_ids_gpu[targets]
            g_msk = group_mask_gpu[targets]
            has_group = g_msk.any(dim=-1)
            if has_group.any():
                idx_h = has_group.nonzero(as_tuple=False).squeeze(-1)
                sub_ids = g_ids[idx_h]
                sub_msk = g_msk[idx_h]
                s_sub = z_s_flat[idx_h].gather(1, sub_ids).masked_fill(~sub_msk, -1e9)
                t_sub = z_t_flat[idx_h].gather(1, sub_ids).masked_fill(~sub_msk, -1e9)
                kl_sub = F.kl_div(F.log_softmax(s_sub, dim=-1), F.softmax(t_sub, dim=-1), reduction="batchmean")
            else:
                kl_sub = torch.tensor(0.0, device=device)

            # 3. Projected Pre-Head Hidden-State Alignment (1024 -> 2560)
            u_s_proj = feat_proj(u_s).float()
            u_t_f = u_t.float()
            feat_cos = 1.0 - F.cosine_similarity(u_s_proj, u_t_f, dim=-1).mean()
            feat_l1 = F.smooth_l1_loss(u_s_proj, u_t_f)

            loss = 0.60 * kl_full + 0.80 * kl_sub + 1.80 * feat_cos + 0.35 * feat_l1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(feat_proj.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 200 == 0 or step == args.steps:
            dt = time.perf_counter() - t0
            p_ok, p_tot = evaluate_probes(student, c_tok, cos_cache, sin_cache, device) if (step == 1 or step % 400 == 0 or step == args.steps) else (-1, 0)
            p_str = f"{p_ok}/{p_tot}" if p_ok >= 0 else "..."
            print(f"[Step {step:4d}/{args.steps}] loss={loss.item():.4f} (kl_full={kl_full.item():.3f}, kl_sub={kl_sub.item():.3f}, cos={1.0 - feat_cos.item():.4f}) | probes={p_str} | {dt:.1f}s", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": student.embed_weight.detach().to(torch.float16).cpu().contiguous(),
        "model.norm.weight": student.norm.weight.detach().to(torch.float16).cpu().contiguous(),
    }
    for idx, blk in enumerate(student.layers):
        p = f"model.layers.{idx}"
        export_tensors[f"{p}.input_layernorm.weight"] = blk.input_layernorm.weight.detach().to(torch.float16).cpu().contiguous()
        export_tensors[f"{p}.post_attention_layernorm.weight"] = blk.post_attention_layernorm.weight.detach().to(torch.float16).cpu().contiguous()
        export_tensors[f"{p}.self_attn.q_norm.weight"] = blk.q_norm.weight.detach().to(torch.float16).cpu().contiguous()
        export_tensors[f"{p}.self_attn.k_norm.weight"] = blk.k_norm.weight.detach().to(torch.float16).cpu().contiguous()
        for name, mod in [
            ("self_attn.q_proj", blk.q_proj),
            ("self_attn.k_proj", blk.k_proj),
            ("self_attn.v_proj", blk.v_proj),
            ("self_attn.o_proj", blk.o_proj),
            ("mlp.gate_proj", blk.gate_proj),
            ("mlp.up_proj", blk.up_proj),
            ("mlp.down_proj", blk.down_proj),
        ]:
            export_tensors[f"{p}.{name}.weight"] = mod.weight.detach().to(torch.float16).cpu().contiguous()
            if mod.bias is not None:
                export_tensors[f"{p}.{name}.bias"] = mod.bias.detach().to(torch.float16).cpu().contiguous()

    save_file(export_tensors, str(out_dir / "student_fp16.safetensors"))
    (out_dir / "vocab.json").write_text(json.dumps(vocab_strings, ensure_ascii=False), encoding="utf-8")
    out_cfg = {
        "vocab_size": len(vocab_strings),
        "hidden_size": orig_cfg["hidden_size"],
        "intermediate_size": orig_cfg["intermediate_size"],
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": orig_cfg["num_attention_heads"],
        "num_key_value_heads": orig_cfg["num_key_value_heads"],
        "head_dim": orig_cfg["head_dim"],
        "rms_norm_eps": float(orig_cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(orig_cfg.get("rope_theta", 1000000.0)),
        "tie_word_embeddings": True,
    }
    (out_dir / "config.json").write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")
    print(f"Exported distilled {args.num_layers}L/{len(vocab_strings)}-vocab student -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
