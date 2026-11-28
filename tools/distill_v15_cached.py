#!/usr/bin/env python3
"""
Chime v0.15 Offline-Cached High-Step Multi-Architecture Distillation Pipeline.

Works on Colab A100 (single or 2x parallel), Modal H100, or any CUDA GPU.

Subcommands:
  1. `prep`          Pack zero-contamination Chinese corpus into Route-A aligned token blocks
                     (82% continuous multi-sentence + 18% "。，" sentence-start anchored),
                     and build the homophone subspace lookup table from `vocab.json` +
                     `token_pinyin.json`.
  2. `diagnose-bi`   Data-driven Block Influence (1 - cos(h_in, h_out)) per layer AND
                     activation-weighted MLP neuron salience on `Qwen3-0.6B-Base` over
                     held-out Chinese calibration blocks (~20s on A100). Eliminates
                     arbitrary hardcoded layer indices!
  3. `cache-teacher` One-time `Qwen3-14B-Base` bf16 forward pass caching per next-token:
                       - Top-32 vocab IDs (uint16) + logits (fp16)
                       - Exact full-vocab logsumexp at T=1.25 (fp16) for zero-bias KL
                       - Exact logits (fp16) for all 16 same-pinyin homophone peers
                     Total storage: only 162 B/token (~9.7 GB for 60M tokens).
                     Supports `--shard-idx` / `--num-shards` for parallel multi-GPU caching!
  4. `train-student` High-step (20K-30K steps) student distillation WITHOUT loading the 14B
                     teacher (~10-12 GB VRAM, 3.5-4.5x faster). Supports:
                       - BI-selected layer subsets (e.g. 16L, 18L)
                       - Salience-pruned MLP width (e.g. 28L MLP 3072 -> 1792, or 18L MLP 2048)
                       - Automatic resumable checkpoints + Drive/Volume sync + live IME probes.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pathlib
import random
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")

CRITICAL_CONFUSABLE_CLUSTERS: list[list[str]] = [
    ["的", "地", "得"],
    ["在", "再"],
    ["他", "她", "它"],
    ["吧", "爸", "把", "八", "拔"],
    ["做", "作", "坐", "座"],
    ["那", "哪"],
    ["是", "时", "事", "试", "实", "使"],
    ["就", "久", "旧", "救"],
    ["带", "代", "袋", "大", "戴", "待"],
    ["像", "象", "相", "香", "想", "向"],
    ["盛", "剩", "生", "声", "胜"],
    ["版", "班", "半", "般", "办", "板"],
    ["写", "些", "谢", "血", "鞋"],
    ["兵", "饼", "病", "冰", "并"],
    ["清", "轻", "青", "情", "请"],
    ["推", "退", "腿"],
    ["消", "小", "笑", "校", "效"],
    ["进", "近", "尽", "禁", "仅"],
    ["已", "以", "一", "意", "易"],
    ["须", "需", "许", "续"],
    ["练", "炼", "联", "连"],
    ["账", "帐", "涨", "长", "张"],
    ["支", "只", "知", "之", "直", "值", "指"],
    ["气", "器", "起", "期", "其", "齐"],
    ["游", "有", "又", "由", "右", "油"],
    ["到", "倒", "道", "导", "岛"],
    ["见", "建", "件", "间", "健", "检", "简"],
    ["买", "卖", "迈", "脉"],
    ["签", "欠", "前", "千", "迁"],
    ["唱", "尝", "长", "常", "场", "厂"],
    ["种", "中", "重", "终", "钟"],
    ["服", "附", "付", "副", "幅", "复", "福"],
    ["盖", "改", "该", "概"],
    ["权利", "权力"],
    ["反应", "反映"],
    ["必须", "必需"],
    ["以致", "以至"],
    ["交代", "交待"],
    ["原型", "原形"],
]

TOP_K = 32
SUB_K = 16
TEMP_CACHE = 1.25


def is_all_cjk(s: str) -> bool:
    return bool(s) and all("\u4e00" <= ch <= "\u9fa5" for ch in s)


class CompactTokenizer:
    def __init__(self, vocab_strings: list[str]) -> None:
        self.vocab = vocab_strings
        self.p2i = {p: i for i, p in enumerate(vocab_strings)}
        self.unk = 1

    def encode(self, text: str) -> list[int]:
        chars = list(text)
        out: list[int] = []
        i = 0
        n = len(chars)
        while i < n:
            matched = None
            max_end = min(n, i + 16)
            for end in range(max_end, i, -1):
                piece = "".join(chars[i:end])
                tid = self.p2i.get(piece)
                if tid is not None:
                    matched = (tid, end)
                    break
            if matched is not None:
                out.append(matched[0])
                i = matched[1]
            else:
                out.append(self.unk)
                i += 1
        return out


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone().float())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(orig_dtype)


def build_rope_cache(max_seq: int, head_dim: int, theta: float, device: torch.device):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(max_seq, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (x.float() * cos + rot.float() * sin).to(x.dtype)


class Qwen3Layer(nn.Module):
    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        prefix: str,
        cfg: dict,
        mlp_keep_idx: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = int(cfg["num_attention_heads"])
        self.n_kv = int(cfg["num_key_value_heads"])
        self.head_dim = int(cfg.get("head_dim", int(cfg["hidden_size"]) // self.n_heads))
        eps = float(cfg.get("rms_norm_eps", 1e-6))

        self.input_norm = RMSNorm(weights[f"{prefix}.input_layernorm.weight"], eps)
        self.post_norm = RMSNorm(weights[f"{prefix}.post_attention_layernorm.weight"], eps)

        def make_lin(w: torch.Tensor) -> nn.Linear:
            w = w.float()
            lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
            lin.weight.data.copy_(w)
            return lin

        self.q_proj = make_lin(weights[f"{prefix}.self_attn.q_proj.weight"])
        self.k_proj = make_lin(weights[f"{prefix}.self_attn.k_proj.weight"])
        self.v_proj = make_lin(weights[f"{prefix}.self_attn.v_proj.weight"])
        self.o_proj = make_lin(weights[f"{prefix}.self_attn.o_proj.weight"])
        self.q_norm = RMSNorm(weights[f"{prefix}.self_attn.q_norm.weight"], eps)
        self.k_norm = RMSNorm(weights[f"{prefix}.self_attn.k_norm.weight"], eps)

        w_gate = weights[f"{prefix}.mlp.gate_proj.weight"]
        w_up = weights[f"{prefix}.mlp.up_proj.weight"]
        w_down = weights[f"{prefix}.mlp.down_proj.weight"]
        if mlp_keep_idx is not None:
            idx = torch.tensor(mlp_keep_idx, dtype=torch.long)
            w_gate = w_gate.index_select(0, idx)
            w_up = w_up.index_select(0, idx)
            w_down = w_down.index_select(1, idx)

        self.gate_proj = make_lin(w_gate)
        self.up_proj = make_lin(w_up)
        self.down_proj = make_lin(w_down)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.input_norm(x)
        q = self.q_norm(self.q_proj(h).view(b, t, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(h).view(b, t, self.n_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:, :, :t, :], sin[:, :, :t, :])
        k = apply_rope(k, cos[:, :, :t, :], sin[:, :, :t, :])

        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        x = x + self.o_proj(attn)
        m = self.post_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(m)) * self.up_proj(m))
        return x


class Qwen3Student(nn.Module):
    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        embed_pruned: torch.Tensor,
        layer_indices: list[int],
        cfg: dict,
        mlp_keep_by_orig_layer: dict[int, list[int]] | None = None,
    ) -> None:
        super().__init__()
        self.embed_weight = nn.Parameter(embed_pruned.clone().float())
        self.layers = nn.ModuleList(
            [
                Qwen3Layer(
                    weights,
                    f"model.layers.{li}",
                    cfg,
                    mlp_keep_idx=(
                        mlp_keep_by_orig_layer[li]
                        if mlp_keep_by_orig_layer is not None and li in mlp_keep_by_orig_layer
                        else None
                    ),
                )
                for li in layer_indices
            ]
        )
        self.final_norm = RMSNorm(
            weights["model.norm.weight"], float(cfg.get("rms_norm_eps", 1e-6))
        )

    def forward(self, input_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = F.embedding(input_ids, self.embed_weight)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.final_norm(x)

    def export_safetensors(self, out_path: pathlib.Path) -> None:
        state: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": self.embed_weight.detach().to(torch.float16).cpu().contiguous(),
            "model.norm.weight": self.final_norm.weight.detach().to(torch.float16).cpu().contiguous(),
        }
        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}"
            state[f"{p}.input_layernorm.weight"] = layer.input_norm.weight.detach().to(torch.float16).cpu().contiguous()
            state[f"{p}.post_attention_layernorm.weight"] = layer.post_norm.weight.detach().to(torch.float16).cpu().contiguous()
            state[f"{p}.self_attn.q_proj.weight"] = layer.q_proj.weight.detach().to(torch.float16).cpu().contiguous()
            state[f"{p}.self_attn.k_proj.weight"] = layer.k_proj.weight.detach().to(torch.float16).cpu().contiguous()
            state[f"{p}.self_attn.v_proj.weight"] = layer.v_proj.weight.detach().to(torch.float16).cpu().contiguous()
            state[f"{p}.self_attn.o_proj.weight"] = layer.o_proj.weight.detach().to(torch.float16).cpu().contiguous()
            state[f"{p}.self_attn.q_norm.weight"] = layer.q_norm.weight.detach().to(torch.float16).cpu().contiguous()
            state[f"{p}.self_attn.k_norm.weight"] = layer.k_norm.weight.detach().to(torch.float16).cpu().contiguous()
            state[f"{p}.mlp.gate_proj.weight"] = layer.gate_proj.weight.detach().to(torch.float16).cpu().contiguous()
            state[f"{p}.mlp.up_proj.weight"] = layer.up_proj.weight.detach().to(torch.float16).cpu().contiguous()
            state[f"{p}.mlp.down_proj.weight"] = layer.down_proj.weight.detach().to(torch.float16).cpu().contiguous()
        tmp_path = out_path.with_suffix(".tmp.safetensors")
        save_file(state, str(tmp_path))
        tmp_path.replace(out_path)


def build_homophone_table(
    vocab_strings: list[str],
    token_pinyin_path: pathlib.Path | None,
    max_group: int = SUB_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds exact homophone peer groups (`[V, SUB_K]`) using `token_pinyin.json` (which includes
    polyphone readings and multi-char words!) with `pypinyin` fallback, plus boosted critical
    grammatical & spoken confusable clusters.
    """
    p2i = {p: i for i, p in enumerate(vocab_strings)}
    by_py: dict[str, list[int]] = defaultdict(list)

    if token_pinyin_path is not None and token_pinyin_path.exists():
        tp = json.loads(token_pinyin_path.read_text(encoding="utf-8"))
        for tid, readings in tp.get("cjk", []):
            if tid < 2 or tid >= len(vocab_strings):
                continue
            piece = vocab_strings[tid]
            if not (1 <= len(piece) <= 4) or not is_all_cjk(piece):
                continue
            for syls in readings:
                py = "".join(syls)
                if py and py.isalpha():
                    by_py[py].append(tid)
        print(f"Loaded homophone readings from {token_pinyin_path} ({len(by_py):,} pinyin keys)", flush=True)
    else:
        from pypinyin import Style, lazy_pinyin

        for tid, piece in enumerate(vocab_strings):
            if tid < 2 or not (1 <= len(piece) <= 4) or not is_all_cjk(piece):
                continue
            py = "".join(lazy_pinyin(piece, style=Style.NORMAL))
            if py and py.isalpha():
                by_py[py].append(tid)
        print(f"Built homophone readings via pypinyin ({len(by_py):,} pinyin keys)", flush=True)

    v = len(vocab_strings)
    group_ids = torch.zeros((v, max_group), dtype=torch.long)
    group_mask = torch.zeros((v, max_group), dtype=torch.bool)
    group_weight = torch.ones((v,), dtype=torch.float32)

    # Deduplicate and keep highest-frequency (lowest token id in vocab_36k) peers per pinyin
    peers_per_tid: dict[int, list[int]] = defaultdict(list)
    for tids in by_py.values():
        uniq = sorted(set(tids))
        if len(uniq) < 2:
            continue
        top_peers = uniq[:max_group]
        for tid in top_peers:
            peers_per_tid[tid].extend(top_peers)

    covered = 0
    for tid, raw_peers in peers_per_tid.items():
        # Always put `tid` itself first, then highest-frequency peers
        ordered = [tid] + [p for p in dict.fromkeys(raw_peers) if p != tid]
        ordered = ordered[:max_group]
        if len(ordered) < 2:
            continue
        for j, peer in enumerate(ordered):
            group_ids[tid, j] = peer
            group_mask[tid, j] = True
        covered += 1

    crit_count = 0
    for cluster in CRITICAL_CONFUSABLE_CLUSTERS:
        c_ids = [p2i[w] for w in cluster if w in p2i]
        if len(c_ids) < 2:
            continue
        for tid in c_ids:
            existing = [int(group_ids[tid, j]) for j in range(max_group) if group_mask[tid, j]]
            merged = list(dict.fromkeys([tid] + c_ids + existing))[:max_group]
            group_ids[tid].zero_()
            group_mask[tid].zero_()
            for j, peer in enumerate(merged):
                group_ids[tid, j] = peer
                group_mask[tid, j] = True
            group_weight[tid] = 2.2
            crit_count += 1

    print(
        f"Homophone table ready: {covered:,} CJK tokens covered ({crit_count} critical boosted), SUB_K={max_group}",
        flush=True,
    )
    return group_ids, group_mask, group_weight


def build_new_to_old_map(vocab_strings: list[str], init_base_dir: pathlib.Path) -> torch.Tensor:
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(str(init_base_dir))
    bos_id = 151643
    new_to_old = [bos_id, bos_id]
    for piece in vocab_strings[2:]:
        ids = hf_tok.encode(piece, add_special_tokens=False)
        new_to_old.append(ids[0] if len(ids) == 1 else bos_id)
    return torch.tensor(new_to_old, dtype=torch.long)


def collect_zero_contamination_corpus(target_chars: int = 120_000_000) -> list[str]:
    """
    Streams diverse modern Chinese prose, dialogue, tech, and encyclopedic corpora.
    Strictly excludes `m-a-p/COIG-CQIA` and `clue/clue` so `dev_suite_10k.tsv` and
    `test_holdout_10k.tsv` stay 100% zero-contamination.
    """
    from datasets import load_dataset

    texts: list[str] = []
    total_chars = 0
    candidate_sources = [
        ("BelleGroup/train_0.5M_CN", None, "train", ["instruction", "output"], 0.28),
        ("opencsg/FineWeb-Edu-Chinese-V2.1", None, "train", ["text"], 0.42),
        ("FreedomIntelligence/Alpaca-GPT4-ZH", None, "train", ["instruction", "output"], 0.10),
        ("wikimedia/wikipedia", "20231101.zh", "train", ["text"], 0.20),
    ]
    for repo, subset, split, fields, frac in candidate_sources:
        if total_chars >= target_chars:
            break
        source_cap = int(target_chars * frac)
        try:
            print(f"Streaming {repo} ({subset or 'default'}, cap={source_cap / 1e6:.1f}M chars)...", flush=True)
            ds = (
                load_dataset(repo, subset, split=split, streaming=True)
                if subset
                else load_dataset(repo, split=split, streaming=True)
            )
            source_chars = 0
            taken = 0
            for row in ds:
                parts = [str(row.get(f, "")).strip() for f in fields if row.get(f)]
                merged = "\n".join(p for p in parts if p)
                if len(merged) < 20:
                    continue
                cjk = sum(1 for c in merged[:300] if "\u4e00" <= c <= "\u9fa5")
                if cjk < 12:
                    continue
                snippet = merged[:800].replace("\n\n", "\n")
                texts.append(snippet)
                total_chars += len(snippet)
                source_chars += len(snippet)
                taken += 1
                if source_chars >= source_cap or total_chars >= target_chars:
                    break
            print(f"  -> collected {taken:,} passages ({source_chars / 1e6:.2f}M chars)", flush=True)
        except Exception as exc:
            print(f"  -> skipped {repo}: {exc}", flush=True)

    random.Random(2026).shuffle(texts)
    print(f"Total corpus: {len(texts):,} passages, {total_chars / 1e6:.2f}M chars", flush=True)
    return texts


@torch.no_grad()
def evaluate_probes(
    student: Qwen3Student,
    c_tok: CompactTokenizer,
    cos: torch.Tensor,
    sin: torch.Tensor,
    device: torch.device,
) -> tuple[int, int]:
    student.eval()
    probes = [
        ("。，", "这件事交给我吧", "这件事交给我爸"),
        ("。，", "还是你想得比较周到", "还是你想的比较周到"),
        ("。，", "他说得非常有道理", "他说的非常有道理"),
        ("。，", "干得漂亮", "干的漂亮"),
        ("。，", "不好意思让你久等了", "不好意思让你就等了"),
        ("。，", "盛了一碗米饭", "剩了一碗米饭"),
        ("。，", "电影院的座位已经选好了", "电影院的作为已经选好了"),
        ("。，", "把手机从口袋里拿出来", "拔手机从口袋里拿出来"),
        ("。，", "方案", "反感"),
        ("。，", "协议", "叶也"),
        ("。如果把人比作食物的话，", "我是一个饼", "我是一个病"),
        ("。我家养了一只橘猫，每天早上跳到床上叫我起床，今天", "它", "他"),
        ("。这一版和", "上一版", "上一般"),
        ("。你把", "这几个都放在一起", "这几个豆放在一起"),
        ("。我如果自己", "写", "些"),
        ("。保护公民合法", "权利", "权力"),
        ("。绝不能滥用", "权力", "权利"),
        ("。，", "请再试一次", "请在试一次"),
        ("。，", "慢慢地走过来", "慢慢的走过来"),
        ("。，", "这个问题必须解决", "这个问题必需解决"),
        ("。，", "部署到生产环境", "部属到生产环境"),
        ("。，", "代码评审已经通过了", "代码平审已经通过了"),
        ("。，", "我们需要重新梳理一下需求", "我们需要重新书理一下需求"),
        ("。，", "银行卡余额不足", "银行卡于额不足"),
    ]
    passed = 0
    for ctx, good, bad in probes:
        ctx_ids = [i for i in c_tok.encode(ctx) if i != 1]
        scores = []
        for cand in (good, bad):
            c_ids = [i for i in c_tok.encode(cand) if i != 1]
            full = torch.tensor([ctx_ids + c_ids], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = student(full, cos, sin)
                logits = F.linear(h, student.embed_weight).float()
            lp = F.log_softmax(logits[0], dim=-1)
            s = sum(float(lp[len(ctx_ids) - 1 + j, c_ids[j]]) for j in range(len(c_ids)))
            scores.append(s)
        if scores[0] > scores[1]:
            passed += 1
    student.train()
    return passed, len(probes)


def cmd_prep(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    work_dir = pathlib.Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    vocab_strings: list[str] = json.loads(pathlib.Path(args.vocab).read_text(encoding="utf-8"))
    c_tok = CompactTokenizer(vocab_strings)

    group_ids, group_mask, group_weight = build_homophone_table(
        vocab_strings,
        pathlib.Path(args.token_pinyin) if args.token_pinyin else None,
        max_group=SUB_K,
    )
    torch.save(
        {
            "group_ids": group_ids,
            "group_mask": group_mask,
            "group_weight": group_weight,
        },
        work_dir / "homophone_table.pt",
    )

    blocks_path = work_dir / "blocks.pt"
    if blocks_path.exists() and not args.force:
        packed = torch.load(blocks_path, weights_only=True)
        print(f"Reusing existing {blocks_path}: shape={tuple(packed.shape)}", flush=True)
        return

    passages = collect_zero_contamination_corpus(target_chars=args.corpus_chars)
    seq_len = args.seq_len
    anchor_ids = c_tok.encode("。，")
    sep_id = c_tok.p2i.get("。", 2)
    split_idx = int(len(passages) * 0.82)

    blocks: list[list[int]] = []
    flat: list[int] = []
    for text in passages[:split_idx]:
        ids = [i for i in c_tok.encode(text) if i != 1]
        if len(ids) >= 8:
            flat.extend(ids)
            flat.append(sep_id)
    n_cont = len(flat) // seq_len
    for k in range(n_cont):
        blocks.append(flat[k * seq_len : (k + 1) * seq_len])

    cur: list[int] = []
    for text in passages[split_idx:]:
        sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if len(s.strip()) >= 6]
        for s in sents:
            s_ids = [i for i in c_tok.encode(s) if i != 1]
            if len(s_ids) < 4:
                continue
            if not cur:
                cur.extend(anchor_ids)
            if len(cur) + len(s_ids) + 1 <= seq_len:
                cur.extend(s_ids)
                cur.append(sep_id)
            else:
                need = seq_len - len(cur)
                if need > 0:
                    cur.extend(s_ids[:need])
                if len(cur) == seq_len:
                    blocks.append(cur)
                cur = list(anchor_ids) + s_ids[: seq_len - len(anchor_ids)]
                if len(cur) == seq_len:
                    blocks.append(cur)
                    cur = []

    if args.max_tokens > 0:
        max_blocks = args.max_tokens // seq_len
        if len(blocks) > max_blocks:
            blocks = blocks[:max_blocks]

    packed = torch.tensor(blocks, dtype=torch.int32)
    perm = torch.randperm(len(packed), generator=torch.Generator().manual_seed(2026))
    packed = packed[perm].contiguous()
    torch.save(packed, blocks_path)
    total_toks = packed.numel()
    print(
        f"Saved {len(packed):,} blocks ({total_toks / 1e6:.2f}M tokens, seq_len={seq_len}) "
        f"in {time.perf_counter() - t0:.1f}s -> {blocks_path}",
        flush=True,
    )


@torch.no_grad()
def cmd_diagnose_bi(args: argparse.Namespace) -> None:
    """
    Measures per-layer Block Influence (ShortGPT: BI_l = 1 - mean cos(h_in, h_out)) AND
    per-layer activation-weighted MLP neuron salience (`mean(|silu(gate)*up|) * ||down_col||_2`)
    on Chinese calibration blocks from `blocks.pt`.
    """
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = pathlib.Path(args.work_dir)
    vocab_strings: list[str] = json.loads(pathlib.Path(args.vocab).read_text(encoding="utf-8"))

    init_snap = pathlib.Path(
        snapshot_download(
            args.init_base,
            allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"],
        )
    )
    orig_cfg = json.loads((init_snap / "config.json").read_text(encoding="utf-8"))
    idx_cpu = build_new_to_old_map(vocab_strings, init_snap)
    raw_st = load_file(str(init_snap / "model.safetensors"))
    embed_pruned = raw_st["model.embed_tokens.weight"].index_select(0, idx_cpu).float().contiguous()

    n_layers = int(orig_cfg["num_hidden_layers"])
    inter_dim = int(orig_cfg["intermediate_size"])
    student = Qwen3Student(raw_st, embed_pruned, list(range(n_layers)), orig_cfg).to(device).eval()
    del raw_st
    gc.collect()

    cos_cache, sin_cache = build_rope_cache(
        512,
        int(orig_cfg["head_dim"]),
        float(orig_cfg.get("rope_theta", 1_000_000.0)),
        device,
    )

    packed = torch.load(work_dir / "blocks.pt", weights_only=True)
    calib = packed[: min(args.calib_blocks, len(packed))].long()

    cos_sums = torch.zeros(n_layers, dtype=torch.float64)
    mlp_act_sums = torch.zeros((n_layers, inter_dim), dtype=torch.float64, device=device)
    total_toks = 0

    bs = 16
    for i in range(0, len(calib), bs):
        batch = calib[i : i + bs].to(device)
        b, t = batch.shape
        total_toks += b * t
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x = F.embedding(batch, student.embed_weight)
            for li, layer in enumerate(student.layers):
                x_in = x.float()
                # Inline forward to capture MLP intermediate activation
                h = layer.input_norm(x)
                q = layer.q_norm(layer.q_proj(h).view(b, t, layer.n_heads, layer.head_dim)).transpose(1, 2)
                k = layer.k_norm(layer.k_proj(h).view(b, t, layer.n_kv, layer.head_dim)).transpose(1, 2)
                v = layer.v_proj(h).view(b, t, layer.n_kv, layer.head_dim).transpose(1, 2)
                q = apply_rope(q, cos_cache[:, :, :t, :], sin_cache[:, :, :t, :])
                k = apply_rope(k, cos_cache[:, :, :t, :], sin_cache[:, :, :t, :])
                if layer.n_kv != layer.n_heads:
                    rep = layer.n_heads // layer.n_kv
                    k = k.repeat_interleave(rep, dim=1)
                    v = v.repeat_interleave(rep, dim=1)
                attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn = attn.transpose(1, 2).contiguous().view(b, t, layer.n_heads * layer.head_dim)
                x_mid = x + layer.o_proj(attn)
                m = layer.post_norm(x_mid)
                act = F.silu(layer.gate_proj(m)) * layer.up_proj(m)
                mlp_act_sums[li] += act.float().abs().sum(dim=(0, 1))
                x = x_mid + layer.down_proj(act)
                x_out = x.float()
                cos_val = F.cosine_similarity(x_in, x_out, dim=-1).sum().item()
                cos_sums[li] += cos_val

    bi_scores = [round(1.0 - float(cos_sums[li] / total_toks), 6) for li in range(n_layers)]
    print("\n=== Layer Block Influence (higher = more essential) ===", flush=True)
    for li, bi in enumerate(bi_scores):
        bar = "#" * int(bi * 80)
        print(f"  Layer {li:2d}: BI={bi:.4f} {bar}", flush=True)

    # Always preserve boundary layers (0, n_layers - 1); select top-(K-2) interior layers by BI
    interior = list(range(1, n_layers - 1))
    interior_sorted_desc = sorted(interior, key=lambda li: bi_scores[li], reverse=True)
    selected_by_k: dict[str, list[int]] = {}
    for k in (14, 16, 18, 20, 22, 24, n_layers):
        if k >= n_layers:
            selected_by_k[str(k)] = list(range(n_layers))
        else:
            keep = sorted([0, n_layers - 1] + interior_sorted_desc[: k - 2])
            selected_by_k[str(k)] = keep
            dropped = [li for li in range(n_layers) if li not in keep]
            print(f"  BI-{k:2d}L keep={keep} | dropped={dropped}", flush=True)

    # Per-layer MLP neuron salience: mean(|act|) * ||down_proj_col||_2
    mlp_ranks: dict[str, list[int]] = {}
    for li, layer in enumerate(student.layers):
        mean_act = (mlp_act_sums[li] / total_toks).float()
        col_norm = layer.down_proj.weight.detach().float().norm(dim=0)
        salience = (mean_act * col_norm).cpu()
        order = torch.argsort(salience, descending=True).tolist()
        mlp_ranks[str(li)] = order

    report = {
        "init_base": args.init_base,
        "num_layers": n_layers,
        "intermediate_size": inter_dim,
        "calib_blocks": len(calib),
        "bi_scores": bi_scores,
        "selected_layers_by_k": selected_by_k,
        "mlp_neuron_rank_desc": mlp_ranks,
    }
    out_path = work_dir / "bi_report.json"
    out_path.write_text(json.dumps(report), encoding="utf-8")
    print(f"Saved BI & MLP salience report in {time.perf_counter() - t0:.1f}s -> {out_path}", flush=True)


@torch.no_grad()
def cmd_cache_teacher(args: argparse.Namespace) -> None:
    """
    Runs one-time bf16 forward pass of `Qwen3-14B-Base` over a shard of `blocks.pt` and writes
    compact memmap arrays (`162 B/token`):
      - `top_ids.npy`    uint16  [N_shard, seq_len-1, 32]
      - `top_logits.npy` float16 [N_shard, seq_len-1, 32]
      - `lse_t.npy`      float16 [N_shard, seq_len-1]       (full-vocab logsumexp of z_t / 1.25)
      - `sub_logits.npy` float16 [N_shard, seq_len-1, 16]   (teacher logits on group_ids[target])
    Resumable at batch granularity via `meta.json`!
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    work_dir = pathlib.Path(args.work_dir)
    vocab_strings: list[str] = json.loads(pathlib.Path(args.vocab).read_text(encoding="utf-8"))
    packed = torch.load(work_dir / "blocks.pt", weights_only=True)
    hom = torch.load(work_dir / "homophone_table.pt", weights_only=True)
    group_ids_gpu = hom["group_ids"].to(device)

    val_size = args.val_blocks
    train_blocks = packed[val_size:]
    n_total = len(train_blocks)
    seq_len = int(train_blocks.shape[1])
    pred_len = seq_len - 1

    shard_idx = args.shard_idx
    num_shards = args.num_shards
    s_start = (n_total * shard_idx) // num_shards
    s_end = (n_total * (shard_idx + 1)) // num_shards
    shard_blocks = train_blocks[s_start:s_end]
    n_shard = len(shard_blocks)

    shard_dir = pathlib.Path(args.cache_dir) / f"shard_{shard_idx:02d}_of_{num_shards:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    meta_path = shard_dir / "meta.json"

    done_blocks = 0
    if meta_path.exists() and not args.force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete") and meta.get("n_shard") == n_shard:
            print(f"Shard {shard_idx}/{num_shards} already complete ({n_shard:,} blocks) -> {shard_dir}", flush=True)
            return
        if meta.get("n_shard") == n_shard:
            done_blocks = int(meta.get("done_blocks", 0))
            print(f"Resuming shard {shard_idx}/{num_shards} from block {done_blocks:,}/{n_shard:,}", flush=True)

    mode = "r+" if done_blocks > 0 and (shard_dir / "top_ids.bin").exists() else "w+"
    top_ids_mm = np.memmap(shard_dir / "top_ids.bin", dtype=np.uint16, mode=mode, shape=(n_shard, pred_len, TOP_K))
    top_logits_mm = np.memmap(shard_dir / "top_logits.bin", dtype=np.float16, mode=mode, shape=(n_shard, pred_len, TOP_K))
    lse_t_mm = np.memmap(shard_dir / "lse_t.bin", dtype=np.float16, mode=mode, shape=(n_shard, pred_len))
    sub_logits_mm = np.memmap(shard_dir / "sub_logits.bin", dtype=np.float16, mode=mode, shape=(n_shard, pred_len, SUB_K))

    init_snap = pathlib.Path(
        snapshot_download(
            args.init_base,
            allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
        )
    )
    new_to_old_gpu = build_new_to_old_map(vocab_strings, init_snap).to(device)

    print(
        f"=== Caching teacher {args.teacher} shard {shard_idx}/{num_shards} "
        f"({n_shard:,} blocks = {n_shard * pred_len / 1e6:.2f}M pred tokens, bs={args.batch_size}) ===",
        flush=True,
    )
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(device)
    teacher.eval()
    teacher_lm_head = teacher.get_output_embeddings().weight.detach().index_select(0, new_to_old_gpu).contiguous()
    if hasattr(teacher, "lm_head"):
        del teacher.lm_head
    torch.cuda.empty_cache()

    inv_t = 1.0 / TEMP_CACHE
    bs = args.batch_size
    t_loop = time.perf_counter()
    processed_since_start = 0

    for pos in range(done_blocks, n_shard, bs):
        end = min(pos + bs, n_shard)
        batch_new = shard_blocks[pos:end].long().to(device, non_blocking=True)
        batch_old = new_to_old_gpu[batch_new]
        targets = batch_new[:, 1:]  # [B, pred_len]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            u_t = teacher.model(input_ids=batch_old, use_cache=False).last_hidden_state[:, :-1, :]
            z_t = F.linear(u_t, teacher_lm_head)  # [B, pred_len, V] bf16

        # Full-vocab logsumexp at T=1.25 in fp32 for zero-bias truncated KL
        lse_t = torch.logsumexp(z_t.float() * inv_t, dim=-1).to(torch.float16)

        # Top-32 token IDs & raw logits
        t_vals, t_idx = torch.topk(z_t, k=TOP_K, dim=-1, sorted=True)

        # Exact homophone subspace raw logits for `group_ids[target]`
        g_ids = group_ids_gpu[targets]  # [B, pred_len, SUB_K]
        s_vals = z_t.gather(-1, g_ids).to(torch.float16)

        top_ids_mm[pos:end] = t_idx.to(torch.int32).cpu().numpy().astype(np.uint16)
        top_logits_mm[pos:end] = t_vals.to(torch.float16).cpu().numpy()
        lse_t_mm[pos:end] = lse_t.cpu().numpy()
        sub_logits_mm[pos:end] = s_vals.cpu().numpy()

        processed_since_start += end - pos
        if (end % (bs * 50) == 0) or end == n_shard:
            top_ids_mm.flush()
            top_logits_mm.flush()
            lse_t_mm.flush()
            sub_logits_mm.flush()
            meta_path.write_text(
                json.dumps(
                    {
                        "teacher": args.teacher,
                        "shard_idx": shard_idx,
                        "num_shards": num_shards,
                        "s_start": s_start,
                        "s_end": s_end,
                        "n_shard": n_shard,
                        "seq_len": seq_len,
                        "pred_len": pred_len,
                        "top_k": TOP_K,
                        "sub_k": SUB_K,
                        "temp": TEMP_CACHE,
                        "done_blocks": end,
                        "complete": end == n_shard,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            dt = max(1e-6, time.perf_counter() - t_loop)
            toks_per_s = (processed_since_start * pred_len) / dt
            eta_min = ((n_shard - end) * pred_len) / max(1.0, toks_per_s) / 60.0
            print(
                f"  [Shard {shard_idx}/{num_shards}] {end:7d}/{n_shard} blocks "
                f"({100.0 * end / n_shard:5.1f}%) | {toks_per_s / 1e3:5.1f}k tok/s | ETA {eta_min:4.1f} min",
                flush=True,
            )

    del teacher, teacher_lm_head
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"=== Shard {shard_idx}/{num_shards} COMPLETE in {(time.perf_counter() - t0) / 60:.1f} min -> {shard_dir} ===",
        flush=True,
    )


class CachedTeacherStore:
    """Zero-copy memmap reader spanning 1 or more teacher cache shards."""

    def __init__(self, blocks: torch.Tensor, cache_dir: pathlib.Path, val_blocks: int) -> None:
        shard_dirs = sorted(p for p in cache_dir.iterdir() if p.is_dir() and (p / "meta.json").exists())
        if not shard_dirs:
            raise FileNotFoundError(f"No teacher cache shards found under {cache_dir}")

        self.shards = []
        self.cum_sizes = [0]
        train_blocks = blocks[val_blocks:]
        for sd in shard_dirs:
            meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
            if not meta.get("complete"):
                raise RuntimeError(f"Incomplete shard: {sd} ({meta.get('done_blocks')}/{meta.get('n_shard')})")
            n_s = int(meta["n_shard"])
            pred_len = int(meta["pred_len"])
            s_start = int(meta["s_start"])
            s_end = int(meta["s_end"])
            self.shards.append(
                {
                    "n": n_s,
                    "blocks": train_blocks[s_start:s_end],
                    "top_ids": np.memmap(sd / "top_ids.bin", dtype=np.uint16, mode="r", shape=(n_s, pred_len, TOP_K)),
                    "top_logits": np.memmap(sd / "top_logits.bin", dtype=np.float16, mode="r", shape=(n_s, pred_len, TOP_K)),
                    "lse_t": np.memmap(sd / "lse_t.bin", dtype=np.float16, mode="r", shape=(n_s, pred_len)),
                    "sub_logits": np.memmap(sd / "sub_logits.bin", dtype=np.float16, mode="r", shape=(n_s, pred_len, SUB_K)),
                }
            )
            self.cum_sizes.append(self.cum_sizes[-1] + n_s)

        self.total_blocks = self.cum_sizes[-1]
        print(f"Mounted {len(self.shards)} teacher cache shard(s): {self.total_blocks:,} blocks ready", flush=True)

    def fetch_contiguous(self, start: int, count: int):
        """Fast zero-scatter contiguous read (blocks.pt is already globally permuted)."""
        if len(self.shards) == 1:
            sh = self.shards[0]
            end = start + count
            return (
                sh["blocks"][start:end].long().pin_memory(),
                torch.from_numpy(np.array(sh["top_ids"][start:end], dtype=np.int64)).pin_memory(),
                torch.from_numpy(np.array(sh["top_logits"][start:end], dtype=np.float16)).pin_memory(),
                torch.from_numpy(np.array(sh["lse_t"][start:end], dtype=np.float16)).pin_memory(),
                torch.from_numpy(np.array(sh["sub_logits"][start:end], dtype=np.float16)).pin_memory(),
            )
        return self.fetch_batch(np.arange(start, start + count, dtype=np.int64))

    def fetch_batch(self, indices: np.ndarray):
        # Fast path: single shard
        if len(self.shards) == 1:
            sh = self.shards[0]
            return (
                sh["blocks"][indices].long(),
                torch.from_numpy(np.array(sh["top_ids"][indices], dtype=np.int64)),
                torch.from_numpy(np.array(sh["top_logits"][indices], dtype=np.float16)),
                torch.from_numpy(np.array(sh["lse_t"][indices], dtype=np.float16)),
                torch.from_numpy(np.array(sh["sub_logits"][indices], dtype=np.float16)),
            )

        b_list, ti_list, tl_list, lse_list, sl_list = [], [], [], [], []
        for s_idx, sh in enumerate(self.shards):
            lo, hi = self.cum_sizes[s_idx], self.cum_sizes[s_idx + 1]
            mask = (indices >= lo) & (indices < hi)
            if not mask.any():
                continue
            loc = indices[mask] - lo
            b_list.append(sh["blocks"][loc].long())
            ti_list.append(torch.from_numpy(np.array(sh["top_ids"][loc], dtype=np.int64)))
            tl_list.append(torch.from_numpy(np.array(sh["top_logits"][loc], dtype=np.float16)))
            lse_list.append(torch.from_numpy(np.array(sh["lse_t"][loc], dtype=np.float16)))
            sl_list.append(torch.from_numpy(np.array(sh["sub_logits"][loc], dtype=np.float16)))
        return (
            torch.cat(b_list, dim=0),
            torch.cat(ti_list, dim=0),
            torch.cat(tl_list, dim=0),
            torch.cat(lse_list, dim=0),
            torch.cat(sl_list, dim=0),
        )


@torch.no_grad()
def evaluate_val_nll(
    student: Qwen3Student,
    val_blocks: torch.Tensor,
    group_ids_gpu: torch.Tensor,
    group_mask_gpu: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (full_vocab_val_nll, homophone_subspace_top1_acc) on held-out val_blocks."""
    student.eval()
    total_nll = 0.0
    total_toks = 0
    hom_ok = 0
    hom_tot = 0
    bs = 16
    for i in range(0, len(val_blocks), bs):
        batch = val_blocks[i : i + bs].long().to(device)
        targets = batch[:, 1:].reshape(-1)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            u_s = student(batch, cos, sin)[:, :-1, :]
            z_s = F.linear(u_s, student.embed_weight).reshape(-1, student.embed_weight.shape[0]).float()
        total_nll += float(F.cross_entropy(z_s, targets, reduction="sum").item())
        total_toks += int(targets.numel())

        g_ids = group_ids_gpu[targets]
        g_msk = group_mask_gpu[targets]
        has_g = g_msk.any(dim=-1)
        if has_g.any():
            idx_h = has_g.nonzero(as_tuple=False).squeeze(-1)
            sub_logits = z_s[idx_h].gather(1, g_ids[idx_h]).masked_fill(~g_msk[idx_h], -1e9)
            # In build_homophone_table, `targets` itself is always placed at slot 0!
            pred_slot = sub_logits.argmax(dim=-1)
            hom_ok += int((pred_slot == 0).sum().item())
            hom_tot += int(idx_h.numel())
    student.train()
    return total_nll / max(1, total_toks), 100.0 * hom_ok / max(1, hom_tot)


def cmd_train_student(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    work_dir = pathlib.Path(args.work_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sync_dir = pathlib.Path(args.sync_dir) if args.sync_dir else None
    if sync_dir is not None:
        sync_dir.mkdir(parents=True, exist_ok=True)

    vocab_strings: list[str] = json.loads(pathlib.Path(args.vocab).read_text(encoding="utf-8"))
    c_tok = CompactTokenizer(vocab_strings)
    vocab_size = len(vocab_strings)

    hom = torch.load(work_dir / "homophone_table.pt", weights_only=True)
    group_ids_gpu = hom["group_ids"].to(device)
    group_mask_gpu = hom["group_mask"].to(device)
    group_weight_gpu = hom["group_weight"].to(device)

    if args.init_student_ckpt:
        ckpt_path = pathlib.Path(args.init_student_ckpt)
        orig_cfg = json.loads((ckpt_path.parent / "config.json").read_text(encoding="utf-8"))
        num_layers = int(orig_cfg["num_hidden_layers"])
        target_inter = int(orig_cfg["intermediate_size"])
        selected_layers = list(range(num_layers))
        raw_st = load_file(str(ckpt_path))
        embed_pruned = raw_st["model.embed_tokens.weight"].float().contiguous()
        student = Qwen3Student(raw_st, embed_pruned, selected_layers, orig_cfg, None).to(device)
        del raw_st
        gc.collect()
        print(
            f"[{args.route_name}] Initialized directly from student checkpoint {ckpt_path} "
            f"({num_layers}L, MLP={target_inter})",
            flush=True,
        )
    else:
        bi_report = json.loads((work_dir / "bi_report.json").read_text(encoding="utf-8"))
        selected_layers = bi_report["selected_layers_by_k"][str(args.num_layers)]

        init_snap = pathlib.Path(
            snapshot_download(
                args.init_base,
                allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"],
            )
        )
        orig_cfg = json.loads((init_snap / "config.json").read_text(encoding="utf-8"))
        orig_inter = int(orig_cfg["intermediate_size"])
        target_inter = args.intermediate_size if args.intermediate_size > 0 else orig_inter
        if target_inter % 64 != 0:
            raise ValueError(f"--intermediate-size ({target_inter}) must be divisible by 64 for 4-bit MLX quantization")

        mlp_keep_by_orig_layer: dict[int, list[int]] | None = None
        if target_inter < orig_inter:
            mlp_keep_by_orig_layer = {}
            for li in selected_layers:
                ranked = bi_report["mlp_neuron_rank_desc"][str(li)]
                mlp_keep_by_orig_layer[li] = sorted(ranked[:target_inter])
            print(f"Salience-pruned MLP width: {orig_inter} -> {target_inter} across {len(selected_layers)} layers", flush=True)

        idx_cpu = build_new_to_old_map(vocab_strings, init_snap)
        raw_st = load_file(str(init_snap / "model.safetensors"))
        embed_pruned = raw_st["model.embed_tokens.weight"].index_select(0, idx_cpu).float().contiguous()
        student = Qwen3Student(raw_st, embed_pruned, selected_layers, orig_cfg, mlp_keep_by_orig_layer).to(device)
        del raw_st
        gc.collect()

    student_cfg = {
        "model_type": "qwen3",
        "vocab_size": vocab_size,
        "hidden_size": int(orig_cfg["hidden_size"]),
        "intermediate_size": target_inter,
        "num_hidden_layers": len(selected_layers),
        "num_attention_heads": int(orig_cfg["num_attention_heads"]),
        "num_key_value_heads": int(orig_cfg["num_key_value_heads"]),
        "head_dim": int(orig_cfg["head_dim"]),
        "rms_norm_eps": float(orig_cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(orig_cfg.get("rope_theta", 1_000_000.0)),
        "tie_word_embeddings": True,
        "distill_v15_meta": {
            "route_name": args.route_name,
            "selected_layers": selected_layers,
            "intermediate_size": target_inter,
            "steps": args.steps,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(student_cfg, indent=2), encoding="utf-8")
    (out_dir / "vocab.json").write_text(json.dumps(vocab_strings, ensure_ascii=False), encoding="utf-8")

    cos_cache, sin_cache = build_rope_cache(
        512,
        int(orig_cfg["head_dim"]),
        float(orig_cfg.get("rope_theta", 1_000_000.0)),
        device,
    )

    packed = torch.load(work_dir / "blocks.pt", weights_only=True)
    val_blocks = packed[: args.val_blocks]
    store = CachedTeacherStore(packed, pathlib.Path(args.cache_dir), args.val_blocks)

    matrix_params = [p for n, p in student.named_parameters() if n != "embed_weight" and p.ndim >= 2]
    norm_params = [p for n, p in student.named_parameters() if n != "embed_weight" and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": matrix_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": norm_params, "lr": args.lr, "weight_decay": 0.0},
            {"params": [student.embed_weight], "lr": args.lr * args.embed_lr_mult, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        fused=True,
    )
    warmup = min(400, max(50, args.steps // 20))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup)
        if s < warmup
        else 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, args.steps - warmup))),
    )

    start_step = 0
    resume_ckpt = out_dir / "train_state.pt"
    if resume_ckpt.exists() and not args.force:
        st_ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        student.load_state_dict(st_ckpt["model"])
        optimizer.load_state_dict(st_ckpt["optimizer"])
        scheduler.load_state_dict(st_ckpt["scheduler"])
        start_step = int(st_ckpt["step"])
        print(f"[{args.route_name}] Resumed from step {start_step}/{args.steps}", flush=True)

    p0_ok, p0_tot = evaluate_probes(student, c_tok, cos_cache, sin_cache, device)
    nll0, hom0 = evaluate_val_nll(student, val_blocks, group_ids_gpu, group_mask_gpu, cos_cache, sin_cache, device)
    print(
        f"=== [{args.route_name}] GPU={torch.cuda.get_device_name(0)} | {len(selected_layers)}L "
        f"MLP={target_inter} | layers={selected_layers} | steps={args.steps} "
        f"bs={args.batch_size}x{args.grad_accum} lr={args.lr} ===",
        flush=True,
    )
    print(
        f"[{args.route_name}] Step {start_step} baseline: probes={p0_ok}/{p0_tot} "
        f"val_nll={nll0:.4f} val_hom_acc={hom0:.2f}%",
        flush=True,
    )

    model_fwd = torch.compile(student) if args.compile else student

    def compute_distill_losses(
        z_s_bf16: torch.Tensor,
        targets: torch.Tensor,
        t_ids: torch.Tensor,
        t_logits: torch.Tensor,
        lse_t: torch.Tensor,
        sub_t_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Single-pass vocab logsumexp at T=1.25 (eliminates redundant fp32 full-vocab copies & 2nd softmax!)
        inv_t_val = 1.0 / TEMP_CACHE
        t_sq_val = TEMP_CACHE * TEMP_CACHE

        s_scaled = z_s_bf16.float() * inv_t_val
        lse_s = torch.logsumexp(s_scaled, dim=-1, keepdim=True)  # [B, L, 1]

        # 1. Exact Bucketed Soft-KL over {Top-32} U {Tail} using exact full-vocab `lse_t`!
        #    Including `p_t_tail * (log p_t_tail - log p_s_tail)` is mathematically critical:
        #    without it, at high-entropy positions (sentence starts, rare words, English/tech)
        #    where Top-32 holds <50% mass, Top-32 tokens get inflated 2-3x while all 36,832
        #    non-Top-32 tail tokens are crushed toward p_s = 0!
        log_p_t_top = t_logits * inv_t_val - lse_t.unsqueeze(-1)
        p_t_top = log_p_t_top.exp()
        log_p_s_top = s_scaled.gather(-1, t_ids) - lse_s
        p_s_top = log_p_s_top.exp()

        p_t_tail = (1.0 - p_t_top.sum(dim=-1)).clamp(min=1e-6, max=1.0)
        p_s_tail = (1.0 - p_s_top.sum(dim=-1)).clamp(min=1e-6, max=1.0)
        kl_top = (p_t_top * (log_p_t_top - log_p_s_top)).sum(dim=-1)
        kl_tail = p_t_tail * (p_t_tail.log() - p_s_tail.log())
        kl_full = (kl_top + kl_tail).mean() * t_sq_val

        # 2. Temperature-scaled ground-truth CE anchor reusing `lse_s` (0 extra 36K softmax!)
        s_tgt = s_scaled.gather(-1, targets.unsqueeze(-1)) - lse_s
        ce_loss = (-s_tgt.squeeze(-1)).mean() * TEMP_CACHE

        # 3. Exact Homophone Subspace KL (`group_ids[target, :SUB_K]`)
        targets_flat = targets.reshape(-1)
        z_s_flat = z_s_bf16.reshape(-1, vocab_size)
        sub_t_flat = sub_t_logits.reshape(-1, SUB_K)

        g_ids = group_ids_gpu[targets_flat]
        g_msk = group_mask_gpu[targets_flat]
        has_g = g_msk.any(dim=-1)
        sub_w = group_weight_gpu[targets_flat] * has_g.float()
        s_sub = z_s_flat.gather(1, g_ids).float().masked_fill(~g_msk, -1e9)
        t_sub = sub_t_flat.masked_fill(~g_msk, -1e9)
        per_tok_kl = F.kl_div(
            F.log_softmax(s_sub, dim=-1),
            F.softmax(t_sub, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        kl_sub = (per_tok_kl * sub_w).sum() / sub_w.sum().clamp_min(1.0)
        return kl_full, kl_sub, ce_loss

    loss_fn = torch.compile(compute_distill_losses) if args.compile else compute_distill_losses

    rng = np.random.default_rng(2026 + start_step)
    bs = args.batch_size
    n_chunks = store.total_blocks // bs
    chunk_order = rng.permutation(n_chunks)
    chunk_offset = int(rng.integers(0, bs)) if store.total_blocks > bs * 2 else 0
    cursor = 0

    def next_chunk_start() -> int:
        nonlocal chunk_order, chunk_offset, cursor
        if cursor >= n_chunks - 1:
            chunk_order = rng.permutation(n_chunks)
            chunk_offset = int(rng.integers(0, bs)) if store.total_blocks > bs * 2 else 0
            cursor = 0
        start = int(chunk_order[cursor]) * bs + chunk_offset
        if start + bs > store.total_blocks:
            start = max(0, store.total_blocks - bs)
        cursor += 1
        return start

    ema_loss = None
    best_hom_acc = hom0
    t_train = time.perf_counter()

    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    next_future = prefetch_pool.submit(store.fetch_contiguous, next_chunk_start(), bs)

    student.train()
    try:
        for step in range(start_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            acc_loss = 0.0
            acc_kl_f = 0.0
            acc_kl_s = 0.0
            acc_ce = 0.0

            for _ in range(args.grad_accum):
                b_new, t_ids, t_logits, lse_t, sub_t_logits = next_future.result()
                next_future = prefetch_pool.submit(store.fetch_contiguous, next_chunk_start(), bs)

                b_new = b_new.to(device, non_blocking=True)
                t_ids = t_ids.to(device, non_blocking=True)
                t_logits = t_logits.to(device, non_blocking=True).float()
                lse_t = lse_t.to(device, non_blocking=True).float()
                sub_t_logits = sub_t_logits.to(device, non_blocking=True).float()

                targets = b_new[:, 1:]  # [B, pred_len]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    u_s = model_fwd(b_new, cos_cache, sin_cache)[:, :-1, :]
                    z_s = F.linear(u_s, student.embed_weight)  # [B, pred_len, V]

                kl_full, kl_sub, ce_loss = loss_fn(z_s, targets, t_ids, t_logits, lse_t, sub_t_logits)
                loss_raw = args.kl_full_w * kl_full + args.kl_sub_w * kl_sub + args.ce_w * ce_loss
                loss = loss_raw / args.grad_accum
                loss.backward()

                acc_loss += float(loss.item())
                acc_kl_f += float(kl_full.item()) / args.grad_accum
                acc_kl_s += float(kl_sub.item()) / args.grad_accum
                acc_ce += float(ce_loss.item()) / args.grad_accum

            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema_loss = acc_loss if ema_loss is None else (0.95 * ema_loss + 0.05 * acc_loss)

            if step in (start_step + 1, start_step + 100) or step % 500 == 0 or step == args.steps:
                dt = time.perf_counter() - t_train
                done_steps = step - start_step
                ms_per_step = dt / max(1, done_steps) * 1000.0
                eta_min = (args.steps - step) * (dt / max(1, done_steps)) / 60.0
                lr_now = scheduler.get_last_lr()[0]

                if step % 1000 == 0 or step == args.steps:
                    p_ok, p_tot = evaluate_probes(student, c_tok, cos_cache, sin_cache, device)
                    val_nll, val_hom = evaluate_val_nll(
                        student, val_blocks, group_ids_gpu, group_mask_gpu, cos_cache, sin_cache, device
                    )
                    extra = f"| probes={p_ok}/{p_tot} val_nll={val_nll:.4f} val_hom={val_hom:.2f}%"
                    if val_hom > best_hom_acc:
                        best_hom_acc = val_hom
                        student.export_safetensors(out_dir / "student_best_fp16.safetensors")
                else:
                    extra = ""

                print(
                    f"[{args.route_name} {step:5d}/{args.steps}] loss={acc_loss:.4f} (ema={ema_loss:.4f}) "
                    f"kl_f={acc_kl_f:.3f} kl_s={acc_kl_s:.3f} ce={acc_ce:.3f} lr={lr_now:.2e} "
                    f"| {ms_per_step:.1f}ms/step ETA={eta_min:.1f}m {extra}",
                    flush=True,
                )

            if step % args.save_every == 0 or step == args.steps:
                fp16_path = out_dir / "student_fp16.safetensors"
                student.export_safetensors(fp16_path)
                torch.save(
                    {
                        "step": step,
                        "model": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    resume_ckpt,
                )
                print(
                    f"[{args.route_name}] Saved checkpoint step={step} -> {fp16_path} "
                    f"({fp16_path.stat().st_size / 1e6:.1f} MB)",
                    flush=True,
                )
                if sync_dir is not None:
                    for fn in ("config.json", "vocab.json", "student_fp16.safetensors"):
                        src = out_dir / fn
                        if src.exists():
                            shutil.copy2(src, sync_dir / fn)
                    print(f"[{args.route_name}] Synced fp16 bundle -> {sync_dir}", flush=True)
    finally:
        prefetch_pool.shutdown(wait=False)

    print(
        f"=== [{args.route_name}] DONE steps={args.steps} best_val_hom={best_hom_acc:.2f}% "
        f"elapsed={(time.perf_counter() - t0) / 60:.1f} min ===",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chime v0.15 Offline-Cached Distillation Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prep")
    p_prep.add_argument("--work-dir", required=True)
    p_prep.add_argument("--vocab", required=True)
    p_prep.add_argument("--token-pinyin", default="")
    p_prep.add_argument("--corpus-chars", type=int, default=120_000_000)
    p_prep.add_argument("--max-tokens", type=int, default=60_000_000)
    p_prep.add_argument("--seq-len", type=int, default=192)
    p_prep.add_argument("--force", action="store_true")

    p_bi = sub.add_parser("diagnose-bi")
    p_bi.add_argument("--work-dir", required=True)
    p_bi.add_argument("--vocab", required=True)
    p_bi.add_argument("--init-base", default="Qwen/Qwen3-0.6B-Base")
    p_bi.add_argument("--calib-blocks", type=int, default=256)

    p_cache = sub.add_parser("cache-teacher")
    p_cache.add_argument("--work-dir", required=True)
    p_cache.add_argument("--cache-dir", required=True)
    p_cache.add_argument("--vocab", required=True)
    p_cache.add_argument("--teacher", default="Qwen/Qwen3-14B-Base")
    p_cache.add_argument("--init-base", default="Qwen/Qwen3-0.6B-Base")
    p_cache.add_argument("--batch-size", type=int, default=32)
    p_cache.add_argument("--val-blocks", type=int, default=512)
    p_cache.add_argument("--shard-idx", type=int, default=0)
    p_cache.add_argument("--num-shards", type=int, default=1)
    p_cache.add_argument("--force", action="store_true")

    p_tr = sub.add_parser("train-student")
    p_tr.add_argument("--route-name", required=True)
    p_tr.add_argument("--work-dir", required=True)
    p_tr.add_argument("--cache-dir", required=True)
    p_tr.add_argument("--out-dir", required=True)
    p_tr.add_argument("--sync-dir", default="")
    p_tr.add_argument("--vocab", required=True)
    p_tr.add_argument("--init-base", default="Qwen/Qwen3-0.6B-Base")
    p_tr.add_argument("--init-student-ckpt", default="")
    p_tr.add_argument("--num-layers", type=int, default=0)
    p_tr.add_argument("--intermediate-size", type=int, default=0)
    p_tr.add_argument("--steps", type=int, default=25000)
    p_tr.add_argument("--batch-size", type=int, default=32)
    p_tr.add_argument("--grad-accum", type=int, default=2)
    p_tr.add_argument("--lr", type=float, default=2.2e-4)
    p_tr.add_argument("--embed-lr-mult", type=float, default=0.12)
    p_tr.add_argument("--kl-full-w", type=float, default=0.80)
    p_tr.add_argument("--kl-sub-w", type=float, default=1.25)
    p_tr.add_argument("--ce-w", type=float, default=0.18)
    p_tr.add_argument("--val-blocks", type=int, default=512)
    p_tr.add_argument("--save-every", type=int, default=5000)
    p_tr.add_argument("--compile", action="store_true")
    p_tr.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.cmd == "prep":
        cmd_prep(args)
    elif args.cmd == "diagnose-bi":
        cmd_diagnose_bi(args)
    elif args.cmd == "cache-teacher":
        cmd_cache_teacher(args)
    elif args.cmd == "train-student":
        cmd_train_student(args)


if __name__ == "__main__":
    main()
