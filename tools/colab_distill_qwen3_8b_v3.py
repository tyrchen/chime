#!/usr/bin/env python3
"""
Colab L4/A100 Route-A-Aligned Distillation v3:
Distills `Qwen/Qwen3-8B-Base` (36L, hidden=4096, strong Chinese semantics & spoken nuance) ->
8-Layer / 36,864-vocab Chime Student (`hidden=1024`, ~88 MB 4-bit MLX) with:

1. Exact Route-A Inference Alignment:
   - 50% of training sequences start cleanly at real sentence boundaries prefixed with `"。，"`
     (aligning RoPE pos 0/1/2 sentence-start logits directly with Route A's neutral anchor!).
   - 50% of training sequences are continuous multi-sentence blocks (aligning long-range
     contextual disambiguation like `橘猫...它`, `这一版和...上一版`).
2. Clean Modern Homophone-Subspace KL (`1.10x` weight) + Explicit Grammatical Confusable Clusters
   (`的/地/得`, `在/再`, `他/她/它`, `吧/爸/把`, `做/作/坐`, `那/哪`, `就/久/旧`, `盛/剩/生`,
   `版/班/半/般`, `兵/饼/病/冰/并`).
3. Training-only linear projector (`1024 -> 4096`) for pre-head hidden-state cosine + SmoothL1
   distillation (0 bytes inference overhead).
4. Strictly Zero-Contamination Corpora: excludes ALL `m-a-p/COIG-CQIA` and `clue/clue` subsets
   used in `dev_suite_10k.tsv` and `test_holdout_10k.tsv`!
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import random
import re
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from pypinyin import Style, lazy_pinyin
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

CJK_RE = re.compile(r"^[\u4e00-\u9fa5]+$")
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
]


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
            max_end = min(n, i + 6)
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
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
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
    def __init__(self, weights: dict[str, torch.Tensor], prefix: str, cfg: dict) -> None:
        super().__init__()
        self.n_heads = int(cfg["num_attention_heads"])
        self.n_kv = int(cfg["num_key_value_heads"])
        self.head_dim = int(cfg.get("head_dim", int(cfg["hidden_size"]) // self.n_heads))
        eps = float(cfg.get("rms_norm_eps", 1e-6))

        self.input_norm = RMSNorm(weights[f"{prefix}.input_layernorm.weight"], eps)
        self.post_norm = RMSNorm(weights[f"{prefix}.post_attention_layernorm.weight"], eps)

        def make_lin(name: str) -> nn.Linear:
            w = weights[f"{prefix}.{name}.weight"].float()
            lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
            lin.weight.data.copy_(w)
            return lin

        self.q_proj = make_lin("self_attn.q_proj")
        self.k_proj = make_lin("self_attn.k_proj")
        self.v_proj = make_lin("self_attn.v_proj")
        self.o_proj = make_lin("self_attn.o_proj")
        self.q_norm = RMSNorm(weights[f"{prefix}.self_attn.q_norm.weight"], eps)
        self.k_norm = RMSNorm(weights[f"{prefix}.self_attn.k_norm.weight"], eps)

        self.gate_proj = make_lin("mlp.gate_proj")
        self.up_proj = make_lin("mlp.up_proj")
        self.down_proj = make_lin("mlp.down_proj")

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
    def __init__(self, weights: dict[str, torch.Tensor], embed_pruned: torch.Tensor, layer_indices: list[int], cfg: dict) -> None:
        super().__init__()
        self.embed_weight = nn.Parameter(embed_pruned.clone().float())
        self.layers = nn.ModuleList([
            Qwen3Layer(weights, f"model.layers.{li}", cfg) for li in layer_indices
        ])
        self.final_norm = RMSNorm(weights["model.norm.weight"], float(cfg.get("rms_norm_eps", 1e-6)))

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
        save_file(state, str(out_path))


def build_clean_homophone_groups(vocab_strings: list[str], max_group: int = 18) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Maps every CJK token id -> padded tensor of its clean modern-Mandarin homophone class,
    plus a per-token weight multiplier (`2.2x` on critical confusable clusters like 的/地/得,
    在/再, 他/她/它, 吧/爸/把, 就/久/旧, 兵/饼/病/冰/并).
    """
    p2i = {p: i for i, p in enumerate(vocab_strings)}
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
    group_weight = torch.ones((v,), dtype=torch.float32)

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

    # Inject explicit high-priority grammatical & spoken confusable clusters (guarantees every
    # pair like 的/地/得, 在/再, 他/她/它, 吧/爸/把 is always in the exact same subspace!).
    crit_count = 0
    for cluster in CRITICAL_CONFUSABLE_CLUSTERS:
        c_ids = [p2i[w] for w in cluster if w in p2i]
        if len(c_ids) < 2:
            continue
        for tid in c_ids:
            # Merge cluster members into the front of `tid`'s group
            existing = [int(group_ids[tid, j]) for j in range(max_group) if group_mask[tid, j]]
            merged = list(dict.fromkeys(c_ids + existing))[:max_group]
            group_ids[tid].zero_()
            group_mask[tid].zero_()
            for j, peer in enumerate(merged):
                group_ids[tid, j] = peer
                group_mask[tid, j] = True
            group_weight[tid] = 2.2
            crit_count += 1

    print(
        f"Built clean homophone table: {covered} CJK tokens ({crit_count} boosted critical confusables)",
        flush=True,
    )
    return group_ids, group_mask, group_weight


def collect_zero_contamination_corpus(target_chars: int = 60_000_000) -> list[str]:
    """
    Streams diverse modern Chinese prose, spoken dialogue, tech, and encyclopedic corpora.
    Strictly excludes `m-a-p/COIG-CQIA` and `clue/clue` so `dev_suite_10k.tsv` and
    `test_holdout_10k.tsv` stay 100% zero-contamination!
    """
    texts: list[str] = []
    total_chars = 0
    candidate_sources = [
        ("BelleGroup/train_0.5M_CN", None, "train", ["instruction", "output"], 0.30),
        ("opencsg/FineWeb-Edu-Chinese-V2.1", None, "train", ["text"], 0.35),
        ("FreedomIntelligence/Alpaca-GPT4-ZH", None, "train", ["instruction", "output"], 0.15),
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
                snippet = merged[:700].replace("\n\n", "\n")
                texts.append(snippet)
                total_chars += len(snippet)
                source_chars += len(snippet)
                taken += 1
                if source_chars >= source_cap or total_chars >= target_chars:
                    break
            print(f"  -> collected {taken} passages ({source_chars / 1e6:.2f}M chars)", flush=True)
        except Exception as exc:
            print(f"  -> skipped {repo}: {exc}", flush=True)

    random.Random(2026).shuffle(texts)
    print(f"Total corpus: {len(texts)} passages, {total_chars / 1e6:.2f}M chars", flush=True)
    return texts


def pack_route_a_blocks(
    passages: list[str], c_tok: CompactTokenizer, seq_len: int
) -> torch.Tensor:
    """
    Packs 50% sentence-start anchored blocks (`"。，"` at pos 0,1 followed by clean sentences,
    matching Route A's exact standalone decoding manifold!) and 50% continuous multi-sentence
    long-context blocks.
    """
    anchor_ids = c_tok.encode("。，")
    sep_id = c_tok.p2i.get("。", 2)

    blocks: list[list[int]] = []
    # 1. Half continuous long-context packed blocks
    flat: list[int] = []
    half_idx = len(passages) // 2
    for text in passages[:half_idx]:
        ids = [i for i in c_tok.encode(text) if i != 1]
        if len(ids) >= 8:
            flat.extend(ids)
            flat.append(sep_id)
    n_cont = len(flat) // seq_len
    for k in range(n_cont):
        blocks.append(flat[k * seq_len : (k + 1) * seq_len])

    # 2. Half `"。，"` sentence-start anchored blocks!
    cur: list[int] = []
    for text in passages[half_idx:]:
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

    packed = torch.tensor(blocks, dtype=torch.long)
    packed = packed[torch.randperm(len(packed), generator=torch.Generator().manual_seed(1337))]
    print(f"Packed {len(packed)} Route-A blocks ({n_cont} long-ctx + {len(packed) - n_cont} anchored, seq_len={seq_len})", flush=True)
    return packed


@torch.no_grad()
def evaluate_probes(student: Qwen3Student, c_tok: CompactTokenizer, cos: torch.Tensor, sin: torch.Tensor, device: torch.device) -> tuple[int, int]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--init-base", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=9000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2.6e-4)
    parser.add_argument("--corpus-chars", type=int, default=60_000_000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vocab_strings: list[str] = json.loads(args.vocab.read_text(encoding="utf-8"))
    c_tok = CompactTokenizer(vocab_strings)

    group_ids, group_mask, group_weight = build_clean_homophone_groups(vocab_strings)
    group_ids_gpu = group_ids.to(device)
    group_mask_gpu = group_mask.to(device)
    group_weight_gpu = group_weight.to(device)

    print(f"Downloading {args.init_base} for 8-layer student init...", flush=True)
    init_snap = pathlib.Path(
        snapshot_download(
            args.init_base,
            allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"],
        )
    )
    orig_cfg = json.loads((init_snap / "config.json").read_text(encoding="utf-8"))
    hf_tok = AutoTokenizer.from_pretrained(str(init_snap))
    bos_id = 151643
    new_to_old = [bos_id, bos_id]
    for piece in vocab_strings[2:]:
        ids = hf_tok.encode(piece, add_special_tokens=False)
        new_to_old.append(ids[0] if len(ids) == 1 else bos_id)
    idx_cpu = torch.tensor(new_to_old, dtype=torch.long)
    new_to_old_gpu = idx_cpu.to(device)

    raw_st = load_file(str(init_snap / "model.safetensors"))
    embed_pruned = raw_st["model.embed_tokens.weight"].index_select(0, idx_cpu).float().contiguous()
    selected_layers = [0, 3, 7, 11, 15, 19, 23, 27] if args.num_layers == 8 else [0, 5, 11, 16, 22, 27]
    student = Qwen3Student(raw_st, embed_pruned, selected_layers, orig_cfg).to(device)
    del raw_st
    gc.collect()

    print(f"Loading strong teacher {args.teacher} in bfloat16...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_lm_head = teacher.get_output_embeddings().weight.detach().index_select(0, new_to_old_gpu).contiguous()
    if hasattr(teacher, "lm_head"):
        del teacher.lm_head
    torch.cuda.empty_cache()

    t_hidden = int(teacher.config.hidden_size)
    s_hidden = int(orig_cfg["hidden_size"])
    feat_proj = nn.Linear(s_hidden, t_hidden, bias=False).to(device)
    print(f"Teacher hidden={t_hidden} -> Student hidden={s_hidden} (projector {s_hidden}->{t_hidden})", flush=True)

    cos_cache, sin_cache = build_rope_cache(512, orig_cfg["head_dim"], float(orig_cfg.get("rope_theta", 1000000.0)), device)

    passages = collect_zero_contamination_corpus(target_chars=args.corpus_chars)
    packed = pack_route_a_blocks(passages, c_tok, args.seq_len)
    val_size = min(256, max(32, len(packed) // 25))
    train_seqs = packed[val_size:]

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
    warmup = min(250, args.steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup) if s < warmup else 0.04 + 0.96 * 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, args.steps - warmup))),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    student_cfg = {
        "model_type": "qwen3",
        "vocab_size": len(vocab_strings),
        "hidden_size": int(orig_cfg["hidden_size"]),
        "intermediate_size": int(orig_cfg["intermediate_size"]),
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": int(orig_cfg["num_attention_heads"]),
        "num_key_value_heads": int(orig_cfg["num_key_value_heads"]),
        "head_dim": int(orig_cfg["head_dim"]),
        "rms_norm_eps": float(orig_cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(orig_cfg.get("rope_theta", 1000000.0)),
        "tie_word_embeddings": True,
    }
    (args.out_dir / "config.json").write_text(json.dumps(student_cfg, indent=2), encoding="utf-8")
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab_strings, ensure_ascii=False), encoding="utf-8")

    t0 = time.perf_counter()
    cursor = 0
    student.train()
    feat_proj.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        acc_kl_f = 0.0
        acc_kl_s = 0.0
        acc_cos = 0.0

        for _ in range(args.grad_accum):
            if cursor + args.batch_size > len(train_seqs):
                train_seqs = train_seqs[torch.randperm(len(train_seqs))]
                cursor = 0
            batch_new = train_seqs[cursor : cursor + args.batch_size].to(device, non_blocking=True)
            cursor += args.batch_size
            batch_old = new_to_old_gpu[batch_new]

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                u_t = teacher.model(input_ids=batch_old, use_cache=False).last_hidden_state
                z_t = F.linear(u_t, teacher_lm_head)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                u_s = student(batch_new, cos_cache, sin_cache)
                z_s = F.linear(u_s, student.embed_weight)

                z_s_flat = z_s[:, :-1, :].reshape(-1, len(vocab_strings)).float()
                z_t_flat = z_t[:, :-1, :].reshape(-1, len(vocab_strings)).float()
                inv_t = 1.0 / 1.25
                kl_full = (
                    F.kl_div(
                        F.log_softmax(z_s_flat * inv_t, dim=-1),
                        F.softmax(z_t_flat * inv_t, dim=-1),
                        reduction="batchmean",
                    )
                    * (1.25 * 1.25)
                )

                targets = batch_new[:, 1:].reshape(-1)
                g_ids = group_ids_gpu[targets]
                g_msk = group_mask_gpu[targets]
                has_group = g_msk.any(dim=-1)
                if has_group.any():
                    idx_h = has_group.nonzero(as_tuple=False).squeeze(-1)
                    sub_ids = g_ids[idx_h]
                    sub_msk = g_msk[idx_h]
                    sub_w = group_weight_gpu[targets[idx_h]]
                    s_sub = z_s_flat[idx_h].gather(1, sub_ids).masked_fill(~sub_msk, -1e9)
                    t_sub = z_t_flat[idx_h].gather(1, sub_ids).masked_fill(~sub_msk, -1e9)
                    per_tok_kl = F.kl_div(
                        F.log_softmax(s_sub, dim=-1),
                        F.softmax(t_sub, dim=-1),
                        reduction="none",
                    ).sum(dim=-1)
                    kl_sub = (per_tok_kl * sub_w).sum() / sub_w.sum().clamp_min(1.0)
                else:
                    kl_sub = torch.tensor(0.0, device=device)

                u_s_proj = feat_proj(u_s).float()
                u_t_f = u_t.float()
                feat_cos = 1.0 - F.cosine_similarity(u_s_proj, u_t_f, dim=-1).mean()
                feat_l1 = F.smooth_l1_loss(u_s_proj, u_t_f)

                loss = (0.55 * kl_full + 1.10 * kl_sub + 1.60 * feat_cos + 0.30 * feat_l1) / args.grad_accum

            loss.backward()
            acc_loss += float(loss.item())
            acc_kl_f += float(kl_full.item()) / args.grad_accum
            acc_kl_s += float(kl_sub.item()) / args.grad_accum
            acc_cos += float(1.0 - feat_cos.item()) / args.grad_accum

        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(feat_proj.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            dt = time.perf_counter() - t0
            p_ok, p_tot = (
                evaluate_probes(student, c_tok, cos_cache, sin_cache, device)
                if (step == 1 or step % 500 == 0 or step == args.steps)
                else (-1, 0)
            )
            p_str = f"{p_ok}/{p_tot}" if p_ok >= 0 else "..."
            print(
                f"[Step {step:5d}/{args.steps}] loss={acc_loss:.4f} "
                f"(kl_full={acc_kl_f:.3f}, kl_sub={acc_kl_s:.3f}, cos={acc_cos:.4f}) "
                f"| probes={p_str} | {dt:.1f}s",
                flush=True,
            )

        if step in (2500, 4000) or step == args.steps:
            student.export_safetensors(args.out_dir / "student_fp16.safetensors")
            print(f"Saved checkpoint at step {step} -> {args.out_dir / 'student_fp16.safetensors'}", flush=True)

    print("DONE v3 distillation!", flush=True)


if __name__ == "__main__":
    main()
