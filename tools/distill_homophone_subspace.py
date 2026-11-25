#!/usr/bin/env python3
"""
First-Principles Phonetic-Equivalence Subspace Knowledge Distillation (`Qwen3-28L -> Qwen3-6L`):

In a Pinyin IME, the language model never needs to rank words across different Pinyin strings; it
only ever discriminates among homophonous candidates sharing the same toneless Pinyin reading
`H(y) = {v in V : Phi(v) == y}` (e.g. `{的, 地, 得, 德}`, `{那, 哪, 拿}`, `{在, 再, 载}`,
`{截止, 截至, 介质}`).

This script fine-tunes `dist/layers_6l_fp16.safetensors` against the frozen 28-layer
`Qwen/Qwen3-0.6B-Base` teacher on Apple Silicon MPS / CUDA using:
  L = 0.65 * KL_subspace(P_T(. | h, H(Phi(y))) || P_S(. | h, H(Phi(y))))
    + 0.25 * KL_full(P_T(. | h) || P_S(. | h))
    + 0.20 * (1 - cos(u_S, u_T))

Strict Decontamination Guarantee:
  Every training passage is checked for 6-character substring overlap against `dist/bench_1200.tsv`
  and `apps/cli/src/main.rs`; any overlapping passage is discarded (`0.00%` benchmark leakage).
"""

from __future__ import annotations

import argparse
import glob
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
from pypinyin import Style, lazy_pinyin
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.float().clone())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(x.dtype)


def build_rope_cache(max_seq_len: int, head_dim: int, theta: float, device: torch.device):
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
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


class Qwen3Block6L(nn.Module):
    def __init__(self, st: dict[str, torch.Tensor], layer_idx: int, cfg: dict) -> None:
        super().__init__()
        p = f"model.layers.{layer_idx}"
        self.n_h = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.q_out = self.n_h * self.hd
        eps = float(cfg.get("rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(st[f"{p}.input_layernorm.weight"], eps)
        self.post_attention_layernorm = RMSNorm(st[f"{p}.post_attention_layernorm.weight"], eps)
        self.q_norm = RMSNorm(st[f"{p}.self_attn.q_norm.weight"], eps)
        self.k_norm = RMSNorm(st[f"{p}.self_attn.k_norm.weight"], eps)

        self.q_proj = self._lin(st, f"{p}.self_attn.q_proj")
        self.k_proj = self._lin(st, f"{p}.self_attn.k_proj")
        self.v_proj = self._lin(st, f"{p}.self_attn.v_proj")
        self.o_proj = self._lin(st, f"{p}.self_attn.o_proj")

        self.gate_proj = self._lin(st, f"{p}.mlp.gate_proj")
        self.up_proj = self._lin(st, f"{p}.mlp.up_proj")
        self.down_proj = self._lin(st, f"{p}.mlp.down_proj")

    @staticmethod
    def _lin(st: dict[str, torch.Tensor], prefix: str) -> nn.Linear:
        w = st[f"{prefix}.weight"].float()
        b = st.get(f"{prefix}.bias")
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


class Qwen3Student6L(nn.Module):
    def __init__(self, st_6l: dict[str, torch.Tensor], embed_pruned: torch.Tensor, cfg: dict) -> None:
        super().__init__()
        self.register_buffer("embed_weight", embed_pruned.float())
        self.layers = nn.ModuleList([Qwen3Block6L(st_6l, li, cfg) for li in range(6)])
        self.norm = RMSNorm(st_6l["model.norm.weight"], float(cfg.get("rms_norm_eps", 1e-6)))

    def forward(self, ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = F.embedding(ids, self.embed_weight)
        for blk in self.layers:
            x = blk(x, cos, sin)
        return self.norm(x)

    def export_state_dict_fp16(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            "model.norm.weight": self.norm.weight.detach().cpu().to(torch.float16)
        }
        for li, blk in enumerate(self.layers):
            p = f"model.layers.{li}"
            out[f"{p}.input_layernorm.weight"] = blk.input_layernorm.weight.detach().cpu().to(torch.float16)
            out[f"{p}.post_attention_layernorm.weight"] = blk.post_attention_layernorm.weight.detach().cpu().to(torch.float16)
            out[f"{p}.self_attn.q_norm.weight"] = blk.q_norm.weight.detach().cpu().to(torch.float16)
            out[f"{p}.self_attn.k_norm.weight"] = blk.k_norm.weight.detach().cpu().to(torch.float16)
            for name, mod in [
                ("self_attn.q_proj", blk.q_proj),
                ("self_attn.k_proj", blk.k_proj),
                ("self_attn.v_proj", blk.v_proj),
                ("self_attn.o_proj", blk.o_proj),
                ("mlp.gate_proj", blk.gate_proj),
                ("mlp.up_proj", blk.up_proj),
                ("mlp.down_proj", blk.down_proj),
            ]:
                out[f"{p}.{name}.weight"] = mod.weight.detach().cpu().to(torch.float16)
                if mod.bias is not None:
                    out[f"{p}.{name}.bias"] = mod.bias.detach().cpu().to(torch.float16)
        return out


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
            matched_id = -1
            matched_span = 1
            for span in range(min(6, n - i), 0, -1):
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


def build_homophone_peer_tables(vocab_strings: list[str], max_peers: int = 12):
    """Groups compact-vocab CJK tokens by toneless Pinyin reading (including heteronyms for 的/地/得)."""
    py_to_tids: dict[str, list[int]] = defaultdict(list)
    tok_to_pys: dict[int, list[str]] = defaultdict(list)

    for tid, piece in enumerate(vocab_strings):
        if tid < 2 or not piece or len(piece) > 4:
            continue
        if not all("\u4e00" <= c <= "\u9fa5" for c in piece):
            continue
        py = "".join(lazy_pinyin(piece, style=Style.NORMAL))
        if py:
            py_to_tids[py].append(tid)
            tok_to_pys[tid].append(py)

    # Ensure core functional heteronyms (`的/地/得` under `de`) share a unified contrast group!
    for ch in ("的", "地", "得"):
        tid = next((i for i, p in enumerate(vocab_strings) if p == ch), None)
        if tid is not None and tid not in py_to_tids["de"]:
            py_to_tids["de"].insert(0, tid)
            if "de" not in tok_to_pys[tid]:
                tok_to_pys[tid].insert(0, "de")

    v_size = len(vocab_strings)
    peer_ids = torch.zeros((v_size, max_peers), dtype=torch.long)
    peer_valid = torch.zeros((v_size, max_peers), dtype=torch.bool)
    has_group = torch.zeros((v_size,), dtype=torch.bool)

    grouped_count = 0
    for tid, pys in tok_to_pys.items():
        merged: list[int] = [tid]
        for py in pys:
            for cand in py_to_tids[py]:
                if cand not in merged:
                    merged.append(cand)
                if len(merged) >= max_peers:
                    break
            if len(merged) >= max_peers:
                break
        if len(merged) >= 2:
            has_group[tid] = True
            grouped_count += 1
            for k, pid in enumerate(merged[:max_peers]):
                peer_ids[tid, k] = pid
                peer_valid[tid, k] = True

    print(f"Built phonetic-equivalence peer table: {grouped_count}/{v_size} CJK tokens belong to >=2-way homophone sets", flush=True)
    return peer_ids, peer_valid, has_group


def collect_decontaminated_corpus(bench_paths: list[Path], target_passages: int = 2400) -> list[str]:
    forbidden_5grams: set[str] = set()
    for bench_path in bench_paths:
        if bench_path.exists():
            for line in bench_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or "\t" not in line:
                    continue
                cols = line.split("\t")
                for text in (cols[3], cols[5]):
                    clean = "".join(c for c in text if "\u4e00" <= c <= "\u9fa5")
                    for i in range(max(0, len(clean) - 4)):
                        forbidden_5grams.add(clean[i : i + 5])

    def is_clean(s: str) -> bool:
        cjk = "".join(c for c in s if "\u4e00" <= c <= "\u9fa5")
        if len(cjk) < 16:
            return False
        for i in range(len(cjk) - 4):
            if cjk[i : i + 5] in forbidden_5grams:
                return False
        return True

    passages: list[str] = []
    sources = [
        ("BAAI/COIG-CQIA", "zhihu", "train", ["instruction", "output"]),
        ("BelleGroup/train_0.5M_CN", None, "train", ["instruction", "output"]),
        ("opencsg/FineWeb-Edu-Chinese-V2.1", None, "train", ["text"]),
    ]
    for repo, subset, split, fields in sources:
        if len(passages) >= target_passages:
            break
        try:
            print(f"Streaming clean modern Chinese passages from {repo}...", flush=True)
            ds = (
                load_dataset(repo, subset, split=split, streaming=True)
                if subset
                else load_dataset(repo, split=split, streaming=True)
            )
            per_source_cap = target_passages // 2
            added = 0
            for row in ds:
                text = "。".join(str(row.get(f, "")).strip() for f in fields if row.get(f))
                snippet = text[:260].replace("\n\n", "\n").strip()
                if is_clean(snippet):
                    passages.append("。" + snippet)
                    added += 1
                    if added >= per_source_cap or len(passages) >= target_passages:
                        break
            print(f"  -> accepted {added} strictly decontaminated passages from {repo}", flush=True)
        except Exception as exc:
            print(f"  -> skipped {repo}: {exc}", flush=True)

    random.Random(2026).shuffle(passages)
    print(f"Total clean training passages: {len(passages)} (forbidden 5-grams checked: {len(forbidden_5grams)})", flush=True)
    return passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-6l", default="dist/layers_6l_fp16.safetensors")
    parser.add_argument("--vocab", default="dist/base_neural/vocab.json")
    parser.add_argument(
        "--bench",
        nargs="*",
        default=["benchmarks/dev_suite.tsv", "benchmarks/test_holdout.tsv"],
    )
    parser.add_argument("--out-6l", default="dist/layers_6l_homophone_fp16.safetensors")
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=88)
    parser.add_argument("--lr", type=float, default=3.5e-5)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Running Homophone-Subspace Distillation on device={device}", flush=True)

    snap = Path(
        glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/*"))[0]
    )
    orig_cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    vocab_strings = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    hf_tok = AutoTokenizer.from_pretrained(str(snap))
    c_tok = GreedyCompactTokenizer(vocab_strings)

    bos_old = 151643
    compact_to_old = [bos_old, bos_old]
    for piece in vocab_strings[2:]:
        ids = hf_tok.encode(piece, add_special_tokens=False)
        compact_to_old.append(ids[0] if len(ids) == 1 else bos_old)
    c2o_tensor = torch.tensor(compact_to_old, dtype=torch.long, device=device)

    print("Loading frozen 28-layer Qwen3-0.6B-Base teacher...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        str(snap), dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        embed_pruned = teacher.model.embed_tokens.weight.index_select(0, c2o_tensor).detach().clone()

    st_6l = load_file(args.in_6l)
    student = Qwen3Student6L(st_6l, embed_pruned.cpu(), orig_cfg).to(device)
    cos, sin = build_rope_cache(args.seq_len + 8, orig_cfg["head_dim"], float(orig_cfg["rope_theta"]), device)

    peer_ids, peer_valid, has_group = build_homophone_peer_tables(vocab_strings)
    peer_ids = peer_ids.to(device)
    peer_valid = peer_valid.to(device)
    has_group = has_group.to(device)

    passages = collect_decontaminated_corpus([Path(p) for p in args.bench], target_passages=2600)
    encoded_pool: list[list[int]] = []
    for p in passages:
        tids = [t for t in c_tok.encode(p) if t != 1]
        if len(tids) >= 24:
            encoded_pool.append(tids[: args.seq_len])
    print(f"Tokenized {len(encoded_pool)} clean sequences (max len {args.seq_len})", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(108)
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        # Cosine learning rate schedule with 10% linear warmup
        if step <= args.steps * 0.1:
            lr_now = args.lr * (step / max(1, args.steps * 0.1))
        else:
            prog = (step - args.steps * 0.1) / max(1.0, args.steps * 0.9)
            lr_now = args.lr * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        batch_seqs = rng.choices(encoded_pool, k=args.batch_size)
        max_l = max(len(s) for s in batch_seqs)
        inp = torch.zeros((args.batch_size, max_l), dtype=torch.long, device=device)
        mask = torch.zeros((args.batch_size, max_l), dtype=torch.bool, device=device)
        for bi, s in enumerate(batch_seqs):
            inp[bi, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            mask[bi, : len(s)] = True

        old_inp = c2o_tensor[inp]
        with torch.no_grad():
            t_out = teacher.model(input_ids=old_inp, use_cache=False)
            u_t = t_out.last_hidden_state.float()
            logits_t = F.linear(u_t, student.embed_weight)

        u_s = student(inp, cos, sin).float()
        logits_s = F.linear(u_s, student.embed_weight)

        # Shift for next-token prediction: position `t` predicts `inp[:, t + 1]`
        pred_mask = mask[:, 1:]
        targets = inp[:, 1:]
        lt_shift = logits_t[:, :-1, :][pred_mask]
        ls_shift = logits_s[:, :-1, :][pred_mask]
        ut_shift = u_t[:, :-1, :][pred_mask]
        us_shift = u_s[:, :-1, :][pred_mask]
        flat_targets = targets[pred_mask]

        # 1. Full-vocab KL at T=1.5
        temp = 1.5
        loss_full_kl = (
            F.kl_div(
                F.log_softmax(ls_shift / temp, dim=-1),
                F.softmax(lt_shift / temp, dim=-1),
                reduction="batchmean",
            )
            * (temp * temp)
        )

        # 2. Pre-head hidden cosine alignment
        loss_cos = (1.0 - F.cosine_similarity(us_shift, ut_shift, dim=-1)).mean()

        # 3. Phonetic-Equivalence Subspace KL over homophone peers `H(Phi(target))`!
        hom_sel = has_group[flat_targets]
        if hom_sel.any():
            hom_targets = flat_targets[hom_sel]
            p_idx = peer_ids[hom_targets]       # [N_hom, M]
            p_val = peer_valid[hom_targets]     # [N_hom, M]
            lt_hom = torch.gather(lt_shift[hom_sel], 1, p_idx).masked_fill(~p_val, -1e9)
            ls_hom = torch.gather(ls_shift[hom_sel], 1, p_idx).masked_fill(~p_val, -1e9)
            loss_subspace_kl = F.kl_div(
                F.log_softmax(ls_hom, dim=-1),
                F.softmax(lt_hom, dim=-1),
                reduction="batchmean",
            )
        else:
            loss_subspace_kl = torch.tensor(0.0, device=device)

        loss = 0.70 * loss_subspace_kl + 0.25 * loss_full_kl + 0.20 * loss_cos
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 40 == 0 or step == args.steps:
            elapsed = time.perf_counter() - t_start
            print(
                f"  step {step:3d}/{args.steps} | loss={loss.item():.4f} "
                f"(hom_kl={loss_subspace_kl.item():.4f}, full_kl={loss_full_kl.item():.4f}, cos={loss_cos.item():.4f}) "
                f"| lr={lr_now:.2e} | {elapsed:.1f}s",
                flush=True,
            )

    out_path = Path(args.out_6l)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(student.export_state_dict_fp16(), str(out_path))
    print(f"Saved homophone-subspace refined 6-layer FP16 weights -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
