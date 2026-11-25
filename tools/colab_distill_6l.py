#!/usr/bin/env python3
"""
Google Colab GPU Layer-to-Layer + Pre-Head Cosine + Dual-Temperature KL Distillation:
Distills 28-layer `Qwen/Qwen3-0.6B-Base` into a 6-layer `Qwen3-6L-Chime` student (`24,576` vocab,
`head_dim=128`, QK-Norm, tied frozen `embed_tokens`) ready for 64 MB 4-bit MLX deployment!
"""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

CURATED_IME_SENTENCES = [
    "依法保障合法权利，严禁任何机构滥用公共权力。",
    "群众向上级如实地反映问题，身体却出现了轻微的过敏反应。",
    "我们在苹果 macOS 系统上使用 Rust 和 Swift 编写超低延迟的本地输入法引擎。",
    "帮我看看这个 GitHub 上的 PR 代码有没有内存泄漏问题，顺便测试一下 API 接口。",
    "通过端侧小模型的量化推理技术，大幅降低了响应延迟与内存占用。",
    "我在使用的过程中，感觉第一候选字有时候不那么靠谱，需要结合上下文窗口来优化。",
    "我截了几个图，你看看是怎么回事，如何让初始的状态能够足够好。",
    "打字的过程中输入法要维护一个上下文窗口，这样才能推荐出来最合适的输入。",
    "这个候选还是差些意思，八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
    "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。",
    "天有不测风云，人有旦夕祸福。天生我材必有用，千金散尽还复来。",
    "有个问题：为什么是用旧模型呢？为什么不用更新版本的开源大模型来做词库蒸馏呢？",
    "模糊音处理和候选字还有不少细节问题，一些冷门的词和跟上下文无关的词不该排在前面。",
    "比如第四和第五个候选究竟有多大可能被选上？感觉现在的排序算法还有很大的提升空间。",
    "中英文混合输入的时候自动在英文单词前后添加空格，并把大小写自动纠正成 ChatGPT。",
    "今天我们使用本模型进行端到端的基准评测，验证四比特量化后的推理速度与准确度。",
    "报名时间截至明天下午五点，逾期系统将自动截止所有申请。",
    "政府深入研究并制定了长期战略方针，编辑部则仔细制订了具体的出版细则。",
    "空气和水是人类生存所必需的条件，因此我们每个人都必须高度重视环境保护。",
    "面对瞬息万变幻莫测的国际形势，艺术作品也展现出了丰富多样的表现形式。",
    "公安机关依法立案侦查这起案件，侦察兵则连夜潜入前线侦察敌情。",
    "这位医生很有高尚的医德，每天都耐心地为每一位患者说明治疗方案。",
    "数学家推导出了全新的公式，前线部队则发起了猛烈的攻势，我明天还要出差办公事。",
    "根据用户的日常打字习惯不断学习调整，根本不需要手动维护复杂的个人词库。",
    "似乎每次切换到新的输入源时，系统都需要同步刷新状态栏图标和上下文缓存。",
]

DISAMBIGUATION_PROBES = [
    ("4/5究竟有", ["多大", "多达", "多打", "躲大"]),
    ("打字的过程中要维护一个上下文窗口，这样才能推荐出来最", ["合适", "何时", "何事", "和是"]),
    ("感觉第一候选字有时候不那么", ["靠谱", "靠铺", "考铺"]),
    ("依法保障", ["权利", "权力", "全力"]),
    ("严禁滥用", ["权力", "权利", "全力"]),
    ("候选字还是有", ["不少", "不稍", "步哨"]),
    ("现在的排序算法还有很大的提升", ["空间", "空降", "控件"]),
    ("为什么不用更新", ["版本", "板本", "半本"]),
    ("今天我们", ["使用本模型", "试用本模型"]),
]


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


class Qwen3Student6L(nn.Module):
    def __init__(
        self,
        raw_st: dict[str, torch.Tensor],
        embed_pruned: torch.Tensor,
        selected_layers: list[int],
        cfg: dict,
    ) -> None:
        super().__init__()
        self.register_buffer("embed_weight", embed_pruned.bfloat16())
        self.layers = nn.ModuleList([Qwen3Block(raw_st, li, cfg) for li in selected_layers])
        self.norm = RMSNorm(raw_st["model.norm.weight"], float(cfg.get("rms_norm_eps", 1e-6)))

    def forward(self, new_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = F.embedding(new_ids, self.embed_weight)
        inter_states = []
        for idx, blk in enumerate(self.layers):
            x = blk(x, cos, sin)
            if idx < len(self.layers) - 1:
                inter_states.append(x)
        u_s = self.norm(x)
        return u_s, inter_states


class GreedyCompactTokenizer:
    """Exact mirror of `CompactVocab::encode` in `crates/neural/src/lib.rs` (0 train/test skew)."""

    def __init__(self, vocab_strings: list[str]) -> None:
        self.p2i = {p: i for i, p in enumerate(vocab_strings)}
        self.unk = 1

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        n = len(text)
        i = 0
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            max_span = min(6, n - i)
            matched_id = -1
            matched_span = 1
            for span in range(max_span, 0, -1):
                sub = text[i : i + span]
                tid = self.p2i.get(sub)
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


def collect_chinese_corpus(target_chars: int = 16_000_000) -> list[str]:
    texts: list[str] = []
    # Oversample high-value IME conversational, homophone, code-switch & poetry sentences
    for _ in range(80):
        texts.extend(CURATED_IME_SENTENCES)

    total_chars = sum(len(t) for t in texts)
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
            taken = 0
            source_cap = target_chars // 2
            source_chars = 0
            for row in ds:
                parts = [str(row.get(f, "")).strip() for f in fields if row.get(f)]
                merged = " ".join(p for p in parts if p)
                if len(merged) < 20:
                    continue
                # Keep paragraphs with natural Chinese density
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
            print(f"  -> collected {taken} passages ({source_chars / 1e6:.2f}M chars) from {repo}", flush=True)
        except Exception as exc:
            print(f"  -> skipped {repo}: {exc}", flush=True)

    random.Random(42).shuffle(texts)
    print(f"Total distillation corpus: {len(texts)} passages, {total_chars / 1e6:.2f}M chars", flush=True)
    return texts


@torch.no_grad()
def evaluate_probes(student: Qwen3Student6L, c_tok: GreedyCompactTokenizer, cos: torch.Tensor, sin: torch.Tensor, device: torch.device):
    student.eval()
    passed = 0
    details = []
    for ctx_s, cands in DISAMBIGUATION_PROBES:
        c_ids = c_tok.encode(ctx_s)
        scores = []
        for cand in cands:
            t_ids = c_tok.encode(cand)
            seq = torch.tensor([c_ids + t_ids], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                u_s, _ = student(seq, cos, sin)
                logits = F.linear(u_s, student.embed_weight).float()
                lps = F.log_softmax(logits[0], dim=-1)
            lp = sum(float(lps[len(c_ids) - 1 + i, tid].item()) for i, tid in enumerate(t_ids))
            scores.append((cand, round(lp, 2)))
        best = max(scores, key=lambda x: x[1])[0]
        if best == cands[0]:
            passed += 1
        details.append((ctx_s, scores))
    student.train()
    return passed, len(DISAMBIGUATION_PROBES), details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="/content/vocab.json")
    parser.add_argument("--out-dir", default="/content/student_6l")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=4.0e-4)
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vocab_strings: list[str] = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    c_tok = GreedyCompactTokenizer(vocab_strings)
    print(f"Loaded compact vocab: {len(vocab_strings)} tokens", flush=True)

    snap = Path(
        snapshot_download(
            "Qwen/Qwen3-0.6B-Base",
            allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"],
        )
    )
    orig_cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    hf_tok = AutoTokenizer.from_pretrained(str(snap))

    bos_id = 151643
    selected_old_ids = [bos_id, bos_id]
    for piece in vocab_strings[2:]:
        ids = hf_tok.encode(piece, add_special_tokens=False)
        selected_old_ids.append(ids[0] if len(ids) == 1 else bos_id)
    new_to_old_gpu = torch.tensor(selected_old_ids, dtype=torch.long, device=device)

    print("Loading 28-layer Qwen3-0.6B-Base teacher...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(str(snap), dtype=torch.bfloat16).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    raw_st = load_file(str(snap / "model.safetensors"))
    idx_cpu = torch.tensor(selected_old_ids, dtype=torch.long)
    embed_pruned = raw_st["model.embed_tokens.weight"].index_select(0, idx_cpu).float().contiguous()

    selected_layers = [0, 5, 11, 16, 22, 27]
    teacher_inter_targets = [5, 10, 16, 21, 25]  # hidden_states indices (1-based after embedding)
    student = Qwen3Student6L(raw_st, embed_pruned, selected_layers, orig_cfg).to(device)
    del raw_st

    cos_cache, sin_cache = build_rope_cache(
        max_seq_len=512,
        head_dim=orig_cfg["head_dim"],
        theta=float(orig_cfg.get("rope_theta", 1000000.0)),
        device=device,
    )

    p0, n0, det0 = evaluate_probes(student, c_tok, cos_cache, sin_cache, device)
    print(f"[Step 0 Before Distillation] Probe accuracy: {p0}/{n0}", flush=True)
    for ctx_s, sc in det0[:4]:
        print(f"   {ctx_s!r} -> {sc}", flush=True)

    # Build packed token stream (`[N_seqs, seq_len]`)
    passages = collect_chinese_corpus(target_chars=16_000_000)
    flat_tokens: list[int] = []
    sep_id = c_tok.p2i.get("。", 0)
    for text in passages:
        ids = c_tok.encode(text)
        ids = [i for i in ids if i != 1]
        if len(ids) >= 8:
            flat_tokens.extend(ids)
            flat_tokens.append(sep_id)

    n_seqs = len(flat_tokens) // args.seq_len
    packed = torch.tensor(flat_tokens[: n_seqs * args.seq_len], dtype=torch.long).view(n_seqs, args.seq_len)
    perm = torch.randperm(n_seqs, generator=torch.Generator().manual_seed(42))
    packed = packed[perm]

    val_size = min(256, max(32, n_seqs // 20))
    val_seqs = packed[:val_size]
    train_seqs = packed[val_size:]
    print(f"Tokenized {len(flat_tokens) / 1e6:.2f}M compact tokens -> {len(train_seqs)} train blocks, {len(val_seqs)} val blocks", flush=True)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    print(f"Trainable student parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M (frozen tied embed: {student.embed_weight.numel() / 1e6:.2f}M)", flush=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    warmup_steps = min(150, args.steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.04 + 0.96 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    t_start = time.perf_counter()
    train_cursor = 0
    student.train()

    for step in range(1, args.steps + 1):
        if train_cursor + args.batch_size > len(train_seqs):
            shuf = torch.randperm(len(train_seqs))
            train_seqs = train_seqs[shuf]
            train_cursor = 0

        batch_new = train_seqs[train_cursor : train_cursor + args.batch_size].to(device, non_blocking=True)
        train_cursor += args.batch_size
        batch_old = new_to_old_gpu[batch_new]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            t_out = teacher.model(input_ids=batch_old, output_hidden_states=True, use_cache=False)
            u_t = t_out.last_hidden_state  # Already passed through `teacher.model.norm`!
            t_inter = [t_out.hidden_states[idx] for idx in teacher_inter_targets]
            z_t = F.linear(u_t, student.embed_weight)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            u_s, s_inter = student(batch_new, cos_cache, sin_cache)
            z_s = F.linear(u_s, student.embed_weight)

            # 1. Exact vocabulary KL divergence (T=1.5) + Pre-head feature alignment
            bs_tokens = batch_new.numel()
            inv_t = 1.0 / 1.5
            kl_loss = (
                F.kl_div(
                    F.log_softmax(z_s.view(bs_tokens, -1).float() * inv_t, dim=-1),
                    F.softmax(z_t.view(bs_tokens, -1).float() * inv_t, dim=-1),
                    reduction="batchmean",
                )
                * (1.5 * 1.5)
            )

            # 2. Pre-LM-head normalized hidden-state cosine + SmoothL1 alignment
            # Because `embed_weight` is shared & frozen, `u_s -> u_t` guarantees exact T=1.0 logit scale!
            feat_cos = 1.0 - F.cosine_similarity(u_s.float(), u_t.float(), dim=-1).mean()
            feat_l1 = F.smooth_l1_loss(u_s.float(), u_t.float())

            # 3. Decaying intermediate trajectory cosine alignment
            prog = step / args.steps
            inter_w = 0.30 * (1.0 - 0.8 * prog)
            inter_cos = sum(
                1.0 - F.cosine_similarity(si.float(), ti.float(), dim=-1).mean()
                for si, ti in zip(s_inter, t_inter)
            ) / len(s_inter)

            loss = 0.70 * kl_loss + 2.80 * feat_cos + 0.60 * feat_l1 + inter_w * inter_cos

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            elapsed = time.perf_counter() - t_start
            tok_per_s = (step * args.batch_size * args.seq_len) / max(1e-3, elapsed)
            if step == 1 or step % 200 == 0 or step == args.steps:
                p_ok, p_tot, p_det = evaluate_probes(student, c_tok, cos_cache, sin_cache, device)
                probe_str = f"{p_ok}/{p_tot}"
            else:
                p_det = []
                probe_str = "..."
            print(
                f"[Step {step:4d}/{args.steps}] loss={loss.item():.4f} "
                f"(kl={kl_loss.item():.3f}, cos={1.0 - feat_cos.item():.4f}) | "
                f"probes={probe_str} | lr={scheduler.get_last_lr()[0]:.2e} | {tok_per_s / 1e3:.1f}k tok/s ({elapsed:.1f}s)",
                flush=True,
            )
            if step == args.steps or step % 400 == 0:
                for ctx_s, sc in p_det:
                    print(f"   {ctx_s!r} -> {sc}", flush=True)

    # Validation KL & cosine summary
    student.eval()
    val_kl_sum = 0.0
    val_cos_sum = 0.0
    val_batches = 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(val_seqs), args.batch_size):
            vb_new = val_seqs[i : i + args.batch_size].to(device)
            vb_old = new_to_old_gpu[vb_new]
            u_t = teacher.model(input_ids=vb_old, use_cache=False).last_hidden_state
            u_s, _ = student(vb_new, cos_cache, sin_cache)
            z_t = F.linear(u_t, student.embed_weight).view(-1, len(vocab_strings)).float()
            z_s = F.linear(u_s, student.embed_weight).view(-1, len(vocab_strings)).float()
            val_kl_sum += float(F.kl_div(F.log_softmax(z_s, dim=-1), F.softmax(z_t, dim=-1), reduction="batchmean").item())
            val_cos_sum += float(F.cosine_similarity(u_s.float(), u_t.float(), dim=-1).mean().item())
            val_batches += 1
    print(
        f"\nFinal Held-Out Validation: KL(Teacher||Student) = {val_kl_sum / max(1, val_batches):.4f}, "
        f"Pre-Head Cosine Similarity = {val_cos_sum / max(1, val_batches):.4f}",
        flush=True,
    )

    # Export trained 6-layer weights (`layers.0..5` + `model.norm.weight`) in FP16 (~180 MB)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_tensors: dict[str, torch.Tensor] = {
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

    out_st = out_dir / "layers_6l_fp16.safetensors"
    save_file(export_tensors, str(out_st))
    print(f"Saved {out_st} ({out_st.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
