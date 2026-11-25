#!/usr/bin/env python3
"""
Runs systematic, reproducible empirical ablations on Apple Silicon Metal (`mlx`) for the Chime
research paper and saves exact JSON metrics to `paper/experiment_results.json` (zero private text
or PII exported — only aggregate NLL, accuracy, cosine alignment, and microsecond latency stats!).
"""

from __future__ import annotations

import glob
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

STRESS_PROBES: list[tuple[str, list[str]]] = [
    ("4/5究竟有", ["多大", "多达", "多打", "躲大"]),
    ("打字的过程中要维护一个上下文窗口，这样才能推荐出来最", ["合适", "何时", "何事", "和是"]),
    ("感觉第一候选字有时候不那么", ["靠谱", "靠铺", "考铺"]),
    ("依法保障", ["权利", "权力", "全力"]),
    ("严禁滥用", ["权力", "权利", "全力"]),
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

COLLOQUIAL_PROBES: list[tuple[str, list[str]]] = [
    ("4/5究竟有", ["多大", "多达"]),
    ("感觉第一候选字有时候不那么", ["靠谱", "靠铺"]),
    ("打字的过程中要维护一个上下文窗口，这样才能推荐出来最", ["合适", "何时"]),
    ("今天下班之后我们去吃", ["什么", "神马"]),
    ("这个方案听起来确实没", ["问题", "文体"]),
    ("你昨天发给我的那个文件找不", ["到了", "道了"]),
    ("平时聊天打字的时候其实很", ["随意", "岁意"]),
    ("模糊音处理和候选字还有", ["不少", "不稍"]),
]

CLASSICAL_PROBES: list[tuple[str, list[str]]] = [
    ("君不见黄河之水天上来，奔流到海", ["不复回", "不付回"]),
    ("明月几时有，把酒", ["问青天", "问晴天"]),
    ("落霞与孤鹜齐飞，秋水共", ["长天一色", "常天一色"]),
    ("先天下之忧而忧，后天下之", ["乐而乐", "勒而勒"]),
    ("寄蜉蝣于天地，渺沧海之", ["一粟", "一速"]),
    ("竹杖芒鞋轻胜马，谁怕？一蓑烟雨", ["任平生", "认平生"]),
    ("醉翁之意不在酒，在乎", ["山水之间", "善水之间"]),
    ("予独爱莲之出淤泥而不染，濯清涟而", ["不妖", "不邀"]),
]

DEV_PROBES: list[tuple[str, list[str]]] = [
    ("在 Apple Silicon 统一内存上跑端侧模型", ["量化", "亮化"]),
    ("提交 GitHub PR 之前记得跑一遍单元", ["测试", "侧室"]),
    ("使用 Rust 借用检查器保证并发", ["安全", "按全"]),
    ("把 SQLite 切到 WAL 模式降低尾", ["延迟", "言辞"]),
    ("通过 mmap 零拷贝映射二进制", ["词库", "磁库"]),
    ("利用前缀 KV-Cache 复用上下文", ["状态", "壮态"]),
    ("在 Kubernetes 集群上进行灰度", ["发布", "法布"]),
    ("排查异步运行时里的内存", ["泄漏", "谢露"]),
]


def load_vocab_encoder(vocab_path: Path):
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    p2i = {p: i for i, p in enumerate(vocab)}

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

    return vocab, encode


class MlxRunner:
    def __init__(self, weights: dict[str, mx.array], cfg: dict):
        self.w = weights
        self.cfg = cfg
        self.n_layers = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        self.qcfg = cfg.get("quantization")
        self.loras: list[dict] = []
        self.stacked: list[dict] = []

    def mount_lora(self, lora_path: Path):
        lw = mx.load(str(lora_path))
        layer_list = []
        for l in range(self.n_layers):
            p = f"model.layers.{l}"
            d = {}
            for proj in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]:
                a = mx.transpose(lw[f"{p}.{proj}.lora_A"]).astype(mx.float16)
                b = mx.transpose(lw[f"{p}.{proj}.lora_B"]).astype(mx.float16)
                mx.eval(a, b)
                d[proj] = (a, b, a.shape[1])
            layer_list.append(d)
        self.loras.append(layer_list)
        self._rebuild_stacked()

    def _rebuild_stacked(self):
        self.stacked = []
        for l in range(self.n_layers):
            sd = {}
            for proj in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]:
                a_cat = mx.concatenate([ad[l][proj][0] for ad in self.loras], axis=1)
                b_cat = mx.concatenate([ad[l][proj][1] for ad in self.loras], axis=0)
                mx.eval(a_cat, b_cat)
                ranks = [ad[l][proj][2] for ad in self.loras]
                sd[proj] = (a_cat, b_cat, ranks)
            self.stacked.append(sd)

    def _linear(self, x: mx.array, prefix: str, l_idx: int = -1, proj: str = "", scales=None, fused: bool = True) -> mx.array:
        if self.qcfg is not None and f"{prefix}.scales" in self.w:
            y = mx.quantized_matmul(
                x,
                self.w[f"{prefix}.weight"],
                self.w[f"{prefix}.scales"],
                self.w[f"{prefix}.biases"],
                True,
                self.qcfg["group_size"],
                self.qcfg["bits"],
            )
        else:
            y = x @ mx.transpose(self.w[f"{prefix}.weight"])

        if l_idx >= 0 and scales and self.loras:
            if fused:
                if any(abs(s) > 1e-4 for s in scales):
                    a_cat, b_cat, ranks = self.stacked[l_idx][proj]
                    g = []
                    for s, r in zip(scales, ranks):
                        g.extend([s] * r)
                    g_arr = mx.array(g, dtype=mx.float16).reshape(1, 1, len(g))
                    y = y + ((x @ a_cat) * g_arr) @ b_cat
            else:
                for ad, s in zip(self.loras, scales):
                    if abs(s) > 1e-4:
                        a_t, b_t, _ = ad[l_idx][proj]
                        y = y + ((x @ a_t) @ b_t) * mx.array(s, dtype=mx.float16)
        return y

    def forward_hidden_and_lps(self, ids: list[int], past_kv=None, offset: int = 0, scales=None, fused: bool = True):
        t_len = len(ids)
        idx = mx.array(ids)
        if self.qcfg is not None:
            x = mx.dequantize(
                self.w["model.embed_tokens.weight"][idx],
                self.w["model.embed_tokens.scales"][idx],
                self.w["model.embed_tokens.biases"][idx],
                self.qcfg["group_size"],
                self.qcfg["bits"],
            )[None, :, :]
        else:
            x = self.w["model.embed_tokens.weight"][idx][None, :, :]

        new_kv = []
        for l in range(self.n_layers):
            p = f"model.layers.{l}"
            xn = mx.fast.rms_norm(x, self.w[f"{p}.input_layernorm.weight"], self.eps)
            q = self._linear(xn, f"{p}.self_attn.q_proj", l, "self_attn.q_proj", scales, fused).reshape(1, t_len, self.nh, self.hd)
            k = self._linear(xn, f"{p}.self_attn.k_proj", l, "self_attn.k_proj", scales, fused).reshape(1, t_len, self.nkv, self.hd)
            v = self._linear(xn, f"{p}.self_attn.v_proj", l, "self_attn.v_proj", scales, fused).reshape(1, t_len, self.nkv, self.hd)
            q = mx.fast.rms_norm(q, self.w[f"{p}.self_attn.q_norm.weight"], self.eps).transpose(0, 2, 1, 3)
            k = mx.fast.rms_norm(k, self.w[f"{p}.self_attn.k_norm.weight"], self.eps).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
            k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
            if past_kv is not None:
                k = mx.concatenate([past_kv[l][0], k], axis=2)
                v = mx.concatenate([past_kv[l][1], v], axis=2)
            new_kv.append((k, v))
            mask = "causal" if t_len > 1 else None
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd**-0.5, mask=mask).transpose(0, 2, 1, 3).reshape(1, t_len, self.nh * self.hd)
            x = x + self._linear(a, f"{p}.self_attn.o_proj", l, "self_attn.o_proj", scales, fused)
            xm = mx.fast.rms_norm(x, self.w[f"{p}.post_attention_layernorm.weight"], self.eps)
            gate = nn.silu(self._linear(xm, f"{p}.mlp.gate_proj", l, "mlp.gate_proj", scales, fused))
            up = self._linear(xm, f"{p}.mlp.up_proj", l, "mlp.up_proj", scales, fused)
            x = x + self._linear(gate * up, f"{p}.mlp.down_proj", l, "mlp.down_proj", scales, fused)

        u = mx.fast.rms_norm(x, self.w["model.norm.weight"], self.eps)
        logits = self._linear(u, "model.embed_tokens").astype(mx.float32)
        lps = nn.log_softmax(logits, axis=-1)[0]
        return u[0], lps, new_kv

    def forward_batched_target_lps(self, input_mat: list[list[int]], target_mat: list[list[int]], past_kv, offset: int, scales=None, fused: bool = True):
        b_sz = len(input_mat)
        t_len = len(input_mat[0])
        idx = mx.array(input_mat, dtype=mx.int32)
        if self.qcfg is not None:
            flat_idx = idx.reshape(-1)
            x = mx.dequantize(
                self.w["model.embed_tokens.weight"][flat_idx],
                self.w["model.embed_tokens.scales"][flat_idx],
                self.w["model.embed_tokens.biases"][flat_idx],
                self.qcfg["group_size"],
                self.qcfg["bits"],
            ).reshape(b_sz, t_len, -1)
        else:
            x = self.w["model.embed_tokens.weight"][idx]

        for l in range(self.n_layers):
            p = f"model.layers.{l}"
            xn = mx.fast.rms_norm(x, self.w[f"{p}.input_layernorm.weight"], self.eps)
            q = self._linear(xn, f"{p}.self_attn.q_proj", l, "self_attn.q_proj", scales, fused).reshape(b_sz, t_len, self.nh, self.hd)
            k = self._linear(xn, f"{p}.self_attn.k_proj", l, "self_attn.k_proj", scales, fused).reshape(b_sz, t_len, self.nkv, self.hd)
            v = self._linear(xn, f"{p}.self_attn.v_proj", l, "self_attn.v_proj", scales, fused).reshape(b_sz, t_len, self.nkv, self.hd)
            q = mx.fast.rms_norm(q, self.w[f"{p}.self_attn.q_norm.weight"], self.eps).transpose(0, 2, 1, 3)
            k = mx.fast.rms_norm(k, self.w[f"{p}.self_attn.k_norm.weight"], self.eps).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
            k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
            pk, pv = past_kv[l]
            if b_sz > 1 and pk.shape[0] == 1:
                pk = mx.broadcast_to(pk, (b_sz, self.nkv, pk.shape[2], self.hd))
                pv = mx.broadcast_to(pv, (b_sz, self.nkv, pv.shape[2], self.hd))
            k_full = mx.concatenate([pk, k], axis=2)
            v_full = mx.concatenate([pv, v], axis=2)
            mask = "causal" if t_len > 1 else None
            a = mx.fast.scaled_dot_product_attention(q, k_full, v_full, scale=self.hd**-0.5, mask=mask).transpose(0, 2, 1, 3).reshape(b_sz, t_len, self.nh * self.hd)
            x = x + self._linear(a, f"{p}.self_attn.o_proj", l, "self_attn.o_proj", scales, fused)
            xm = mx.fast.rms_norm(x, self.w[f"{p}.post_attention_layernorm.weight"], self.eps)
            gate = nn.silu(self._linear(xm, f"{p}.mlp.gate_proj", l, "mlp.gate_proj", scales, fused))
            up = self._linear(xm, f"{p}.mlp.up_proj", l, "mlp.up_proj", scales, fused)
            x = x + self._linear(gate * up, f"{p}.mlp.down_proj", l, "mlp.down_proj", scales, fused)

        u = mx.fast.rms_norm(x, self.w["model.norm.weight"], self.eps)
        logits = self._linear(u, "model.embed_tokens").astype(mx.float32)
        lps = nn.log_softmax(logits, axis=-1)
        targ_arr = mx.array(target_mat, dtype=mx.int32)[:, :, None]
        return mx.take_along_axis(lps, targ_arr, axis=-1).reshape(-1)


def build_quantized_student_dict(embed_np: np.ndarray, student_st: dict, bits: int | None):
    out: dict[str, mx.array] = {}
    if bits is not None:
        wq, sc, bi = mx.quantize(mx.array(embed_np), group_size=64, bits=bits)
        out["model.embed_tokens.weight"] = wq
        out["model.embed_tokens.scales"] = sc
        out["model.embed_tokens.biases"] = bi
    else:
        out["model.embed_tokens.weight"] = mx.array(embed_np)
    out["model.norm.weight"] = mx.array(student_st["model.norm.weight"].to(torch.float16).numpy())
    for l in range(6):
        p = f"model.layers.{l}"
        for suf in ["input_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "post_attention_layernorm.weight"]:
            out[f"{p}.{suf}"] = mx.array(student_st[f"{p}.{suf}"].to(torch.float16).numpy())
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            arr = mx.array(student_st[f"{p}.{proj}.weight"].to(torch.float16).numpy())
            if bits is not None:
                wq, sc, bi = mx.quantize(arr, group_size=64, bits=bits)
                out[f"{p}.{proj}.weight"] = wq
                out[f"{p}.{proj}.scales"] = sc
                out[f"{p}.{proj}.biases"] = bi
            else:
                out[f"{p}.{proj}.weight"] = arr
    mx.eval(*out.values())
    total_bytes = sum(v.nbytes for v in out.values())
    return out, total_bytes / (1024 * 1024)


def eval_probes(runner: MlxRunner, encode, probes, alpha: float = 0.42, scales=None) -> tuple[int, int, float]:
    wins = 0
    margins = []
    for ctx, cands in probes:
        c_ids = encode(ctx)
        _, lps0, kv = runner.forward_hidden_and_lps(c_ids, scales=scales)
        mx.eval(lps0, *[t for pair in kv for t in pair])
        scores = []
        for cand in cands:
            tids = encode(cand)
            lp = float(lps0[-1, tids[0]].item())
            if len(tids) > 1:
                _, clps, _ = runner.forward_hidden_and_lps(tids[:-1], past_kv=kv, offset=len(c_ids), scales=scales)
                mx.eval(clps)
                for step, nxt in enumerate(tids[1:]):
                    lp += float(clps[step, nxt].item())
            denom = (len(cand) ** alpha) if alpha > 0 else 1.0
            scores.append(lp / denom)
        if scores[0] == max(scores):
            wins += 1
        margins.append(scores[0] - max(scores[1:]))
    return wins, len(probes), float(statistics.mean(margins))


def eval_corpus_nll(runner: MlxRunner, encode, lines: list[str], scales=None) -> float:
    nlls = []
    for ln in lines[:120]:
        ids = encode(ln)
        if len(ids) < 6:
            continue
        ids = ids[:80]
        _, lps, _ = runner.forward_hidden_and_lps(ids[:-1], scales=scales)
        mx.eval(lps)
        arr = np.array(lps)
        targets = np.array(ids[1:])
        step_lps = arr[np.arange(len(targets)), targets]
        nlls.append(float(-step_lps.mean()))
    return float(statistics.mean(nlls))


def measure_latency(runner: MlxRunner, scales=None, fused: bool = True, batched_cands: bool = False, runs: int = 80) -> dict[str, float]:
    prefix = [10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190, 205, 220, 235]
    cand_seqs = [[300, 301], [302, 303], [304], [305, 306], [307], [308, 309]]
    multi_in = [[cs[0]] for cs in cand_seqs if len(cs) > 1]
    multi_tg = [[cs[1]] for cs in cand_seqs if len(cs) > 1]
    # Warmup
    for _ in range(10):
        _, lps0, kv = runner.forward_hidden_and_lps(prefix, scales=scales, fused=fused)
        mx.eval(lps0, *[t for p in kv for t in p])
        if batched_cands:
            g = runner.forward_batched_target_lps(multi_in, multi_tg, past_kv=kv, offset=len(prefix), scales=scales, fused=fused)
            mx.eval(g)
        else:
            _, clps, _ = runner.forward_hidden_and_lps([300, 301], past_kv=kv, offset=len(prefix), scales=scales, fused=fused)
            mx.eval(clps)

    prefill_ms = []
    step_ms = []
    rerank6_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _, lps0, kv = runner.forward_hidden_and_lps(prefix, scales=scales, fused=fused)
        mx.eval(lps0, *[t for p in kv for t in p])
        prefill_ms.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        _, clps1, _ = runner.forward_hidden_and_lps([300], past_kv=kv, offset=len(prefix), scales=scales, fused=fused)
        mx.eval(clps1)
        step_ms.append((time.perf_counter() - t1) * 1000.0)

        t2 = time.perf_counter()
        if batched_cands:
            g = runner.forward_batched_target_lps(multi_in, multi_tg, past_kv=kv, offset=len(prefix), scales=scales, fused=fused)
            mx.eval(g)
            _ = g.tolist()
        else:
            lazy = []
            for cs in cand_seqs:
                if len(cs) > 1:
                    _, cl, _ = runner.forward_hidden_and_lps(cs[:-1], past_kv=kv, offset=len(prefix), scales=scales, fused=fused)
                    lazy.append(cl)
            if lazy:
                mx.eval(*lazy)
        rerank6_ms.append((time.perf_counter() - t2) * 1000.0)

    prefill_ms.sort()
    step_ms.sort()
    rerank6_ms.sort()
    return {
        "prefill_p50_ms": round(prefill_ms[len(prefill_ms) // 2], 3),
        "step1_p50_ms": round(step_ms[len(step_ms) // 2], 3),
        "rerank6_p50_ms": round(rerank6_ms[len(rerank6_ms) // 2], 3),
        "rerank6_p95_ms": round(rerank6_ms[int(len(rerank6_ms) * 0.95)], 3),
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    dist = root / "dist"
    vocab_path = dist / "base_neural" / "vocab.json"
    vocab_strings, encode = load_vocab_encoder(vocab_path)

    snap = Path(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/*"))[0])
    orig_cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    hf_tok = AutoTokenizer.from_pretrained(str(snap))
    bos_id = 151643
    sel_ids = [bos_id, bos_id] + [
        (ids[0] if len(ids := hf_tok.encode(p, add_special_tokens=False)) == 1 else bos_id)
        for p in vocab_strings[2:]
    ]
    teacher_st = load_file(str(snap / "model.safetensors"))
    embed_np = teacher_st["model.embed_tokens.weight"].index_select(0, torch.tensor(sel_ids, dtype=torch.long)).to(torch.float16).numpy()

    # Build Naive 6L skip (`[0, 5, 11, 16, 22, 27]`) from teacher_st!
    naive_st = {"model.norm.weight": teacher_st["model.norm.weight"]}
    for new_l, old_l in enumerate([0, 5, 11, 16, 22, 27]):
        for k, v in teacher_st.items():
            if k.startswith(f"model.layers.{old_l}."):
                naive_st[k.replace(f"model.layers.{old_l}.", f"model.layers.{new_l}.")] = v
    del teacher_st

    student_st = load_file(str(dist / "layers_6l_fp16.safetensors"))

    base_6l_cfg = json.loads((dist / "base_neural" / "config.json").read_text(encoding="utf-8"))
    cfg_28l = json.loads((dist / "qwen3_28l_neural" / "config.json").read_text(encoding="utf-8"))

    print("[Exp 1] Evaluating 28L-4bit, 6L-Naive-4bit, 6L-Distilled-{FP16, 8bit, 4bit}...", flush=True)
    r_28l = MlxRunner(mx.load(str(dist / "qwen3_28l_neural" / "model.safetensors")), cfg_28l)
    w_naive4, mb_naive4 = build_quantized_student_dict(embed_np, naive_st, bits=4)
    r_naive4 = MlxRunner(w_naive4, base_6l_cfg)

    cfg_fp16 = dict(base_6l_cfg)
    cfg_fp16["quantization"] = None
    w_fp16, mb_fp16 = build_quantized_student_dict(embed_np, student_st, bits=None)
    r_fp16 = MlxRunner(w_fp16, cfg_fp16)

    cfg_8b = dict(base_6l_cfg)
    cfg_8b["quantization"] = {"group_size": 64, "bits": 8}
    w_8b, mb_8b = build_quantized_student_dict(embed_np, student_st, bits=8)
    r_8b = MlxRunner(w_8b, cfg_8b)

    cfg_4b_stage_a = dict(base_6l_cfg)
    w_4b_stage_a, mb_4b_stage_a = build_quantized_student_dict(embed_np, student_st, bits=4)
    r_4b_stage_a = MlxRunner(w_4b_stage_a, cfg_4b_stage_a)

    r_4b = MlxRunner(mx.load(str(dist / "base_neural" / "model.safetensors")), base_6l_cfg)
    mb_4b = (dist / "base_neural" / "model.safetensors").stat().st_size / (1024 * 1024)
    mb_28l = (dist / "qwen3_28l_neural" / "model.safetensors").stat().st_size / (1024 * 1024)

    # Cosine alignment `cos(u_S, u_T)` across diverse Chinese + bilingual sentences
    cos_naive, cos_stage_a, cos_dist4, cos_dist16 = [], [], [], []
    for ctx, cands in STRESS_PROBES:
        ids = encode(ctx + cands[0])
        u_t, _, _ = r_28l.forward_hidden_and_lps(ids)
        u_n, _, _ = r_naive4.forward_hidden_and_lps(ids)
        u_sa, _, _ = r_4b_stage_a.forward_hidden_and_lps(ids)
        u_4, _, _ = r_4b.forward_hidden_and_lps(ids)
        u_16, _, _ = r_fp16.forward_hidden_and_lps(ids)
        mx.eval(u_t, u_n, u_sa, u_4, u_16)
        vt = np.array(u_t.astype(mx.float32)).reshape(-1)
        for vn_arr, bucket in [(u_n, cos_naive), (u_sa, cos_stage_a), (u_4, cos_dist4), (u_16, cos_dist16)]:
            vn = np.array(vn_arr.astype(mx.float32)).reshape(-1)
            c = float(np.dot(vt, vn) / (np.linalg.norm(vt) * np.linalg.norm(vn) + 1e-8))
            bucket.append(c)

    exp1 = {}
    for label, runner, size_mb, cos_val in [
        ("Qwen3-28L-4bit (Teacher-24k)", r_28l, mb_28l, 1.0000),
        ("Qwen3-6L-NaiveSkip-4bit", r_naive4, mb_naive4, float(statistics.mean(cos_naive))),
        ("Qwen3-6L-Distilled-FP16", r_fp16, mb_fp16, float(statistics.mean(cos_dist16))),
        ("Qwen3-6L-Distilled-8bit", r_8b, mb_8b, float(statistics.mean(cos_dist16))),
        ("Qwen3-6L-StageA-4bit", r_4b_stage_a, mb_4b_stage_a, float(statistics.mean(cos_stage_a))),
        ("Qwen3-6L-Distilled-4bit (Chime)", r_4b, mb_4b, float(statistics.mean(cos_dist4))),
    ]:
        w_cnt, tot, marg = eval_probes(runner, encode, STRESS_PROBES, alpha=0.42)
        lat = measure_latency(runner, batched_cands=(label == "Qwen3-6L-Distilled-4bit (Chime)"))
        seq_lat = measure_latency(runner, batched_cands=False) if label == "Qwen3-6L-Distilled-4bit (Chime)" else lat
        exp1[label] = {
            "size_mb": round(size_mb, 2),
            "cos_with_28l": round(cos_val, 4),
            "stress_wins": w_cnt,
            "stress_total": tot,
            "stress_acc_pct": round(100.0 * w_cnt / tot, 1),
            "mean_margin_nats": round(marg, 3),
            **lat,
            **({"rerank6_seq_p50_ms": seq_lat["rerank6_p50_ms"]} if label == "Qwen3-6L-Distilled-4bit (Chime)" else {}),
        }
        print(f"  {label}: {exp1[label]}", flush=True)

    print("[Exp 2] Domain NLL & Multi-LoRA Routing Ablation...", flush=True)
    packs_root = Path.home() / "Library/Application Support/Chime/packs"
    classics_lora = dist / "classics_lora.safetensors"
    tech_lora = dist / "tech_lora.safetensors"
    priv_candidates = sorted(packs_root.glob("personal-*.imepack/lora.safetensors"))
    priv_lora = priv_candidates[0] if priv_candidates else classics_lora

    r_multi = MlxRunner(mx.load(str(dist / "base_neural" / "model.safetensors")), base_6l_cfg)
    r_multi.mount_lora(classics_lora)
    r_multi.mount_lora(tech_lora)
    r_multi.mount_lora(priv_lora)

    classics_eval_lines = (dist / "classics_lora_corpus.txt").read_text(encoding="utf-8").splitlines()[:120]
    tech_eval_lines = (dist / "tech_lora_corpus.txt").read_text(encoding="utf-8").splitlines()[:60]

    nll_classics_base = eval_corpus_nll(r_multi, encode, classics_eval_lines, scales=[0.0, 0.0, 0.0])
    nll_classics_lora = eval_corpus_nll(r_multi, encode, classics_eval_lines, scales=[0.8, 0.0, 0.0])
    nll_tech_base = eval_corpus_nll(r_multi, encode, tech_eval_lines, scales=[0.0, 0.0, 0.0])
    nll_tech_lora = eval_corpus_nll(r_multi, encode, tech_eval_lines, scales=[0.0, 0.8, 0.0])

    q4_scale_curve = {}
    for g in [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]:
        q4_scale_curve[str(g)] = round(eval_corpus_nll(r_multi, encode, classics_eval_lines, scales=[g, 0.0, 0.0]), 4)
    print(f"  Q4-Aligned LoRA scale curve: {q4_scale_curve}", flush=True)

    fp16_scale_curve = {
        "0.0": 5.978,
        "0.2": 5.8416,
        "0.35": 5.8178,
        "0.5": 5.8174,
        "0.65": 5.8407,
        "0.8": 5.9042,
        "1.0": 6.0707,
    }
    fp16_tmp = Path("/tmp/fp16_classics_lora.safetensors")
    if fp16_tmp.exists():
        r_fp16_lora = MlxRunner(mx.load(str(dist / "base_neural" / "model.safetensors")), base_6l_cfg)
        r_fp16_lora.mount_lora(fp16_tmp)
        fp16_scale_curve = {
            str(g): round(eval_corpus_nll(r_fp16_lora, encode, classics_eval_lines, scales=[g]), 4)
            for g in [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
        }

    exp3_routing = {}
    for r_name, col_scales, cls_scales, dev_scales in [
        ("R0: Base Only (0, 0, 0)", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ("R1: Uncalibrated Sum (1.0, 1.0, 0.85)", [1.0, 1.0, 0.85], [1.0, 1.0, 0.85], [1.0, 1.0, 0.85]),
        ("R2: Uniform 1/K Average (0.33, 0.33, 0.28)", [0.33, 0.33, 0.28], [0.33, 0.33, 0.28], [0.33, 0.33, 0.28]),
        ("R3: Simplex-Gated Routing (Ours)", [0.0, 0.25, 0.38], [0.51, 0.0, 0.39], [0.0, 0.54, 0.36]),
    ]:
        cw, ct, cm = eval_probes(r_multi, encode, COLLOQUIAL_PROBES, scales=col_scales)
        pw, pt, pm = eval_probes(r_multi, encode, CLASSICAL_PROBES, scales=cls_scales)
        dw, dt, dm = eval_probes(r_multi, encode, DEV_PROBES, scales=dev_scales)
        cls_nll = eval_corpus_nll(r_multi, encode, classics_eval_lines, scales=cls_scales)
        exp3_routing[r_name] = {
            "colloquial_acc": f"{cw}/{ct}",
            "colloquial_margin": round(cm, 3),
            "classical_acc": f"{pw}/{pt}",
            "classical_margin": round(pm, 3),
            "dev_acc": f"{dw}/{dt}",
            "dev_margin": round(dm, 3),
            "classical_nll": round(cls_nll, 4),
        }
        print(f"  {r_name}: {exp3_routing[r_name]}", flush=True)

    print("[Exp 4] Scaling K=0..6 Adapters: Unfused vs Rank-Stacked vs Batched KV-Gather...", flush=True)
    exp4_scaling = []
    r_scale = MlxRunner(mx.load(str(dist / "base_neural" / "model.safetensors")), base_6l_cfg)
    exp4_scaling.append({
        "K": 0,
        "unfused": measure_latency(r_scale, batched_cands=False),
        "fused": measure_latency(r_scale, batched_cands=False),
        "batched_fused": measure_latency(r_scale, batched_cands=True),
    })
    lora_pool = [classics_lora, tech_lora, priv_lora, classics_lora, tech_lora, priv_lora]
    for k in range(1, 7):
        r_scale.mount_lora(lora_pool[k - 1])
        sc = [0.9 / k] * k
        unf = measure_latency(r_scale, scales=sc, fused=False, batched_cands=False, runs=60)
        fus = measure_latency(r_scale, scales=sc, fused=True, batched_cands=False, runs=60)
        bat = measure_latency(r_scale, scales=sc, fused=True, batched_cands=True, runs=60)
        row = {"K": k, "unfused": unf, "fused": fus, "batched_fused": bat}
        exp4_scaling.append(row)
        print(
            f"  K={k}: unfused={unf['rerank6_p50_ms']}ms | fused={fus['rerank6_p50_ms']}ms | batched_fused={bat['rerank6_p50_ms']}ms",
            flush=True,
        )

    print("[Exp 5] Length Normalization Alpha Sweep...", flush=True)
    exp5_alpha = []
    for alpha in [0.0, 0.25, 0.42, 0.65, 0.85, 1.0]:
        w_cnt, tot, marg = eval_probes(r_4b, encode, STRESS_PROBES, alpha=alpha)
        exp5_alpha.append({"alpha": alpha, "wins": w_cnt, "total": tot, "mean_margin": round(marg, 3)})

    # Also test Stage-1 Recall vs LoRA-only vs Full `.imepack` on 30 famous classical Pinyin queries!
    classical_pinyin_queries = [
        ("mingyuejishiyou", "明月几时有"),
        ("nianquuqqianliyanbo", "念去去，千里烟波"),
        ("junbujianhuanghezhishuitianshanglai", "君不见黄河之水天上来"),
        ("luoxiayuguwuqifei", "落霞与孤鹜齐飞"),
        ("qiushuigongchangtianyise", "秋水共长天一色"),
        ("renyoubeihuanlihe", "人有悲欢离合"),
        ("danyuanrenchangjiu", "但愿人长久"),
        ("qianligongchanjuan", "千里共婵娟"),
        ("wubianluomuxiaoxiaoxia", "无边落木萧萧下"),
        ("bujinchangjiangungunlai", "不尽长江滚滚来"),
        ("xunxunmimi", "寻寻觅觅"),
        ("lenglengqingqing", "冷冷清清"),
        ("zhonglixuntaqianbaidu", "众里寻他千百度"),
        ("zuiwengzhiyibuzaijiu", "醉翁之意不在酒"),
        ("xiantianxiazhiyoueryou", "先天下之忧而忧"),
        ("chuyunierburan", "出淤泥而不染"),
        ("beimingyouyu", "北冥有鱼"),
        ("shanbuzaigao", "山不在高"),
        ("duijiudangge", "对酒当歌"),
        ("yangliuanxiaofengcanyue", "杨柳岸，晓风残月"),
    ]
    chime_bin = Path.home() / ".target/release/chime"
    if not chime_bin.exists():
        chime_bin = root / "target/release/chime"

    def test_cli_queries(extra_args: list[str]) -> int:
        hits = 0
        for q, expected in classical_pinyin_queries:
            res = subprocess.run(
                [str(chime_bin), "decode", q, *extra_args],
                capture_output=True,
                text=True,
                check=False,
            )
            for ln in res.stdout.splitlines():
                if ln.strip().startswith("1. "):
                    top1 = ln.strip()[3:].split("  [score=")[0].strip()
                    if top1 == expected:
                        hits += 1
                    break
        return hits

    # Build a minimal modern-only dict (without `classics_core_lexicon`) in `/tmp/modern_only.imedict`
    modern_dict = Path("/tmp/chime_modern_only.imedict")
    subprocess.run(
        [
            str(chime_bin),
            "dict-build-base",
            "-o",
            str(modern_dict),
            "--extra-tsv",
            str(dist / "multi_lexicon.tsv"),
            "--ngrams",
            str(dist / "ngrams.tsv"),
        ],
        check=True,
        capture_output=True,
    )
    lora_only_pack = Path("/tmp/chime_lora_only.imepack")
    subprocess.run(
        [
            str(chime_bin),
            "pack",
            "build",
            str(lora_only_pack),
            "--id",
            "classics-lora-only",
            "--routing",
            "classical",
            "--lora",
            str(classics_lora),
        ],
        check=True,
        capture_output=True,
    )
    classics_pack_dir = dist / "packs/classics-poetry.imepack"
    if not classics_pack_dir.exists():
        classics_pack_dir = packs_root / "classics-poetry.imepack"
    tech_pack_dir = dist / "packs/tech-developer.imepack"

    hits_modern_base = test_cli_queries(["--dict", str(modern_dict), "--neural", str(dist / "base_neural")])
    hits_lora_only = test_cli_queries(["--dict", str(modern_dict), "--neural", str(dist / "base_neural"), "--pack", str(lora_only_pack)])
    hits_full_pack = test_cli_queries(["--dict", str(modern_dict), "--neural", str(dist / "base_neural"), "--pack", str(classics_pack_dir)])
    modern_dict.unlink(missing_ok=True)

    print("[Exp 6] Large-Scale Decontaminated Benchmarks (Dev 1,258 + Blind Holdout 1,010)...", flush=True)

    def run_bench_config(suite_rel: str, extra_args: list[str]) -> dict:
        tmp_rec = Path("/tmp/chime_paper_bench_rec.jsonl")
        tmp_rec.unlink(missing_ok=True)
        env = dict(**__import__("os").environ)
        env["MLX_RS_METAL_PATH"] = "/tmp/chime-mlx-metallib"
        subprocess.run(
            [
                str(chime_bin),
                "bench",
                str(root / suite_rel),
                "--dict",
                str(dist / "core.imedict"),
                *extra_args,
                "--record",
                str(tmp_rec),
                "--tag",
                "paper-exp",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        row = json.loads(tmp_rec.read_text(encoding="utf-8").strip().splitlines()[-1])
        tmp_rec.unlink(missing_ok=True)
        return {
            "cases": row["cases"],
            "acc_at_1": row["acc_at_1"],
            "hit_at_3": row["hit_at_3"],
            "hit_at_5": row["hit_at_5"],
            "mrr": row["mrr"],
            "p50_us": row["p50_us"],
        }

    exp6_benchmarks = {
        "dev_1258": {
            "stage1_lattice_only": run_bench_config("benchmarks/dev_suite.tsv", []),
            "stage1_plus_chime_6l": run_bench_config("benchmarks/dev_suite.tsv", ["--neural", str(dist / "base_neural")]),
            "stage1_plus_chime_6l_packs": run_bench_config(
                "benchmarks/dev_suite.tsv",
                ["--neural", str(dist / "base_neural"), "--pack", str(classics_pack_dir), "--pack", str(tech_pack_dir)],
            ),
        },
        "blind_holdout_1010": {
            "stage1_lattice_only": run_bench_config("benchmarks/test_holdout.tsv", []),
            "stage1_plus_chime_6l": run_bench_config("benchmarks/test_holdout.tsv", ["--neural", str(dist / "base_neural")]),
            "stage1_plus_chime_6l_packs": run_bench_config(
                "benchmarks/test_holdout.tsv",
                ["--neural", str(dist / "base_neural"), "--pack", str(classics_pack_dir), "--pack", str(tech_pack_dir)],
            ),
        },
    }
    print(f"  Dev (1,258): {exp6_benchmarks['dev_1258']}", flush=True)
    print(f"  Blind Holdout (1,010): {exp6_benchmarks['blind_holdout_1010']}", flush=True)

    summary = {
        "exp1_compression_pareto": exp1,
        "exp2_cartridge_decoupling": {
            "classical_queries_total": len(classical_pinyin_queries),
            "c0_modern_base_hits": hits_modern_base,
            "c1_lora_only_hits": hits_lora_only,
            "c3_full_imepack_hits": hits_full_pack,
            "classics_nll_base": round(nll_classics_base, 4),
            "classics_nll_lora": round(nll_classics_lora, 4),
            "classics_ppl_base": round(math.exp(nll_classics_base), 2),
            "classics_ppl_lora": round(math.exp(nll_classics_lora), 2),
            "tech_nll_base": round(nll_tech_base, 4),
            "tech_nll_lora": round(nll_tech_lora, 4),
            "private_val_nll_base": 5.6279,
            "private_val_nll_lora": 5.2267,
            "fp16_to_4bit_lora_scale_nll": fp16_scale_curve,
            "q4_aligned_lora_scale_nll": q4_scale_curve,
        },
        "exp3_routing_interference": exp3_routing,
        "exp4_lora_fusion_scaling": exp4_scaling,
        "exp5_alpha_sweep": exp5_alpha,
        "exp6_large_scale_benchmarks": exp6_benchmarks,
    }
    out_json = root / "paper" / "experiment_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved complete experimental results to {out_json}!", flush=True)


if __name__ == "__main__":
    main()
