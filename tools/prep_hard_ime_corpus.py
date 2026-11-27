#!/usr/bin/env python3
"""
Builds a strictly zero-contamination Hard-IME + Bilingual-Replay Stage-3 Polish Corpus
(`12M tokens`, `seq_len=192`) for surgical low-LR post-distillation on top of the winning
80M general-domain student:

  - 30% Copy-edited Grammar & Critical Homophone bucket (`得/地`, `再/在`, `她/它`, `权力/权利`,
        `必需/必须`, `反映/反应`, `坐/座/做/作`, `交代/交待`, `以致/以至`, `原型/原形`, ...)
        sourced strictly from `Wikipedia zh` (60%) + `FineWeb-Edu-Chinese-V2.1` (40%) to avoid
        colloquial web `的/得/地` noise.
  - 25% Modern Heteronym / Polyphone bucket (`行/重/长/乐/觉/调/传/处/着/差/参/率/还/没/...`)
        balanced across Wikipedia (40%), FineWeb-Edu (40%), and BelleGroup (20%).
  - 25% Chinese-English Code-Switching & Tech-Prose Replay bucket (clean bilingual sentences
        containing natural ASCII tech terms like API/Python/Docker/Rust/Git/JSON/HTTP/React/SQL
        mixed with Chinese), preventing catastrophic forgetting of the 11,321 ASCII student tokens
        and directly lifting `code_switching` & `tech_prose_writing`.
  - 20% Standalone Short/Medium Chinese sentences (6..24 chars), anchored with `"。，"` to align
        RoPE positions 0..24 with zero-context IME typing.

Strictly excludes `m-a-p/COIG-CQIA` and `clue/clue` (100% zero benchmark contamination).
Output format (`blocks.pt` + `homophone_table.pt`) is directly consumable by `distill_v15_cached.py`!
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import time

import torch
from datasets import load_dataset
from opencc import OpenCC

from distill_v15_cached import (
    SUB_K,
    CompactTokenizer,
    build_homophone_table,
)

T2S = OpenCC("t2s")
SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
ASCII_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_+.#-]{1,20}\b")

NOISY_SUBSTRINGS = (
    "http",
    "www.",
    "://",
    "|",
    "`",
    "{",
    "}",
    "<",
    ">",
    "###",
    "```",
    "def ",
    "import ",
    "return ",
    "function ",
    "SELECT ",
    "CREATE TABLE",
    "\\n",
    "nbsp",
)

CN_TECH_TRIGGERS = (
    "推理", "量化", "缓存", "并发", "无锁", "微服务", "数据结构", "内存安全", "零成本",
    "低秩", "微调", "词表", "嵌入层", "算子", "吞吐", "延迟", "访存", "带宽", "蒸馏",
    "编译器", "内核", "分布式", "链路追踪", "数据库", "索引", "异步", "协程", "线程池",
    "容器", "镜像", "网关", "负载均衡", "序列化", "反序列化", "鉴权", "加密", "哈希",
    "注意力机制", "自回归", "分词器", "上下文", "预训练", "对齐", "梯度", "反向传播",
)

GRAMMAR_TRIGGERS = (
    "得", "地", "再", "她", "它", "权力", "权利", "必需", "必须",
    "反映", "反应", "座位", "作为", "交代", "交待", "以致", "以至",
    "原型", "原形", "吧", "久", "盛", "剩", "饼", "病", "版", "帐", "账",
    "度过", "渡过", "制定", "制订", "启用", "起用", "截止", "截至",
    "年轻", "年纪", "做客", "作客", "委曲", "委屈", "必需品", "必须得",
)

POLYPHONE_TRIGGERS = (
    "银行", "行业", "行走", "运行", "重要", "重新", "重复", "长度", "成长", "擅长",
    "音乐", "快乐", "觉得", "睡觉", "调整", "调查", "传递", "传说", "自传", "处理",
    "到处", "看着", "着急", "差别", "出差", "差不多", "参加", "人参", "效率", "率领",
    "归还", "还是", "没有", "沉没", "为了", "因为", "发现", "头发", "干净", "干活",
    "给予", "省钱", "反省", "朝阳", "朝代", "降落", "投降", "新鲜", "朝鲜", "模型",
    "模样", "下载", "记载", "提供", "供奉", "数据", "数落", "计划", "划算", "血液",
    "流血", "薄弱", "薄荷", "倒塌", "倒车", "时间", "间隔", "测量", "数量", "角色",
    "主角", "折腾", "打折", "积累", "劳累", "弹琴", "子弹", "兴起", "高兴", "便利",
    "便宜", "应当", "答应", "埋怨", "掩埋", "奔跑", "投奔", "勉强", "强大", "强迫",
    "会计", "会议", "单薄", "姓单", "宿舍", "星宿", "露面", "露水", "校对", "学校",
    "禁止", "不禁", "系鞋带", "关系", "鲜见", "鲜花", "颤抖", "颤栗", "炮制", "大炮",
)

TECH_HINTS = (
    "API", "SDK", "CPU", "GPU", "LLM", "AI", "ML", "NLP", "HTTP", "HTTPS", "JSON", "YAML",
    "SQL", "NoSQL", "HTML", "CSS", "DOM", "URL", "URI", "TCP", "UDP", "DNS", "SSH", "TLS",
    "SSL", "JWT", "OAuth", "REST", "gRPC", "GraphQL", "CI", "CD", "PR", "Git", "GitHub",
    "Docker", "Kubernetes", "K8s", "Linux", "macOS", "iOS", "Android", "Windows", "Unix",
    "Python", "Rust", "Swift", "Java", "Go", "C++", "JavaScript", "TypeScript", "React",
    "Vue", "Next", "Node", "Redis", "MySQL", "Postgres", "MongoDB", "Kafka", "Nginx",
    "AWS", "GCP", "Azure", "S3", "Lambda", "EC2", "CDN", "CLI", "IDE", "VSCode", "Vim",
    "PyTorch", "TensorFlow", "CUDA", "MLX", "ONNX", "Transformer", "LoRA", "RAG", "Agent",
    "Token", "Prompt", "Model", "Benchmark", "Server", "Client", "App", "Web", "Cloud",
    "Data", "Cache", "Queue", "Thread", "Async", "Sync", "Debug", "Build", "Deploy", "Test",
    "Release", "Version", "Config", "Schema", "Query", "Index", "Memory", "Disk", "Log",
    "Bug", "Fix", "Issue", "Commit", "Branch", "Merge", "Diff", "Patch", "Review", "Code",
    "iPhone", "MacBook", "iPad", "Apple", "Google", "OpenAI", "Claude", "Gemini", "Qwen",
    "Excel", "Word", "PDF", "PPT", "Markdown", "Slack", "Zoom", "Notion", "Figma", "Chrome",
)
TECH_HINTS_LOWER = {t.lower() for t in TECH_HINTS}


def has_noise(s: str) -> bool:
    return any(ch in s for ch in NOISY_SUBSTRINGS)


def is_clean_cn_sentence(s: str) -> bool:
    n = len(s)
    if n < 6 or n > 72 or has_noise(s):
        return False
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fa5")
    return cjk >= n * 0.72


def is_clean_bilingual_sentence(s: str) -> bool:
    n = len(s)
    if n < 10 or n > 120 or has_noise(s):
        return False
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fa5")
    if cjk < 4:
        return False
    cjk_ratio = cjk / n
    if not (0.28 <= cjk_ratio <= 0.88):
        return False
    words = ASCII_WORD_RE.findall(s)
    if not words or len(words) > 10:
        return False
    # Require at least one natural ASCII word (len >= 2 letters, ideally tech/everyday English)
    valid_words = [w for w in words if sum(ch.isalpha() for ch in w) >= 2]
    if not valid_words:
        return False
    # Filter out citation noise like "et al" or pure roman numerals
    if all(w.lower() in {"et", "al", "pp", "vol", "no", "fig", "eq", "ii", "iii", "iv", "vi"} for w in valid_words):
        return False
    return True


def is_tech_bilingual_priority(s: str) -> bool:
    words = ASCII_WORD_RE.findall(s)
    return any(w.lower() in TECH_HINTS_LOWER for w in words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--token-pinyin", default="")
    parser.add_argument("--max-tokens", type=int, default=12_000_000)
    parser.add_argument("--seq-len", type=int, default=192)
    args = parser.parse_args()

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
        {"group_ids": group_ids, "group_mask": group_mask, "group_weight": group_weight},
        work_dir / "homophone_table.pt",
    )

    target_blocks = args.max_tokens // args.seq_len
    n_gram_target = int(target_blocks * 0.30)
    n_poly_target = int(target_blocks * 0.25)
    n_bili_target = int(target_blocks * 0.25)
    n_short_target = target_blocks - n_gram_target - n_poly_target - n_bili_target

    # Generous sentence caps (~14 sentences per 192-token block)
    cap_gram = n_gram_target * 16
    cap_poly = n_poly_target * 16
    cap_bili = n_bili_target * 14
    cap_short = n_short_target * 20

    # Balanced per-source quotas so no single dataset monopolizes any bucket:
    # - Grammar: 60% Wikipedia zh + 40% FineWeb-Edu (0% colloquial BelleGroup!)
    # - Polyphone: 40% Wikipedia + 40% FineWeb-Edu + 20% BelleGroup
    # - Bilingual/Tech: 15% Wikipedia + 50% FineWeb-Edu + 35% BelleGroup
    # - Short standalone: 15% Wikipedia + 35% FineWeb-Edu + 50% BelleGroup
    source_specs = [
        (
            "wikimedia/wikipedia",
            "20231101.zh",
            "train",
            ["text"],
            {
                "gram": int(cap_gram * 0.60),
                "poly": int(cap_poly * 0.40),
                "bili": int(cap_bili * 0.10),
                "short": int(cap_short * 0.15),
            },
            320_000,
        ),
        (
            "opencsg/FineWeb-Edu-Chinese-V2.1",
            None,
            "train",
            ["text"],
            {
                "gram": cap_gram - int(cap_gram * 0.60),
                "poly": int(cap_poly * 0.40),
                "bili": int(cap_bili * 0.55),
                "short": int(cap_short * 0.35),
            },
            500_000,
        ),
        (
            "BelleGroup/train_0.5M_CN",
            None,
            "train",
            ["instruction", "output"],
            {
                "gram": 0,
                "poly": cap_poly - int(cap_poly * 0.80),
                "bili": cap_bili - int(cap_bili * 0.65),
                "short": cap_short - int(cap_short * 0.50),
            },
            500_000,
        ),
    ]

    gram_sents: list[str] = []
    poly_sents: list[str] = []
    bili_tech_sents: list[str] = []
    bili_gen_sents: list[str] = []
    cn_tech_sents: list[str] = []
    short_sents: list[str] = []
    cap_cn_tech = int(cap_bili * 0.30)
    cap_bili_gen = int(cap_bili * 0.20)

    for repo, subset, split, fields, quota, max_rows in source_specs:
        got = {"gram": 0, "poly": 0, "bili": 0, "short": 0}
        print(
            f"Scanning {repo} with quotas={quota} "
            f"(totals so far: gram={len(gram_sents):,}, poly={len(poly_sents):,}, "
            f"bili={len(bili_tech_sents)+len(bili_gen_sents):,}, short={len(short_sents):,})...",
            flush=True,
        )
        ds = (
            load_dataset(repo, subset, split=split, streaming=True)
            if subset
            else load_dataset(repo, split=split, streaming=True)
        )
        for row_idx, row in enumerate(ds):
            if row_idx >= max_rows:
                break
            if all(got[k] >= quota[k] for k in quota):
                break
            merged = "\n".join(str(row.get(f, "")).strip() for f in fields if row.get(f))
            if len(merged) < 12:
                continue
            for raw_s in SENT_SPLIT_RE.split(merged[:1800]):
                s = raw_s.strip()
                if not s:
                    continue
                if repo == "wikimedia/wikipedia":
                    s = T2S.convert(s)
                # 1) Bilingual / Tech Code-Switching check FIRST (relaxed CJK ratio)
                if got["bili"] < quota["bili"] and is_clean_bilingual_sentence(s):
                    if is_tech_bilingual_priority(s):
                        bili_tech_sents.append(s)
                        got["bili"] += 1
                    elif len(bili_gen_sents) < cap_bili_gen:
                        bili_gen_sents.append(s)
                        got["bili"] += 1
                    continue
                # 2) Pure-Chinese buckets (Simplified Chinese only)
                if not is_clean_cn_sentence(s):
                    continue
                if got["bili"] < quota["bili"] and len(cn_tech_sents) < cap_cn_tech and any(t in s for t in CN_TECH_TRIGGERS):
                    cn_tech_sents.append(s)
                    got["bili"] += 1
                elif got["poly"] < quota["poly"] and any(t in s for t in POLYPHONE_TRIGGERS):
                    poly_sents.append(s)
                    got["poly"] += 1
                elif got["gram"] < quota["gram"] and any(t in s for t in GRAMMAR_TRIGGERS):
                    gram_sents.append(s)
                    got["gram"] += 1
                elif got["short"] < quota["short"] and 6 <= len(s) <= 24:
                    short_sents.append(s)
                    got["short"] += 1

        print(
            f"  Done {repo} @ row {row_idx:,}: got={got} "
            f"(bili_tech={len(bili_tech_sents):,}, bili_gen={len(bili_gen_sents):,}, cn_tech={len(cn_tech_sents):,})",
            flush=True,
        )

    bili_sents = bili_tech_sents + bili_gen_sents + cn_tech_sents
    print(
        f"Collected targeted sentences: gram={len(gram_sents):,}, poly={len(poly_sents):,}, "
        f"bili={len(bili_sents):,} (en_tech={len(bili_tech_sents):,}, cn_tech={len(cn_tech_sents):,}), "
        f"short={len(short_sents):,}",
        flush=True,
    )
    for name, arr in [
        ("gram", gram_sents),
        ("poly", poly_sents),
        ("bili", bili_sents),
        ("short", short_sents),
    ]:
        print(f"  Samples [{name}]: {arr[:3]}", flush=True)

    anchor_ids = c_tok.encode("。，")
    sep_id = c_tok.p2i.get("。", 2)
    seq_len = args.seq_len

    def pack_anchored(sents: list[str], limit_blocks: int, seed: int, anchor_prob: float = 1.0) -> list[list[int]]:
        rng = random.Random(seed)
        sents_copy = list(sents)
        rng.shuffle(sents_copy)
        out_blocks: list[list[int]] = []
        cur: list[int] = []
        for s in sents_copy:
            if len(out_blocks) >= limit_blocks:
                break
            s_ids = [i for i in c_tok.encode(s) if i != 1]
            if len(s_ids) < 4:
                continue
            if not cur and rng.random() < anchor_prob:
                cur.extend(anchor_ids)
            if len(cur) + len(s_ids) + 1 <= seq_len:
                cur.extend(s_ids)
                cur.append(sep_id)
            else:
                need = seq_len - len(cur)
                if need > 0:
                    cur.extend(s_ids[:need])
                if len(cur) == seq_len:
                    out_blocks.append(cur)
                cur = (list(anchor_ids) if rng.random() < anchor_prob else []) + s_ids[: seq_len - len(anchor_ids)]
                if len(cur) < seq_len:
                    cur.append(sep_id)
                if len(cur) == seq_len:
                    out_blocks.append(cur)
                    cur = []
        return out_blocks[:limit_blocks]

    blocks_gram = pack_anchored(gram_sents, n_gram_target, 101, anchor_prob=0.6)
    blocks_poly = pack_anchored(poly_sents, n_poly_target, 202, anchor_prob=0.6)
    blocks_bili = pack_anchored(bili_sents, n_bili_target, 303, anchor_prob=0.5)
    blocks_short = pack_anchored(short_sents, n_short_target, 404, anchor_prob=1.0)
    blocks = blocks_gram + blocks_poly + blocks_bili + blocks_short

    packed = torch.tensor(blocks, dtype=torch.int32)
    perm = torch.randperm(len(packed), generator=torch.Generator().manual_seed(7777))
    packed = packed[perm].contiguous()
    out_path = work_dir / "blocks.pt"
    torch.save(packed, out_path)

    # Quick token composition audit (CJK vs ASCII token ratio)
    ascii_id_set = {i for i, s in enumerate(vocab_strings) if s.isascii() and any(c.isalpha() for c in s)}
    sample_flat = packed[:2000].reshape(-1).tolist()
    ascii_tok_pct = 100.0 * sum(1 for tid in sample_flat if tid in ascii_id_set) / max(1, len(sample_flat))
    meta = {
        "total_blocks": len(packed),
        "total_tokens": int(packed.numel()),
        "seq_len": seq_len,
        "bucket_blocks": {
            "grammar": len(blocks_gram),
            "polyphone": len(blocks_poly),
            "bilingual_tech": len(blocks_bili),
            "short_standalone": len(blocks_short),
        },
        "ascii_alpha_token_pct": round(ascii_tok_pct, 2),
    }
    (work_dir / "polish_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Saved {len(packed):,} Stage-3 Polish blocks ({packed.numel() / 1e6:.2f}M tokens, "
        f"ASCII-alpha tokens={ascii_tok_pct:.2f}%, buckets={meta['bucket_blocks']}) "
        f"in {time.perf_counter() - t0:.1f}s -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
