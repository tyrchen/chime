#!/usr/bin/env python3
"""
Distills high-PMI modern Mandarin Bigram collocations (`2\tw1\tw2\tbonus`) directly from the
Apache-2.0 28-layer `Qwen/Qwen3-0.6B-Base` teacher on Apple Silicon MPS.

First-principles PMI law:
    PMI(w1, w2) = ln P_Qwen3(w2 | "。" + w1) - ln P_Qwen3(w2 | "。")

No benchmark data is ever touched; `w1` is simply the top-N most frequent Mandarin words/characters
from `dist/multi_lexicon.tsv`.
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_all_cjk(s: str) -> bool:
    return bool(s) and all("\u4e00" <= ch <= "\u9fa5" for ch in s)


def load_top_mandarin_anchors(lex_tsv: Path, max_words: int) -> tuple[list[str], set[str]]:
    scored: dict[str, float] = {}
    for line in lex_tsv.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("EN\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        w = parts[0]
        if not (1 <= len(w) <= 4) or not is_all_cjk(w):
            continue
        try:
            lp = float(parts[2])
        except ValueError:
            continue
        if w not in scored or lp > scored[w]:
            scored[w] = lp
    ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    anchors = [w for w, _ in ordered[:max_words]]
    vocab_set = set(scored.keys())
    return anchors, vocab_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", default="dist/multi_lexicon.tsv")
    parser.add_argument("--base-ngrams", default="dist/ngrams.tsv")
    parser.add_argument("--out", default="dist/ngrams_expanded.tsv")
    parser.add_argument("--max-anchors", type=int, default=12000)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--min-pmi", type=float, default=1.35)
    parser.add_argument("--min-prob", type=float, default=1.2e-3)
    args = parser.parse_args()

    snaps = sorted(
        glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/*"))
    )
    if not snaps:
        raise FileNotFoundError("Qwen/Qwen3-0.6B-Base snapshot not found in HuggingFace cache")
    snap_dir = snaps[-1]

    anchors, lex_vocab = load_top_mandarin_anchors(Path(args.lexicon), args.max_anchors)
    print(f"Loaded {len(anchors)} Mandarin anchor words (vocab={len(lex_vocab)})", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(snap_dir)
    model = AutoModelForCausalLM.from_pretrained(
        snap_dir,
        dtype=torch.float16 if device == "mps" else torch.float32,
    ).to(device)
    model.eval()

    # Map valid pure-CJK single-token pieces (1..=4 chars) that exist in our lexicon
    cjk_tok_ids: list[int] = []
    cjk_tok_strs: list[str] = []
    for tid in range(tok.vocab_size):
        piece = tok.decode([tid])
        if 1 <= len(piece) <= 4 and is_all_cjk(piece) and piece in lex_vocab:
            cjk_tok_ids.append(tid)
            cjk_tok_strs.append(piece)
    cjk_idx_tensor = torch.tensor(cjk_tok_ids, dtype=torch.long, device=device)
    print(f"Mapped {len(cjk_tok_ids)} CJK target tokens in Qwen3 vocab", flush=True)

    anchor_prefix_id = tok.encode("。", add_special_tokens=False)[0]
    with torch.no_grad():
        base_logits = model(
            torch.tensor([[anchor_prefix_id]], dtype=torch.long, device=device)
        ).logits[0, -1].float()
        base_lps = F.log_softmax(base_logits, dim=-1).index_select(0, cjk_idx_tensor).cpu()

    # Group anchors by token length for zero-padding exact batched inference
    by_len: dict[int, list[tuple[str, list[int]]]] = {}
    for w in anchors:
        tids = tok.encode(w, add_special_tokens=False)
        if 1 <= len(tids) <= 4:
            by_len.setdefault(len(tids), []).append((w, tids))

    bigrams: dict[tuple[str, str], float] = {}
    # Preserve existing hand-verified / base ngrams first
    base_path = Path(args.base_ngrams)
    extra_trigrams: list[str] = []
    if base_path.exists():
        for raw in base_path.read_text(encoding="utf-8").splitlines():
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) == 4 and parts[0] == "2":
                try:
                    bigrams[(parts[1], parts[2])] = float(parts[3])
                except ValueError:
                    pass
            elif len(parts) == 5 and parts[0] == "3":
                extra_trigrams.append(ln)

    min_log_prob = math.log(args.min_prob)
    batch_size = 512
    with torch.no_grad():
        for k_len, items in sorted(by_len.items()):
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                mat = torch.tensor(
                    [[anchor_prefix_id] + tids for _, tids in chunk],
                    dtype=torch.long,
                    device=device,
                )
                logits = model(mat).logits[:, -1, :].float()
                lps = F.log_softmax(logits, dim=-1).index_select(1, cjk_idx_tensor)
                top_vals, top_idxs = torch.topk(lps, k=min(args.top_k, len(cjk_tok_ids)), dim=-1)
                top_vals_cpu = top_vals.cpu()
                top_idxs_cpu = top_idxs.cpu()
                for row_i, (w1, _) in enumerate(chunk):
                    for col_j in range(top_vals_cpu.shape[1]):
                        lp = float(top_vals_cpu[row_i, col_j].item())
                        if lp < min_log_prob:
                            break
                        c_idx = int(top_idxs_cpu[row_i, col_j].item())
                        w2 = cjk_tok_strs[c_idx]
                        if w2 == w1:
                            continue
                        pmi = lp - float(base_lps[c_idx].item())
                        if pmi < args.min_pmi:
                            continue
                        # Smoothly map PMI in [1.35, 6.5] -> additive bigram bonus in [0.30, 1.95]
                        bonus = min(1.95, max(0.30, 0.30 + (pmi - args.min_pmi) * 0.32))
                        key = (w1, w2)
                        if key not in bigrams or bonus > bigrams[key]:
                            bigrams[key] = bonus

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Distilled Apache-2.0 Qwen3-0.6B-Base PMI Bigrams & Trigrams"]
    for (w1, w2), bonus in sorted(bigrams.items()):
        lines.append(f"2\t{w1}\t{w2}\t{bonus:.3f}")
    lines.extend(extra_trigrams)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(bigrams)} bigrams + {len(extra_trigrams)} trigrams -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
