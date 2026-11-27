#!/usr/bin/env python3
"""
Measures the exact Teacher LM Upper Bound (`Qwen3-14B-Base`, optional `Qwen3-32B-Base`)
on `dev_suite_10k` candidate pools dumped by `chime bench`.

NEVER run on `test_holdout*`! Strictly for `dev_suite_10k` diagnostics.

Outputs per-category error decomposition:
  - `Cur@1`:         Current Chime student Top-1 accuracy
  - `Hit@5`:         Current Chime decoder Top-5 candidate recall
  - `T_Rerank5`:     Teacher accuracy reranking ONLY the current Top-5 candidates
                     (= pure-LM headroom with zero decoder changes!)
  - `T_Oracle+Exp`:  Teacher accuracy when `expected` is added to the candidate set
                     (T_Oracle+Exp - T_Rerank5 = decoder/channel/beam generation gap!)
  - `T_36k` vs `T_151k`: compares 36K Chime vocab vs full 151K Qwen3 vocab.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


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


@torch.no_grad()
def score_batch_full_vocab(
    model,
    hf_tok,
    pairs: list[tuple[str, str]],
    device: torch.device,
    batch_size: int = 32,
) -> list[float]:
    """Computes exact sum log P_151k(cand | ctx) + sentence-closure log P(。| ctx+cand)."""
    eos_char = "。"
    encoded = []
    for ctx, cand in pairs:
        prefix = ctx if ctx else "。，"
        p_ids = hf_tok.encode(prefix, add_special_tokens=False)
        c_ids = hf_tok.encode(cand, add_special_tokens=False)
        cl_ids = hf_tok.encode(eos_char, add_special_tokens=False)
        if not p_ids:
            p_ids = [151643]
        full = p_ids + c_ids + cl_ids
        encoded.append((full, len(p_ids), len(c_ids)))

    scores = [0.0] * len(pairs)
    order = sorted(range(len(pairs)), key=lambda i: len(encoded[i][0]))
    pad_id = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else 151643

    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        max_len = max(len(encoded[i][0]) for i in idxs)
        inp = torch.full((len(idxs), max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(idxs), max_len), dtype=torch.long, device=device)
        for bi, orig_i in enumerate(idxs):
            seq = encoded[orig_i][0]
            inp[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attn[bi, : len(seq)] = 1

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits

        lp = F.log_softmax(logits.float(), dim=-1)
        for bi, orig_i in enumerate(idxs):
            seq, p_len, c_len = encoded[orig_i]
            # Score candidate tokens + 0.5 weight on sentence-end closure (matches Route A ClosedSentence)
            s = 0.0
            for j in range(c_len):
                pos = p_len - 1 + j
                s += float(lp[bi, pos, seq[p_len + j]].item())
            if len(seq) > p_len + c_len:
                s += 0.5 * float(lp[bi, p_len + c_len - 1, seq[p_len + c_len]].item())
            scores[orig_i] = s
    return scores


@torch.no_grad()
def score_batch_36k_vocab(
    model,
    lm_head_36k: torch.Tensor,
    new_to_old_gpu: torch.Tensor,
    c_tok: CompactTokenizer,
    pairs: list[tuple[str, str]],
    device: torch.device,
    batch_size: int = 32,
) -> list[float]:
    """Computes exact sum log P_36k(cand | ctx) on the exact Chime 36,864 vocabulary."""
    eos_id = c_tok.p2i.get("。", 2)
    anchor_ids = [i for i in c_tok.encode("。，") if i != 1]
    encoded = []
    for ctx, cand in pairs:
        p_ids = [i for i in c_tok.encode(ctx) if i != 1] if ctx else list(anchor_ids)
        if not p_ids:
            p_ids = list(anchor_ids)
        c_ids = [i for i in c_tok.encode(cand) if i != 1]
        full = p_ids + c_ids + [eos_id]
        encoded.append((full, len(p_ids), len(c_ids)))

    scores = [0.0] * len(pairs)
    order = sorted(range(len(pairs)), key=lambda i: len(encoded[i][0]))

    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        max_len = max(len(encoded[i][0]) for i in idxs)
        inp_new = torch.zeros((len(idxs), max_len), dtype=torch.long, device=device)
        attn = torch.zeros((len(idxs), max_len), dtype=torch.long, device=device)
        for bi, orig_i in enumerate(idxs):
            seq = encoded[orig_i][0]
            inp_new[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attn[bi, : len(seq)] = 1

        inp_old = new_to_old_gpu[inp_new]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=inp_old, attention_mask=attn, use_cache=False).last_hidden_state
            logits = F.linear(h, lm_head_36k)

        lp = F.log_softmax(logits.float(), dim=-1)
        for bi, orig_i in enumerate(idxs):
            seq, p_len, c_len = encoded[orig_i]
            s = 0.0
            for j in range(c_len):
                pos = p_len - 1 + j
                s += float(lp[bi, pos, seq[p_len + j]].item())
            if len(seq) > p_len + c_len:
                s += 0.5 * float(lp[bi, p_len + c_len - 1, seq[p_len + c_len]].item())
            scores[orig_i] = s
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-tsv", required=True, help="dev_suite_10k subset TSV (6 cols)")
    parser.add_argument("--failures-tsv", required=True, help="chime bench --failures-out TSV")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--teacher", default="Qwen/Qwen3-14B-Base")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    t0 = time.perf_counter()
    device = torch.device("cuda")
    vocab_strings = json.loads(pathlib.Path(args.vocab).read_text(encoding="utf-8"))
    c_tok = CompactTokenizer(vocab_strings)

    # Parse all rows from bench-tsv
    all_rows = []
    for line in pathlib.Path(args.bench_tsv).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("category\t"):
            continue
        cols = line.split("\t")
        all_rows.append(
            {
                "cat": cols[0].strip(),
                "scheme": cols[1].strip(),
                "ctx": cols[3].strip(),
                "keys": cols[4].strip(),
                "expected": [x.strip() for x in cols[5].split("|")],
            }
        )

    # Map failures by (cat, scheme, ctx, keys, expected_raw)
    fail_map: dict[tuple[str, str, str, str], tuple[str, list[str]]] = {}
    for line in pathlib.Path(args.failures_tsv).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("category\t"):
            continue
        cols = line.split("\t")
        key = (cols[0].strip(), cols[1].strip(), cols[2].strip(), cols[3].strip())
        rank_str = cols[5].strip()
        cands = [c.strip() for c in cols[6].split(" / ") if c.strip()] if len(cols) > 6 else []
        fail_map[key] = (rank_str, cands)

    print(f"Loaded {len(all_rows)} bench rows ({len(fail_map)} student failures)", flush=True)

    # Build scoring pairs ONLY for failures (measures: how many current failures can the teacher fix?
    # Note: we report both Rerank-Top5 and Oracle+{expected} on failures, giving exact upper-bound
    # gain assuming passing rows stay passing.)
    pairs: list[tuple[str, str]] = []
    row_pair_slices: dict[int, tuple[list[str], int, int]] = {}
    for idx, r in enumerate(all_rows):
        key = (r["cat"], r["scheme"], r["ctx"], r["keys"])
        if key not in fail_map:
            continue
        _, top5 = fail_map[key]
        exp0 = r["expected"][0]
        pool = list(dict.fromkeys(top5 + [exp0]))
        s_idx = len(pairs)
        for cand in pool:
            pairs.append((r["ctx"], cand))
        row_pair_slices[idx] = (pool, s_idx, len(pairs))

    print(f"Scoring {len(pairs)} candidate pairs with {args.teacher}...", flush=True)
    hf_tok = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(device).eval()

    bos_id = 151643
    new_to_old = [bos_id, bos_id]
    for piece in vocab_strings[2:]:
        ids = hf_tok.encode(piece, add_special_tokens=False)
        new_to_old.append(ids[0] if len(ids) == 1 else bos_id)
    new_to_old_gpu = torch.tensor(new_to_old, dtype=torch.long, device=device)
    lm_head_36k = model.get_output_embeddings().weight.detach().index_select(0, new_to_old_gpu).contiguous()

    scores_36k = score_batch_36k_vocab(
        model, lm_head_36k, new_to_old_gpu, c_tok, pairs, device, batch_size=args.batch_size
    )
    scores_151k = score_batch_full_vocab(model, hf_tok, pairs, device, batch_size=args.batch_size)

    stats = defaultdict(lambda: {"n": 0, "cur1": 0, "hit5": 0, "rr5_36k": 0, "ora_36k": 0, "rr5_151k": 0, "ora_151k": 0})
    cat_order: list[str] = []

    for idx, r in enumerate(all_rows):
        cat = r["cat"]
        if cat not in stats:
            cat_order.append(cat)
        st = stats[cat]
        st["n"] += 1
        key = (r["cat"], r["scheme"], r["ctx"], r["keys"])
        if key not in fail_map:
            for k in ("cur1", "hit5", "rr5_36k", "ora_36k", "rr5_151k", "ora_151k"):
                st[k] += 1
            continue

        rank_str, top5 = fail_map[key]
        in_top5 = rank_str != "MISS" and int(rank_str) <= 5
        if in_top5:
            st["hit5"] += 1

        pool, s_lo, s_hi = row_pair_slices[idx]
        s36 = scores_36k[s_lo:s_hi]
        s151 = scores_151k[s_lo:s_hi]
        exp_set = set(r["expected"])

        # Rerank ONLY existing top5
        n_top5 = len(top5)
        if n_top5 > 0:
            best_36_rr = pool[max(range(n_top5), key=lambda i: s36[i])]
            best_151_rr = pool[max(range(n_top5), key=lambda i: s151[i])]
            if best_36_rr in exp_set:
                st["rr5_36k"] += 1
            if best_151_rr in exp_set:
                st["rr5_151k"] += 1

        # Oracle+{expected}: does teacher prefer expected over every generated candidate?
        best_36_ora = pool[max(range(len(pool)), key=lambda i: s36[i])]
        best_151_ora = pool[max(range(len(pool)), key=lambda i: s151[i])]
        if best_36_ora in exp_set:
            st["ora_36k"] += 1
        if best_151_ora in exp_set:
            st["ora_151k"] += 1

    print(
        f"\n=== Teacher Ceiling Decomposition ({args.teacher}, elapsed={time.perf_counter() - t0:.1f}s) ==="
    )
    print(
        f"{'Category':<24} {'N':>4} {'Cur@1':>7} {'Hit@5':>7} {'T36k_RR5':>9} {'T36k_Ora':>9} {'T151k_RR5':>10} {'T151k_Ora':>10}"
    )
    print("-" * 88)
    tot = defaultdict(int)
    for cat in cat_order:
        st = stats[cat]
        n = st["n"]
        for k, v in st.items():
            tot[k] += v
        print(
            f"{cat:<24} {n:>4} {100*st['cur1']/n:6.2f}% {100*st['hit5']/n:6.2f}% "
            f"{100*st['rr5_36k']/n:8.2f}% {100*st['ora_36k']/n:8.2f}% "
            f"{100*st['rr5_151k']/n:9.2f}% {100*st['ora_151k']/n:9.2f}%"
        )
    print("-" * 88)
    n = tot["n"]
    print(
        f"{'OVERALL':<24} {n:>4} {100*tot['cur1']/n:6.2f}% {100*tot['hit5']/n:6.2f}% "
        f"{100*tot['rr5_36k']/n:8.2f}% {100*tot['ora_36k']/n:8.2f}% "
        f"{100*tot['rr5_151k']/n:9.2f}% {100*tot['ora_151k']/n:9.2f}%"
    )


if __name__ == "__main__":
    main()
