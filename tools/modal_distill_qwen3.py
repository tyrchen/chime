#!/usr/bin/env python3
"""
Modal H100 Dual-Route Pure-LM Soft-KL Distillation (`Qwen/Qwen3-14B-Base` -> 14L & 28L 36K-vocab).

Stage 0 (CPU-only, <$0.01):
  Caches `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-14B-Base`, and 65M-char zero-contamination packed
  Route-A token blocks (`82%` natural multi-sentence + `18%` `"。，"` sentence-start anchored) on
  Modal Volume `chime-train-vol`, so neither H100 wastes a single second on downloads or tokenization.

Stage 1 (2x H100 80GB HBM3 in parallel):
  - Route 1 (`14L`, ~135 MB 4-bit MLX, ~2x faster than 28L):
      Uniformly sampled 14 layers from `Qwen3-0.6B-Base`, `steps=5500`, `eff_bs=32`, `lr=1.6e-4`.
  - Route 2 (`28L`, ~257 MB 4-bit MLX, zero layer pruning):
      All 28 intact layers of `Qwen3-0.6B-Base`, gentle `steps=3600`, `eff_bs=32`, `lr=3.5e-5`
      (absorbs `Qwen3-14B` semantics & homophone disambiguation with zero catastrophic forgetting).

Usage:
    modal run tools/modal_distill_qwen3.py
"""

from __future__ import annotations

import json
import pathlib

import modal

APP_NAME = "chime-distill-q14b-h100"
VOL_NAME = "chime-train-vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "datasets>=3.2.0",
        "huggingface_hub>=0.28.0",
        "pypinyin>=0.53.0",
        "safetensors>=0.5.0",
        "numpy>=2.0.0",
        "accelerate>=1.3.0",
    )
    .add_local_file(
        "tools/colab_distill_qwen3_8b_v3.py",
        remote_path="/root/distill_v3_lib.py",
    )
    .add_local_file(
        "dist/base_neural/vocab.json",
        remote_path="/root/vocab_36k.json",
    )
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def _select_layer_indices(num_layers: int, total_layers: int = 28) -> list[int]:
    if num_layers >= total_layers:
        return list(range(total_layers))
    if num_layers == 14 and total_layers == 28:
        # Preserve boundary layers (0, 27) and uniform stride across the full depth
        return [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 27]
    if num_layers == 16 and total_layers == 28:
        return [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27]
    step = (total_layers - 1) / max(1, num_layers - 1)
    return sorted({round(i * step) for i in range(num_layers)})


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=1800,
    volumes={"/vol": vol},
)
def prepare_assets(
    teacher_repo: str = "Qwen/Qwen3-14B-Base",
    init_base: str = "Qwen/Qwen3-0.6B-Base",
    corpus_chars: int = 65_000_000,
    seq_len: int = 192,
) -> str:
    import os
    import sys
    import time

    os.environ["HF_HOME"] = "/vol/hf_cache"
    sys.path.insert(0, "/root")

    import torch
    from huggingface_hub import snapshot_download

    import distill_v3_lib as v3

    t0 = time.perf_counter()
    print(f"=== Stage 0 (CPU): Caching {init_base} & {teacher_repo} ===", flush=True)
    snapshot_download(
        init_base,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"],
    )
    snapshot_download(
        teacher_repo,
        allow_patterns=["*.json", "*.safetensors", "vocab.json", "merges.txt"],
    )
    vol.commit()
    print(f"Cached model weights in {time.perf_counter() - t0:.1f}s", flush=True)

    blocks_path = pathlib.Path(f"/vol/distill_blocks_{corpus_chars // 1_000_000}m_{seq_len}_v4.pt")
    if blocks_path.exists():
        packed = torch.load(blocks_path, weights_only=True)
        print(f"Reusing cached blocks {blocks_path}: shape={tuple(packed.shape)}", flush=True)
        return str(blocks_path)

    vocab_strings: list[str] = json.loads(pathlib.Path("/root/vocab_36k.json").read_text(encoding="utf-8"))
    c_tok = v3.CompactTokenizer(vocab_strings)

    passages = v3.collect_zero_contamination_corpus(target_chars=corpus_chars)

    # Pack 82% natural continuous multi-sentence blocks + 18% `"。，"` sentence-start anchored blocks!
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
        sents = [s.strip() for s in v3.SENT_SPLIT_RE.split(text) if len(s.strip()) >= 6]
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
    packed = packed[torch.randperm(len(packed), generator=torch.Generator().manual_seed(2026))]
    torch.save(packed, blocks_path)
    vol.commit()
    print(
        f"Saved {len(packed):,} v4 blocks ({n_cont:,} continuous + {len(packed) - n_cont:,} anchored) "
        f"in {time.perf_counter() - t0:.1f}s -> {blocks_path}",
        flush=True,
    )
    return str(blocks_path)


@app.function(
    image=image,
    gpu="H100",
    cpu=4.0,
    memory=49152,
    timeout=3600 * 2,
    volumes={"/vol": vol},
)
def distill_on_h100(
    route_name: str,
    num_layers: int,
    steps: int,
    lr: float,
    blocks_path_str: str,
    teacher_repo: str = "Qwen/Qwen3-14B-Base",
    init_base: str = "Qwen/Qwen3-0.6B-Base",
    batch_size: int = 16,
    grad_accum: int = 2,
    embed_lr_mult: float = 0.25,
    kl_full_w: float = 0.65,
    kl_sub_w: float = 1.15,
    feat_cos_w: float = 1.20,
    feat_l1_w: float = 0.20,
) -> str:
    import gc
    import math
    import os
    import sys
    import time

    os.environ["HF_HOME"] = "/vol/hf_cache"
    sys.path.insert(0, "/root")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import distill_v3_lib as v3

    vol.reload()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(
        f"=== [{route_name}] GPU={torch.cuda.get_device_name(0)} | teacher={teacher_repo} "
        f"-> {num_layers}L | steps={steps} bs={batch_size}x{grad_accum} lr={lr} ===",
        flush=True,
    )

    vocab_strings: list[str] = json.loads(pathlib.Path("/root/vocab_36k.json").read_text(encoding="utf-8"))
    c_tok = v3.CompactTokenizer(vocab_strings)

    group_ids, group_mask, group_weight = v3.build_clean_homophone_groups(vocab_strings)
    group_ids_gpu = group_ids.to(device)
    group_mask_gpu = group_mask.to(device)
    group_weight_gpu = group_weight.to(device)

    init_snap = pathlib.Path(
        snapshot_download(
            init_base,
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
    selected_layers = _select_layer_indices(num_layers, int(orig_cfg["num_hidden_layers"]))
    print(f"[{route_name}] Selected {len(selected_layers)} layers: {selected_layers}", flush=True)
    student = v3.Qwen3Student(raw_st, embed_pruned, selected_layers, orig_cfg).to(device)
    del raw_st
    gc.collect()

    print(f"[{route_name}] Loading teacher {teacher_repo} in bfloat16...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_repo, dtype=torch.bfloat16).to(device)
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

    cos_cache, sin_cache = v3.build_rope_cache(
        512,
        int(orig_cfg["head_dim"]),
        float(orig_cfg.get("rope_theta", 1_000_000.0)),
        device,
    )

    packed = torch.load(blocks_path_str, weights_only=True)
    val_size = min(256, max(32, len(packed) // 25))
    train_seqs = packed[val_size:]
    print(f"[{route_name}] Loaded {len(train_seqs):,} train blocks (seq_len={train_seqs.shape[1]})", flush=True)

    # Baseline probe score BEFORE distillation
    p0_ok, p0_tot = v3.evaluate_probes(student, c_tok, cos_cache, sin_cache, device)
    print(f"[{route_name}] Init probes BEFORE distillation: {p0_ok}/{p0_tot}", flush=True)

    backbone_params = [p for n, p in student.named_parameters() if n != "embed_weight"]
    param_groups = [
        {"params": backbone_params, "lr": lr},
        {"params": [student.embed_weight], "lr": lr * embed_lr_mult},
    ]
    if feat_cos_w > 0.0 or feat_l1_w > 0.0:
        param_groups.append({"params": list(feat_proj.parameters()), "lr": max(lr * 2.5, 2.0e-4)})
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    warmup = min(250, steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup)
        if s < warmup
        else 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))),
    )

    out_dir = pathlib.Path(f"/vol/{route_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    student_cfg = {
        "model_type": "qwen3",
        "vocab_size": len(vocab_strings),
        "hidden_size": int(orig_cfg["hidden_size"]),
        "intermediate_size": int(orig_cfg["intermediate_size"]),
        "num_hidden_layers": len(selected_layers),
        "num_attention_heads": int(orig_cfg["num_attention_heads"]),
        "num_key_value_heads": int(orig_cfg["num_key_value_heads"]),
        "head_dim": int(orig_cfg["head_dim"]),
        "rms_norm_eps": float(orig_cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(orig_cfg.get("rope_theta", 1_000_000.0)),
        "tie_word_embeddings": True,
    }
    (out_dir / "config.json").write_text(json.dumps(student_cfg, indent=2), encoding="utf-8")
    (out_dir / "vocab.json").write_text(json.dumps(vocab_strings, ensure_ascii=False), encoding="utf-8")
    vol.commit()

    t0 = time.perf_counter()
    cursor = 0
    student.train()
    feat_proj.train()
    ckpt_steps = {steps // 2, steps}
    ema_loss = None

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        acc_kl_f = 0.0
        acc_kl_s = 0.0
        acc_cos = 0.0

        for _ in range(grad_accum):
            if cursor + batch_size > len(train_seqs):
                train_seqs = train_seqs[torch.randperm(len(train_seqs))]
                cursor = 0
            batch_new = train_seqs[cursor : cursor + batch_size].to(device, non_blocking=True)
            cursor += batch_size
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

                loss_raw = kl_full_w * kl_full + kl_sub_w * kl_sub
                if feat_cos_w > 0.0 or feat_l1_w > 0.0:
                    u_s_proj = feat_proj(u_s).float()
                    u_t_f = u_t.float()
                    feat_cos = 1.0 - F.cosine_similarity(u_s_proj, u_t_f, dim=-1).mean()
                    loss_raw = loss_raw + feat_cos_w * feat_cos
                    if feat_l1_w > 0.0:
                        loss_raw = loss_raw + feat_l1_w * F.smooth_l1_loss(u_s_proj, u_t_f)
                    acc_cos += float(1.0 - feat_cos.item()) / grad_accum
                loss = loss_raw / grad_accum

            loss.backward()
            acc_loss += float(loss.item())
            acc_kl_f += float(kl_full.item()) / grad_accum
            acc_kl_s += float(kl_sub.item()) / grad_accum

        all_params = list(student.parameters()) + (
            list(feat_proj.parameters()) if (feat_cos_w > 0.0 or feat_l1_w > 0.0) else []
        )
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        scheduler.step()
        ema_loss = acc_loss if ema_loss is None else (0.95 * ema_loss + 0.05 * acc_loss)

        if step in (1, 50) or step % 250 == 0 or step == steps:
            dt = time.perf_counter() - t0
            p_ok, p_tot = (
                v3.evaluate_probes(student, c_tok, cos_cache, sin_cache, device)
                if (step == 1 or step % 500 == 0 or step == steps)
                else (-1, 0)
            )
            p_str = f"{p_ok}/{p_tot}" if p_ok >= 0 else "..."
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"[{route_name} Step {step:5d}/{steps}] loss={acc_loss:.4f} (ema={ema_loss:.4f}) "
                f"kl_f={acc_kl_f:.3f} kl_s={acc_kl_s:.3f} cos={acc_cos:.4f} lr={lr_now:.2e} "
                f"| probes={p_str} | {dt:.1f}s ({dt / step * 1000:.1f}ms/step)",
                flush=True,
            )

        if step in ckpt_steps:
            ckpt = out_dir / "student_fp16.safetensors"
            student.export_safetensors(ckpt)
            vol.commit()
            print(
                f"[{route_name}] Committed checkpoint step={step} -> {ckpt} ({ckpt.stat().st_size / 1e6:.1f} MB)",
                flush=True,
            )

    return f"[{route_name}] DONE steps={steps} ema={ema_loss:.4f} elapsed={time.perf_counter() - t0:.1f}s"


@app.local_entrypoint()
def main() -> None:
    blocks_path = prepare_assets.remote(
        teacher_repo="Qwen/Qwen3-14B-Base",
        init_base="Qwen/Qwen3-0.6B-Base",
        corpus_chars=65_000_000,
        seq_len=192,
    )
    print(f"\nAssets ready at {blocks_path}. Spawning 2x H100 distillation jobs in parallel...\n", flush=True)

    # Route 1: 14L sweet-spot (~135 MB 4-bit MLX, ~2x faster than 28L)
    h_14l = distill_on_h100.spawn(
        route_name="distill_14l_q14b",
        num_layers=14,
        steps=5500,
        lr=1.6e-4,
        blocks_path_str=blocks_path,
        teacher_repo="Qwen/Qwen3-14B-Base",
        batch_size=16,
        grad_accum=2,
        embed_lr_mult=0.25,
        kl_full_w=0.65,
        kl_sub_w=1.15,
        feat_cos_w=1.20,
        feat_l1_w=0.20,
    )

    # Route 2: Intact 28L ceiling (~257 MB 4-bit MLX, gentle low-LR refinement on top of 83.80% base)
    h_28l = distill_on_h100.spawn(
        route_name="distill_28l_q14b",
        num_layers=28,
        steps=3600,
        lr=3.5e-5,
        blocks_path_str=blocks_path,
        teacher_repo="Qwen/Qwen3-14B-Base",
        batch_size=16,
        grad_accum=2,
        embed_lr_mult=0.20,
        kl_full_w=0.85,
        kl_sub_w=1.25,
        feat_cos_w=0.40,
        feat_l1_w=0.0,
    )

    res_14l = h_14l.get()
    res_28l = h_28l.get()
    print("\n=== BOTH H100 DISTILLATION ROUTES COMPLETED ===")
    print(res_14l)
    print(res_28l)
