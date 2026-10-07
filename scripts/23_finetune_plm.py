#!/usr/bin/env python3
"""
PHASE II, STEP 1 - Fine-tune ESM-2 on CryptoBench and emit a leakage-clean embedding cache.

Why
---
Phase I measured what a spatial residual adds on top of a FROZEN ESM-2 head. The open
question, and the one that decides whether the method is competitive with the published
state of the art, is whether that residual survives when the language model itself is
adapted to the task. This script produces the fine-tuned counterpart of every Phase-I
input so that the whole Phase-I pipeline can be re-run unchanged on top of it.

Cross-fitting (the point of this script)
---------------------------------------
A fine-tuned encoder has seen the labels of its training folds, so its embeddings are not
exchangeable across folds. For official fold f the encoder is trained on the other three
folds and applied to f; for the 222 test proteins the encoder is trained on all 885
training proteins. The resulting cache is therefore exactly as leakage-clean as the frozen
one and can be dropped into scripts 10-19 by pointing `work.esm_cache` at it.

Outputs
-------
work/cache_esm_ft/<pid>.npz                per-residue fine-tuned embeddings, residue order
                                           identical to work/cache_struct/<pid>.npz
work/cache_esm_ft/model_name.txt           provenance
work/cache_esm_ft/manifest.json            which encoder produced each protein
results/exp/base_plm_ft/{oof,test}_predictions.csv
                                           the fine-tuned model's OWN head, i.e. the
                                           published-style sequence baseline, used as the
                                           Phase-II anchor for reporting
checkpoints/ft_<tag>.pt                    adapter/head weights (small with --mode lora)

Modes
-----
--mode lora      (default) rank-r adapters on attention query/value; trains ~0.1-0.5% of
                 parameters, fits ESM2-650M on ~12 GB and ESM2-3B on ~40 GB with
                 --grad-checkpointing.
--mode top<N>    full fine-tuning of the top N transformer layers.
--mode full      everything (needs a large GPU; included for completeness).

Usage
-----
  python scripts/23_finetune_plm.py --config config_ft.yaml \
      --model-name facebook/esm2_t33_650M_UR50D --mode lora --epochs 3
  # then, with work.esm_cache -> ./work/cache_esm_ft and work.results -> ./results_ft:
  python scripts/11_baseline_multiseed.py --config config_ft.yaml
  python scripts/15_anchored_spatial_gnn.py --config config_ft.yaml

Resumable: finished folds are skipped. Use --only-fold to shard across GPUs.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from cryptogrid.common import ensure_dir
from cryptogrid.exp_common import Store, canonical_frame, exp_dir, make_ctx, save_preds, write_json
from cryptogrid.finetune import (FineTuneModel, build_samples, inject_lora, predict_samples,
                                 train_finetune, trainable_report, unfreeze_top_layers)


def build_model(args, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name)
    enc = AutoModel.from_pretrained(args.model_name)
    if args.grad_checkpointing:
        enc.gradient_checkpointing_enable()
        enc.config.use_cache = False
    if args.mode == "lora":
        n = inject_lora(enc, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        print(f"[model] LoRA on {n} projections (r={args.lora_r}, alpha={args.lora_alpha})")
    elif args.mode == "full":
        for p in enc.parameters():
            p.requires_grad_(True)
    else:
        m = re.fullmatch(r"top(\d+)", args.mode)
        if not m:
            raise SystemExit(f"unknown --mode {args.mode}")
        unfreeze_top_layers(enc, int(m.group(1)))
        print(f"[model] fine-tuning the top {m.group(1)} encoder layers")
    model = FineTuneModel(enc, hidden=args.hidden, dropout=args.dropout).to(device)
    print(f"[model] {args.model_name} | {trainable_report(model)}")
    return model, tok


def run_one(args, ctx, store, device, tag, fit_ids, pred_ids):
    """Train on fit_ids, predict pred_ids. Returns {pid: {'z','emb'}}."""
    ck_root = args.checkpoint_dir or (Path(ctx.cfg["work"].get("root", "work")) / "checkpoints")
    ck = ensure_dir(Path(ck_root)) / f"ft_{tag}.pt"
    cache = ensure_dir(Path(ctx.cfg["work"]["esm_cache"]))
    done = all((cache / f"{p}.npz").exists() for p in pred_ids)
    pred_file = cache / f"_pred_{tag}.npz"
    if done and pred_file.exists() and not args.overwrite:
        print(f"[{tag}] cached")
        z = np.load(pred_file, allow_pickle=False)
        return {p: {"z": z[p]} for p in pred_ids}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    model, tok = build_model(args, device)
    fit = build_samples(store, fit_ids, with_labels=True)
    print(f"[{tag}] train on {len(fit_ids)} proteins / {len(fit)} chains, predict {len(pred_ids)}")
    t0 = time.time()
    train_finetune(model, tok, fit, device=device, epochs=args.epochs, lr_head=args.lr_head,
                   lr_encoder=args.lr_encoder, weight_decay=args.weight_decay,
                   max_tokens=args.max_tokens, accum=args.accum, seed=args.seed,
                   amp=not args.no_amp, tag=tag)
    if args.save_checkpoints:
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        torch.save({k: v.cpu() for k, v in model.state_dict().items() if k in trainable}, ck)
        print(f"  [{tag}] checkpoint -> {ck}")

    pred = predict_samples(model, tok, build_samples(store, pred_ids, with_labels=False),
                           device=device, max_tokens=args.max_tokens, overlap=args.overlap,
                           want_embeddings=True)
    for pid in pred_ids:
        keys = store.labels(pid)[1]
        e = pred[pid]["emb"]
        if len(e) != len(keys):
            raise RuntimeError(f"{pid}: {len(e)} embedded residues vs {len(keys)} in the struct cache")
        np.savez_compressed(cache / f"{pid}.npz", X_esm=e.astype(np.float16),
                            residue_keys=keys.astype("U"), pdb_id=np.asarray(pid))
    np.savez_compressed(pred_file, **{p: pred[p]["z"] for p in pred_ids})
    print(f"[{tag}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_ft.yaml")
    ap.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--mode", default="lora", help="lora | top<N> | full")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=1022)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--accum", type=int, default=8, help="chains per optimiser step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--checkpoint-dir", default=None, help="default: <work.root>/checkpoints")
    ap.add_argument("--only-fold", default=None, help="train-0 | train-1 | train-2 | train-3 | full")
    ap.add_argument("--embed-only", action="store_true",
                    help="skip training: load --checkpoint and embed/predict every protein in this config. "
                         "Used to apply an encoder trained on all 885 CryptoBench proteins to an EXTERNAL set.")
    ap.add_argument("--checkpoint", default=None, help="checkpoint written by a previous run with --save-checkpoints")
    ap.add_argument("--name", default="base_plm_ft")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ctx = make_ctx(args.config, require_esm=False)   # this script creates the cache
    store = Store(ctx)
    cache = Path(ctx.cfg["work"]["esm_cache"])
    frozen = Path(ctx.cfg["work"].get("frozen_esm_cache", "")) if ctx.cfg["work"].get("frozen_esm_cache") else None
    if frozen and cache.resolve() == frozen.resolve():
        raise SystemExit("work.esm_cache must differ from the Phase-I frozen cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[WARN] no CUDA device: fine-tuning on CPU is not practical for real runs")
    print(f"[device] {device} | cache -> {cache} | results -> {ctx.cfg['work']['results']}")

    if args.embed_only:
        if not args.checkpoint or not Path(args.checkpoint).exists():
            raise SystemExit("--embed-only needs an existing --checkpoint")
        model, tok = build_model(args, device)
        sd = torch.load(args.checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        trained = [k for k in sd]
        print(f"[embed-only] loaded {len(trained)} tensors from {args.checkpoint}; "
              f"{len(unexpected)} unexpected")
        if unexpected:
            raise SystemExit(f"checkpoint does not match this architecture: {unexpected[:5]}")
        ids = list(ctx.train_ids) + list(ctx.test_ids)
        todo = [p for p in ids if args.overwrite or not (cache / f"{p}.npz").exists()]
        print(f"[embed-only] {len(todo)}/{len(ids)} proteins to embed")
        pred = predict_samples(model, tok, build_samples(store, todo, with_labels=False),
                               device=device, max_tokens=args.max_tokens, overlap=args.overlap,
                               want_embeddings=True)
        ensure_dir(cache)
        zz = {}
        for pid in todo:
            keys = store.labels(pid)[1]
            e = pred[pid]["emb"]
            if len(e) != len(keys):
                raise RuntimeError(f"{pid}: {len(e)} embedded residues vs {len(keys)} in the struct cache")
            np.savez_compressed(cache / f"{pid}.npz", X_esm=e.astype(np.float16),
                                residue_keys=keys.astype("U"), pdb_id=np.asarray(pid))
            zz[pid] = np.asarray(pred[pid]["z"], dtype=np.float64)
        if set(ctx.test_ids) <= set(zz):
            df = canonical_frame(ctx, "test")
            df["z"] = np.concatenate([zz[p] for p in ctx.test_ids])
            save_preds(ctx, args.name, "test", df)
            print(f"[embed-only] wrote results/exp/{args.name}/test_predictions.csv")
        (cache / "model_name.txt").write_text(
            f"{args.model_name} (fine-tuned, mode={args.mode}, applied from {args.checkpoint})\n", encoding="utf-8")
        print(f"[DONE] embeddings -> {cache}")
        return

    jobs = []
    for f in ctx.fold_names:
        jobs.append((f, [p for p in ctx.train_ids if ctx.fold_of[p] != f],
                     [p for p in ctx.train_ids if ctx.fold_of[p] == f]))
    jobs.append(("full", list(ctx.train_ids), list(ctx.test_ids)))
    if args.only_fold:
        jobs = [j for j in jobs if j[0] == args.only_fold]
        if not jobs:
            raise SystemExit(f"--only-fold {args.only_fold} is not one of {ctx.fold_names + ['full']}")

    z_oof, z_test, manifest = {}, {}, {}
    for tag, fit_ids, pred_ids in jobs:
        pred = run_one(args, ctx, store, device, tag, fit_ids, pred_ids)
        for pid in pred_ids:
            manifest[pid] = tag
            (z_test if tag == "full" else z_oof)[pid] = np.asarray(pred[pid]["z"], dtype=np.float64)

    ensure_dir(cache)
    (cache / "model_name.txt").write_text(f"{args.model_name} (fine-tuned, mode={args.mode})\n", encoding="utf-8")
    old = json.loads((cache / "manifest.json").read_text()) if (cache / "manifest.json").exists() else {}
    old.update(manifest)
    (cache / "manifest.json").write_text(json.dumps(old, indent=2), encoding="utf-8")

    if args.only_fold:
        print(f"\n[PARTIAL] only {args.only_fold} was run; re-run without --only-fold (or for the "
              "remaining folds) before writing predictions.")
        return
    for split, zz, ids in (("oof", z_oof, ctx.train_ids), ("test", z_test, ctx.test_ids)):
        df = canonical_frame(ctx, split)
        df["z"] = np.concatenate([zz[p] for p in ids])
        save_preds(ctx, args.name, split, df)
    write_json(exp_dir(ctx, args.name) / "meta.json", {
        "experiment": args.name, "model": args.model_name, "mode": args.mode,
        "epochs": args.epochs, "lr_head": args.lr_head, "lr_encoder": args.lr_encoder,
        "max_tokens": args.max_tokens, "accum": args.accum, "seed": args.seed,
        "cross_fitting": "fold f predicted by an encoder trained on the other three folds; "
                         "test predicted by an encoder trained on all 885 training proteins",
        "embedding_cache": str(cache),
    })
    print(f"\n[DONE] fine-tuned anchor -> results/exp/{args.name}; embeddings -> {cache}")
    print("Next: run scripts 10/11/15/12/18/19 with the Phase-II config.")


if __name__ == "__main__":
    main()
