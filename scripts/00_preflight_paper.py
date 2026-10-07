#!/usr/bin/env python3
"""
Read-only preflight for run_paper.sh. Changes nothing. Exit code 1 on any blocking problem,
so a 15-20 h replicate run cannot end in a crash at the analysis / plotting stage.

Checks: source files, imports (incl. matplotlib, needed by scripts 21/22), official folds and
caches, frozen selection + SHA-256 of the frozen prediction files, frozen stack inputs vs what
scripts/20_replicate_runs.py can rebuild, test-unlock state, disk space, GPU, and reports the
tree count of the frozen GBM component.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = [
    "run_paper.sh", "cryptogrid/exp_common.py", "cryptogrid/exp_models.py", "cryptogrid/exp_stats.py",
    *[f"scripts/{s}" for s in (
        "11_baseline_multiseed.py", "12_spatial_smoothing.py", "13_pocket_consensus.py",
        "14_gbm_residual_boost.py", "15_anchored_spatial_gnn.py", "16_rank_loss.py",
        "17_aux_distance_multitask.py", "18_constrained_stacker.py", "19_stats_and_freeze.py",
        "20_replicate_runs.py", "21_replicate_analysis.py", "22_paper_report.py")],
]
HARD_IMPORTS = ["numpy", "pandas", "scipy", "sklearn", "torch", "yaml", "lightgbm", "Bio", "matplotlib"]
SOFT_IMPORTS = ["tabulate", "rdkit"]          # optional: nicer PAPER_NUMBERS.md / legacy scripts only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--expected-winner", default="stack_anchor")
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    args = ap.parse_args()
    problems, warns = [], []
    ok = lambda m: print(f"[OK]   {m}")
    info = lambda m: print(f"[INFO] {m}")

    print("=" * 60 + "\nPAPER FINAL PREFLIGHT (read-only)\n" + "=" * 60)

    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    problems += [f"missing source file: {p}" for p in missing]
    if not missing:
        ok("source files")

    for m in HARD_IMPORTS:
        try:
            importlib.import_module(m)
        except Exception as e:
            problems.append(f"cannot import {m}: {e}")
    for m in SOFT_IMPORTS:
        try:
            importlib.import_module(m)
        except Exception:
            warns.append(f"optional package not installed: {m}")
    if not any("cannot import" in p for p in problems):
        ok("runtime dependencies (incl. matplotlib for scripts 21/22)")
    if problems:
        return finish(problems, warns)

    import torch
    from cryptogrid.exp_common import BASELINE, make_ctx
    from cryptogrid.exp_stats import test_gate

    try:
        ctx = make_ctx(args.config)          # raises if a protein lacks a struct/ESM cache or folds overlap
    except Exception as e:
        problems.append(f"config/folds/caches: {e}")
        return finish(problems, warns)
    n = len(ctx.train_ids) + len(ctx.test_ids)
    ok(f"folds + struct/ESM caches: {len(ctx.train_ids)} train / {len(ctx.test_ids)} test proteins, folds {ctx.fold_names}")
    for d, name in ((ctx.geom_dir, "cache_geom"), (ctx.pocket_dir, "cache_pockets")):
        have = {p.stem for p in d.glob("*.npz")}
        lack = [p for p in ctx.train_ids + ctx.test_ids if p not in have]
        if lack:
            problems.append(f"{name}: {len(lack)}/{n} proteins missing (e.g. {lack[:3]})")
        else:
            ok(f"{name}: {n} proteins")

    gate = test_gate(ctx.exp_root)
    fz = gate["frozen"]
    if fz is None:
        problems.append("results/exp/_stats/frozen_selection.json missing")
        return finish(problems, warns)
    if fz["winner"] != args.expected_winner:
        problems.append(f"frozen winner is {fz['winner']}, expected {args.expected_winner}")
    elif set(fz.get("nested_winner_per_fold", {}).values()) != {args.expected_winner}:
        warns.append(f"nested winners differ across folds: {fz.get('nested_winner_per_fold')}")
    else:
        ok(f"frozen model = {fz['winner']} (all nested folds agree), criterion {fz['criterion']}, frozen {fz['frozen_at']}")
    if gate["hash_ok"]:
        ok("frozen prediction files match the SHA-256 recorded at freeze time")
    else:
        problems.append("frozen prediction files do NOT match the recorded SHA-256 (were pilot experiments re-run?)")

    spec = importlib.util.spec_from_file_location("rep", ROOT / "scripts" / "20_replicate_runs.py")
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)
    sel = ctx.exp_root / fz["winner"] / "selection.json"
    if not sel.exists():
        problems.append(f"{sel} missing (stack inputs unknown)")
    else:
        inputs = json.loads(sel.read_text()).get("inputs", [])
        unknown = [i for i in inputs if i not in rep.STEPS]
        lost = [i for i in inputs if not (ctx.exp_root / i / "oof_predictions.csv").exists()]
        if unknown or lost:
            problems.append(f"stack inputs not rebuildable {unknown} / missing in pilot {lost}")
        else:
            ok(f"frozen stack inputs ({len(inputs)}): {inputs}")
    for e in (BASELINE, "gnn_anchor"):
        if not (ctx.exp_root / e / "test_predictions.csv").exists():
            problems.append(f"pilot experiment {e} has no predictions")

    gsel = ctx.exp_root / "gbm_resid_grid_geom_ctx" / "selection.json"
    if gsel.exists():
        rounds = json.loads(gsel.read_text()).get("meta", {}).get("final_rounds")
        note = (" = the 10-tree floor of the early-stopping rule. Kept as frozen (describe it as such in Methods); "
                "the corrected stopping criterion is evaluated post hoc, OOF only, by script 14b") if rounds == 10 else ""
        info(f"frozen GBM component uses {rounds} trees{note}")
    if not (ROOT / "scripts" / "14b_gbm_residual_weighted_earlystop.py").exists():
        warns.append("scripts/14b_gbm_residual_weighted_earlystop.py missing: post-hoc Grid sensitivity phase will be skipped")

    if gate["unlocked"]:
        warns.append(f"test set ALREADY unlocked {gate['n_unlocks']} time(s) - fine if you are resuming after Phase E, "
                     "otherwise the run is no longer the first look")
    else:
        ok("frozen model not yet scored on the test set")
    reps = sorted(p.name for p in (Path(ctx.cfg["work"]["results"]) / "replicates").glob("rep_*"))
    info(f"existing replicate folders (resumable): {len(reps)}")

    if "created_after_test_unlock" not in (ROOT / "run_paper.sh").read_text():
        warns.append("run_paper.sh has no pre-test snapshot guard (old version?)")
    ext = list((ROOT / "external_predictions").glob("*.csv")) if (ROOT / "external_predictions").is_dir() else []
    info(f"external predictor CSVs for Table 7: {len(ext)}" + ("" if ext else "  (Table 7 will be skipped)"))

    free = shutil.disk_usage(Path(ctx.cfg["work"]["results"]).resolve()).free / 1e9
    (ok if free >= args.min_free_gb else warns.append)(f"free disk space under results/: {free:.0f} GB"
                                                       + ("" if free >= args.min_free_gb else f" (< {args.min_free_gb:.0f} GB)"))
    if torch.cuda.is_available():
        ok(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        warns.append("no CUDA device visible - replicates will run on CPU (very slow)")
    return finish(problems, warns)


def finish(problems, warns):
    for w in warns:
        print(f"[WARN] {w}")
    for p in problems:
        print(f"[FAIL] {p}")
    print("=" * 60)
    print("PREFLIGHT " + ("FAILED - fix the [FAIL] items before running" if problems else "PASSED"))
    print("=" * 60)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
