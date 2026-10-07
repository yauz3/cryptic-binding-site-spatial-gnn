#!/usr/bin/env python3
"""
STEP 20 - Replicate the FROZEN pipeline with independent seeds (default: 30 seeds, 100..129).

What a replicate is
-------------------
Replicate r re-runs the complete, already frozen pipeline with seed r for every stochastic
component, single-seed (no ensembling inside a replicate):

    base_plm(r) -> { gnn_anchor(r), rank_*(r), aux_dist(r), smooth3d, pocket_consensus, gbm(r) }
                -> stack_anchor   (inputs = exactly the inputs of the frozen stack)

Nothing is re-designed: architectures, hyper-parameters, candidate list, the nested-OOF
selection rule and the stack inputs are those of the frozen pilot. The data-driven choices
inside the pipeline (smoothing scale, pocket pooling, eta, stack lambda) are re-derived by the
same nested train-OOF rule inside each replicate - that is part of the algorithm being replicated.

Seeds 100..129 are disjoint from the pilot seeds that were used for model selection, so the
replicates are an honest replication of the selected method, not a re-use of the selection data.

Layout (caches are shared, results are separate; the pilot results are never touched)
    results/replicates/rep_<seed>/config.yaml
    results/replicates/rep_<seed>/exp/<experiment>/{oof,test}_predictions.csv
    results/replicates/rep_<seed>/logs/<experiment>.log
    results/replicates/manifest.json

Test predictions are written by the pipeline scripts but are NOT scored here.

Usage
    python scripts/20_replicate_runs.py                       # 30 replicates, full frozen stack
    python scripts/20_replicate_runs.py --components core     # only base_plm + gnn_anchor (fast)
    python scripts/20_replicate_runs.py --ablation-reps 10    # + GNN ablations/controls on 10 replicates
    python scripts/20_replicate_runs.py --shard 0/2           # split over 2 GPUs / terminals (0/2 and 1/2)
    python scripts/20_replicate_runs.py --dry-run
Resumable: finished steps (test_predictions.csv present) are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from cryptogrid.common import ensure_dir, load_config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# experiment name -> (script, extra args, uses per-replicate --seeds)
STEPS = {
    "base_plm": ("11_baseline_multiseed.py", [], True),
    "gnn_anchor": ("15_anchored_spatial_gnn.py", [], True),
    "rank_toppair_lam0.5": ("16_rank_loss.py", ["--rank-loss", "toppair", "--lam", "0.5"], True),
    "rank_softbedroc_lam1": ("16_rank_loss.py", ["--rank-loss", "softbedroc", "--lam", "1.0"], True),
    "aux_dist_s5_g0.5": ("17_aux_distance_multitask.py", [], True),
    "smooth3d": ("12_spatial_smoothing.py", [], False),
    "pocket_consensus": ("13_pocket_consensus.py", [], False),
    "gbm_resid_grid_geom_ctx": ("14_gbm_residual_boost.py", ["--features", "grid,geom,ctx"], True),
}
ABLATIONS = {
    "ablation_gnn_no_graph": ("15_anchored_spatial_gnn.py", ["--no-graph", "--name", "ablation_gnn_no_graph"], True),
    "ablation_gnn_no_geom": ("15_anchored_spatial_gnn.py", ["--no-geom", "--name", "ablation_gnn_no_geom"], True),
    "ablation_gnn_no_bfactor": ("15_anchored_spatial_gnn.py", ["--no-bfactor", "--name", "ablation_gnn_no_bfactor"], True),
    "smoothseq_ablation": ("12_spatial_smoothing.py", ["--mode", "sequence"], False),
    "pocket_consensus_shuffled_ablation": ("13_pocket_consensus.py", ["--shuffle-pockets"], False),
}
DEFAULT_STACK_INPUTS = ["smooth3d", "pocket_consensus", "gbm_resid_grid_geom_ctx", "gnn_anchor",
                        "rank_toppair_lam0.5", "rank_softbedroc_lam1", "aux_dist_s5_g0.5"]


def frozen_stack_inputs(pilot_exp: Path, stack_name: str):
    sel = pilot_exp / stack_name / "selection.json"
    if sel.exists():
        return list(json.loads(sel.read_text())["inputs"]), True
    return list(DEFAULT_STACK_INPUTS), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 130)))
    ap.add_argument("--components", choices=["full", "core"], default="full",
                    help="full = everything the frozen stack needs; core = base_plm + gnn_anchor only")
    ap.add_argument("--ablation-reps", type=int, default=0,
                    help="also run GNN ablations + cheap controls on the first N replicates")
    ap.add_argument("--stack-name", default="stack_anchor")
    ap.add_argument("--gbm-extra", default="", help="extra args for script 14 (must equal the pilot's, if any)")
    ap.add_argument("--shard", default="0/1", help="i/n: run replicates whose index %% n == i")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = Path(cfg["work"]["results"])
    pilot_exp = results / "exp"
    rep_root = ensure_dir(results / "replicates")
    inputs, from_frozen = frozen_stack_inputs(pilot_exp, args.stack_name)
    unknown = [i for i in inputs if i not in STEPS]
    if args.components == "full" and unknown:
        raise SystemExit(f"frozen stack uses inputs this driver cannot rebuild: {unknown}")
    if args.components == "full" and not from_frozen:
        print(f"[WARN] {pilot_exp / args.stack_name / 'selection.json'} not found; using the default input list")

    plan = ["base_plm", "gnn_anchor"]
    if args.components == "full":
        plan += [i for i in inputs if i not in plan]
    si, sn = (int(x) for x in args.shard.split("/"))
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    manifest_path = rep_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"replicates": {}}
    manifest.update({"stack_inputs": inputs, "stack_inputs_from_frozen_pilot": from_frozen,
                     "components": args.components, "stack_name": args.stack_name})

    def run(seed, name, script, extra, seeded, rep_dir, cfg_path):
        done = rep_dir / "exp" / name / "test_predictions.csv"
        if done.exists():
            return "cached", 0.0
        cmd = [sys.executable, str(HERE / script), "--config", str(cfg_path)] + list(extra)
        if seeded:
            cmd += ["--seeds", str(seed)]
        if script.startswith("14_") and args.gbm_extra:
            cmd += args.gbm_extra.split()
        if args.dry_run:
            print("   $", " ".join(cmd))
            return "dry", 0.0
        t0 = time.time()
        log = ensure_dir(rep_dir / "logs") / f"{name}.log"
        with log.open("w") as f:
            rc = subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not done.exists():
            tail = "\n".join(log.read_text().splitlines()[-25:])
            raise SystemExit(f"[rep {seed}] step {name} failed (see {log}):\n{tail}")
        return "ok", (time.time() - t0) / 60.0

    for k, seed in enumerate(args.seeds):
        if k % sn != si:
            continue
        rep_dir = ensure_dir(rep_root / f"rep_{seed}")
        rcfg = json.loads(json.dumps(cfg))
        rcfg["work"]["results"] = str(rep_dir)            # caches (work.*) stay shared
        cfg_path = rep_dir / "config.yaml"
        if not args.dry_run:
            cfg_path.write_text(yaml.safe_dump(rcfg), encoding="utf-8")
        print(f"\n=== replicate seed={seed} ({k + 1}/{len(args.seeds)}) -> {rep_dir}", flush=True)
        times = manifest["replicates"].setdefault(str(seed), {})
        for name in plan:
            status, minutes = run(seed, name, *STEPS[name], rep_dir, cfg_path)
            print(f"   {name:28s} {status} {minutes:6.1f} min", flush=True)
            if status == "ok":
                times[name] = round(minutes, 2)
        if args.components == "full":
            status, minutes = run(seed, args.stack_name, "18_constrained_stacker.py",
                                  ["--name", args.stack_name, "--inputs"] + inputs, False, rep_dir, cfg_path)
            print(f"   {args.stack_name:28s} {status} {minutes:6.1f} min", flush=True)
        if k < args.ablation_reps:
            for name, spec in ABLATIONS.items():
                status, minutes = run(seed, name, *spec, rep_dir, cfg_path)
                print(f"   {name:28s} {status} {minutes:6.1f} min", flush=True)
        if not args.dry_run:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n[DONE] replicates under {rep_root}. Next: python scripts/21_replicate_analysis.py --config {args.config}")


if __name__ == "__main__":
    main()
