#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

CFG="${CFG:-config.yaml}"

# Independent replication seeds.
# Pilot/model-selection seeds were 0,1,2.
REP_SEEDS="${REP_SEEDS:-$(seq 100 129)}"

# GNN ablations and cheap controls need not all be repeated 30 times.
ABLATION_REPS="${ABLATION_REPS:-10}"

# Must match the frozen pilot settings.
GBM_EXTRA="${GBM_EXTRA:-}"

mkdir -p \
  results/paper \
  results/replicates/_analysis \
  results/exp/_stats

export CFG REP_SEEDS

# Post-hoc, train-OOF-only sensitivity analysis of the Grid descriptors with a corrected GBM
# early-stopping criterion (scripts/14b). NOT part of the frozen pipeline; set POSTHOC_GBM=0 to skip.
POSTHOC_GBM="${POSTHOC_GBM:-1}"
POSTHOC_GBM_SEEDS="${POSTHOC_GBM_SEEDS:-0 1 2}"
POSTHOC_GBM_EXTRA="${POSTHOC_GBM_EXTRA:-}"

echo
echo "============================================================"
echo "PHASE 0 — READ-ONLY PREFLIGHT"
echo "============================================================"
echo

python scripts/00_preflight_paper.py --config "$CFG"

echo
echo "============================================================"
echo "PHASE A — FROZEN PIPELINE REPLICATION"
echo "30 seeds; NO model redesign"
echo "============================================================"
echo

python scripts/20_replicate_runs.py \
  --config "$CFG" \
  --seeds $REP_SEEDS \
  --ablation-reps "$ABLATION_REPS" \
  ${GBM_EXTRA:+--gbm-extra "$GBM_EXTRA"}

echo
echo "============================================================"
echo "PHASE B — OOF-ONLY REPLICATE ANALYSIS"
echo "============================================================"
echo

python scripts/21_replicate_analysis.py \
  --config "$CFG"

echo
echo "============================================================"
echo "PHASE B2 — POST-HOC (OOF-ONLY) GRID DESCRIPTOR SENSITIVITY"
echo "frozen model and frozen stack inputs are NOT touched"
echo "============================================================"
echo

if [[ "$POSTHOC_GBM" == "1" && -f scripts/14b_gbm_residual_weighted_earlystop.py ]]; then

    for FEATS in grid geom,ctx grid,geom,ctx; do
        NAME="posthoc_gbmw_${FEATS//,/_}"
        if [[ -f "results/exp/$NAME/test_predictions.csv" ]]; then
            echo "[INFO] $NAME already present; skipping"
            continue
        fi
        python scripts/14b_gbm_residual_weighted_earlystop.py \
          --config "$CFG" \
          --features "$FEATS" \
          --seeds $POSTHOC_GBM_SEEDS \
          $POSTHOC_GBM_EXTRA \
          || echo "[WARN] post-hoc GBM ($FEATS) failed; continuing - it is not part of the frozen pipeline"
    done

    python - <<'PY' || echo "[WARN] post-hoc Grid summary failed; continuing"
import json, os
from pathlib import Path
import pandas as pd
from cryptogrid.exp_common import load_preds, make_ctx, per_protein_table
from cryptogrid.exp_stats import full_paired

ctx = make_ctx(os.environ.get("CFG", "config.yaml"))
out = Path(ctx.cfg["work"]["results"]) / "paper"
out.mkdir(parents=True, exist_ok=True)
pairs = [
    ("posthoc_gbmw_geom_ctx", "posthoc_gbmw_grid_geom_ctx", "adding Grid descriptors to geometry+context"),
    ("base_plm", "posthoc_gbmw_grid", "Grid descriptors alone vs pLM anchor"),
    ("base_plm", "posthoc_gbmw_geom_ctx", "geometry+context (no Grid) vs pLM anchor"),
    ("gbm_resid_grid_geom_ctx", "posthoc_gbmw_grid_geom_ctx", "corrected early stopping vs frozen 10-tree GBM"),
]
rows = []
for ref, new, question in pairs:
    if not all((ctx.exp_root / e / "oof_predictions.csv").exists() for e in (ref, new)):
        continue
    st = full_paired(per_protein_table(load_preds(ctx, ref, "oof")), per_protein_table(load_preds(ctx, new, "oof")),
                     metrics=["ap", "ef1", "bedroc20", "auc"], n_perm=20000, n_boot=5000)
    sel = ctx.exp_root / new / "selection.json"
    trees = json.loads(sel.read_text()).get("meta", {}).get("final_rounds") if sel.exists() else None
    st.insert(0, "trees_in_final_model", trees)
    st.insert(0, "model", new)
    st.insert(0, "reference", ref)
    st.insert(0, "question", question)
    rows.append(st)
if rows:
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "posthoc_grid_descriptor_sensitivity_oof.csv", index=False)
    print(df[["question", "metric", "mean_base", "mean_new", "mean_delta", "ci95_low", "ci95_high",
              "p_signflip", "cohen_dz", "trees_in_final_model"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("[OK] wrote", out / "posthoc_grid_descriptor_sensitivity_oof.csv", "(post hoc, train OOF only, exploratory)")
PY

else
    echo "[INFO] post-hoc GBM phase skipped"
fi

echo
echo "============================================================"
echo "PHASE C — OOF PAPER TABLES / FIGURES"
echo "============================================================"
echo

python scripts/22_paper_report.py \
  --config "$CFG" \
  --loo \
  --no-test

# Separate EF@3 analysis without altering the already-frozen core
# experimental code.
if [[ -f scripts/20_early_enrichment_report.py ]]; then

    python scripts/20_early_enrichment_report.py \
      --config "$CFG" \
      --split oof \
      --n-boot 5000

    cp -f \
      results/exp/_stats/early_enrichment_oof_summary.csv \
      results/paper/ \
      2>/dev/null || true

    cp -f \
      results/exp/_stats/early_enrichment_oof_paired_stats.csv \
      results/paper/ \
      2>/dev/null || true
fi

echo
echo "============================================================"
echo "PHASE D — WRITE PRE-TEST PROTOCOL SNAPSHOT"
echo "============================================================"
echo

python - <<'PY'
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os

root = Path(".")
stats = root / "results" / "exp" / "_stats"
paper = root / "results" / "paper"

fz_path = stats / "frozen_selection.json"

if not fz_path.exists():
    raise SystemExit(
        "frozen_selection.json missing. "
        "Do not unlock test."
    )

fz = json.loads(
    fz_path.read_text()
)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)
    return h.hexdigest()

tracked = [
    "run_paper.sh",
    "config.yaml",
    "scripts/11_baseline_multiseed.py",
    "scripts/12_spatial_smoothing.py",
    "scripts/13_pocket_consensus.py",
    "scripts/14_gbm_residual_boost.py",
    "scripts/14b_gbm_residual_weighted_earlystop.py",
    "scripts/16_rank_loss.py",
    "scripts/17_aux_distance_multitask.py",
    "scripts/15_anchored_spatial_gnn.py",
    "scripts/18_constrained_stacker.py",
    "scripts/19_stats_and_freeze.py",
    "scripts/20_replicate_runs.py",
    "scripts/21_replicate_analysis.py",
    "scripts/22_paper_report.py",
    "cryptogrid/exp_common.py",
    "cryptogrid/exp_models.py",
    "cryptogrid/exp_stats.py",
]

protocol = {
    "created_before_final_test_unlock":
        datetime.now().isoformat(),

    "dataset":
        "CryptoBench",

    "core_architecture":
        "gnn_anchor",

    "frozen_performance_model":
        fz["winner"],

    "model_selection_criterion":
        fz["criterion"],

    "frozen_primary_family":
        fz["primary_family"],

    "secondary_metrics": [
        "AP",
        "ROC-AUC",
        "EF@3",
        "EF@5",
        "BEDROC50",
        "BEDROC80",
        "MCC",
        "F1",
    ],

    "replicate_seeds":
        [int(s) for s in os.environ.get("REP_SEEDS", "").split()],

    "post_hoc_analyses": [
        "scripts/14b: GBM residual with class-weighted early stopping "
        "(posthoc_gbmw_*), train-OOF only, exploratory; added after the "
        "freeze because every frozen GBM fit stopped at the 10-tree floor. "
        "Not an input of the frozen stack."
    ],

    "pilot_seeds":
        [0, 1, 2],

    "inference_unit":
        "protein",

    "primary_inference":
        (
            "paired protein-level sign-flip "
            "permutation test with Holm correction"
        ),

    "seed_statistics":
        (
            "training-stability analysis only; "
            "not treated as independent proteins"
        ),

    "scientific_claim":
        (
            "pLM-anchored spatial GNN is the "
            "mechanistic core; stack_anchor is the "
            "frozen performance ensemble; Grid is "
            "evaluated as a secondary/marginal "
            "structural contribution."
        ),

    "frozen_selection":
        fz,

    "code_sha256": {
        p: sha(p)
        for p in tracked
        if Path(p).exists()
    },
}

paper.mkdir(
    parents=True,
    exist_ok=True
)

out = (
    paper /
    "PRETEST_FINAL_PROTOCOL.json"
)

# The snapshot must really predate the test unlock: never rewrite it afterwards
# (e.g. when this script is re-run after an interruption).
if (stats / "test_unlock_log.jsonl").exists():
    if out.exists():
        print("[OK] test already unlocked; keeping existing", out)
        raise SystemExit(0)
    protocol["WARNING"] = (
        "written AFTER the test unlock; "
        "not a pre-test snapshot"
    )
    protocol["created_after_test_unlock"] = protocol.pop(
        "created_before_final_test_unlock"
    )

out.write_text(
    json.dumps(
        protocol,
        indent=2
    )
)

print(
    "[OK] wrote",
    out
)
PY

echo
echo "============================================================"
echo "PHASE E — ONE FINAL TEST UNLOCK"
echo "NO MODEL CHANGES AFTER THIS POINT"
echo "============================================================"
echo

if [[ ! -f results/exp/_stats/test_unlock_log.jsonl ]]; then

    python scripts/19_stats_and_freeze.py \
      test \
      --unlock-test \
      --config "$CFG" \
      2>&1 | tee \
      results/exp/_stats/final_test_unlock.log

else

    echo \
      "[INFO] Test unlock already recorded; "\
      "not unlocking again."

fi

echo
echo "============================================================"
echo "PHASE F — FINAL TEST + 30-SEED ANALYSIS"
echo "============================================================"
echo

python scripts/21_replicate_analysis.py \
  --config "$CFG" \
  --include-test

echo
echo "============================================================"
echo "PHASE G — FINAL PAPER TABLES / FIGURES"
echo "============================================================"
echo

python scripts/22_paper_report.py \
  --config "$CFG" \
  --loo

# EF@3 + extended early enrichment on the now-unlocked
# frozen final test.
if [[ -f scripts/20_early_enrichment_report.py ]]; then

    python scripts/20_early_enrichment_report.py \
      --config "$CFG" \
      --split test \
      --allow-test \
      --n-boot 5000

    cp -f \
      results/exp/_stats/early_enrichment_test_summary.csv \
      results/paper/ \
      2>/dev/null || true

    cp -f \
      results/exp/_stats/early_enrichment_test_paired_stats.csv \
      results/paper/ \
      2>/dev/null || true
fi

echo
echo "============================================================"
echo "FINAL PAPER PIPELINE COMPLETE"
echo "============================================================"
echo

echo "Main outputs:"
echo "  results/paper/"
echo "  results/replicates/_analysis/"
echo "  results/exp/_stats/final_test_unlock.log"
echo
echo "IMPORTANT:"
echo "  Do not redesign/tune the model from final test results."
