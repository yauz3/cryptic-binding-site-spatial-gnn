#!/usr/bin/env bash
# Leakage-aware experiment suite (scripts 10-19). Nothing here scores the 222 test proteins
# except the very last, commented-out command.
#
#   CFG=config.yaml SEEDS="0 1 2 3 4" bash run_experiments.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CFG="${CFG:-config.yaml}"
SEEDS="${SEEDS:-0 1 2 3 4}"
GBM_SEEDS="${GBM_SEEDS:-0 1 2}"
GBM_EXTRA="${GBM_EXTRA:-}"

# ---- 0. caches + anchor -------------------------------------------------------------------
python scripts/10_build_geom_cache.py   --config "$CFG"
python scripts/11_baseline_multiseed.py --config "$CFG" --seeds $SEEDS

# ---- 1. candidates (PRESPECIFIED list; do not add/remove after looking at results) ---------
python scripts/12_spatial_smoothing.py  --config "$CFG"
python scripts/13_pocket_consensus.py   --config "$CFG"
python scripts/14_gbm_residual_boost.py --config "$CFG" --features grid,geom,ctx --seeds $GBM_SEEDS $GBM_EXTRA
python scripts/15_anchored_spatial_gnn.py --config "$CFG" --seeds $SEEDS
python scripts/16_rank_loss.py --config "$CFG" --rank-loss toppair    --lam 0.5 --seeds $SEEDS
python scripts/16_rank_loss.py --config "$CFG" --rank-loss softbedroc --lam 1.0 --seeds $SEEDS
python scripts/17_aux_distance_multitask.py --config "$CFG" --seeds $SEEDS

CANDIDATES="smooth3d pocket_consensus gbm_resid_grid_geom_ctx gnn_anchor rank_toppair_lam0.5 rank_softbedroc_lam1 aux_dist_s5_g0.5"
python scripts/18_constrained_stacker.py --config "$CFG" --inputs $CANDIDATES

# ---- 2. ablations / controls (reported, never eligible for freezing) ----------------------
python scripts/12_spatial_smoothing.py  --config "$CFG" --mode sequence
python scripts/13_pocket_consensus.py   --config "$CFG" --shuffle-pockets
python scripts/14_gbm_residual_boost.py --config "$CFG" --features summary  --name ablation_gbm_support_only --seeds $GBM_SEEDS $GBM_EXTRA
python scripts/14_gbm_residual_boost.py --config "$CFG" --features geom,ctx --name ablation_gbm_no_grid      --seeds $GBM_SEEDS $GBM_EXTRA
python scripts/14_gbm_residual_boost.py --config "$CFG" --features grid     --name ablation_gbm_grid_only    --seeds $GBM_SEEDS $GBM_EXTRA
python scripts/15_anchored_spatial_gnn.py --config "$CFG" --seeds $SEEDS --no-graph   --name ablation_gnn_no_graph
python scripts/15_anchored_spatial_gnn.py --config "$CFG" --seeds $SEEDS --no-geom    --name ablation_gnn_no_geom
python scripts/15_anchored_spatial_gnn.py --config "$CFG" --seeds $SEEDS --no-bfactor --name ablation_gnn_no_bfactor

# ---- 3. statistics on TRAIN OOF + freeze one model ---------------------------------------
python scripts/19_stats_and_freeze.py oof --config "$CFG" --candidates $CANDIDATES stack_anchor

echo
echo "Inspect results/exp/_stats/oof_leaderboard.csv and frozen_selection.json."
echo "Only when the frozen model is final, run ONCE:"
echo "  python scripts/19_stats_and_freeze.py test --unlock-test --config $CFG"
