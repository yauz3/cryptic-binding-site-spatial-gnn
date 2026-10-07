#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CFG="${CFG:-config.yaml}"

python scripts/01_prepare_grid_residue_features.py --config "$CFG"
python scripts/06_residue_feature_ablation.py --config "$CFG"
python scripts/07_candidate_ranking_ablation.py --config "$CFG"

echo "[DONE] Ablations -> results/ablation/"
