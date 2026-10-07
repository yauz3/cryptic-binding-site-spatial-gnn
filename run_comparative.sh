#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CFG="${CFG:-config.yaml}"

python scripts/01_prepare_grid_residue_features.py --config "$CFG"

if [[ ! -d work/cache_esm ]] || ! find work/cache_esm -name '*.npz' -print -quit | grep -q .; then
  if [[ -n "${ESM_MODEL:-}" ]]; then
    python scripts/02_extract_esm_embeddings.py --config "$CFG" --model-name "$ESM_MODEL"
  else
    python scripts/02_extract_esm_embeddings.py --config "$CFG"
  fi
else
  echo "[cache] ESM embeddings already exist. Use script 02 --overwrite to regenerate."
fi

python scripts/03_train_hybrid_models.py --config "$CFG"

for f in external_predictions/*.csv; do
  [[ -e "$f" ]] || continue
  echo "[external fusion] $f"
  python scripts/04_fuse_external_predictions.py --config "$CFG" --predictions "$f" || \
    echo "[WARN] External fusion failed for $f"
done

python scripts/05_comparative_analysis.py --config "$CFG"
echo "[DONE] Comparative analysis -> results/comparative/"
