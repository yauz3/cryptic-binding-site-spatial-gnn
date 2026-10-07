#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/scripts/dataset_analysis:${PYTHONPATH:-}"
CFG="${CFG:-config.yaml}"

python scripts/01_prepare_grid_residue_features.py --config "$CFG"
python scripts/dataset_analysis/pca_analysis.py --config "$CFG"
python scripts/dataset_analysis/anova_analysis.py --config "$CFG"
python scripts/dataset_analysis/mutual_information_analysis.py --config "$CFG"
python scripts/dataset_analysis/heatmap_analysis.py --config "$CFG"
python scripts/dataset_analysis/radviz_analysis.py --config "$CFG"
python scripts/dataset_analysis/networkx_analysis.py --config "$CFG"
python scripts/dataset_analysis/dendrogram_analysis.py --config "$CFG"
python scripts/dataset_analysis/tsne_analysis.py --config "$CFG"
python scripts/dataset_analysis/umap_analysis.py --config "$CFG"
if [[ "${RUN_SHAP:-1}" == "1" ]]; then
  python scripts/dataset_analysis/shap_analysis.py --config "$CFG"
fi

echo "[DONE] Dataset analysis -> results/dataset_analysis/"
