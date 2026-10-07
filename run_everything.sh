#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
bash run_comparative.sh
bash run_ablation.sh
bash run_dataset_analysis.sh
echo "[DONE] Core comparative + ablation + dataset analysis complete."
