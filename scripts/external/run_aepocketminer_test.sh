#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$ROOT/external_repos/ae-pocketminer"
: "${PROJECT_PDB:?Set PROJECT_PDB to the directory containing the CryptoBench protein structures}"
: "${FOLDS:?Set FOLDS to the CryptoBench folds.json file}"
if [[ ! -d "$REPO" ]]; then echo "Missing $REPO; run fetch_external_repos.sh"; exit 1; fi
mkdir -p "$REPO/inputs"
while read -r id; do [[ -f "$PROJECT_PDB/$id.pdb" ]] && cp -n "$PROJECT_PDB/$id.pdb" "$REPO/inputs/"; done < <(PYTHONPATH="$ROOT" python "$ROOT/scripts/external/list_split_ids.py" "$FOLDS" --split test)
(
  cd "$REPO"
  if [[ -f src/example_config.yaml ]]; then
    python src/xtal_predict.py src/example_config.yaml
  else
    echo "[ERROR] AE-PocketMiner CLI/layout changed. Inspect repository README."; exit 2
  fi
)
PYTHONPATH="$ROOT" python "$ROOT/scripts/external/convert_aepocketminer.py" --results "$REPO/results" --pdb-root "$PROJECT_PDB" --out "$ROOT/external_predictions/aepocketminer.csv"
echo "[DONE] AE-PocketMiner standardized predictions."
