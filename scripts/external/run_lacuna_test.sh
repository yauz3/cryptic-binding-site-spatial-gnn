#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PROJECT_PDB:?Set PROJECT_PDB to the directory containing the CryptoBench protein structures}"
: "${FOLDS:?Set FOLDS to the CryptoBench folds.json file}"
OUT="$ROOT/external_runs/lacuna"; CONFORMERS="${CONFORMERS:-20}"; mkdir -p "$OUT"
command -v lacuna >/dev/null 2>&1 || { echo "Install Lacuna according to its current GitHub README first."; exit 1; }
while read -r id; do
  pdb="$PROJECT_PDB/$id.pdb"; [[ -f "$pdb" ]] || continue; mkdir -p "$OUT/$id"
  if [[ ! -f "$OUT/$id/pocket_report.json" ]]; then
    echo "[Lacuna] $id"
    ( cd "$OUT/$id" && lacuna discover "$pdb" --conformers "$CONFORMERS" )
  fi
done < <(PYTHONPATH="$ROOT" python "$ROOT/scripts/external/list_split_ids.py" "$FOLDS" --split test)
PYTHONPATH="$ROOT" python "$ROOT/scripts/external/convert_lacuna.py" --root "$OUT" --out "$ROOT/external_predictions/lacuna.csv"
echo "[DONE] Lacuna standardized predictions."
