#!/usr/bin/env bash
# Install the Phase-III (Q1) code into the suite. Idempotent; backs up what it replaces.
#
#   bash install.sh                 # SUITE defaults to the parent directory
#   SUITE=/path/to/suite bash install.sh
set -euo pipefail
Q1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE="${SUITE:-$(cd "$Q1/.." && pwd)}"
[[ -f "$SUITE/config.yaml" && -d "$SUITE/cryptogrid" ]] || {
    echo "[FAIL] $SUITE does not look like the suite (no config.yaml / cryptogrid). Set SUITE=..."; exit 1; }
BACKUP="$Q1/backup_$(date +%Y%m%d_%H%M%S)"

echo "suite:  $SUITE"
echo "backup: $BACKUP"
mkdir -p "$BACKUP"

for f in cryptogrid/pockets.py \
         scripts/30_pocket_level_eval.py \
         scripts/31_predicted_structure_benchmark.py \
         scripts/32_subset_report.py \
         scripts/external/convert_comparators.py; do
    mkdir -p "$SUITE/$(dirname "$f")"
    if [[ -f "$SUITE/$f" ]]; then
        mkdir -p "$BACKUP/$(dirname "$f")"; cp "$SUITE/$f" "$BACKUP/$f"
    fi
    cp "$Q1/$f" "$SUITE/$f"
    echo "[new]      $f"
done

# Phase-II files are reused as they are. Fail loudly if an older copy is installed, because
# run_q1.sh passes --out-dir to both of them and a silent fallback would scatter the Q1 results
# into the Phase-I results tree.
missing=0
for f in scripts/25_external_report.py scripts/26_matched_external_predictors.py \
         scripts/24_build_external_benchmark.py scripts/19_stats_and_freeze.py; do
    if [[ ! -f "$SUITE/$f" ]]; then
        echo "[FAIL] $f missing - install the Phase-II package (additional_experiment) first"; missing=1
    fi
done
for f in scripts/25_external_report.py scripts/26_matched_external_predictors.py; do
    if ! grep -q '"--out-dir"' "$SUITE/$f" 2>/dev/null; then
        echo "[FAIL] $f has no --out-dir option; re-install the latest Phase-II package"; missing=1
    fi
done
if ! grep -q '"--baseline"' "$SUITE/scripts/19_stats_and_freeze.py" 2>/dev/null; then
    echo "[FAIL] scripts/19_stats_and_freeze.py has no --baseline option; re-install Phase II"; missing=1
fi
((missing == 0)) || exit 1
echo "[OK]       Phase-II scripts present and current"

python - "$SUITE" <<'PY'
import py_compile, sys
from pathlib import Path
root = Path(sys.argv[1])
for f in ["cryptogrid/pockets.py", "scripts/30_pocket_level_eval.py",
          "scripts/31_predicted_structure_benchmark.py", "scripts/32_subset_report.py",
          "scripts/external/convert_comparators.py"]:
    py_compile.compile(str(root / f), doraise=True)
print("[OK] all files compile")
PY

echo
echo "Done. Next: bash $Q1/run_q1.sh check"
