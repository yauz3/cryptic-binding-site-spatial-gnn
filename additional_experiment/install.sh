#!/usr/bin/env bash
# Install the Phase-II code into the Phase-I suite. Idempotent; backs up what it replaces.
#
#   bash install.sh            # SUITE defaults to the parent directory
#   SUITE=/path/to/suite bash install.sh
set -euo pipefail
PH2="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE="${SUITE:-$(cd "$PH2/.." && pwd)}"
[[ -f "$SUITE/config.yaml" && -d "$SUITE/cryptogrid" ]] || {
    echo "[FAIL] $SUITE does not look like the suite (no config.yaml / cryptogrid). Set SUITE=..."; exit 1; }
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$PH2/backup_$STAMP"

echo "suite:  $SUITE"
echo "backup: $BACKUP"
mkdir -p "$BACKUP"

# 1. new files
for f in cryptogrid/finetune.py scripts/23_finetune_plm.py scripts/24_build_external_benchmark.py \
         scripts/25_external_report.py scripts/26_matched_external_predictors.py \
         scripts/external/convert_predictions.py; do
    mkdir -p "$SUITE/$(dirname "$f")"
    [[ -f "$SUITE/$f" ]] && { mkdir -p "$BACKUP/$(dirname "$f")"; cp "$SUITE/$f" "$BACKUP/$f"; }
    cp "$PH2/$f" "$SUITE/$f"
    echo "[new]      $f"
done

# 2. three Phase-I files that gained a backward-compatible option
#    exp_common.py            make_ctx(require_esm=...)   - scripts that CREATE an embedding cache
#    10_build_geom_cache.py   no longer requires the embedding cache
#    19_stats_and_freeze.py   --baseline                  - freeze against the fine-tuned anchor
for f in cryptogrid/exp_common.py scripts/10_build_geom_cache.py scripts/19_stats_and_freeze.py; do
    if [[ -f "$SUITE/$f" ]] && cmp -s "$SUITE/$f" "$PH2/patch/$f"; then
        echo "[current]  $f"; continue
    fi
    mkdir -p "$BACKUP/$(dirname "$f")"
    [[ -f "$SUITE/$f" ]] && cp "$SUITE/$f" "$BACKUP/$f"
    cp "$PH2/patch/$f" "$SUITE/$f"
    echo "[updated]  $f   (previous version in $BACKUP/$f)"
done

python - "$SUITE" <<'PY'
import py_compile, sys
from pathlib import Path
root = Path(sys.argv[1])
for f in ["cryptogrid/finetune.py", "cryptogrid/exp_common.py", "scripts/10_build_geom_cache.py",
          "scripts/19_stats_and_freeze.py", "scripts/23_finetune_plm.py",
          "scripts/24_build_external_benchmark.py", "scripts/25_external_report.py",
          "scripts/26_matched_external_predictors.py", "scripts/external/convert_predictions.py"]:
    py_compile.compile(str(root / f), doraise=True)
print("[OK] all files compile")
PY
echo
echo "Done. Next: bash $PH2/run_phase2.sh check"
