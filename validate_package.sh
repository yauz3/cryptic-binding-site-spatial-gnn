#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[1/5] Python syntax"
python -m compileall -q cryptogrid scripts

echo "[2/5] Bash syntax"
for f in *.sh scripts/external/*.sh; do bash -n "$f"; done

echo "[3/5] Config syntax"
python - <<'PY2'
import yaml
from pathlib import Path
yaml.safe_load(Path('config.yaml').read_text())
print('config.yaml OK')
PY2

echo "[4/5] Expected local inputs"
python - <<'PY2'
from pathlib import Path
import yaml
cfg=yaml.safe_load(Path('config.yaml').read_text())
for k in ('folds_json','dataset_json','pdb_root','grid_features_dir'):
    p=Path(cfg['project'][k])
    print(f"{k:20s} {'OK' if p.exists() else 'MISSING'}  {p}")
PY2

echo "[5/5] Core imports"
python - <<'PY2'
mods=['numpy','pandas','sklearn','yaml','torch','transformers','networkx','scipy','matplotlib']
for m in mods:
    try:
        __import__(m); print(f"{m:15s} OK")
    except Exception as e:
        print(f"{m:15s} MISSING/ERROR: {e}")
PY2

echo "[DONE] Package validation completed."
