#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/external_repos"
mkdir -p "$DEST"

clone() {
  local url="$1"; local name="$2"; local branch="${3:-}"
  if [[ -d "$DEST/$name/.git" ]]; then
    echo "[exists] $name"
  else
    if [[ -n "$branch" ]]; then
      git clone --depth 1 --branch "$branch" "$url" "$DEST/$name"
    else
      git clone --depth 1 "$url" "$DEST/$name"
    fi
  fi
}

clone https://github.com/skrhakv/CryptoBench.git CryptoBench
clone https://github.com/skrhakv/cryptic-finetuning.git cryptic-finetuning
clone https://github.com/skrhakv/seq2pocket.git seq2pocket
clone https://github.com/bowman-lab/ae-pocketminer.git ae-pocketminer
clone https://github.com/mooreneural/lacuna.git lacuna
clone https://github.com/zetayue/ProMoNet.git ProMoNet
clone https://github.com/Mickdub/gvp.git pocketminer-gvp pocket_pred
clone https://github.com/rdk/p2rank.git p2rank

mkdir -p "$ROOT/external_data/cryptobench_osf"
echo "[INFO] Attempting official CryptoBench OSF download (project pz4a9)."
PYTHONPATH="$ROOT" python "$ROOT/scripts/download_osf_project.py" --project pz4a9 --out "$ROOT/external_data/cryptobench_osf" || \
  echo "[WARN] OSF download failed; existing local CryptoBench data can still be used."

cat <<'EOF'

External repositories cloned.

Notes:
- cryptic-finetuning and Seq2Pocket may require separately hosted model weights/data referenced by their README files.
- UniPocket is kept as a literature comparator unless an official public implementation is available.
- External repositories have different environments; do not install every requirements file into one environment.

EOF
