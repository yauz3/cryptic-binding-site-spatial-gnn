#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

bash snapshot_code_state.sh

VERSION="$(cat VERSION 2>/dev/null || echo dev)"
STAMP="$(date +%Y%m%d_%H%M%S)"

PARENT="$(dirname "$ROOT")"
NAME="cryptobench_gridfusion_suite_${VERSION}_${STAMP}.zip"
LATEST="cryptobench_gridfusion_suite_latest.zip"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

DEST="$TMP/cryptobench_gridfusion_suite"

mkdir -p "$DEST"

rsync -a \
  --exclude '.git/' \
  --exclude 'work/' \
  --exclude 'external_data/' \
  --exclude 'external_repos/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.npz' \
  --exclude '*.pkl' \
  --exclude '*.log' \
  --exclude 'nohup.out' \
  --exclude 'results/replicates/' \
  "$ROOT/" \
  "$DEST/"

cd "$TMP"
zip -qr "$PARENT/$NAME" cryptobench_gridfusion_suite

cd "$PARENT"
cp -f "$NAME" "$LATEST"

echo
echo "[DONE]"
echo "$PARENT/$NAME"
echo "$PARENT/$LATEST"
echo
ls -lh "$NAME" "$LATEST"
