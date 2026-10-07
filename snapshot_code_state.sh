#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OUT="CODE_STATE.txt"
HASH="CODE_SHA256SUMS.txt"

{
    echo "CryptoBench GridFusion code state"
    echo "Generated: $(date -Iseconds)"
    echo "Root: $ROOT"
    echo
    echo "VERSION:"
    cat VERSION 2>/dev/null || true
    echo
    echo "===== FILE LIST ====="

    find . \
      -type f \
      \( -name '*.py' \
         -o -name '*.sh' \
         -o -name '*.yaml' \
         -o -name '*.yml' \
         -o -name '*.md' \
         -o -name '*.txt' \) \
      ! -path './.git/*' \
      ! -path './work/*' \
      ! -path './external_data/*' \
      ! -path './external_repos/*' \
      ! -path './additional_experiment/results/*' \
      | sort
} > "$OUT"

find . \
  -type f \
  \( -name '*.py' \
     -o -name '*.sh' \
     -o -name '*.yaml' \
     -o -name '*.yml' \) \
  ! -path './.git/*' \
  ! -path './work/*' \
  ! -path './external_data/*' \
  ! -path './external_repos/*' \
  ! -path './additional_experiment/results/*' \
  -print0 \
| sort -z \
| xargs -0 sha256sum \
> "$HASH"

echo "[OK] $OUT"
echo "[OK] $HASH"
