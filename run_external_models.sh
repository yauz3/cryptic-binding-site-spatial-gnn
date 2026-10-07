#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

cat <<'EOF'
External methods use incompatible environments and are intentionally isolated.

Automated wrappers included:
  RUN_AE=1      -> AE-PocketMiner
  RUN_LACUNA=1  -> Lacuna

Repositories cloned for controlled/manual reproduction:
  CryptoBench, cryptic-finetuning, Seq2Pocket, ProMoNet, PocketMiner, P2Rank.
EOF

if [[ "${RUN_AE:-0}" == "1" ]]; then bash scripts/external/run_aepocketminer_test.sh; fi
if [[ "${RUN_LACUNA:-0}" == "1" ]]; then bash scripts/external/run_lacuna_test.sh; fi

echo "After standardized external predictions exist, run: bash run_comparative.sh"
