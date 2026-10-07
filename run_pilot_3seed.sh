#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

CFG="${CFG:-config.yaml}"

mkdir -p results/exp/_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/exp/_logs/pilot3_${STAMP}.log"

echo "========================================"
echo "GridFusion 3-seed OOF pilot"
echo "========================================"
echo "Started: $(date)"
echo "Config : $CFG"
echo "Seeds  : 0 1 2"
echo "Log    : $LOG"
echo

{
    echo "========================================"
    echo "MAIN EXPERIMENTS"
    echo "========================================"

    SEEDS="0 1 2" \
    GBM_SEEDS="0 1 2" \
    CFG="$CFG" \
    bash run_experiments.sh

    echo
    echo "========================================"
    echo "EARLY ENRICHMENT ANALYSIS"
    echo "EF@1 / EF@3 / BEDROC20/50/80"
    echo "========================================"

    python scripts/20_early_enrichment_report.py \
        --config "$CFG" \
        --split oof \
        --n-boot 5000

    echo
    echo "========================================"
    echo "PILOT FINISHED"
    echo "========================================"
    echo "Finished: $(date)"

} 2>&1 | tee "$LOG"

echo
echo "[DONE]"
echo "Log:"
echo "$LOG"
echo
echo "Expected outputs:"
echo "results/exp/_stats/oof_leaderboard.csv"
echo "results/exp/_stats/frozen_selection.json"
echo "results/exp/_stats/early_enrichment_oof_summary.csv"
echo "results/exp/_stats/early_enrichment_oof_paired_stats.csv"
