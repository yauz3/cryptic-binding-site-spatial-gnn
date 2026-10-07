#!/usr/bin/env bash
# Phase II: fine-tuned pLM + homology-filtered external benchmark.
# READ PHASE2_PROTOCOL.md BEFORE RUNNING ANYTHING.
#
# Run this from inside additional_experiment/. Everything it produces lands in
# additional_experiment/results/ ; the Phase-I suite (its results/, work/cache_esm, its
# freeze and its unlock log) is only ever read.
#
#   bash run_phase2.sh check         # read-only preflight
#   bash run_phase2.sh finetune      # II-A/II-B: fine-tune, re-run the pipeline, decide H1 on OOF
#   bash run_phase2.sh unlock        # II-C: freeze one Phase-II model, open CryptoBench test ONCE
#   bash run_phase2.sh external      # II-D: build + score the external benchmark (frozen pLM)
#   bash run_phase2.sh external-ft   # II-D: same external proteins, fine-tuned encoder
#   bash run_phase2.sh matched       # II-E: head-to-head with external predictors
#
# Environment (all optional):
#   SUITE       path to the Phase-I suite        (default: the parent directory)
#   CFG         Phase-I config                   (default: $SUITE/config.yaml)
#   FT_MODEL    encoder to fine-tune             (default facebook/esm2_t33_650M_UR50D)
#   FT_MODE     lora | top6 | full               (default lora)
#   FT_EPOCHS   fine-tuning epochs               (default 3)
#   FT_EXTRA    extra args for scripts/23        (e.g. "--grad-checkpointing --accum 4")
#   SEEDS       seeds for scripts 11/15          (default "0 1 2")
#   EXT_NAME / EXT_STRUCTURES / EXT_LABELS       external benchmark inputs
set -euo pipefail

PH2="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE="${SUITE:-$(cd "$PH2/.." && pwd)}"
cd "$SUITE"
export PYTHONPATH="$SUITE:${PYTHONPATH:-}"

RESULTS="$PH2/results"
CFG="${CFG:-$SUITE/config.yaml}"
FT_CFG="$RESULTS/config_ft.yaml"
FT_MODEL="${FT_MODEL:-facebook/esm2_t33_650M_UR50D}"
FT_MODE="${FT_MODE:-lora}"
FT_EPOCHS="${FT_EPOCHS:-3}"
FT_EXTRA="${FT_EXTRA:-}"
SEEDS="${SEEDS:-0 1 2}"
EXT_NAME="${EXT_NAME:-external}"
EXT_STRUCTURES="${EXT_STRUCTURES:-$PH2/external_data/${EXT_NAME}_structures}"
EXT_LABELS="${EXT_LABELS:-$PH2/external_data/${EXT_NAME}_labels.json}"
EXT_CFG="$RESULTS/config_external_${EXT_NAME}.yaml"
EXT_FT_CFG="$RESULTS/config_external_${EXT_NAME}_ft.yaml"
STAGE="${1:-check}"
mkdir -p "$RESULTS"

cfg_get() {   # cfg_get <config> <dotted.key>
    python - "$1" "$2" <<'PY'
import sys, yaml
from pathlib import Path
c = yaml.safe_load(Path(sys.argv[1]).read_text())
for k in sys.argv[2].split("."):
    c = c[k]
print(c)
PY
}

write_cfg() {  # write_cfg <src config> <dst config> <esm cache> <results dir>
    python - "$@" <<'PY'
import sys, yaml
from pathlib import Path
src, dst, esm, res = sys.argv[1:5]
c = yaml.safe_load(Path(src).read_text())
w = dict(c["work"])
w.setdefault("frozen_esm_cache", w["esm_cache"])     # provenance: the Phase-I cache
w["esm_cache"] = str(Path(esm).resolve())
w["results"] = str(Path(res).resolve())
c["work"] = w
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(yaml.safe_dump(c))
print(f"[config] {dst}\n         esm_cache -> {w['esm_cache']}\n         results   -> {w['results']}")
PY
}

case "$STAGE" in

check)
    echo "=== PHASE II PREFLIGHT (read-only) ==="
    python scripts/00_preflight_paper.py --config "$CFG" || true
    echo
    for f in cryptogrid/finetune.py scripts/23_finetune_plm.py scripts/24_build_external_benchmark.py \
             scripts/25_external_report.py scripts/26_matched_external_predictors.py \
             scripts/external/convert_predictions.py; do
        if [[ -f "$SUITE/$f" ]]; then echo "[OK]   $f"
        else echo "[FAIL] $f missing - run: bash $PH2/install.sh"; exit 1; fi
    done
    python - <<'PY'
import shutil
try:
    import torch, transformers
    print(f"[OK]   transformers {transformers.__version__}, torch {torch.__version__}, cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[WARN] no CUDA device: fine-tuning is not practical on CPU")
except Exception as e:
    raise SystemExit(f"[FAIL] {e}   (pip install transformers)")
print("[OK]   mmseqs found" if shutil.which("mmseqs")
      else "[WARN] mmseqs not found: homology filtering falls back to Biopython (slower, still correct)")
PY
    echo
    echo "Phase-I results are read from: $(cfg_get "$CFG" work.results)"
    echo "Phase-II outputs go to:        $RESULTS"
    ;;

finetune)
    write_cfg "$CFG" "$FT_CFG" "$RESULTS/cache_esm_ft" "$RESULTS/phase2_cryptobench"
    echo; echo "=== II-A  fine-tune $FT_MODEL (mode=$FT_MODE) with fold cross-fitting ==="
    python scripts/23_finetune_plm.py --config "$FT_CFG" --model-name "$FT_MODEL" \
        --mode "$FT_MODE" --epochs "$FT_EPOCHS" --save-checkpoints \
        --checkpoint-dir "$RESULTS/checkpoints" $FT_EXTRA

    echo; echo "=== II-A  re-run the Phase-I pipeline on the fine-tuned embeddings ==="
    python scripts/11_baseline_multiseed.py   --config "$FT_CFG" --seeds $SEEDS
    python scripts/15_anchored_spatial_gnn.py --config "$FT_CFG" --seeds $SEEDS
    python scripts/12_spatial_smoothing.py    --config "$FT_CFG"
    python scripts/15_anchored_spatial_gnn.py --config "$FT_CFG" --seeds $SEEDS --no-graph --name ablation_gnn_no_graph
    python scripts/15_anchored_spatial_gnn.py --config "$FT_CFG" --seeds $SEEDS --no-geom  --name ablation_gnn_no_geom

    echo; echo "=== II-B  H1 decision on TRAINING-FOLD OOF ONLY ==="
    python scripts/25_external_report.py --config "$FT_CFG" --split oof \
        --reference base_plm_ft --title "CryptoBench training folds, fine-tuned pLM (OOF)"
    echo
    echo "Read $RESULTS/phase2_cryptobench/report_oof/summary.md"
    echo "H1 holds only if the anchored GNN beats base_plm_ft on BOTH BEDROC20 and EF@1% there."
    echo "If it does not: STOP, report Phase II as a negative result, do NOT run 'unlock'."
    ;;

unlock)
    [[ -f "$FT_CFG" ]] || { echo "run 'bash run_phase2.sh finetune' first"; exit 1; }
    RES_FT="$(cfg_get "$FT_CFG" work.results)"
    CAND=""
    for c in gnn_anchor smooth3d stack_anchor; do
        [[ -f "$RES_FT/exp/$c/test_predictions.csv" ]] && CAND="$CAND $c"
    done
    [[ -n "$CAND" ]] || { echo "no Phase-II candidates found"; exit 1; }
    echo "=== II-C  freeze on OOF, then open the CryptoBench test split ONCE ==="
    echo "[freeze] candidates:$CAND"
    python scripts/19_stats_and_freeze.py oof --config "$FT_CFG" --baseline base_plm_ft --candidates $CAND
    echo
    read -r -p "Did H1 hold on OOF and is this freeze final? Type UNLOCK to open the test split: " ans
    [[ "$ans" == "UNLOCK" ]] || { echo "aborted"; exit 1; }
    mkdir -p "$RES_FT/exp/_stats"
    python scripts/19_stats_and_freeze.py test --unlock-test --config "$FT_CFG" \
        2>&1 | tee "$RES_FT/exp/_stats/final_test_unlock.log"
    python scripts/25_external_report.py --config "$FT_CFG" --split test \
        --reference base_plm_ft --title "CryptoBench test, fine-tuned pLM"
    echo
    echo "REMINDER: the CryptoBench test split has now been opened in BOTH phases."
    echo "Report both unlocks (PHASE2_PROTOCOL.md section 3)."
    ;;

external)
    [[ -d "$EXT_STRUCTURES" ]] || { echo "missing structures: $EXT_STRUCTURES"; exit 1; }
    [[ -f "$EXT_LABELS" ]]     || { echo "missing labels: $EXT_LABELS"; exit 1; }
    echo "=== II-D  build and score the homology-filtered external benchmark (frozen pLM) ==="
    python scripts/24_build_external_benchmark.py --config "$CFG" --name "$EXT_NAME" \
        --structures "$EXT_STRUCTURES" --labels "$EXT_LABELS" \
        --out-root "$RESULTS/external" --config-out "$EXT_CFG"
    python scripts/02_extract_esm_embeddings.py --config "$EXT_CFG"
    python scripts/10_build_geom_cache.py       --config "$EXT_CFG"
    python scripts/11_baseline_multiseed.py     --config "$EXT_CFG" --seeds $SEEDS
    python scripts/15_anchored_spatial_gnn.py   --config "$EXT_CFG" --seeds $SEEDS
    python scripts/12_spatial_smoothing.py      --config "$EXT_CFG"
    python scripts/25_external_report.py        --config "$EXT_CFG" --split test \
        --title "external benchmark ($EXT_NAME), frozen pLM"
    ;;

external-ft)
    [[ -f "$EXT_CFG" ]] || { echo "run 'bash run_phase2.sh external' first"; exit 1; }
    CK="$RESULTS/checkpoints/ft_full.pt"
    [[ -f "$CK" ]] || { echo "missing $CK (run 'finetune', which saves checkpoints)"; exit 1; }
    echo "=== II-D  external benchmark with the encoder fine-tuned on all 885 CryptoBench proteins ==="
    write_cfg "$EXT_CFG" "$EXT_FT_CFG" "$RESULTS/external/${EXT_NAME}/cache_esm_ft" \
              "$RESULTS/external/${EXT_NAME}/results_ft"
    python scripts/23_finetune_plm.py --config "$EXT_FT_CFG" --model-name "$FT_MODEL" \
        --mode "$FT_MODE" --embed-only --checkpoint "$CK" --name base_plm_ft $FT_EXTRA
    python scripts/11_baseline_multiseed.py   --config "$EXT_FT_CFG" --seeds $SEEDS
    python scripts/15_anchored_spatial_gnn.py --config "$EXT_FT_CFG" --seeds $SEEDS
    python scripts/25_external_report.py      --config "$EXT_FT_CFG" --split test \
        --reference base_plm_ft --title "external benchmark ($EXT_NAME), fine-tuned pLM"
    ;;

matched)
    ran=0
    if [[ -d "$PH2/external_predictions" ]] && compgen -G "$PH2/external_predictions/*.csv" >/dev/null; then
        echo "=== II-E  matched head-to-head on the CryptoBench test split ==="
        python scripts/26_matched_external_predictors.py --config "$CFG" --split test \
            --predictions-dir "$PH2/external_predictions" \
            --out-dir "$RESULTS/matched_cryptobench_test"
        ran=1
    else
        echo "[skip] no competitor CSVs in $PH2/external_predictions/"
    fi
    if [[ -f "$EXT_CFG" ]] && compgen -G "$PH2/external_predictions_${EXT_NAME}/*.csv" >/dev/null; then
        echo "=== II-E  matched head-to-head on the external benchmark ==="
        python scripts/26_matched_external_predictors.py --config "$EXT_CFG" --split test \
            --predictions-dir "$PH2/external_predictions_${EXT_NAME}" \
            --out-dir "$RESULTS/matched_external_${EXT_NAME}"
        ran=1
    fi
    [[ "$ran" == 1 ]] || { echo "nothing to compare; see README.md for the competitor CSV format"; exit 1; }
    ;;

*)
    echo "usage: bash run_phase2.sh [check|finetune|unlock|external|external-ft|matched]"; exit 1 ;;
esac
