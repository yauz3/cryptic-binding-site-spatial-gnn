#!/usr/bin/env bash
# Phase III: the five experiments that decide whether this is a Q1 paper.
# READ Q1_PROTOCOL.md FIRST. It fixes the confirmatory family, the success criteria and the
# test-set accounting before any result is seen.
#
# Run from inside additional_experiment_2/. Everything produced lands in
# additional_experiment_2/results/ ; Phase-I and Phase-II trees are only read.
#
#   bash run_q1.sh check          read-only preflight
#   bash run_q1.sh e1a            E1 panel A: our metrics on published protein subsets
#   bash run_q1.sh e3             E3: pocket-level retrieval (DCC / RRO), all methods, one pipeline
#   bash run_q1.sh e1b            E1 panel B: matched head-to-head on identical residues
#   bash run_q1.sh e2-embed       E2: extract frozen ESM2-3B embeddings
#   bash run_q1.sh e2-train       E2: OOF models on the 3B backbone, H-E2 decision on OOF only
#   bash run_q1.sh e2-unlock      E2: freeze + the THIRD and final CryptoBench test unlock
#   bash run_q1.sh e5             E5: predicted (AlphaFold) structures
#   bash run_q1.sh e4             E4: homology-filtered external benchmark, scored once
#
# Environment:
#   SUITE      Phase-I suite root            (default: the parent directory)
#   CFG        Phase-I config                (default: $SUITE/config.yaml)
#   SEEDS      seeds for scripts 11/15       (default "0 1 2")
#   MODELS     our models to evaluate        (default "base_plm gnn_anchor stack_anchor")
#   ESM3B      backbone for E2               (default facebook/esm2_t36_3B_UR50D)
#   PRED_DIR   predicted structures for E5
#   EXT_*      external benchmark inputs for E4 (see phase2/README.md)
set -euo pipefail

Q1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE="${SUITE:-$(cd "$Q1/.." && pwd)}"
cd "$SUITE"
export PYTHONPATH="$SUITE:${PYTHONPATH:-}"

RESULTS="$Q1/results"
CFG="${CFG:-$SUITE/config.yaml}"
SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-base_plm gnn_anchor stack_anchor}"
ESM3B="${ESM3B:-facebook/esm2_t36_3B_UR50D}"
PRED_DIR="${PRED_DIR:-$Q1/predicted_structures}"
COMP_DIR="${COMP_DIR:-$Q1/comparator_predictions}"
E2_CFG="$RESULTS/config_esm3b.yaml"
STAGE="${1:-check}"
mkdir -p "$RESULTS"

cfg_get() { python - "$1" "$2" <<'PY'
import sys, yaml
from pathlib import Path
c = yaml.safe_load(Path(sys.argv[1]).read_text())
for k in sys.argv[2].split("."):
    c = c[k]
print(c)
PY
}

case "$STAGE" in

check)
    echo "=== PHASE III PREFLIGHT (read-only) ==="
    python scripts/00_preflight_paper.py --config "$CFG" || true
    echo
    for f in cryptogrid/pockets.py scripts/30_pocket_level_eval.py \
             scripts/31_predicted_structure_benchmark.py scripts/32_subset_report.py \
             scripts/external/convert_comparators.py; do
        [[ -f "$SUITE/$f" ]] && echo "[OK]   $f" || { echo "[FAIL] $f missing - run: bash $Q1/install.sh"; exit 1; }
    done
    python - "$Q1" <<'PY'
import sys, json
from pathlib import Path
q1 = Path(sys.argv[1])
for name, d in (("comparator predictions (E1-B, E3)", q1 / "comparator_predictions"),
                ("published subsets (E1-A)", q1 / "published_subsets"),
                ("predicted structures (E5)", q1 / "predicted_structures"),
                ("external benchmark (E4)", q1 / "external_data")):
    n = len(list(d.glob("*"))) - len(list(d.glob(".gitkeep"))) if d.is_dir() else 0
    print(f"[{'OK  ' if n else 'TODO'}] {name}: {n} file(s) in {d.name}/")
PY
    echo
    echo "Phase-I results:  $(cfg_get "$CFG" work.results)"
    echo "Phase-III output: $RESULTS"
    echo
    echo "Suggested order: e1a -> e3 -> e1b -> e2-embed -> e2-train -> (e2-unlock) -> e5 -> e4"
    ;;

e1a)
    shopt -s nullglob
    files=("$Q1"/published_subsets/*.txt "$Q1"/published_subsets/*.csv)
    ((${#files[@]})) || { echo "put one protein-id list per published method in $Q1/published_subsets/"; exit 1; }
    for f in "${files[@]}"; do
        name="$(basename "${f%.*}")"
        [[ "$name" == *_values ]] && continue        # that is a published-values file, not a subset list
        pub="$Q1/published_subsets/${name}_values.csv"
        echo; echo "=== E1-A  $name ==="
        python scripts/32_subset_report.py --config "$CFG" --split test --subset "$f" --name "$name" \
            --models $MODELS --out-dir "$RESULTS/e1a_subset_$name" \
            ${pub:+$([[ -f "$pub" ]] && echo --published "$pub")}
    done
    echo
    echo "Panel A is aggregate-vs-aggregate. No p-values. Panel B (e1b) carries any significance claim."
    ;;

e3)
    echo "=== E3  pocket-level retrieval, identical post-processing for every method ==="
    python scripts/30_pocket_level_eval.py --config "$CFG" --split test --models $MODELS \
        --reference base_plm --predictions-dir "$COMP_DIR" \
        --out-dir "$RESULTS/e3_pocket_test" --sweep
    echo
    echo "Read $RESULTS/e3_pocket_test/summary.md and sweep.csv (the ordering must survive the sweep)."
    ;;

e1b)
    shopt -s nullglob
    files=("$COMP_DIR"/*.csv)
    ((${#files[@]})) || { echo "no competitor CSVs in $COMP_DIR (see README.md for the format)"; exit 1; }
    echo "=== E1-B  matched head-to-head on identical residues ==="
    python scripts/26_matched_external_predictors.py --config "$CFG" --split test \
        --models $MODELS --predictions-dir "$COMP_DIR" --out-dir "$RESULTS/e1b_matched_test"
    echo
    echo "Confirmatory metrics: AP and BEDROC20, Holm within metrics and across competitors."
    ;;

e2-embed)
    echo "=== E2  frozen ESM2-3B embeddings ($ESM3B) ==="
    python - "$CFG" "$E2_CFG" "$RESULTS/cache_esm3b" "$RESULTS/e2_esm3b" <<'PY'
import sys, yaml
from pathlib import Path
src, dst, esm, res = sys.argv[1:5]
c = yaml.safe_load(Path(src).read_text())
w = dict(c["work"]); w.setdefault("frozen_esm_cache", w["esm_cache"])
w["esm_cache"] = str(Path(esm).resolve()); w["results"] = str(Path(res).resolve())
c["work"] = w
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(yaml.safe_dump(c))
print(f"[config] {dst}\n         esm_cache -> {w['esm_cache']}\n         results   -> {w['results']}")
PY
    python scripts/02_extract_esm_embeddings.py --config "$E2_CFG" --model-name "$ESM3B"
    echo "Embeddings are frozen: no fine-tuning, so no cross-fitting is required."
    ;;

e2-train)
    [[ -f "$E2_CFG" ]] || { echo "run 'bash run_q1.sh e2-embed' first"; exit 1; }
    echo "=== E2  OOF models on the ESM2-3B backbone ==="
    python scripts/11_baseline_multiseed.py   --config "$E2_CFG" --seeds $SEEDS
    python scripts/15_anchored_spatial_gnn.py --config "$E2_CFG" --seeds $SEEDS
    python scripts/12_spatial_smoothing.py    --config "$E2_CFG"
    python scripts/15_anchored_spatial_gnn.py --config "$E2_CFG" --seeds $SEEDS --no-graph --name ablation_gnn_no_graph
    python scripts/15_anchored_spatial_gnn.py --config "$E2_CFG" --seeds $SEEDS --no-geom  --name ablation_gnn_no_geom
    echo; echo "=== E2  decision on TRAINING-FOLD OOF ONLY ==="
    python scripts/25_external_report.py --config "$E2_CFG" --split oof --reference base_plm \
        --title "CryptoBench training folds, frozen ESM2-3B (OOF)" \
        --out-dir "$RESULTS/e2_esm3b/report_oof"
    echo
    echo "If the spatial gain is absent here, STOP: report E2 as an OOF-only negative result"
    echo "and do NOT run e2-unlock. The test split has already been used twice."
    ;;

e2-unlock)
    [[ -f "$E2_CFG" ]] || { echo "run e2-embed and e2-train first"; exit 1; }
    RES="$(cfg_get "$E2_CFG" work.results)"
    CAND=""
    for c in gnn_anchor smooth3d; do
        [[ -f "$RES/exp/$c/test_predictions.csv" ]] && CAND="$CAND $c"
    done
    [[ -n "$CAND" ]] || { echo "no E2 candidates found"; exit 1; }
    echo "=== E2  freeze on OOF, then the THIRD and FINAL CryptoBench test unlock ==="
    echo "[freeze] candidates:$CAND"
    python scripts/19_stats_and_freeze.py oof --config "$E2_CFG" --baseline base_plm --candidates $CAND
    echo
    read -r -p "Did the OOF gain hold, and is this freeze final? Type UNLOCK: " ans
    [[ "$ans" == "UNLOCK" ]] || { echo "aborted"; exit 1; }
    mkdir -p "$RES/exp/_stats"
    python scripts/19_stats_and_freeze.py test --unlock-test --config "$E2_CFG" \
        2>&1 | tee "$RES/exp/_stats/final_test_unlock.log"
    python scripts/25_external_report.py --config "$E2_CFG" --split test --reference base_plm \
        --title "CryptoBench test, frozen ESM2-3B" --out-dir "$RESULTS/e2_esm3b/report_test"
    echo
    echo "The CryptoBench test split has now been opened three times across the project."
    echo "Report all three unlocks."
    ;;

e5)
    [[ -d "$PRED_DIR" ]] || { echo "missing predicted structures: $PRED_DIR"; exit 1; }
    echo "=== E5  predicted structures, pLDDT as deposited ==="
    python scripts/31_predicted_structure_benchmark.py --config "$CFG" --structures "$PRED_DIR" \
        --split test --out-root "$RESULTS/e5_predicted"
    P="$RESULTS/e5_predicted/config_predicted.yaml"
    python scripts/11_baseline_multiseed.py   --config "$P" --seeds $SEEDS
    python scripts/15_anchored_spatial_gnn.py --config "$P" --seeds $SEEDS
    python scripts/25_external_report.py      --config "$P" --split test \
        --title "predicted structures (pLDDT as deposited)" \
        --out-dir "$RESULTS/e5_predicted/report_test"

    echo; echo "=== E5  sensitivity: 100 - pLDDT in the B-factor slot ==="
    python scripts/31_predicted_structure_benchmark.py --config "$CFG" --structures "$PRED_DIR" \
        --split test --out-root "$RESULTS/e5_predicted_inv" --invert-plddt
    P2="$RESULTS/e5_predicted_inv/config_predicted.yaml"
    python scripts/11_baseline_multiseed.py   --config "$P2" --seeds $SEEDS
    python scripts/15_anchored_spatial_gnn.py --config "$P2" --seeds $SEEDS
    python scripts/25_external_report.py      --config "$P2" --split test \
        --title "predicted structures (100 - pLDDT)" \
        --out-dir "$RESULTS/e5_predicted_inv/report_test"
    echo
    echo "E5 is a sensitivity analysis of the frozen model, not a new unlock. Say so in the paper."
    ;;

e4)
    echo "=== E4  homology-filtered external benchmark (scored once, ever) ==="
    EXT_NAME="${EXT_NAME:-external}"
    EXT_STRUCTURES="${EXT_STRUCTURES:-$Q1/external_data/${EXT_NAME}_structures}"
    EXT_LABELS="${EXT_LABELS:-$Q1/external_data/${EXT_NAME}_labels.json}"
    [[ -d "$EXT_STRUCTURES" && -f "$EXT_LABELS" ]] || {
        echo "missing $EXT_STRUCTURES or $EXT_LABELS"; exit 1; }
    for ID in 0.20 0.30; do
        TAG="${EXT_NAME}_id$(echo "$ID" | tr -d '.')"
        CFGX="$RESULTS/config_external_${TAG}.yaml"
        echo; echo "--- cohort $TAG (max identity to CryptoBench $ID) ---"
        python scripts/24_build_external_benchmark.py --config "$CFG" --name "$TAG" \
            --structures "$EXT_STRUCTURES" --labels "$EXT_LABELS" \
            --max-identity "$ID" --out-root "$RESULTS/external" --config-out "$CFGX"
        python scripts/02_extract_esm_embeddings.py --config "$CFGX"
        python scripts/10_build_geom_cache.py       --config "$CFGX"
        python scripts/11_baseline_multiseed.py     --config "$CFGX" --seeds $SEEDS
        python scripts/15_anchored_spatial_gnn.py   --config "$CFGX" --seeds $SEEDS
        python scripts/12_spatial_smoothing.py      --config "$CFGX"
        python scripts/25_external_report.py        --config "$CFGX" --split test \
            --title "external benchmark $TAG" --out-dir "$RESULTS/e4_external_$TAG/report_test"
        python scripts/30_pocket_level_eval.py      --config "$CFGX" --split test \
            --models $MODELS --reference base_plm --out-dir "$RESULTS/e4_external_$TAG/pocket"
    done
    echo
    echo "EXT20 is the primary cohort; EXT30 is the sensitivity analysis."
    ;;

*)
    echo "usage: bash run_q1.sh [check|e1a|e3|e1b|e2-embed|e2-train|e2-unlock|e5|e4]"; exit 1 ;;
esac
