#!/usr/bin/env bash
#
# collect_results.sh — gather every result artefact needed to write the paper into one
# folder and one zip, small enough to upload.
#
# Run it from the suite root:
#     bash collect_results.sh
#     MAXMB=5 bash collect_results.sh          # stricter per-file size cap (default 3 MB)
#     OUT=/tmp/paper_results bash collect_results.sh
#
# It only copies. Nothing in results/, additional_experiment/ or additional_experiment_2/ is
# modified. Prediction files, caches, checkpoints and replicate folders are skipped: they are
# large and regenerable, and nothing in the manuscript is written from them directly.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$SRC/paper_results}"
MAXMB="${MAXMB:-3}"
MAXB=$((MAXMB * 1024 * 1024))

echo "source: $SRC"
echo "output: $OUT  (per-file cap ${MAXMB} MB)"
rm -rf "$OUT"
mkdir -p "$OUT"

FOUND=(); MISSING=(); TOOBIG=()

take() {   # take <source dir> <destination subdir>
    local s="$1" d="$OUT/$2"
    if [[ ! -d "$s" ]]; then MISSING+=("$2"); return; fi
    local n=0
    while IFS= read -r -d '' f; do
        local rel="${f#"$s"/}"
        case "$rel" in
            *_predictions.csv|*.npz|*.pt|*.pth|parts/*|*/parts/*) continue ;;
        esac
        local sz; sz=$(stat -c%s "$f")
        if (( sz > MAXB )); then TOOBIG+=("$2/$rel ($((sz / 1024 / 1024)) MB)"); continue; fi
        mkdir -p "$d/$(dirname "$rel")"
        cp -a "$f" "$d/$rel"
        n=$((n + 1))
    done < <(find "$s" -type f -print0)
    (( n > 0 )) && FOUND+=("$2 ($n files)") || MISSING+=("$2 (empty)")
}

take_file() {  # take_file <source file> <destination path>
    [[ -f "$1" ]] && { mkdir -p "$OUT/$(dirname "$2")"; cp -a "$1" "$OUT/$2"; FOUND+=("$2"); } \
                  || MISSING+=("$2")
}

# ---------------------------------------------------------------- Phase I
take "$SRC/results/exp/_stats"                  "phase1/stats"
take "$SRC/results/paper"                       "phase1/paper"
take "$SRC/results/replicates/_analysis"        "phase1/replicates_analysis"
take "$SRC/results/comparative"                 "phase1/comparative"
for e in "$SRC"/results/exp/*/; do
    n="$(basename "$e")"; [[ "$n" == _* ]] && continue
    for f in selection.json meta.json stack_weights.csv oof_quick_stats.csv \
             config_search_oof.csv feature_importance.csv; do
        [[ -f "$e/$f" ]] && { mkdir -p "$OUT/phase1/experiments/$n"; cp -a "$e/$f" "$OUT/phase1/experiments/$n/"; }
    done
done
[[ -d "$OUT/phase1/experiments" ]] && FOUND+=("phase1/experiments ($(ls "$OUT/phase1/experiments" | wc -l) models)")

# ---------------------------------------------------------------- Phase II
P2="$SRC/additional_experiment/results"
take "$P2/phase2_cryptobench/report_oof"        "phase2/report_oof"
take "$P2/phase2_cryptobench/report_test"       "phase2/report_test"
take "$P2/phase2_cryptobench/exp/_stats"        "phase2/stats"
take_file "$P2/config_ft.yaml"                  "phase2/config_ft.yaml"
take_file "$P2/cache_esm_ft/model_name.txt"     "phase2/finetuned_encoder.txt"
take_file "$P2/cache_esm_ft/manifest.json"      "phase2/embedding_crossfit_manifest.json"
for e in "$P2"/phase2_cryptobench/exp/*/; do
    n="$(basename "$e")"; [[ "$n" == _* ]] && continue
    for f in selection.json meta.json oof_quick_stats.csv; do
        [[ -f "$e/$f" ]] && { mkdir -p "$OUT/phase2/experiments/$n"; cp -a "$e/$f" "$OUT/phase2/experiments/$n/"; }
    done
done

# ---------------------------------------------------------------- Phase III
P3="$SRC/additional_experiment_2/results"
for d in "$P3"/e1a_subset_*; do
    [[ -d "$d" ]] && take "$d" "phase3/E1A_$(basename "$d" | sed 's/^e1a_subset_//')"
done
take "$P3/e1b_matched_test"                     "phase3/E1B_matched_test"
take "$P3/e2_esm3b/report_oof"                  "phase3/E2_esm3b/report_oof"
take "$P3/e2_esm3b/report_test"                 "phase3/E2_esm3b/report_test"
take "$P3/e2_esm3b/exp/_stats"                  "phase3/E2_esm3b/stats"
take_file "$P3/config_esm3b.yaml"               "phase3/E2_esm3b/config_esm3b.yaml"
for e in "$P3"/e2_esm3b/exp/*/; do
    n="$(basename "$e")"; [[ "$n" == _* ]] && continue
    for f in selection.json meta.json oof_quick_stats.csv; do
        [[ -f "$e/$f" ]] && { mkdir -p "$OUT/phase3/E2_esm3b/experiments/$n"; cp -a "$e/$f" "$OUT/phase3/E2_esm3b/experiments/$n/"; }
    done
done
take "$P3/e3_pocket_test"                       "phase3/E3_pocket_test"
for d in "$P3"/e4_external_*; do
    [[ -d "$d" ]] && take "$d" "phase3/E4_$(basename "$d" | sed 's/^e4_external_//')"
done
for d in "$P3"/external/*/; do
    [[ -d "$d" ]] || continue
    n="$(basename "$d")"
    for f in report.json homology_report.csv; do
        [[ -f "$d/$f" ]] && { mkdir -p "$OUT/phase3/E4_$n"; cp -a "$d/$f" "$OUT/phase3/E4_$n/"; }
    done
done
for d in "$P3"/e5_predicted "$P3"/e5_predicted_inv; do
    [[ -d "$d" ]] || continue
    n="$(basename "$d")"
    take "$d/report_test" "phase3/E5_${n#e5_}/report_test"
    take_file "$d/mapping_report.csv" "phase3/E5_${n#e5_}/mapping_report.csv"
done

# ---------------------------------------------------------------- provenance
{
    echo "# paper_results"
    echo
    echo "Collected $(date -u '+%Y-%m-%d %H:%M UTC') from"
    echo "  $SRC"
    echo
    echo "Per-file cap: ${MAXMB} MB. Prediction CSVs, caches, checkpoints and replicate folders"
    echo "are not included; they are large and regenerable."
    echo
    echo "## Present"
    for x in "${FOUND[@]:-}"; do [[ -n "$x" ]] && echo "- $x"; done
    echo
    echo "## Not found (experiment not run yet, or path differs)"
    for x in "${MISSING[@]:-}"; do [[ -n "$x" ]] && echo "- $x"; done
    if ((${#TOOBIG[@]})); then
        echo
        echo "## Skipped for size"
        for x in "${TOOBIG[@]}"; do echo "- $x"; done
    fi
    echo
    echo "## Test-split unlock log"
    for p in results/exp/_stats/test_unlock_log.jsonl \
             additional_experiment/results/phase2_cryptobench/exp/_stats/test_unlock_log.jsonl \
             additional_experiment_2/results/e2_esm3b/exp/_stats/test_unlock_log.jsonl; do
        if [[ -f "$SRC/$p" ]]; then
            echo "- $p: $(grep -c . "$SRC/$p") unlock(s)"
        else
            echo "- $p: absent"
        fi
    done
} > "$OUT/MANIFEST.md"

(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt)

cd "$SRC"
rm -f paper_results.zip
zip -qr paper_results.zip "$(basename "$OUT")"

echo
echo "=== collected ==="
printf '  %s\n' "${FOUND[@]:-(nothing)}"
if ((${#MISSING[@]})); then
    echo
    echo "not found (fine if that experiment has not been run):"
    printf '  %s\n' "${MISSING[@]}"
fi
if ((${#TOOBIG[@]})); then
    echo
    echo "skipped for size (raise MAXMB if you want them):"
    printf '  %s\n' "${TOOBIG[@]}"
fi
echo
echo "folder: $OUT  ($(du -sh "$OUT" | cut -f1), $(find "$OUT" -type f | wc -l) files)"
echo "zip:    $SRC/paper_results.zip  ($(du -h "$SRC/paper_results.zip" | cut -f1))"
echo
echo "Upload paper_results.zip. If it is over ~30 MB, re-run with MAXMB=1."
