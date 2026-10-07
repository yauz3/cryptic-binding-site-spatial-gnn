#!/usr/bin/env bash
#
# organize_repo.sh — build a clean, GitHub-ready copy of this project.
#
# It COPIES files; it never edits, renames the contents of, or deletes anything in the
# source tree. Every Python/shell file lands byte-identical, at a path that keeps the
# pipeline runnable exactly as it was run, so a re-run reproduces the published numbers.
#
# Usage (from inside the suite directory):
#     bash organize_repo.sh                       # -> ../cryptobench-anchored-gnn
#     bash organize_repo.sh ../my-repo-name       # custom destination
#     RESULTS=full bash organize_repo.sh          # include the large prediction files too
#     RESULTS=none bash organize_repo.sh          # code and manuscript only
#
# RESULTS modes
#   summary (default) everything a reader needs to check the paper: statistics, reports,
#                     tables, figures, freeze records, unlock logs, protocol snapshots,
#                     stack weights and per-protein metrics. A few tens of MB.
#   full              adds every *_predictions.csv (~440 MB). Use Git LFS or a data
#                     repository for this; GitHub rejects single files over 100 MB.
#   none              no results at all.
#
# What is deliberately left out: superseded scripts (01..08 of the pre-registration era are
# kept because the caches were built with them; the *.before_cif_fix backup is not),
# the installer and patch copies from the Phase-II delivery zip, nested release zips,
# __pycache__, PID files and editor leftovers. The exclusion list is printed at the end.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:-$(dirname "$SRC")/cryptobench-anchored-gnn}"
RESULTS="${RESULTS:-summary}"
PHASE2_SRC=""
for cand in "$SRC/additional_experiment" "$SRC/phase2"; do
    [[ -d "$cand" ]] && PHASE2_SRC="$cand"
done

echo "source:      $SRC"
echo "destination: $DST"
echo "results:     $RESULTS"
[[ -n "$PHASE2_SRC" ]] && echo "phase II:    $PHASE2_SRC"
echo

[[ -e "$DST" ]] && { echo "[FAIL] $DST already exists; remove it or pass another path"; exit 1; }
mkdir -p "$DST"

MISSING=()
SKIPPED=()

copy() {  # copy <relative source> <relative destination>
    local s="$SRC/$1" d="$DST/$2"
    if [[ ! -e "$s" ]]; then MISSING+=("$1"); return; fi
    mkdir -p "$(dirname "$d")"
    cp -a "$s" "$d"
}

copy_tree() {  # copy_tree <relative source dir> <relative destination dir> [find-args...]
    local rel="$1" s="$SRC/$1" d="$DST/$2"; shift 2
    [[ -d "$s" ]] || { MISSING+=("$rel/"); return; }
    mkdir -p "$d"
    (cd "$s" && find . -type f "$@" -print0) | while IFS= read -r -d '' f; do
        mkdir -p "$d/$(dirname "$f")"
        cp -a "$s/$f" "$d/$f"
    done
}

# ---------------------------------------------------------------------------
# 1. library  — the modules the pipeline imports
# ---------------------------------------------------------------------------
for f in __init__.py common.py cif_support.py exp_common.py exp_models.py exp_stats.py finetune.py; do
    copy "cryptogrid/$f" "cryptogrid/$f"
done

# ---------------------------------------------------------------------------
# 2. scripts — numbered exactly as they were run; paths unchanged so the
#    `python scripts/NN_*.py --config config.yaml` calls still work
# ---------------------------------------------------------------------------
for f in \
    00_preflight_paper.py \
    01_prepare_grid_residue_features.py \
    02_extract_esm_embeddings.py \
    03_train_hybrid_models.py \
    04_fuse_external_predictions.py \
    05_comparative_analysis.py \
    06_residue_feature_ablation.py \
    07_candidate_ranking_ablation.py \
    08_residual_gated_fusion.py \
    10_build_geom_cache.py \
    11_baseline_multiseed.py \
    12_spatial_smoothing.py \
    13_pocket_consensus.py \
    14_gbm_residual_boost.py \
    14b_gbm_residual_weighted_earlystop.py \
    15_anchored_spatial_gnn.py \
    16_rank_loss.py \
    17_aux_distance_multitask.py \
    18_constrained_stacker.py \
    19_stats_and_freeze.py \
    20_early_enrichment_report.py \
    20_replicate_runs.py \
    21_replicate_analysis.py \
    22_paper_report.py \
    23_finetune_plm.py \
    24_build_external_benchmark.py \
    25_external_report.py \
    26_matched_external_predictors.py \
    download_osf_project.py
do
    copy "scripts/$f" "scripts/$f"
done
copy_tree "scripts/external" "scripts/external" ! -name "*.pyc"
copy_tree "scripts/dataset_analysis" "scripts/dataset_analysis" ! -name "*.pyc"

# ---------------------------------------------------------------------------
# 3. runners and configuration (root level, so their relative paths hold)
# ---------------------------------------------------------------------------
for f in run_experiments.sh run_paper.sh run_pilot_3seed.sh run_comparative.sh run_ablation.sh \
         run_dataset_analysis.sh run_external_models.sh run_everything.sh setup_final_workspace.sh \
         fetch_external_repos.sh validate_package.sh snapshot_code_state.sh \
         config.yaml requirements.txt; do
    copy "$f" "$f"
done
copy_tree "tests" "tests" ! -name "*.pyc"

# ---------------------------------------------------------------------------
# 4. Phase II — same folder-relative layout, so run_phase2.sh keeps working
#    (it resolves the suite as its own parent directory)
# ---------------------------------------------------------------------------
if [[ -n "$PHASE2_SRC" ]]; then
    P2REL="$(basename "$PHASE2_SRC")"
    mkdir -p "$DST/phase2"
    for f in run_phase2.sh PHASE2_PROTOCOL.md README.md; do
        [[ -f "$PHASE2_SRC/$f" ]] && cp -a "$PHASE2_SRC/$f" "$DST/phase2/$f" || MISSING+=("$P2REL/$f")
    done
    for d in external_data external_predictions; do
        mkdir -p "$DST/phase2/$d"
        [[ -f "$PHASE2_SRC/$d/.gitkeep" ]] && cp -a "$PHASE2_SRC/$d/.gitkeep" "$DST/phase2/$d/"
    done
    SKIPPED+=("$P2REL/install.sh and $P2REL/patch/ (delivery mechanism; the files are already installed)")
    SKIPPED+=("$P2REL/backup_*/ (installer backups)")
fi

# ---------------------------------------------------------------------------
# 5. documentation
# ---------------------------------------------------------------------------
for f in README.md README_PAPER.md SCIENTIFIC_GUARDRAILS.md EXPERIMENT_MATRIX.md MANIFEST.txt \
         CODE_STATE.txt VERSION LAST_UPDATED.txt literature_reference_metrics.csv; do
    copy "$f" "docs/$f"
done
[[ -f "$DST/docs/README.md" ]] && mv "$DST/docs/README.md" "$DST/docs/README_suite_original.md"

# ---------------------------------------------------------------------------
# 6. manuscript (if the LaTeX folder sits next to or inside the suite)
# ---------------------------------------------------------------------------
for cand in "$SRC/manuscript_latex" "$SRC/manuscript" "$(dirname "$SRC")/manuscript_latex"; do
    if [[ -d "$cand" ]]; then
        mkdir -p "$DST/manuscript"
        (cd "$cand" && find . -type f ! -name "*.aux" ! -name "*.log" ! -name "*.out" \
            ! -name "*.fls" ! -name "*.fdb_latexmk" ! -name "*.toc" ! -name "*.blg" -print0) |
        while IFS= read -r -d '' f; do
            mkdir -p "$DST/manuscript/$(dirname "$f")"
            cp -a "$cand/$f" "$DST/manuscript/$f"
        done
        echo "[manuscript] copied from $cand"
        break
    fi
done

# ---------------------------------------------------------------------------
# 7. results
# ---------------------------------------------------------------------------
if [[ "$RESULTS" != "none" ]]; then
    # Phase I: statistics, manuscript tables/figures, replicate analysis, freeze records
    copy_tree "results/exp/_stats" "results/phase1/stats"
    copy_tree "results/paper" "results/phase1/paper"
    copy_tree "results/replicates/_analysis" "results/phase1/replicates_analysis"
    for e in "$SRC"/results/exp/*/; do
        n="$(basename "$e")"
        [[ "$n" == _* ]] && continue
        for f in selection.json meta.json config_search_oof.csv oof_quick_stats.csv \
                 stack_weights.csv feature_importance.csv; do
            [[ -f "$e/$f" ]] && { mkdir -p "$DST/results/phase1/experiments/$n"; cp -a "$e/$f" "$DST/results/phase1/experiments/$n/$f"; }
        done
    done
    [[ -f "$SRC/results/comparative/literature_reference_metrics.csv" ]] &&
        copy "results/comparative/literature_reference_metrics.csv" "results/phase1/literature_reference_metrics.csv"

    # Phase II
    if [[ -n "$PHASE2_SRC" && -d "$PHASE2_SRC/results" ]]; then
        R2="$PHASE2_SRC/results"
        mkdir -p "$DST/results/phase2"
        for sub in phase2_cryptobench/report_oof phase2_cryptobench/report_test \
                   phase2_cryptobench/exp/_stats; do
            [[ -d "$R2/$sub" ]] || continue
            mkdir -p "$DST/results/phase2/$sub"
            cp -a "$R2/$sub/." "$DST/results/phase2/$sub/"
        done
        for e in "$R2"/phase2_cryptobench/exp/*/; do
            n="$(basename "$e")"; [[ "$n" == _* ]] && continue
            for f in selection.json meta.json config_search_oof.csv oof_quick_stats.csv; do
                [[ -f "$e/$f" ]] && { mkdir -p "$DST/results/phase2/experiments/$n"; cp -a "$e/$f" "$DST/results/phase2/experiments/$n/$f"; }
            done
        done
        [[ -f "$R2/config_ft.yaml" ]] && cp -a "$R2/config_ft.yaml" "$DST/results/phase2/config_ft.yaml"
        [[ -f "$R2/cache_esm_ft/model_name.txt" ]] && cp -a "$R2/cache_esm_ft/model_name.txt" "$DST/results/phase2/finetuned_encoder.txt"
        [[ -f "$R2/cache_esm_ft/manifest.json" ]] && cp -a "$R2/cache_esm_ft/manifest.json" "$DST/results/phase2/embedding_crossfit_manifest.json"
        for d in "$R2"/external/*/; do
            [[ -d "$d" ]] || continue
            n="$(basename "$d")"
            mkdir -p "$DST/results/phase2/external/$n"
            for f in report.json homology_report.csv; do
                [[ -f "$d/$f" ]] && cp -a "$d/$f" "$DST/results/phase2/external/$n/$f"
            done
            for sub in results/report_test results_ft/report_test; do
                [[ -d "$d/$sub" ]] && { mkdir -p "$DST/results/phase2/external/$n/$sub"; cp -a "$d/$sub/." "$DST/results/phase2/external/$n/$sub/"; }
            done
        done
        for d in "$R2"/matched_*; do
            [[ -d "$d" ]] && { mkdir -p "$DST/results/phase2/$(basename "$d")"; cp -a "$d/." "$DST/results/phase2/$(basename "$d")/"; }
        done
    fi

    if [[ "$RESULTS" == "full" ]]; then
        echo "[results] copying prediction files (large) ..."
        for e in "$SRC"/results/exp/*/; do
            n="$(basename "$e")"; [[ "$n" == _* ]] && continue
            for f in oof_predictions.csv test_predictions.csv; do
                [[ -f "$e/$f" ]] && { mkdir -p "$DST/results/phase1/experiments/$n"; cp -a "$e/$f" "$DST/results/phase1/experiments/$n/$f"; }
            done
        done
        if [[ -n "$PHASE2_SRC" ]]; then
            for e in "$PHASE2_SRC"/results/phase2_cryptobench/exp/*/; do
                n="$(basename "$e")"; [[ "$n" == _* ]] && continue
                for f in oof_predictions.csv test_predictions.csv; do
                    [[ -f "$e/$f" ]] && { mkdir -p "$DST/results/phase2/experiments/$n"; cp -a "$e/$f" "$DST/results/phase2/experiments/$n/$f"; }
                done
            done
        fi
    else
        SKIPPED+=("results/**/ *_predictions.csv  (~440 MB; re-run with RESULTS=full)")
    fi
    SKIPPED+=("results/models, results/predictions, results/*.zip, replicate rep_* folders (regenerable, large)")
fi

SKIPPED+=("scripts/01_prepare_grid_residue_features.py.before_cif_fix (superseded backup)")
SKIPPED+=("python_codes_updated.txt, *.pid, __pycache__, nested release zips")

# ---------------------------------------------------------------------------
# 8. generated repository files
# ---------------------------------------------------------------------------
cat > "$DST/.gitignore" <<'GI'
__pycache__/
*.pyc
work/
results/exp/
results/replicates/rep_*/
phase2/results/cache_esm_ft/
phase2/results/checkpoints/
external_data/
external_repos/
*.zip
*.pid
*.aux
*.log
*.out
*.fls
*.fdb_latexmk
GI

(cd "$DST" && find . -type f ! -name "MANIFEST_SHA256.txt" -print0 | sort -z |
    xargs -0 sha256sum > MANIFEST_SHA256.txt)

N_PY=$(find "$DST" -name "*.py" | wc -l)
N_SH=$(find "$DST" -name "*.sh" | wc -l)
N_ALL=$(find "$DST" -type f | wc -l)
SIZE=$(du -sh "$DST" | cut -f1)

cat > "$DST/README.md" <<README
# Anchored spatial GNN for cryptic binding-site residue prediction

Code and analysis artefacts for the CryptoBench study: a protein language model (pLM)
anchored in local three-dimensional structure, evaluated under a leakage-aware protocol in
which model selection happens only on the official training folds, one model is frozen
together with the SHA-256 digests of its predictions, and the held-out split is opened once
per phase.

This repository is a curated copy of the working tree. Every \`.py\` and \`.sh\` file is
byte-identical to the version that produced the reported numbers (see
\`MANIFEST_SHA256.txt\`); nothing was rewritten for presentation.

## Layout

| Path | Contents |
|---|---|
| \`cryptogrid/\` | library: caches, metrics, statistics, models, fine-tuning |
| \`scripts/\` | numbered pipeline steps, run as \`python scripts/NN_name.py --config config.yaml\` |
| \`run_*.sh\` | the exact orchestration used for each stage |
| \`phase2/\` | Phase-II runner and pre-registered protocol (kept one level down, which is what \`run_phase2.sh\` expects) |
| \`results/phase1/\` | Phase-I statistics, manuscript tables and figures, replicate analysis, freeze and unlock records |
| \`results/phase2/\` | Phase-II reports, freeze and unlock records, fine-tuning provenance |
| \`manuscript/\` | LaTeX sources, bibliography and figures |
| \`docs/\` | original suite documentation, guard rails, experiment matrix |

## Reproducing

Dependencies: \`pip install -r requirements.txt\` plus \`transformers\` for Phase II.
CryptoBench (structures, labels, official folds) is obtained from its authors; set the paths
in \`config.yaml\`.

\`\`\`bash
# caches
python scripts/01_prepare_grid_residue_features.py --config config.yaml
python scripts/02_extract_esm_embeddings.py        --config config.yaml
python scripts/10_build_geom_cache.py              --config config.yaml

# Phase I: screen candidates on the training folds, freeze one model
bash run_experiments.sh

# Phase I: 30 replicates, single test unlock, manuscript tables and figures
bash run_paper.sh

# Phase II: fine-tuned encoder, then the external benchmark
bash phase2/run_phase2.sh check
bash phase2/run_phase2.sh finetune
bash phase2/run_phase2.sh unlock
bash phase2/run_phase2.sh external
bash phase2/run_phase2.sh matched
\`\`\`

\`scripts/00_preflight_paper.py\` is a read-only check that the caches are complete, the
frozen predictions still match their recorded digests and the test split has not yet been
opened. Run it before any long job.

## Protocol records

* \`results/phase1/stats/frozen_selection.json\` — frozen model, selection criterion, per-fold
  nested winners, SHA-256 of the prediction files at freeze time
* \`results/phase1/stats/test_unlock_log.jsonl\` — append-only test-unlock log (one record)
* \`results/phase1/paper/PRETEST_FINAL_PROTOCOL.json\` — analysis plan written before the unlock
* \`results/phase2/phase2_cryptobench/exp/_stats/\` — the same records for Phase II
* \`phase2/PHASE2_PROTOCOL.md\` — Phase-II hypotheses and decision rules, fixed in advance

The CryptoBench test split is opened once per phase, twice in total. Both unlocks are logged
and are reported as such in the manuscript.

## Generated

Built by \`organize_repo.sh\` on $(date -u +%Y-%m-%d) — $N_ALL files ($N_PY Python, $N_SH shell), $SIZE.
Prediction files are excluded by default; rebuild with \`RESULTS=full bash organize_repo.sh\`.
README

{
    echo "# Files deliberately not copied"
    echo
    echo "Generated by organize_repo.sh. Everything below is either superseded, regenerable,"
    echo "or a delivery artefact rather than part of the pipeline."
    echo
    for s in "${SKIPPED[@]}"; do echo "- $s"; done
} > "$DST/docs/EXCLUDED_FROM_REPO.md"

echo
echo "=== built: $DST ==="
echo "$N_ALL files, $N_PY Python, $N_SH shell, $SIZE"
if ((${#MISSING[@]})); then
    echo
    echo "[WARN] expected but not found in the source tree:"
    printf '    %s\n' "${MISSING[@]}"
fi
echo
echo "Deliberately excluded (see docs/EXCLUDED_FROM_REPO.md):"
printf '    %s\n' "${SKIPPED[@]}"
echo
echo "Next:"
echo "    cd $DST && git init && git add -A && git commit -m 'Anchored spatial GNN for cryptic binding sites'"
