#!/usr/bin/env bash
# Prepare this (freshly unzipped) suite folder for the final paper run and optionally start it.
#
#   bash setup_final_workspace.sh            # link caches, copy frozen pilot results, run preflight
#   bash setup_final_workspace.sh --start    # same, then start run_paper.sh with nohup if preflight passes
#
# What it does (safe to re-run; never deletes a real directory, never overwrites existing results):
#   1. activates the conda env (ENV_NAME, default rdkit_env) if it is not active
#   2. symlinks work/cache_{struct,esm,geom,pockets} from the old working suite (OLD)
#   3. copies OLD/results into ./results without overwriting anything already here
#      (frozen selection, pilot predictions, hashes stay byte-identical)
#   4. runs scripts/00_preflight_paper.py
#   5. with --start: launches run_paper.sh in the background and prints the PID
#
# Override the old suite location with:  OLD=/path/to/old_suite bash setup_final_workspace.sh
set -euo pipefail

NEW="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLD="${OLD:-/home/yavuz/yavuz_proje/cryptic_binding_site/cryptobench_gridfusion_suite}"
ENV_NAME="${ENV_NAME:-rdkit_env}"
LOG="${LOG:-paper_final_nohup.log}"
START=0
[[ "${1:-}" == "--start" ]] && START=1

cd "$NEW"
say() { echo; echo "=== $* ==="; }

# ---- 0. environment -----------------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi
echo "python: $(command -v python) | conda env: ${CONDA_DEFAULT_ENV:-none}"

# ---- 1. sanity ----------------------------------------------------------------------------
say "SOURCE (old working suite)"
if [[ ! -d "$OLD" ]]; then
    echo "[FAIL] OLD suite not found: $OLD   (set OLD=/path/to/old_suite)"; exit 1
fi
if [[ "$(cd "$OLD" && pwd -P)" == "$(pwd -P)" ]]; then
    echo "[INFO] OLD and NEW are the same folder - nothing to link or copy"
    SAME=1
else
    SAME=0
    echo "OLD = $OLD"
    echo "NEW = $NEW"
fi

# ---- 2. caches: symlink, never delete real data ----------------------------------------------
if [[ "$SAME" == "0" ]]; then
    say "LINK CACHES"
    mkdir -p work
    for d in cache_struct cache_esm cache_geom cache_pockets; do
        src="$OLD/work/$d"
        dst="work/$d"
        if [[ ! -d "$src" ]]; then
            echo "[FAIL] missing in old suite: $src"; exit 1
        fi
        if [[ -L "$dst" ]]; then
            ln -sfn "$src" "$dst"; echo "[RELINKED] $dst -> $src"
        elif [[ -d "$dst" ]] && [[ -n "$(ls -A "$dst" 2>/dev/null)" ]]; then
            echo "[KEPT]     $dst is a real, non-empty directory - left untouched"
        else
            [[ -d "$dst" ]] && rmdir "$dst"
            ln -s "$src" "$dst"; echo "[LINKED]   $dst -> $src"
        fi
    done

    # ---- 3. frozen pilot results: copy without overwriting ---------------------------------
    say "COPY FROZEN PILOT RESULTS (existing files here are never overwritten)"
    if [[ ! -f "$OLD/results/exp/_stats/frozen_selection.json" ]]; then
        echo "[FAIL] $OLD/results/exp/_stats/frozen_selection.json not found"; exit 1
    fi
    mkdir -p results
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --ignore-existing "$OLD/results/" "results/"
    else
        cp -an "$OLD/results/." "results/"
    fi
    echo "[OK] results/ ready ($(du -sh results | cut -f1))"
fi

say "COUNTS"
for d in cache_struct cache_esm cache_geom cache_pockets; do
    printf "%-14s %s npz\n" "$d" "$(find -L "work/$d" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)"
done
ls -l results/exp/_stats/frozen_selection.json

# ---- 4. preflight ---------------------------------------------------------------------------
say "PREFLIGHT"
export PYTHONPATH="$NEW:${PYTHONPATH:-}"
if ! python scripts/00_preflight_paper.py --config "${CFG:-config.yaml}"; then
    echo; echo "[STOP] preflight failed - nothing was started. Send the output above."; exit 1
fi

# ---- 5. start -------------------------------------------------------------------------------
if [[ "$START" == "0" ]]; then
    echo
    echo "Workspace is ready. Start the final run with:"
    echo "    bash setup_final_workspace.sh --start"
    exit 0
fi
if pgrep -fx "bash run_paper.sh" >/dev/null 2>&1; then
    echo "[STOP] run_paper.sh is already running (PID $(pgrep -fx 'bash run_paper.sh' | tr '\n' ' ')) - not starting a second one."
    exit 1
fi
[[ -f "$LOG" ]] && mv "$LOG" "${LOG%.log}_$(date +%Y%m%d_%H%M%S).log"
nohup bash run_paper.sh > "$LOG" 2>&1 &
PID=$!
echo "$PID" > paper_final.pid
say "STARTED"
echo "PID=$PID   (also in paper_final.pid)"
echo "follow:  tail -f $NEW/$LOG"
echo "resume after an interruption:  bash setup_final_workspace.sh --start"
