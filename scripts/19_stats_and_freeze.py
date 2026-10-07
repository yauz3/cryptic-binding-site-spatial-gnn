#!/usr/bin/env python3
"""
IDEA 10 - One statistical protocol for every experiment, and a single locked look at the test set.

Why this script exists
----------------------
* EF@1% on a ~340-residue protein is decided by ~4 residues, so most paired differences are
  exactly zero and a Wilcoxon test (which discards zeros) has almost no power. The primary test
  here is a paired sign-flip permutation test on the MEAN per-protein difference; Wilcoxon and a
  bootstrap CI are reported next to it.
* If ten methods are each tested on the 222 test proteins and the best p-value is reported, a
  "significant" result is expected by chance. Selection therefore happens on the 885 training
  proteins (OOF) and the test set is opened once, for one frozen model.

Mode `oof` (default) - uses ONLY train OOF predictions
    1. per-protein AP / AUC / EF@1 / EF@5 / BEDROC20/50/80 for every experiment
    2. paired stats vs the seed-ensembled pLM anchor; Holm inside the prespecified primary
       family {EF@1, BEDROC20}; Holm across candidates for each primary metric
    3. seed robustness (per-seed deltas; no seed is selected)
    4. NESTED_WINNER: the experiment is chosen on 3 folds and scored on the 4th -> unbiased
       estimate of "pick the best candidate by OOF J"
    5. freezes the candidate with the highest all-OOF J into frozen_selection.json
       (with SHA-256 of its prediction files)

Mode `test --unlock-test` - scores ONLY the frozen model against the pLM anchor on the 222
    test proteins, verifies the SHA-256 recorded at freeze time, and appends to
    test_unlock_log.jsonl. Every additional unlock is recorded; report the count in the paper.

Outputs (results/exp/_stats/)
    oof_per_protein.csv, oof_stats_long.csv, oof_leaderboard.csv, oof_seed_robustness.csv,
    frozen_selection.json, test_stats_frozen.csv, test_per_protein_frozen.csv, test_unlock_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from cryptogrid.common import ensure_dir
from cryptogrid.exp_common import (BASELINE, METRICS, PRIMARY, SELECTION_WEIGHTS, criterion, holm, j_vector,
                                   list_experiments, load_preds, make_ctx, paired_stats, per_protein_table,
                                   sha256_file, write_json)

SHOW = ["metric", "mean_base", "mean_new", "mean_delta", "ci95_low", "ci95_high", "n_better", "n_worse",
        "n_equal", "p_signflip", "p_signflip_holm_primary", "p_wilcoxon", "p_wilcoxon_holm_primary"]
FMT = lambda x: f"{x:.5f}"


def seed_robustness(base, new, name):
    seeds = sorted(set(c for c in base.columns if c.startswith("z_s")) & set(c for c in new.columns if c.startswith("z_s")))
    rows = []
    for c in seeds:
        b, n = per_protein_table(base, c, ["ap", "ef1", "bedroc20"]), per_protein_table(new, c, ["ap", "ef1", "bedroc20"])
        rows.append({"experiment": name, "seed": c[3:], **{f"delta_{m}": float((n[m] - b[m]).mean()) for m in ("ap", "ef1", "bedroc20")}})
    return rows


def run_oof(ctx, args, out):
    base_name = args.baseline
    base = load_preds(ctx, base_name, "oof")
    ppb = per_protein_table(base)
    if base_name not in list_experiments(ctx):
        raise SystemExit(f"--baseline {base_name} has no predictions")
    exps = [e for e in list_experiments(ctx) if e != base_name and not e.startswith("_")]
    if args.candidates:
        missing = set(args.candidates) - set(exps)
        if missing:
            raise SystemExit(f"unknown experiments: {sorted(missing)}")
    eligible = [e for e in exps if (args.candidates is None and "ablation" not in e) or (args.candidates and e in args.candidates)]
    eligible = [e for e in eligible if (ctx.exp_root / e / "test_predictions.csv").exists()]

    pps, long, per_prot, seeds, frames = {base_name: ppb}, [], [ppb.assign(experiment=base_name)], [], {base_name: base}
    for e in exps:
        df = load_preds(ctx, e, "oof")
        frames[e] = df
        pps[e] = per_protein_table(df)
        st = paired_stats(ppb, pps[e], n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed)
        st.insert(0, "experiment", e)
        long.append(st)
        per_prot.append(pps[e].assign(experiment=e))
        seeds += seed_robustness(base, df, e)

    # ---- nested winner: choose the experiment on 3 folds, score it on the 4th ------------------
    pfold = base.groupby("pdb_id", sort=False)["fold"].first().reindex(ppb.index).to_numpy()
    pool = [base_name] + eligible                        # baseline first = conservative tie-break
    Jm = np.stack([j_vector(pps[e]) for e in pool])
    nested = ppb.copy()
    chosen = {}
    for f in ctx.fold_names:
        best = pool[int(np.argmax(np.nanmean(Jm[:, pfold != f], axis=1)))]
        chosen[f] = best
        nested.loc[pfold == f, METRICS] = pps[best].loc[pfold == f, METRICS].to_numpy()
    st = paired_stats(ppb, nested, n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed)
    st.insert(0, "experiment", "NESTED_WINNER")
    long.append(st)

    long = pd.concat(long, ignore_index=True)
    long.to_csv(out / "oof_stats_long.csv", index=False)
    pd.concat(per_prot).reset_index().to_csv(out / "oof_per_protein.csv", index=False)
    pd.DataFrame(seeds).to_csv(out / "oof_seed_robustness.csv", index=False)

    # ---- leaderboard ---------------------------------------------------------------------------
    rows = []
    for e in exps + ["NESTED_WINNER"]:
        s = long[long["experiment"] == e].set_index("metric")
        r = {"experiment": e, "eligible": e in eligible or e == "NESTED_WINNER",
             "J": criterion(pps[e]) if e in pps else criterion(nested), "J_base": criterion(ppb)}
        for m in ("ef1", "bedroc20", "ap", "auc"):
            r[f"{m}_delta"] = s.loc[m, "mean_delta"]
            r[f"{m}_p_signflip"] = s.loc[m, "p_signflip"]
            r[f"{m}_p_wilcoxon"] = s.loc[m, "p_wilcoxon"]
        for m in PRIMARY:
            r[f"{m}_p_signflip_holm_primary"] = s.loc[m, "p_signflip_holm_primary"]
        rows.append(r)
    lb = pd.DataFrame(rows)
    lb["J_delta"] = lb["J"] - lb["J_base"]
    el = lb["experiment"].isin(eligible).to_numpy()
    for m in PRIMARY:                                   # multiplicity across the candidates that could be frozen
        adj = np.full(len(lb), np.nan)
        adj[el] = holm(lb.loc[el, f"{m}_p_signflip"].to_numpy())
        lb[f"{m}_p_signflip_holm_across_candidates"] = adj
    lb = lb.sort_values("J", ascending=False)
    lb.to_csv(out / "oof_leaderboard.csv", index=False)

    print(f"\n=== TRAIN-OOF leaderboard vs {base_name} ({len(ppb)} proteins; J = {SELECTION_WEIGHTS}) ===")
    print(lb[["experiment", "eligible", "J_delta", "ef1_delta", "ef1_p_signflip", "bedroc20_delta",
              "bedroc20_p_signflip", "ap_delta", "ap_p_signflip"]].to_string(index=False, float_format=FMT))
    print(f"\nNESTED_WINNER choices per fold: {chosen}")
    print(long[long["experiment"] == "NESTED_WINNER"][SHOW].to_string(index=False, float_format=FMT))

    # ---- freeze --------------------------------------------------------------------------------
    Jall = {e: criterion(pps[e]) for e in pool}
    winner = max(pool, key=lambda e: (Jall[e], -pool.index(e)))
    frozen_path = out / "frozen_selection.json"
    log_path = out / "test_unlock_log.jsonl"
    if frozen_path.exists():
        prev = json.loads(frozen_path.read_text())
        if prev["winner"] != winner and not args.refreeze:
            print(f"\n[FROZEN] existing selection = {prev['winner']} (new all-OOF argmax would be {winner}). "
                  "Kept unchanged; pass --refreeze to replace it.")
            return
        if log_path.exists() and prev["winner"] != winner:
            print("\n[WARNING] the test set was already unlocked for a previous frozen model. A new freeze after "
                  "that is NOT a confirmatory analysis any more - report it as exploratory.")
    files = {}
    for e in {base_name, winner}:
        for split in ("oof", "test"):
            p = ctx.exp_root / e / f"{split}_predictions.csv"
            files[f"{e}/{split}"] = sha256_file(p)
    write_json(frozen_path, {
        "winner": winner, "baseline": base_name, "J_all_oof": Jall, "eligible": eligible,
        "criterion": SELECTION_WEIGHTS, "primary_family": PRIMARY,
        "primary_test": "paired sign-flip permutation test on the mean per-protein difference, Holm over the primary family",
        "nested_winner_per_fold": chosen, "sha256": files, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Selected on the 885 training proteins (official-fold OOF) only.",
    })
    msg = "no candidate beats the anchor on OOF - there is nothing to confirm on test" if winner == base_name \
        else "run `19_stats_and_freeze.py test --unlock-test` ONCE for the confirmatory comparison"
    print(f"\n[FROZEN] winner = {winner} (J={Jall[winner]:.5f} vs anchor {Jall[base_name]:.5f}); {msg}")


def run_test(ctx, args, out):
    frozen_path = out / "frozen_selection.json"
    if not args.unlock_test:
        raise SystemExit("test mode needs --unlock-test (this is the single confirmatory look at the 222 proteins)")
    if not frozen_path.exists():
        raise SystemExit("no frozen_selection.json - run OOF mode first")
    fz = json.loads(frozen_path.read_text())
    winner = fz["winner"]
    base_name = fz.get("baseline", BASELINE)
    if winner == base_name:
        raise SystemExit("frozen winner is the anchor itself: nothing to test")
    for key, digest in fz["sha256"].items():
        e, split = key.split("/")
        if sha256_file(ctx.exp_root / e / f"{split}_predictions.csv") != digest:
            raise SystemExit(f"{key} predictions changed after freezing - refusing to score the test set")
    base, new = load_preds(ctx, base_name, "test"), load_preds(ctx, winner, "test")
    ppb, ppn = per_protein_table(base), per_protein_table(new)
    st = paired_stats(ppb, ppn, n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed)
    st.insert(0, "experiment", winner)
    st.to_csv(out / "test_stats_frozen.csv", index=False)
    pd.concat([ppb.assign(experiment=base_name), ppn.assign(experiment=winner)]).reset_index().to_csv(
        out / "test_per_protein_frozen.csv", index=False)
    log_path = out / "test_unlock_log.jsonl"
    n_prev = len(log_path.read_text().splitlines()) if log_path.exists() else 0
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "winner": winner,
                            "frozen_at": fz["frozen_at"], "unlock_number": n_prev + 1}) + "\n")
    print(f"\n=== TEST ({len(ppb)} proteins): frozen model `{winner}` vs {base_name} | unlock #{n_prev + 1} ===")
    print(st[SHOW].to_string(index=False, float_format=FMT))
    print("\nPrimary family:", PRIMARY, "-> read p_signflip_holm_primary.")
    if n_prev:
        print(f"[WARNING] the test set has now been unlocked {n_prev + 1} times; state this in the manuscript.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["oof", "test"], default="oof")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--candidates", nargs="*", default=None,
                    help="experiments allowed to be frozen (default: all non-ablation runs with test predictions)")
    ap.add_argument("--refreeze", action="store_true")
    ap.add_argument("--baseline", default=BASELINE,
                    help="anchor experiment (Phase II: base_plm_ft). The test mode reads it from the freeze file.")
    ap.add_argument("--unlock-test", action="store_true")
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    out = ensure_dir(ctx.exp_root / "_stats")
    (run_oof if args.mode == "oof" else run_test)(ctx, args, out)


if __name__ == "__main__":
    main()
