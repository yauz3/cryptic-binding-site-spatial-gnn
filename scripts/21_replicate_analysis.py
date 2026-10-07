#!/usr/bin/env python3
"""
STEP 21 - Analysis of the seed replicates produced by script 20 (paper-ready tables + figures).

Three questions, three kinds of numbers (do not mix them in the text)
----------------------------------------------------------------------
A. Is the gain robust to training randomness?            -> unit = SEED
     mean +- SD over seeds, seeds improved k/S, paired t / Wilcoxon / sign test over seeds,
     seed-level Cohen's d_z and d_av.   (Describes stability. It is NOT evidence about new
     proteins: every seed sees the same proteins, so these p-values shrink without bound as S grows.)
B. Does the method generalise to unseen proteins?         -> unit = PROTEIN
     per-protein deltas averaged over seeds; sign-flip permutation p (Holm over {EF@1, BEDROC20}),
     Cohen's d_z with bootstrap CI, win/tie/loss.
C. One interval that carries both sources of uncertainty -> crossed bootstrap (proteins x seeds)
     plus the variance components (protein / seed / residual).

Also: the all-seed logit ENSEMBLE of each method against the all-seed ensemble of the anchor,
and GNN ablations paired by seed against the full GNN (if script 20 was run with --ablation-reps).

Splits
    oof  : always (885 training proteins, official folds)
    test : only with --include-test AND after the preregistered unlock of the frozen pilot model
           (scripts/19 ... test --unlock-test). Nothing is selected from these numbers.

Outputs: results/replicates/_analysis/
    replicate_macro_long.csv           one row per split x seed x method x metric
    replicate_summary.csv              mean, SD, min, max over seeds
    replicate_delta_vs_anchor.csv      A + B + C for every method vs base_plm
    replicate_ablation_vs_full_gnn.csv A + B + C for GNN ablations vs gnn_anchor
    replicate_ensemble_stats.csv       all-seed ensembles, protein-level paired stats
    table_replicates_<split>.csv/.tex  compact manuscript table
    fig_replicates_<split>.png/.pdf    per-seed macro metrics (anchor vs methods)
    fig_replicate_deltas_<split>.png/.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cryptogrid.common import ensure_dir, load_config
from cryptogrid.exp_common import (BASELINE, METRICS, PRIMARY, holm, list_experiments, load_preds, make_ctx,
                                   per_protein_table, signflip_pvalue)
from cryptogrid.exp_stats import (METRIC_LABELS, crossed_bootstrap_ci, effect_sizes, full_paired, label, new_fig,
                                  save_fig, seed_level_stats, test_gate, variance_components, write_table)

KEY = ["ap", "ef1", "bedroc20"]


def load_split(rep_dirs, split):
    """-> PP[method][seed] = per-protein table, ENS[method] = (sum of logits, n), template frame."""
    PP, ENS, template = {}, {}, None
    for seed, rd in rep_dirs:
        ctx = make_ctx(rd / "config.yaml")
        for m in list_experiments(ctx, need_test=(split == "test")):
            df = load_preds(ctx, m, split)
            if template is None:
                template = df[["pdb_id", "residue_key", "y_true", "fold"]].copy()
            PP.setdefault(m, {})[seed] = per_protein_table(df)
            s, n = ENS.get(m, (0.0, 0))
            ENS[m] = (s + df["z"].to_numpy(dtype=float), n + 1)
        print(f"  [{split}] loaded replicate seed={seed}", flush=True)
    return PP, ENS, template


def delta_rows(PP, ref, split, n_perm, n_boot, methods=None):
    rows = []
    for m in (methods or [k for k in PP if k != ref]):
        seeds = sorted(set(PP[m]) & set(PP[ref]))
        if len(seeds) < 2:
            continue
        per_metric = []
        for metric in METRICS:
            A = np.stack([PP[ref][s][metric].to_numpy(dtype=float) for s in seeds])     # (S, P)
            B = np.stack([PP[m][s][metric].reindex(PP[ref][s].index).to_numpy(dtype=float) for s in seeds])
            D = B - A
            r = {"split": split, "method": m, "reference": ref, "metric": metric,
                 "ref_mean": float(np.nanmean(A)), "method_mean": float(np.nanmean(B)),
                 "ref_sd_seeds": float(np.nanmean(A, axis=1).std(ddof=1)),
                 "method_sd_seeds": float(np.nanmean(B, axis=1).std(ddof=1))}
            r.update(seed_level_stats(np.nanmean(A, axis=1), np.nanmean(B, axis=1)))            # A: seeds
            dbar = np.nanmean(D, axis=0)                                                        # B: proteins
            es = effect_sizes(dbar, n_boot=n_boot, seed=1)
            r.update({"protein_n": int(np.isfinite(dbar).sum()), "protein_mean_delta": float(np.nanmean(dbar)),
                      "protein_cohen_dz": es["cohen_dz"], "protein_dz_ci95_low": es["dz_ci95_low"],
                      "protein_dz_ci95_high": es["dz_ci95_high"], "protein_hedges_gz": es["hedges_gz"],
                      "protein_wins": int((dbar > 0).sum()), "protein_losses": int((dbar < 0).sum()),
                      "protein_ties": int((dbar == 0).sum()),
                      "protein_p_signflip": signflip_pvalue(dbar, n_perm=n_perm, seed=2),
                      "protein_p_ttest": es["p_ttest"]})
            _, lo, hi = crossed_bootstrap_ci(D, n_boot=n_boot, seed=3)                           # C: both
            r.update({"crossed_ci95_low": lo, "crossed_ci95_high": hi})
            r.update(variance_components(D))
            per_metric.append(r)
        prim = [i for i, r in enumerate(per_metric) if r["metric"] in PRIMARY]
        adj = holm([per_metric[i]["protein_p_signflip"] for i in prim])
        for r in per_metric:
            r["protein_p_signflip_holm_primary"] = np.nan
        for i, a in zip(prim, adj):
            per_metric[i]["protein_p_signflip_holm_primary"] = float(a)
        rows += per_metric
    return rows


def figures(PP, split, out, show):
    show = [m for m in show if m in PP]
    seeds = sorted(set.intersection(*[set(PP[m]) for m in show]))
    fig, axes = new_fig(ncols=len(KEY), w=2.9, h=3.0)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes[0], KEY):
        vals = [[PP[m][s][metric].mean() for s in seeds] for m in show]
        ax.boxplot(vals, widths=0.5, showfliers=False)
        for i, v in enumerate(vals, 1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, alpha=0.7, zorder=3)
        ax.set_xticks(range(1, len(show) + 1))
        ax.set_xticklabels([label(m) for m in show], rotation=25, ha="right")
        ax.set_ylabel(f"macro {METRIC_LABELS[metric]}")
        ax.set_title(f"{len(seeds)} seeds, {split.upper()}")
    save_fig(fig, out / f"fig_replicates_{split}")

    others = sorted((m for m in PP if m != BASELINE), key=lambda m: -np.mean(
        [PP[m][s]["bedroc20"].mean() - PP[BASELINE][s]["bedroc20"].mean() for s in set(PP[m]) & set(PP[BASELINE])]))
    fig, axes = new_fig(ncols=len(KEY), w=3.4, h=0.33 * len(others) + 1.4)
    for ax, metric in zip(axes[0], KEY):
        for i, m in enumerate(others):
            ss = sorted(set(PP[m]) & set(PP[BASELINE]))
            d = [PP[m][s][metric].mean() - PP[BASELINE][s][metric].mean() for s in ss]
            ax.scatter(d, np.full(len(d), i) + rng.uniform(-0.15, 0.15, len(d)), s=8, alpha=0.7)
            ax.plot([np.mean(d)] * 2, [i - 0.3, i + 0.3], color="k", lw=1.2)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks(range(len(others)))
        ax.set_yticklabels([label(m) for m in others] if ax is axes[0][0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(f"$\\Delta$ macro {METRIC_LABELS[metric]} vs anchor (per seed)")
    save_fig(fig, out / f"fig_replicate_deltas_{split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--show", nargs="*", default=[BASELINE, "gnn_anchor", "stack_anchor"],
                    help="methods in the compact table / first figure")
    ap.add_argument("--n-perm", type=int, default=100000)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = Path(cfg["work"]["results"])
    rep_root = results / "replicates"
    rep_dirs = sorted(((int(p.name.split("_")[1]), p) for p in rep_root.glob("rep_*")
                       if (p / "exp" / BASELINE / "oof_predictions.csv").exists()), key=lambda t: t[0])
    if len(rep_dirs) < 2:
        raise SystemExit(f"need >= 2 finished replicates under {rep_root} (run scripts/20_replicate_runs.py)")
    out = ensure_dir(rep_root / "_analysis")
    splits = ["oof"]
    if args.include_test:
        gate = test_gate(results / "exp")
        if not gate["unlocked"]:
            raise SystemExit("--include-test refused: unlock the frozen pilot model first "
                             "(python scripts/19_stats_and_freeze.py test --unlock-test).")
        splits.append("test")
    print(f"[replicates] {len(rep_dirs)} seeds: {[s for s, _ in rep_dirs]} | splits: {splits}")

    long, summ, deltas, abl, ens_rows = [], [], [], [], []
    for split in splits:
        PP, ENS, template = load_split(rep_dirs, split)
        for m, by_seed in PP.items():
            for metric in METRICS:
                v = np.array([by_seed[s][metric].mean() for s in sorted(by_seed)])
                long += [{"split": split, "seed": s, "method": m, "metric": metric, "macro_value": by_seed[s][metric].mean()}
                         for s in sorted(by_seed)]
                summ.append({"split": split, "method": m, "metric": metric, "n_seeds": len(v), "mean": v.mean(),
                             "sd": v.std(ddof=1) if len(v) > 1 else np.nan, "min": v.min(), "max": v.max()})
        deltas += delta_rows(PP, BASELINE, split, args.n_perm, args.n_boot)
        if "gnn_anchor" in PP:
            abl += delta_rows(PP, "gnn_anchor", split, args.n_perm, args.n_boot,
                              methods=[m for m in PP if m.startswith("ablation_gnn")])
        # all-seed ensembles (logit mean) - protein-level paired comparison
        n_all = ENS[BASELINE][1]
        base_ens = template.assign(z=ENS[BASELINE][0] / n_all)
        ppb = per_protein_table(base_ens)
        for m, (zs, n) in ENS.items():
            if m == BASELINE or n != n_all:
                continue
            st = full_paired(ppb, per_protein_table(template.assign(z=zs / n)), n_perm=args.n_perm, n_boot=args.n_boot)
            st.insert(0, "method", m)
            st.insert(0, "split", split)
            st.insert(2, "ensemble_size", n)
            ens_rows.append(st)
        figures(PP, split, out, args.show)

    pd.DataFrame(long).to_csv(out / "replicate_macro_long.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "replicate_summary.csv", index=False)
    D = pd.DataFrame(deltas)
    D.to_csv(out / "replicate_delta_vs_anchor.csv", index=False)
    if abl:
        pd.DataFrame(abl).to_csv(out / "replicate_ablation_vs_full_gnn.csv", index=False)
    if ens_rows:
        pd.concat(ens_rows, ignore_index=True).to_csv(out / "replicate_ensemble_stats.csv", index=False)

    for split in splits:
        rows = []
        for m in [x for x in args.show if x != BASELINE]:
            for _, r in D[(D["split"] == split) & (D["method"] == m)].iterrows():
                rows.append({
                    "Method": label(m), "Metric": METRIC_LABELS[r["metric"]],
                    "Anchor (mean$\\pm$SD)": f"{r['ref_mean']:.4f}$\\pm${r['ref_sd_seeds']:.4f}",
                    "Method (mean$\\pm$SD)": f"{r['method_mean']:.4f}$\\pm${r['method_sd_seeds']:.4f}",
                    "Delta": r["seed_mean_delta"],
                    "95% CI (proteins x seeds)": f"[{r['crossed_ci95_low']:.4f}, {r['crossed_ci95_high']:.4f}]",
                    "Seeds improved": f"{int(r['seeds_improved'])}/{int(r['n_seeds'])}",
                    "d_z (proteins)": r["protein_cohen_dz"], "p (proteins)": r["protein_p_signflip"],
                    "d_z (seeds)": r["seed_cohen_dz"]})
        if rows:
            write_table(pd.DataFrame(rows), out / f"table_replicates_{split}", nd=4, p_cols=["p (proteins)"],
                        caption=f"Seed replicates ({len(rep_dirs)} independent seeds), {split.upper()} split. "
                                "SD is over seeds. The confidence interval resamples proteins and seeds jointly. "
                                "p: paired sign-flip permutation test over proteins on seed-averaged differences. "
                                "Seed-level $d_z$ quantifies training stability only.")
        sub = D[(D["split"] == split) & D["method"].isin(args.show) & D["metric"].isin(KEY)]
        print(f"\n=== {split.upper()} | {len(rep_dirs)} seeds ===")
        print(sub[["method", "metric", "ref_mean", "method_mean", "seed_mean_delta", "seed_sd_delta", "seeds_improved",
                   "crossed_ci95_low", "crossed_ci95_high", "protein_cohen_dz", "protein_p_signflip_holm_primary",
                   "seed_cohen_dz", "share_var_of_mean_from_seed"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[DONE] -> {out}")


if __name__ == "__main__":
    main()
