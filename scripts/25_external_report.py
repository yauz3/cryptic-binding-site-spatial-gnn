#!/usr/bin/env python3
"""
PHASE II, STEP 3 - Evaluate models on an arbitrary split and report it to manuscript standard.

This is the reporting counterpart of scripts/22 for splits that are not the frozen
CryptoBench test set: the homology-filtered external benchmark built by scripts/24, and the
Phase-II (fine-tuned pLM) runs. It makes no selection decisions and has no freeze gate,
because by construction these splits are evaluation-only.

What it produces (under <results>/report_<split>/)
--------------------------------------------------
  table_absolute.csv/.tex        macro AP / ROC-AUC / EF@1,3,5 / BEDROC20,50,80 / MCC / F1
                                 with 95% bootstrap intervals over proteins, plus pooled
                                 ROC-AUC and AP
  table_paired_vs_<ref>.csv/.tex every model against the reference: delta, 95% CI,
                                 sign-flip permutation p, Holm within {EF@1%, BEDROC20},
                                 Wilcoxon p, Cohen's d_z with CI, rank-biserial, W/T/L
  per_protein.csv                per-protein metrics for every model
  fig_enrichment.png/.pdf        macro recall vs top-x% with bootstrap bands
  fig_paired_scatter.png/.pdf    per-protein scatter + delta histogram against the reference
  summary.md                     the numbers in a form that can be pasted into a draft

Usage
-----
  # external benchmark, frozen pLM pipeline
  python scripts/25_external_report.py --config config_external_pocketminer.yaml
  # Phase-II CryptoBench, fine-tuned anchor as the reference
  python scripts/25_external_report.py --config config_ft.yaml --split test \
      --reference base_plm_ft --models base_plm_ft base_plm gnn_anchor stack_anchor
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cryptogrid.common import ensure_dir
from cryptogrid.exp_common import (BASELINE, METRICS, PRIMARY, Store, list_experiments, load_preds,
                                   make_ctx, per_protein_table)
from cryptogrid.exp_stats import (METRIC_LABELS, crossfit_thresholds, enrichment_curve, fmt_p, full_paired,
                                  label, macro_ci, mcc_threshold, new_fig, pooled_compare, save_fig,
                                  threshold_tables, write_table)

FRACS = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20])
KEY = ["ap", "ef1", "bedroc20"]


def thresholds_for(ctx, name, split):
    """Always tuned on training-fold predictions; never on the split being reported."""
    oof = load_preds(ctx, name, "oof")
    if split == "oof":
        return crossfit_thresholds(oof, ctx.fold_names)
    return mcc_threshold(oof["y_true"].to_numpy().astype(int), oof["z"].to_numpy(dtype=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["test", "oof"])
    ap.add_argument("--models", nargs="*", default=None, help="default: every experiment with predictions")
    ap.add_argument("--reference", default=BASELINE)
    ap.add_argument("--title", default=None, help="label for the split, e.g. 'external (PocketMiner)'")
    ap.add_argument("--n-perm", type=int, default=100000)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-boot-pooled", type=int, default=300)
    ap.add_argument("--no-thresholded", action="store_true", help="skip MCC/F1 (needs OOF predictions)")
    ap.add_argument("--out-dir", default=None, help="default: <work.results>/report_<split>")
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    store = Store(ctx)
    have = list_experiments(ctx, need_test=(args.split == "test"))
    models = args.models or have
    missing = [m for m in models if m not in have]
    if missing:
        raise SystemExit(f"no {args.split} predictions for: {missing}")
    if args.reference not in models:
        models = [args.reference] + models
    out = ensure_dir(Path(args.out_dir) if args.out_dir
                     else Path(ctx.cfg["work"]["results"]) / f"report_{args.split}")
    title = args.title or f"{args.split} split"
    ids = ctx.test_ids if args.split == "test" else ctx.train_ids
    print(f"[report] {title}: {len(ids)} proteins | models: {models} | reference: {args.reference}")

    frames = {m: load_preds(ctx, m, args.split) for m in models}
    pps = {m: per_protein_table(frames[m]) for m in models}
    thr_pp, thr_pool = {}, {}
    if not args.no_thresholded:
        for m in models:
            try:
                thr_pp[m], thr_pool[m] = threshold_tables(frames[m], thresholds_for(ctx, m, args.split))
            except FileNotFoundError:
                print(f"[report] {m}: no OOF predictions, skipping MCC/F1")

    # ---- absolute table --------------------------------------------------------
    rows = []
    for m in models:
        r = {"Model": label(m), "n proteins": len(pps[m])}
        for k in METRICS:
            mean, lo, hi = macro_ci(pps[m][k].to_numpy(), args.n_boot)
            r[METRIC_LABELS[k]] = f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
        if m in thr_pp:
            for k in ("mcc", "f1"):
                mean, lo, hi = macro_ci(thr_pp[m][k].to_numpy(), args.n_boot)
                r[f"macro {METRIC_LABELS[k]}"] = f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
            r["pooled MCC"], r["pooled F1"] = thr_pool[m]["pooled_mcc"], thr_pool[m]["pooled_f1"]
        pc = pooled_compare(frames[args.reference], frames[m], n_boot=2, seed=0).set_index("metric")
        r["pooled ROC-AUC"], r["pooled AP"] = pc.loc["pooled_auc", "new"], pc.loc["pooled_ap", "new"]
        rows.append(r)
    write_table(pd.DataFrame(rows), out / "table_absolute",
                caption=f"Residue-level performance on the {title} ({len(ids)} proteins). Macro metrics are means "
                        "over proteins with 95\\% bootstrap intervals over proteins; pooled metrics concatenate all "
                        "residues. Thresholds for MCC and F1 were tuned on CryptoBench training-fold predictions only.")

    # ---- paired table ----------------------------------------------------------
    paired, pooled = [], []
    for m in models:
        if m == args.reference:
            continue
        st = full_paired(pps[args.reference], pps[m], n_perm=args.n_perm, n_boot=args.n_boot)
        if m in thr_pp and args.reference in thr_pp:
            st = pd.concat([st, full_paired(thr_pp[args.reference], thr_pp[m], metrics=["mcc", "f1"],
                                            n_perm=args.n_perm, n_boot=args.n_boot)], ignore_index=True)
        st.insert(0, "model", m)
        st["win/tie/loss"] = [f"{a}/{c}/{b}" for a, b, c in zip(st["n_better"], st["n_worse"], st["n_equal"])]
        paired.append(st)
        pc = pooled_compare(frames[args.reference], frames[m], n_boot=args.n_boot_pooled)
        pc.insert(0, "model", m)
        pooled.append(pc)
    if not paired:
        raise SystemExit("nothing to compare: only the reference model was given")
    paired = pd.concat(paired, ignore_index=True)
    paired.to_csv(out / f"table_paired_vs_{args.reference}_full.csv", index=False)
    pd.concat(pooled, ignore_index=True).to_csv(out / "table_pooled_delta.csv", index=False)
    slim = pd.DataFrame({
        "Model": [label(m) for m in paired["model"]], "Metric": [METRIC_LABELS[k] for k in paired["metric"]],
        "Reference": paired["mean_base"], "Model value": paired["mean_new"], "Delta": paired["mean_delta"],
        "95% CI": [f"[{a:.4f}, {b:.4f}]" for a, b in zip(paired["ci95_low"], paired["ci95_high"])],
        "p (perm.)": paired["p_signflip"], "p Holm": paired["p_signflip_holm_primary"],
        "p (Wilcoxon)": paired["p_wilcoxon"], "d_z": paired["cohen_dz"],
        "d_z 95% CI": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(paired["dz_ci95_low"], paired["dz_ci95_high"])],
        "r_rb": paired["rank_biserial"], "W/T/L": paired["win/tie/loss"]})
    write_table(slim, out / f"table_paired_vs_{args.reference}", p_cols=["p (perm.)", "p Holm", "p (Wilcoxon)"],
                caption=f"Paired per-protein comparison against {label(args.reference)} on the {title}. "
                        "p (perm.): two-sided sign-flip permutation test on the mean per-protein difference; Holm "
                        f"adjustment within the prespecified primary family {PRIMARY}; $d_z$: paired Cohen's d.")
    pd.concat([pps[m].assign(model=m) for m in models]).reset_index().to_csv(out / "per_protein.csv", index=False)

    # ---- figures ---------------------------------------------------------------
    rng = np.random.default_rng(0)
    fig, axes = new_fig(w=4.0, h=3.3)
    ax = axes[0][0]
    for m in models:
        E = enrichment_curve(frames[m], FRACS)
        b = E[rng.integers(0, len(E), size=(1000, len(E)))].mean(axis=1)
        lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
        ax.plot(FRACS * 100, E.mean(axis=0), marker="o", ms=3, label=label(m))
        ax.fill_between(FRACS * 100, lo, hi, alpha=0.18)
    ax.plot(FRACS * 100, FRACS, color="grey", ls="--", lw=0.8, label="random")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("top x % of residues per protein")
    ax.set_ylabel("recall of site residues (macro)")
    ax.set_title(f"Early enrichment, {title}")
    ax.legend(fontsize=7)
    save_fig(fig, out / "fig_enrichment")

    best = max((m for m in models if m != args.reference),
               key=lambda m: pps[m]["bedroc20"].mean() - pps[args.reference]["bedroc20"].mean())
    b, n = pps[args.reference], pps[best].reindex(pps[args.reference].index)
    fig, axes = new_fig(ncols=3, w=3.1, h=3.0)
    for axx, k in zip(axes[0][:2], ("ap", "bedroc20")):
        axx.scatter(b[k], n[k], s=6, alpha=0.5, color="k", linewidths=0)
        axx.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
        axx.set_xlabel(f"{METRIC_LABELS[k]} - {label(args.reference)}")
        axx.set_ylabel(f"{METRIC_LABELS[k]} - {label(best)}")
        axx.set_title(f"{int((n[k] > b[k]).sum())} better / {int((n[k] < b[k]).sum())} worse")
    d = (n["ap"] - b["ap"]).dropna()
    axes[0][2].hist(d, bins=min(40, max(8, len(d) // 4)), color="0.4")
    axes[0][2].axvline(0, color="grey", ls="--", lw=0.8)
    axes[0][2].axvline(d.mean(), color="k", lw=1.2)
    axes[0][2].set_xlabel("per-protein $\\Delta$AP")
    axes[0][2].set_ylabel("proteins")
    save_fig(fig, out / "fig_paired_scatter")

    # ---- summary ---------------------------------------------------------------
    lines = [f"# {title}: {len(ids)} proteins", "",
             f"- Reference: `{args.reference}`. Unit of analysis: protein.",
             f"- Primary test: paired sign-flip permutation ({args.n_perm} permutations), Holm within {PRIMARY}.",
             f"- Residues: {int(sum(len(store.labels(p)[0]) for p in ids))}; "
             f"positives: {int(sum(int(store.labels(p)[0].sum()) for p in ids))}.", ""]
    for m in models:
        if m == args.reference:
            continue
        s = paired[paired["model"] == m].set_index("metric")
        lines += [f"## {label(m)} vs {label(args.reference)}", "",
                  "| metric | ref | model | delta | 95% CI | p | p Holm | d_z | W/T/L |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for k in [x for x in METRICS if x in s.index]:
            r = s.loc[k]
            lines.append(f"| {k} | {r['mean_base']:.4f} | {r['mean_new']:.4f} | {r['mean_delta']:+.4f} | "
                         f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {fmt_p(r['p_signflip'])} | "
                         f"{fmt_p(r['p_signflip_holm_primary'])} | {r['cohen_dz']:.3f} | {r['win/tie/loss']} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    show = paired[paired["metric"].isin(KEY)]
    print(f"\n=== {title} ===")
    print(show[["model", "metric", "mean_base", "mean_new", "mean_delta", "ci95_low", "ci95_high",
                "p_signflip", "p_signflip_holm_primary", "cohen_dz", "win/tie/loss"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[DONE] -> {out}")


if __name__ == "__main__":
    main()
