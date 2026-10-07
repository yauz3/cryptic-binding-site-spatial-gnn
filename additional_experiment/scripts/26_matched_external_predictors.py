#!/usr/bin/env python3
"""
PHASE II, STEP 4 - Matched head-to-head comparison with external predictors.

Why
---
Phase I compared the method only against its own pLM anchor, and placed published numbers
in a table explicitly labelled as context rather than a benchmark, because those numbers
come from different protein subsets and different filtering. A reviewer at a strong journal
will ask for a like-for-like comparison. This script provides it: every method is scored on
EXACTLY the same residues of exactly the same proteins, and any protein a competitor cannot
handle is either excluded for everyone or reported separately, never silently dropped.

This matters concretely for CryptoBench: in the benchmark paper PocketMiner failed on 22
test structures and cannot predict multi-chain structures, which removed 60 of 222 proteins,
more than a quarter of the test set. Comparing a number computed on 222 proteins with one
computed on 162 is not a comparison.

Input format
------------
One CSV per competitor in --predictions-dir, with columns
    pdb_id, residue_key, score           (higher = more likely to be a site residue)
and optionally a `method` column giving the display name (otherwise the file stem is used).
`residue_key` must be <auth_asym_id>_<auth_seq_id><insertion code>, i.e. the same namespace
as work/cache_struct. Residues that are missing from a file are treated as not predicted.
Helper converters for PocketMiner (.npy per structure) and P2Rank (residue CSV) are in
scripts/external/.

What it reports
---------------
For every competitor, and for every one of our models:
  * the matched protein set (proteins where the competitor covers at least
    --min-coverage of the residues AND has at least one positive and one negative)
  * macro AP / ROC-AUC / EF@1,3,5 / BEDROC20 computed on the intersection of residues
  * paired per-protein differences of each of our models against the competitor, with
    sign-flip permutation p, Holm within the primary family, Cohen's d_z and W/T/L
  * pooled ROC-AUC / AUPRC / MCC / F1 for comparability with the CryptoBench paper, with
    the competitor's own threshold when it supplies a binary column
  * a coverage table, so the cost of each exclusion is visible

Usage
-----
  python scripts/26_matched_external_predictors.py --config config.yaml --split test \
      --predictions-dir external_predictions --models base_plm gnn_anchor stack_anchor
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cryptogrid.common import ensure_dir
from cryptogrid.exp_common import (BASELINE, PRIMARY, canonical_frame, holm, list_experiments, load_preds,
                                   make_ctx, per_protein_table, protein_slices, rank_metrics, signflip_pvalue)
from cryptogrid.exp_stats import METRIC_LABELS, effect_sizes, fmt_p, label, macro_ci, write_table

METRICS = ["ap", "auc", "ef1", "ef3", "ef5", "bedroc20"]


def rank_metrics_ext(y, s):
    m = rank_metrics(y, s, ["ap", "auc", "ef1", "ef5", "bedroc20"])
    n, npos = len(y), int(np.sum(y))
    if n and 0 < npos < n:
        order = np.argsort(-np.asarray(s, dtype=float), kind="stable")
        cum = np.cumsum(np.asarray(y)[order])
        k = max(1, int(np.ceil(0.03 * n)))
        m["ef3"] = float((cum[k - 1] / k) / (npos / n))
    else:
        m["ef3"] = float("nan")
    return m


def load_competitor(path: Path, ref: pd.DataFrame):
    df = pd.read_csv(path, dtype={"pdb_id": str, "residue_key": str})
    need = {"pdb_id", "residue_key", "score"}
    if not need <= set(df.columns):
        print(f"[skip] {path.name}: needs columns {sorted(need)}")
        return None, None
    name = str(df["method"].iloc[0]) if "method" in df.columns and len(df) else path.stem
    df["pdb_id"] = df["pdb_id"].str.lower().str.strip()
    df["residue_key"] = df["residue_key"].str.strip()
    cols = ["pdb_id", "residue_key", "score"] + (["prediction"] if "prediction" in df.columns else [])
    merged = ref.merge(df[cols].drop_duplicates(["pdb_id", "residue_key"]),
                       on=["pdb_id", "residue_key"], how="left")
    return name, merged


def pooled_binary(y, pred):
    tp = float(((pred == 1) & (y == 1)).sum()); fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum()); tn = float(((pred == 0) & (y == 0)).sum())
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return mcc, f1, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test", choices=["test", "oof"])
    ap.add_argument("--predictions-dir", default="external_predictions")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--out-dir", default=None, help="default: <work.results>/matched_<split>")
    ap.add_argument("--n-perm", type=int, default=100000)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    have = list_experiments(ctx, need_test=(args.split == "test"))
    models = [m for m in (args.models or have) if m in have]
    if not models:
        raise SystemExit("none of the requested models has predictions")
    files = sorted(Path(args.predictions_dir).glob("*.csv")) if Path(args.predictions_dir).is_dir() else []
    if not files:
        raise SystemExit(f"no competitor CSVs in {args.predictions_dir}. See the docstring for the format.")
    out = ensure_dir(Path(args.out_dir) if args.out_dir
                     else Path(ctx.cfg["work"]["results"]) / f"matched_{args.split}")

    ref = canonical_frame(ctx, args.split)
    y_all = ref["y_true"].to_numpy()
    ours = {m: load_preds(ctx, m, args.split)["z"].to_numpy(dtype=float) for m in models}
    slices = protein_slices(ref["pdb_id"].to_numpy())
    print(f"[matched] split={args.split} ({len(slices)} proteins) | our models: {models} | "
          f"competitors: {[f.stem for f in files]}")

    abs_rows, paired_rows, cover_rows, pooled_rows = [], [], [], []
    for path in files:
        name, merged = load_competitor(path, ref)
        if merged is None:
            continue
        s_all = merged["score"].to_numpy(dtype=float)
        bin_all = merged["prediction"].to_numpy(dtype=float) if "prediction" in merged.columns else None
        per_comp, per_ours, kept, dropped = [], {m: [] for m in models}, [], []
        pooled_idx = []
        for pid, a, b in slices:
            ok = np.isfinite(s_all[a:b])
            cov = float(ok.mean())
            y = y_all[a:b][ok]
            reason = ""
            if cov < args.min_coverage:
                reason = f"coverage {cov:.2f}"
            elif y.sum() == 0 or y.sum() == len(y):
                reason = "no positive or no negative residue after intersection"
            cover_rows.append({"competitor": name, "pdb_id": pid, "coverage": cov,
                               "n_residues": b - a, "n_matched": int(ok.sum()), "kept": reason == "",
                               "reason": reason})
            if reason:
                dropped.append(pid)
                continue
            kept.append(pid)
            idx = np.arange(a, b)[ok]
            pooled_idx.append(idx)
            per_comp.append({"pdb_id": pid, **rank_metrics_ext(y, s_all[idx])})
            for m in models:
                per_ours[m].append({"pdb_id": pid, **rank_metrics_ext(y, ours[m][idx])})
        if len(kept) < 5:
            print(f"[skip] {name}: only {len(kept)} evaluable proteins")
            continue
        print(f"[{name}] matched on {len(kept)} proteins ({len(dropped)} excluded)")
        pc = pd.DataFrame(per_comp).set_index("pdb_id")
        pooled_idx = np.concatenate(pooled_idx)
        yp = y_all[pooled_idx]

        def absrow(nm, pp, scores, binary=None):
            r = {"Method": nm, "Proteins": len(pp)}
            for k in METRICS:
                mean, lo, hi = macro_ci(pp[k].to_numpy(), args.n_boot)
                r[METRIC_LABELS.get(k, k.upper())] = f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
            from sklearn.metrics import average_precision_score, roc_auc_score
            r["pooled ROC-AUC"] = roc_auc_score(yp, scores[pooled_idx])
            r["pooled AUPRC"] = average_precision_score(yp, scores[pooled_idx])
            if binary is not None and np.isfinite(binary[pooled_idx]).all():
                mcc, f1, _, _ = pooled_binary(yp, (binary[pooled_idx] > 0.5).astype(int))
                r["pooled MCC"], r["pooled F1"] = mcc, f1
            return r

        abs_rows.append({"competitor": name, **absrow(name, pc, s_all, bin_all)})
        for m in models:
            pm = pd.DataFrame(per_ours[m]).set_index("pdb_id").reindex(pc.index)
            abs_rows.append({"competitor": name, **absrow(label(m), pm, ours[m])})
            stats = []
            for k in METRICS:
                d = (pm[k] - pc[k]).to_numpy(dtype=float)
                d = d[np.isfinite(d)]
                es = effect_sizes(d, n_boot=args.n_boot, seed=1)
                rng = np.random.default_rng(0)
                boot = d[rng.integers(0, len(d), size=(args.n_boot, len(d)))].mean(axis=1)
                stats.append({"competitor": name, "model": m, "metric": k, "n_proteins": len(d),
                              "competitor_mean": float(pc[k].mean()), "model_mean": float(pm[k].mean()),
                              "mean_delta": float(d.mean()),
                              "ci95_low": float(np.percentile(boot, 2.5)),
                              "ci95_high": float(np.percentile(boot, 97.5)),
                              "p_signflip": signflip_pvalue(d, n_perm=args.n_perm, seed=2),
                              "cohen_dz": es["cohen_dz"], "rank_biserial": es["rank_biserial"],
                              "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()),
                              "n_equal": int((d == 0).sum())})
            prim = [i for i, s in enumerate(stats) if s["metric"] in PRIMARY]
            adj = holm([stats[i]["p_signflip"] for i in prim])
            for s in stats:
                s["p_signflip_holm_primary"] = np.nan
            for i, a_ in zip(prim, adj):
                stats[i]["p_signflip_holm_primary"] = float(a_)
            paired_rows += stats
        pooled_rows.append({"competitor": name, "matched_proteins": len(kept),
                            "excluded_proteins": len(dropped), "matched_residues": len(pooled_idx),
                            "excluded_examples": ", ".join(dropped[:8])})

    if not paired_rows:
        raise SystemExit("no competitor produced an evaluable matched comparison")
    paired = pd.DataFrame(paired_rows)
    paired["win/tie/loss"] = [f"{a}/{c}/{b}" for a, b, c in zip(paired["n_better"], paired["n_worse"], paired["n_equal"])]
    paired.to_csv(out / "matched_paired_full.csv", index=False)
    pd.DataFrame(cover_rows).to_csv(out / "coverage_per_protein.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(out / "matched_sets.csv", index=False)

    absdf = pd.DataFrame(abs_rows).drop_duplicates(subset=["competitor", "Method"])
    write_table(absdf.drop(columns=["competitor"]), out / "table_matched_absolute",
                caption="Head-to-head performance on identical residues. For each competitor, every method is "
                        "scored on the intersection of residues it covers, restricted to proteins where that "
                        "coverage is complete enough and both classes are present; the number of such proteins is "
                        "given. Values are means over proteins with 95\\% bootstrap intervals, plus pooled metrics.")
    slim = pd.DataFrame({
        "Competitor": paired["competitor"], "Our model": [label(m) for m in paired["model"]],
        "Metric": [METRIC_LABELS.get(k, k.upper()) for k in paired["metric"]], "n": paired["n_proteins"],
        "Competitor value": paired["competitor_mean"], "Ours": paired["model_mean"], "Delta": paired["mean_delta"],
        "95% CI": [f"[{a:.4f}, {b:.4f}]" for a, b in zip(paired["ci95_low"], paired["ci95_high"])],
        "p (perm.)": paired["p_signflip"], "p Holm": paired["p_signflip_holm_primary"],
        "d_z": paired["cohen_dz"], "W/T/L": paired["win/tie/loss"]})
    write_table(slim, out / "table_matched_paired", p_cols=["p (perm.)", "p Holm"],
                caption="Paired per-protein comparison against each external predictor on identical residues. "
                        f"Holm adjustment is within the prespecified primary family {PRIMARY}.")

    lines = ["# Matched comparison with external predictors", "",
             f"Split: {args.split}. Minimum residue coverage per protein: {args.min_coverage:.0%}.", ""]
    for _, r in pd.DataFrame(pooled_rows).iterrows():
        lines.append(f"- **{r['competitor']}**: matched on {r['matched_proteins']} proteins "
                     f"({r['matched_residues']} residues); {r['excluded_proteins']} proteins excluded"
                     + (f" (e.g. {r['excluded_examples']})" if r["excluded_examples"] else "") + ".")
    lines.append("")
    for comp in paired["competitor"].unique():
        lines += [f"## vs {comp}", "", "| our model | metric | competitor | ours | delta | 95% CI | p | d_z | W/T/L |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in paired[paired["competitor"] == comp].iterrows():
            lines.append(f"| {label(r['model'])} | {r['metric']} | {r['competitor_mean']:.4f} | "
                         f"{r['model_mean']:.4f} | {r['mean_delta']:+.4f} | "
                         f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {fmt_p(r['p_signflip'])} | "
                         f"{r['cohen_dz']:.3f} | {r['win/tie/loss']} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    show = paired[paired["metric"].isin(["ap", "auc", "bedroc20"])]
    print()
    print(show[["competitor", "model", "metric", "competitor_mean", "model_mean", "mean_delta",
                "ci95_low", "ci95_high", "p_signflip", "cohen_dz", "win/tie/loss"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[DONE] -> {out}")


if __name__ == "__main__":
    main()
