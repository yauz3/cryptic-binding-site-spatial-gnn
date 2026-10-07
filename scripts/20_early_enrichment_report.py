#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC


PRIMARY_FAMILY = [
    "ef1",
    "ef3",
    "bedroc20",
    "bedroc50",
    "bedroc80",
]


def enrichment_factor(y, score, frac):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    n = len(y)
    n_pos = int(y.sum())

    if n == 0 or n_pos == 0:
        return np.nan

    k = max(1, int(np.ceil(frac * n)))

    order = np.argsort(-score, kind="mergesort")
    hits = int(y[order[:k]].sum())

    precision_top = hits / k
    prevalence = n_pos / n

    return precision_top / prevalence


def bedroc(y, score, alpha):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    if y.sum() == 0:
        return np.nan

    order = np.argsort(-score, kind="mergesort")

    ranked = [
        [float(score[i]), int(y[i])]
        for i in order
    ]

    return float(
        CalcBEDROC(
            ranked,
            1,
            float(alpha),
        )
    )


def protein_metrics(df):
    rows = []

    for pid, g in df.groupby(
        "pdb_id",
        sort=False
    ):
        y = g["y_true"].to_numpy(dtype=int)
        z = g["z"].to_numpy(dtype=float)

        row = {
            "pdb_id": str(pid),

            "ap":
                average_precision_score(y, z),

            "auc":
                roc_auc_score(y, z)
                if len(np.unique(y)) > 1
                else np.nan,

            "ef1":
                enrichment_factor(
                    y, z, 0.01
                ),

            "ef3":
                enrichment_factor(
                    y, z, 0.03
                ),

            "ef5":
                enrichment_factor(
                    y, z, 0.05
                ),

            "bedroc20":
                bedroc(y, z, 20),

            "bedroc50":
                bedroc(y, z, 50),

            "bedroc80":
                bedroc(y, z, 80),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def holm(pvalues):
    p = np.asarray(
        pvalues,
        dtype=float
    )

    m = len(p)

    order = np.argsort(p)
    adjusted = np.empty(m)

    running = 0.0

    for rank, idx in enumerate(order):
        val = min(
            1.0,
            (m - rank) * p[idx]
        )

        running = max(
            running,
            val
        )

        adjusted[idx] = running

    return adjusted


def bootstrap_delta(
    d,
    n_boot,
    seed
):
    d = np.asarray(
        d,
        dtype=float
    )

    d = d[np.isfinite(d)]

    rng = np.random.default_rng(seed)

    means = np.empty(
        n_boot,
        dtype=float
    )

    n = len(d)

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        means[i] = d[idx].mean()

    lo, hi = np.percentile(
        means,
        [2.5, 97.5]
    )

    return (
        float(lo),
        float(hi),
        float(np.mean(means > 0))
    )


def paired_stats(
    base,
    new,
    metric,
    n_boot,
    seed
):
    m = base[
        ["pdb_id", metric]
    ].merge(
        new[
            ["pdb_id", metric]
        ],
        on="pdb_id",
        how="inner",
        suffixes=(
            "_base",
            "_new"
        ),
        validate="one_to_one"
    )

    a = m[
        f"{metric}_base"
    ].to_numpy(dtype=float)

    b = m[
        f"{metric}_new"
    ].to_numpy(dtype=float)

    ok = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    a = a[ok]
    b = b[ok]

    d = b - a

    if len(d) == 0:
        raise RuntimeError(
            f"No finite values: {metric}"
        )

    if np.allclose(
        d,
        0
    ):
        p = 1.0
    else:
        try:
            p = float(
                wilcoxon(
                    b,
                    a,
                    alternative="two-sided",
                    zero_method="pratt",
                ).pvalue
            )
        except ValueError:
            p = 1.0

    lo, hi, prob = bootstrap_delta(
        d,
        n_boot=n_boot,
        seed=seed,
    )

    return {
        "metric": metric,

        "n_proteins":
            int(len(d)),

        "baseline_mean":
            float(np.mean(a)),

        "new_mean":
            float(np.mean(b)),

        "mean_delta":
            float(np.mean(d)),

        "median_delta":
            float(np.median(d)),

        "wins":
            int((d > 0).sum()),

        "losses":
            int((d < 0).sum()),

        "ties":
            int((d == 0).sum()),

        "wilcoxon_p":
            p,

        "bootstrap_ci95_low":
            lo,

        "bootstrap_ci95_high":
            hi,

        "bootstrap_prob_delta_gt_0":
            prob,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="config.yaml",
    )

    ap.add_argument(
        "--split",
        choices=["oof", "test"],
        default="oof",
    )

    ap.add_argument(
        "--allow-test",
        action="store_true",
    )

    ap.add_argument(
        "--n-boot",
        type=int,
        default=5000,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = ap.parse_args()

    if (
        args.split == "test"
        and not args.allow_test
    ):
        raise SystemExit(
            "Refusing to inspect test. "
            "Use --allow-test only after "
            "the model is frozen."
        )

    root = Path(
        "results/exp"
    )

    stats_dir = (
        root / "_stats"
    )

    stats_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "oof_predictions.csv"
        if args.split == "oof"
        else "test_predictions.csv"
    )

    files = sorted(
        root.glob(
            f"*/{suffix}"
        )
    )

    if not files:
        raise RuntimeError(
            f"No {suffix} files under {root}"
        )

    datasets = {}

    for fp in files:
        experiment = (
            fp.parent.name
        )

        if experiment.startswith("_"):
            continue

        df = pd.read_csv(fp)

        required = {
            "pdb_id",
            "y_true",
            "z",
        }

        if not required.issubset(
            df.columns
        ):
            print(
                "[SKIP]",
                experiment,
                "missing",
                required - set(df.columns)
            )
            continue

        datasets[experiment] = (
            protein_metrics(df)
        )

    baseline = "base_plm"

    if baseline not in datasets:
        raise RuntimeError(
            "base_plm predictions missing."
        )

    # ------------------------------------
    # Absolute summary
    # ------------------------------------

    summary_rows = []

    for name, pp in datasets.items():
        row = {
            "experiment": name,
            "n_proteins":
                pp["pdb_id"].nunique(),
        }

        for metric in [
            "ap",
            "auc",
            "ef1",
            "ef3",
            "ef5",
            "bedroc20",
            "bedroc50",
            "bedroc80",
        ]:
            row[metric] = float(
                pp[metric].mean()
            )

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "bedroc20",
        ascending=False
    )

    summary.to_csv(
        stats_dir /
        f"early_enrichment_{args.split}_summary.csv",
        index=False
    )

    # ------------------------------------
    # Paired stats versus pLM anchor
    # ------------------------------------

    base = datasets[baseline]

    all_rows = []

    candidates = [
        x
        for x in datasets
        if x != baseline
    ]

    for ci, name in enumerate(
        candidates
    ):
        method_rows = []

        for mi, metric in enumerate(
            PRIMARY_FAMILY
        ):
            s = paired_stats(
                base,
                datasets[name],
                metric,
                n_boot=args.n_boot,
                seed=(
                    args.seed
                    + 1000 * ci
                    + mi
                ),
            )

            s["experiment"] = name

            method_rows.append(s)

        adjusted = holm(
            [
                x["wilcoxon_p"]
                for x in method_rows
            ]
        )

        for row, adj in zip(
            method_rows,
            adjusted
        ):
            row[
                "wilcoxon_p_holm5"
            ] = float(adj)

            row[
                "holm5_significant_0.05"
            ] = bool(
                adj < 0.05
            )

            all_rows.append(row)

    stats = pd.DataFrame(
        all_rows
    )

    # Global correction if one scans
    # all methods x all primary metrics.
    if len(stats):
        stats[
            "wilcoxon_p_holm_global"
        ] = holm(
            stats[
                "wilcoxon_p"
            ].to_numpy()
        )

        stats[
            "holm_global_significant_0.05"
        ] = (
            stats[
                "wilcoxon_p_holm_global"
            ] < 0.05
        )

    stats.to_csv(
        stats_dir /
        f"early_enrichment_{args.split}_paired_stats.csv",
        index=False
    )

    per_protein = []

    for name, pp in datasets.items():
        tmp = pp.copy()
        tmp[
            "experiment"
        ] = name

        per_protein.append(
            tmp
        )

    pd.concat(
        per_protein,
        ignore_index=True
    ).to_csv(
        stats_dir /
        f"early_enrichment_{args.split}_per_protein.csv",
        index=False
    )

    print()
    print(
        "=== EARLY ENRICHMENT SUMMARY ==="
    )

    print(
        summary[
            [
                "experiment",
                "ap",
                "auc",
                "ef1",
                "ef3",
                "bedroc20",
                "bedroc50",
                "bedroc80",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=== PAIRED VS base_plm ==="
    )

    if len(stats):
        print(
            stats[
                [
                    "experiment",
                    "metric",
                    "mean_delta",
                    "wilcoxon_p",
                    "wilcoxon_p_holm5",
                    "wilcoxon_p_holm_global",
                    "bootstrap_ci95_low",
                    "bootstrap_ci95_high",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "[DONE]",
        stats_dir
    )


if __name__ == "__main__":
    main()
