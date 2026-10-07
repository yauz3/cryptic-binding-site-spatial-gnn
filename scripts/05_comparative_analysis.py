#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cryptogrid.common import load_config, metric_summary, per_protein_metrics, bootstrap_macro_ci, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pred_dir = Path(cfg["work"]["results"]) / "predictions"
    out = ensure_dir(Path(cfg["work"]["results"]) / "comparative")

    summaries = []
    pps = []
    for fp in sorted(pred_dir.glob("*_test_predictions.csv")):
        df = pd.read_csv(fp)
        if not {"pdb_id", "y_true", "y_score"}.issubset(df.columns):
            continue
        method = str(df["method"].iloc[0]) if "method" in df and len(df) else fp.stem.replace("_test_predictions", "")
        thr = float(df["threshold"].iloc[0]) if "threshold" in df.columns and len(df) else 0.5
        mpath = Path(cfg["work"]["results"]) / "models" / f"{method}_metrics.json"
        if mpath.exists():
            try:
                thr = float(json.loads(mpath.read_text())["threshold"])
            except Exception:
                pass
        sm = metric_summary(df, thr)
        sm["method"] = method
        sm["prediction_file"] = str(fp)
        pp = per_protein_metrics(df, thr)
        pp["method"] = method
        lo, hi = bootstrap_macro_ci(pp, "roc_auc", cfg["evaluation"].get("bootstrap_reps", 1000))
        sm["macro_roc_auc_ci_low"] = lo
        sm["macro_roc_auc_ci_high"] = hi
        lo, hi = bootstrap_macro_ci(pp, "average_precision", cfg["evaluation"].get("bootstrap_reps", 1000))
        sm["macro_ap_ci_low"] = lo
        sm["macro_ap_ci_high"] = hi
        summaries.append(sm)
        pps.append(pp)

    rerun = pd.DataFrame(summaries)
    if len(rerun):
        rerun = rerun.sort_values("macro_roc_auc", ascending=False)
    rerun.to_csv(out / "rerun_metrics.csv", index=False)
    if pps:
        pd.concat(pps, ignore_index=True).to_csv(out / "per_protein_metrics.csv", index=False)

    lit = Path("literature_reference_metrics.csv")
    if lit.exists():
        pd.read_csv(lit).to_csv(out / "literature_reference_metrics.csv", index=False)

    if len(rerun):
        plot = rerun.sort_values("macro_roc_auc")
        fig, ax = plt.subplots(figsize=(9, max(4, .5 * len(plot))))
        x = plot["macro_roc_auc"].to_numpy()
        lo = plot["macro_roc_auc_ci_low"].to_numpy()
        hi = plot["macro_roc_auc_ci_high"].to_numpy()
        ax.barh(plot["method"], x)
        ax.errorbar(x, np.arange(len(plot)), xerr=np.vstack([x - lo, hi - x]), fmt="none", capsize=3)
        ax.set_xlabel("Protein-macro ROC-AUC")
        ax.set_xlim(0.45, 1.0)
        ax.set_title("CryptoBench same-split rerun comparison")
        fig.tight_layout()
        fig.savefig(out / "macro_roc_auc_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        plot = rerun.sort_values("macro_average_precision")
        fig, ax = plt.subplots(figsize=(9, max(4, .5 * len(plot))))
        ax.barh(plot["method"], plot["macro_average_precision"])
        ax.set_xlabel("Protein-macro Average Precision")
        ax.set_title("CryptoBench same-split rerun comparison")
        fig.tight_layout()
        fig.savefig(out / "macro_ap_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(rerun.to_string(index=False) if len(rerun) else "[WARN] no predictions found")
    print("[DONE]", out)


if __name__ == "__main__":
    main()
