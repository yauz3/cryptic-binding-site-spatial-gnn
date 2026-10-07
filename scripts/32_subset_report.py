#!/usr/bin/env python3
"""
Q1 EXPERIMENT E1, PANEL A - our metrics recomputed on exactly the protein subset a published
method reported on.

Why a separate script
---------------------
Published cryptic-site numbers are computed on different subsets of the CryptoBench test split:
the benchmark paper's PocketMiner comparison drops 60 of 222 proteins, and at least one recent
method reports on 193. Quoting our 222-protein number next to their subset number is not a
comparison. This script recomputes our metrics on exactly their protein list, so the two
columns describe the same proteins.

What it is NOT
--------------
This is still an aggregate-versus-aggregate comparison. Without the competitor's per-protein
predictions there is no paired test and no confidence interval on the difference, so no p-value
can be reported. That is Panel B's job (scripts/26), which needs the competitor's actual
predictions. Panel A exists to make the published-number comparison honest, not to replace
Panel B, and the output says so in its own caption.

Inputs
------
--subset FILE       one PDB identifier per line (or a CSV with a pdb_id column) listing the
                    proteins the published method evaluated on
--published FILE    optional CSV of literature values with columns
                    method,metric,value[,subset_name,source]; these are printed beside ours
                    and never mixed into any statistical test

Output (under <results>/subset_<name>/)
    subset_coverage.csv     which requested proteins exist in our split, which do not
    table_subset.csv/.tex   our models on that subset, with the published rows appended
    summary.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cryptogrid.common import ensure_dir
from cryptogrid.exp_common import METRICS, Store, list_experiments, load_preds, make_ctx, per_protein_table
from cryptogrid.exp_stats import METRIC_LABELS, label, macro_ci, write_table

POOLED = ["pooled ROC-AUC", "pooled AP"]


def read_subset(path: Path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = next((c for c in df.columns if c.lower() in {"pdb_id", "pdb", "id", "apo_pdb_id"}), df.columns[0])
        ids = df[col].astype(str).tolist()
    else:
        ids = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return [i.lower().strip() for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test", choices=["test", "oof"])
    ap.add_argument("--subset", required=True)
    ap.add_argument("--name", required=True, help="short label, e.g. unipocket193 or pocketminer162")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--published", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    have = list_experiments(ctx, need_test=(args.split == "test"))
    models = [m for m in (args.models or have) if m in have]
    if not models:
        raise SystemExit("no model predictions found")
    out = ensure_dir(Path(args.out_dir) if args.out_dir
                     else Path(ctx.cfg["work"]["results"]) / f"subset_{args.name}")

    want = read_subset(Path(args.subset))
    split_ids = set(ctx.test_ids if args.split == "test" else ctx.train_ids)
    present = [p for p in want if p in split_ids]
    missing = [p for p in want if p not in split_ids]
    pd.DataFrame({"pdb_id": want,
                  "in_split": [p in split_ids for p in want]}).to_csv(out / "subset_coverage.csv", index=False)
    print(f"[subset {args.name}] requested {len(want)}, present in the {args.split} split {len(present)}"
          + (f", absent {len(missing)} (e.g. {missing[:5]})" if missing else ""))
    if len(present) < 5:
        raise SystemExit("fewer than five requested proteins are in this split; check the identifiers")

    rows = []
    for m in models:
        df = load_preds(ctx, m, args.split)
        sub = df[df["pdb_id"].isin(present)]
        pp = per_protein_table(sub)
        r = {"Method": label(m), "Subset": args.name, "Proteins": len(pp), "Source": "this work"}
        for k in METRICS:
            mean, lo, hi = macro_ci(pp[k].to_numpy(dtype=float), args.n_boot)
            r[METRIC_LABELS[k]] = f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
        from sklearn.metrics import average_precision_score, roc_auc_score
        y, z = sub["y_true"].to_numpy(), sub["z"].to_numpy(dtype=float)
        r["pooled ROC-AUC"] = roc_auc_score(y, z)
        r["pooled AP"] = average_precision_score(y, z)
        rows.append(r)
    table = pd.DataFrame(rows)

    if args.published and Path(args.published).exists():
        pub = pd.read_csv(args.published)
        need = {"method", "metric", "value"}
        if need <= set(c.lower() for c in pub.columns):
            pub.columns = [c.lower() for c in pub.columns]
            for meth, g in pub.groupby("method"):
                r = {"Method": meth, "Subset": str(g["subset_name"].iloc[0]) if "subset_name" in g else args.name,
                     "Proteins": np.nan, "Source": str(g["source"].iloc[0]) if "source" in g else "published"}
                for _, x in g.iterrows():
                    key = METRIC_LABELS.get(str(x["metric"]).lower(), str(x["metric"]))
                    r[key] = f"{float(x['value']):.4f}"
                rows.append(r)
            table = pd.DataFrame(rows)
        else:
            print(f"[WARN] {args.published}: expected columns method,metric,value")

    write_table(table, out / "table_subset",
                caption=f"Performance on the {len(present)}-protein subset reported by the "
                        f"`{args.name}` source, recomputed for our models on exactly those proteins. "
                        "Rows whose Source is not ``this work'' are values quoted from the respective "
                        "publication; they were computed by their authors with their own residue "
                        "universe and aggregation, so they are shown for context and are NOT entered "
                        "into any statistical comparison. A paired test requires the competitor's "
                        "per-protein predictions and is reported separately.")

    lines = [f"# Subset comparison: {args.name} ({len(present)} proteins, {args.split} split)", "",
             f"- Requested {len(want)} identifiers; {len(present)} are in our {args.split} split.",
             "- Our rows are recomputed on exactly these proteins. Published rows are quoted, not tested.", ""]
    for _, r in table.iterrows():
        bits = [f"{k} {r[k]}" for k in ("AP", "ROC-AUC", "BEDROC$_{20}$", "pooled AP", "pooled ROC-AUC")
                if k in r and pd.notna(r[k])]
        lines.append(f"- **{r['Method']}** ({r['Source']}): " + ", ".join(str(b) for b in bits))
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(table[[c for c in ("Method", "Source", "Proteins", "AP", "BEDROC$_{20}$", "pooled AP") if c in table]]
          .to_string(index=False))
    print(f"\n[DONE] -> {out}")


if __name__ == "__main__":
    main()
