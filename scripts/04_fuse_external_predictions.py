#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

from cryptogrid.common import (
    load_config, load_folds, split_ids, best_threshold,
    metric_summary, ensure_dir,
)


def load_struct_rows(ids, struct_dir, feature_indices=None):
    rows = []
    for pid in ids:
        p = Path(struct_dir) / f"{pid}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=False)
        x = z["X_struct"].astype(np.float32)
        if feature_indices is not None:
            x = x[:, feature_indices]
        for i, (rk, y) in enumerate(zip(z["residue_keys"].astype(str), z["y"])):
            rows.append((pid, rk, int(y), x[i]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--predictions", required=True, help="CSV: method,pdb_id,residue_key,score")
    ap.add_argument("--top-struct-features", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ext = pd.read_csv(args.predictions)
    need = {"pdb_id", "residue_key", "score"}
    if not need.issubset(ext.columns):
        raise ValueError(f"External CSV needs {sorted(need)}")
    method = str(ext["method"].iloc[0]) if "method" in ext and len(ext) else Path(args.predictions).stem
    ext["pdb_id"] = ext["pdb_id"].astype(str).str.lower()
    ext["residue_key"] = ext["residue_key"].astype(str)

    folds = load_folds(cfg["project"]["folds_json"])
    train, test, tf = split_ids(folds)
    cal_name = cfg["training"].get("calibration_fold", "train_3")
    cal = set(tf.get(cal_name, []))
    fit = [x for x in train if x not in cal]
    val = [x for x in train if x in cal]
    sdir = Path(cfg["work"]["struct_cache"])

    trrows = load_struct_rows(train, sdir)
    tr_map = {(p, r): (y, x) for p, r, y, x in trrows}
    matched = []
    for _, r in ext[ext.pdb_id.isin(train)].iterrows():
        k = (r.pdb_id, r.residue_key)
        if k in tr_map and np.isfinite(r.score):
            y, x = tr_map[k]
            matched.append((r.pdb_id, r.residue_key, y, float(r.score), x))
    if not matched:
        raise RuntimeError("No matched training residues between external predictions and structural cache")

    Xs = np.stack([m[4] for m in matched])
    yy = np.array([m[2] for m in matched])
    imp0 = SimpleImputer(strategy="median")
    X0 = imp0.fit_transform(Xs)
    f, _ = f_classif(X0, yy)
    f = np.nan_to_num(f, nan=-np.inf)
    k = min(args.top_struct_features, len(f))
    idx = np.argsort(-f)[:k]

    def build(ids):
        rows = load_struct_rows(ids, sdir, idx)
        mp = {(p, r): (y, x) for p, r, y, x in rows}
        out = []
        for _, rr in ext[ext.pdb_id.isin(ids)].iterrows():
            key = (rr.pdb_id, rr.residue_key)
            if key in mp and np.isfinite(rr.score):
                y, x = mp[key]
                out.append((rr.pdb_id, rr.residue_key, y, float(rr.score), x))
        return out

    fitrows = build(fit)
    valrows = build(val)
    testrows = build(test)
    if not fitrows or not valrows or not testrows:
        raise RuntimeError("Need matched fit/validation/test residues for external fusion")

    imp = SimpleImputer(strategy="median")
    Xfit_s = imp.fit_transform(np.stack([r[4] for r in fitrows]))
    Xfit = np.c_[np.array([r[3] for r in fitrows]), Xfit_s]
    yfit = np.array([r[2] for r in fitrows])
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=42
    )
    clf.fit(Xfit, yfit)

    def pred(rows):
        xs = imp.transform(np.stack([r[4] for r in rows]))
        X = np.c_[np.array([r[3] for r in rows]), xs]
        s = clf.predict_proba(X)[:, 1]
        return pd.DataFrame({
            "pdb_id": [r[0] for r in rows],
            "residue_key": [r[1] for r in rows],
            "y_true": [r[2] for r in rows],
            "external_score": [r[3] for r in rows],
            "y_score": s,
        })

    vd = pred(valrows)
    thr, _ = best_threshold(vd.y_true, vd.y_score, cfg["training"].get("threshold_metric", "mcc"))

    allrows = build(train)
    Xall_s = imp.fit_transform(np.stack([r[4] for r in allrows]))
    Xall = np.c_[np.array([r[3] for r in allrows]), Xall_s]
    yall = np.array([r[2] for r in allrows])
    clf.fit(Xall, yall)
    td = pred(testrows)

    outdir = ensure_dir(Path(cfg["work"]["results"]) / "predictions")
    td["method"] = f"{method}+gridfusion"
    td["threshold"] = thr
    td.to_csv(outdir / f"{method}_gridfusion_test_predictions.csv", index=False)

    base = td[["pdb_id", "residue_key", "y_true", "external_score"]].rename(columns={"external_score": "y_score"})
    base["method"] = method
    base_val = vd[["pdb_id", "residue_key", "y_true", "external_score"]].rename(columns={"external_score": "y_score"})
    bthr, _ = best_threshold(base_val.y_true, base_val.y_score, cfg["training"].get("threshold_metric", "mcc"))
    base["threshold"] = bthr
    base.to_csv(outdir / f"{method}_test_predictions.csv", index=False)

    report = {
        "external": metric_summary(base, bthr),
        "external_plus_grid": metric_summary(td, thr),
        "selected_structural_features": idx.tolist(),
        "method": method,
        "matched_test_proteins": int(td.pdb_id.nunique()),
    }
    (outdir / f"{method}_gridfusion_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
