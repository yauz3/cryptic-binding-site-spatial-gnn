#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, average_precision_score

from cryptogrid.common import (
    load_config, load_folds, split_ids, load_dataset_json, cryptic_truth,
    find_feature_file, parse_residue_list, feature_family, ensure_dir,
)

TARGETS = ["jaccard_index", "overlap_ratio", "overlap_count"]
MODELS = ["histgb", "extratrees"]
SCORINGS = ["rank", "pred_rank", "exp"]
TOPM = [None, 20, 50, 100]


def load_proteins(ids, cfg, feature_subset=None):
    frames = []
    drop = set(cfg["features"]["drop_columns"])
    for pid in ids:
        fp = find_feature_file(cfg["project"]["grid_features_dir"], pid)
        if fp is None:
            continue
        df = pd.read_csv(fp, low_memory=False)
        df["pdb_id"] = pid
        frames.append(df)
    if not frames:
        return pd.DataFrame(), []
    df = pd.concat(frames, ignore_index=True, sort=False)
    numeric = []
    for c in df.columns:
        if c in drop:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().any():
            numeric.append(c)
    if feature_subset and feature_subset != "all":
        numeric = [c for c in numeric if feature_family(c) == feature_subset]
    return df, numeric


def score_residues(g, pred, mode, topm):
    order = np.argsort(-pred)
    if topm is not None:
        order = order[:topm]
    scores = {}
    p = pred.copy().astype(float)
    pmin, pmax = np.nanmin(p), np.nanmax(p)
    pn = (p - pmin) / (pmax - pmin + 1e-12)
    for rank, j in enumerate(order, 1):
        if mode == "rank":
            w = 1.0 / rank
        elif mode == "pred_rank":
            w = float(pn[j]) / rank
        elif mode == "exp":
            w = float(np.exp(-0.03 * (rank - 1)))
        else:
            raise ValueError(mode)
        for r in parse_residue_list(g.iloc[j]["pocket_residues_list"]):
            scores[r] = scores.get(r, 0.0) + w
    return scores


def macro_eval(df, pred, dataset, label_side, mode, topm):
    vals = []
    for pid, g in df.assign(_pred=pred).groupby("pdb_id", sort=False):
        truth = cryptic_truth(dataset, pid, label_side)
        if not truth:
            continue
        scores = score_residues(g, g["_pred"].to_numpy(), mode, topm)
        universe = set(scores) | truth
        y = np.array([1 if r in truth else 0 for r in universe])
        s = np.array([scores.get(r, 0.0) for r in universe])
        if len(np.unique(y)) < 2:
            continue
        vals.append((roc_auc_score(y, s), average_precision_score(y, s)))
    if not vals:
        return np.nan, np.nan, 0
    a = np.array(vals)
    return a[:, 0].mean(), a[:, 1].mean(), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--final-test", action="store_true", help="Use official test. Default uses held-out train fold.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    folds = load_folds(cfg["project"]["folds_json"])
    train, test, tf = split_ids(folds)
    vname = cfg["ablation"].get("validation_fold", "train_3")
    if args.final_test:
        fit, eva, split_name = train, test, "official_test"
    else:
        eva = tf.get(vname, [])
        fit = [x for x in train if x not in set(eva)]
        split_name = vname
    dataset = load_dataset_json(cfg["project"]["dataset_json"])
    side = cfg["evaluation"].get("label_side", "apo")

    feature_sets = ["all", "distance_geometry", "counts", "face", "filtration_energy"]
    rows = []
    for fset in feature_sets:
        tr, cols = load_proteins(fit, cfg, fset)
        va, _ = load_proteins(eva, cfg, fset)
        if len(tr) == 0 or len(va) == 0 or not cols:
            continue
        for target in TARGETS:
            if target not in tr or target not in va:
                continue
            y = pd.to_numeric(tr[target], errors="coerce")
            m = y.notna()
            X = tr.loc[m, cols].apply(pd.to_numeric, errors="coerce")
            y = y[m].to_numpy(float)
            Xt = va.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
            imp = SimpleImputer(strategy="median")
            A = imp.fit_transform(X)
            B = imp.transform(Xt)
            for model_name in MODELS:
                if model_name == "histgb":
                    model = HistGradientBoostingRegressor(
                        max_iter=250, learning_rate=.05, max_leaf_nodes=31,
                        l2_regularization=1, random_state=42
                    )
                else:
                    model = ExtraTreesRegressor(
                        n_estimators=400, max_features=0.7, min_samples_leaf=1,
                        n_jobs=-1, random_state=42
                    )
                model.fit(A, y)
                pred = model.predict(B)
                scorings = SCORINGS if fset == "all" else ["pred_rank"]
                topms = TOPM if fset == "all" else [50]
                for scoring, topm in itertools.product(scorings, topms):
                    auc, apv, n = macro_eval(va, pred, dataset, side, scoring, topm)
                    rows.append({
                        "feature_set": fset, "target": target, "model": model_name,
                        "scoring": scoring, "top_m": ("all" if topm is None else topm),
                        "macro_roc_auc": auc, "macro_average_precision": apv,
                        "n_proteins": n, "evaluation_split": split_name,
                        "n_features": len(cols),
                    })
                    print(rows[-1])

    out = ensure_dir(Path(cfg["work"]["results"]) / "ablation")
    pd.DataFrame(rows).sort_values(
        ["macro_roc_auc", "macro_average_precision"], ascending=False
    ).to_csv(out / "candidate_ranking_ablation.csv", index=False)
    print("[DONE]", out)


if __name__ == "__main__":
    main()
