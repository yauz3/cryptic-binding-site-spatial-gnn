#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from cryptogrid.common import load_config, load_folds, split_ids, ensure_dir


def sample_rows(ids, sdir, max_rows, indices):
    xs, ys, pids = [], [], []
    rng = np.random.default_rng(42)
    for pid in ids:
        p = Path(sdir) / f"{pid}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=False)
        x = z["X_struct"][:, indices].astype(np.float32)
        y = z["y"].astype(int)
        if len(y) > 1500:
            take = rng.choice(len(y), 1500, replace=False)
            x, y = x[take], y[take]
        xs.append(x)
        ys.append(y)
        pids.extend([pid] * len(y))
    if not xs:
        return None, None, None
    X = np.vstack(xs)
    Y = np.concatenate(ys)
    P = np.array(pids)
    if len(Y) > max_rows:
        take = rng.choice(len(Y), max_rows, replace=False)
        X, Y, P = X[take], Y[take], P[take]
    return X, Y, P


def macro_auc_ap(y, s, p):
    vals = []
    for pid in np.unique(p):
        m = p == pid
        yy, ss = y[m], s[m]
        if len(np.unique(yy)) < 2:
            continue
        vals.append((roc_auc_score(yy, ss), average_precision_score(yy, ss)))
    if not vals:
        return np.nan, np.nan
    a = np.asarray(vals)
    return a[:, 0].mean(), a[:, 1].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sdir = Path(cfg["work"]["struct_cache"])
    names = json.loads((sdir / "feature_names.json").read_text())
    families = pd.read_csv(sdir / "feature_families.csv")

    folds = load_folds(cfg["project"]["folds_json"])
    train, _, tf = split_ids(folds)
    val_name = cfg["ablation"].get("validation_fold", "train_3")
    val = tf.get(val_name, [])
    fit = [x for x in train if x not in set(val)]
    maxtr = int(cfg["ablation"].get("max_train_residues_classical", 150000))
    maxva = int(cfg["ablation"].get("max_valid_residues_classical", 60000))
    topn = int(cfg["ablation"].get("top_features_per_family", 128))

    configs = {"all": list(range(len(names)))}
    for fam in sorted(families.family.unique()):
        idx = families.index[families.family == fam].tolist()
        if idx:
            configs[fam] = idx
    configs["multiscale_summary"] = [i for i, n in enumerate(names) if n.startswith("summary::")]

    rows = []
    for label, idx0 in configs.items():
        if not idx0:
            continue
        Xtr, ytr, _ = sample_rows(fit, sdir, maxtr, idx0)
        Xv, yv, pv = sample_rows(val, sdir, maxva, idx0)
        if Xtr is None or Xv is None:
            continue
        imp = SimpleImputer(strategy="median")
        A = imp.fit_transform(Xtr)
        B = imp.transform(Xv)
        if A.shape[1] > topn:
            f, _ = f_classif(A, ytr)
            f = np.nan_to_num(f, nan=-np.inf)
            sel = np.argsort(-f)[:topn]
            A, B = A[:, sel], B[:, sel]
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=.05, max_leaf_nodes=31,
            l2_regularization=1, random_state=42
        )
        clf.fit(A, ytr)
        s = clf.predict_proba(B)[:, 1]
        auc, apv = macro_auc_ap(yv, s, pv)
        rows.append({
            "ablation": label,
            "n_input_features": len(idx0),
            "n_model_features": A.shape[1],
            "macro_roc_auc": auc,
            "macro_average_precision": apv,
            "validation_fold": val_name,
            "n_validation_proteins": len(np.unique(pv)),
        })
        print(rows[-1])

    out = ensure_dir(Path(cfg["work"]["results"]) / "ablation")
    pd.DataFrame(rows).sort_values("macro_roc_auc", ascending=False).to_csv(
        out / "residue_feature_ablation.csv", index=False
    )
    print("[DONE]", out)


if __name__ == "__main__":
    main()
