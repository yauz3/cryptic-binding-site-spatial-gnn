from pathlib import Path
import json
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from cryptogrid.common import load_config, load_folds, split_ids


def load_training_matrix(config="config.yaml", max_rows=30000, max_features=250, seed=42):
    cfg = load_config(config)
    sdir = Path(cfg["work"]["struct_cache"])
    names = json.loads((sdir / "feature_names.json").read_text())
    train, _, _ = split_ids(load_folds(cfg["project"]["folds_json"]))
    rng = np.random.default_rng(seed)
    xs, ys, ps = [], [], []
    for pid in train:
        p = sdir / f"{pid}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=False)
        x = z["X_struct"].astype(np.float32)
        y = z["y"].astype(int)
        if len(y) > 600:
            take = rng.choice(len(y), 600, replace=False)
            x, y = x[take], y[take]
        xs.append(x)
        ys.append(y)
        ps.extend([pid] * len(y))
    if not xs:
        raise RuntimeError("No training structural cache found")
    X = np.vstack(xs)
    y = np.concatenate(ys)
    pids = np.asarray(ps)
    if len(y) > max_rows:
        take = rng.choice(len(y), max_rows, replace=False)
        X, y, pids = X[take], y[take], pids[take]
    with np.errstate(all="ignore"):
        vari = np.nanvar(X, axis=0)
    vari = np.nan_to_num(vari, nan=-1)
    idx = np.argsort(-vari)[:min(max_features, X.shape[1])]
    feat = [names[i] for i in idx]
    X = X[:, idx]
    X = SimpleImputer(strategy="median").fit_transform(X)
    return X, y, pids, feat, cfg


def scaled(X):
    return StandardScaler().fit_transform(X)
