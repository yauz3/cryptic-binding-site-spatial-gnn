#!/usr/bin/env python3
"""
IDEA 5 - Gradient-boosted RESIDUAL correction of the pLM (Grid + geometry + spatial context).

Hypothesis
----------
(a) The Grid tower is weak (macro AUC 0.65) partly because 1240 heavy-tailed, mostly-missing
    descriptors are z-scored, mean-imputed and fed to an MLP. Trees handle NaN and outliers natively.
(b) A structure model trained to predict y re-learns what the pLM already knows; fusion then
    averages two correlated answers. Boosting FROM the pLM logit learns only the complementary part.

Equation (pLM is the anchor by construction)
--------------------------------------------
    f = GBM trained with init_score = z_plm (train-OOF logits), binary log-loss,
        scale_pos_weight = neg/pos (same class weighting as the pLM head)
    z' = z_plm + eta * f(x)                         eta = 0 -> exactly the pLM

    x = [Grid descriptors (--features grid) | 8 Grid summary columns (summary) |
         geometry/flexibility from script 10 (geom) |
         spatial context of the pLM logit: z, Gaussian-smoothed z (4 A, 8 A), neighbour max (8 A),
         within-protein rank of z (ctx)]

Searched (nested on train OOF, J = 0.5 AP + 0.5 BEDROC20): eta in {0.25, 0.5, 0.75, 1.0}
Frozen a priori: all GBM hyper-parameters below. Number of trees: early stopping on an INNER
protein-level hold-out taken from the fitting folds only, then refit on all fitting folds.

Cross-fitting: the model applied to official fold k is trained on the other three folds, using
their OOF pLM logits. (Those logits came from base models that saw fold k; this is ordinary
stacking and only affects the single scalar eta.)

Required ablations (run the script again with another --features / --name):
    --features summary          support-only control: is the gain just "residue is in a pocket"?
    --features geom,ctx         no Grid at all: what do coordinates alone add?
    --features grid             Grid only, no geometry/context
    default grid,geom,ctx       full model

Outputs: results/exp/<name>/{oof,test}_predictions.csv, selection.json, config_search_oof.csv,
         feature_importance.csv (gain, final model)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cryptogrid.exp_common import (BASELINE, Store, exp_dir, gaussian_smooth, load_preds, make_ctx,
                                   neighbor_max, neighbor_pairs, protein_slices, run_grid_experiment)

ETAS = (0.25, 0.5, 0.75, 1.0)

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:                       # pragma: no cover
    HAVE_LGB = False


def context_features(z, cb):
    pairs = neighbor_pairs(cb.astype(np.float64), 24.0)
    zs4, zs8 = gaussian_smooth(z, pairs, 4.0), gaussian_smooth(z, pairs, 8.0)
    rank = (np.argsort(np.argsort(z)) + 0.5) / max(1, len(z))
    return np.stack([z, zs4, zs8, neighbor_max(z, pairs, 8.0), z - zs4, rank], axis=1)


CTX_NAMES = ["ctx::z_plm", "ctx::z_smooth4", "ctx::z_smooth8", "ctx::z_nbmax8", "ctx::z_minus_smooth4", "ctx::z_rank"]


def build_matrix(ctx, store, df, groups):
    """Feature matrix in canonical row order + names. Uses logits and structure only (no labels)."""
    z = df["z"].to_numpy(dtype=np.float64)
    names = []
    snames = store.struct_feature_names()
    if "grid" in groups:
        names += [f"grid::{n}" for n in snames]
    elif "summary" in groups:
        sidx = [i for i, n in enumerate(snames) if n.startswith("summary::")]
        names += [f"grid::{snames[i]}" for i in sidx]
    if "geom" in groups:
        names += [f"geom::{n}" for n in store.geom_feature_names()]
    if "ctx" in groups:
        names += CTX_NAMES
    X = np.empty((len(df), len(names)), dtype=np.float32)
    for pid, a, b in protein_slices(df["pdb_id"].to_numpy()):
        blocks = []
        if "grid" in groups:
            blocks.append(store.struct(pid))
        elif "summary" in groups:
            blocks.append(store.struct(pid)[:, sidx])
        if "geom" in groups:
            blocks.append(store.geom(pid)["X_geom"])
        if "ctx" in groups:
            blocks.append(context_features(z[a:b], store.geom(pid)["cb"]))
        X[a:b] = np.concatenate(blocks, axis=1)
    X[~np.isfinite(X)] = np.nan
    return X, names


def fit_margin_model(Xtr, ytr, ztr, groups_tr, args, seed, rounds=None):
    """Returns (predict_margin_fn, n_rounds, importance or None)."""
    pw = float((len(ytr) - ytr.sum()) / max(1, ytr.sum()))
    if HAVE_LGB:
        params = dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves,
                      min_data_in_leaf=args.min_leaf, feature_fraction=args.feature_fraction,
                      bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, scale_pos_weight=pw,
                      max_bin=63, verbose=-1, seed=seed, num_threads=args.threads)
        if rounds is None:              # inner protein-level hold-out, taken from the fitting folds only
            prots = np.unique(groups_tr)
            rng = np.random.default_rng(seed)
            hold = set(rng.choice(prots, size=max(1, int(round(0.15 * len(prots)))), replace=False).tolist())
            va = np.isin(groups_tr, list(hold))
            dtr = lgb.Dataset(Xtr[~va], ytr[~va], init_score=ztr[~va], free_raw_data=False)
            dva = lgb.Dataset(Xtr[va], ytr[va], init_score=ztr[va], reference=dtr)
            bst = lgb.train(params, dtr, num_boost_round=args.max_rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(args.patience, verbose=False)])
            rounds = max(10, int(bst.best_iteration or args.max_rounds))
        bst = lgb.train(params, lgb.Dataset(Xtr, ytr, init_score=ztr), num_boost_round=rounds)
        return (lambda X: bst.predict(X, raw_score=True)), rounds, bst.feature_importance("gain")

    # Fallback without LightGBM: one functional Newton step fitted by a regression GBM.
    from sklearn.ensemble import HistGradientBoostingRegressor
    p = 1.0 / (1.0 + np.exp(-ztr))
    h = np.maximum(p * (1.0 - p), 1e-3)
    target = np.clip((ytr - p) / h, -10.0, 10.0)
    weight = np.where(ytr > 0, pw, 1.0) * h
    reg = HistGradientBoostingRegressor(max_iter=rounds or 200, learning_rate=0.05, max_leaf_nodes=args.leaves,
                                        min_samples_leaf=args.min_leaf, l2_regularization=10.0,
                                        early_stopping=False, random_state=seed)
    reg.fit(Xtr, target, sample_weight=weight)
    return reg.predict, rounds or 200, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--features", default="grid,geom,ctx", help="comma list of grid|summary|geom|ctx")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--min-leaf", type=int, default=100)
    ap.add_argument("--feature-fraction", type=float, default=0.3)
    ap.add_argument("--max-rounds", type=int, default=600)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    groups = [g.strip() for g in args.features.split(",") if g.strip()]
    bad = set(groups) - {"grid", "summary", "geom", "ctx"}
    if bad:
        raise SystemExit(f"unknown feature groups: {bad}")
    name = args.name or "gbm_resid_" + "_".join(groups)
    if not HAVE_LGB:
        print("[WARN] lightgbm not installed -> using the one-step sklearn fallback (pip install lightgbm is recommended)")

    ctx = make_ctx(args.config)
    store = Store(ctx)
    oof = load_preds(ctx, BASELINE, "oof")
    X, names = build_matrix(ctx, store, oof, groups)
    y = oof["y_true"].to_numpy(dtype=np.float64)
    z = oof["z"].to_numpy(dtype=np.float64)
    pid = oof["pdb_id"].to_numpy()
    fold = oof["fold"].to_numpy()
    print(f"[{name}] X={X.shape} groups={groups} backend={'lightgbm' if HAVE_LGB else 'sklearn-fallback'}")

    margin = np.zeros(len(oof))
    rounds_used = []
    for f in ctx.fold_names:
        tr, va = fold != f, fold == f
        for seed in args.seeds:
            pred, r, _ = fit_margin_model(X[tr], y[tr], z[tr], pid[tr], args, seed)
            margin[va] += pred(X[va]) / len(args.seeds)
            rounds_used.append(r)
        print(f"  [{f}] trees={rounds_used[-len(args.seeds):]} mean|margin|={np.abs(margin[va]).mean():.4f}", flush=True)

    cands = {"identity": z}
    for eta in ETAS:
        cands[f"eta={eta:g}"] = z + eta * margin

    final_rounds = int(np.median(rounds_used))
    state = {}

    def test_fn(config: str) -> np.ndarray:
        test = load_preds(ctx, BASELINE, "test")
        zt = test["z"].to_numpy(dtype=np.float64)
        if config == "identity":
            return zt
        Xt, _ = build_matrix(ctx, store, test, groups)
        mt = np.zeros(len(test))
        imp = np.zeros(len(names))
        for seed in args.seeds:                      # trained on ALL train OOF rows, fixed #trees
            pred, _, gain = fit_margin_model(X, y, z, pid, args, seed, rounds=final_rounds)
            mt += pred(Xt) / len(args.seeds)
            if gain is not None:
                imp += gain / len(args.seeds)
        state["imp"] = imp
        return zt + float(config.split("=")[1]) * mt

    run_grid_experiment(ctx, name, oof, cands, test_fn,
                        meta={"idea": "GBM residual boosting from pLM logit", "features": groups,
                              "n_features": len(names), "final_rounds": final_rounds, "seeds": args.seeds,
                              "backend": "lightgbm" if HAVE_LGB else "sklearn-fallback"})
    if "imp" in state and state["imp"].sum() > 0:
        (pd.DataFrame({"feature": names, "gain": state["imp"]}).sort_values("gain", ascending=False)
           .to_csv(exp_dir(ctx, name) / "feature_importance.csv", index=False))


if __name__ == "__main__":
    main()
