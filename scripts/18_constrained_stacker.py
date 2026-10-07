#!/usr/bin/env python3
"""
IDEA 9 - Constrained, pLM-anchored stacking of all structural corrections.

Hypothesis
----------
Smoothing, pocket consensus, GBM residual, GNN and ranking-loss heads make partly different
errors. A stacker that can only ADD non-negative, shrunken amounts of each correction to the
pLM logit keeps the anchor intact while pooling the complementary parts.

Equation
--------
    c_k  = (z_k - z_plm) / sd_k                    correction proposed by experiment k
    z'   = z_plm + b + sum_k a_k c_k ,   a_k >= 0
    (b, a) = argmin  weighted log-loss(y, z') + lam * sum_k a_k^2        (L-BFGS-B, convex)

    a = 0  -> exactly the pLM;  lam large -> shrinks every correction toward zero.

Searched: lam in {1, 0.3, 0.1, 0.03, 0.01, 0.001, 0}, chosen by J = 0.5 AP + 0.5 BEDROC20 with
DOUBLE nesting: for official fold f, lam is chosen by cross-fitting inside the other three
folds, the stacker is refitted on those three folds and applied to f.
Test: lam from 4-fold cross-fitting, weights fitted on all train OOF rows, then frozen.

Inputs: by default every experiment that has OOF and test predictions, except the baseline,
other stackers and runs whose name contains 'ablation'. Override with --inputs.
Report: the weight table (which corrections survive the non-negativity constraint) is the
ablation - a weight of exactly 0 means that source adds nothing beyond the others.

Outputs: results/exp/<name>/{oof,test}_predictions.csv, stack_weights.csv, selection.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from cryptogrid.exp_common import (BASELINE, KEYS, canonical_frame, criterion_from_arrays, exp_dir,
                                   list_experiments, load_preds, make_ctx, oof_report, save_preds, write_json)

LAMS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.001, 0.0)       # conservative first


def fit_stack(C, z0, y, lam):
    pw = (len(y) - y.sum()) / max(1.0, y.sum())
    w = np.where(y > 0, pw, 1.0)
    w = w / w.sum()

    def f(theta):
        z = z0 + theta[0] + C @ theta[1:]
        loss = float((w * (np.logaddexp(0.0, z) - y * z)).sum() + lam * (theta[1:] ** 2).sum())
        g = w * (expit(z) - y)
        return loss, np.r_[g.sum(), C.T @ g + 2.0 * lam * theta[1:]]

    k = C.shape[1]
    res = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B",
                   bounds=[(None, None)] + [(0.0, None)] * k, options={"maxiter": 500})
    return res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--name", default="stack_anchor")
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    base = load_preds(ctx, BASELINE, "oof")
    inputs = args.inputs or [e for e in list_experiments(ctx, need_test=True)
                             if e != BASELINE and not e.startswith("stack") and "ablation" not in e]
    if not inputs:
        raise SystemExit("no input experiments found - run scripts 12-17 first")
    print(f"[{args.name}] inputs: {inputs}")

    z0 = base["z"].to_numpy(dtype=np.float64)
    y = base["y_true"].to_numpy(dtype=np.float64)
    pid = base["pdb_id"].to_numpy()
    fold = base["fold"].to_numpy()
    D = np.stack([load_preds(ctx, e, "oof")["z"].to_numpy(dtype=np.float64) - z0 for e in inputs], axis=1)

    def scaled(rows):
        sd = D[rows].std(axis=0)
        return np.where(sd > 1e-8, sd, 1.0)

    def crossfit(folds, lam):
        """Cross-fitted stacked logits restricted to `folds` (each fold predicted from the others)."""
        out = np.full(len(z0), np.nan)
        for g in folds:
            tr = np.isin(fold, [h for h in folds if h != g])
            sd = scaled(tr)
            th = fit_stack(D[tr] / sd, z0[tr], y[tr], lam)
            va = fold == g
            out[va] = z0[va] + th[0] + (D[va] / sd) @ th[1:]
        return out

    def pick_lam(folds):
        rows = np.isin(fold, folds)
        scores = [criterion_from_arrays(pid[rows], y[rows], crossfit(folds, lam)[rows]) for lam in LAMS]
        return LAMS[int(np.argmax(scores))], scores

    # ---- nested OOF ---------------------------------------------------------
    z_nested = np.empty(len(z0))
    per_fold = {}
    for f in ctx.fold_names:
        others = [g for g in ctx.fold_names if g != f]
        lam, _ = pick_lam(others)
        tr, va = np.isin(fold, others), fold == f
        sd = scaled(tr)
        th = fit_stack(D[tr] / sd, z0[tr], y[tr], lam)
        z_nested[va] = z0[va] + th[0] + (D[va] / sd) @ th[1:]
        per_fold[f] = {"lam": lam, "weights": dict(zip(inputs, np.round(th[1:], 5).tolist()))}
        print(f"  [{f}] lam={lam:g} weights={per_fold[f]['weights']}")
    oof = base[KEYS + ["y_true", "fold"]].copy()
    oof["z"] = z_nested
    save_preds(ctx, args.name, "oof", oof)

    # ---- final model (train OOF only) -> test logits ---------------------------
    lam, scores = pick_lam(ctx.fold_names)
    sd = scaled(np.ones(len(z0), dtype=bool))
    th = fit_stack(D / sd, z0, y, lam)
    tb = load_preds(ctx, BASELINE, "test")["z"].to_numpy(dtype=np.float64)
    Dt = np.stack([load_preds(ctx, e, "test")["z"].to_numpy(dtype=np.float64) - tb for e in inputs], axis=1)
    test = canonical_frame(ctx, "test")
    test["z"] = tb + th[0] + (Dt / sd) @ th[1:]
    save_preds(ctx, args.name, "test", test)                 # written, never scored here

    d = exp_dir(ctx, args.name)
    pd.DataFrame({"input": inputs, "weight_scaled": th[1:], "sd_correction": sd,
                  "weight_per_logit": th[1:] / sd}).to_csv(d / "stack_weights.csv", index=False)
    write_json(d / "selection.json", {"experiment": args.name, "inputs": inputs, "lam_for_test": lam,
                                      "lam_grid": LAMS, "J_crossfit_by_lam": scores, "per_fold": per_fold,
                                      "intercept": th[0], "selected_on": "train OOF only (double nested)"})
    print(f"[{args.name}] final lam={lam:g} weights={dict(zip(inputs, np.round(th[1:], 4).tolist()))}")
    oof_report(ctx, args.name, base_oof=base, new_oof=oof)


if __name__ == "__main__":
    main()
