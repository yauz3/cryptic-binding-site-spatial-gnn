#!/usr/bin/env python3
"""
IDEA 3 - 3D spatial smoothing of pLM logits (binding sites are contiguous surface patches).

Hypothesis
----------
The pLM head scores every residue independently, so its top-ranked residues are
scattered and isolated false positives occupy the top 1 %. A residue whose 3D
neighbours are also scored highly is more likely to be a true site residue.

Equation (pLM stays the anchor; lambda = 0 is exactly the pLM)
---------------------------------------------------------------
    w_ij   = exp(-d_ij^2 / (2 sigma^2))        for CB-CB distance d_ij <= 3 sigma, j != i
    zbar_i = sum_j w_ij z_j / sum_j w_ij       (repeated `steps` times)
    z'_i   = z_i + lambda * (zbar_i - z_i)

Searched (nested on train OOF, criterion J = 0.5 AP + 0.5 BEDROC20)
    sigma in {3,4,5,6,8} A, lambda in {0.1..0.8}, steps in {1,2}; "identity" is the first candidate.
Frozen: the base pLM logits (results/exp/base_plm), coordinates, kernel form.
No trainable parameters -> negligible over-fitting risk.

Required ablation: sequence-only smoothing (--mode sequence: neighbours i-w..i+w along the
chain). If the 3D kernel does not beat it, the gain is not a coordinate effect.

Outputs: results/exp/<name>/{oof,test}_predictions.csv, config_search_oof.csv, selection.json
"""
from __future__ import annotations

import argparse

import numpy as np

from cryptogrid.exp_common import (BASELINE, Store, gaussian_smooth, load_preds, make_ctx,
                                   neighbor_pairs, protein_slices, run_grid_experiment)

SIGMAS = (3.0, 4.0, 5.0, 6.0, 8.0)
LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
STEPS = (1, 2)
WINDOWS = (1, 2, 3, 5)


def sequence_pairs(chain_index, seq_index, w):
    """Chain neighbours within +-w positions, encoded as pseudo-distances for the same kernel code."""
    n = len(chain_index)
    ii, jj, dd = [], [], []
    for off in range(1, w + 1):
        a = np.arange(n - off)
        ok = (chain_index[a] == chain_index[a + off]) & (seq_index[a + off] - seq_index[a] == off)
        a = a[ok]
        ii += [a, a + off]
        jj += [a + off, a]
        dd += [np.full(len(a), float(off))] * 2
    if not ii:
        e = np.zeros(0, dtype=int)
        return e, e, np.zeros(0)
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(dd)


def smoothed_tables(df, store, mode):
    """dict (scale, steps) -> smoothed logits for the whole frame."""
    z = df["z"].to_numpy(dtype=np.float64)
    scales = SIGMAS if mode == "3d" else WINDOWS
    out = {(s, t): np.empty_like(z) for s in scales for t in STEPS}
    for pid, a, b in protein_slices(df["pdb_id"].to_numpy()):
        g = store.geom(pid)
        if mode == "3d":
            pairs = neighbor_pairs(g["cb"].astype(np.float64), 3.0 * max(SIGMAS))
        for s in scales:
            if mode == "sequence":
                pairs = sequence_pairs(g["chain_index"], g["seq_index"], int(s))
                sig = float(s)            # kernel exp(-off^2 / 2w^2), cut at w (pairs only go to w)
            else:
                sig = float(s)
            cur = z[a:b]
            for t in range(1, max(STEPS) + 1):
                cur = gaussian_smooth(cur, pairs, sig)
                if t in STEPS:
                    out[(s, t)][a:b] = cur
    return out


def build_candidates(df, tabs):
    z = df["z"].to_numpy(dtype=np.float64)
    cands = {"identity": z}
    # conservative first: small lambda, small scale, one step
    for lam in LAMBDAS:
        for (s, t), zs in sorted(tabs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            cands[f"scale={s:g}|steps={t}|lam={lam:g}"] = z + lam * (zs - z)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--base", default=BASELINE, help="experiment whose logits are smoothed")
    ap.add_argument("--mode", choices=["3d", "sequence"], default="3d")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    store = Store(ctx)
    name = args.name or ("smooth3d" if args.mode == "3d" else "smoothseq_ablation")
    if args.base != BASELINE:
        name = f"{name}__on__{args.base}"

    oof = load_preds(ctx, args.base, "oof")
    cands = build_candidates(oof, smoothed_tables(oof, store, args.mode))

    def test_fn(config: str) -> np.ndarray:
        test = load_preds(ctx, args.base, "test")          # logits only; labels are not scored here
        return build_candidates(test, smoothed_tables(test, store, args.mode))[config]

    # The paired comparison is always against the pLM anchor.
    anchor = load_preds(ctx, BASELINE, "oof")
    run_grid_experiment(ctx, name, anchor, cands, test_fn,
                        meta={"idea": "spatial smoothing", "mode": args.mode, "input_logits": args.base})


if __name__ == "__main__":
    main()
