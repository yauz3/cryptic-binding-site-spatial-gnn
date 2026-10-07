#!/usr/bin/env python3
"""
IDEA 4 - Pocket-consensus fusion: use Grid pockets as NEIGHBOURHOODS, not as descriptors.

Hypothesis
----------
Averaging 620 pocket descriptors onto residues (mean/max) throws away what Grid-Crypto
is actually good at: proposing geometrically coherent residue sets. The pLM knows
*which residues* look functional, Grid knows *which residues belong together*.
So score each Grid pocket with the pLM, and feed the pocket score back to its members.

Equations (pLM stays the anchor; lambda = 0 or unsupported residue -> exactly the pLM)
----------------------------------------------------------------------------------------
    S_p  = pool_{i in pocket p} z_i            pool in {mean, log-mean-exp}
    c_i  = agg_{p containing i} S_p            agg  in {max, mean}
    z'_i = z_i + lambda * s_i * (c_i - z_i)    s_i = 1 if residue i is in any used pocket

Searched (nested on train OOF, J = 0.5 AP + 0.5 BEDROC20)
    pool x agg x grid-scale subset {all, small, large} x lambda {0.1..0.6}
Frozen: base pLM logits, pocket membership (label-free: only pocket_residues_list and
grid_size are read; overlap/jaccard columns are never loaded).

Required ablation: --shuffle-pockets (same pocket sizes, random residues). If shuffled
pockets do as well, the gain is generic smoothing, not Grid geometry.

Builds work/cache_pockets/<pid>.npz (CSR residue membership) on first use and checks it
against summary::pocket_membership_count in the struct cache.

Outputs: results/exp/<name>/{oof,test}_predictions.csv, config_search_oof.csv, selection.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from cryptogrid.cif_support import find_cif_file, parse_cif_residues, remap_residue_key
from cryptogrid.common import apo_chains, ensure_dir, find_feature_file, load_dataset_json, parse_residue_list
from cryptogrid.exp_common import (BASELINE, Store, load_preds, make_ctx, protein_slices, run_grid_experiment)

LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
POOLS = ("mean", "lme")
AGGS = ("max", "mean")
SCALES = ("all", "small", "large")


# ---------------------------------------------------------------------
# Pocket membership cache (label-free)
# ---------------------------------------------------------------------

def build_pocket_cache(ctx, store, overwrite=False):
    out = ensure_dir(ctx.pocket_dir)
    cfg = ctx.cfg
    pocket_col = cfg["features"].get("pocket_residue_column", "pocket_residues_list")
    dataset = load_dataset_json(cfg["project"]["dataset_json"])     # used ONLY for apo chain ids
    names = store.struct_feature_names()
    mem_idx = names.index("summary::pocket_membership_count") if "summary::pocket_membership_count" in names else None
    mismatches = 0
    for pid in ctx.train_ids + ctx.test_ids:
        dst = out / f"{pid}.npz"
        if dst.exists() and not overwrite:
            continue
        _, keys = store.labels(pid)
        ridx = {k: i for i, k in enumerate(keys)}
        ptr, idx, gsize = [0], [], []
        raw_count = np.zeros(len(keys))
        fp = find_feature_file(cfg["project"]["grid_features_dir"], pid)
        if fp is not None:
            df = pd.read_csv(fp, usecols=lambda c: c in {pocket_col, "grid_size"}, low_memory=False)
            chain_map = {}
            gp = ctx.geom_dir / f"{pid}.npz"
            if gp.exists():
                chain_map = json.loads(str(np.load(gp, allow_pickle=False)["chain_map"]))
            if not chain_map:
                cif = find_cif_file(cfg["project"].get("cif_root"), pid)
                if cif is not None:
                    chain_map = parse_cif_residues(cif)[2]
            chains = set(apo_chains(dataset, pid))
            grid = pd.to_numeric(df.get("grid_size", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy()
            for row, v in enumerate(df[pocket_col].tolist() if pocket_col in df else []):
                rs = parse_residue_list(v)
                if chain_map:
                    rs = [remap_residue_key(r, chain_map) for r in rs]
                if chains:
                    rs = [r for r in rs if r.split("_", 1)[0] in chains]
                hits = [ridx[r] for r in rs if r in ridx]
                if not hits:
                    continue
                np.add.at(raw_count, hits, 1.0)        # counted exactly like script 01
                members = sorted(set(hits))
                idx.extend(members)
                ptr.append(len(idx))
                gsize.append(grid[row])
        idx_a = np.asarray(idx, dtype=np.int32)
        if mem_idx is not None:                       # consistency with script 01
            ref = np.nan_to_num(store.struct(pid)[:, mem_idx], nan=0.0)
            if not np.array_equal(ref, raw_count):
                mismatches += 1
        np.savez_compressed(dst, ptr=np.asarray(ptr, dtype=np.int64), idx=idx_a,
                            grid_size=np.asarray(gsize, dtype=np.float32), n_residues=np.asarray(len(keys)))
    if mismatches:
        raise RuntimeError(f"pocket cache disagrees with summary::pocket_membership_count for {mismatches} proteins; "
                           "the chain remapping differs from script 01 - fix before using this experiment.")


def load_pockets(ctx, pid):
    z = np.load(ctx.pocket_dir / f"{pid}.npz", allow_pickle=False)
    return z["ptr"], z["idx"], z["grid_size"]


# ---------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------

def consensus(z, ptr, idx, keep, pool, agg):
    """Returns (c, support). Residues outside every kept pocket: c = z, support = 0."""
    n = len(z)
    sizes = np.diff(ptr)
    P = len(sizes)
    c, sup = z.copy(), np.zeros(n)
    if P == 0 or not keep.any():
        return c, sup
    pocket_of = np.repeat(np.arange(P), sizes)
    zi = z[idx]
    if pool == "mean":
        S = np.bincount(pocket_of, weights=zi, minlength=P) / sizes
    else:                                              # log-mean-exp: soft maximum over members
        m = np.full(P, -np.inf)
        np.maximum.at(m, pocket_of, zi)
        S = m + np.log(np.bincount(pocket_of, weights=np.exp(zi - m[pocket_of]), minlength=P) / sizes)
    use = keep[pocket_of]
    ri, sv = idx[use], S[pocket_of[use]]
    if agg == "max":
        acc = np.full(n, -np.inf)
        np.maximum.at(acc, ri, sv)
        ok = np.isfinite(acc)
    else:
        cnt = np.bincount(ri, minlength=n)
        ok = cnt > 0
        acc = np.bincount(ri, weights=sv, minlength=n) / np.maximum(cnt, 1)
    c[ok] = acc[ok]
    sup[ok] = 1.0
    return c, sup


def scale_mask(grid_size, scale):
    if scale == "all" or len(grid_size) == 0 or not np.isfinite(grid_size).any():
        return np.ones(len(grid_size), dtype=bool)
    med = np.nanmedian(np.unique(grid_size[np.isfinite(grid_size)]))
    return grid_size <= med if scale == "small" else grid_size >= med


def build_candidates(ctx, df, shuffle=False):
    z = df["z"].to_numpy(dtype=np.float64)
    corr = {(p, a, s): np.zeros_like(z) for p in POOLS for a in AGGS for s in SCALES}
    rng = np.random.default_rng(12345)
    for pid, a0, b0 in protein_slices(df["pdb_id"].to_numpy()):
        ptr, idx, gs = load_pockets(ctx, pid)
        if shuffle and len(idx):                      # ablation: destroy geometry, keep pocket sizes
            n = b0 - a0
            idx = np.concatenate([rng.choice(n, size=min(int(k), n), replace=False) for k in np.diff(ptr)])
            ptr = np.r_[0, np.cumsum(np.minimum(np.diff(ptr), n))]
        zz = z[a0:b0]
        for s in SCALES:
            keep = scale_mask(gs, s)
            for p in POOLS:
                for a in AGGS:
                    c, sup = consensus(zz, ptr, idx, keep, p, a)
                    corr[(p, a, s)][a0:b0] = sup * (c - zz)
    cands = {"identity": z}
    for lam in LAMBDAS:                                # conservative first
        for (p, a, s), dz in corr.items():
            cands[f"pool={p}|agg={a}|scale={s}|lam={lam:g}"] = z + lam * dz
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--base", default=BASELINE)
    ap.add_argument("--shuffle-pockets", action="store_true", help="ablation: random residue sets of equal size")
    ap.add_argument("--rebuild-pocket-cache", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    store = Store(ctx)
    build_pocket_cache(ctx, store, overwrite=args.rebuild_pocket_cache)
    name = args.name or ("pocket_consensus_shuffled_ablation" if args.shuffle_pockets else "pocket_consensus")
    if args.base != BASELINE:
        name = f"{name}__on__{args.base}"

    oof = load_preds(ctx, args.base, "oof")
    cands = build_candidates(ctx, oof, shuffle=args.shuffle_pockets)

    def test_fn(config: str) -> np.ndarray:
        return build_candidates(ctx, load_preds(ctx, args.base, "test"), shuffle=args.shuffle_pockets)[config]

    run_grid_experiment(ctx, name, load_preds(ctx, BASELINE, "oof"), cands, test_fn,
                        meta={"idea": "pocket consensus", "shuffled": bool(args.shuffle_pockets), "input_logits": args.base})


if __name__ == "__main__":
    main()
