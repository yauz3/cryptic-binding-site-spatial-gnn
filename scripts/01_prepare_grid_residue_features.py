#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from cryptogrid.common import (
    load_config, ensure_dir, load_folds, split_ids, load_dataset_json,
    cryptic_truth, apo_chains, find_feature_file, find_pdb_file,
    parse_residue_list, parse_pdb_observed_residues, safe_numeric_frame,
    feature_family,
)

from cryptogrid.cif_support import (
    find_cif_file,
    parse_cif_residues,
    remap_residue_key,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

SUMMARY_FEATURES = [
    "summary::pocket_membership_count",
    "summary::unique_grid_sizes",
    "summary::grid_size_mean",
    "summary::grid_size_min",
    "summary::grid_size_max",
    "summary::center_x_std",
    "summary::center_y_std",
    "summary::center_z_std",
]


def scan_schema(proteins, features_dir, drop_cols):
    """
    Freeze a GLOBAL feature-name union. Deliberately do NOT delete all-NaN columns.
    This keeps the structural dimensionality stable across proteins/runs.
    """
    union = set()
    files = 0
    for pid in proteins:
        fp = find_feature_file(features_dir, pid)
        if fp is None:
            continue
        try:
            cols = pd.read_csv(fp, nrows=0).columns.tolist()
        except Exception as e:
            print(f"[WARN] schema read failed {fp}: {e}")
            continue
        files += 1
        union.update(c for c in cols if c not in drop_cols)
    raw = sorted(union)
    print(f"[schema] files={files} | frozen raw columns={len(raw)}")
    return raw


def safe_nan_reduce(vals, mode):
    if vals.size == 0:
        return np.array([], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode == "mean":
            return np.nanmean(vals, axis=0).astype(np.float32)
        if mode == "max":
            return np.nanmax(vals, axis=0).astype(np.float32)
        if mode == "min":
            return np.nanmin(vals, axis=0).astype(np.float32)
        if mode == "std":
            return np.nanstd(vals, axis=0).astype(np.float32)
    raise ValueError(mode)


def aggregate_one(pid, fp, pdb_root, cif_root, dataset, label_side, raw_cols, aggs, pocket_col):
    df = pd.read_csv(fp, low_memory=False)
    if pocket_col not in df.columns:
        raise ValueError(f"{fp}: missing required column {pocket_col!r}")

    chains = apo_chains(dataset, pid)

    # ---------------------------------------------------------
    # Official CryptoBench residue universe:
    # use mmCIF auth_asym_id/auth_seq_id whenever available.
    #
    # Grid-Crypto residues use label-chain IDs (A/B/C...) with
    # author residue numbering, while CryptoBench truth uses
    # auth-chain IDs (AAA/BBB/EEE...). The CIF mapping is thus
    # required before matching candidate residues to truth.
    # ---------------------------------------------------------

    residues, aas = [], []
    chain_map = {}

    cif_file = find_cif_file(cif_root, pid) if cif_root else None

    if cif_file is not None:
        residues, aas, chain_map = parse_cif_residues(
            cif_file,
            allowed_auth_chains=chains or None,
        )

    # Legacy fallback only when official CIF is unavailable.
    if not residues:
        pdb_file = find_pdb_file(pdb_root, pid)

        if pdb_file is not None:
            residues, aas = parse_pdb_observed_residues(
                pdb_file,
                allowed_chains=None,
            )

    pocket_res = [
        parse_residue_list(v)
        for v in df[pocket_col].tolist()
    ]

    # Convert Grid-Crypto label-chain namespace -> CryptoBench
    # auth-chain namespace. Residue number remains unchanged.
    if chain_map:
        pocket_res = [
            [
                remap_residue_key(r, chain_map)
                for r in rs
            ]
            for rs in pocket_res
        ]

    # Restrict to the official apo chain(s), now that both
    # sources use the same auth-chain namespace.
    if chains:
        chainset = set(chains)

        pocket_res = [
            [
                r for r in rs
                if r.split("_", 1)[0] in chainset
            ]
            for rs in pocket_res
        ]

    truth = cryptic_truth(dataset, pid, label_side)
    if chains:
        truth = {r for r in truth if r.split("_", 1)[0] in set(chains)}

    # Fallback if a PDB cannot be parsed: union of observed candidate residues + truth.
    if not residues:
        seen, universe = set(), []
        for rs in pocket_res:
            for r in rs:
                if r not in seen:
                    seen.add(r); universe.append(r)
        for r in sorted(truth):
            if r not in seen:
                seen.add(r); universe.append(r)
        residues, aas = universe, ["X"] * len(universe)

    ridx = {r: i for i, r in enumerate(residues)}
    memberships = [[] for _ in residues]
    for j, rs in enumerate(pocket_res):
        for r in rs:
            i = ridx.get(r)
            if i is not None:
                memberships[i].append(j)

    # Critical: reindex to frozen schema. Missing columns remain NaN rather than disappearing.
    Xraw = safe_numeric_frame(df, raw_cols).to_numpy(dtype=np.float32)

    grid = pd.to_numeric(df.get("grid_size", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(float)
    center = {}
    for c in ("center_x", "center_y", "center_z"):
        center[c] = pd.to_numeric(df.get(c, pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(float)

    n = len(residues)
    blocks, names = [], []
    for agg in aggs:
        B = np.full((n, len(raw_cols)), np.nan, dtype=np.float32)
        for i, js in enumerate(memberships):
            if js:
                B[i] = safe_nan_reduce(Xraw[js, :], agg)
        blocks.append(B)
        names.extend([f"{agg}::{c}" for c in raw_cols])

    S = np.full((n, len(SUMMARY_FEATURES)), np.nan, dtype=np.float32)
    for i, js in enumerate(memberships):
        if not js:
            S[i, 0] = 0.0
            S[i, 1] = 0.0
            continue
        g = grid[js]
        gv = g[np.isfinite(g)]
        S[i, 0] = float(len(js))
        S[i, 1] = float(len(np.unique(gv))) if len(gv) else 0.0
        if len(gv):
            S[i, 2:5] = [float(np.mean(gv)), float(np.min(gv)), float(np.max(gv))]
        for k, c in enumerate(("center_x", "center_y", "center_z"), start=5):
            z = center[c][js]
            z = z[np.isfinite(z)]
            S[i, k] = float(np.std(z)) if len(z) > 1 else (0.0 if len(z) == 1 else np.nan)
    blocks.append(S)
    names.extend(SUMMARY_FEATURES)

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    y = np.asarray([1 if r in truth else 0 for r in residues], dtype=np.int8)
    return X, y, np.asarray(residues), np.asarray(aas), names, len(df), len(truth), chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    train, test, _ = split_ids(load_folds(cfg["project"]["folds_json"]))
    proteins = sorted(set(train + test))
    dataset = load_dataset_json(cfg["project"]["dataset_json"])

    features_dir = cfg["project"]["grid_features_dir"]
    pdb_root = cfg["project"]["pdb_root"]
    cif_root = cfg["project"].get("cif_root")
    outdir = ensure_dir(cfg["work"]["struct_cache"])
    aggs = list(cfg["features"].get("aggregations", ["mean", "max"]))
    drop_cols = set(cfg["features"].get("drop_columns", []))
    pocket_col = cfg["features"].get("pocket_residue_column", "pocket_residues_list")
    label_side = cfg["evaluation"].get("label_side", "apo")

    raw_cols = scan_schema(proteins, features_dir, drop_cols)
    if not raw_cols:
        raise RuntimeError(f"No structural columns found under {features_dir}")
    (outdir / "raw_feature_schema.json").write_text(json.dumps(raw_cols, indent=2), encoding="utf-8")

    final_names = [f"{a}::{c}" for a in aggs for c in raw_cols] + SUMMARY_FEATURES
    (outdir / "feature_names.json").write_text(json.dumps(final_names, indent=2), encoding="utf-8")
    pd.DataFrame({"feature": final_names, "family": [feature_family(x) for x in final_names]}).to_csv(
        outdir / "feature_families.csv", index=False
    )

    manifest = []
    for ix, pid in enumerate(proteins, 1):
        out = outdir / f"{pid}.npz"
        if out.exists() and not args.overwrite:
            try:
                z = np.load(out, allow_pickle=False)
                if z["X_struct"].shape[1] == len(final_names):
                    manifest.append({"pdb_id": pid, "status": "cached", "n_residues": len(z["y"]),
                                     "positives": int(z["y"].sum()), "n_struct_features": z["X_struct"].shape[1]})
                    print(f"[{ix}/{len(proteins)}] {pid}: cached {z['X_struct'].shape}")
                    continue
            except Exception:
                pass

        fp = find_feature_file(features_dir, pid)
        if fp is None:
            manifest.append({"pdb_id": pid, "status": "missing_feature_csv"})
            print(f"[{ix}/{len(proteins)}] {pid}: missing feature CSV")
            continue
        try:
            X, y, residues, aas, names, npockets, ntruth, chains = aggregate_one(
                pid, fp, pdb_root, cif_root, dataset, label_side, raw_cols, aggs, pocket_col
            )
            if X.shape[1] != len(final_names) or names != final_names:
                raise RuntimeError(f"Feature-schema mismatch: {X.shape[1]} vs {len(final_names)}")
            np.savez_compressed(out, X_struct=X, y=y, residue_keys=residues.astype("U"), aa=aas.astype("U1"), pdb_id=np.asarray(pid))
            manifest.append({"pdb_id": pid, "status": "ok", "n_residues": len(y), "positives": int(y.sum()),
                             "n_pockets": npockets, "n_truth_residues": ntruth,
                             "n_struct_features": X.shape[1], "apo_chains": "-".join(chains), "source_file": str(fp)})
            print(f"[{ix}/{len(proteins)}] {pid}: residues={len(y)} pos={int(y.sum())} features={X.shape[1]}")
        except Exception as e:
            manifest.append({"pdb_id": pid, "status": "error", "error": repr(e), "source_file": str(fp)})
            print(f"[{ix}/{len(proteins)}] {pid}: ERROR {e!r}")

    pd.DataFrame(manifest).to_csv(outdir / "manifest.csv", index=False)
    print(f"[DONE] frozen structural dimension = {len(final_names)} -> {outdir}")


if __name__ == "__main__":
    main()
