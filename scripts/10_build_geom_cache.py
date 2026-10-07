#!/usr/bin/env python3
"""
IDEA 1 - Coordinate cache + label-free geometric / flexibility residue features.

Hypothesis
----------
The current pipeline never sees coordinates: Grid descriptors are pocket-level
statistics averaged onto residues, and the classifier is a per-residue MLP.
Cryptic sites are tied to local flexibility and burial, which can be read
directly from the apo structure (B-factors, contact density, half-sphere
exposure, chain breaks next to unresolved loops).

What it writes (no labels are read in this script)
--------------------------------------------------
work/cache_geom/<pid>.npz
    residue_keys  identical order to work/cache_struct/<pid>.npz
    ca, cb        (n,3) float32 coordinates (virtual CB for Gly), NaN if unresolved
    X_geom        (n,d) float32 features (NaN allowed)
    chain_index   (n,)  int   chain id per residue
    seq_index     (n,)  int   position inside its chain (observed residues)
    chain_map     JSON  label_asym_id -> auth_asym_id
work/cache_geom/feature_names.json, manifest.csv

Consumers: 12 (smoothing), 13 (pocket consensus), 14 (GBM), 15 (GNN), 17 (aux task).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from cryptogrid.cif_support import find_cif_file
from cryptogrid.common import ensure_dir, normalize_residue_token
from cryptogrid.exp_common import make_ctx, neighbor_pairs

BACKBONE = {"N", "CA", "C", "O", "OXT"}
FEATURES = [
    "cn6", "cn8", "cn10", "cn12", "cn16", "shell_ratio_8_16", "wcn",
    "hse_up13", "hse_down13", "hse_up_frac",
    "centroid_dist_over_rg", "protrusion10", "sidechain_inward10",
    "long_range_frac10", "log_mean_seqsep10",
    "b_ca_z", "b_res_z", "b_rank", "b_nb10_z", "b_contrast", "b_available",
    "rel_pos", "term_dist", "gap_adjacent", "dist_to_gap",
    "ca_angle_cos", "ca_torsion_sin", "ca_torsion_cos",
    "n_sidechain_atoms", "coords_available",
]


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def parse_cif_atoms(cif_path: Path):
    """Heavy atoms per residue (first model, first alt-loc), keyed like cif_support.parse_cif_residues."""
    d = MMCIF2Dict(str(cif_path))
    g = lambda k: _as_list(d.get(f"_atom_site.{k}"))
    group, lchain, achain = g("group_PDB"), g("label_asym_id"), g("auth_asym_id")
    aseq, ins, atom = g("auth_seq_id"), g("pdbx_PDB_ins_code"), g("label_atom_id")
    xs, ys, zs, bf = g("Cartn_x"), g("Cartn_y"), g("Cartn_z"), g("B_iso_or_equiv")
    model, elem = g("pdbx_PDB_model_num"), g("type_symbol")
    n = min(len(v) for v in (group, lchain, achain, aseq, atom, xs, ys, zs) if len(v))
    first_model = model[0] if model else None
    res, chain_map = {}, {}
    for i in range(n):
        if str(group[i]).upper() != "ATOM":
            continue
        if first_model is not None and i < len(model) and model[i] != first_model:
            continue
        lc, ac = str(lchain[i]).strip(), str(achain[i]).strip()
        if ac in {".", "?", ""}:
            ac = lc
        chain_map.setdefault(lc, ac)
        resi = str(aseq[i]).strip()
        if resi in {"", ".", "?"}:
            continue
        ic = str(ins[i]).strip() if i < len(ins) else ""
        ic = "" if ic in {"", ".", "?"} else ic
        key = normalize_residue_token(f"{ac}_{resi}{ic}")
        if not key:
            continue
        if i < len(elem) and str(elem[i]).strip().upper() in {"H", "D"}:
            continue
        name = str(atom[i]).strip().strip('"')
        r = res.setdefault(key, {"atoms": {}, "b": {}})
        if name in r["atoms"]:
            continue
        try:
            r["atoms"][name] = (float(xs[i]), float(ys[i]), float(zs[i]))
            r["b"][name] = float(bf[i]) if i < len(bf) and bf[i] not in {".", "?"} else np.nan
        except ValueError:
            continue
    return res, chain_map


def residue_points(r):
    at = {k: np.asarray(v, dtype=np.float64) for k, v in r["atoms"].items()}
    allxyz = np.stack(list(at.values())) if at else np.full((1, 3), np.nan)
    ca = at.get("CA", allxyz.mean(axis=0))
    if "CB" in at:
        cb = at["CB"]
    elif {"N", "CA", "C"} <= set(at):
        b, c = at["CA"] - at["N"], at["C"] - at["CA"]
        cb = -0.58273431 * np.cross(b, c) + 0.56802827 * b - 0.54067466 * c + at["CA"]
    else:
        cb = ca
    bvals = np.array(list(r["b"].values()), dtype=np.float64)
    b_ca = r["b"].get("CA", np.nan)
    b_res = float(np.nanmean(bvals)) if np.isfinite(bvals).any() else np.nan
    n_sc = sum(1 for k in at if k not in BACKBONE)
    return ca, cb, b_ca, b_res, n_sc


def zscore(v):
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.full_like(v, np.nan), False
    sd = np.nanstd(v[ok])
    if sd < 1e-6:                      # NMR / models without real B-factors
        return np.zeros_like(v), False
    return (v - np.nanmean(v[ok])) / sd, True


def resnum_int(key: str) -> int:
    m = re.search(r"_(-?\d+)", key)
    return int(m.group(1)) if m else 0


def build_features(keys, ca, cb, b_ca, b_res, n_sc):
    n = len(keys)
    F = {k: np.full(n, np.nan) for k in FEATURES}
    valid = np.isfinite(cb).all(axis=1) & np.isfinite(ca).all(axis=1)
    F["coords_available"] = valid.astype(float)
    F["n_sidechain_atoms"] = n_sc.astype(float)

    chains = np.array([k.split("_", 1)[0] for k in keys])
    uniq = {c: i for i, c in enumerate(dict.fromkeys(chains.tolist()))}
    chain_index = np.array([uniq[c] for c in chains], dtype=np.int32)
    seq_index = np.zeros(n, dtype=np.int32)
    resnum = np.array([resnum_int(k) for k in keys])

    # ---- sequence-local descriptors (per chain, observed residues) ----------
    gap_adj = np.zeros(n)
    for c in range(len(uniq)):
        idx = np.flatnonzero(chain_index == c)
        L = len(idx)
        seq_index[idx] = np.arange(L)
        F["rel_pos"][idx] = np.arange(L) / max(1, L - 1)
        F["term_dist"][idx] = np.minimum(np.minimum(np.arange(L), L - 1 - np.arange(L)), 50) / 50.0
        if L > 1:
            step = np.linalg.norm(ca[idx[1:]] - ca[idx[:-1]], axis=1)
            brk = (np.diff(resnum[idx]) > 1) | (np.nan_to_num(step, nan=0.0) > 4.3)
            gap_adj[idx[:-1]] = np.maximum(gap_adj[idx[:-1]], brk)
            gap_adj[idx[1:]] = np.maximum(gap_adj[idx[1:]], brk)
        gpos = np.flatnonzero(gap_adj[idx] > 0)
        if len(gpos):
            dist = np.abs(np.arange(L)[:, None] - gpos[None, :]).min(axis=1)
        else:
            dist = np.full(L, 30)
        F["dist_to_gap"][idx] = np.minimum(dist, 30) / 30.0
        # CA-trace angle / pseudo-torsion (secondary-structure proxy)
        for a in range(2, L - 2):
            p = ca[idx[a - 2]], ca[idx[a]], ca[idx[a + 2]]
            v1, v2 = p[0] - p[1], p[2] - p[1]
            den = np.linalg.norm(v1) * np.linalg.norm(v2)
            if np.isfinite(den) and den > 0:
                F["ca_angle_cos"][idx[a]] = float(np.dot(v1, v2) / den)
        for a in range(1, L - 2):
            p0, p1, p2, p3 = (ca[idx[a + o]] for o in (-1, 0, 1, 2))
            b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
            nb1 = np.linalg.norm(b1)
            if not np.isfinite(nb1) or nb1 == 0:
                continue
            b1u = b1 / nb1
            v, w = b0 - np.dot(b0, b1u) * b1u, b2 - np.dot(b2, b1u) * b1u
            x, y = np.dot(v, w), np.dot(np.cross(b1u, v), w)
            if np.isfinite(x) and np.isfinite(y) and (x != 0 or y != 0):
                ang = np.arctan2(y, x)
                F["ca_torsion_sin"][idx[a]], F["ca_torsion_cos"][idx[a]] = np.sin(ang), np.cos(ang)
    F["gap_adjacent"] = gap_adj

    # ---- contact density / exposure -----------------------------------------
    i, j, dd = neighbor_pairs(cb, 16.0)
    for r in (6, 8, 10, 12, 16):
        cnt = np.bincount(i[dd <= r], minlength=n).astype(float)
        F[f"cn{r}"] = np.where(valid, cnt, np.nan)
    F["shell_ratio_8_16"] = F["cn8"] / (F["cn16"] + 1.0)
    F["wcn"] = np.where(valid, np.bincount(i, weights=1.0 / np.maximum(dd, 1.0) ** 2, minlength=n), np.nan)

    u = cb - ca
    un = np.linalg.norm(u, axis=1, keepdims=True)
    u = np.where(un > 1e-6, u / np.maximum(un, 1e-6), 0.0)
    ia, ja, da = neighbor_pairs(ca, 13.0)
    if len(ia):
        up = (np.einsum("ij,ij->i", ca[ja] - ca[ia], u[ia]) > 0).astype(float)
        hu = np.bincount(ia, weights=up, minlength=n)
        hd = np.bincount(ia, weights=1.0 - up, minlength=n)
        F["hse_up13"] = np.where(valid, hu, np.nan)
        F["hse_down13"] = np.where(valid, hd, np.nan)
        F["hse_up_frac"] = np.where(valid, hu / (hu + hd + 1.0), np.nan)

    if valid.sum() >= 3:
        cen = cb[valid].mean(axis=0)
        rg = np.sqrt(((cb[valid] - cen) ** 2).sum(axis=1).mean())
        F["centroid_dist_over_rg"] = np.linalg.norm(cb - cen, axis=1) / max(rg, 1e-6)

    k10 = dd <= 10.0
    i10, j10 = i[k10], j[k10]
    if len(i10):
        cnt = np.maximum(np.bincount(i10, minlength=n), 1)
        mvec = np.stack([np.bincount(i10, weights=(cb[j10] - cb[i10])[:, a], minlength=n) for a in range(3)], axis=1)
        mvec = mvec / cnt[:, None]
        mnorm = np.linalg.norm(mvec, axis=1)
        F["protrusion10"] = np.where(valid, mnorm, np.nan)
        F["sidechain_inward10"] = np.where(valid, np.einsum("ij,ij->i", u, mvec / np.maximum(mnorm, 1e-6)[:, None]), np.nan)
        same = chain_index[i10] == chain_index[j10]
        sep = np.where(same, np.abs(seq_index[i10] - seq_index[j10]), 500)
        F["long_range_frac10"] = np.where(valid, np.bincount(i10, weights=(sep > 12).astype(float), minlength=n) / cnt, np.nan)
        F["log_mean_seqsep10"] = np.where(valid, np.log1p(np.bincount(i10, weights=sep.astype(float), minlength=n) / cnt), np.nan)

    # ---- flexibility (B-factor), normalised inside each protein --------------
    bz, has_b = zscore(b_ca)
    brz, _ = zscore(b_res)
    F["b_ca_z"], F["b_res_z"] = bz, brz
    F["b_available"] = np.full(n, float(has_b))
    okb = np.isfinite(b_ca)
    rank = np.full(n, np.nan)
    if okb.sum() > 1:
        rank[okb] = (np.argsort(np.argsort(b_ca[okb])) + 0.5) / okb.sum()
    F["b_rank"] = rank
    if len(i10):
        bj = np.nan_to_num(bz[j10], nan=0.0)
        nb = np.bincount(i10, weights=bj, minlength=n) / np.maximum(np.bincount(i10, minlength=n), 1)
        F["b_nb10_z"] = np.where(valid, nb, np.nan)
        F["b_contrast"] = bz - F["b_nb10_z"]

    X = np.stack([F[k] for k in FEATURES], axis=1).astype(np.float32)
    return X, chain_index, seq_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ctx = make_ctx(args.config, require_esm=False)   # geometry does not depend on embeddings
    cif_root = ctx.cfg["project"].get("cif_root")
    out = ensure_dir(ctx.geom_dir)
    (out / "feature_names.json").write_text(json.dumps(FEATURES, indent=2), encoding="utf-8")
    ids = ctx.train_ids + ctx.test_ids
    manifest = []

    for ix, pid in enumerate(ids, 1):
        dst = out / f"{pid}.npz"
        s = np.load(ctx.struct_dir / f"{pid}.npz", allow_pickle=False)
        keys = s["residue_keys"].astype(str)          # labels (s["y"]) are deliberately not read
        if dst.exists() and not args.overwrite:
            z = np.load(dst, allow_pickle=False)
            if np.array_equal(z["residue_keys"].astype(str), keys) and z["X_geom"].shape[1] == len(FEATURES):
                manifest.append({"pdb_id": pid, "status": "cached", "n_residues": len(keys),
                                 "coord_coverage": float(np.isfinite(z["ca"]).all(axis=1).mean())})
                continue
        n = len(keys)
        ca = np.full((n, 3), np.nan)
        cb = np.full((n, 3), np.nan)
        b_ca, b_res, n_sc = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
        chain_map, status = {}, "ok"
        cif = find_cif_file(cif_root, pid) if cif_root else None
        if cif is None:
            status = "missing_cif"
        else:
            res, chain_map = parse_cif_atoms(cif)
            for i, k in enumerate(keys):
                r = res.get(k)
                if r is None or not r["atoms"]:
                    continue
                ca[i], cb[i], b_ca[i], b_res[i], n_sc[i] = residue_points(r)
        X, chain_index, seq_index = build_features(keys, ca, cb, b_ca, b_res, n_sc)
        np.savez_compressed(dst, residue_keys=keys.astype("U"), ca=ca.astype(np.float32), cb=cb.astype(np.float32),
                            X_geom=X, chain_index=chain_index, seq_index=seq_index,
                            chain_map=np.asarray(json.dumps(chain_map)), pdb_id=np.asarray(pid))
        cov = float(np.isfinite(ca).all(axis=1).mean())
        manifest.append({"pdb_id": pid, "status": status, "n_residues": n, "coord_coverage": cov,
                         "has_bfactor": bool(X[0, FEATURES.index("b_available")] > 0) if n else False})
        print(f"[{ix}/{len(ids)}] {pid}: residues={n} coord_coverage={cov:.3f} status={status}")

    m = pd.DataFrame(manifest)
    m.to_csv(out / "manifest.csv", index=False)
    low = m[m["coord_coverage"] < 0.999]
    print(f"\n[DONE] {len(m)} proteins -> {out} | {len(FEATURES)} features")
    print(f"proteins with coordinate coverage < 100%: {len(low)}")
    if len(low):
        print(low.head(15).to_string(index=False))
        print("These proteins are KEPT; residues without coordinates get NaN geometry and are left untouched "
              "by the spatial methods.")


if __name__ == "__main__":
    main()
