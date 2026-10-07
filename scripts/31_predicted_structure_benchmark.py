#!/usr/bin/env python3
"""
Q1 EXPERIMENT E5 - Does the method still work on PREDICTED structures?

The anchored GNN consumes experimental apo coordinates, including crystallographic B-factors.
A reviewer will ask what happens on an AlphaFold model, because that is the setting in which a
cryptic-site predictor would actually be deployed on a new target. The Phase-I B-factor
ablation is encouraging ($-0.004$ AP on test) but it is not the experiment.

What this script does
---------------------
It rebuilds ONLY the geometry cache from predicted structures, keeping the labels, the residue
keys and the sequence embeddings of the real benchmark, so that the comparison isolates the
effect of swapping experimental coordinates for predicted ones. Predicted models are numbered
by UniProt position rather than by author numbering, so residues are mapped by global sequence
alignment against the cached sequence and the alignment identity is reported per protein.
pLDDT occupies the B-factor field of AlphaFold models; it is a confidence score rather than a
flexibility measure, and the script records which convention was used so the manuscript can
say so. Use --invert-plddt to feed (100 - pLDDT), which is the monotone transform that makes
low-confidence regions look flexible, and run both as a sensitivity analysis.

Inputs
------
--structures DIR   predicted models, named <pdb_id>.cif/.pdb (the same identifiers as the
                   benchmark split being evaluated)
--split            test (default) or oof

Output
------
<out-root>/cache_geom/<pid>.npz    geometry from predicted coordinates, residue order identical
                                   to the real struct cache
<out-root>/config_predicted.yaml   ready for scripts 11/15/25; struct and ESM caches still point
                                   at the originals, only work.root (hence cache_geom) changes
<out-root>/mapping_report.csv      per protein: alignment identity, residues mapped, mean pLDDT

Then
----
  python scripts/11_baseline_multiseed.py   --config <out>/config_predicted.yaml
  python scripts/15_anchored_spatial_gnn.py --config <out>/config_predicted.yaml
  python scripts/25_external_report.py      --config <out>/config_predicted.yaml --split test

Proteins without a predicted model keep NaN coordinates, which the spatial operators already
handle by leaving those residues untouched; they are counted in the report rather than dropped,
so the evaluation stays on the full split.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cryptogrid.common import ensure_dir, normalize_residue_token
from cryptogrid.exp_common import Store, make_ctx

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}


def load_geom_module(root: Path):
    spec = importlib.util.spec_from_file_location("g10", root / "scripts" / "10_build_geom_cache.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_predicted(path: Path):
    """-> (sequence, per-residue dict of atom coords, per-residue mean B/pLDDT), in file order."""
    from Bio.PDB import MMCIFParser, PDBParser
    p = (MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True))
    model = next(p.get_structure(path.stem, str(path)).get_models())
    seq, atoms, bfac = [], [], []
    for chain in model:
        for res in chain:
            if res.id[0].strip() not in {"", "H_MSE"}:
                continue
            name = res.get_resname().strip().upper()
            if name not in THREE_TO_ONE:
                continue
            at = {a.get_name().strip(): np.asarray(a.get_coord(), dtype=np.float64)
                  for a in res if a.element not in {"H", "D"}}
            if not at:
                continue
            bs = [float(a.get_bfactor()) for a in res]
            seq.append(THREE_TO_ONE[name])
            atoms.append(at)
            bfac.append(float(np.mean(bs)) if bs else np.nan)
    return "".join(seq), atoms, np.asarray(bfac, dtype=np.float64)


def _make_aligner():
    """Global aligner that does not penalise terminal gaps, across Biopython versions."""
    from Bio import Align
    a = Align.PairwiseAligner(mode="global", open_gap_score=-11, extend_gap_score=-1)
    for name in ("end_insertion_score", "target_end_gap_score"):
        try:
            setattr(a, name, 0.0)
            break
        except Exception:
            continue
    for name in ("end_deletion_score", "query_end_gap_score"):
        try:
            setattr(a, name, 0.0)
            break
        except Exception:
            continue
    try:
        from Bio.Align import substitution_matrices
        a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    except Exception:
        a.match_score, a.mismatch_score = 2, -1
    return a


def align_map(query: str, target: str):
    aln = _make_aligner().align(target, query)[0]
    mapping, same = {}, 0
    for (t0, t1), (q0, q1) in zip(aln.aligned[0], aln.aligned[1]):
        for off in range(t1 - t0):
            mapping[t0 + off] = q0 + off
            same += int(target[t0 + off] == query[q0 + off])
    return mapping, same / max(1, len(mapping))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--structures", required=True)
    ap.add_argument("--split", default="test", choices=["test", "oof", "both"])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--invert-plddt", action="store_true",
                    help="store (100 - pLDDT) in the B-factor slot, so low confidence reads as flexible")
    ap.add_argument("--min-identity", type=float, default=0.95)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    ctx = make_ctx(args.config, require_esm=False)
    store = Store(ctx)
    g10 = load_geom_module(root)
    out_root = ensure_dir(Path(args.out_root))
    out_geom = ensure_dir(out_root / "cache_geom")
    (out_geom / "feature_names.json").write_text(json.dumps(g10.FEATURES, indent=2), encoding="utf-8")

    files = {}
    for p in sorted(Path(args.structures).iterdir()):
        if p.suffix.lower() in {".cif", ".mmcif", ".pdb", ".ent"}:
            files.setdefault(p.stem.lower().split(".")[0], p)
    ids = (ctx.test_ids if args.split == "test" else ctx.train_ids if args.split == "oof"
           else ctx.train_ids + ctx.test_ids)
    print(f"[predicted] {len(files)} predicted models for {len(ids)} proteins in split '{args.split}'")

    report = []
    for ix, pid in enumerate(ids, 1):
        dst = out_geom / f"{pid}.npz"
        keys = store.labels(pid)[1]
        if dst.exists() and not args.overwrite:
            continue
        n = len(keys)
        ca = np.full((n, 3), np.nan)
        cb = np.full((n, 3), np.nan)
        b_ca = np.full(n, np.nan)
        b_res = np.full(n, np.nan)
        n_sc = np.zeros(n)
        status, ident, cov, mplddt = "ok", np.nan, 0.0, np.nan
        f = files.get(pid)
        if f is None:
            status = "no_predicted_model"
        else:
            z = np.load(ctx.struct_dir / f"{pid}.npz", allow_pickle=False)
            target = "".join(z["aa"].astype(str).tolist())
            try:
                qseq, atoms, bf = parse_predicted(f)
            except Exception as e:
                status = f"unreadable:{e}"
                qseq = ""
            if qseq:
                mapping, ident = align_map(qseq, target)
                cov = len(mapping) / max(1, n)
                mplddt = float(np.nanmean(bf)) if len(bf) else np.nan
                if ident < args.min_identity:
                    status = f"identity {ident:.2f} below {args.min_identity}"
                else:
                    bvals = (100.0 - bf) if args.invert_plddt else bf
                    for j, i in mapping.items():
                        r = {"atoms": atoms[i], "b": {k: bvals[i] for k in atoms[i]}}
                        ca[j], cb[j], b_ca[j], b_res[j], n_sc[j] = g10.residue_points(r)
        X, chain_index, seq_index = g10.build_features(keys, ca, cb, b_ca, b_res, n_sc)
        np.savez_compressed(dst, residue_keys=keys.astype("U"), ca=ca.astype(np.float32),
                            cb=cb.astype(np.float32), X_geom=X, chain_index=chain_index,
                            seq_index=seq_index, chain_map=np.asarray("{}"), pdb_id=np.asarray(pid))
        report.append({"pdb_id": pid, "status": status, "n_residues": n,
                       "alignment_identity": ident, "coverage": cov, "mean_plddt": mplddt,
                       "coord_coverage": float(np.isfinite(ca).all(axis=1).mean())})
        if ix % 25 == 0:
            print(f"  [{ix}/{len(ids)}]", flush=True)

    # Proteins outside the evaluated split keep their EXPERIMENTAL geometry: the deployment
    # scenario is a model trained on crystal structures and applied to a predicted one.
    src_geom = Path(ctx.cfg["work"].get("root", "work")) / "cache_geom"
    linked = 0
    for pid in ctx.train_ids + ctx.test_ids:
        if pid in set(ids):
            continue
        dst, src = out_geom / f"{pid}.npz", src_geom / f"{pid}.npz"
        if src.exists() and not dst.exists():
            try:
                import os
                os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
            linked += 1
    if linked:
        print(f"[predicted] {linked} proteins outside the evaluated split keep experimental geometry")

    rep = pd.DataFrame(report)
    if not rep.empty:
        rep.to_csv(out_root / "mapping_report.csv", index=False)
        bad = rep[rep["status"] != "ok"]
        print(f"[predicted] {len(rep) - len(bad)}/{len(rep)} proteins mapped; "
              f"mean coordinate coverage {rep['coord_coverage'].mean():.3f}")
        if len(bad):
            print(f"[predicted] {len(bad)} proteins without usable predicted coordinates "
                  "(kept with NaN geometry, never dropped):")
            print(bad[["pdb_id", "status"]].head(10).to_string(index=False))

    cfg = yaml.safe_load(Path(args.config).read_text())
    w = dict(cfg["work"])
    w["root"] = str(out_root.resolve())                       # geometry comes from here
    w["struct_cache"] = str(Path(w["struct_cache"]).resolve())  # labels unchanged
    w["esm_cache"] = str(Path(w["esm_cache"]).resolve())        # embeddings unchanged
    w["results"] = str((out_root / "results").resolve())
    cfg["work"] = w
    cfg_path = out_root / "config_predicted.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    print(f"\n[DONE] geometry from predicted structures -> {out_geom}")
    print(f"       config -> {cfg_path}")
    print(f"       B-factor slot holds {'100 - pLDDT' if args.invert_plddt else 'pLDDT as deposited'}; "
          "run both conventions as a sensitivity analysis.")
    print("\nNext:")
    print(f"  python scripts/11_baseline_multiseed.py   --config {cfg_path}")
    print(f"  python scripts/15_anchored_spatial_gnn.py --config {cfg_path}")
    print(f"  python scripts/25_external_report.py      --config {cfg_path} --split {args.split}")


if __name__ == "__main__":
    main()
