#!/usr/bin/env python3
"""
Convert external predictor output into the format scripts/26 expects:

    pdb_id, residue_key, score[, prediction][, method]

`residue_key` is <auth_asym_id>_<auth_seq_id><insertion code>, the same namespace as
work/cache_struct. Conversions are checked against the struct cache and the script FAILS
LOUDLY on any length or key mismatch rather than writing a silently misaligned file, because
a one-residue offset would quietly invalidate every head-to-head number.

Supported inputs
----------------
--format p2rank       P2Rank `*_residues.csv` (one file per structure). Columns are matched
                      case-insensitively: chain, residue_label (or residue/resi), and the
                      first of probability / score / zscore that is present.
--format pocketminer  One array per structure (.npy or .csv) of per-residue scores in the
                      order PocketMiner emitted them, plus --order-from so that the residue
                      keys can be recovered. Two ordering conventions are supported:
                        --order-from cache      the struct-cache residue order (use when the
                                                tool was run on the same cleaned structures)
                        --order-from structure  the order of observed standard residues in
                                                the structure file (default)
                      Lengths must match exactly; otherwise the structure is reported and
                      skipped, and the skipped list is written next to the output.
--format generic      A CSV that already has pdb_id/residue_key/score under other names;
                      give them with --col-pdb, --col-key, --col-score.

Usage
-----
  python scripts/external/convert_predictions.py --format p2rank \
      --input external_runs/p2rank_out --config config.yaml \
      --method P2Rank --out external_predictions/p2rank.csv

  python scripts/external/convert_predictions.py --format pocketminer \
      --input external_runs/pocketminer_npy --structures external_data/cryptobench_cifs \
      --config config.yaml --method PocketMiner --out external_predictions/pocketminer.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cryptogrid.common import load_config, normalize_residue_token

THREE_TO_ONE_KEEP = {"HOH", "DOD", "WAT"}


def cache_keys(struct_dir: Path, pid: str):
    f = struct_dir / f"{pid}.npz"
    if not f.exists():
        return None
    return np.load(f, allow_pickle=False)["residue_keys"].astype(str)


def structure_keys(path: Path):
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.Polypeptide import is_aa
    p = (MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True))
    model = next(p.get_structure(path.stem, str(path)).get_models())
    out = []
    for chain in model:
        for res in chain:
            het, seq, icode = res.id
            if het.strip() not in {"", "H_MSE"} or res.get_resname().strip().upper() in THREE_TO_ONE_KEEP:
                continue
            if not is_aa(res, standard=True) and res.get_resname().strip().upper() != "MSE":
                continue
            ic = "" if icode in {" ", "", "?"} else icode.strip()
            k = normalize_residue_token(f"{chain.id.strip()}_{seq}{ic}")
            if k:
                out.append(k)
    return np.array(out)


def read_scores(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=float).ravel()
    df = pd.read_csv(path)
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        raise ValueError(f"{path.name}: no numeric column")
    return df[num[-1]].to_numpy(dtype=float)


def convert_p2rank(files, struct_dir):
    rows, skipped = [], []
    for f in files:
        pid = f.stem.split("_residues")[0].split(".")[0].lower()
        df = pd.read_csv(f, skipinitialspace=True)
        df.columns = [c.strip().lower() for c in df.columns]
        ch = next((c for c in ("chain", "chain_id") if c in df.columns), None)
        rs = next((c for c in ("residue_label", "residue", "resi", "residue_number") if c in df.columns), None)
        sc = next((c for c in ("probability", "score", "zscore") if c in df.columns), None)
        if not (ch and rs and sc):
            skipped.append((pid, f"columns {list(df.columns)[:6]}"))
            continue
        keys = [normalize_residue_token(f"{str(a).strip()}_{str(b).strip()}") for a, b in zip(df[ch], df[rs])]
        for k, v in zip(keys, df[sc].to_numpy(dtype=float)):
            if k:
                rows.append((pid, k, float(v)))
    return pd.DataFrame(rows, columns=["pdb_id", "residue_key", "score"]), skipped


def convert_ordered(files, struct_dir, structures, order_from):
    rows, skipped = [], []
    struct_files = {}
    if structures:
        for p in Path(structures).iterdir():
            if p.suffix.lower() in {".cif", ".mmcif", ".pdb", ".ent"}:
                struct_files.setdefault(p.stem.lower(), p)
    for f in files:
        pid = f.stem.split(".")[0].lower()
        try:
            s = read_scores(f)
        except Exception as e:
            skipped.append((pid, str(e)))
            continue
        if order_from == "cache":
            keys = cache_keys(struct_dir, pid)
            src = "struct cache"
        else:
            if pid not in struct_files:
                skipped.append((pid, "no structure file for key recovery"))
                continue
            keys = structure_keys(struct_files[pid])
            src = "structure file"
        if keys is None:
            skipped.append((pid, "no struct cache entry"))
            continue
        if len(keys) != len(s):
            skipped.append((pid, f"{len(s)} scores vs {len(keys)} residues in the {src}"))
            continue
        rows += [(pid, k, float(v)) for k, v in zip(keys, s)]
    return pd.DataFrame(rows, columns=["pdb_id", "residue_key", "score"]), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True, choices=["p2rank", "pocketminer", "generic"])
    ap.add_argument("--input", required=True, help="directory of per-structure files, or one CSV for generic")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--structures", default=None)
    ap.add_argument("--order-from", default="structure", choices=["structure", "cache"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="adds a binary `prediction` column")
    ap.add_argument("--col-pdb", default="pdb_id")
    ap.add_argument("--col-key", default="residue_key")
    ap.add_argument("--col-score", default="score")
    args = ap.parse_args()

    cfg = load_config(args.config)
    struct_dir = Path(cfg["work"]["struct_cache"])
    inp = Path(args.input)

    if args.format == "generic":
        df = pd.read_csv(inp)
        out = pd.DataFrame({"pdb_id": df[args.col_pdb].astype(str).str.lower(),
                            "residue_key": [normalize_residue_token(str(k)) for k in df[args.col_key]],
                            "score": df[args.col_score].astype(float)})
        skipped = []
    else:
        files = sorted(p for p in inp.iterdir() if p.suffix.lower() in {".csv", ".npy"})
        if not files:
            raise SystemExit(f"no .csv/.npy files in {inp}")
        if args.format == "p2rank":
            out, skipped = convert_p2rank(files, struct_dir)
        else:
            out, skipped = convert_ordered(files, struct_dir, args.structures, args.order_from)

    out = out[out["residue_key"].astype(bool)].drop_duplicates(["pdb_id", "residue_key"])
    if out.empty:
        raise SystemExit("no rows converted; check the input format and residue numbering")

    # verify against the struct cache
    bad = []
    for pid, g in out.groupby("pdb_id"):
        keys = cache_keys(struct_dir, pid)
        if keys is None:
            continue
        cov = np.isin(keys, g["residue_key"].to_numpy()).mean()
        if cov < 0.5:
            bad.append((pid, f"only {cov:.0%} of cached residues matched"))
    if bad:
        print(f"[WARN] {len(bad)} structures match fewer than half their cached residues; "
              "this usually means a different residue-numbering convention:")
        for pid, why in bad[:10]:
            print(f"    {pid}: {why}")

    out["method"] = args.method
    if args.threshold is not None:
        out["prediction"] = (out["score"] >= args.threshold).astype(int)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    if skipped:
        (dst.with_suffix(".skipped.json")).write_text(json.dumps(dict(skipped), indent=2), encoding="utf-8")
    print(f"[DONE] {len(out)} rows for {out['pdb_id'].nunique()} structures -> {dst}")
    if skipped:
        print(f"       {len(skipped)} structures skipped (see {dst.with_suffix('.skipped.json')}); "
              "report these as competitor failures, do not drop them silently")


if __name__ == "__main__":
    main()
