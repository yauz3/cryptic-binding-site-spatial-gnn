#!/usr/bin/env python3
"""
Convert any external cryptic-site predictor's output into the one format the comparison
scripts read:   pdb_id, residue_key, score [, prediction] [, method]

The hard part is never the file format, it is the residue namespace. Competitors index their
scores by UniProt position, by observed-residue order, by chain plus author numbering, or by
their own cleaned structure. A one-residue offset silently invalidates every head-to-head
number, so this tool maps by explicit keys where they exist and by SEQUENCE ALIGNMENT where
they do not, verifies the mapping against our struct cache, and refuses to write a file whose
coverage or identity falls below a threshold.

Formats
-------
--format keyed        CSV or JSON that already names chain and residue number.
                      CSV: columns for pdb id, chain, residue number, score (names given with
                      --col-*). JSON: {"<pdb_id>": {"<chain>_<resi>": score}}.
--format seq-aligned  one array per protein (.npy/.csv/.txt, one score per residue) plus the
                      sequence those scores were computed on (--sequences, a FASTA whose
                      records are named <pdb_id> or <pdb_id>_<chain>). The array is aligned to
                      our cached sequence with a global alignment and mapped position by
                      position. This is the right mode for UniProt-indexed output, for
                      whole-sequence pLM predictions, and for any tool that silently dropped
                      or added residues.
--format ordered      one array per protein, assumed to follow the struct cache's residue
                      order exactly. Only safe when the tool was run on the same cleaned
                      structures; lengths must match or the protein is skipped and listed.

Every mode writes <out>.coverage.csv (per protein: matched residues, alignment identity) and
<out>.skipped.json (protein, reason). Report the skipped proteins as competitor failures in
the manuscript; do not drop them silently.

Examples
--------
  python scripts/external/convert_comparators.py --format seq-aligned \
      --input runs/hidden_scores --sequences runs/hidden_sequences.fasta \
      --config config.yaml --method "Hidden (ESM2-3B finetuned)" \
      --out phase2/external_predictions/hidden.csv

  python scripts/external/convert_comparators.py --format keyed \
      --input runs/unipocket_cry.csv --config config.yaml --method UniPocket \
      --col-pdb pdb --col-chain chain --col-resi resnum --col-score cry_prob \
      --out phase2/external_predictions/unipocket.csv
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


def cache_entry(struct_dir: Path, pid: str):
    f = struct_dir / f"{pid}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    return z["residue_keys"].astype(str), z["aa"].astype(str)


def read_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=float).ravel()
    if path.suffix.lower() in {".txt", ".dat"}:
        return np.loadtxt(path, dtype=float).ravel()
    df = pd.read_csv(path)
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        raise ValueError("no numeric column")
    return df[num[-1]].to_numpy(dtype=float)


def read_fasta(path: Path) -> dict:
    seqs, name, buf = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name = line[1:].strip().split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf)
    return seqs


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
    """Map positions of `query` (the competitor's sequence) onto `target` (our cache sequence).

    Returns (mapping, identity) where mapping[j] = i means target position j took query
    position i, and identity is the fraction of aligned target positions that match.
    """
    aln = _make_aligner().align(target, query)[0]
    mapping = {}
    same = 0
    for (t0, t1), (q0, q1) in zip(aln.aligned[0], aln.aligned[1]):
        for off in range(t1 - t0):
            tj, qi = t0 + off, q0 + off
            mapping[tj] = qi
            same += int(target[tj] == query[qi])
    return mapping, same / max(1, len(mapping))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True, choices=["keyed", "seq-aligned", "ordered"])
    ap.add_argument("--input", required=True, help="file (keyed) or directory of per-protein arrays")
    ap.add_argument("--sequences", default=None, help="FASTA for seq-aligned mode")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="adds a binary `prediction` column")
    ap.add_argument("--min-identity", type=float, default=0.95,
                    help="seq-aligned: minimum alignment identity to accept a protein")
    ap.add_argument("--min-coverage", type=float, default=0.80,
                    help="minimum fraction of cached residues that must receive a score")
    ap.add_argument("--col-pdb", default="pdb_id")
    ap.add_argument("--col-chain", default="chain")
    ap.add_argument("--col-resi", default="residue_number")
    ap.add_argument("--col-score", default="score")
    ap.add_argument("--col-key", default=None, help="keyed mode: a single column already of the form CHAIN_RESI")
    args = ap.parse_args()

    cfg = load_config(args.config)
    struct_dir = Path(cfg["work"]["struct_cache"])
    rows, skipped, cover = [], {}, []

    if args.format == "keyed":
        inp = Path(args.input)
        if inp.suffix.lower() == ".json":
            data = json.loads(inp.read_text())
            for pid, d in data.items():
                for k, v in d.items():
                    key = normalize_residue_token(str(k))
                    if key:
                        rows.append((str(pid).lower(), key, float(v)))
        else:
            df = pd.read_csv(inp)
            df.columns = [c.strip() for c in df.columns]
            if args.col_key and args.col_key in df.columns:
                keys = [normalize_residue_token(str(k)) for k in df[args.col_key]]
            else:
                miss = [c for c in (args.col_pdb, args.col_chain, args.col_resi, args.col_score)
                        if c not in df.columns]
                if miss:
                    raise SystemExit(f"missing columns {miss}; available: {list(df.columns)}")
                keys = [normalize_residue_token(f"{str(c).strip()}_{str(r).strip()}")
                        for c, r in zip(df[args.col_chain], df[args.col_resi])]
            pdbs = df[args.col_pdb].astype(str).str.lower().str.strip()
            for pid, key, v in zip(pdbs, keys, df[args.col_score].astype(float)):
                if key:
                    rows.append((pid, key, float(v)))
    else:
        inp = Path(args.input)
        files = sorted(p for p in inp.iterdir() if p.suffix.lower() in {".npy", ".csv", ".txt", ".dat"})
        if not files:
            raise SystemExit(f"no per-protein arrays in {inp}")
        seqs = read_fasta(Path(args.sequences)) if args.sequences else {}
        if args.format == "seq-aligned" and not seqs:
            raise SystemExit("--format seq-aligned needs --sequences")
        for f in files:
            pid = f.stem.split(".")[0].lower()
            ent = cache_entry(struct_dir, pid)
            if ent is None:
                skipped[pid] = "not in the struct cache"
                continue
            keys, aa = ent
            try:
                s = read_array(f)
            except Exception as e:
                skipped[pid] = f"unreadable array: {e}"
                continue
            if args.format == "ordered":
                if len(s) != len(keys):
                    skipped[pid] = f"{len(s)} scores vs {len(keys)} cached residues"
                    continue
                mapped = {j: j for j in range(len(keys))}
                ident = 1.0
            else:
                qs = seqs.get(pid) or seqs.get(f.stem) or seqs.get(pid.upper())
                if qs is None:
                    skipped[pid] = "no sequence record"
                    continue
                if len(qs) != len(s):
                    skipped[pid] = f"sequence length {len(qs)} != {len(s)} scores"
                    continue
                mapped, ident = align_map(qs, "".join(aa.tolist()))
                if ident < args.min_identity:
                    skipped[pid] = f"alignment identity {ident:.2f} < {args.min_identity}"
                    continue
            cov = len(mapped) / max(1, len(keys))
            cover.append({"pdb_id": pid, "cached_residues": len(keys), "matched": len(mapped),
                          "coverage": cov, "alignment_identity": ident})
            if cov < args.min_coverage:
                skipped[pid] = f"coverage {cov:.2f} < {args.min_coverage}"
                continue
            for j, i in mapped.items():
                rows.append((pid, keys[j], float(s[i])))

    out = pd.DataFrame(rows, columns=["pdb_id", "residue_key", "score"]).drop_duplicates(
        ["pdb_id", "residue_key"])
    if out.empty:
        raise SystemExit("nothing converted; check the format options and the residue numbering")

    if args.format == "keyed":
        for pid, g in out.groupby("pdb_id"):
            ent = cache_entry(struct_dir, pid)
            if ent is None:
                continue
            keys = ent[0]
            cov = float(np.isin(keys, g["residue_key"].to_numpy()).mean())
            cover.append({"pdb_id": pid, "cached_residues": len(keys),
                          "matched": int(cov * len(keys)), "coverage": cov,
                          "alignment_identity": np.nan})
            if cov < args.min_coverage:
                skipped[pid] = f"coverage {cov:.2f} < {args.min_coverage}"
        if skipped:
            out = out[~out["pdb_id"].isin(skipped)]

    out["method"] = args.method
    if args.threshold is not None:
        out["prediction"] = (out["score"] >= args.threshold).astype(int)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    cdf = pd.DataFrame(cover)
    if not cdf.empty:
        cdf.to_csv(dst.with_suffix(".coverage.csv"), index=False)
    if skipped:
        dst.with_suffix(".skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    print(f"[DONE] {len(out)} rows for {out['pdb_id'].nunique()} proteins -> {dst}")
    if not cdf.empty:
        kept = cdf[~cdf["pdb_id"].isin(skipped)]
        if not kept.empty:
            print(f"       kept proteins: mean coverage {kept['coverage'].mean():.3f}"
                  + (f", mean alignment identity {kept['alignment_identity'].mean():.3f}"
                     if kept["alignment_identity"].notna().any() else ""))
    if skipped:
        print(f"       {len(skipped)} proteins skipped -> {dst.with_suffix('.skipped.json')}")
        print("       these are competitor failures and must be reported, not dropped silently")


if __name__ == "__main__":
    main()
