#!/usr/bin/env python3
"""
PHASE II, STEP 2 - Build an EXTERNAL, homology-filtered evaluation set.

Why
---
The CryptoBench test split was inspected during the exploratory phase of this project, so
even with the freeze-and-single-unlock protocol it is a frozen-after-OOF held-out set, not
a pristine one. A confirmatory claim needs proteins that were never seen in any form. This
script turns any collection of apo structures with cryptic-site residue annotations into a
drop-in evaluation split for the existing pipeline, after removing everything homologous
to CryptoBench.

Inputs
------
--structures DIR   .cif / .mmcif / .pdb / .ent files, one per protein, named <pdb_id>.<ext>
--labels JSON      the CryptoBench dataset.json schema:
                     {"<pdb_id>": [{"apo_chain": "A",
                                    "apo_pocket_selection": ["A_41", "A_42", ...]}, ...]}
                   Residue keys are <auth_asym_id>_<auth_seq_id><insertion code>. Several
                   records per protein are merged into one binary label vector, exactly as
                   CryptoBench does.

What it does
------------
1. Parses every structure, builds the observed-residue universe (chain, author numbering,
   one-letter code) and the binary label vector, and reports proteins whose annotation does
   not match any observed residue.
2. Removes proteins that are homologous to ANY CryptoBench protein (train or test) above
   --max-identity (default 30%), and proteins that are redundant with each other above
   --max-internal-identity (default 90%). Uses MMseqs2 when available, otherwise a
   k-mer-prefiltered Smith-Waterman via Biopython. Every exclusion is logged.
3. Writes a self-contained work tree:
     work/external/<name>/cache_struct/   external proteins + symlinks to the 885 CryptoBench
                                          training proteins, so the pipeline can train on
                                          CryptoBench and evaluate on the external set
     work/external/<name>/cache_geom/     same, built with the script-10 feature code
     work/external/<name>/folds.json      train = CryptoBench 885 (official folds),
                                          test  = the external proteins
     config_external_<name>.yaml          ready to use with scripts 02/11/12/15/18/23
     work/external/<name>/report.json     counts, exclusions, label statistics

Note: Grid-Crypto pocket descriptors do not exist for external proteins, so X_struct is
written as zeros with the eight summary feature names. Grid-dependent models (scripts 13,
14) are therefore not applicable externally; the sequence anchor and the spatial GNN are,
and those are the models the manuscript recommends.

Usage
-----
  python scripts/24_build_external_benchmark.py --name pocketminer \
      --structures external_data/pocketminer_apo --labels external_data/pocketminer_labels.json
  python scripts/02_extract_esm_embeddings.py --config config_external_pocketminer.yaml
  python scripts/11_baseline_multiseed.py      --config config_external_pocketminer.yaml
  python scripts/15_anchored_spatial_gnn.py    --config config_external_pocketminer.yaml
  python scripts/25_external_report.py         --config config_external_pocketminer.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptogrid.common import ensure_dir, load_config, load_folds, normalize_residue_token, split_ids

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}
SUMMARY = ["summary::pocket_membership_count", "summary::unique_grid_sizes", "summary::grid_size_mean",
           "summary::grid_size_min", "summary::grid_size_max", "summary::center_x_std",
           "summary::center_y_std", "summary::center_z_std"]


# ---------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------

def parse_structure(path: Path):
    """-> list of (residue_key, one_letter) for observed standard residues, first model/altloc."""
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.Polypeptide import is_aa
    suffix = path.suffix.lower()
    parser = MMCIFParser(QUIET=True) if suffix in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    out = []
    for chain in model:
        for res in chain:
            het, seq, icode = res.id
            if het.strip() not in {"", "H_MSE"}:
                continue
            name = res.get_resname().strip().upper()
            if name not in THREE_TO_ONE and not is_aa(res, standard=True):
                continue
            ic = "" if icode in {" ", "", "?"} else icode.strip()
            key = normalize_residue_token(f"{chain.id.strip()}_{seq}{ic}")
            if not key:
                continue
            out.append((key, THREE_TO_ONE.get(name, "X")))
    return out


# ---------------------------------------------------------------------
# Homology filtering
# ---------------------------------------------------------------------

def kmer_set(seq: str, k: int = 5) -> set:
    return {seq[i:i + k] for i in range(max(0, len(seq) - k + 1))}


def identity(a: str, b: str) -> float:
    """Local alignment identity normalised by the shorter sequence."""
    from Bio import Align
    aligner = Align.PairwiseAligner(mode="local", open_gap_score=-11, extend_gap_score=-1)
    try:
        from Bio.Align import substitution_matrices
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    except Exception:
        aligner.match_score, aligner.mismatch_score = 2, -1
    try:
        aln = aligner.align(a, b)[0]
    except Exception:
        return 0.0
    ta, tb = aln[0], aln[1]
    same = sum(1 for x, y in zip(ta, tb) if x == y and x != "-")
    return same / max(1, min(len(a), len(b)))



def _normalized_fullseq_identity(fident, alnlen, qlen, tlen):
    """
    Identity definition used for E4 homology filtering:

        identical aligned residues / shorter FULL sequence length

    MMseqs:
        fident = identical residues / alignment length

    therefore:
        identical residues = fident * alnlen
    """
    fid = float(fident)

    # fident should be in [0,1], but tolerate percentage representation.
    if fid > 1.0:
        fid /= 100.0

    aln = float(alnlen)
    ql = float(qlen)
    tl = float(tlen)

    return (fid * aln) / max(1.0, min(ql, tl))


def mmseqs_identities(query: dict, target: dict, tmp: Path) -> dict:
    """
    Return:
        {query_id: maximum full-sequence-normalized identity
                   against ANY target}

    IMPORTANT:
    query and target are different datasets here.
    Therefore q == t identifiers must NOT be discarded.
    """

    if shutil.which("mmseqs") is None:
        return {}

    qf = tmp / "q.fasta"
    tf = tmp / "t.fasta"

    for f, d in ((qf, query), (tf, target)):
        f.write_text(
            "".join(f">{k}\n{v}\n" for k, v in d.items())
        )

    res = tmp / "res.m8"

    cmd = [
        "mmseqs", "easy-search",
        str(qf),
        str(tf),
        str(res),
        str(tmp / "mm"),
        "-s", "7.5",
        "--max-seqs", "1000",
        "-e", "1e5",
        "--format-output",
        "query,target,fident,alnlen,qlen,tlen"
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
    except Exception as e:
        print(
            f"[homology] mmseqs failed ({e}); "
            "falling back to Biopython"
        )
        return {}

    best = defaultdict(float)
    n_lines = 0

    for line in res.read_text().splitlines():

        p = line.split("\t")

        if len(p) < 6:
            continue

        q, t = p[0], p[1]

        val = _normalized_fullseq_identity(
            p[2], p[3], p[4], p[5]
        )

        best[q] = max(best[q], val)

        n_lines += 1

    print(
        f"[homology] MMseqs parsed {n_lines} "
        f"external-vs-reference alignments"
    )

    return dict(best)


def mmseqs_pairwise_identities(seqs: dict, tmp: Path) -> dict:
    """
    Pairwise identities for internal external-set de-replication.

    Self-hits ARE discarded here because query and target
    are the same sequence collection.
    """

    if shutil.which("mmseqs") is None:
        return {}

    fasta = tmp / "internal.fasta"

    fasta.write_text(
        "".join(f">{k}\n{v}\n" for k, v in seqs.items())
    )

    res = tmp / "internal.m8"

    cmd = [
        "mmseqs", "easy-search",
        str(fasta),
        str(fasta),
        str(res),
        str(tmp / "mm_internal"),
        "-s", "7.5",
        "--max-seqs", "1000",
        "-e", "1e5",
        "--format-output",
        "query,target,fident,alnlen,qlen,tlen"
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

    except Exception as e:
        print(
            f"[homology] internal MMseqs failed ({e}); "
            "falling back to Biopython"
        )
        return {}

    out = {}
    n_lines = 0

    for line in res.read_text().splitlines():

        p = line.split("\t")

        if len(p) < 6:
            continue

        q, t = p[0], p[1]

        if q == t:
            continue

        val = _normalized_fullseq_identity(
            p[2], p[3], p[4], p[5]
        )

        out[(q, t)] = max(
            out.get((q, t), 0.0),
            val
        )

        n_lines += 1

    print(
        f"[homology] MMseqs parsed {n_lines} "
        f"non-self external-vs-external alignments"
    )

    return out


def max_identity_against(query: dict, target: dict, kmer_cut: float, label: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        mm = mmseqs_identities(query, target, Path(td))
    if mm:
        print(f"[homology] {label}: MMseqs2 compared {len(query)} x {len(target)} sequences")
        return {q: mm.get(q, 0.0) for q in query}
    print(f"[homology] {label}: MMseqs2 not available, using k-mer prefilter + Biopython alignment")
    tks = {k: kmer_set(v) for k, v in target.items()}
    best, n_aln = {}, 0
    for i, (q, qs) in enumerate(query.items(), 1):
        qk = kmer_set(qs)
        b = 0.0
        for t, ts in tks.items():
            if q == t:
                continue
            share = len(qk & ts) / max(1, min(len(qk), len(ts)))
            if share < kmer_cut:
                continue
            n_aln += 1
            b = max(b, identity(qs, target[t]))
            if b >= 0.99:
                break
        best[q] = b
        if i % 25 == 0:
            print(f"  [homology] {i}/{len(query)} ({n_aln} alignments)", flush=True)
    return best


# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="the Phase-I config (source of the CryptoBench caches)")
    ap.add_argument("--name", required=True, help="short identifier, e.g. pocketminer or cryptosite")
    ap.add_argument("--structures", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--max-identity", type=float, default=0.30,
                    help="drop external proteins with identity above this to any CryptoBench protein")
    ap.add_argument("--max-internal-identity", type=float, default=0.90)
    ap.add_argument("--kmer-cut", type=float, default=0.05, help="k-mer prefilter for the Biopython fallback")
    ap.add_argument("--min-residues", type=int, default=30)
    ap.add_argument("--keep-homologs", action="store_true", help="report but do not remove homologs (NOT recommended)")
    ap.add_argument("--out-root", default=None, help="where to put the external work tree (default <work.root>/external)")
    ap.add_argument("--config-out", default=None, help="path of the generated config (default ./config_external_<name>.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    src_struct = Path(cfg["work"]["struct_cache"])
    src_geom = Path(cfg["work"].get("root", "work")) / "cache_geom"
    train, test, folds = split_ids(load_folds(cfg["project"]["folds_json"]))
    root = ensure_dir(Path(args.out_root or (Path(cfg["work"].get("root", "work")) / "external")) / args.name)
    out_struct = ensure_dir(root / "cache_struct")
    out_geom = ensure_dir(root / "cache_geom")

    labels = json.loads(Path(args.labels).read_text())
    files = {}
    for p in sorted(Path(args.structures).iterdir()):
        if p.suffix.lower() in {".cif", ".mmcif", ".pdb", ".ent"}:
            files.setdefault(p.stem.lower(), p)
    print(f"[input] {len(files)} structures, {len(labels)} label records")

    # ---- parse + label ---------------------------------------------------------
    seqs, data, report = {}, {}, {"excluded": {}, "name": args.name}
    for pid_raw, recs in labels.items():
        pid = pid_raw.lower()
        if pid in set(train) | set(test):
            report["excluded"][pid] = "identifier also present in CryptoBench"
            continue
        if pid not in files:
            report["excluded"][pid] = "no structure file"
            continue
        res = parse_structure(files[pid])
        if len(res) < args.min_residues:
            report["excluded"][pid] = f"only {len(res)} observed residues"
            continue
        keys = np.array([k for k, _ in res])
        aa = np.array([a for _, a in res])
        want = set()
        for r in (recs if isinstance(recs, list) else [recs]):
            for t in r.get("apo_pocket_selection", []):
                k = normalize_residue_token(t)
                if k:
                    want.add(k)
        pos = np.isin(keys, list(want))
        if pos.sum() == 0:
            report["excluded"][pid] = f"none of the {len(want)} annotated residues is observed"
            continue
        if pos.all():
            report["excluded"][pid] = "every observed residue is annotated"
            continue
        matched = len(want & set(keys.tolist()))
        data[pid] = {"keys": keys, "aa": aa, "y": pos.astype(np.int8),
                     "annotated": len(want), "matched": matched}
        seqs[pid] = "".join(aa.tolist())
    print(f"[parse] {len(data)} usable proteins, {len(report['excluded'])} excluded so far")
    if not data:
        raise SystemExit("no usable proteins; check --labels residue keys against the structures")

    # ---- homology filtering ----------------------------------------------------
    cb_seqs = {}
    for pid in list(train) + list(test):
        f = src_struct / f"{pid}.npz"
        if f.exists():
            cb_seqs[pid] = "".join(np.load(f, allow_pickle=False)["aa"].astype(str).tolist())
    ext_vs_cb = max_identity_against(
        seqs, cb_seqs, args.kmer_cut,
        "external vs CryptoBench"
    )

    # Correct pairwise external redundancy graph.
    with tempfile.TemporaryDirectory() as td:
        internal_pairs = mmseqs_pairwise_identities(
            seqs, Path(td)
        )

    # Safe fallback if MMseqs pairwise search failed.
    if not internal_pairs:
        print(
            "[homology] building exact external pairwise "
            "identities with Biopython fallback"
        )

        pids = sorted(seqs)

        for i, q in enumerate(pids):
            for t in pids[i + 1:]:
                val = identity(seqs[q], seqs[t])

                internal_pairs[(q, t)] = val
                internal_pairs[(t, q)] = val

    def pair_id(a, b):
        return max(
            internal_pairs.get((a, b), 0.0),
            internal_pairs.get((b, a), 0.0)
        )

    rows = []
    kept = []

    for pid in sorted(data):

        a = ext_vs_cb.get(pid, 0.0)

        b = max(
            [
                pair_id(pid, other)
                for other in data
                if other != pid
            ] or [0.0]
        )

        drop = None
        redundant_to = ""

        if a > args.max_identity:
            drop = (
                f"identity {a:.2f} to CryptoBench "
                f"> {args.max_identity}"
            )

        else:
            # Deterministic greedy de-replication:
            # only compare the current protein with ACTUAL
            # retained proteins.
            conflicts = [
                (k, pair_id(pid, k))
                for k in kept
                if pair_id(pid, k) > args.max_internal_identity
            ]

            if conflicts:
                redundant_to, sim = max(
                    conflicts,
                    key=lambda x: x[1]
                )

                drop = (
                    f"internal redundancy {sim:.2f} "
                    f"to retained {redundant_to} "
                    f"> {args.max_internal_identity}"
                )

        rows.append({
            "pdb_id": pid,
            "n_residues": len(data[pid]["keys"]),
            "n_site_residues": int(data[pid]["y"].sum()),
            "annotated": data[pid]["annotated"],
            "matched": data[pid]["matched"],
            "max_identity_to_cryptobench": a,
            "max_identity_internal": b,
            "kept": drop is None,
            "redundant_to": redundant_to,
            "reason": drop or "",
        })

        if drop is None or args.keep_homologs:
            kept.append(pid)
        else:
            report["excluded"][pid] = drop
    pd.DataFrame(rows).to_csv(root / "homology_report.csv", index=False)
    print(f"[homology] kept {len(kept)} / {len(data)} proteins "
          f"(threshold {args.max_identity:.0%} vs CryptoBench)")
    if not kept:
        raise SystemExit("every external protein is homologous to CryptoBench at this threshold")

    # ---- caches ----------------------------------------------------------------
    for pid in kept:
        d = data[pid]
        np.savez_compressed(out_struct / f"{pid}.npz",
                            X_struct=np.zeros((len(d["keys"]), len(SUMMARY)), dtype=np.float32),
                            y=d["y"], residue_keys=d["keys"].astype("U"), aa=d["aa"].astype("U"),
                            pdb_id=np.asarray(pid))
    (out_struct / "feature_names.json").write_text(json.dumps(SUMMARY, indent=2), encoding="utf-8")
    linked = 0
    for pid in train:
        for src, dst in ((src_struct / f"{pid}.npz", out_struct / f"{pid}.npz"),
                         (src_geom / f"{pid}.npz", out_geom / f"{pid}.npz")):
            if src.exists() and not dst.exists():
                try:
                    os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)
                except OSError:
                    shutil.copy2(src, dst)
                linked += 1
    for nm in ("feature_names.json",):
        if (src_geom / nm).exists():
            shutil.copy2(src_geom / nm, out_geom / nm)
    print(f"[cache] {len(kept)} external + {linked // 2} linked CryptoBench training proteins")

    ext_folds = {k: list(v) for k, v in folds.items() if v}
    ext_folds["test"] = kept
    (root / "folds.json").write_text(json.dumps(ext_folds, indent=2), encoding="utf-8")

    new_cfg = json.loads(json.dumps(cfg))
    new_cfg["project"]["folds_json"] = str((root / "folds.json").resolve())
    new_cfg["project"]["cif_root"] = str(Path(args.structures).resolve())
    new_cfg["work"] = dict(cfg["work"])
    new_cfg["work"]["root"] = str(root.resolve())
    new_cfg["work"]["struct_cache"] = str(out_struct.resolve())
    new_cfg["work"]["esm_cache"] = str((root / "cache_esm").resolve())
    new_cfg["work"]["results"] = str((root / "results").resolve())
    cfg_path = Path(args.config_out or f"config_external_{args.name}.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(new_cfg), encoding="utf-8")

    report.update({"n_input": len(labels), "n_parsed": len(data), "n_kept": len(kept),
                   "max_identity_to_cryptobench": args.max_identity,
                   "residues": int(sum(len(data[p]["keys"]) for p in kept)),
                   "positives": int(sum(int(data[p]["y"].sum()) for p in kept)),
                   "config": str(cfg_path)})
    report["prevalence"] = report["positives"] / max(1, report["residues"])
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[DONE] external benchmark `{args.name}`: {len(kept)} proteins, {report['residues']} residues, "
          f"{report['positives']} positives ({report['prevalence']:.2%})")
    print(f"       config -> {cfg_path}")
    print("\nNext steps:")
    print(f"  python scripts/10_build_geom_cache.py        --config {cfg_path}")
    print(f"  python scripts/02_extract_esm_embeddings.py  --config {cfg_path}      # frozen pLM")
    print(f"  python scripts/11_baseline_multiseed.py      --config {cfg_path}")
    print(f"  python scripts/15_anchored_spatial_gnn.py    --config {cfg_path}")
    print(f"  python scripts/25_external_report.py         --config {cfg_path}")


if __name__ == "__main__":
    main()
