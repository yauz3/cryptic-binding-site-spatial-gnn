#!/usr/bin/env python3

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def load_seq2pocket(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))

    spec = importlib.util.spec_from_file_location(
        "seq2pocket_runner",
        repo / "run_seq2pocket.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def find_npz(struct_dir, pid):
    candidates = [
        struct_dir / f"{pid}.npz",
        struct_dir / f"{pid.lower()}.npz",
        struct_dir / f"{pid.upper()}.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def chain_from_key(key):
    # author-chain may contain multiple characters; residue number is last token
    s = str(key)
    if "_" not in s:
        return ""
    return s.rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seq2pocket-repo", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    struct_dir = Path(cfg["work"]["struct_cache"])
    results_dir = Path(cfg["work"]["results"])

    base_csv = results_dir / "exp" / "base_plm" / "test_predictions.csv"
    if not base_csv.exists():
        raise SystemExit(f"Missing test prediction file: {base_csv}")

    base = pd.read_csv(base_csv)

    if "pdb_id" not in base.columns:
        raise SystemExit(
            f"test_predictions.csv has no pdb_id column: {list(base.columns)}"
        )

    test_ids = list(dict.fromkeys(
        base["pdb_id"].astype(str).str.lower().tolist()
    ))

    if args.limit:
        test_ids = test_ids[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Test proteins:", len(test_ids))
    print("Struct cache:", struct_dir)
    print("Output:", out_dir)

    s2p = load_seq2pocket(args.seq2pocket_repo)
    s2p._register_unpickle_classes()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    tokenizer, model, _, _ = s2p.load_models(
        task="cbs",
        size="3B",
        models_dir=Path(args.models_dir),
        device=device,
        use_smoother=False,
    )

    manifest = []
    fasta_lines = []

    for n, pid in enumerate(test_ids, 1):

        npz = find_npz(struct_dir, pid)

        if npz is None:
            print(f"[{n}/{len(test_ids)}] SKIP {pid}: struct cache missing")
            manifest.append({
                "pdb_id": pid,
                "status": "missing_cache"
            })
            continue

        z = np.load(npz, allow_pickle=False)

        keys = z["residue_keys"].astype(str)
        aa = z["aa"].astype(str)

        if len(keys) != len(aa):
            raise RuntimeError(
                f"{pid}: residue_keys={len(keys)} aa={len(aa)}"
            )

        scores = np.full(len(keys), np.nan, dtype=np.float32)

        chains = np.array([chain_from_key(k) for k in keys], dtype=object)

        unique_chains = []
        for c in chains:
            if c not in unique_chains:
                unique_chains.append(c)

        try:
            for chain in unique_chains:

                idx = np.where(chains == chain)[0]
                sequence = "".join(aa[idx].tolist())

                if not sequence:
                    continue

                probs = s2p.compute_prediction(
                    sequence,
                    model,
                    tokenizer,
                    device,
                )

                probs = np.asarray(probs, dtype=np.float32).ravel()

                if len(probs) != len(idx):
                    raise RuntimeError(
                        f"{pid}/{chain}: model returned {len(probs)} "
                        f"scores for {len(idx)} residues"
                    )

                scores[idx] = probs

                fasta_lines.append(
                    f">{pid}_{chain}\n{sequence}\n"
                )

            if np.isnan(scores).any():
                raise RuntimeError(
                    f"{pid}: {np.isnan(scores).sum()} residues have no score"
                )

            np.save(out_dir / f"{pid}.npy", scores)

            manifest.append({
                "pdb_id": pid,
                "status": "ok",
                "n_residues": len(scores),
                "n_chains": len(unique_chains),
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
            })

            print(
                f"[{n}/{len(test_ids)}] OK {pid}: "
                f"{len(scores)} residues / {len(unique_chains)} chain(s) "
                f"[{scores.min():.3f}, {scores.max():.3f}]"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"CUDA OOM at {pid}")
                raise

            print(f"[{n}/{len(test_ids)}] ERROR {pid}: {e}")

            manifest.append({
                "pdb_id": pid,
                "status": "error",
                "reason": str(e),
            })

    pd.DataFrame(manifest).to_csv(
        out_dir.parent / "seq2pocket_cbs3b_manifest.csv",
        index=False,
    )

    (out_dir.parent / "seq2pocket_cbs3b_sequences.fasta").write_text(
        "".join(fasta_lines)
    )

    print()
    print("DONE")
    print(pd.DataFrame(manifest)["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
