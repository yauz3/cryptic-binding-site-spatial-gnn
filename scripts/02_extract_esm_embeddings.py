#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from cryptogrid.common import load_config, ensure_dir, residue_chain


@torch.inference_mode()
def embed_sequence(model, tokenizer, seq, device, max_tokens=1000, overlap=128):
    if not seq:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    hidden = int(model.config.hidden_size)
    n = len(seq)
    acc = np.zeros((n, hidden), dtype=np.float32)
    cnt = np.zeros(n, dtype=np.float32)
    step = max(1, max_tokens - overlap)
    for st in range(0, n, step):
        en = min(n, st + max_tokens)
        sub = seq[st:en]
        toks = tokenizer(sub, return_tensors="pt", add_special_tokens=True)
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(**toks).last_hidden_state[0]
        emb = out[1:1 + len(sub)].float().cpu().numpy()
        acc[st:en] += emb
        cnt[st:en] += 1.0
        if en == n:
            break
    cnt[cnt == 0] = 1.0
    return acc / cnt[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    struct_dir = Path(cfg["work"]["struct_cache"])
    outdir = ensure_dir(cfg["work"]["esm_cache"])
    model_name = args.model_name or os.environ.get("ESM_MODEL") or cfg["esm"]["model_name"]
    max_tokens = int(cfg["esm"].get("max_tokens", 1000))
    overlap = int(cfg["esm"].get("overlap", 128))
    fp16_cache = bool(cfg["esm"].get("fp16_cache", True))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"[ESM] model={model_name} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype if device.type == "cuda" else None)
    model.to(device)
    model.eval()

    files = sorted(struct_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No structural cache files in {struct_dir}. Run script 01 first.")

    for ix, sf in enumerate(files, 1):
        pid = sf.stem
        out = outdir / f"{pid}.npz"
        if out.exists() and not args.overwrite:
            print(f"[{ix}/{len(files)}] {pid}: cached")
            continue

        z = np.load(sf, allow_pickle=False)
        residue_keys = z["residue_keys"].astype(str)
        aas = z["aa"].astype(str)
        X = np.zeros((len(residue_keys), int(model.config.hidden_size)), dtype=np.float32)

        chains = []
        for r in residue_keys:
            c = residue_chain(r)
            if c not in chains:
                chains.append(c)

        for ch in chains:
            idx = np.array([i for i, r in enumerate(residue_keys) if residue_chain(r) == ch], dtype=int)
            seq = "".join(aas[idx].tolist()).replace("U", "X").replace("O", "X")
            e = embed_sequence(model, tokenizer, seq, device, max_tokens=max_tokens, overlap=overlap)
            if len(e) != len(idx):
                raise RuntimeError(f"{pid}/{ch}: embedding length {len(e)} != residue length {len(idx)}")
            X[idx] = e

        if fp16_cache:
            X = X.astype(np.float16)
        np.savez_compressed(out, X_esm=X, residue_keys=residue_keys.astype("U"), pdb_id=np.asarray(pid))
        print(f"[{ix}/{len(files)}] {pid}: {X.shape}")

    (outdir / "model_name.txt").write_text(model_name + "\n", encoding="utf-8")
    print(f"[DONE] ESM cache -> {outdir}")


if __name__ == "__main__":
    main()
