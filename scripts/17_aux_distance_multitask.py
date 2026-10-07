#!/usr/bin/env python3
"""
IDEA 8 - Coordinates in the LOSS, not in the input: auxiliary "distance-to-site" task.

Hypothesis
----------
CryptoBench labels are a hard distance cut-off around the holo ligand. A residue 5 A from the
site and a residue 40 A away are both "negative", so BCE pushes them to the same target and
the shared representation never learns site proximity. Regressing a smooth proximity target
derived from TRAINING labels and apo coordinates gives the trunk a spatially organised
representation. At inference only the sequence embedding is needed (no structure, no Grid).

Model / loss
------------
    h          = trunk(ESM-2)                       (same trunk as the pLM baseline)
    z          = main head(h)                       -> scored output
    a          = aux head(h)
    t_i        = exp(-d_i^2 / (2 sigma^2)),  d_i = CB distance from residue i to the nearest
                 positive residue of the same protein (t = 1 on positives)
    L          = BCE_pw(z, y) + gamma * BCE(a, t)   (aux term masked where coordinates are missing)

Leakage check: t is built from the labels of the proteins being TRAINED on only; validation
and test proteins contribute no targets, and `a` is never used for prediction.

Frozen a priori: sigma = 5 A, gamma = 0.5, everything else = pLM baseline. Nothing is searched.
Ablation = baseline (gamma = 0, script 11). Optional sensitivity run: --sigma 8.

Outputs: results/exp/aux_dist_s<sigma>_g<gamma>/{oof,test}_predictions.csv, meta.json
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from cryptogrid.exp_common import Store, get_device, make_ctx, pos_weight_of, run_nn_protocol
from cryptogrid.exp_models import PlmHead


class AuxHead(nn.Module):
    def __init__(self, d_esm, hidden, dropout):
        super().__init__()
        self.plm = PlmHead(d_esm, hidden, dropout)
        self.aux = nn.Linear(hidden, 1)

    def forward(self, b):
        p = self.plm(b)
        return {"z": p["z"], "aux": self.aux(p["h"]).squeeze(-1)}


def proximity_target(y, cb, sigma):
    """(target, mask). Uses labels -> call ONLY for proteins in the fitting folds."""
    n = len(y)
    ok = np.isfinite(cb).all(axis=1)
    pos = np.flatnonzero((y > 0) & ok)
    t, m = y.astype(np.float32).copy(), np.zeros(n, dtype=np.float32)
    if len(pos) == 0 or ok.sum() == 0:
        return t, m
    d, _ = cKDTree(cb[pos].astype(np.float64)).query(cb[ok].astype(np.float64))
    t[ok] = np.exp(-(d ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    m[ok] = 1.0
    return t, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    hp = ctx.cfg["training"]
    store, device = Store(ctx), get_device(ctx.cfg)
    d_esm = store.esm(ctx.train_ids[0]).shape[1]
    name = args.name or f"aux_dist_s{args.sigma:g}_g{args.gamma:g}"

    def prepare(fit_ids):
        fit = set(fit_ids)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_of(store, fit_ids), device=device))
        targets = {p: proximity_target(store.labels(p)[0], store.geom(p)["cb"], args.sigma) for p in fit_ids}

        def batch_fn(pid):
            b = {"xe": torch.from_numpy(store.esm(pid).astype(np.float32)).to(device),
                 "y": torch.from_numpy(store.labels(pid)[0].astype(np.float32)).to(device)}
            if pid in fit:                           # held-out proteins never get an aux target
                t, m = targets[pid]
                b["t"], b["m"] = torch.from_numpy(t).to(device), torch.from_numpy(m).to(device)
            return b

        def loss_fn(out, b):
            aux = F.binary_cross_entropy_with_logits(out["aux"], b["t"], reduction="none")
            aux = (aux * b["m"]).sum() / b["m"].sum().clamp_min(1.0)
            return bce(out["z"], b["y"]) + args.gamma * aux

        return batch_fn, loss_fn

    print(f"[device] {device} | experiment={name}")
    run_nn_protocol(ctx, name, prepare,
                    lambda: AuxHead(d_esm, int(hp["hidden_plm"]), float(hp["dropout"])),
                    seeds=args.seeds, epochs=int(hp["epochs"]), lr=float(hp["learning_rate"]),
                    weight_decay=float(hp["weight_decay"]), resume=not args.no_resume,
                    meta={"idea": "auxiliary distance-to-site task", **{k: v for k, v in vars(args).items() if k != "config"}})


if __name__ == "__main__":
    main()
