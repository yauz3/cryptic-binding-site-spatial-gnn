#!/usr/bin/env python3
"""
IDEA 7 - Top-of-ranking-aware training loss for the pLM head (targets EF@1% / BEDROC20 directly).

Hypothesis
----------
BCE with pos_weight optimises calibration over ALL residues; EF@1% and BEDROC20 depend only on
which residues win the first few ranks inside each protein. Adding a per-protein ranking term
that concentrates on the hardest negatives should move early-enrichment metrics more than AP/AUC.

Loss (one protein = one list)
-----------------------------
    L = BCE_pw(z, y) + lambda * L_rank(z, y)

    --rank-loss toppair    : mean_{i in P} mean_{j in H} tau*softplus((z_j - z_i)/tau)
                             H = top 5 % highest-scoring negatives of that protein (re-selected every step)
    --rank-loss softbedroc : 1 - sum_{i in P} exp(-alpha * softrank_i / n) / ideal,  alpha = 20
                             softrank_i = 1 + sum_{j != i} sigmoid((z_j - z_i)/tau)

Frozen a priori: architecture, optimiser, epochs = pLM baseline; tau = 1, top_frac = 0.05, alpha = 20.
Searched: only (rank-loss, lambda) - run each as its own experiment, e.g.
    --rank-loss toppair --lam 0.5      --rank-loss toppair --lam 1.0
    --rank-loss softbedroc --lam 0.5   --rank-loss softbedroc --lam 1.0
Script 19 then chooses among experiments with NESTED train-OOF selection; do not pick by hand.

Ablation = the baseline itself (lambda = 0, script 11). Same seeds -> same protein order per epoch.

Outputs: results/exp/rank_<loss>_lam<lambda>/{oof,test}_predictions.csv, meta.json
The resulting model can replace the anchor for scripts 12-14 via their --base option.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from cryptogrid.exp_common import Store, get_device, make_ctx, pos_weight_of, run_nn_protocol
from cryptogrid.exp_models import PlmHead, soft_bedroc_loss, top_pair_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--rank-loss", choices=["toppair", "softbedroc"], default="toppair")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--top-frac", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    hp = ctx.cfg["training"]
    store, device = Store(ctx), get_device(ctx.cfg)
    d_esm = store.esm(ctx.train_ids[0]).shape[1]
    name = args.name or f"rank_{args.rank_loss}_lam{args.lam:g}"

    def prepare(fit_ids):
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_of(store, fit_ids), device=device))

        def batch_fn(pid):
            return {"xe": torch.from_numpy(store.esm(pid).astype(np.float32)).to(device),
                    "y": torch.from_numpy(store.labels(pid)[0].astype(np.float32)).to(device)}

        def loss_fn(out, b):
            if args.rank_loss == "toppair":
                r = top_pair_loss(out["z"], b["y"], top_frac=args.top_frac, tau=args.tau)
            else:
                r = soft_bedroc_loss(out["z"], b["y"], alpha=args.alpha, tau=args.tau)
            return bce(out["z"], b["y"]) + args.lam * r

        return batch_fn, loss_fn

    print(f"[device] {device} | experiment={name}")
    run_nn_protocol(ctx, name, prepare,
                    lambda: PlmHead(d_esm, int(hp["hidden_plm"]), float(hp["dropout"])),
                    seeds=args.seeds, epochs=int(hp["epochs"]), lr=float(hp["learning_rate"]),
                    weight_decay=float(hp["weight_decay"]), resume=not args.no_resume,
                    meta={"idea": "ranking-aware loss", **{k: v for k, v in vars(args).items() if k != "config"}})


if __name__ == "__main__":
    main()
