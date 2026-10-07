#!/usr/bin/env python3
"""
IDEA 6 - pLM-anchored spatial graph network (the direct way to "use the coordinates").

Hypothesis
----------
A per-residue MLP cannot see that residue i sits next to residues j, k in 3D. Message passing
over a CB kNN graph lets the model (i) pool pLM evidence across a surface patch and (ii) read
local geometry/flexibility, while the pLM prediction stays the anchor.

Model (cryptogrid/exp_models.AnchoredGNN)
-----------------------------------------
    h, z_plm = PlmHead(ESM-2)                               # identical to the baseline head
    g0   = MLP([h, stopgrad(z_plm), geom_features])
    g    = L x attention message passing over kNN(CB, k) with invariant edge features
           (16 RBFs of distance, side-chain orientation dot-products, sequence separation)
    z    = z_plm + z_res(g)                                 # z_res head is ZERO-initialised

Loss
----
    L = BCE_pw(z, y) + mu * BCE_pw(z_plm, y) + rho * mean(z_res^2)
        mu  : deep supervision keeps the anchor a valid stand-alone pLM predictor
        rho : shrinkage of the structural correction toward zero

Frozen a priori (no search): k=16, layers=2, hidden=128, mu=0.5, rho=0.01, epochs/lr/wd from
config.yaml, structural branch lr multiplier 3. Seeds are ensembled, never selected.
Geometry scaler is fitted on the fitting folds only.

Required ablations (same seeds; give each a --name):
    --no-graph      geometry features only, no message passing (layers=0)
    --no-geom       graph over pLM states only, no geometry node features
    --no-bfactor    drop B-factor-derived features (they do not exist for predicted/NMR structures)

Outputs: results/exp/<name>/{oof,test}_predictions.csv (z = seed mean, z_s<seed>), meta.json
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from cryptogrid.exp_common import RobustScaler, Store, get_device, make_ctx, pos_weight_of, run_nn_protocol
from cryptogrid.exp_models import AnchoredGNN, build_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=0.01)
    ap.add_argument("--struct-lr-mult", type=float, default=3.0)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-geom", action="store_true")
    ap.add_argument("--no-bfactor", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    hp = ctx.cfg["training"]
    store, device = Store(ctx), get_device(ctx.cfg)
    gnames = store.geom_feature_names()
    keep = np.array([not (args.no_bfactor and n.startswith("b_")) for n in gnames])
    layers = 0 if args.no_graph else args.layers
    name = args.name or ("gnn_anchor" + ("_nograph" if args.no_graph else "") + ("_nogeom" if args.no_geom else "")
                         + ("_nobfactor" if args.no_bfactor else ""))
    d_esm = store.esm(ctx.train_ids[0]).shape[1]
    graphs = {}                                   # fold-independent, label-free

    def graph(pid):
        if pid not in graphs:
            g = build_graph(store.geom(pid), k=args.k)
            graphs[pid] = {k: torch.from_numpy(v) for k, v in g.items()}
        return graphs[pid]

    def prepare(fit_ids):
        scaler = RobustScaler().fit([store.geom(p)["X_geom"][:, keep] for p in fit_ids])
        pw = torch.tensor(pos_weight_of(store, fit_ids), device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)

        def batch_fn(pid):
            g = graph(pid)
            return {"xe": torch.from_numpy(store.esm(pid).astype(np.float32)).to(device),
                    "xg": torch.from_numpy(scaler.transform(store.geom(pid)["X_geom"][:, keep])).to(device),
                    "idx": g["idx"].to(device), "mask": g["mask"].to(device), "edge": g["edge"].to(device),
                    "y": torch.from_numpy(store.labels(pid)[0].astype(np.float32)).to(device)}

        def loss_fn(out, b):
            return (bce(out["z"], b["y"]) + args.mu * bce(out["z_plm"], b["y"])
                    + args.rho * out["z_res"].pow(2).mean())

        return batch_fn, loss_fn

    def build_model():
        return AnchoredGNN(d_esm, int(keep.sum()), int(hp["hidden_plm"]), float(hp["dropout"]),
                           hidden=args.hidden, layers=layers, use_geom=not args.no_geom)

    def param_groups(model, lr):
        return [{"params": model.plm.parameters(), "lr": lr},
                {"params": model.structural_parameters(), "lr": lr * args.struct_lr_mult}]

    print(f"[device] {device} | experiment={name} | layers={layers} geom={not args.no_geom} "
          f"geom_dim={int(keep.sum())}")
    run_nn_protocol(ctx, name, prepare, build_model, seeds=args.seeds, epochs=int(hp["epochs"]),
                    lr=float(hp["learning_rate"]), weight_decay=float(hp["weight_decay"]),
                    param_groups=param_groups, resume=not args.no_resume,
                    meta={"idea": "anchored spatial GNN", **{k: v for k, v in vars(args).items() if k != "config"}})


if __name__ == "__main__":
    main()
