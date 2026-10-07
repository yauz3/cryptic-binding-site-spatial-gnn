#!/usr/bin/env python3
"""
IDEA 2 - Seed-ensembled pLM anchor (the comparator every other experiment is paired against).

Hypothesis
----------
The reported gain of residual-scalar over pLM (+0.003 macro AP) was measured with ONE
seed per model. Seed-to-seed variation of an MLP head on frozen ESM-2 is of the same
order, so (a) the comparison is noisy and (b) part of any "fusion gain" may just be an
ensembling effect. A K-seed logit ensemble of the SAME pLM-only architecture is the
fair anchor: a candidate has to beat it, not a single-seed run.

What is frozen
--------------
Architecture and optimiser = plm_only from scripts/03_train_hybrid_models.py
(Linear-ReLU-Dropout x2, AdamW, BCE with pos_weight, config.yaml epochs/lr/wd).
Nothing is searched. No seed is selected; the ensemble mean is the model.

Outputs
-------
results/exp/base_plm/oof_predictions.csv    z (seed mean) + z_s<seed>, official 4-fold OOF
results/exp/base_plm/test_predictions.csv   written, NOT scored (see script 19)
results/exp/base_plm/meta.json
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from cryptogrid.exp_common import BASELINE, Store, get_device, make_ctx, pos_weight_of, run_nn_protocol
from cryptogrid.exp_models import PlmHead


def make_plm_prepare(ctx, store, device):
    def prepare(fit_ids):
        pw = torch.tensor(pos_weight_of(store, fit_ids), device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)

        def batch_fn(pid):
            return {"xe": torch.from_numpy(store.esm(pid).astype(np.float32)).to(device),
                    "y": torch.from_numpy(store.labels(pid)[0].astype(np.float32)).to(device)}

        return batch_fn, (lambda out, b: bce(out["z"], b["y"]))
    return prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    hp = ctx.cfg["training"]
    store, device = Store(ctx), get_device(ctx.cfg)
    d_esm = store.esm(ctx.train_ids[0]).shape[1]
    print(f"[device] {device} | train={len(ctx.train_ids)} test={len(ctx.test_ids)} folds={ctx.fold_names}")

    run_nn_protocol(
        ctx, BASELINE,
        prepare=make_plm_prepare(ctx, store, device),
        build_model=lambda: PlmHead(d_esm, int(hp["hidden_plm"]), float(hp["dropout"])),
        seeds=args.seeds, epochs=int(hp["epochs"]), lr=float(hp["learning_rate"]),
        weight_decay=float(hp["weight_decay"]), resume=not args.no_resume,
        meta={"model": "plm_only MLP head on frozen ESM-2", "loss": "BCE pos_weight"},
    )


if __name__ == "__main__":
    main()
