#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cryptogrid.common import (
    load_config, ensure_dir, load_folds, split_ids, best_threshold,
    metric_summary, per_protein_metrics, seed_everything,
)


class ResidueMLP(nn.Module):
    def __init__(self, variant, d_esm, d_struct, hp):
        super().__init__()
        drop = float(hp["dropout"])
        hplm = int(hp["hidden_plm"])
        hst = int(hp["hidden_struct"])
        hfu = int(hp["hidden_fusion"])
        self.variant = variant
        if variant == "plm_only":
            self.net = nn.Sequential(
                nn.Linear(d_esm, hplm), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hplm, hplm), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hplm, 1),
            )
        elif variant == "grid_only":
            h2 = max(64, hst // 2)
            self.net = nn.Sequential(
                nn.Linear(d_struct, hst), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hst, h2), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(h2, 1),
            )
        elif variant == "early_concat":
            self.net = nn.Sequential(
                nn.Linear(d_esm + d_struct, hfu), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hfu, hfu), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hfu, 1),
            )
        elif variant == "dual_tower":
            self.esm_tower = nn.Sequential(
                nn.Linear(d_esm, hplm), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hplm, hplm), nn.ReLU(),
            )
            self.struct_tower = nn.Sequential(
                nn.Linear(d_struct, hst), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hst, hst), nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(hplm + hst, hfu), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hfu, 1),
            )
        else:
            raise ValueError(variant)

    def forward(self, xe, xs):
        if self.variant == "plm_only":
            return self.net(xe).squeeze(-1)
        if self.variant == "grid_only":
            return self.net(xs).squeeze(-1)
        if self.variant == "early_concat":
            return self.net(torch.cat([xe, xs], dim=-1)).squeeze(-1)
        return self.head(torch.cat([self.esm_tower(xe), self.struct_tower(xs)], dim=-1)).squeeze(-1)


def available_ids(struct_dir, esm_dir):
    s = {p.stem for p in Path(struct_dir).glob("*.npz")}
    e = {p.stem for p in Path(esm_dir).glob("*.npz")}
    return s & e


def load_one(pid, struct_dir, esm_dir, mean=None, std=None):
    a = np.load(Path(struct_dir) / f"{pid}.npz", allow_pickle=False)
    b = np.load(Path(esm_dir) / f"{pid}.npz", allow_pickle=False)
    rk1 = a["residue_keys"].astype(str)
    rk2 = b["residue_keys"].astype(str)
    if len(rk1) != len(rk2) or not np.array_equal(rk1, rk2):
        raise ValueError(f"{pid}: residue cache mismatch")
    xs = a["X_struct"].astype(np.float32)
    if mean is not None:
        xs = np.where(np.isfinite(xs), xs, mean)
        xs = (xs - mean) / std
        xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    xe = b["X_esm"].astype(np.float32)
    y = a["y"].astype(np.float32)
    return xe, xs, y, rk1


def stream_struct_stats(ids, struct_dir):
    sumv = sumsq = count = None
    for pid in ids:
        z = np.load(Path(struct_dir) / f"{pid}.npz", allow_pickle=False)
        x = z["X_struct"].astype(np.float64)
        if sumv is None:
            d = x.shape[1]
            sumv = np.zeros(d)
            sumsq = np.zeros(d)
            count = np.zeros(d)
        m = np.isfinite(x)
        xx = np.where(m, x, 0.0)
        sumv += xx.sum(0)
        sumsq += (xx * xx).sum(0)
        count += m.sum(0)
    if sumv is None:
        raise RuntimeError("No training structural cache available")
    count_safe = np.maximum(count, 1)
    mean = sumv / count_safe
    var = np.maximum(sumsq / count_safe - mean * mean, 1e-8)
    std = np.sqrt(var)
    mean[count == 0] = 0.0
    std[count == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def pos_weight(ids, struct_dir):
    pos = neg = 0
    for pid in ids:
        y = np.load(Path(struct_dir) / f"{pid}.npz", allow_pickle=False)["y"]
        pos += int(y.sum())
        neg += int(len(y) - y.sum())
    return float(neg / max(1, pos))


def train_model(variant, ids, struct_dir, esm_dir, cfg, device, epochs=None):
    hp = cfg["training"]
    mean, std = stream_struct_stats(ids, struct_dir)
    xe, xs, _, _ = load_one(ids[0], struct_dir, esm_dir, mean, std)
    model = ResidueMLP(variant, xe.shape[1], xs.shape[1], hp).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(hp["learning_rate"]), weight_decay=float(hp["weight_decay"])
    )
    pw = torch.tensor(pos_weight(ids, struct_dir), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    epochs = int(epochs or hp["epochs"])
    rng = random.Random(int(hp["seed"]))

    for ep in range(1, epochs + 1):
        order = list(ids)
        rng.shuffle(order)
        model.train()
        losses = []
        for pid in order:
            xe, xs, y, _ = load_one(pid, struct_dir, esm_dir, mean, std)
            te = torch.from_numpy(xe).to(device)
            ts = torch.from_numpy(xs).to(device)
            ty = torch.from_numpy(y).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(te, ts)
            loss = loss_fn(logits, ty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[{variant}] epoch {ep}/{epochs} loss={np.mean(losses):.5f}")
    return model, mean, std


@torch.inference_mode()
def predict(model, ids, struct_dir, esm_dir, mean, std, device):
    model.eval()
    rows = []
    for pid in ids:
        xe, xs, y, rk = load_one(pid, struct_dir, esm_dir, mean, std)
        logits = model(torch.from_numpy(xe).to(device), torch.from_numpy(xs).to(device))
        score = torch.sigmoid(logits).cpu().numpy()
        rows.extend(
            {"pdb_id": pid, "residue_key": r, "y_true": int(yy), "y_score": float(ss)}
            for r, yy, ss in zip(rk, y, score)
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["training"]["seed"])
    seed_everything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and cfg["training"].get("device", "auto") != "cpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("[device]", device)

    struct_dir = Path(cfg["work"]["struct_cache"])
    esm_dir = Path(cfg["work"]["esm_cache"])
    outroot = ensure_dir(Path(cfg["work"]["results"]) / "models")
    predroot = ensure_dir(Path(cfg["work"]["results"]) / "predictions")

    folds = load_folds(cfg["project"]["folds_json"])
    train_ids, test_ids, train_folds = split_ids(folds)
    avail = available_ids(struct_dir, esm_dir)
    train_ids = [x for x in train_ids if x in avail]
    test_ids = [x for x in test_ids if x in avail]
    if not train_ids or not test_ids:
        raise RuntimeError("No matched train/test cache files. Check folds.json and caches.")

    cal_name = cfg["training"].get("calibration_fold", "train_3")
    cal_ids = [x for x in train_folds.get(cal_name, []) if x in avail]
    fit_ids = [x for x in train_ids if x not in set(cal_ids)]
    if not fit_ids or not cal_ids:
        cut = max(1, int(len(train_ids) * 0.2))
        cal_ids = train_ids[-cut:]
        fit_ids = train_ids[:-cut]

    variants = args.variants or cfg["training"]["variants"]
    summaries = []
    for variant in variants:
        print("\n" + "=" * 90 + f"\n{variant}")
        m0, mu0, sd0 = train_model(variant, fit_ids, struct_dir, esm_dir, cfg, device)
        val = predict(m0, cal_ids, struct_dir, esm_dir, mu0, sd0, device)
        thr, val_obj = best_threshold(
            val.y_true, val.y_score, cfg["training"].get("threshold_metric", "mcc")
        )
        print(f"[{variant}] calibrated threshold={thr:.4f} objective={val_obj:.4f}")

        model, mu, sd = train_model(variant, train_ids, struct_dir, esm_dir, cfg, device)
        test = predict(model, test_ids, struct_dir, esm_dir, mu, sd, device)
        test["method"] = variant
        test.to_csv(predroot / f"{variant}_test_predictions.csv", index=False)

        sm = metric_summary(test, thr)
        sm["method"] = variant
        sm["calibration_objective"] = val_obj
        pp = per_protein_metrics(test, thr)
        pp["method"] = variant
        pp.to_csv(predroot / f"{variant}_per_protein.csv", index=False)
        summaries.append(sm)

        torch.save(
            {
                "state_dict": model.state_dict(), "variant": variant,
                "struct_mean": mu, "struct_std": sd,
                "threshold": thr, "config": cfg["training"],
            },
            outroot / f"{variant}.pt",
        )
        (outroot / f"{variant}_metrics.json").write_text(json.dumps(sm, indent=2), encoding="utf-8")
        print(json.dumps(sm, indent=2))

    pd.DataFrame(summaries).to_csv(predroot / "hybrid_summary.csv", index=False)
    print("[DONE]")


if __name__ == "__main__":
    main()
