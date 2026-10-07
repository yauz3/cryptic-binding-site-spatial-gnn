#!/usr/bin/env python3
from __future__ import annotations

"""
Leakage-aware residual / gated fusion for CryptoBench GridFusion.

What this script does
---------------------
1) Reuses the existing pLM-only and Grid-only architectures from
   scripts/03_train_hybrid_models.py.
2) Builds 4-fold out-of-fold (OOF) predictions on the 885 CryptoBench
   training proteins. The official 222-protein test set is NOT used
   for alpha selection or gate fitting.
3) Selects a conservative scalar residual weight on OOF predictions:
       z_final = z_plm + alpha * support * (z_grid - z_plm)
4) Fits a residue-level gate on OOF predictions and structural-support
   features:
       z_final = z_plm + gate(x) * support * (z_grid - z_plm)
5) Only after all train-side choices are frozen, evaluates the already
   existing full-train pLM-only and Grid-only predictions on the test set.
6) Writes predictions, per-protein metrics, summaries, alpha search,
   gate diagnostics, and paired protein-bootstrap deltas.

Important
---------
- ESM and structural caches are reused; they are NOT regenerated.
- Existing pLM-only / Grid-only test predictions are reused.
- OOF fold predictions are cached per fold, so interrupted runs can resume.
- Use --rebuild-oof to force retraining the OOF base models.
"""

import argparse
import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from cryptogrid.common import (
    load_config,
    ensure_dir,
    load_folds,
    split_ids,
    best_threshold,
    metric_summary,
    per_protein_metrics,
    seed_everything,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def load_train_module():
    """Dynamically load scripts/03_train_hybrid_models.py."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "03_train_hybrid_models.py",
        Path.cwd() / "scripts" / "03_train_hybrid_models.py",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            "Could not find scripts/03_train_hybrid_models.py. "
            "Place this file inside the suite's scripts/ directory "
            "or run it from the suite root."
        )

    spec = importlib.util.spec_from_file_location("gridfusion_train03", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge_pair(plm: pd.DataFrame, grid: pd.DataFrame, fold_name: str) -> pd.DataFrame:
    keys = ["pdb_id", "residue_key"]
    a = plm[keys + ["y_true", "y_score"]].rename(
        columns={"y_true": "y_plm", "y_score": "p_plm"}
    )
    b = grid[keys + ["y_true", "y_score"]].rename(
        columns={"y_true": "y_grid", "y_score": "p_grid"}
    )
    m = a.merge(b, on=keys, how="inner", validate="one_to_one")
    if len(m) != len(a) or len(m) != len(b):
        raise RuntimeError(
            f"{fold_name}: pLM/Grid prediction merge lost rows "
            f"(plm={len(a)}, grid={len(b)}, merged={len(m)})"
        )
    if not np.array_equal(m["y_plm"].to_numpy(), m["y_grid"].to_numpy()):
        raise RuntimeError(f"{fold_name}: pLM/Grid y_true mismatch")
    m["y_true"] = m["y_plm"].astype(int)
    m.drop(columns=["y_plm", "y_grid"], inplace=True)
    m["fold"] = fold_name
    return m


def build_support_table(ids, struct_dir: Path, feature_names_path: Path) -> pd.DataFrame:
    names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    name_to_idx = {n: i for i, n in enumerate(names)}
    summary_names = [n for n in names if n.startswith("summary::")]
    raw_idx = np.array(
        [i for i, n in enumerate(names) if not n.startswith("summary::")],
        dtype=int,
    )

    membership_name = "summary::pocket_membership_count"
    membership_idx = name_to_idx.get(membership_name)

    frames = []
    for pid in ids:
        z = np.load(struct_dir / f"{pid}.npz", allow_pickle=False)
        rk = z["residue_keys"].astype(str)
        xs = z["X_struct"].astype(np.float32)

        if raw_idx.size:
            finite_fraction = np.isfinite(xs[:, raw_idx]).mean(axis=1).astype(np.float32)
        else:
            finite_fraction = np.zeros(len(rk), dtype=np.float32)

        if membership_idx is not None:
            membership = np.nan_to_num(
                xs[:, membership_idx], nan=0.0, posinf=0.0, neginf=0.0
            )
            support = (membership > 0).astype(np.float32)
        else:
            # Fallback: any finite non-summary structural descriptor.
            support = (finite_fraction > 0).astype(np.float32)

        data = {
            "pdb_id": np.repeat(str(pid), len(rk)),
            "residue_key": rk,
            "finite_fraction": finite_fraction,
            "struct_support": support,
        }

        for n in summary_names:
            v = xs[:, name_to_idx[n]]
            data[n] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )

        frames.append(pd.DataFrame(data))

    if not frames:
        raise RuntimeError("No structural support rows were built.")
    return pd.concat(frames, ignore_index=True)


def attach_support(
    pred: pd.DataFrame,
    struct_dir: Path,
    feature_names_path: Path,
) -> pd.DataFrame:
    ids = sorted(pred["pdb_id"].astype(str).unique().tolist())
    sup = build_support_table(ids, struct_dir, feature_names_path)
    out = pred.merge(
        sup,
        on=["pdb_id", "residue_key"],
        how="left",
        validate="one_to_one",
    )
    if out["struct_support"].isna().any():
        bad = out.loc[out["struct_support"].isna(), "pdb_id"].unique()[:10]
        raise RuntimeError(f"Missing structural support rows; examples={bad.tolist()}")
    return out


def macro_scores(df: pd.DataFrame, score_col: str) -> dict:
    aucs, aps = [], []
    for _, g in df.groupby("pdb_id", sort=False):
        y = g["y_true"].to_numpy(dtype=int)
        p = g[score_col].to_numpy(dtype=float)
        aps.append(average_precision_score(y, p))
        if len(np.unique(y)) >= 2:
            aucs.append(roc_auc_score(y, p))
    return {
        "macro_roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "macro_average_precision": float(np.mean(aps)) if aps else float("nan"),
    }


def pooled_scores(df: pd.DataFrame, score_col: str) -> dict:
    y = df["y_true"].to_numpy(dtype=int)
    p = df[score_col].to_numpy(dtype=float)
    return {
        "pooled_roc_auc": float(roc_auc_score(y, p)),
        "pooled_average_precision": float(average_precision_score(y, p)),
    }


def score_frame(df: pd.DataFrame, score_col: str = "y_score") -> dict:
    out = {}
    out.update(macro_scores(df, score_col))
    out.update(pooled_scores(df, score_col))
    return out


# ---------------------------------------------------------------------
# OOF base-model predictions
# ---------------------------------------------------------------------

def build_oof(
    trainmod,
    cfg,
    train_ids,
    train_folds,
    avail,
    struct_dir,
    esm_dir,
    device,
    outdir,
    rebuild=False,
):
    fold_names = sorted(
        [k for k, v in train_folds.items() if isinstance(v, (list, tuple)) and len(v) > 0]
    )
    if len(fold_names) < 2:
        raise RuntimeError(
            f"Expected multiple official train folds; found {fold_names}"
        )

    all_parts = []
    full_train_set = set(train_ids)

    print("\n" + "=" * 88)
    print("BUILDING / LOADING OOF BASE PREDICTIONS")
    print("=" * 88)
    print("folds:", fold_names)

    for fi, fold_name in enumerate(fold_names):
        cache_csv = outdir / f"oof_{fold_name}.csv"

        if cache_csv.exists() and not rebuild:
            part = pd.read_csv(cache_csv)
            needed = {"pdb_id", "residue_key", "y_true", "p_plm", "p_grid", "fold"}
            if needed.issubset(part.columns):
                print(f"[OOF cache] {fold_name}: {len(part):,} residues")
                all_parts.append(part)
                continue
            print(f"[OOF cache invalid] rebuilding {fold_name}")

        val_ids = [x for x in train_folds[fold_name] if x in avail]
        val_set = set(val_ids)
        fit_ids = [x for x in train_ids if x in full_train_set and x not in val_set]

        if not fit_ids or not val_ids:
            raise RuntimeError(
                f"{fold_name}: empty fit/validation split "
                f"(fit={len(fit_ids)}, val={len(val_ids)})"
            )

        print(
            f"\n[{fold_name}] fit proteins={len(fit_ids)} "
            f"validation proteins={len(val_ids)}"
        )

        # Fold-specific reproducible initialization.
        seed = int(cfg["training"]["seed"]) + fi
        seed_everything(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        plm_model, plm_mu, plm_sd = trainmod.train_model(
            "plm_only", fit_ids, struct_dir, esm_dir, cfg, device
        )
        plm_pred = trainmod.predict(
            plm_model, val_ids, struct_dir, esm_dir, plm_mu, plm_sd, device
        )
        del plm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seed_everything(seed + 1000)
        torch.manual_seed(seed + 1000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 1000)

        grid_model, grid_mu, grid_sd = trainmod.train_model(
            "grid_only", fit_ids, struct_dir, esm_dir, cfg, device
        )
        grid_pred = trainmod.predict(
            grid_model, val_ids, struct_dir, esm_dir, grid_mu, grid_sd, device
        )
        del grid_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        part = merge_pair(plm_pred, grid_pred, fold_name)
        part.to_csv(cache_csv, index=False)
        print(f"[OOF saved] {cache_csv} ({len(part):,} residues)")
        all_parts.append(part)

    oof = pd.concat(all_parts, ignore_index=True)

    # Every training protein should occur in exactly one official OOF fold.
    got = set(oof["pdb_id"].astype(str).unique())
    expected = set(train_ids)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise RuntimeError(
            f"OOF protein mismatch: missing={missing[:10]} extra={extra[:10]}"
        )

    dup = oof.duplicated(["pdb_id", "residue_key"]).sum()
    if dup:
        raise RuntimeError(f"OOF contains {dup} duplicated residue keys")

    oof.to_csv(outdir / "oof_predictions.csv", index=False)
    return oof, fold_names


# ---------------------------------------------------------------------
# Scalar residual
# ---------------------------------------------------------------------

def apply_scalar(df: pd.DataFrame, alpha: float) -> np.ndarray:
    lp = logit_np(df["p_plm"].to_numpy())
    lg = logit_np(df["p_grid"].to_numpy())
    support = df["struct_support"].to_numpy(dtype=float)
    lf = lp + float(alpha) * support * (lg - lp)
    return sigmoid_np(lf)


def select_alpha(oof: pd.DataFrame, fold_names, alpha_max: float, alpha_step: float):
    alphas = np.arange(0.0, alpha_max + alpha_step * 0.5, alpha_step)
    rows = []

    for alpha in alphas:
        tmp = oof[["pdb_id", "y_true", "fold"]].copy()
        tmp["score"] = apply_scalar(oof, float(alpha))

        fold_ap = []
        fold_auc = []
        for fold in fold_names:
            d = tmp[tmp["fold"] == fold]
            s = macro_scores(d, "score")
            fold_ap.append(s["macro_average_precision"])
            fold_auc.append(s["macro_roc_auc"])

        all_s = macro_scores(tmp, "score")
        rows.append(
            {
                "alpha": float(alpha),
                "mean_fold_macro_ap": float(np.nanmean(fold_ap)),
                "mean_fold_macro_auc": float(np.nanmean(fold_auc)),
                "oof_macro_ap": all_s["macro_average_precision"],
                "oof_macro_auc": all_s["macro_roc_auc"],
            }
        )

    tab = pd.DataFrame(rows)

    # Primary objective: mean macro AP across official folds.
    # Tie-break: higher macro AUC, then smaller alpha (more conservative).
    best = (
        tab.sort_values(
            ["mean_fold_macro_ap", "mean_fold_macro_auc", "alpha"],
            ascending=[False, False, True],
        )
        .iloc[0]
    )
    return float(best["alpha"]), tab


# ---------------------------------------------------------------------
# Gated residual
# ---------------------------------------------------------------------

class GateNet(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32, dropout: float = 0.10,
                 gate_max: float = 1.0, init_gate: float = 0.05):
        super().__init__()
        h2 = max(8, hidden // 2)
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, h2),
            nn.ReLU(),
        )
        self.out = nn.Linear(h2, 1)
        self.gate_max = float(gate_max)

        # Conservative start close to pLM-only.
        target_sigmoid = np.clip(init_gate / max(self.gate_max, 1e-8), 1e-4, 1 - 1e-4)
        bias = math.log(target_sigmoid / (1.0 - target_sigmoid))
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, bias)

    def forward(self, x):
        return self.gate_max * torch.sigmoid(self.out(self.body(x)).squeeze(-1))


def gate_feature_columns(df: pd.DataFrame):
    summary_cols = sorted([c for c in df.columns if c.startswith("summary::")])
    return summary_cols


def make_gate_matrix(df: pd.DataFrame, summary_cols):
    pp = np.clip(df["p_plm"].to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)
    pg = np.clip(df["p_grid"].to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)
    lp = logit_np(pp)
    lg = logit_np(pg)

    cols = [
        lp,
        lg,
        lg - lp,
        np.abs(lg - lp),
        np.abs(pp - 0.5) * 2.0,
        np.abs(pg - 0.5) * 2.0,
        df["finite_fraction"].to_numpy(dtype=np.float64),
        df["struct_support"].to_numpy(dtype=np.float64),
    ]
    names = [
        "logit_plm",
        "logit_grid",
        "delta_logit",
        "abs_delta_logit",
        "plm_confidence",
        "grid_confidence",
        "finite_fraction",
        "struct_support",
    ]

    for c in summary_cols:
        cols.append(
            np.nan_to_num(
                df[c].to_numpy(dtype=np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        names.append(c)

    X = np.column_stack(cols).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def fit_scaler(X: np.ndarray):
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.std(axis=0, dtype=np.float64)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def train_gate(
    df: pd.DataFrame,
    summary_cols,
    device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden: int,
    dropout: float,
    gate_max: float,
    init_gate: float,
    gate_l1: float,
):
    seed_everything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X, feature_names = make_gate_matrix(df, summary_cols)
    mu, sd = fit_scaler(X)
    Xn = (X - mu) / sd

    y = df["y_true"].to_numpy(dtype=np.float32)
    support = df["struct_support"].to_numpy(dtype=np.float32)
    lp = logit_np(df["p_plm"].to_numpy()).astype(np.float32)
    lg = logit_np(df["p_grid"].to_numpy()).astype(np.float32)

    X_t = torch.from_numpy(Xn).to(device)
    y_t = torch.from_numpy(y).to(device)
    s_t = torch.from_numpy(support).to(device)
    lp_t = torch.from_numpy(lp).to(device)
    lg_t = torch.from_numpy(lg).to(device)

    model = GateNet(
        Xn.shape[1],
        hidden=hidden,
        dropout=dropout,
        gate_max=gate_max,
        init_gate=init_gate,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    pw = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    n = len(df)
    batch_size = min(int(batch_size), n)

    for ep in range(1, int(epochs) + 1):
        model.train()
        perm = torch.randperm(n, generator=rng)
        losses = []
        gate_means = []

        for start in range(0, n, batch_size):
            idx_cpu = perm[start : start + batch_size]
            idx = idx_cpu.to(device)

            opt.zero_grad(set_to_none=True)
            gate = model(X_t[idx]) * s_t[idx]
            final_logit = lp_t[idx] + gate * (lg_t[idx] - lp_t[idx])

            loss = bce(final_logit, y_t[idx])
            if gate_l1 > 0:
                loss = loss + float(gate_l1) * gate.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))
            gate_means.append(float(gate.detach().mean().cpu()))

        if ep == 1 or ep == epochs or ep % max(1, epochs // 5) == 0:
            print(
                f"[gate] epoch {ep}/{epochs} "
                f"loss={np.mean(losses):.6f} "
                f"mean_gate={np.mean(gate_means):.4f}"
            )

    return model, mu, sd, feature_names


@torch.inference_mode()
def predict_gate(model, df, summary_cols, mu, sd, device):
    X, _ = make_gate_matrix(df, summary_cols)
    Xn = (X - mu) / sd

    pp = df["p_plm"].to_numpy(dtype=np.float64)
    pg = df["p_grid"].to_numpy(dtype=np.float64)
    lp = logit_np(pp)
    lg = logit_np(pg)
    support = df["struct_support"].to_numpy(dtype=np.float64)

    model.eval()
    x_t = torch.from_numpy(Xn.astype(np.float32)).to(device)
    gate = model(x_t).cpu().numpy().astype(np.float64)
    gate *= support

    score = sigmoid_np(lp + gate * (lg - lp))
    return score, gate


def crossfit_gate(oof, fold_names, summary_cols, device, args, seed):
    pieces = []

    print("\n" + "=" * 88)
    print("CROSS-FITTED GATE DIAGNOSTIC")
    print("=" * 88)

    for fi, fold in enumerate(fold_names):
        tr = oof[oof["fold"] != fold].reset_index(drop=True)
        va = oof[oof["fold"] == fold].reset_index(drop=True)

        print(
            f"\n[gate CV {fold}] train residues={len(tr):,} "
            f"valid residues={len(va):,}"
        )

        model, mu, sd, _ = train_gate(
            tr,
            summary_cols,
            device,
            seed=seed + 5000 + fi,
            epochs=args.gate_epochs,
            batch_size=args.gate_batch_size,
            lr=args.gate_lr,
            weight_decay=args.gate_weight_decay,
            hidden=args.gate_hidden,
            dropout=args.gate_dropout,
            gate_max=args.gate_max,
            init_gate=args.init_gate,
            gate_l1=args.gate_l1,
        )

        score, gate = predict_gate(model, va, summary_cols, mu, sd, device)
        part = va[["pdb_id", "residue_key", "y_true", "fold"]].copy()
        part["y_score"] = score
        part["gate"] = gate
        pieces.append(part)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------
# Paired bootstrap on proteins
# ---------------------------------------------------------------------

def per_protein_auc_ap(df: pd.DataFrame, score_col: str):
    rows = []
    for pid, g in df.groupby("pdb_id", sort=False):
        y = g["y_true"].to_numpy(dtype=int)
        p = g[score_col].to_numpy(dtype=float)
        ap = average_precision_score(y, p)
        auc = roc_auc_score(y, p) if len(np.unique(y)) >= 2 else np.nan
        rows.append({"pdb_id": pid, "ap": ap, "auc": auc})
    return pd.DataFrame(rows)


def paired_bootstrap_delta(base_df, new_df, reps=1000, seed=42):
    a = per_protein_auc_ap(base_df, "y_score").rename(
        columns={"ap": "ap_base", "auc": "auc_base"}
    )
    b = per_protein_auc_ap(new_df, "y_score").rename(
        columns={"ap": "ap_new", "auc": "auc_new"}
    )
    m = a.merge(b, on="pdb_id", how="inner", validate="one_to_one")

    dap = (m["ap_new"] - m["ap_base"]).to_numpy(dtype=float)
    dauc = (m["auc_new"] - m["auc_base"]).to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    n = len(m)
    boot_ap, boot_auc = [], []

    for _ in range(int(reps)):
        idx = rng.integers(0, n, size=n)
        boot_ap.append(float(np.nanmean(dap[idx])))
        boot_auc.append(float(np.nanmean(dauc[idx])))

    def summarize(obs, boot):
        boot = np.asarray(boot, dtype=float)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        return {
            "observed_mean_delta": float(np.nanmean(obs)),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "bootstrap_prob_delta_gt_0": float(np.nanmean(boot > 0)),
        }

    return {
        "n_proteins": int(n),
        "macro_average_precision_delta": summarize(dap, boot_ap),
        "macro_roc_auc_delta": summarize(dauc, boot_auc),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--rebuild-oof", action="store_true")
    ap.add_argument("--alpha-max", type=float, default=0.50)
    ap.add_argument("--alpha-step", type=float, default=0.025)

    # Gate hyperparameters are intentionally fixed on train-side data.
    ap.add_argument("--gate-epochs", type=int, default=20)
    ap.add_argument("--gate-batch-size", type=int, default=8192)
    ap.add_argument("--gate-lr", type=float, default=3e-4)
    ap.add_argument("--gate-weight-decay", type=float, default=1e-4)
    ap.add_argument("--gate-hidden", type=int, default=32)
    ap.add_argument("--gate-dropout", type=float, default=0.10)
    ap.add_argument("--gate-max", type=float, default=1.0)
    ap.add_argument("--init-gate", type=float, default=0.05)
    ap.add_argument("--gate-l1", type=float, default=1e-4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["training"]["seed"])
    seed_everything(seed)

    if torch.cuda.is_available() and cfg["training"].get("device", "auto") != "cpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("[device]", device)

    struct_dir = Path(cfg["work"]["struct_cache"])
    esm_dir = Path(cfg["work"]["esm_cache"])
    results_root = Path(cfg["work"]["results"])
    predroot = results_root / "predictions"
    outdir = ensure_dir(results_root / "residual_gated")

    feature_names_path = struct_dir / "feature_names.json"
    if not feature_names_path.exists():
        raise FileNotFoundError(feature_names_path)

    trainmod = load_train_module()

    folds = load_folds(cfg["project"]["folds_json"])
    train_ids, test_ids, train_folds = split_ids(folds)
    avail = trainmod.available_ids(struct_dir, esm_dir)
    train_ids = [x for x in train_ids if x in avail]
    test_ids = [x for x in test_ids if x in avail]

    print(
        f"[data] train proteins={len(train_ids)} "
        f"test proteins={len(test_ids)} available={len(avail)}"
    )

    # --------------------------------------------------------------
    # 1) OOF base predictions. No test scores are read yet.
    # --------------------------------------------------------------
    oof, fold_names = build_oof(
        trainmod,
        cfg,
        train_ids,
        train_folds,
        avail,
        struct_dir,
        esm_dir,
        device,
        Path(outdir),
        rebuild=args.rebuild_oof,
    )

    print("\n[OOF] attaching structural-support features...")
    oof = attach_support(oof, struct_dir, feature_names_path)
    oof.to_csv(Path(outdir) / "oof_predictions_with_support.csv", index=False)

    # --------------------------------------------------------------
    # 2) Scalar residual alpha selected ONLY on OOF train predictions.
    # --------------------------------------------------------------
    alpha, alpha_tab = select_alpha(
        oof,
        fold_names,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
    )
    alpha_tab.to_csv(Path(outdir) / "alpha_search_oof.csv", index=False)

    oof_scalar = oof[["pdb_id", "residue_key", "y_true", "fold"]].copy()
    oof_scalar["y_score"] = apply_scalar(oof, alpha)

    scalar_thr, scalar_thr_obj = best_threshold(
        oof_scalar.y_true,
        oof_scalar.y_score,
        cfg["training"].get("threshold_metric", "mcc"),
    )

    print("\n" + "=" * 88)
    print("OOF SCALAR RESIDUAL SELECTION")
    print("=" * 88)
    print(f"selected alpha={alpha:.4f}")
    print(f"OOF threshold={scalar_thr:.4f} objective={scalar_thr_obj:.6f}")
    print(json.dumps(score_frame(oof_scalar), indent=2))

    # --------------------------------------------------------------
    # 3) Cross-fitted gate diagnostic + train final gate on all OOF.
    # --------------------------------------------------------------
    summary_cols = gate_feature_columns(oof)
    print("\n[gate] summary/support columns:")
    for c in summary_cols:
        print("  ", c)

    gate_cv = crossfit_gate(
        oof,
        fold_names,
        summary_cols,
        device,
        args,
        seed,
    )
    gate_cv.to_csv(Path(outdir) / "gated_crossfit_oof_predictions.csv", index=False)

    gate_thr, gate_thr_obj = best_threshold(
        gate_cv.y_true,
        gate_cv.y_score,
        cfg["training"].get("threshold_metric", "mcc"),
    )

    print("\n" + "=" * 88)
    print("CROSS-FITTED GATE OOF")
    print("=" * 88)
    print(f"OOF threshold={gate_thr:.4f} objective={gate_thr_obj:.6f}")
    print(json.dumps(score_frame(gate_cv), indent=2))
    print(
        "[gate CV] mean gate=",
        float(gate_cv["gate"].mean()),
        "median gate=",
        float(gate_cv["gate"].median()),
    )

    print("\n[gate] fitting final gate on all OOF training predictions...")
    gate_model, gate_mu, gate_sd, gate_features = train_gate(
        oof,
        summary_cols,
        device,
        seed=seed + 9000,
        epochs=args.gate_epochs,
        batch_size=args.gate_batch_size,
        lr=args.gate_lr,
        weight_decay=args.gate_weight_decay,
        hidden=args.gate_hidden,
        dropout=args.gate_dropout,
        gate_max=args.gate_max,
        init_gate=args.init_gate,
        gate_l1=args.gate_l1,
    )

    torch.save(
        {
            "state_dict": gate_model.state_dict(),
            "feature_names": gate_features,
            "summary_cols": summary_cols,
            "feature_mean": gate_mu,
            "feature_std": gate_sd,
            "gate_hidden": args.gate_hidden,
            "gate_dropout": args.gate_dropout,
            "gate_max": args.gate_max,
            "threshold": gate_thr,
            "config": vars(args),
        },
        Path(outdir) / "gated_residual.pt",
    )

    selection = {
        "selected_alpha": alpha,
        "scalar_threshold": float(scalar_thr),
        "scalar_threshold_objective": float(scalar_thr_obj),
        "gate_threshold": float(gate_thr),
        "gate_threshold_objective": float(gate_thr_obj),
        "gate_features": gate_features,
        "fold_names": fold_names,
        "note": "All selection/fitting above used training-fold OOF predictions only.",
    }
    (Path(outdir) / "train_side_selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # 4) TEST: only now read existing full-train test predictions.
    # --------------------------------------------------------------
    plm_test_path = predroot / "plm_only_test_predictions.csv"
    grid_test_path = predroot / "grid_only_test_predictions.csv"
    if not plm_test_path.exists() or not grid_test_path.exists():
        raise FileNotFoundError(
            "Existing full-train test predictions are required:\n"
            f"  {plm_test_path}\n  {grid_test_path}\n"
            "Run scripts/03_train_hybrid_models.py first."
        )

    print("\n" + "=" * 88)
    print("FINAL TEST EVALUATION")
    print("=" * 88)
    print("Train-side choices are now frozen. Reading test predictions.")

    plm_test = pd.read_csv(plm_test_path)
    grid_test = pd.read_csv(grid_test_path)
    test_pair = merge_pair(plm_test, grid_test, "test")
    test_pair = attach_support(test_pair, struct_dir, feature_names_path)

    # Scalar
    scalar_test = test_pair[["pdb_id", "residue_key", "y_true"]].copy()
    scalar_test["y_score"] = apply_scalar(test_pair, alpha)
    scalar_test["method"] = "residual_scalar"
    scalar_test.to_csv(
        Path(outdir) / "residual_scalar_test_predictions.csv", index=False
    )

    scalar_sm = metric_summary(scalar_test, scalar_thr)
    scalar_sm["method"] = "residual_scalar"
    scalar_sm["selected_alpha"] = alpha
    scalar_sm["selection_source"] = "training_OOF_only"

    scalar_pp = per_protein_metrics(scalar_test, scalar_thr)
    scalar_pp["method"] = "residual_scalar"
    scalar_pp.to_csv(
        Path(outdir) / "residual_scalar_per_protein.csv", index=False
    )

    # Gate
    gate_score, gate_value = predict_gate(
        gate_model,
        test_pair,
        summary_cols,
        gate_mu,
        gate_sd,
        device,
    )
    gated_test = test_pair[["pdb_id", "residue_key", "y_true"]].copy()
    gated_test["y_score"] = gate_score
    gated_test["gate"] = gate_value
    gated_test["method"] = "gated_residual"
    gated_test.to_csv(
        Path(outdir) / "gated_residual_test_predictions.csv", index=False
    )

    gated_metric_input = gated_test[
        ["pdb_id", "residue_key", "y_true", "y_score"]
    ].copy()
    gate_sm = metric_summary(gated_metric_input, gate_thr)
    gate_sm["method"] = "gated_residual"
    gate_sm["selection_source"] = "training_OOF_only"
    gate_sm["mean_gate_test"] = float(np.mean(gate_value))
    gate_sm["median_gate_test"] = float(np.median(gate_value))
    gate_sm["zero_gate_fraction_test"] = float(np.mean(gate_value == 0))

    gate_pp = per_protein_metrics(gated_metric_input, gate_thr)
    gate_pp["method"] = "gated_residual"
    gate_pp.to_csv(
        Path(outdir) / "gated_residual_per_protein.csv", index=False
    )

    # Baseline test dataframe for paired bootstrap.
    base_test = plm_test[
        ["pdb_id", "residue_key", "y_true", "y_score"]
    ].copy()

    reps = int(cfg.get("evaluation", {}).get("bootstrap_reps", 1000))
    bootstrap = {
        "residual_scalar_vs_plm": paired_bootstrap_delta(
            base_test,
            scalar_test[["pdb_id", "residue_key", "y_true", "y_score"]],
            reps=reps,
            seed=seed + 111,
        ),
        "gated_residual_vs_plm": paired_bootstrap_delta(
            base_test,
            gated_metric_input,
            reps=reps,
            seed=seed + 222,
        ),
    }
    (Path(outdir) / "paired_bootstrap_vs_plm.json").write_text(
        json.dumps(bootstrap, indent=2),
        encoding="utf-8",
    )

    # Comparison summary.
    rows = []

    # Existing pLM metric JSON if available.
    plm_metric_json = results_root / "models" / "plm_only_metrics.json"
    if plm_metric_json.exists():
        p = json.loads(plm_metric_json.read_text(encoding="utf-8"))
        p["method"] = "plm_only"
        rows.append(p)
    else:
        p = score_frame(base_test)
        p["method"] = "plm_only"
        rows.append(p)

    dual_metric_json = results_root / "models" / "dual_tower_metrics.json"
    if dual_metric_json.exists():
        d = json.loads(dual_metric_json.read_text(encoding="utf-8"))
        d["method"] = "dual_tower"
        rows.append(d)

    rows.extend([scalar_sm, gate_sm])
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(Path(outdir) / "final_comparison.csv", index=False)

    (Path(outdir) / "residual_scalar_metrics.json").write_text(
        json.dumps(scalar_sm, indent=2), encoding="utf-8"
    )
    (Path(outdir) / "gated_residual_metrics.json").write_text(
        json.dumps(gate_sm, indent=2), encoding="utf-8"
    )

    print("\n--- residual_scalar ---")
    print(json.dumps(scalar_sm, indent=2))
    print("\n--- gated_residual ---")
    print(json.dumps(gate_sm, indent=2))
    print("\n--- paired bootstrap vs pLM ---")
    print(json.dumps(bootstrap, indent=2))

    print("\n[DONE]")
    print("Outputs ->", Path(outdir))
    print("Main table ->", Path(outdir) / "final_comparison.csv")


if __name__ == "__main__":
    main()
