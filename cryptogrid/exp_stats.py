"""
Statistics / reporting helpers for the publication scripts 20-22.

Units of analysis (keep them apart in the manuscript)
------------------------------------------------------
* PROTEIN is the unit for every generalisation claim ("better on unseen proteins"):
  paired sign-flip permutation p, bootstrap CI, Cohen's d_z computed over proteins.
* SEED is the unit only for training-stochasticity claims ("the gain does not depend on
  initialisation"): mean +- SD over seeds, seeds improved k/S, seed-level d_z.
  A seed-level p-value / Cohen's d does NOT measure generalisation: all seeds share the
  same proteins, so it becomes arbitrarily extreme as S grows.
* The crossed bootstrap (resample proteins AND seeds) gives one CI that carries both
  sources of uncertainty.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from .exp_common import (METRICS, criterion_from_arrays, paired_stats, protein_slices, sha256_file)

LABELS = {
    "base_plm": "ESM-2 MLP (pLM anchor)",
    "gnn_anchor": "Anchored spatial GNN",
    "stack_anchor": "Anchored stack",
    "smooth3d": "3D logit smoothing",
    "pocket_consensus": "Grid pocket consensus",
    "gbm_resid_grid_geom_ctx": "GBM residual (Grid+geom+ctx)",
    "rank_softbedroc_lam1": "Soft-BEDROC loss",
    "rank_toppair_lam0.5": "Top-pair ranking loss",
    "aux_dist_s5_g0.5": "Aux. distance task",
    "ablation_gnn_no_graph": "GNN w/o message passing",
    "ablation_gnn_no_geom": "GNN w/o geometry features",
    "ablation_gnn_no_bfactor": "GNN w/o B-factor features",
    "smoothseq_ablation": "Sequence-window smoothing",
    "pocket_consensus_shuffled_ablation": "Shuffled-pocket consensus",
    "ablation_gbm_support_only": "GBM residual (support only)",
    "ablation_gbm_no_grid": "GBM residual (geom+ctx, no Grid)",
    "ablation_gbm_grid_only": "GBM residual (Grid only)",
}
METRIC_LABELS = {"ap": "AP", "auc": "ROC-AUC", "ef1": "EF@1%", "ef5": "EF@5%", "bedroc20": "BEDROC$_{20}$",
                 "bedroc50": "BEDROC$_{50}$", "bedroc80": "BEDROC$_{80}$", "mcc": "MCC", "f1": "F1"}
# stack inputs that depend on Grid-Crypto output (pocket lists or pocket descriptors)
GRID_DERIVED = ("pocket_consensus", "gbm_resid_grid")


def label(name: str) -> str:
    return LABELS.get(name, name)


# =====================================================================
# Effect sizes
# =====================================================================

def cohen_dz(d: np.ndarray) -> Tuple[float, float]:
    """Paired standardised effect d_z = mean(diff) / SD(diff), and Hedges-corrected g_z."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return float("nan"), float("nan")
    dz = d.mean() / sd
    return float(dz), float(dz * (1.0 - 3.0 / (4.0 * (n - 1) - 1.0)))


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (effect size of the Wilcoxon signed-rank test)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    r = sps.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def effect_sizes(d: np.ndarray, n_boot: int = 5000, seed: int = 0) -> Dict[str, float]:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    dz, gz = cohen_dz(d)
    out = {"cohen_dz": dz, "hedges_gz": gz, "dz_ci95_low": np.nan, "dz_ci95_high": np.nan,
           "rank_biserial": rank_biserial(d), "p_ttest": np.nan,
           "prob_superiority": float(((d > 0).sum() + 0.5 * (d == 0).sum()) / n) if n else np.nan}
    if n >= 3 and np.isfinite(dz):
        out["p_ttest"] = float(sps.ttest_1samp(d, 0.0).pvalue)
        rng = np.random.default_rng(seed)
        b = d[rng.integers(0, n, size=(n_boot, n))]
        sd = b.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            bz = b.mean(axis=1) / sd
        bz = bz[np.isfinite(bz)]
        if len(bz):
            out["dz_ci95_low"], out["dz_ci95_high"] = (float(x) for x in np.percentile(bz, [2.5, 97.5]))
    return out


def full_paired(pp_base: pd.DataFrame, pp_new: pd.DataFrame, metrics: Sequence[str] = METRICS,
                n_perm: int = 100000, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """paired_stats (mean delta, CI, sign-flip p, Wilcoxon p, Holm) + effect sizes, protein-level."""
    pp_new = pp_new.reindex(pp_base.index)
    st = paired_stats(pp_base, pp_new, metrics=metrics, n_perm=n_perm, n_boot=n_boot, seed=seed)
    extra = [effect_sizes(pp_new[m].to_numpy(dtype=float) - pp_base[m].to_numpy(dtype=float), n_boot, seed + 7)
             for m in metrics]
    return pd.concat([st, pd.DataFrame(extra)], axis=1)


def macro_ci(v: np.ndarray, n_boot: int = 5000, seed: int = 0) -> Tuple[float, float, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


# =====================================================================
# Crossed (protein x seed) analysis for replicate experiments
# =====================================================================

def crossed_bootstrap_ci(D: np.ndarray, n_boot: int = 5000, seed: int = 0) -> Tuple[float, float, float]:
    """D: (n_seeds, n_proteins) paired deltas. Resamples seeds AND proteins (pigeonhole bootstrap)."""
    D = np.asarray(D, dtype=float)
    S, P = D.shape
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        out[b] = np.nanmean(D[np.ix_(rng.integers(0, S, S), rng.integers(0, P, P))])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(np.nanmean(D)), float(lo), float(hi)


def variance_components(D: np.ndarray) -> Dict[str, float]:
    """Two-way crossed random-effects decomposition of D[seed, protein] (one value per cell)."""
    D = np.asarray(D, dtype=float)
    keep = np.isfinite(D).all(axis=0)
    D = D[:, keep]
    S, P = D.shape
    gm = D.mean()
    if S < 2 or P < 2:
        return {"sd_protein": np.nan, "sd_seed": np.nan, "sd_residual": np.nan, "se_mean_total": np.nan,
                "share_var_of_mean_from_seed": np.nan}
    ms_seed = P * ((D.mean(axis=1) - gm) ** 2).sum() / (S - 1)
    ms_prot = S * ((D.mean(axis=0) - gm) ** 2).sum() / (P - 1)
    resid = D - D.mean(axis=1, keepdims=True) - D.mean(axis=0, keepdims=True) + gm
    ms_res = (resid ** 2).sum() / ((S - 1) * (P - 1))
    v_seed, v_prot = max(0.0, (ms_seed - ms_res) / P), max(0.0, (ms_prot - ms_res) / S)
    var_mean = v_prot / P + v_seed / S + ms_res / (P * S)
    return {"sd_protein": float(np.sqrt(v_prot)), "sd_seed": float(np.sqrt(v_seed)),
            "sd_residual": float(np.sqrt(ms_res)), "se_mean_total": float(np.sqrt(var_mean)),
            "share_var_of_mean_from_seed": float((v_seed / S) / var_mean) if var_mean > 0 else np.nan}


def seed_level_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """a, b: per-seed macro metric for baseline / method (paired by seed). Training-stochasticity only."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    S = len(d)
    dz, gz = cohen_dz(d)
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0) if S > 1 else np.nan
    out = {"n_seeds": S, "seed_mean_delta": float(d.mean()), "seed_sd_delta": float(d.std(ddof=1)) if S > 1 else np.nan,
           "seeds_improved": int((d > 0).sum()), "seed_cohen_dz": dz, "seed_hedges_gz": gz,
           "seed_cohen_d_av": float(d.mean() / pooled_sd) if pooled_sd and pooled_sd > 0 else np.nan,
           "seed_p_ttest": np.nan, "seed_p_wilcoxon": np.nan, "seed_p_sign": np.nan}
    if S >= 3 and np.any(d != 0):
        out["seed_p_ttest"] = float(sps.ttest_1samp(d, 0.0).pvalue)
        try:
            out["seed_p_wilcoxon"] = float(sps.wilcoxon(d).pvalue)
        except ValueError:
            pass
        nz = int((d != 0).sum())
        out["seed_p_sign"] = float(sps.binomtest(int((d > 0).sum()), nz, 0.5).pvalue) if nz else 1.0
    return out


# =====================================================================
# Pooled and thresholded metrics (for comparability with the literature)
# =====================================================================

def pooled_compare(base: pd.DataFrame, new: pd.DataFrame, n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """Pooled (residue-level) ROC-AUC / AP with a protein-cluster bootstrap CI for the difference."""
    y = base["y_true"].to_numpy()
    zb, zn = base["z"].to_numpy(dtype=float), new["z"].to_numpy(dtype=float)
    sl = protein_slices(base["pdb_id"].to_numpy())
    fn = {"pooled_auc": roc_auc_score, "pooled_ap": average_precision_score}
    obs = {k: (f(y, zb), f(y, zn)) for k, f in fn.items()}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in fn}
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(sl[i][1], sl[i][2]) for i in rng.integers(0, len(sl), len(sl))])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for k, f in fn.items():
            boot[k].append(f(yy, zn[idx]) - f(yy, zb[idx]))
    rows = []
    for k in fn:
        lo, hi = (np.percentile(boot[k], [2.5, 97.5]) if boot[k] else (np.nan, np.nan))
        rows.append({"metric": k, "base": obs[k][0], "new": obs[k][1], "delta": obs[k][1] - obs[k][0],
                     "ci95_low": lo, "ci95_high": hi, "ci_method": f"protein-cluster bootstrap ({n_boot})"})
    return pd.DataFrame(rows)


def mcc_threshold(y: np.ndarray, z: np.ndarray) -> float:
    """Logit threshold maximising pooled MCC (exact search over all cut points)."""
    order = np.argsort(-z, kind="stable")
    ys = y[order].astype(float)
    P, N = ys.sum(), len(ys) - ys.sum()
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    fp = k - tp
    fn, tn = P - tp, N - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)
    return float(z[order][int(np.argmax(mcc))])


def _confusion(y, pred):
    tp = float(((pred == 1) & (y == 1)).sum()); fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum()); tn = float(((pred == 0) & (y == 0)).sum())
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return mcc, f1, prec, rec


def threshold_tables(df: pd.DataFrame, thr: np.ndarray | float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Per-protein MCC/F1 table and pooled MCC/F1/precision/recall at the given (row-wise) threshold."""
    y = df["y_true"].to_numpy().astype(int)
    pred = (df["z"].to_numpy(dtype=float) >= thr).astype(int)
    rows = []
    for pid, a, b in protein_slices(df["pdb_id"].to_numpy()):
        mcc, f1, _, _ = _confusion(y[a:b], pred[a:b])
        rows.append({"pdb_id": pid, "mcc": mcc, "f1": f1})
    mcc, f1, prec, rec = _confusion(y, pred)
    return pd.DataFrame(rows).set_index("pdb_id"), {"pooled_mcc": mcc, "pooled_f1": f1,
                                                    "pooled_precision": prec, "pooled_recall": rec}


def crossfit_thresholds(oof: pd.DataFrame, fold_names: Sequence[str]) -> np.ndarray:
    """Row-wise thresholds for OOF rows: the threshold for fold f is tuned on the other folds."""
    y, z, fold = oof["y_true"].to_numpy().astype(int), oof["z"].to_numpy(dtype=float), oof["fold"].to_numpy()
    thr = np.empty(len(oof))
    for f in fold_names:
        thr[fold == f] = mcc_threshold(y[fold != f], z[fold != f])
    return thr


# =====================================================================
# Nested anchored stacking (same algorithm as scripts/18)
# =====================================================================

STACK_LAMS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.001, 0.0)


def fit_stack(C, z0, y, lam):
    pw = (len(y) - y.sum()) / max(1.0, y.sum())
    w = np.where(y > 0, pw, 1.0)
    w = w / w.sum()

    def f(theta):
        z = z0 + theta[0] + C @ theta[1:]
        loss = float((w * (np.logaddexp(0.0, z) - y * z)).sum() + lam * (theta[1:] ** 2).sum())
        g = w * (expit(z) - y)
        return loss, np.r_[g.sum(), C.T @ g + 2.0 * lam * theta[1:]]

    k = C.shape[1]
    return minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B",
                    bounds=[(None, None)] + [(0.0, None)] * k, options={"maxiter": 500}).x


def nested_stack(z0, D, y, pid, fold, fold_names) -> np.ndarray:
    """Double-nested OOF logits of the non-negative anchored stacker for corrections D (n, k)."""
    y = y.astype(float)
    if D.shape[1] == 0:
        return z0.copy()

    def scaled(rows):
        sd = D[rows].std(axis=0)
        return np.where(sd > 1e-8, sd, 1.0)

    def crossfit(folds, lam):
        out = np.full(len(z0), np.nan)
        for g in folds:
            tr = np.isin(fold, [h for h in folds if h != g])
            sd = scaled(tr)
            th = fit_stack(D[tr] / sd, z0[tr], y[tr], lam)
            va = fold == g
            out[va] = z0[va] + th[0] + (D[va] / sd) @ th[1:]
        return out

    z_nested = np.empty(len(z0))
    for f in fold_names:
        others = [g for g in fold_names if g != f]
        rows = np.isin(fold, others)
        scores = [criterion_from_arrays(pid[rows], y[rows], crossfit(others, lam)[rows]) for lam in STACK_LAMS]
        lam = STACK_LAMS[int(np.argmax(scores))]
        sd = scaled(rows)
        th = fit_stack(D[rows] / sd, z0[rows], y[rows], lam)
        va = fold == f
        z_nested[va] = z0[va] + th[0] + (D[va] / sd) @ th[1:]
    return z_nested


# =====================================================================
# Test-set gate, LaTeX and figure helpers
# =====================================================================

def test_gate(pilot_exp_root: Path) -> dict:
    """The test set may be reported only after the preregistered unlock of the frozen model (script 19)."""
    stats = Path(pilot_exp_root) / "_stats"
    fz_path, log = stats / "frozen_selection.json", stats / "test_unlock_log.jsonl"
    out = {"frozen": None, "unlocked": False, "n_unlocks": 0, "hash_ok": False}
    if fz_path.exists():
        out["frozen"] = json.loads(fz_path.read_text())
        out["hash_ok"] = all(sha256_file(Path(pilot_exp_root) / k.split("/")[0] / f"{k.split('/')[1]}_predictions.csv") == v
                             for k, v in out["frozen"]["sha256"].items())
    if log.exists():
        out["n_unlocks"] = len([ln for ln in log.read_text().splitlines() if ln.strip()])
        out["unlocked"] = out["n_unlocks"] > 0
    return out


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.4f}"


def fmt_num(x, nd: int = 4) -> str:
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def write_table(df: pd.DataFrame, stem: Path, caption: str = "", nd: int = 4, p_cols: Sequence[str] = ()) -> None:
    """Writes <stem>.csv (full precision) and <stem>.tex (booktabs; needs \\usepackage{booktabs})."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}",
             rf"\label{{tab:{stem.name}}}", r"\begin{tabular}{" + "l" + "r" * (df.shape[1] - 1) + "}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if c in p_cols and not isinstance(v, str):
                cells.append(fmt_p(float(v)))
            elif isinstance(v, str):
                cells.append(v if "$" in v else esc(v))
            else:
                cells.append(fmt_num(v, nd))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    stem.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def new_fig(ncols: int = 1, nrows: int = 1, w: float = 3.6, h: float = 3.2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 9, "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    return fig, axes


def save_fig(fig, stem: Path) -> None:
    import matplotlib.pyplot as plt
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def enrichment_curve(df: pd.DataFrame, fracs: np.ndarray) -> np.ndarray:
    """(n_proteins, len(fracs)) recall of true residues within the top x% ranked residues."""
    y, z = df["y_true"].to_numpy(), df["z"].to_numpy(dtype=float)
    out = []
    for _, a, b in protein_slices(df["pdb_id"].to_numpy()):
        ys = y[a:b][np.argsort(-z[a:b], kind="stable")]
        cum = np.cumsum(ys)
        k = np.maximum(1, np.ceil(fracs * (b - a)).astype(int))
        out.append(cum[k - 1] / max(1, ys.sum()))
    return np.asarray(out)
