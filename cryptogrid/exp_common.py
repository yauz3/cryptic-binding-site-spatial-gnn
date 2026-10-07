"""
Shared, leakage-aware infrastructure for experiment scripts 10-19.

Design rules enforced here
--------------------------
* Every experiment writes residue-level LOGITS in one canonical row order:
      results/exp/<name>/oof_predictions.csv    (885 train proteins, official folds)
      results/exp/<name>/test_predictions.csv   (222 test proteins, written but NOT scored)
* Hyper-parameters are chosen by NESTED selection on train OOF only
  (choose on 3 official folds, apply to the 4th), so OOF numbers are not
  optimistically biased by the search.
* The selection criterion is prespecified:  J = 0.5*macro AP + 0.5*macro BEDROC20.
* Test metrics are computed in exactly one place: scripts/19_stats_and_freeze.py
  with --unlock-test, after a model has been frozen from OOF.
* No protein is ever dropped (zero-Grid-support proteins stay in).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, wilcoxon

from .common import ensure_dir, load_config, load_folds, split_ids

KEYS = ["pdb_id", "residue_key"]
METRICS = ["ap", "auc", "ef1", "ef5", "bedroc20", "bedroc50", "bedroc80"]
PRIMARY = ["ef1", "bedroc20"]                      # prespecified confirmatory family
SELECTION_WEIGHTS = {"ap": 0.5, "bedroc20": 0.5}   # prespecified OOF selection criterion J
BASELINE = "base_plm"


# =====================================================================
# Context
# =====================================================================

@dataclass
class Ctx:
    cfg: dict
    struct_dir: Path
    esm_dir: Path
    geom_dir: Path
    pocket_dir: Path
    exp_root: Path
    train_ids: List[str]
    test_ids: List[str]
    fold_names: List[str]
    fold_of: Dict[str, str]
    _frames: Dict[str, pd.DataFrame] = field(default_factory=dict)


def make_ctx(config: str | Path = "config.yaml", require_esm: bool = True) -> Ctx:
    """require_esm=False is for scripts that CREATE the embedding cache (e.g. scripts/23)."""
    cfg = load_config(config)
    struct_dir = Path(cfg["work"]["struct_cache"])
    esm_dir = Path(cfg["work"]["esm_cache"])
    work_root = Path(cfg["work"].get("root", struct_dir.parent))
    exp_root = ensure_dir(Path(cfg["work"]["results"]) / "exp")

    train, test, tf = split_ids(load_folds(cfg["project"]["folds_json"]))
    avail = {p.stem for p in struct_dir.glob("*.npz")}
    if require_esm:
        avail &= {p.stem for p in esm_dir.glob("*.npz")}
    missing = sorted((set(train) | set(test)) - avail)
    if missing:
        # Never silently shrink the benchmark.
        what = "struct/ESM" if require_esm else "struct"
        raise RuntimeError(
            f"{len(missing)} proteins have no {what} cache (e.g. {missing[:5]}). "
            "Fix the caches instead of dropping proteins."
        )
    fold_names = sorted(k for k, v in tf.items() if v)
    fold_of: Dict[str, str] = {}
    for f in fold_names:
        for pid in tf[f]:
            if pid in fold_of:
                raise RuntimeError(f"{pid} occurs in two official folds")
            fold_of[pid] = f
    if set(fold_of) != set(train):
        raise RuntimeError("official train folds do not partition the training set")
    if set(train) & set(test):
        raise RuntimeError("train/test overlap in folds.json")
    return Ctx(cfg, struct_dir, esm_dir, work_root / "cache_geom", work_root / "cache_pockets",
               exp_root, list(train), list(test), fold_names, fold_of)


def exp_dir(ctx: Ctx, name: str) -> Path:
    return ensure_dir(ctx.exp_root / name)


def write_json(path: Path, obj) -> None:
    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================
# Cached per-protein data
# =====================================================================

class Store:
    """In-memory access to struct / ESM / geometry caches (read-only)."""

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self._lab: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._esm: Dict[str, np.ndarray] = {}
        self._geom: Dict[str, dict] = {}

    def labels(self, pid: str) -> Tuple[np.ndarray, np.ndarray]:
        if pid not in self._lab:
            z = np.load(self.ctx.struct_dir / f"{pid}.npz", allow_pickle=False)
            self._lab[pid] = (z["y"].astype(np.int8), z["residue_keys"].astype(str))
        return self._lab[pid]

    def struct(self, pid: str) -> np.ndarray:
        z = np.load(self.ctx.struct_dir / f"{pid}.npz", allow_pickle=False)
        return z["X_struct"].astype(np.float32)

    def esm(self, pid: str) -> np.ndarray:
        if pid not in self._esm:
            z = np.load(self.ctx.esm_dir / f"{pid}.npz", allow_pickle=False)
            if not np.array_equal(z["residue_keys"].astype(str), self.labels(pid)[1]):
                raise RuntimeError(f"{pid}: ESM/struct residue order mismatch")
            self._esm[pid] = z["X_esm"]          # keep float16 in RAM
        return self._esm[pid]

    def geom(self, pid: str) -> dict:
        if pid not in self._geom:
            p = self.ctx.geom_dir / f"{pid}.npz"
            if not p.exists():
                raise FileNotFoundError(f"{p} missing. Run scripts/10_build_geom_cache.py first.")
            z = np.load(p, allow_pickle=False)
            if not np.array_equal(z["residue_keys"].astype(str), self.labels(pid)[1]):
                raise RuntimeError(f"{pid}: geometry/struct residue order mismatch")
            self._geom[pid] = {k: z[k] for k in ("X_geom", "ca", "cb", "chain_index", "seq_index")}
        return self._geom[pid]

    def geom_feature_names(self) -> List[str]:
        return json.loads((self.ctx.geom_dir / "feature_names.json").read_text())

    def struct_feature_names(self) -> List[str]:
        return json.loads((self.ctx.struct_dir / "feature_names.json").read_text())


class RobustScaler:
    """Median/IQR scaling fitted on TRAINING-FOLD proteins only; clips and zero-fills NaN."""

    def __init__(self, clip: float = 5.0):
        self.clip = clip
        self.med = None
        self.scale = None

    def fit(self, arrays: Sequence[np.ndarray]) -> "RobustScaler":
        X = np.concatenate(arrays, axis=0).astype(np.float64)
        with np.errstate(all="ignore"):
            self.med = np.nanmedian(X, axis=0)
            q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
            sd = np.nanstd(X, axis=0)
        scale = (q75 - q25) / 1.349
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, sd)
        self.scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X.astype(np.float64) - self.med) / self.scale
        Z = np.clip(Z, -self.clip, self.clip)
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)


# =====================================================================
# Canonical prediction frames
# =====================================================================

def canonical_frame(ctx: Ctx, split: str, store: Store | None = None) -> pd.DataFrame:
    """Rows = residues in cache order; proteins in folds.json order. split in {'oof','test'}."""
    if split not in ctx._frames:
        store = store or Store(ctx)
        ids = ctx.train_ids if split == "oof" else ctx.test_ids
        pid_col, key_col, y_col, fold_col = [], [], [], []
        for pid in ids:
            y, keys = store.labels(pid)
            pid_col.append(np.repeat(pid, len(y)))
            key_col.append(keys)
            y_col.append(y.astype(int))
            fold_col.append(np.repeat(ctx.fold_of.get(pid, "test"), len(y)))
        ctx._frames[split] = pd.DataFrame({
            "pdb_id": np.concatenate(pid_col), "residue_key": np.concatenate(key_col),
            "y_true": np.concatenate(y_col), "fold": np.concatenate(fold_col),
        })
    return ctx._frames[split].copy()


def frame_from_dict(ctx: Ctx, split: str, z_by_pid: Dict[str, np.ndarray], col: str = "z") -> pd.DataFrame:
    df = canonical_frame(ctx, split)
    ids = ctx.train_ids if split == "oof" else ctx.test_ids
    df[col] = np.concatenate([np.asarray(z_by_pid[pid], dtype=np.float64) for pid in ids])
    return df


def save_preds(ctx: Ctx, name: str, split: str, df: pd.DataFrame) -> Path:
    ref = canonical_frame(ctx, split)
    if len(df) != len(ref) or not (df["pdb_id"].to_numpy() == ref["pdb_id"].to_numpy()).all() \
            or not (df["residue_key"].to_numpy() == ref["residue_key"].to_numpy()).all():
        raise RuntimeError(f"{name}/{split}: predictions are not in canonical order")
    if not np.isfinite(df["z"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{name}/{split}: non-finite logits")
    path = exp_dir(ctx, name) / f"{split}_predictions.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    return path


def load_preds(ctx: Ctx, name: str, split: str) -> pd.DataFrame:
    path = ctx.exp_root / name / f"{split}_predictions.csv"
    if not path.exists():
        hint = " Run scripts/11_baseline_multiseed.py first." if name == BASELINE else ""
        raise FileNotFoundError(f"{path} not found.{hint}")
    df = pd.read_csv(path, dtype={"pdb_id": str, "residue_key": str, "fold": str})
    ref = canonical_frame(ctx, split)
    if len(df) != len(ref) or not (df["pdb_id"].to_numpy() == ref["pdb_id"].to_numpy()).all() \
            or not (df["residue_key"].to_numpy() == ref["residue_key"].to_numpy()).all():
        raise RuntimeError(f"{path}: row order differs from the canonical frame (stale cache?)")
    if not (df["y_true"].to_numpy() == ref["y_true"].to_numpy()).all():
        raise RuntimeError(f"{path}: labels differ from the struct cache")
    return df


def protein_slices(pdb_ids: np.ndarray) -> List[Tuple[str, int, int]]:
    pdb_ids = np.asarray(pdb_ids)
    change = np.flatnonzero(pdb_ids[1:] != pdb_ids[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(pdb_ids)]
    names = pdb_ids[starts]
    if len(set(names.tolist())) != len(names):
        raise RuntimeError("proteins are not stored as contiguous blocks")
    return [(str(n), int(a), int(b)) for n, a, b in zip(names, starts, ends)]


# =====================================================================
# Ranking metrics (per protein, ranking unit = residue)
# =====================================================================

def bedroc_from_ranks(pos_ranks: np.ndarray, n: int, alpha: float) -> float:
    """Truchon & Bayly BEDROC; identical to RDKit CalcBEDROC for a given ordering.

    NOTE: BEDROC is only interpretable when alpha * Ra << 1 (Ra = active fraction).
    With Ra ~ 0.054, alpha=20 gives ~1.1 (borderline) and alpha=50/80 are outside
    that regime, so BEDROC50/80 are reported as secondary only.
    """
    n_act = len(pos_ranks)
    if n_act == 0 or n_act == n:
        return float("nan")
    ra = n_act / n
    s = float(np.exp(-alpha * pos_ranks / n).sum())
    denom = (1.0 / n) * ((1.0 - math.exp(-alpha)) / (math.exp(alpha / n) - 1.0))
    rie = s / (n_act * denom)
    rie_max = (1.0 - math.exp(-alpha * ra)) / (ra * (1.0 - math.exp(-alpha)))
    rie_min = (1.0 - math.exp(alpha * ra)) / (ra * (1.0 - math.exp(alpha)))
    if rie_max == rie_min:
        return 1.0
    return float((rie - rie_min) / (rie_max - rie_min))


def rank_metrics(y: np.ndarray, s: np.ndarray, which: Sequence[str] = METRICS) -> Dict[str, float]:
    """AP, ROC-AUC, EF@1%, EF@5%, BEDROC(20/50/80) for one protein.

    EF@x% uses k = ceil(x * n) residues, precision@k / prevalence (same as cryptogrid.common).
    Ties are broken by residue order (stable sort), deterministically.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    n, npos = len(y), int(y.sum())
    out = {m: float("nan") for m in which}
    if n == 0 or npos == 0 or npos == n:
        return out
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    cum = np.cumsum(ys)
    ranks = np.flatnonzero(ys) + 1
    if "ap" in out:
        out["ap"] = float((cum[ranks - 1] / ranks).mean())
    if "auc" in out:
        r = rankdata(s)
        out["auc"] = float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * (n - npos)))
    for frac, key in ((0.01, "ef1"), (0.05, "ef5")):
        if key in out:
            k = max(1, int(math.ceil(n * frac)))
            out[key] = float((cum[k - 1] / k) / (npos / n))
    for a in (20, 50, 80):
        if f"bedroc{a}" in out:
            out[f"bedroc{a}"] = bedroc_from_ranks(ranks, n, float(a))
    return out


def per_protein_table(df: pd.DataFrame, col: str = "z", which: Sequence[str] = METRICS) -> pd.DataFrame:
    y = df["y_true"].to_numpy()
    z = df[col].to_numpy(dtype=np.float64)
    rows = []
    for pid, a, b in protein_slices(df["pdb_id"].to_numpy()):
        m = rank_metrics(y[a:b], z[a:b], which)
        m.update(pdb_id=pid, n=b - a, positives=int(y[a:b].sum()))
        rows.append(m)
    return pd.DataFrame(rows).set_index("pdb_id")


def j_vector(pp: pd.DataFrame) -> np.ndarray:
    return sum(w * pp[m].to_numpy(dtype=float) for m, w in SELECTION_WEIGHTS.items())


def criterion(pp: pd.DataFrame) -> float:
    return float(np.nanmean(j_vector(pp)))


def criterion_from_arrays(pdb_ids: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """J for an arbitrary contiguous-by-protein subset of rows."""
    vals = []
    which = list(SELECTION_WEIGHTS)
    for _, a, b in protein_slices(pdb_ids):
        m = rank_metrics(y[a:b], z[a:b], which)
        vals.append(sum(w * m[k] for k, w in SELECTION_WEIGHTS.items()))
    return float(np.nanmean(vals))


def list_experiments(ctx: "Ctx", need_test: bool = False) -> List[str]:
    out = []
    for d in sorted(ctx.exp_root.iterdir()):
        if d.is_dir() and (d / "oof_predictions.csv").exists() and (not need_test or (d / "test_predictions.csv").exists()):
            out.append(d.name)
    return out


# =====================================================================
# Nested (cross-fitted) configuration selection on OOF
# =====================================================================

def nested_select(base: pd.DataFrame, cands: Dict[str, np.ndarray], fold_names: Sequence[str]) -> dict:
    """Pick a configuration for each official fold using the OTHER folds only.

    `cands` must be ordered from most conservative to least conservative: ties
    go to the earlier (more conservative) configuration.
    Returns nested OOF logits, per-fold choices, the all-OOF choice (to be
    applied to test) and the search table.
    """
    y = base["y_true"].to_numpy()
    sl = protein_slices(base["pdb_id"].to_numpy())
    pfold = np.array([base["fold"].iat[a] for _, a, _ in sl])
    names = list(cands)
    which = list(SELECTION_WEIGHTS)
    J = np.full((len(names), len(sl)), np.nan)
    for ci, name in enumerate(names):
        z = np.asarray(cands[name], dtype=np.float64)
        for pi, (_, a, b) in enumerate(sl):
            m = rank_metrics(y[a:b], z[a:b], which)
            J[ci, pi] = sum(w * m[k] for k, w in SELECTION_WEIGHTS.items())
    overall_scores = np.nanmean(J, axis=1)
    overall = names[int(np.argmax(overall_scores))]
    z_nested = np.empty(len(base), dtype=np.float64)
    chosen = {}
    table = pd.DataFrame({"config": names, "J_all_oof": overall_scores})
    for f in fold_names:
        other = pfold != f
        sc = np.nanmean(J[:, other], axis=1)
        best = names[int(np.argmax(sc))]
        chosen[f] = best
        rows = (base["fold"] == f).to_numpy()
        z_nested[rows] = np.asarray(cands[best], dtype=np.float64)[rows]
        table[f"J_{f}"] = np.nanmean(J[:, pfold == f], axis=1)
    return {"z_nested": z_nested, "chosen_per_fold": chosen, "overall": overall, "table": table}


def run_grid_experiment(ctx: Ctx, name: str, base_oof: pd.DataFrame, cands_oof: Dict[str, np.ndarray],
                        test_fn: Callable[[str], np.ndarray], meta: dict | None = None) -> dict:
    """Second-stage method with a small config grid: nested OOF + frozen config applied to test."""
    sel = nested_select(base_oof, cands_oof, ctx.fold_names)
    oof = base_oof[KEYS + ["y_true", "fold"]].copy()
    oof["z"] = sel["z_nested"]
    save_preds(ctx, name, "oof", oof)
    test = canonical_frame(ctx, "test")
    test["z"] = np.asarray(test_fn(sel["overall"]), dtype=np.float64)
    save_preds(ctx, name, "test", test)       # written, never scored here
    d = exp_dir(ctx, name)
    sel["table"].to_csv(d / "config_search_oof.csv", index=False)
    write_json(d / "selection.json", {
        "experiment": name, "criterion": SELECTION_WEIGHTS, "selected_on": "train OOF only (nested)",
        "config_applied_to_test": sel["overall"], "config_per_fold": sel["chosen_per_fold"],
        "n_configs": len(cands_oof), "meta": meta or {}, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"[{name}] config for test = {sel['overall']} | per fold = {sel['chosen_per_fold']}")
    oof_report(ctx, name, base_oof=base_oof, new_oof=oof)
    return sel


# =====================================================================
# Paired statistics (protein = unit of analysis)
# =====================================================================

def holm(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def signflip_pvalue(d: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on the MEAN difference."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_perm:
        m = min(2000, n_perm - done)
        signs = rng.integers(0, 2, size=(m, len(d)), dtype=np.int8) * 2 - 1
        hits += int((np.abs(signs @ d) / len(d) >= obs - 1e-15).sum())
        done += m
    return (hits + 1.0) / (n_perm + 1.0)


def paired_stats(pp_base: pd.DataFrame, pp_new: pd.DataFrame, metrics: Sequence[str] = METRICS,
                 n_perm: int = 20000, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Per-metric paired comparison new - base over proteins.

    p_signflip : permutation test on the mean paired difference (primary test; this is
                 the hypothesis behind 'macro EF@1 / BEDROC20 improved').
    p_wilcoxon : signed-rank test (sensitivity analysis; zeros dropped, so it is weak
                 for EF@1 where most proteins have identical top-k hits).
    Holm adjustment is applied inside the prespecified PRIMARY family.
    """
    if not pp_base.index.equals(pp_new.index):
        pp_new = pp_new.reindex(pp_base.index)
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics:
        a = pp_base[m].to_numpy(dtype=float)
        b = pp_new[m].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        d = b[ok] - a[ok]
        n = len(d)
        row = {"metric": m, "n_proteins": n, "mean_base": float(a[ok].mean()) if n else np.nan,
               "mean_new": float(b[ok].mean()) if n else np.nan, "mean_delta": float(d.mean()) if n else np.nan,
               "median_delta": float(np.median(d)) if n else np.nan,
               "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "n_equal": int((d == 0).sum())}
        if n and np.any(d != 0):
            boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
            row["ci95_low"], row["ci95_high"] = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
            row["p_signflip"] = signflip_pvalue(d, n_perm=n_perm, seed=seed + 1)
            try:
                row["p_wilcoxon"] = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                row["p_wilcoxon"] = 1.0
        else:
            row.update(ci95_low=0.0, ci95_high=0.0, p_signflip=1.0, p_wilcoxon=1.0)
        rows.append(row)
    out = pd.DataFrame(rows)
    prim = out["metric"].isin(PRIMARY).to_numpy()
    for pcol in ("p_signflip", "p_wilcoxon"):
        adj = np.full(len(out), np.nan)
        adj[prim] = holm(out.loc[prim, pcol].to_numpy())
        out[pcol + "_holm_primary"] = adj
    return out


def oof_report(ctx: Ctx, name: str, base_oof: pd.DataFrame | None = None,
               new_oof: pd.DataFrame | None = None, n_perm: int = 5000) -> pd.DataFrame:
    """Quick train-OOF comparison against the pLM anchor (no test data involved)."""
    base_oof = base_oof if base_oof is not None else load_preds(ctx, BASELINE, "oof")
    new_oof = new_oof if new_oof is not None else load_preds(ctx, name, "oof")
    ppb, ppn = per_protein_table(base_oof), per_protein_table(new_oof)
    st = paired_stats(ppb, ppn, n_perm=n_perm, n_boot=2000)
    print(f"\n[{name}] TRAIN-OOF vs {BASELINE} ({len(ppb)} proteins) | J: {criterion(ppb):.5f} -> {criterion(ppn):.5f}")
    print(st[["metric", "mean_base", "mean_new", "mean_delta", "ci95_low", "ci95_high",
              "n_better", "n_worse", "p_signflip", "p_wilcoxon"]].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    st.to_csv(exp_dir(ctx, name) / "oof_quick_stats.csv", index=False)
    return st


# =====================================================================
# Geometry helpers (label-free)
# =====================================================================

def neighbor_pairs(xyz: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All ordered pairs (i, j, d), i != j, d <= radius, among residues with finite coordinates."""
    valid = np.flatnonzero(np.isfinite(xyz).all(axis=1))
    if len(valid) < 2:
        e = np.zeros(0, dtype=int)
        return e, e, np.zeros(0)
    pts = xyz[valid].astype(np.float64)
    pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        e = np.zeros(0, dtype=int)
        return e, e, np.zeros(0)
    d = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
    i = np.concatenate([valid[pairs[:, 0]], valid[pairs[:, 1]]])
    j = np.concatenate([valid[pairs[:, 1]], valid[pairs[:, 0]]])
    return i, j, np.concatenate([d, d])


def gaussian_smooth(z: np.ndarray, pairs: Tuple[np.ndarray, np.ndarray, np.ndarray], sigma: float) -> np.ndarray:
    """Kernel-weighted mean of NEIGHBOUR logits (self excluded); residues without neighbours keep z."""
    i, j, d = pairs
    keep = d <= 3.0 * sigma
    i, j, d = i[keep], j[keep], d[keep]
    w = np.exp(-(d * d) / (2.0 * sigma * sigma))
    n = len(z)
    num = np.bincount(i, weights=w * z[j], minlength=n)
    den = np.bincount(i, weights=w, minlength=n)
    out = np.array(z, dtype=np.float64, copy=True)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def neighbor_max(z: np.ndarray, pairs: Tuple[np.ndarray, np.ndarray, np.ndarray], radius: float) -> np.ndarray:
    i, j, d = pairs
    keep = d <= radius
    out = np.full(len(z), -np.inf)
    np.maximum.at(out, i[keep], z[j[keep]])
    return np.where(np.isfinite(out), out, z)


# =====================================================================
# Multi-seed neural protocol (OOF on official folds + full-train -> test)
# =====================================================================

def get_device(cfg: dict):
    import torch
    if torch.cuda.is_available() and cfg["training"].get("device", "auto") != "cpu":
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pos_weight_of(store: Store, ids: Sequence[str]) -> float:
    pos = sum(int(store.labels(p)[0].sum()) for p in ids)
    tot = sum(len(store.labels(p)[0]) for p in ids)
    return float((tot - pos) / max(1, pos))


def train_one(model, ids, batch_fn, loss_fn, *, epochs, lr, weight_decay, seed, param_groups=None, tag=""):
    import torch
    params = param_groups(model, lr) if param_groups else model.parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    for ep in range(1, epochs + 1):
        order = list(ids)
        rng.shuffle(order)
        model.train()
        tot = 0.0
        for pid in order:
            b = batch_fn(pid)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(b), b)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{tag}: non-finite loss on {pid}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
        print(f"  [{tag}] epoch {ep}/{epochs} loss={tot / max(1, len(order)):.5f}", flush=True)
    return model


def predict_one(model, ids, batch_fn) -> Dict[str, np.ndarray]:
    import torch
    model.eval()
    out = {}
    with torch.no_grad():          # not inference_mode: cached graph tensors are reused for training
        for pid in ids:
            out[pid] = model(batch_fn(pid))["z"].float().cpu().numpy().astype(np.float64)
    return out


def run_nn_protocol(ctx: Ctx, name: str, prepare: Callable, build_model: Callable, *, seeds: Sequence[int],
                    epochs: int, lr: float, weight_decay: float, meta: dict | None = None,
                    param_groups: Callable | None = None, resume: bool = True) -> None:
    """Train `len(seeds)` x (4 official folds + 1 full-train) models and save seed-AVERAGED logits.

    prepare(fit_ids) -> (batch_fn, loss_fn)   # all preprocessing fitted on fit_ids only
    build_model()    -> torch.nn.Module       # called after set_seed
    Epoch count is fixed a priori (no early stopping on any held-out data).
    The seed ensemble is the reported model; individual seeds are kept as z_s<seed>
    columns for robustness reporting, never for picking a best seed.
    """
    device = get_device(ctx.cfg)
    d = exp_dir(ctx, name)
    parts = ensure_dir(d / "parts")
    oof_seed: Dict[int, Dict[str, np.ndarray]] = {}
    test_seed: Dict[int, Dict[str, np.ndarray]] = {}
    t0 = time.time()

    # Cached fold models may only be reused if the experiment definition is unchanged.
    sig = json.dumps({"epochs": epochs, "lr": lr, "weight_decay": weight_decay,
                      "meta": {k: v for k, v in (meta or {}).items() if k not in ("seeds", "no_resume", "name")}},
                     sort_keys=True, default=str)
    sig_path = parts / "signature.json"
    if resume and sig_path.exists() and sig_path.read_text() != sig:
        raise RuntimeError(f"{name}: cached fold predictions were made with different settings. "
                           "Use a new --name or pass --no-resume to retrain.")
    if not resume:
        for old in parts.glob("*.npz"):
            old.unlink()
    sig_path.write_text(sig)

    def _run(tag: str, fit_ids, pred_ids, seed: int) -> Dict[str, np.ndarray]:
        cache = parts / f"{tag}.npz"
        if resume and cache.exists():
            z = np.load(cache, allow_pickle=False)
            if set(z.files) == set(pred_ids):
                print(f"[{name}] {tag}: cached")
                return {p: z[p] for p in pred_ids}
        set_seed(seed)
        batch_fn, loss_fn = prepare(fit_ids)
        model = build_model().to(device)
        train_one(model, fit_ids, batch_fn, loss_fn, epochs=epochs, lr=lr, weight_decay=weight_decay,
                  seed=seed, param_groups=param_groups, tag=f"{name}/{tag}")
        pred = predict_one(model, pred_ids, batch_fn)
        np.savez_compressed(cache, **pred)
        return pred

    for seed in seeds:
        oof_seed[seed] = {}
        for fi, f in enumerate(ctx.fold_names):
            val_ids = [p for p in ctx.train_ids if ctx.fold_of[p] == f]
            fit_ids = [p for p in ctx.train_ids if ctx.fold_of[p] != f]
            oof_seed[seed].update(_run(f"seed{seed}_{f}", fit_ids, val_ids, seed * 1000 + fi))
        test_seed[seed] = _run(f"seed{seed}_full", ctx.train_ids, ctx.test_ids, seed * 1000 + 99)

    for split, per_seed, ids in (("oof", oof_seed, ctx.train_ids), ("test", test_seed, ctx.test_ids)):
        df = canonical_frame(ctx, split)
        cols = []
        for seed in seeds:
            c = f"z_s{seed}"
            df[c] = np.concatenate([per_seed[seed][p] for p in ids])
            cols.append(c)
        df["z"] = df[cols].mean(axis=1)          # logit-space seed ensemble
        save_preds(ctx, name, split, df)

    write_json(d / "meta.json", {"experiment": name, "seeds": list(seeds), "epochs": epochs, "lr": lr,
                                 "weight_decay": weight_decay, "folds": ctx.fold_names,
                                 "minutes": round((time.time() - t0) / 60, 2), "meta": meta or {}})
    if name != BASELINE:
        try:
            oof_report(ctx, name)
        except FileNotFoundError as e:
            print(f"[{name}] (no OOF report: {e})")
    else:
        pp = per_protein_table(load_preds(ctx, name, "oof"))
        print(f"\n[{name}] TRAIN-OOF seed-ensemble: " + "  ".join(f"{m}={pp[m].mean():.4f}" for m in METRICS))
