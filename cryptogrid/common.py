from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yaml

AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U","PYL":"O",
}


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_pdb_id(x: str) -> str:
    return Path(str(x).strip()).stem.lower()


def normalize_residue_token(token: str) -> str:
    """Normalize CryptoBench/Grid-Crypto residue ids to CHAIN_RESNUM[ICODE]."""
    s = str(token).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.replace(":", "_")
    s = re.sub(r"\s+", "", s)
    # Do not replace '-' globally because negative residue ids are legal.
    m = re.match(r"^([A-Za-z0-9])[_-](-?\d+[A-Za-z]?)$", s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = re.match(r"^([A-Za-z0-9])(-?\d+[A-Za-z]?)$", s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    parts = [p for p in re.split(r"[_:]+", s) if p]
    if len(parts) >= 2 and re.match(r"^-?\d+[A-Za-z]?$", parts[1]):
        return f"{parts[0]}_{parts[1]}"
    return ""


def parse_residue_list(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    toks = re.split(r"[;,\s]+", str(value))
    out, seen = [], set()
    for t in toks:
        n = normalize_residue_token(t)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def residue_chain(residue_key: str) -> str:
    return str(residue_key).split("_", 1)[0] if "_" in str(residue_key) else "?"


def split_chain_spec(chain_spec: str) -> List[str]:
    s = str(chain_spec or "").strip()
    if not s:
        return []
    return [x for x in re.split(r"[-,;/\s]+", s) if x]


def dataset_entries(dataset: Mapping, pdb_id: str):
    pid = clean_pdb_id(pdb_id)
    e = dataset.get(pid)
    if e is None:
        e = dataset.get(pid.upper())
    if e is None:
        return []
    return [e] if isinstance(e, dict) else list(e)


def apo_chains(dataset: Mapping, pdb_id: str) -> List[str]:
    e = dataset_entries(dataset, pdb_id)
    if not e:
        return []
    chains = []
    for row in e:
        for ch in split_chain_spec(row.get("apo_chain", "")):
            if ch not in chains:
                chains.append(ch)
    return chains


def _parse_selection_item(item) -> str:
    if isinstance(item, dict):
        chain = item.get("chain") or item.get("chain_id") or item.get("auth_chain")
        resi = item.get("resi") or item.get("residue") or item.get("residue_id") or item.get("resnum")
        if chain is not None and resi is not None:
            return normalize_residue_token(f"{chain}_{resi}")
    return normalize_residue_token(item)


def cryptic_truth(dataset: Mapping, pdb_id: str, label_side: str = "apo") -> Set[str]:
    selection_key = f"{label_side}_pocket_selection"
    truth: Set[str] = set()
    for e in dataset_entries(dataset, pdb_id):
        vals = e.get(selection_key, []) or []
        if isinstance(vals, str):
            vals = re.split(r"[;,\s]+", vals)
        for r in vals:
            nr = _parse_selection_item(r)
            if nr:
                truth.add(nr)
    return truth


def parse_pdb_observed_residues(pdb_path: str | Path, allowed_chains: Sequence[str] | None = None) -> Tuple[List[str], List[str]]:
    """Unique ATOM residues in file order. Returns residue keys and AA1 sequence."""
    p = Path(pdb_path)
    keys, aas, seen = [], [], set()
    allow = set(allowed_chains or [])
    if not p.exists():
        return keys, aas
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21].strip() or "?"
            if allow and chain not in allow:
                continue
            resseq = line[22:26].strip()
            icode = line[26].strip()
            if not resseq:
                continue
            rk = normalize_residue_token(f"{chain}_{resseq}{icode}" if icode else f"{chain}_{resseq}")
            if not rk or rk in seen:
                continue
            seen.add(rk)
            keys.append(rk)
            aas.append(AA3_TO_1.get(line[17:20].strip().upper(), "X"))
    return keys, aas


def load_folds(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_ids(obj) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        out.append(clean_pdb_id(obj))
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out.extend(_flatten_ids(x))
    elif isinstance(obj, dict):
        for x in obj.values():
            out.extend(_flatten_ids(x))
    return out


def split_ids(folds: dict) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Handles train-0, train_0 and nested train dictionaries."""
    test = sorted(set(_flatten_ids(folds.get("test", []))))
    train_folds: Dict[str, List[str]] = {}
    for k, v in folds.items():
        kl = str(k).lower()
        if kl.startswith("train-") or kl.startswith("train_"):
            train_folds[str(k)] = sorted(set(_flatten_ids(v)))
    if not train_folds and "train" in folds:
        tr = folds["train"]
        if isinstance(tr, dict):
            for k, v in tr.items():
                train_folds[str(k)] = sorted(set(_flatten_ids(v)))
        else:
            train_folds["train"] = sorted(set(_flatten_ids(tr)))
    train = sorted(set(x for xs in train_folds.values() for x in xs))
    return train, test, train_folds


def resolve_fold(train_folds: Dict[str, List[str]], requested: str) -> List[str]:
    """Accept train_3 or train-3 regardless of source spelling."""
    if requested in train_folds:
        return train_folds[requested]
    alt = requested.replace("_", "-") if "_" in requested else requested.replace("-", "_")
    if alt in train_folds:
        return train_folds[alt]
    if train_folds:
        return train_folds[sorted(train_folds)[-1]]
    return []


def load_dataset_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def find_pdb_file(pdb_root: str | Path, pdb_id: str) -> Path | None:
    d = Path(pdb_root)
    pid = clean_pdb_id(pdb_id)
    for c in (d / f"{pid}.pdb", d / f"{pid.upper()}.pdb"):
        if c.exists():
            return c
    h = sorted(d.glob(f"{pid}*.pdb"))
    return h[0] if h else None


def find_feature_file(features_dir: str | Path, pdb_id: str) -> Path | None:
    d = Path(features_dir)
    pid = clean_pdb_id(pdb_id)
    for c in (d/f"{pid}_features.csv", d/f"{pid}.csv", d/f"{pid.upper()}_features.csv", d/f"{pid.upper()}.csv"):
        if c.exists():
            return c
    h = sorted(d.glob(f"{pid}*features*.csv"))
    return h[0] if h else None


def feature_family(name: str) -> str:
    c = name.lower()
    if c.startswith("summary::"):
        return "grid_summary"
    base = c.split("::", 1)[-1]
    if any(x in base for x in ("gaff", "uff", "sasa", "mda", "vormqa", "voromqa")):
        return "filtration_energy"
    if re.match(r"^[1-6]_face_", base) or "_face_" in base:
        return "face"
    if base in {"deg_seq", "n_atom"} or base.endswith("_count") or any(x in base for x in ("_n1", "_n2")):
        return "counts"
    if any(x in base for x in ("distance", "_mean", "_median", "_min", "_max", "_stdev", "_variance", "_nn_", "_ns_", "_no_", "_ss_", "_so_", "_oo_")):
        return "distance_geometry"
    return "other"


def safe_numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    x = df.reindex(columns=list(cols)).copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def best_threshold(y_true, y_score, metric: str = "mcc"):
    from sklearn.metrics import matthews_corrcoef, f1_score
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    if len(y) == 0:
        return 0.5, np.nan
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 99), np.quantile(s, np.linspace(0, 1, 101))])
    bt, bv = 0.5, -np.inf
    for t in candidates:
        yp = (s >= t).astype(int)
        try:
            if metric.lower() == "f1":
                v = f1_score(y, yp, zero_division=0)
            else:
                v = matthews_corrcoef(y, yp) if len(np.unique(yp)) > 1 else -1.0
        except Exception:
            continue
        if np.isfinite(v) and v > bv:
            bt, bv = float(t), float(v)
    return bt, bv


def per_protein_metrics(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score
    rows = []
    for pid, g in df.groupby("pdb_id", sort=False):
        y = g["y_true"].astype(int).to_numpy()
        s = g["y_score"].astype(float).to_numpy()
        ok = np.isfinite(s)
        y, s = y[ok], s[ok]
        if len(y) == 0:
            continue
        yp = (s >= threshold).astype(int)
        row = {"pdb_id": pid, "n": len(y), "positives": int(y.sum())}
        row["roc_auc"] = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
        row["average_precision"] = average_precision_score(y, s) if y.sum() > 0 else np.nan
        row["mcc"] = matthews_corrcoef(y, yp) if len(np.unique(yp)) > 1 else 0.0
        row["f1"] = f1_score(y, yp, zero_division=0)
        row["accuracy"] = accuracy_score(y, yp)
        row["precision"] = precision_score(y, yp, zero_division=0)
        row["recall"] = recall_score(y, yp, zero_division=0)
        for frac in (0.01, 0.05):
            k = max(1, int(math.ceil(len(y) * frac)))
            ii = np.argsort(-s)[:k]
            row[f"recall_at_{int(frac*100)}pct"] = float(y[ii].sum() / max(1, y.sum()))
            base = float(y.mean())
            pk = float(y[ii].mean()) if k else 0.0
            row[f"ef_at_{int(frac*100)}pct"] = pk / base if base > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def metric_summary(pred_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score
    pp = per_protein_metrics(pred_df, threshold)
    y = pred_df["y_true"].astype(int).to_numpy()
    s = pred_df["y_score"].astype(float).to_numpy()
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    yp = (s >= threshold).astype(int)
    def mmean(c):
        return float(pp[c].mean(skipna=True)) if len(pp) and c in pp else np.nan
    return {
        "n_proteins": int(pred_df["pdb_id"].nunique()),
        "n_residues": int(len(y)),
        "threshold": float(threshold),
        "macro_roc_auc": mmean("roc_auc"),
        "macro_average_precision": mmean("average_precision"),
        "macro_mcc": mmean("mcc"),
        "macro_f1": mmean("f1"),
        "macro_recall_at_1pct": mmean("recall_at_1pct"),
        "macro_ef_at_1pct": mmean("ef_at_1pct"),
        "pooled_roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan,
        "pooled_average_precision": float(average_precision_score(y, s)) if y.sum() > 0 else np.nan,
        "pooled_mcc": float(matthews_corrcoef(y, yp)) if len(np.unique(yp)) > 1 else 0.0,
        "pooled_f1": float(f1_score(y, yp, zero_division=0)),
        "pooled_accuracy": float(accuracy_score(y, yp)),
        "pooled_precision": float(precision_score(y, yp, zero_division=0)),
        "pooled_recall": float(recall_score(y, yp, zero_division=0)),
    }


def bootstrap_macro_ci(per_protein_df: pd.DataFrame, col: str, reps: int = 1000, seed: int = 42):
    vals = per_protein_df[col].dropna().to_numpy(float)
    if len(vals) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(reps)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
