"""
Residue scores -> 3D pockets -> pocket-level retrieval metrics.

Why this exists
---------------
Everything else in this project ranks residues. A medicinal chemist does not act on a
residue ranking; they act on a pocket. Two methods with the same AP can differ sharply in
whether their top-ranked *pocket* is the real one, and the cryptic-site literature reports
pocket-level DCC/RRO precisely for that reason. This module converts any per-residue score
into ranked pockets and scores them, so that every method in the comparison goes through one
identical post-processing pipeline. Cross-paper pocket numbers are not comparable unless the
clustering and the success criteria match; running every method through this module is what
makes the comparison fair, and it is the reason we do not quote pocket numbers from other
papers alongside ours.

Definitions used here (stated explicitly because they vary between papers)
-------------------------------------------------------------------------
* Candidate residues: the top `top_frac` of a protein's residues by score, or all residues
  above `threshold`, whichever selector is requested.
* Pockets: single-linkage clusters of candidate residues whose CB atoms are within
  `link_dist` of each other, keeping clusters of at least `min_size` residues.
* Pocket score: mean of the `score_top_k` highest member scores (a soft maximum that does not
  reward a single lucky residue, and does not dilute a large pocket).
* Pocket centre: mean CB coordinate of the member residues.
* DCC (distance to the centre of the closest true pocket centre): a predicted pocket is a hit
  if its centre lies within `dcc_cut` of a true pocket centre. DCC@k = the fraction of true
  pockets hit by at least one of the k top-ranked predicted pockets.
* RRO (relative residue overlap): |predicted ∩ true| / |predicted ∪ true| for the best-matching
  predicted pocket; success at `rro_cut`. This is the Jaccard form; some papers use
  |∩| / |true| instead, so we report both (`rro` and `rro_recall`).
* N is the number of annotated cryptic pockets of that protein, so DCC@N and DCC@(N+2) follow
  the convention used for ligand-binding-site predictors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


@dataclass
class Pocket:
    residues: np.ndarray          # indices into the protein's residue arrays
    score: float
    center: np.ndarray            # (3,)
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.residues)


def predict_pockets(scores: np.ndarray, cb: np.ndarray, *, top_frac: float = 0.10,
                    threshold: float | None = None, link_dist: float = 8.0,
                    min_size: int = 3, score_top_k: int = 5, max_pockets: int = 20) -> List[Pocket]:
    """Cluster the highest-scoring residues into ranked pockets. Deterministic."""
    scores = np.asarray(scores, dtype=np.float64)
    cb = np.asarray(cb, dtype=np.float64)
    ok = np.isfinite(cb).all(axis=1) & np.isfinite(scores)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return []
    if threshold is not None:
        sel = idx[scores[idx] >= threshold]
    else:
        k = max(min_size, int(np.ceil(top_frac * len(idx))))
        sel = idx[np.argsort(-scores[idx], kind="stable")[:k]]
    if len(sel) < min_size:
        return []
    if len(sel) == 1:
        labels = np.array([1])
    else:
        Z = linkage(pdist(cb[sel]), method="single")
        labels = fcluster(Z, t=link_dist, criterion="distance")
    pockets = []
    for lab in np.unique(labels):
        members = sel[labels == lab]
        if len(members) < min_size:
            continue
        s = np.sort(scores[members])[::-1][:score_top_k]
        pockets.append(Pocket(residues=np.sort(members), score=float(s.mean()),
                              center=cb[members].mean(axis=0)))
    pockets.sort(key=lambda p: (-p.score, -p.size))
    return pockets[:max_pockets]


def true_pockets(residue_keys: Sequence[str], cb: np.ndarray,
                 selections: Sequence[Sequence[str]]) -> List[Pocket]:
    """One Pocket per annotated cryptic site; selections are lists of residue keys."""
    pos = {k: i for i, k in enumerate(residue_keys)}
    out = []
    for sel in selections:
        members = np.array(sorted({pos[k] for k in sel if k in pos}), dtype=int)
        if len(members) == 0:
            continue
        c = cb[members]
        c = c[np.isfinite(c).all(axis=1)]
        if len(c) == 0:
            continue
        out.append(Pocket(residues=members, score=1.0, center=c.mean(axis=0)))
    return out


def _overlap(pred: Pocket, true: Pocket):
    a, b = set(pred.residues.tolist()), set(true.residues.tolist())
    inter = len(a & b)
    return inter / max(1, len(a | b)), inter / max(1, len(b))


def evaluate_protein(pred: Sequence[Pocket], true: Sequence[Pocket], *, dcc_cut: float = 4.0,
                     rro_cut: float = 0.5, ks: Sequence[int] = (1, 3, 5)) -> Dict[str, float]:
    """Pocket-level metrics for one protein. NaN where undefined (no annotated pocket)."""
    out: Dict[str, float] = {"n_true_pockets": len(true), "n_pred_pockets": len(pred)}
    if not true:
        return out
    N = len(true)
    cutoffs = list(ks) + [N, N + 2]
    names = [f"dcc_top{k}" for k in ks] + ["dcc_topN", "dcc_topN2"]
    centers_t = np.stack([t.center for t in true])
    for name, k in zip(names, cutoffs):
        if not pred:
            out[name] = 0.0
            continue
        cen = np.stack([p.center for p in pred[:k]])
        d = np.linalg.norm(cen[:, None, :] - centers_t[None, :, :], axis=2)   # (k, N)
        out[name] = float((d.min(axis=0) <= dcc_cut).mean())
    best_j = best_r = 0.0
    rank_hit = np.nan
    for t in true:
        js = [_overlap(p, t) for p in pred] or [(0.0, 0.0)]
        j = max(x[0] for x in js)
        r = max(x[1] for x in js)
        best_j, best_r = max(best_j, j), max(best_r, r)
        hits = [i for i, (jj, _) in enumerate(js, 1) if jj >= rro_cut]
        if hits:
            rank_hit = hits[0] if not np.isfinite(rank_hit) else min(rank_hit, hits[0])
    out["rro"] = best_j
    out["rro_recall"] = best_r
    out["rro_success"] = float(best_j >= rro_cut)
    out["first_hit_rank"] = float(rank_hit) if np.isfinite(rank_hit) else np.nan
    out["site_found"] = float(out.get("dcc_topN2", 0.0) > 0)
    return out


POCKET_METRICS = ["dcc_top1", "dcc_top3", "dcc_top5", "dcc_topN", "dcc_topN2",
                  "rro", "rro_recall", "rro_success", "site_found"]
