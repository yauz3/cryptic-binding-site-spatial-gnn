"""
Neural building blocks shared by experiment scripts 11, 15, 16, 17.

* PlmHead          - the plm_only MLP of scripts/03 (exposes its hidden state)
* top_pair_loss    - pairwise logistic loss of positives vs the HARDEST negatives of the same protein
* soft_bedroc_loss - differentiable BEDROC(alpha) surrogate built on soft ranks
* build_graph      - rotation/translation-invariant kNN residue graph from CA/CB coordinates
* AnchoredGNN      - pLM logit + zero-initialised graph residual  (z = z_plm + z_res)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

RBF_CENTERS = np.linspace(2.0, 22.0, 16)
EDGE_DIM = len(RBF_CENTERS) + 6


class PlmHead(nn.Module):
    """Same network as ResidueMLP(variant='plm_only') in scripts/03_train_hybrid_models.py."""

    def __init__(self, d_esm: int, hidden: int, dropout: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_esm, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, b):
        h = self.trunk(b["xe"])
        return {"z": self.out(h).squeeze(-1), "h": h}


# ---------------------------------------------------------------------
# Ranking losses (one protein = one list)
# ---------------------------------------------------------------------

def top_pair_loss(z: torch.Tensor, y: torch.Tensor, top_frac: float = 0.05, tau: float = 1.0) -> torch.Tensor:
    """mean_{i in P} mean_{j in hardest negatives} tau * softplus((z_j - z_i) / tau).

    The hardest negatives are the top `top_frac` of the protein's residues by score among
    negatives, i.e. exactly the residues that compete for the EF@1% / EF@5% slots.
    """
    pos, neg = z[y > 0.5], z[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return z.sum() * 0.0
    k = min(neg.numel(), max(1, int(math.ceil(top_frac * z.numel()))))
    hard = torch.topk(neg, k).values
    return (tau * F.softplus((hard[None, :] - pos[:, None]) / tau)).mean()


def soft_bedroc_loss(z: torch.Tensor, y: torch.Tensor, alpha: float = 20.0, tau: float = 1.0) -> torch.Tensor:
    """1 - (soft RIE numerator / best achievable numerator); 0 when all positives are ranked first.

    soft rank_i = 1 + sum_{j != i} sigmoid((z_j - z_i) / tau)
    """
    pos = z[y > 0.5]
    n, npos = z.numel(), pos.numel()
    if npos == 0 or npos == n:
        return z.sum() * 0.0
    soft_rank = 0.5 + torch.sigmoid((z[None, :] - pos[:, None]) / tau).sum(dim=1)   # self term contributes 0.5
    gain = torch.exp(-alpha * soft_rank / n).sum()
    ideal = torch.exp(-alpha * torch.arange(1, npos + 1, device=z.device, dtype=z.dtype) / n).sum()
    return 1.0 - gain / ideal


# ---------------------------------------------------------------------
# Residue graph (label-free, SE(3)-invariant edge features)
# ---------------------------------------------------------------------

def build_graph(geom: dict, k: int = 16) -> dict:
    """kNN graph on CB atoms. Residues without coordinates get no edges (mask = False)."""
    ca = geom["ca"].astype(np.float64)
    cb = geom["cb"].astype(np.float64)
    chain, seq = geom["chain_index"], geom["seq_index"]
    n = len(ca)
    K = k + 1                                    # one slot may be the residue itself and is masked
    idx = np.tile(np.arange(n)[:, None], (1, K))
    mask = np.zeros((n, K), dtype=bool)
    dist = np.zeros((n, K))
    valid = np.isfinite(ca).all(axis=1) & np.isfinite(cb).all(axis=1)
    vi = np.flatnonzero(valid)
    if len(vi) >= 2:
        kk = min(K, len(vi))
        d, j = cKDTree(cb[vi]).query(cb[vi], k=kk)
        idx[vi, :kk] = vi[j]
        dist[vi, :kk] = d
        mask[vi, :kk] = vi[j] != vi[:, None]
    u = cb - ca
    un = np.linalg.norm(u, axis=1, keepdims=True)
    u = np.where(np.isfinite(un) & (un > 1e-6), u / np.maximum(un, 1e-6), 0.0)
    u[~valid] = 0.0
    cbz = np.where(valid[:, None], cb, 0.0)
    r = cbz[idx] - cbz[:, None, :]
    rn = r / np.maximum(dist, 1e-6)[..., None]
    rbf = np.exp(-((dist[..., None] - RBF_CENTERS) / 1.5) ** 2)
    ui_uj = np.einsum("nd,nkd->nk", u, u[idx])
    ui_r = np.einsum("nd,nkd->nk", u, rn)
    uj_r = -np.einsum("nkd,nkd->nk", u[idx], rn)
    same = chain[idx] == chain[:, None]
    sep = np.abs(seq[idx] - seq[:, None])
    sepf = np.where(same, np.log1p(sep) / math.log(1000.0), 1.0)
    adj = (same & (sep == 1)).astype(np.float64)
    edge = np.concatenate([rbf, np.stack([ui_uj, ui_r, uj_r, sepf, adj, same.astype(np.float64)], axis=-1)], axis=-1)
    edge = edge * mask[..., None]
    return {"idx": idx.astype(np.int64), "mask": mask, "edge": edge.astype(np.float32)}


class GraphLayer(nn.Module):
    def __init__(self, d: int, d_edge: int, dropout: float):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * d + d_edge, d), nn.ReLU(), nn.Linear(d, d))
        self.att = nn.Linear(2 * d + d_edge, 1)
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, g, idx, edge, mask):
        gj = g[idx]                                            # (n, K, d)
        x = torch.cat([g.unsqueeze(1).expand_as(gj), gj, edge], dim=-1)
        a = self.att(x).squeeze(-1).masked_fill(~mask, -1e9)
        a = torch.softmax(a, dim=1) * mask                     # rows without neighbours -> all zeros
        m = (a.unsqueeze(-1) * self.msg(x)).sum(dim=1)
        return self.norm(g + self.upd(torch.cat([g, m], dim=-1)))


class AnchoredGNN(nn.Module):
    """z = z_plm + z_res.

    z_plm : PlmHead on the ESM-2 embedding (identical to the pLM-only baseline).
    z_res : graph branch over [pLM hidden state, detached pLM logit, geometry features];
            its last layer is ZERO-initialised, so training starts at the pLM-only solution
            and the structural branch can only add a learned correction.
    """

    def __init__(self, d_esm: int, d_geom: int, hidden_plm: int, dropout: float,
                 hidden: int = 128, layers: int = 2, use_geom: bool = True):
        super().__init__()
        self.plm = PlmHead(d_esm, hidden_plm, dropout)
        self.use_geom = use_geom
        d_in = hidden_plm + 1 + (d_geom if use_geom else 0)
        self.node_in = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.layers = nn.ModuleList([GraphLayer(hidden, EDGE_DIM, dropout) for _ in range(layers)])
        self.res = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.res[-1].weight)
        nn.init.zeros_(self.res[-1].bias)

    def structural_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("plm.")]

    def forward(self, b):
        p = self.plm(b)
        feats = [p["h"], p["z"].detach().unsqueeze(-1)]
        if self.use_geom:
            feats.append(b["xg"])
        g = self.node_in(torch.cat(feats, dim=-1))
        for layer in self.layers:
            g = layer(g, b["idx"], b["edge"], b["mask"])
        z_res = self.res(g).squeeze(-1)
        return {"z": p["z"] + z_res, "z_plm": p["z"], "z_res": z_res}
