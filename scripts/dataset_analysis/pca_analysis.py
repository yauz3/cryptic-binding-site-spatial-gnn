#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from analysis_common import load_training_matrix, scaled

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args()
X, y, pids, feat, cfg = load_training_matrix(args.config, max_rows=40000, max_features=250)
Z = scaled(X)
out = Path(cfg["work"]["results"]) / "dataset_analysis" / "pca"; out.mkdir(parents=True, exist_ok=True)
p = PCA(n_components=min(30, Z.shape[1], Z.shape[0])).fit(Z)
pc = p.transform(Z)
fig, ax = plt.subplots(figsize=(9, 6)); sc = ax.scatter(pc[:, 0], pc[:, 1], c=y, s=7, alpha=.35)
ax.set_xlabel(f"PC1 ({100*p.explained_variance_ratio_[0]:.2f}%)"); ax.set_ylabel(f"PC2 ({100*p.explained_variance_ratio_[1]:.2f}%)")
ax.set_title("Grid-Crypto residue feature space — training proteins only"); fig.colorbar(sc, ax=ax, label="cryptic residue")
fig.tight_layout(); fig.savefig(out / "pca_scatter.png", dpi=300); plt.close(fig)
cum = np.cumsum(p.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(np.arange(1, len(cum)+1), cum, marker="o"); ax.axhline(.8, ls="--"); ax.axhline(.95, ls=":")
ax.set_xlabel("PC"); ax.set_ylabel("Cumulative explained variance"); fig.tight_layout(); fig.savefig(out / "pca_scree.png", dpi=300); plt.close(fig)
load = np.abs(p.components_[:2]).max(axis=0); ii = np.argsort(-load)[:30]
pd.DataFrame({"feature": [feat[i] for i in ii], "max_abs_loading_pc1_pc2": load[ii]}).to_csv(out / "pca_top_loadings.csv", index=False)
print("[DONE]", out)
