#!/usr/bin/env python3
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness
import umap
from analysis_common import load_training_matrix, scaled

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args()
X, y, pids, feat, cfg = load_training_matrix(args.config, max_rows=20000, max_features=150)
Z = scaled(X); emb = umap.UMAP(n_neighbors=15, min_dist=.1, metric="euclidean", random_state=42).fit_transform(Z)
tw = trustworthiness(Z, emb, n_neighbors=10)
out = Path(cfg["work"]["results"]) / "dataset_analysis" / "umap"; out.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 6)); sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, s=7, alpha=.35); fig.colorbar(sc, ax=ax, label="cryptic residue")
ax.set_title(f"UMAP — training only | trustworthiness@10={tw:.3f}"); fig.tight_layout(); fig.savefig(out / "umap.png", dpi=300); plt.close(fig)
(out / "metrics.txt").write_text(f"trustworthiness_at_10={tw}\n")
print("[DONE]", out)
