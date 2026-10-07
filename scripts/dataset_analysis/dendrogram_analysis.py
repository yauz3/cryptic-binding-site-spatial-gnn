#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from analysis_common import load_training_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args()
X, y, pids, feat, cfg = load_training_matrix(args.config, max_rows=25000, max_features=80)
corr = pd.DataFrame(X, columns=feat).corr().fillna(0).clip(-1, 1); dist = 1 - np.abs(corr); np.fill_diagonal(dist.values, 0)
Z = linkage(squareform(dist.values, checks=False), method="average")
out = Path(cfg["work"]["results"]) / "dataset_analysis" / "dendrogram"; out.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(16, 9)); dendrogram(Z, labels=feat, leaf_rotation=90, leaf_font_size=5, ax=ax)
ax.set_ylabel("1 - |correlation|"); ax.set_title("Structural feature dendrogram — training only"); fig.tight_layout(); fig.savefig(out / "feature_dendrogram.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("[DONE]", out)
