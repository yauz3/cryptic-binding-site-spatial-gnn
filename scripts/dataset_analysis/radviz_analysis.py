#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.plotting import radviz
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from analysis_common import load_training_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args()
X, y, pids, feat, cfg = load_training_matrix(args.config, max_rows=8000, max_features=250)
mi = mutual_info_classif(X, y, random_state=42); ii = np.argsort(-mi)[:min(15, len(mi))]
Z = MinMaxScaler().fit_transform(X[:, ii]); names = [feat[i] for i in ii]
df = pd.DataFrame(Z, columns=names); df["class"] = np.where(y == 1, "cryptic", "noncryptic")
out = Path(cfg["work"]["results"]) / "dataset_analysis" / "radviz"; out.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(11, 11)); radviz(df, "class", ax=ax, alpha=.25, s=12); ax.set_title("Radviz of MI-selected structural features — training only")
fig.tight_layout(); fig.savefig(out / "radviz.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("[DONE]", out)
