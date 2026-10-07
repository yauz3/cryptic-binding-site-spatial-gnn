#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis_common import load_training_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args()
X, y, pids, feat, cfg = load_training_matrix(args.config, max_rows=30000, max_features=60)
df = pd.DataFrame(X, columns=feat); df["target"] = y; corr = df.corr()
out = Path(cfg["work"]["results"]) / "dataset_analysis" / "heatmap"; out.mkdir(parents=True, exist_ok=True); corr.to_csv(out / "correlation_matrix.csv")
fig, ax = plt.subplots(figsize=(15, 13)); im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto", cmap="coolwarm"); fig.colorbar(im, ax=ax)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=5); ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=5)
ax.set_title("Feature correlation heatmap — training only"); fig.tight_layout(); fig.savefig(out / "correlation_heatmap.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("[DONE]", out)
