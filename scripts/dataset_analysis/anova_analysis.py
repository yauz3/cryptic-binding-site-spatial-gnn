#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from analysis_common import load_training_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args()
X, y, pids, feat, cfg = load_training_matrix(args.config, max_rows=100000, max_features=1000)
f, p = f_classif(X, y)
df = pd.DataFrame({"feature": feat, "F_score": f, "p_value": p}).replace([np.inf, -np.inf], np.nan).dropna(subset=["F_score"]).sort_values("F_score", ascending=False)
out = Path(cfg["work"]["results"]) / "dataset_analysis" / "anova"; out.mkdir(parents=True, exist_ok=True); df.to_csv(out / "anova_all.csv", index=False)
top = df.head(30).iloc[::-1]; fig, ax = plt.subplots(figsize=(10, 8)); ax.barh(top.feature, top.F_score); ax.set_xlabel("ANOVA F-score"); ax.set_title("Top structural features — training only")
fig.tight_layout(); fig.savefig(out / "anova_top30.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("[DONE]", out)
