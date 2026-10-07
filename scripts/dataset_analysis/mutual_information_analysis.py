#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from analysis_common import load_training_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args()
X, y, pids, feat, cfg = load_training_matrix(args.config, max_rows=60000, max_features=600)
mi = mutual_info_classif(X, y, random_state=42); df = pd.DataFrame({"feature": feat, "mutual_information": mi}).sort_values("mutual_information", ascending=False)
out = Path(cfg["work"]["results"]) / "dataset_analysis" / "mutual_information"; out.mkdir(parents=True, exist_ok=True); df.to_csv(out / "mutual_information_all.csv", index=False)
top = df.head(30).iloc[::-1]; fig, ax = plt.subplots(figsize=(10, 8)); ax.barh(top.feature, top.mutual_information); ax.set_xlabel("Mutual information"); ax.set_title("Top nonlinear feature associations — training only")
fig.tight_layout(); fig.savefig(out / "mutual_information_top30.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("[DONE]", out)
