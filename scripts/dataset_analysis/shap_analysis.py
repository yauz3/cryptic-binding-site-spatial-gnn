#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from analysis_common import load_training_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args()
X, y, pids, feat, cfg = load_training_matrix(args.config, max_rows=50000, max_features=250)
rng = np.random.default_rng(42); idx = rng.permutation(len(y)); cut = int(.8 * len(y)); tr, va = idx[:cut], idx[cut:]
m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8, eval_metric="logloss", n_jobs=-1, random_state=42)
m.fit(X[tr], y[tr]); take = va[:min(3000, len(va))]
ex = shap.TreeExplainer(m); sv = ex.shap_values(X[take])
out = Path(cfg["work"]["results"]) / "dataset_analysis" / "shap"; out.mkdir(parents=True, exist_ok=True)
mean = np.abs(sv).mean(0); pd.DataFrame({"feature": feat, "mean_abs_shap": mean}).sort_values("mean_abs_shap", ascending=False).to_csv(out / "shap_importance.csv", index=False)
shap.summary_plot(sv, X[take], feature_names=feat, max_display=25, show=False); plt.tight_layout(); plt.savefig(out / "shap_beeswarm.png", dpi=300, bbox_inches="tight"); plt.close()
print("[DONE]", out)
