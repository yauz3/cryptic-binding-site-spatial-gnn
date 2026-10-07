#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from analysis_common import load_training_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--threshold", type=float, default=.75); args = ap.parse_args()
X, y, pids, feat, cfg = load_training_matrix(args.config, max_rows=20000, max_features=250)
mi = mutual_info_classif(X, y, random_state=42); ii = np.argsort(-mi)[:min(40, len(mi))]
df = pd.DataFrame(X[:, ii], columns=[feat[i] for i in ii]); corr = df.corr()
G = nx.Graph(); edges = []
for c in corr.columns: G.add_node(c)
for i in range(len(corr)):
    for j in range(i+1, len(corr)):
        r = float(corr.iloc[i, j])
        if abs(r) >= args.threshold:
            G.add_edge(corr.index[i], corr.columns[j], weight=abs(r), corr=r); edges.append((corr.index[i], corr.columns[j], r))
out = Path(cfg["work"]["results"]) / "dataset_analysis" / "networkx"; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(edges, columns=["feature_1", "feature_2", "correlation"]).to_csv(out / "edges.csv", index=False)
fig, ax = plt.subplots(figsize=(14, 14)); pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=[400 + 100*G.degree(n) for n in G], ax=ax, alpha=.8)
nx.draw_networkx_edges(G, pos, width=[1 + 2*G[u][v]["weight"] for u, v in G.edges()], ax=ax, alpha=.4)
nx.draw_networkx_labels(G, pos, font_size=6, ax=ax); ax.axis("off"); ax.set_title(f"Feature network | |r| ≥ {args.threshold}")
fig.tight_layout(); fig.savefig(out / "feature_network.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("[DONE]", out)
