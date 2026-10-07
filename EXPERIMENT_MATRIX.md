# Experiment matrix

## A. Direct same-split models

| ID | Representation | Head |
|---|---|---|
| A0 | ESM2 | MLP |
| A1 | Grid-Crypto residue structural vector | MLP |
| A2 | ESM2 + Grid-Crypto concatenation | MLP |
| A3 | ESM2 tower + structural tower | fusion MLP |

Primary endpoints: protein-macro ROC-AUC, macro AP, macro MCC, F1, Recall@1%, EF@1%.

## B. External score augmentation

For a standardized external score:

- B0: external score alone
- B1: external score + selected Grid-Crypto features

## C. Structural ablation

- all
- distance_geometry
- counts
- face
- filtration_energy
- grid_summary

## D. Original grid-ranking ablation

- 3 labels: Jaccard / overlap ratio / overlap count
- 2 regressors: HGB / ExtraTrees
- score: reciprocal-rank / pred-rank / exponential / reward-penalty
- Top-M: all / 100 / 50 / 20

## E. Dataset analysis

PCA, t-SNE, UMAP, ANOVA, mutual information, heatmap, Radviz, NetworkX, dendrogram, SHAP.
