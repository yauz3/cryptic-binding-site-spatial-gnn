# Scientific guardrails

1. `jaccard_index`, `overlap_ratio`, `overlap_count`, cryptic residue counts, and other truth-derived columns are never used as features in residue-fusion models.
2. The official CryptoBench split is protein-level and must remain protein-level.
3. Apo/Holo label definitions are never mixed inside one evaluation.
4. Supervised feature selection is learned on training proteins only.
5. Threshold selection is done on a training fold.
6. The official test set is reserved for the final comparison.
7. Primary performance is protein-macro; pooled performance is secondary.
8. A candidate-level Top-K recovery number is not described as residue-level ROC-AUC.
9. Literature-only results are kept separate from rerun results.
10. A method is not named as an official external method unless the actual repository/model is executed.
