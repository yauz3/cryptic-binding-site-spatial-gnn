E1-B — MATCHED Seq2Pocket COMPARISON
====================================

Comparator:
Seq2Pocket CBS ESM2-3B published checkpoint

Evaluation:
222 / 222 CryptoBench test proteins
73,933 matched residues
100% residue coverage
0 excluded proteins
0 inference failures

GNN anchor vs Seq2Pocket
------------------------
AP:
Seq2Pocket = 0.459210
GNN        = 0.465649
Delta      = +0.006439
95% CI     = [-0.013000, +0.025099]
p          = 0.511865
Cohen dz   = 0.044166
W/T/L      = 118/0/104

ROC-AUC:
Seq2Pocket = 0.839778
GNN        = 0.854293
Delta      = +0.014515
95% CI     = [+0.001347, +0.027650]
p          = 0.026930
Cohen dz   = 0.148631
W/T/L      = 119/0/103

BEDROC20:
Seq2Pocket = 0.532473
GNN        = 0.548723
Delta      = +0.016250
95% CI     = [-0.005245, +0.036904]
p          = 0.130479
Holm p     = 0.260957
Cohen dz   = 0.102018
W/T/L      = 124/0/98

Interpretation:
The anchored GNN reaches numerically higher AP and BEDROC20 than
Seq2Pocket under an identical 222-protein residue-level evaluation,
but these differences are not statistically significant.
ROC-AUC is significantly higher nominally.

The frozen pLM baseline is substantially worse than Seq2Pocket,
whereas spatial anchoring closes that gap completely.
