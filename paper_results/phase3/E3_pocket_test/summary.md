# Pocket-level retrieval, test split

- 222 proteins, 1100 annotated cryptic pockets.
- Identical post-processing for every method: top 10% of residues, single-linkage at 8 A, min 3 residues, pocket score = mean of top 5.
- DCC success at 4 A; RRO = Jaccard, success at 0.50.
- Reference: ESM-2 MLP (pLM anchor). Pocket primary family: ['dcc_topN', 'rro_success'].

- **ESM-2 MLP (pLM anchor)** (n=222): DCC@1 0.223, DCC@N 0.233, DCC@(N+2) 0.237, RRO 0.266, RRO success 0.162
- **Anchored spatial GNN** (n=222): DCC@1 0.264, DCC@N 0.292, DCC@(N+2) 0.297, RRO 0.306, RRO success 0.230
- **Anchored stack** (n=222): DCC@1 0.265, DCC@N 0.288, DCC@(N+2) 0.292, RRO 0.308, RRO success 0.230

| method | metric | ref | method | delta | 95% CI | p | d_z | W/T/L |
|---|---|---|---|---|---|---|---|---|
| Anchored spatial GNN | dcc_top1 | 0.2233 | 0.2636 | +0.0403 | [+0.0013, +0.0803] | 0.0521 | 0.131 | 22/186/14 |
| Anchored spatial GNN | dcc_top3 | 0.2373 | 0.2967 | +0.0594 | [+0.0199, +0.1008] | 0.0048 | 0.189 | 26/183/13 |
| Anchored spatial GNN | dcc_top5 | 0.2373 | 0.2967 | +0.0594 | [+0.0199, +0.1008] | 0.0048 | 0.189 | 26/183/13 |
| Anchored spatial GNN | dcc_topN | 0.2328 | 0.2922 | +0.0594 | [+0.0199, +0.1008] | 0.0048 | 0.189 | 26/183/13 |
| Anchored spatial GNN | dcc_topN2 | 0.2373 | 0.2967 | +0.0594 | [+0.0199, +0.1008] | 0.0048 | 0.189 | 26/183/13 |
| Anchored spatial GNN | rro | 0.2659 | 0.3059 | +0.0399 | [+0.0295, +0.0510] | 1.0e-05 | 0.482 | 128/42/52 |
| Anchored spatial GNN | rro_recall | 0.5196 | 0.6106 | +0.0910 | [+0.0729, +0.1097] | 1.0e-05 | 0.636 | 124/72/26 |
| Anchored spatial GNN | rro_success | 0.1622 | 0.2297 | +0.0676 | [+0.0314, +0.1081] | 0.0016 | 0.225 | 18/201/3 |
| Anchored spatial GNN | site_found | 0.2883 | 0.3559 | +0.0676 | [+0.0270, +0.1126] | 0.0041 | 0.205 | 20/197/5 |
| Anchored stack | dcc_top1 | 0.2233 | 0.2654 | +0.0421 | [+0.0027, +0.0827] | 0.0427 | 0.137 | 25/181/16 |
| Anchored stack | dcc_top3 | 0.2373 | 0.2920 | +0.0548 | [+0.0136, +0.0955] | 0.0099 | 0.173 | 29/178/15 |
| Anchored stack | dcc_top5 | 0.2373 | 0.2920 | +0.0548 | [+0.0136, +0.0955] | 0.0099 | 0.173 | 29/178/15 |
| Anchored stack | dcc_topN | 0.2328 | 0.2875 | +0.0548 | [+0.0136, +0.0955] | 0.0099 | 0.173 | 29/178/15 |
| Anchored stack | dcc_topN2 | 0.2373 | 0.2920 | +0.0548 | [+0.0136, +0.0955] | 0.0099 | 0.173 | 29/178/15 |
| Anchored stack | rro | 0.2659 | 0.3083 | +0.0424 | [+0.0315, +0.0538] | 1.0e-05 | 0.500 | 137/37/48 |
| Anchored stack | rro_recall | 0.5196 | 0.6198 | +0.1002 | [+0.0815, +0.1196] | 1.0e-05 | 0.676 | 133/63/26 |
| Anchored stack | rro_success | 0.1622 | 0.2297 | +0.0676 | [+0.0270, +0.1081] | 0.0027 | 0.214 | 19/199/4 |
| Anchored stack | site_found | 0.2883 | 0.3468 | +0.0586 | [+0.0090, +0.1081] | 0.0296 | 0.158 | 22/191/9 |