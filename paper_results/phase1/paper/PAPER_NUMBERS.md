# Key numbers for the manuscript

- Frozen model: `stack_anchor` (frozen 2026-10-02 13:19:05; criterion {'ap': 0.5, 'bedroc20': 0.5}; selected on the 885 training proteins, official-fold OOF, nested selection).
- Test status: test reported; number of unlocks recorded: 1.
- Unit of analysis: protein. Primary test: paired two-sided sign-flip permutation test on the mean per-protein difference (100000 permutations); Holm adjustment within the primary family ['ef1', 'bedroc20'].
- Effect size: paired Cohen's d_z = mean(diff)/SD(diff) over proteins, percentile-bootstrap 95% CI.
- BEDROC50/80 are reported as secondary: alpha * (active fraction) is not << 1 at residue level.

## OOF (885 proteins): Anchored stack vs pLM anchor

| metric | anchor | method | delta | 95% CI | p (perm.) | p Holm | d_z | W/T/L |
|---|---|---|---|---|---|---|---|---|
| ap | 0.3766 | 0.4354 | +0.0588 | [+0.0525, +0.0652] | 1.0e-05 | -- | 0.621 | 654/1/230 |
| auc | 0.8024 | 0.8350 | +0.0327 | [+0.0277, +0.0376] | 1.0e-05 | -- | 0.431 | 725/1/159 |
| ef1 | 7.4830 | 8.6304 | +1.1474 | [+0.8122, +1.5101] | 1.0e-05 | 2.0e-05 | 0.217 | 213/579/93 |
| ef5 | 5.8219 | 6.7119 | +0.8900 | [+0.7312, +1.0543] | 1.0e-05 | -- | 0.361 | 427/308/150 |
| bedroc20 | 0.4571 | 0.5127 | +0.0555 | [+0.0486, +0.0627] | 1.0e-05 | 2.0e-05 | 0.513 | 625/1/259 |
| bedroc50 | 0.4368 | 0.4860 | +0.0492 | [+0.0410, +0.0577] | 1.0e-05 | -- | 0.388 | 578/1/306 |
| bedroc80 | 0.4459 | 0.4927 | +0.0469 | [+0.0374, +0.0563] | 1.0e-05 | -- | 0.324 | 564/1/320 |
| mcc | 0.3070 | 0.3343 | +0.0273 | [+0.0191, +0.0356] | 1.0e-05 | -- | 0.218 | 543/32/310 |
| f1 | 0.3311 | 0.3502 | +0.0191 | [+0.0113, +0.0270] | 1.0e-05 | -- | 0.161 | 442/167/276 |

## TEST (222 proteins): Anchored stack vs pLM anchor

| metric | anchor | method | delta | 95% CI | p (perm.) | p Holm | d_z | W/T/L |
|---|---|---|---|---|---|---|---|---|
| ap | 0.4038 | 0.4700 | +0.0662 | [+0.0532, +0.0798] | 1.0e-05 | -- | 0.634 | 168/0/54 |
| auc | 0.8244 | 0.8542 | +0.0298 | [+0.0199, +0.0398] | 1.0e-05 | -- | 0.391 | 181/1/40 |
| ef1 | 7.8757 | 9.1863 | +1.3106 | [+0.4943, +2.2035] | 0.0020 | 0.0020 | 0.205 | 61/133/28 |
| ef5 | 6.2448 | 7.3093 | +1.0644 | [+0.7958, +1.3280] | 1.0e-05 | -- | 0.516 | 113/78/31 |
| bedroc20 | 0.4874 | 0.5481 | +0.0607 | [+0.0465, +0.0750] | 1.0e-05 | 2.0e-05 | 0.558 | 161/0/61 |
| bedroc50 | 0.4551 | 0.5133 | +0.0581 | [+0.0393, +0.0773] | 1.0e-05 | -- | 0.397 | 149/0/73 |
| bedroc80 | 0.4603 | 0.5152 | +0.0549 | [+0.0322, +0.0791] | 2.0e-05 | -- | 0.310 | 144/0/78 |
| mcc | 0.3275 | 0.3846 | +0.0571 | [+0.0413, +0.0724] | 1.0e-05 | -- | 0.479 | 154/6/62 |
| f1 | 0.3493 | 0.3989 | +0.0496 | [+0.0346, +0.0650] | 1.0e-05 | -- | 0.436 | 132/37/53 |

## Marginal contribution of Grid-derived stack inputs (train OOF, nested re-stacking)

| Stack variant                           | Metric        |   Value |   Delta vs anchor |   p vs anchor |   Delta vs full stack | 95% CI vs full     |   p vs full |   d_z vs full |
|:----------------------------------------|:--------------|--------:|------------------:|--------------:|----------------------:|:-------------------|------------:|--------------:|
| all inputs (frozen stack definition)    | AP            |  0.4354 |            0.0588 |        0.0000 |                0.0000 | [0.0000, 0.0000]   |      1.0000 |      nan      |
| all inputs (frozen stack definition)    | EF@1%         |  8.6304 |            1.1474 |        0.0000 |                0.0000 | [0.0000, 0.0000]   |      1.0000 |      nan      |
| all inputs (frozen stack definition)    | BEDROC$_{20}$ |  0.5127 |            0.0555 |        0.0000 |                0.0000 | [0.0000, 0.0000]   |      1.0000 |      nan      |
| without Grid-derived inputs             | AP            |  0.4275 |            0.0509 |        0.0000 |               -0.0079 | [-0.0105, -0.0054] |      0.0000 |       -0.2089 |
| without Grid-derived inputs             | EF@1%         |  8.4437 |            0.9607 |        0.0000 |               -0.1867 | [-0.4060, 0.0257]  |      0.0932 |       -0.0565 |
| without Grid-derived inputs             | BEDROC$_{20}$ |  0.5044 |            0.0473 |        0.0000 |               -0.0082 | [-0.0108, -0.0058] |      0.0000 |       -0.2164 |
| Grid-derived inputs only                | AP            |  0.3953 |            0.0186 |        0.0000 |               -0.0401 | [-0.0459, -0.0345] |      0.0000 |       -0.4730 |
| Grid-derived inputs only                | EF@1%         |  7.8118 |            0.3288 |        0.0000 |               -0.8186 | [-1.1573, -0.4815] |      0.0001 |       -0.1625 |
| Grid-derived inputs only                | BEDROC$_{20}$ |  0.4726 |            0.0155 |        0.0000 |               -0.0401 | [-0.0466, -0.0339] |      0.0000 |       -0.4137 |
| spatial GNN only                        | AP            |  0.4269 |            0.0503 |        0.0000 |               -0.0085 | [-0.0106, -0.0064] |      0.0000 |       -0.2623 |
| spatial GNN only                        | EF@1%         |  8.4553 |            0.9723 |        0.0000 |               -0.1751 | [-0.3858, 0.0259]  |      0.0901 |       -0.0572 |
| spatial GNN only                        | BEDROC$_{20}$ |  0.5063 |            0.0492 |        0.0000 |               -0.0063 | [-0.0086, -0.0042] |      0.0000 |       -0.1858 |
| leave out: 3D logit smoothing           | AP            |  0.4356 |            0.0590 |        0.0000 |                0.0002 | [-0.0001, 0.0005]  |      0.2534 |        0.0391 |
| leave out: 3D logit smoothing           | EF@1%         |  8.6653 |            1.1823 |        0.0000 |                0.0349 | [-0.0168, 0.0918]  |      0.2222 |        0.0420 |
| leave out: 3D logit smoothing           | BEDROC$_{20}$ |  0.5129 |            0.0558 |        0.0000 |                0.0002 | [-0.0000, 0.0005]  |      0.0993 |        0.0552 |
| leave out: Grid pocket consensus        | AP            |  0.4330 |            0.0564 |        0.0000 |               -0.0024 | [-0.0034, -0.0014] |      0.0000 |       -0.1538 |
| leave out: Grid pocket consensus        | EF@1%         |  8.6476 |            1.1646 |        0.0000 |                0.0172 | [-0.1302, 0.1667]  |      0.8222 |        0.0077 |
| leave out: Grid pocket consensus        | BEDROC$_{20}$ |  0.5109 |            0.0538 |        0.0000 |               -0.0017 | [-0.0027, -0.0008] |      0.0008 |       -0.1189 |
| leave out: GBM residual (Grid+geom+ctx) | AP            |  0.4345 |            0.0579 |        0.0000 |               -0.0009 | [-0.0026, 0.0007]  |      0.2655 |       -0.0371 |
| leave out: GBM residual (Grid+geom+ctx) | EF@1%         |  8.7154 |            1.2324 |        0.0000 |                0.0850 | [-0.0926, 0.2674]  |      0.3554 |        0.0315 |
| leave out: GBM residual (Grid+geom+ctx) | BEDROC$_{20}$ |  0.5106 |            0.0535 |        0.0000 |               -0.0021 | [-0.0037, -0.0004] |      0.0130 |       -0.0822 |
| leave out: Anchored spatial GNN         | AP            |  0.4042 |            0.0276 |        0.0000 |               -0.0312 | [-0.0366, -0.0258] |      0.0000 |       -0.3854 |
| leave out: Anchored spatial GNN         | EF@1%         |  8.1370 |            0.6540 |        0.0000 |               -0.4934 | [-0.8447, -0.1495] |      0.0024 |       -0.0978 |
| leave out: Anchored spatial GNN         | BEDROC$_{20}$ |  0.4809 |            0.0238 |        0.0000 |               -0.0317 | [-0.0379, -0.0260] |      0.0000 |       -0.3423 |
| leave out: Top-pair ranking loss        | AP            |  0.4326 |            0.0559 |        0.0000 |               -0.0028 | [-0.0044, -0.0013] |      0.0003 |       -0.1192 |
| leave out: Top-pair ranking loss        | EF@1%         |  8.6776 |            1.1946 |        0.0000 |                0.0472 | [-0.1114, 0.2036]  |      0.5622 |        0.0198 |
| leave out: Top-pair ranking loss        | BEDROC$_{20}$ |  0.5095 |            0.0524 |        0.0000 |               -0.0032 | [-0.0046, -0.0017] |      0.0001 |       -0.1433 |
| leave out: Soft-BEDROC loss             | AP            |  0.4353 |            0.0587 |        0.0000 |               -0.0001 | [-0.0005, 0.0003]  |      0.7068 |       -0.0125 |
| leave out: Soft-BEDROC loss             | EF@1%         |  8.6119 |            1.1289 |        0.0000 |               -0.0185 | [-0.0817, 0.0375]  |      0.5515 |       -0.0210 |
| leave out: Soft-BEDROC loss             | BEDROC$_{20}$ |  0.5126 |            0.0555 |        0.0000 |               -0.0001 | [-0.0004, 0.0003]  |      0.6476 |       -0.0156 |
| leave out: Aux. distance task           | AP            |  0.4354 |            0.0588 |        0.0000 |               -0.0000 | [-0.0002, 0.0001]  |      0.7801 |       -0.0098 |
| leave out: Aux. distance task           | EF@1%         |  8.6406 |            1.1576 |        0.0000 |                0.0102 | [0.0000, 0.0305]   |      1.0000 |        0.0336 |
| leave out: Aux. distance task           | BEDROC$_{20}$ |  0.5126 |            0.0555 |        0.0000 |               -0.0000 | [-0.0002, 0.0002]  |      0.9575 |       -0.0019 |
