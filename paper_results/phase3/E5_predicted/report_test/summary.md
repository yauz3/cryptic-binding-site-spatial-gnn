# predicted structures (pLDDT as deposited): 222 proteins

- Reference: `base_plm`. Unit of analysis: protein.
- Primary test: paired sign-flip permutation (100000 permutations), Holm within ['ef1', 'bedroc20'].
- Residues: 73933; positives: 3994.

## Anchored spatial GNN vs ESM-2 MLP (pLM anchor)

| metric | ref | model | delta | 95% CI | p | p Holm | d_z | W/T/L |
|---|---|---|---|---|---|---|---|---|
| ap | 0.4038 | 0.3895 | -0.0143 | [-0.0198, -0.0088] | 1.0e-05 | -- | -0.340 | 82/0/140 |
| auc | 0.8244 | 0.8181 | -0.0062 | [-0.0088, -0.0035] | 1.0e-05 | -- | -0.308 | 70/1/151 |
| ef1 | 7.8757 | 7.6023 | -0.2734 | [-0.7194, +0.2106] | 0.2663 | 0.2663 | -0.076 | 20/172/30 |
| ef5 | 6.2448 | 6.0514 | -0.1934 | [-0.3523, -0.0409] | 0.0147 | -- | -0.162 | 46/116/60 |
| bedroc20 | 0.4874 | 0.4724 | -0.0150 | [-0.0212, -0.0084] | 2.0e-05 | 4.0e-05 | -0.307 | 80/0/142 |
| bedroc50 | 0.4551 | 0.4418 | -0.0134 | [-0.0229, -0.0036] | 0.0087 | -- | -0.176 | 94/0/128 |
| bedroc80 | 0.4603 | 0.4472 | -0.0131 | [-0.0262, +0.0003] | 0.0492 | -- | -0.132 | 97/0/125 |
