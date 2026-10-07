# Final paper pipeline (scripts 10-22)

This file supersedes the Grid-centred description in `README.md` for the manuscript.

## What is claimed
- **Mechanistic core:** `gnn_anchor` - a pLM-anchored spatial GNN (z = z_pLM + zero-initialised graph residual).
- **Frozen performance model:** `stack_anchor`, selected on the 885 training proteins only
  (official-fold OOF, nested selection, J = 0.5 AP + 0.5 BEDROC20).
- **Grid-Crypto:** evaluated as a marginal structural contribution (Table 4 and the post-hoc
  sensitivity analysis); it is not assumed to be the main contribution.

## One command
```bash
bash setup_final_workspace.sh --start
```
This links the caches and copies the frozen pilot results from the old working suite
(`OLD=...` to override), runs the read-only preflight and, only if it passes, starts the
paper run in the background (log: `paper_final_nohup.log`, PID in `paper_final.pid`).
Phases: 0 preflight (read-only) -> A 30 independent replicates of the frozen pipeline (seeds 100-129)
-> B OOF replicate analysis -> B2 post-hoc OOF-only Grid sensitivity (script 14b) -> C OOF tables/figures
-> D pre-test protocol snapshot -> E single test unlock -> F replicate analysis incl. test -> G final tables/figures.
Interrupted runs resume with the same command; the snapshot and the unlock are never repeated.

## Statistics
- Unit of inference: protein. Primary test: paired sign-flip permutation test on the mean per-protein
  difference, Holm within {EF@1%, BEDROC20}. Effect size: paired Cohen's d_z with bootstrap CI.
- Seeds describe training stability only (mean +- SD, seeds improved); seed-level p/d are not generalisation evidence.
- Crossed (protein x seed) bootstrap CI carries both sources of uncertainty.
- The CryptoBench test set was inspected during earlier development: report it as a
  "frozen-after-OOF held-out evaluation", not as a pristine untouched test.

## Known limitation kept on purpose
In the frozen pilot every GBM fit stopped at the 10-tree floor (class-weighted objective, unweighted
early-stopping metric). The frozen pipeline is replicated unchanged. `scripts/14b_*` repeats the GBM
arm with class-weighted early stopping as a post-hoc, train-OOF-only analysis (`posthoc_gbmw_*`,
`results/paper/posthoc_grid_descriptor_sensitivity_oof.csv`); it never enters the frozen stack.

## Outputs
`results/paper/` (Tables 1-7, Figures 1-5, PAPER_NUMBERS.md, PRETEST_FINAL_PROTOCOL.json),
`results/replicates/_analysis/` (replicate tables/figures), `results/exp/_stats/` (freeze, unlock log).
