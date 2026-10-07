# Phase II protocol (write this down BEFORE running anything)

Phase I asked: *does local 3D context add anything on top of a **frozen** ESM-2 head?*
The answer was yes, with a confirmatory test-split result and 30-seed replication.

Phase II asks the two questions a strong journal will ask about that answer:

1. **Does the spatial residual survive fine-tuning of the language model itself?**
   If the gain disappears once ESM-2 is adapted to the task, the Phase-I result is about a
   weak baseline rather than about cryptic sites.
2. **Does it hold on proteins that were never part of CryptoBench?**
   The CryptoBench test split was inspected during the exploratory phase of this project.
   It is a frozen-after-OOF held-out set, not a pristine one.

Everything below is fixed before the first Phase-II run. Deviations get written into the
manuscript as deviations.

---

## 1. What is fixed in advance

| Item | Decision |
|---|---|
| Primary endpoint family | `{EF@1%, BEDROC20}`, Holm-adjusted within the family |
| Unit of inference | protein (never seed, never residue) |
| Primary test | paired two-sided sign-flip permutation on the mean per-protein difference, 10^5 permutations |
| Effect size | paired Cohen's d_z with percentile bootstrap CI |
| Selection criterion | `J = 0.5·AP + 0.5·BEDROC20`, nested on the four official training folds |
| Architecture | unchanged from Phase I: `AnchoredGNN`, k=16, 2 layers, hidden 128, μ=0.5, ρ=0.01 |
| Fine-tuning mode | LoRA (r=8, α=16) on attention query/value; 3 epochs; encoder LR = head LR = 1e-4 |
| Fine-tuned encoder cross-fitting | fold *f* predicted by an encoder trained on the other three folds; test predicted by an encoder trained on all 885 |
| External homology cut-off | drop any external protein with >30% identity to any CryptoBench protein (train or test) |
| Matched-comparison coverage | a competitor must cover ≥95% of a protein's residues for that protein to enter the head-to-head set; excluded proteins are listed, never dropped silently |

## 2. Hypotheses

* **H1 (survival).** On the fine-tuned embeddings, the anchored spatial GNN beats the
  fine-tuned pLM's own head on BEDROC20 and EF@1% on training-fold OOF.
  *Decision rule:* if H1 fails on OOF, Phase II is reported as a negative result and the
  CryptoBench test set is **not** opened again. The manuscript then states plainly that the
  spatial residual is complementary to a frozen pLM but not to a fine-tuned one.
* **H2 (confirmation).** If H1 holds, the frozen Phase-II model beats the fine-tuned anchor
  on the CryptoBench test split.
* **H3 (external generalisation).** The Phase-I and Phase-II models beat their respective
  anchors on the homology-filtered external benchmark. This is the confirmatory evidence
  for the paper; the CryptoBench test result is supporting.
* **H4 (competitiveness).** On identical residues of identical proteins, our best model is
  at least as good as the published predictors we can run.

## 3. Test-set accounting (report this verbatim)

The CryptoBench test split will have been used **twice** by the end of Phase II:

1. Phase I, single recorded unlock of the frozen `stack_anchor`
   (`results/exp/_stats/test_unlock_log.jsonl`, one record).
2. Phase II, single recorded unlock of the frozen Phase-II model
   (`results_ft/exp/_stats/test_unlock_log.jsonl`, one record).

Two unlocks of one test set is two looks. The manuscript must say so, and must not present
the Phase-II test number as if it were a first look. This is exactly why the external
benchmark exists: it is the only split in this project that is opened once, ever.

Guard rails that are enforced in code:

* `scripts/19_stats_and_freeze.py` refuses to score the test split unless a model has been
  frozen and its prediction files still match the SHA-256 digests recorded at freeze time,
  and it appends to an append-only unlock log.
* The Phase-II runs live in a separate results tree (`results_ft/`), so the Phase-I freeze
  and unlock log are never overwritten.
* `scripts/24_build_external_benchmark.py` refuses to keep an external protein whose
  identifier appears in CryptoBench and logs every homology exclusion.

## 4. What must not happen

* No architecture or hyper-parameter chosen from any test or external number.
* No seed selection; seed statistics describe stability only.
* No protein dropped for being inconvenient. Proteins without Grid-Crypto support,
  without B-factors or with missing coordinates stay in; competitors' failures are reported.
* No re-freezing after an unlock in order to report a better model. If the frozen Phase-II
  model loses on test, that is the result.
* The external benchmark is not used to tune anything. It is scored once, at the end.

## 5. Order of operations

```
Phase II-A  fine-tune and re-run the pipeline on fine-tuned embeddings   (scripts 23, 10-19)
Phase II-B  decide H1 on OOF only
Phase II-C  if H1 holds: freeze one Phase-II model and unlock CryptoBench test once
Phase II-D  build the external benchmark and score every frozen model on it   (scripts 24, 25)
Phase II-E  matched head-to-head against external predictors                  (scripts 26)
```

`run_phase2.sh` executes A, B and D. Step C is a separate, deliberate command, and step E
needs the competitors' outputs, which have to be produced outside this repository.
