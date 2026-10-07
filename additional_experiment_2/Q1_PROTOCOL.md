# Q1 experiment protocol (fix this before running anything)

Phase I established that a pLM-anchored spatial residual improves early enrichment over our own
frozen ESM-2 head. Phase II established that the residual is unaffected by a light adaptation of
the encoder, and that the adaptation itself did nothing. Neither phase answered the question a
strong journal will actually ask:

> Does this beat the methods already published on this benchmark, on the same proteins and the
> same residues, and does it generalise beyond CryptoBench?

Phase III is five experiments aimed at exactly that. No new architecture is developed. After
these five, model development stops.

---

## 1. Experiments

| ID | Question | Why it matters | New model training |
|---|---|---|---|
| **E1** | Do we beat published predictors on identical proteins and residues? | The single biggest gap: every p-value so far compares us with *our own* baseline | none |
| **E2** | Does the spatial gain survive a stronger backbone (ESM2-3B)? | Closes "your gain only fixes a weak 650M anchor" | yes |
| **E3** | Do better residue rankings become better *pocket* retrieval? | Turns a classification paper into a retrieval paper | none |
| **E4** | Does the gain hold on homology-filtered proteins outside CryptoBench? | The only split that is opened once, ever | yes (frozen models applied) |
| **E5** | Does it work on predicted (AlphaFold) structures? | Deployment setting; closes a stated limitation | yes (geometry swapped) |

E1 and E4 are the two that change the paper's standing. E3 and E5 close specific reviewer
objections. E2 removes the backbone objection.

### Why this differs from the obvious four-experiment plan

Three changes, each for a concrete reason.

1. **E5 (predicted structures) is added.** The method consumes experimental coordinates and
   B-factors. On a new target there is no crystal structure, so a reviewer will ask what happens
   on an AlphaFold model. The Phase-I B-factor ablation ($-0.004$ AP) hints that the answer is
   "not much", but a hint is not an experiment, and this one is cheap.
2. **E1 is split into two panels that are never mixed.** Panel A recomputes our metrics on
   exactly the protein subset a published method reported on, so the quoted numbers describe the
   same proteins; it yields no p-value, because a paired test needs the competitor's per-protein
   predictions. Panel B is the paired head-to-head on identical residues, and is the only panel
   that may carry a significance claim. Conflating them would repeat the mistake the current
   Table 11 avoids.
3. **Multiplicity and test-set accounting are fixed in advance** (Sections 2 and 3). Five
   experiments, up to four comparators and eight metrics is roughly a hundred tests. Without a
   prespecified confirmatory family, "p < 0.05 somewhere" is guaranteed.

---

## 2. Statistics, fixed before any result is seen

* **Unit of inference: the protein.** Always. Residue-level pooled metrics are descriptive.
* **Primary test:** paired two-sided sign-flip permutation on the mean per-protein difference,
  $10^5$ permutations. Wilcoxon reported as a sensitivity analysis.
* **Effect size:** paired Cohen's $d_z$ with percentile bootstrap CI, plus win/tie/loss.
* **Confirmatory family, fixed now:**

  | Experiment | Confirmatory comparison | Metrics |
  |---|---|---|
  | E1-B | anchored GNN vs each competitor, matched residues | AP, BEDROC20 |
  | E2 | anchored GNN vs seed-matched sequence head, ESM2-3B | AP, BEDROC20 |
  | E4 | anchored GNN vs sequence anchor, EXT20 | AP, BEDROC20 |

  Holm correction **within each experiment across its two metrics**, and **across the
  competitors within E1-B**. Everything else in this protocol, including every EF@1, every
  pocket metric, EXT30, E3 and E5, is secondary and reported without significance claims unless
  it survives the same correction, which will be stated explicitly where it does.
* **EF@1% is secondary everywhere.** For a 300-residue protein the top 1% is three residues; the
  metric is too discrete to anchor a claim, as both earlier phases showed ($d_z \approx 0.2$
  against $0.55$ for BEDROC20).
* **No metric, threshold or clustering parameter is chosen after seeing a result.** E3's
  clustering settings are fixed below and a sweep is reported to show the ordering is not an
  artefact of them.

### Success criteria, written in advance

| Experiment | Success | Partial success | Failure |
|---|---|---|---|
| E1-B | ΔAP > 0 **and** ΔBEDROC20 > 0 with Holm $p<0.05$ against at least one strong comparator | comparable AP but better BEDROC20 (claim: "equivalent global discrimination, better early enrichment") | loses on both against every comparator |
| E2 | spatial gain on ESM2-3B is within the CI of the 650M gain | gain shrinks but CI excludes 0 | gain disappears |
| E3 | DCC@N and RRO-success higher than the sequence anchor | DCC@(N+2) only | no pocket-level difference |
| E4 | ΔAP > 0 and ΔBEDROC20 > 0 with 95% CI excluding 0 on EXT20 | holds on EXT30 only | CI includes 0 on both |
| E5 | gain retained within its CI on predicted structures | gain reduced but positive | gain disappears |

A partial success is a publishable result and must be written as one. A failure is written as a
failure; that is what the protocol is for.

---

## 3. Test-set accounting

The CryptoBench test split has already been scored **twice** (Phase I, Phase II). Phase III adds
at most **one** more:

* **E1, E3** re-score predictions that already exist. They open no new model and add no unlock;
  they are re-analyses of the frozen Phase-I predictions under different evaluators.
* **E2 trains a new model** (ESM2-3B backbone). It therefore follows the full protocol: nested
  OOF selection, one freeze against the matched sequence head, **one unlock = the third and
  final use of the CryptoBench test split in this project.**
* **E5 swaps geometry** for proteins already in the test split. It introduces no selection, so
  it is reported as a sensitivity analysis of the frozen model rather than a new unlock. The
  manuscript must say this plainly rather than burying it.
* **E4 is scored once, ever.** It is the only clean confirmatory split and carries the
  generalisation claim.

Hard rule: if E2's OOF result does not show the spatial gain on the 3B backbone, **do not unlock
the test split for E2**. Report it as an OOF-only negative result.

---

## 4. Comparators

| Method | Access | Role |
|---|---|---|
| CryptoBench pLM-NN (ESM2-3B + MLP) | OSF `pz4a9`, GitHub `skrhakv/CryptoBench` | benchmark baseline |
| Hidden in protein sequences (fine-tuned ESM2-3B) | GitHub `skrhakv/cryptic-finetuning` | strongest sequence comparator |
| Seq2Pocket | GitHub `skrhakv/seq2pocket` | residue + pocket-level comparator |
| UniPocket | published 193-protein CryptoBench subset | Panel A; Panel B only if predictions are obtainable |
| PocketMiner, P2Rank | public tools | structure-based comparators from the benchmark paper |

Running a competitor yourself is always preferred to quoting it. Where a competitor cannot be
run, it appears in Panel A only. Where a competitor fails on some proteins, those proteins are
listed and the matched set is reported, never silently reduced.

---

## 5. Order of work

```
E1-A  subset recomputation            (hours; no GPU)
E3    pocket-level evaluation         (hours; no GPU)
E1-B  matched head-to-head            (days; depends on running competitors)
E2    ESM2-3B backbone                (GPU; embeddings + OOF + one freeze)
E5    predicted structures            (GPU-light; needs AlphaFold models)
E4    external benchmark              (GPU; last, scored once)
```

E1-A and E3 need no new computation beyond what is already on disk and should be run first,
because they tell you within a day whether the paper's claim survives contact with the published
numbers.

---

## 6. What must not happen

* No architecture or hyper-parameter chosen from any test, subset or external number.
* No comparator dropped because it is inconvenient; no protein dropped because a competitor
  failed on it.
* No pocket-level number quoted from another paper beside ours. Pocket metrics depend on the
  clustering and the success radius; only numbers produced by the identical pipeline in
  `scripts/30` are comparable.
* No second unlock of the external split under any circumstances.
* If E1-B shows we lose to a fine-tuned 3B comparator, that goes in the abstract, not in a
  footnote.
