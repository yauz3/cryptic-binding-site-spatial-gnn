# additional_experiment_2 — Phase III (the Q1 package)

Phase I: a pLM-anchored spatial residual beats **our own** frozen ESM-2 head.
Phase II: that residual is unchanged by a light adaptation of the encoder, and the adaptation
itself did nothing.

Neither answers the question a strong journal asks. Every p-value in the paper so far compares
us with our own baseline; our pooled AUPRC (0.408) sits below the published fine-tuned-pLM value
(0.475) on a different protein subset; and nothing has been evaluated outside CryptoBench.
Phase III is five experiments aimed exactly at that, after which model development stops.

**Read `Q1_PROTOCOL.md` before running anything.** It fixes the confirmatory family, the success
criteria (including what counts as failure) and the test-set accounting in advance.

| ID | Question | New training | Script |
|---|---|---|---|
| E1-A | Our metrics on exactly the protein subset a published method reported on | none | `32_subset_report.py` |
| E1-B | Paired head-to-head with competitors on identical residues | none | `26_matched_external_predictors.py` |
| E2 | Does the spatial gain survive a stronger backbone (ESM2-3B)? | yes | `02`, `11`, `15`, `19` |
| E3 | Do better residue rankings become better *pocket* retrieval (DCC / RRO)? | none | `30_pocket_level_eval.py` |
| E4 | Does it hold on homology-filtered proteins outside CryptoBench? | frozen models applied | `24`, `25`, `30` |
| E5 | Does it work on predicted (AlphaFold) structures? | geometry swapped | `31_predicted_structure_benchmark.py` |

E1 and E4 are the two that change the paper's standing. E1-A and E3 need no new computation and
should be run first: within a day they tell you whether the claim survives contact with the
published numbers.

## Install

```bash
cd cryptobench_gridfusion_suite          # the Phase-I suite, with Phase II already installed
unzip additional_experiment_2.zip        # into this folder
bash additional_experiment_2/install.sh
bash additional_experiment_2/run_q1.sh check
```

`install.sh` adds five new files and changes nothing that exists. It fails loudly if the
Phase-II package is missing or out of date, because `run_q1.sh` relies on `--out-dir` in scripts
25/26 and `--baseline` in script 19; without them the Q1 outputs would be written into the
Phase-I results tree.

Everything Phase III produces goes to `additional_experiment_2/results/`.

## Running

```bash
cd cryptobench_gridfusion_suite

bash additional_experiment_2/run_q1.sh e1a         # published subsets (no GPU, minutes)
bash additional_experiment_2/run_q1.sh e3          # pocket-level DCC/RRO (no GPU, ~1 h with --sweep)
bash additional_experiment_2/run_q1.sh e1b         # matched head-to-head (needs competitor output)
bash additional_experiment_2/run_q1.sh e2-embed    # ESM2-3B embeddings (GPU, large)
bash additional_experiment_2/run_q1.sh e2-train    # OOF models on 3B; decision on OOF only
bash additional_experiment_2/run_q1.sh e2-unlock   # ONLY if the OOF gain held
bash additional_experiment_2/run_q1.sh e5          # predicted structures + pLDDT sensitivity
bash additional_experiment_2/run_q1.sh e4          # external benchmark, EXT20 and EXT30
```

Useful variables: `SEEDS="0 1 2"`, `MODELS="base_plm gnn_anchor stack_anchor"`,
`ESM3B=facebook/esm2_t36_3B_UR50D`, `PRED_DIR=...`, `EXT_NAME`/`EXT_STRUCTURES`/`EXT_LABELS`.

ESM2-3B embeddings are 2560-dimensional: roughly twice the disk of the 650M cache, and the
extraction needs a GPU with enough memory for the model. `scripts/02` already takes
`--model-name`, and `AnchoredGNN` takes the embedding width from the cache, so nothing else
changes. The architecture is identical across backbones, which is the point of E2.

## What you need to supply

```
additional_experiment_2/
  published_subsets/       <name>.txt            one PDB id per line: the proteins a published
                                                  method evaluated on (E1-A)
                           <name>_values.csv     optional: method,metric,value[,subset_name,source]
  comparator_predictions/  <method>.csv          pdb_id,residue_key,score[,prediction][,method]
                                                  (E1-B and E3)
  predicted_structures/    <pdb_id>.cif          AlphaFold models for the test proteins (E5)
  external_data/           <name>_structures/    apo structures outside CryptoBench (E4)
                           <name>_labels.json    CryptoBench dataset.json schema
```

### Competitor predictions

Residue numbering is where this goes wrong, so use the converter rather than writing CSVs by
hand. It maps by explicit keys where they exist and by global sequence alignment where they do
not, checks the result against the struct cache, and refuses to write a file whose coverage or
alignment identity is too low.

```bash
# scores indexed by UniProt position or by the tool's own sequence
python scripts/external/convert_comparators.py --format seq-aligned \
    --input runs/hidden_scores --sequences runs/hidden_sequences.fasta \
    --config config.yaml --method "Hidden (ESM2-3B finetuned)" \
    --out additional_experiment_2/comparator_predictions/hidden.csv

# scores already carrying chain and residue number
python scripts/external/convert_comparators.py --format keyed \
    --input runs/unipocket_cry.csv --config config.yaml --method UniPocket \
    --col-pdb pdb --col-chain chain --col-resi resnum --col-score cry_prob \
    --out additional_experiment_2/comparator_predictions/unipocket.csv
```

Each run also writes `<out>.coverage.csv` and `<out>.skipped.json`. Proteins a competitor could
not handle are listed there and must be reported as competitor failures; the benchmark paper's
own PocketMiner comparison lost 60 of 222 proteins this way, and hiding that would repeat the
problem we are trying to fix.

Comparators worth running, in order of value: the fine-tuned ESM2-3B model from *Hidden in
protein sequences* (`github.com/skrhakv/cryptic-finetuning`), Seq2Pocket
(`github.com/skrhakv/seq2pocket`), the CryptoBench pLM-NN baseline (OSF `pz4a9`), then
PocketMiner and P2Rank. Running a competitor yourself always beats quoting it; where it cannot be
run, it belongs in Panel A only.

## Reading the results

| Experiment | Where | Decision |
|---|---|---|
| E1-A | `results/e1a_subset_*/summary.md` | same-protein comparison; no p-values by construction |
| E1-B | `results/e1b_matched_test/summary.md` | confirmatory: AP and BEDROC20, Holm within metrics and across competitors |
| E2 | `results/e2_esm3b/report_oof/` then `report_test/` | if the OOF gain is absent, stop and do not unlock |
| E3 | `results/e3_pocket_test/summary.md` + `sweep.csv` | the ordering must survive the clustering sweep |
| E4 | `results/e4_external_*_id020/report_test/` | EXT20 primary, EXT30 sensitivity |
| E5 | `results/e5_predicted*/report_test/` | sensitivity analysis of the frozen model, not a new unlock |

## Pocket-level metrics: definitions used here

Pocket numbers are not comparable across papers unless the clustering and the success criteria
match, so `scripts/30` puts every method through one identical pipeline and we never quote
another paper's pocket numbers beside ours. Fixed a priori: top 10% of residues by score,
single-linkage clustering at 8 Å on CB atoms, clusters of at least 3 residues, pocket score =
mean of the 5 highest member scores, DCC success at 4 Å, RRO = Jaccard overlap with success at
0.5 (`rro_recall` is the |∩|/|true| variant, reported alongside). `--sweep` repeats the headline
metrics over a grid of these settings.

## Test-set accounting

The CryptoBench test split has been scored twice already. Phase III adds **at most one more**,
and only for E2, which trains a new model. E1, E3 and E5 re-analyse predictions that already
exist and add no unlock. E4 is scored once, ever, and is the only clean confirmatory split.
`Q1_PROTOCOL.md` section 3 states this in the form it should appear in the manuscript.

## Status

These scripts were tested end to end on a synthetic mini-benchmark: pocket clustering and
DCC/RRO, the predicted-structure pipeline with perturbed and renumbered models, subset
recomputation, and the converter including its length-mismatch and low-identity refusals. They
have not been run on real CryptoBench data or against a real competitor's output. Treat the
first `e1a` and `e3` runs as the real test.
