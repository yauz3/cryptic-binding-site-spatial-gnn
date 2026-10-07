# additional_experiment — Phase II

Everything needed to answer the two questions that stand between the Phase-I manuscript and
a Q1 submission:

1. **Does the spatial residual survive fine-tuning of the language model?**
   Phase I measured it on top of a *frozen* ESM-2 head. The published state of the art on
   CryptoBench fine-tunes a larger ESM-2, so a reviewer will ask whether the gain is about
   cryptic sites or about a weak baseline.
2. **Does it hold on proteins that were never in CryptoBench?**
   The CryptoBench test split was inspected during the exploratory phase, so it is a
   frozen-after-OOF held-out set, not a pristine one.

Plus the comparison the Phase-I manuscript explicitly deferred:

3. **Head-to-head against published predictors on identical residues**, with every
   competitor failure reported instead of silently dropped.

Read `PHASE2_PROTOCOL.md` before running anything. It fixes the hypotheses, the decision
rules and the test-set accounting in advance.

## Install

Put this folder inside the Phase-I suite and run:

```bash
cd cryptobench_gridfusion_suite
bash additional_experiment/install.sh
pip install transformers            # the only new dependency
bash additional_experiment/run_phase2.sh check
```

`install.sh` copies six new files in and replaces three Phase-I files with versions that
gained a backward-compatible option. It backs up whatever it overwrites into
`additional_experiment/backup_<timestamp>/`, and it is safe to re-run.

| File | Change |
|---|---|
| `cryptogrid/exp_common.py` | `make_ctx(..., require_esm=False)` for scripts that *create* an embedding cache |
| `scripts/10_build_geom_cache.py` | no longer demands an embedding cache (geometry does not need one) |
| `scripts/19_stats_and_freeze.py` | new `--baseline`, so the Phase-II freeze can be judged against the fine-tuned anchor |

Everything else in the Phase-I suite is untouched, and every Phase-II output goes to
`additional_experiment/results/`. The Phase-I `results/`, its freeze and its unlock log are
only ever read.

## Run

```bash
cd cryptobench_gridfusion_suite        # run from the suite root

bash additional_experiment/run_phase2.sh check
bash additional_experiment/run_phase2.sh finetune       # II-A, II-B
#   -> read results/phase2_cryptobench/report_oof/summary.md and decide H1
bash additional_experiment/run_phase2.sh unlock         # II-C, only if H1 held
bash additional_experiment/run_phase2.sh external       # II-D, frozen pLM
bash additional_experiment/run_phase2.sh external-ft    # II-D, fine-tuned encoder
bash additional_experiment/run_phase2.sh matched        # II-E
```

Useful environment variables:

```bash
FT_MODEL=facebook/esm2_t33_650M_UR50D   # or facebook/esm2_t36_3B_UR50D
FT_MODE=lora                            # or top6 / full
FT_EPOCHS=3
FT_EXTRA="--grad-checkpointing --accum 4"   # for the 3B model or a small GPU
SEEDS="0 1 2"
EXT_NAME=pocketminer
EXT_STRUCTURES=additional_experiment/external_data/pocketminer_structures
EXT_LABELS=additional_experiment/external_data/pocketminer_labels.json
```

`finetune` trains five encoders (four folds + one on all 885). With LoRA on ESM2-650M that
is the dominant cost of Phase II; I have not measured it on your hardware. Stages are
resumable: re-running skips finished folds.

## What goes where

```
additional_experiment/
  results/
    config_ft.yaml                       generated Phase-II config
    cache_esm_ft/                        fine-tuned embeddings, cross-fitted by fold
    checkpoints/ft_{train-0..3,full}.pt  LoRA adapters + head
    phase2_cryptobench/
      exp/<model>/{oof,test}_predictions.csv
      exp/_stats/{frozen_selection.json,test_unlock_log.jsonl,final_test_unlock.log}
      report_oof/   report_test/         tables (.csv + .tex), figures (.png + .pdf), summary.md
    external/<name>/
      cache_struct/ cache_geom/ cache_esm/ [cache_esm_ft/]
      homology_report.csv  report.json
      results/ results_ft/                same report layout as above
    matched_cryptobench_test/            head-to-head tables + coverage per protein
    matched_external_<name>/
  external_data/                         your external structures + labels go here
  external_predictions/                  competitor CSVs for the CryptoBench test split
  external_predictions_<name>/           competitor CSVs for the external split
```

## External benchmark inputs

`external_data/<name>_structures/` holds one `.cif`/`.pdb` per protein named `<pdb_id>.<ext>`.
`external_data/<name>_labels.json` uses the CryptoBench `dataset.json` schema:

```json
{"1abc": [{"apo_chain": "A", "apo_pocket_selection": ["A_41", "A_42", "A_118"]}]}
```

Residue keys are `<auth_asym_id>_<auth_seq_id><insertion code>`. Candidate sources: the
PocketMiner test set and the CryptoSite set. `scripts/24` removes anything above 30%
identity to any CryptoBench protein (MMseqs2 if installed, Biopython otherwise) and logs
every exclusion to `homology_report.csv`. Expect it to remove a lot; that is the point.

Grid-Crypto descriptors do not exist for external proteins, so `X_struct` is written as
zeros and the Grid-dependent models (scripts 13, 14) are not applicable there. The sequence
anchor and the anchored GNN are, and those are the models the manuscript recommends.

## Competitor predictions

One CSV per competitor, columns `pdb_id, residue_key, score` and optionally `prediction`
(binary) and `method` (display name). Converters:

```bash
python scripts/external/convert_predictions.py --format p2rank \
    --input runs/p2rank_out --config config.yaml --method P2Rank \
    --out additional_experiment/external_predictions/p2rank.csv

python scripts/external/convert_predictions.py --format pocketminer \
    --input runs/pocketminer_npy --structures external_data/cryptobench_cifs \
    --config config.yaml --method PocketMiner \
    --out additional_experiment/external_predictions/pocketminer.csv
```

The converter checks lengths and residue keys against the struct cache and fails loudly
rather than writing a misaligned file; a one-residue offset would quietly invalidate every
head-to-head number. Structures a competitor cannot process are written to a `.skipped.json`
beside the output and must be reported as competitor failures.

## Reading the results

* **H1** — `results/phase2_cryptobench/report_oof/summary.md`. The anchored GNN must beat
  `base_plm_ft` on both BEDROC20 and EF@1%. If it does not, stop: Phase II is a negative
  result and the test split stays closed.
* **H2** — `results/phase2_cryptobench/report_test/`, after exactly one unlock.
* **H3** — `results/external/<name>/results*/report_test/`. This is the only split in the
  project that is opened once, ever, so it carries the confirmatory claim.
* **H4** — `results/matched_*/table_matched_paired.*` plus `coverage_per_protein.csv`.

Statistics are the Phase-I ones throughout: protein as the unit of inference, paired
sign-flip permutation test on the mean per-protein difference, Holm within
`{EF@1%, BEDROC20}`, Cohen's d_z with bootstrap intervals.

## Status

These scripts were tested end to end on a synthetic mini-benchmark with a tiny locally built
ESM model, on CPU. They have not been run on real CryptoBench data or with a real ESM-2
checkpoint, so treat the first `finetune` run as the real test and check
`report_oof/summary.md` before committing GPU time to the rest.
