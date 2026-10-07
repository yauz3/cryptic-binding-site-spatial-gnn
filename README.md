# Cryptic Binding-Site Prediction with a pLM-Anchored Spatial GNN

Official implementation and reproducibility repository for the study:

**Spatial Anchoring of a Protein Language Model Improves Cryptic Binding-Site Prediction on CryptoBench**

Repository:

https://github.com/yauz3/cryptic-binding-site-spatial-gnn

## Overview

This repository contains the code used to investigate whether local three-dimensional
protein structure can improve residue-level cryptic binding-site predictions obtained
from protein language models.

The main model is a **protein-language-model-anchored spatial graph neural network
(pLM-anchored spatial GNN)**.

Rather than replacing the sequence-model prediction, the structural branch learns a
residual correction using apo-protein geometry and explicit three-dimensional
message passing.

The repository includes the scripts used for:

- sequence-only pLM baselines;
- ESM-2 embedding extraction;
- residue-level geometry and flexibility features;
- multiscale Grid-Crypto feature processing;
- three-dimensional logit smoothing;
- pLM-anchored spatial GNN training;
- GNN ablation experiments;
- ranking-aware losses;
- constrained stacking;
- repeated-seed stability analysis;
- cross-fitted LoRA experiments;
- frozen ESM2-3B experiments;
- pocket-level evaluation;
- matched Seq2Pocket comparison;
- CryptoBank low-homology external validation;
- statistical testing and manuscript tables/figures.

## Scientific scope

The main analysis uses the official CryptoBench training and test partitions.

The study evaluates whether structural information improves a strong pLM anchor while
maintaining strict separation between training-fold model selection and prospective
test evaluation.

The primary result is that residue-level local geometry and three-dimensional
neighbourhood coherence provide useful information beyond the sequence-only pLM
prediction.

The study also evaluates high-dimensional Grid-Crypto pocket descriptors. These
features provide little additional information once direct residue geometry and
spatial message passing are available.

For exact metrics, confidence intervals, confirmatory families, statistical tests,
and limitations, please refer to the associated manuscript.

## Repository structure

    cryptogrid/
        Core reusable Python modules.

    scripts/
        Main training, analysis, evaluation, reporting, and external-validation scripts.

    scripts/dataset_analysis/
        Dataset characterization and exploratory analyses.

    scripts/external/
        Wrappers for selected external comparison methods.

    additional_experiment/
        Phase-II experiments involving adapted ESM-2 representations.

    additional_experiment_2/
        Phase-III experiments, larger-backbone analyses, and external validation.

    E1B_Seq2Pocket_share/
        Scripts and compact outputs for the matched Seq2Pocket evaluation.

    paper_results/
        Curated manuscript-level outputs, manifests, and checksums.

    tests/
        Lightweight synthetic-data tests.

    README_PAPER.md
        Additional manuscript-oriented reproducibility notes.

    EXPERIMENT_MATRIX.md
        Overview of experiment families and their relationships.

    SCIENTIFIC_GUARDRAILS.md
        Experimental and leakage-control rules used by the pipeline.

    config.example.yaml
        Portable configuration template.

    requirements.txt
        Python package requirements.

    environment.yml
        Conda environment definition.

    LICENSE
        Academic research license.

    COPYRIGHT.md
        Copyright and academic-use statement.

    CITATION.cff
        Machine-readable citation metadata.

## Main scripts

Important scripts include:

    scripts/00_preflight_paper.py
    scripts/01_prepare_grid_residue_features.py
    scripts/02_extract_esm_embeddings.py
    scripts/10_build_geom_cache.py
    scripts/11_baseline_multiseed.py
    scripts/12_spatial_smoothing.py
    scripts/13_pocket_consensus.py
    scripts/14_gbm_residual_boost.py
    scripts/14b_gbm_residual_weighted_earlystop.py
    scripts/15_anchored_spatial_gnn.py
    scripts/16_rank_loss.py
    scripts/17_aux_distance_multitask.py
    scripts/18_constrained_stacker.py
    scripts/19_stats_and_freeze.py
    scripts/20_early_enrichment_report.py
    scripts/20_replicate_runs.py
    scripts/21_replicate_analysis.py
    scripts/22_paper_report.py
    scripts/23_finetune_plm.py
    scripts/24_build_external_benchmark.py
    scripts/25_external_report.py
    scripts/26_matched_external_predictors.py
    scripts/30_pocket_level_eval.py
    scripts/31_predicted_structure_benchmark.py
    scripts/32_subset_report.py

## Installation

Clone the repository:

    git clone https://github.com/yauz3/cryptic-binding-site-spatial-gnn.git
    cd cryptic-binding-site-spatial-gnn

Create the Conda environment:

    conda env create -f environment.yml
    conda activate cryptic-spatial-gnn

Alternatively, install the Python dependencies into an existing environment:

    pip install -r requirements.txt

A CUDA-capable GPU is recommended for ESM embedding extraction and the larger-model
experiments.

PyTorch/CUDA versions may need to be adapted to the CUDA driver installed on the
local system.

## Configuration

Copy the portable configuration template:

    cp config.example.yaml config.yaml

Edit `config.yaml` and provide the paths to the required local resources.

The machine-specific `config.yaml` file is intentionally excluded from Git.

## Required external resources

Large datasets, pretrained model weights, and third-party repositories are not
redistributed in this repository.

### CryptoBench

The primary experiments require the official CryptoBench resources, including:

- the training/test definition;
- fold assignments;
- residue-level labels;
- apo protein structures.

Users should obtain these resources from the original CryptoBench distribution and
comply with its original terms.

### ESM-2

The experiments use pretrained ESM-2 models, including:

    facebook/esm2_t33_650M_UR50D
    facebook/esm2_t36_3B_UR50D

Model weights are obtained from their original distribution and are not included in
this repository.

### Grid-Crypto-derived representations

The Grid-related experiments require multiscale pocket/grid features generated by the
corresponding Grid-Crypto pipeline.

These descriptors are conceptually separate from the direct residue-level geometry
and flexibility features used by the pLM-anchored spatial GNN.

### Seq2Pocket

The matched-comparator experiment uses the publicly released Seq2Pocket model.

The original model checkpoint is not redistributed here.

Relevant scripts and compact results are located under:

    E1B_Seq2Pocket_share/

### CryptoBank

The external validation analysis uses an independently constructed low-homology
CryptoBank cohort.

Raw CryptoBank files are not redistributed.

The repository contains the scripts used for cohort preparation, homology filtering,
prediction, and evaluation.

## Typical workflow

### 1. Configure the project

    cp config.example.yaml config.yaml

Edit the paths in `config.yaml`.

### 2. Run the preflight checks

    python scripts/00_preflight_paper.py

### 3. Prepare representations

Depending on the experiment:

    python scripts/01_prepare_grid_residue_features.py
    python scripts/02_extract_esm_embeddings.py
    python scripts/10_build_geom_cache.py

### 4. Run the primary manuscript pipeline

    bash run_paper.sh

For broader orchestration:

    bash run_everything.sh

Individual experiment families can also be run separately:

    bash run_experiments.sh
    bash run_ablation.sh
    bash run_comparative.sh
    bash run_dataset_analysis.sh

### 5. Phase II

Phase II evaluates the architecture using adapted ESM-2 representations.

    bash additional_experiment/run_phase2.sh

The corresponding prospectively specified analysis protocol is:

    additional_experiment/PHASE2_PROTOCOL.md

### 6. Phase III

Phase III contains the larger-backbone and additional confirmatory experiments.

    bash additional_experiment_2/run_q1.sh

The corresponding protocol is:

    additional_experiment_2/Q1_PROTOCOL.md

### 7. Matched Seq2Pocket evaluation

See:

    E1B_Seq2Pocket_share/

The Seq2Pocket checkpoint must be downloaded from its original source.

### 8. External validation

External-validation scripts are located under:

    scripts/
    additional_experiment_2/scripts/

Raw external data and downloaded structures are intentionally excluded from the
repository.

## External comparator paths

External wrappers should be supplied explicit local paths.

For example:

    export PROJECT_PDB=/path/to/cryptobench/protein_structures
    export FOLDS=/path/to/cryptobench/folds.json

Then run the desired wrapper under:

    scripts/external/

## Reproducibility and leakage control

The repository implements phase-specific separation between model development and
test evaluation.

Reproducibility material includes:

- training-fold out-of-fold model selection;
- phase-specific analysis protocols;
- frozen model-selection records;
- SHA-256 checksums;
- test-unlock logging;
- repeated-seed evaluation;
- protein-level paired statistics;
- external validation scripts.

Curated manuscript outputs are provided under:

    paper_results/

Important audit files include:

    paper_results/MANIFEST.md
    paper_results/SHA256SUMS.txt

Large intermediate caches, full model checkpoints, embeddings, and large prediction
matrices are intentionally excluded from Git.

## Metrics

The prediction unit is the residue, while the main statistical unit is the protein.

Reported metrics include:

- average precision;
- ROC-AUC;
- EF@1%;
- EF@3%;
- EF@5%;
- BEDROC;
- MCC;
- F1;
- pocket-level distance and overlap metrics.

The manuscript defines the exact confirmatory metric families and statistical
procedures.

## Academically free

The original code developed for this repository is **academically free**.

It may be used, studied, modified, and redistributed without a licensing fee for
non-commercial academic, educational, and research purposes, subject to the
conditions specified in the `LICENSE` file.

Appropriate scholarly attribution is required.

Commercial use requires prior written permission from the copyright holder.

Third-party software, datasets, pretrained models, model checkpoints, protein
structures, and external resources are not relicensed by this repository and remain
subject to their original licenses and terms.

Because commercial use is restricted, this repository uses a project-specific
academic research license rather than an OSI-approved open-source license.

## Citation

If you use this repository in academic research, please cite the associated manuscript
and this repository.

Machine-readable citation metadata are available in:

    CITATION.cff

## Funding

This research was supported by the Scientific and Technological Research Council of
Türkiye (TÜBİTAK) through the 3501 Career Development Program, Project No. 126S368.

## Contact

Sadettin Yavuz Uğurlu, PhD

Department of Materials Science and Engineering  
Akdeniz University  
Antalya, Türkiye

Repository:

https://github.com/yauz3/cryptic-binding-site-spatial-gnn

## Disclaimer

This software is provided for research purposes without warranty.

Predictions generated by the software are computational research outputs and should
not be interpreted as clinical, diagnostic, or therapeutic recommendations.
