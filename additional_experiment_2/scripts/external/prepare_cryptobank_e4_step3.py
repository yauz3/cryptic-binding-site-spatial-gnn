#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner

ROOT = Path(
    "/path/to/cryptic_binding_site/"
    "cryptobench_gridfusion_suite_github_2/"
    "cryptobench_gridfusion_suite"
)

RAW = ROOT / "additional_experiment_2/external_data/cryptobank_raw"

CAND = RAW / "E4_cryptobank_experimental_apo_candidates.csv"
LABEL = RAW / "E4_cryptobank_official_test_labels.csv"

OUT_ALL = RAW / "E4_step3_all_candidate_alignments.csv"
OUT_SEL = RAW / "E4_step3_selected_experimental_apo.csv"
OUT_EXC = RAW / "E4_step3_excluded_no_strict_apo.csv"


PRIMARY_MIN_IDENTITY = 0.95
PRIMARY_MIN_OFFICIAL_COVERAGE = 0.90


aligner = PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 2.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -10.0
aligner.extend_gap_score = -0.5


def clean_seq(x):
    if pd.isna(x):
        return ""
    return "".join(str(x).split()).upper()


def alignment_metrics(seq_official, seq_apo):
    a = clean_seq(seq_official)
    b = clean_seq(seq_apo)

    if not a or not b:
        return {
            "aligned_pairs": 0,
            "matches": 0,
            "identity_aligned": np.nan,
            "official_coverage": 0.0,
            "apo_coverage": 0.0,
            "exact_seq_equal_recalc": False,
        }

    if a == b:
        return {
            "aligned_pairs": len(a),
            "matches": len(a),
            "identity_aligned": 1.0,
            "official_coverage": 1.0,
            "apo_coverage": 1.0,
            "exact_seq_equal_recalc": True,
        }

    aln = aligner.align(a, b)[0]

    blocks_a, blocks_b = aln.aligned

    paired = 0
    matches = 0

    for (a0, a1), (b0, b1) in zip(blocks_a, blocks_b):
        la = int(a1 - a0)
        lb = int(b1 - b0)

        if la != lb:
            raise RuntimeError(
                f"Unexpected aligned block lengths: {la} vs {lb}"
            )

        paired += la

        sa = a[int(a0):int(a1)]
        sb = b[int(b0):int(b1)]

        matches += sum(x == y for x, y in zip(sa, sb))

    return {
        "aligned_pairs": paired,
        "matches": matches,
        "identity_aligned": matches / paired if paired else np.nan,
        "official_coverage": paired / len(a),
        "apo_coverage": paired / len(b),
        "exact_seq_equal_recalc": False,
    }


print("Loading candidate table...")
cand = pd.read_csv(CAND)

print("candidate rows:", len(cand))
print("official proteins:", cand["structureId"].nunique())

# Ensure official sequence is available
if "official_test_sequence" not in cand.columns:
    lab = pd.read_csv(LABEL)[["structureId", "sequence"]].rename(
        columns={"sequence": "official_test_sequence"}
    )
    cand = cand.merge(
        lab,
        on="structureId",
        how="left",
        validate="many_to_one"
    )

# Defensive: experimental only
if "alphafold" in cand.columns:
    cand = cand[
        cand["alphafold"].astype(str).str.lower().eq("false")
    ].copy()

cand = cand.drop_duplicates(
    subset=["structureId", "structureId_apo"]
).reset_index(drop=True)

print("experimental unique pairs:", len(cand))

metrics = []

for i, row in cand.iterrows():

    if i % 100 == 0:
        print(f"aligning {i}/{len(cand)}")

    m = alignment_metrics(
        row["official_test_sequence"],
        row["sequence_apo"]
    )

    metrics.append(m)

met = pd.DataFrame(metrics)
cand = pd.concat([cand.reset_index(drop=True), met], axis=1)

cand["primary_mapping_eligible"] = (
    (cand["identity_aligned"] >= PRIMARY_MIN_IDENTITY)
    &
    (cand["official_coverage"] >= PRIMARY_MIN_OFFICIAL_COVERAGE)
)

# Resolution: missing values rank last
cand["_resolution_rank"] = pd.to_numeric(
    cand.get("resolution_apo"),
    errors="coerce"
).fillna(np.inf)

# Exact equality from recalculation has priority.
# Then identity, coverage, apo coverage, crystallographic resolution,
# finally deterministic lexical tie-break.
cand = cand.sort_values(
    by=[
        "structureId",
        "primary_mapping_eligible",
        "exact_seq_equal_recalc",
        "identity_aligned",
        "official_coverage",
        "apo_coverage",
        "_resolution_rank",
        "structureId_apo",
    ],
    ascending=[
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ],
).reset_index(drop=True)

cand.to_csv(OUT_ALL, index=False)

eligible = cand[cand["primary_mapping_eligible"]].copy()

selected = (
    eligible
    .groupby("structureId", as_index=False, group_keys=False)
    .head(1)
    .copy()
)

selected["selection_rule"] = (
    "experimental_apo;"
    "identity>=0.95;"
    "official_coverage>=0.90;"
    "rank=exact,identity,coverage,apo_coverage,resolution,structureId"
)

selected = selected.drop(columns=["_resolution_rank"], errors="ignore")
selected.to_csv(OUT_SEL, index=False)

all_proteins = set(cand["structureId"])
sel_proteins = set(selected["structureId"])

excluded_ids = sorted(all_proteins - sel_proteins)

excluded = pd.DataFrame({"structureId": excluded_ids})
excluded.to_csv(OUT_EXC, index=False)


print("\n" + "=" * 100)
print("STEP 3 SUMMARY")
print("=" * 100)

print("experimental proteins entering step:", len(all_proteins))
print("experimental candidate pairs:", len(cand))

print(
    "eligible candidate pairs:",
    int(cand["primary_mapping_eligible"].sum())
)

print(
    "proteins with >=1 strict eligible apo:",
    selected["structureId"].nunique()
)

print(
    "proteins excluded for mapping quality:",
    len(excluded_ids)
)

print("\nSelected alignment identity:")
print(selected["identity_aligned"].describe())

print("\nSelected official-sequence coverage:")
print(selected["official_coverage"].describe())

print("\nSelected apo coverage:")
print(selected["apo_coverage"].describe())

print(
    "\nSelected exact sequence matches:",
    int(selected["exact_seq_equal_recalc"].sum()),
    "/",
    len(selected)
)

if "resolution_apo" in selected.columns:
    print("\nSelected resolution:")
    print(
        pd.to_numeric(
            selected["resolution_apo"],
            errors="coerce"
        ).describe()
    )

print("\nTop selected examples:")

show = [
    c for c in [
        "structureId",
        "structureId_apo",
        "identity_aligned",
        "official_coverage",
        "apo_coverage",
        "exact_seq_equal_recalc",
        "resolution_apo",
        "n_positive",
    ]
    if c in selected.columns
]

print(selected[show].head(25).to_string(index=False))

print("\nSaved:")
print(OUT_ALL)
print(OUT_SEL)
print(OUT_EXC)

print("\nDONE")
