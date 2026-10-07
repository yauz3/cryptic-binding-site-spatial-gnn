#!/usr/bin/env python3

from pathlib import Path
import gzip
import re
import shutil
import time
import urllib.request

import numpy as np
import pandas as pd

from Bio.Align import PairwiseAligner
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Data.PDBData import protein_letters_3to1_extended


ROOT = Path(
    "/path/to/cryptic_binding_site/"
    "cryptobench_gridfusion_suite_github_2/"
    "cryptobench_gridfusion_suite"
)

RAW = ROOT / "additional_experiment_2/external_data/cryptobank_raw"
E4  = ROOT / "additional_experiment_2/external_data/cryptobank_e4"

SELECTED = RAW / "E4_step3_selected_experimental_apo.csv"

CIF_DIR = E4 / "raw_cif"
CIF_DIR.mkdir(parents=True, exist_ok=True)

OUT_LONG = E4 / "E4_step4_residue_labels_long.csv"
OUT_QC   = E4 / "E4_step4_mapping_qc.csv"
OUT_MAN  = E4 / "E4_step4_structure_manifest.csv"

MIN_STRUCTURE_OFFICIAL_COVERAGE = 0.80
MIN_ATOM_SEQUENCE_AGREEMENT = 0.95


aligner = PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 2.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -10.0
aligner.extend_gap_score = -0.5


def canonical_seq(x):
    """
    RCSB entity sequences may contain modified residues as e.g. (OCS).
    Treat each parenthesised modified residue as one X.
    """
    if pd.isna(x):
        return ""

    s = "".join(str(x).split()).upper()
    s = re.sub(r"\([^)]*\)", "X", s)

    return s


def as_list(x):
    if isinstance(x, list):
        return x
    return [x]


def download_cif(pdb_id):
    pdb = pdb_id.upper()
    gz = CIF_DIR / f"{pdb}.cif.gz"
    cif = CIF_DIR / f"{pdb}.cif"

    if cif.exists() and cif.stat().st_size > 1000:
        return cif

    url = f"https://files.rcsb.org/download/{pdb}.cif.gz"

    for attempt in range(1, 5):
        try:
            print(f"[download] {pdb} attempt={attempt}")

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "CryptoBank-E4/1.0"}
            )

            with urllib.request.urlopen(req, timeout=90) as r:
                with open(gz, "wb") as f:
                    shutil.copyfileobj(r, f)

            with gzip.open(gz, "rb") as src:
                with open(cif, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            if cif.stat().st_size < 1000:
                raise RuntimeError("Downloaded CIF too small")

            return cif

        except Exception as e:
            print(f"[WARN] {pdb}: {e}")
            time.sleep(3 * attempt)

    return None


def build_apo_to_official_map(official_seq, apo_seq, labels):
    """
    Return:
      apo_position (0-based) -> official_position (0-based)
    """

    official_seq = canonical_seq(official_seq)
    apo_seq = canonical_seq(apo_seq)
    labels = str(labels).strip()

    if len(official_seq) != len(labels):
        raise ValueError(
            f"official sequence/label mismatch: "
            f"{len(official_seq)} vs {len(labels)}"
        )

    aln = aligner.align(official_seq, apo_seq)[0]

    out = {}

    blocks_a, blocks_b = aln.aligned

    for (o0, o1), (a0, a1) in zip(blocks_a, blocks_b):

        lo = int(o1 - o0)
        la = int(a1 - a0)

        if lo != la:
            raise RuntimeError(
                f"Unexpected alignment block sizes {lo} vs {la}"
            )

        for k in range(lo):
            oi = int(o0) + k
            ai = int(a0) + k
            out[ai] = oi

    return out


def extract_observed_polymer_residues(cif_path, auth_chain):
    """
    Parse ATOM records directly from mmCIF.

    label_seq_id provides the polymer sequence position.
    auth_seq_id + auth_asym_id preserve the residue namespace
    actually used in the structure.
    """

    d = MMCIF2Dict(str(cif_path))

    group = as_list(d["_atom_site.group_PDB"])
    auth_chain_col = as_list(d["_atom_site.auth_asym_id"])
    auth_seq = as_list(d["_atom_site.auth_seq_id"])
    label_seq = as_list(d["_atom_site.label_seq_id"])
    comp = as_list(d["_atom_site.label_comp_id"])

    icode = as_list(
        d.get(
            "_atom_site.pdbx_PDB_ins_code",
            ["?"] * len(group)
        )
    )

    n = len(group)

    cols = [
        auth_chain_col,
        auth_seq,
        label_seq,
        comp,
        icode
    ]

    if any(len(x) != n for x in cols):
        raise RuntimeError("mmCIF atom_site column-length mismatch")

    residues = []
    seen = set()

    for i in range(n):

        # Polymer residues only.
        if str(group[i]).upper() != "ATOM":
            continue

        chain = str(auth_chain_col[i]).strip()

        if chain != str(auth_chain).strip():
            continue

        ls = str(label_seq[i]).strip()

        if ls in {"", ".", "?"}:
            continue

        try:
            label_pos = int(float(ls))
        except Exception:
            continue

        aseq = str(auth_seq[i]).strip()

        ins = str(icode[i]).strip()

        if ins in {"?", "."}:
            ins = ""

        resname = str(comp[i]).strip().upper()

        key_tuple = (
            chain,
            aseq,
            ins,
            label_pos
        )

        if key_tuple in seen:
            continue

        seen.add(key_tuple)

        residue_key = f"{chain}_{aseq}{ins}"

        one = protein_letters_3to1_extended.get(
            resname,
            "X"
        )

        residues.append({
            "auth_chain": chain,
            "auth_seq_id": aseq,
            "insertion_code": ins,
            "label_seq_id": label_pos,
            "resname": resname,
            "aa_observed": one,
            "residue_key": residue_key,
        })

    residues = sorted(
        residues,
        key=lambda x: x["label_seq_id"]
    )

    return residues


selected = pd.read_csv(SELECTED)

print("=" * 100)
print("CRYPTOBANK E4 STEP 4")
print("=" * 100)
print("selected proteins:", len(selected))
print(
    "unique apo structures:",
    selected["structureId_apo"].nunique()
)

all_rows = []
qc_rows = []
manifest_rows = []

download_fail = 0
parse_fail = 0

for idx, row in selected.iterrows():

    external_id = str(row["structureId"]).upper()
    apo_id = str(row["structureId_apo"]).upper()

    if "_" not in apo_id:
        print(f"[FAIL] invalid structureId_apo: {apo_id}")
        parse_fail += 1
        continue

    pdb_id, auth_chain = apo_id.split("_", 1)

    print(
        f"\n[{idx+1}/{len(selected)}] "
        f"{external_id} -> {apo_id}"
    )

    cif = download_cif(pdb_id)

    if cif is None:
        download_fail += 1

        qc_rows.append({
            "external_id": external_id,
            "structureId_apo": apo_id,
            "status": "download_failed",
        })

        continue

    official_seq = canonical_seq(
        row["official_test_sequence"]
    )

    apo_seq = canonical_seq(
        row["sequence_apo"]
    )

    labels = str(
        row["sequence_crypticity_combined"]
    ).strip()

    try:
        apo_to_official = build_apo_to_official_map(
            official_seq,
            apo_seq,
            labels
        )

        residues = extract_observed_polymer_residues(
            cif,
            auth_chain
        )

    except Exception as e:
        print("[FAIL]", repr(e))
        parse_fail += 1

        qc_rows.append({
            "external_id": external_id,
            "structureId_apo": apo_id,
            "status": "parse_or_mapping_failed",
            "error": repr(e),
        })

        continue

    if not residues:
        print("[FAIL] no ATOM polymer residues")
        parse_fail += 1

        qc_rows.append({
            "external_id": external_id,
            "structureId_apo": apo_id,
            "status": "no_observed_residues",
        })

        continue

    represented_official = set()
    agreement_good = 0
    agreement_total = 0

    n_labeled_observed = 0
    observed_positive = 0

    for r in residues:

        apo_pos = r["label_seq_id"] - 1

        apo_aa = (
            apo_seq[apo_pos]
            if 0 <= apo_pos < len(apo_seq)
            else None
        )

        if apo_aa is not None:
            agreement_total += 1

            if (
                r["aa_observed"] == apo_aa
                or r["aa_observed"] == "X"
                or apo_aa == "X"
            ):
                agreement_good += 1

        oi = apo_to_official.get(apo_pos)

        if oi is None:
            y = np.nan
            official_position = np.nan
            official_aa = None

        else:
            y = int(labels[oi])
            official_position = oi + 1
            official_aa = official_seq[oi]

            represented_official.add(oi)
            n_labeled_observed += 1
            observed_positive += y

        all_rows.append({
            "external_id": external_id,
            "source_holo_structureId": external_id,
            "structureId_apo": apo_id,
            "pdb_id": pdb_id,
            "auth_chain": auth_chain,
            "residue_key": r["residue_key"],
            "auth_seq_id": r["auth_seq_id"],
            "insertion_code": r["insertion_code"],
            "label_seq_id": r["label_seq_id"],
            "resname": r["resname"],
            "aa_observed": r["aa_observed"],
            "apo_sequence_aa": apo_aa,
            "official_position": official_position,
            "official_sequence_aa": official_aa,
            "label": y,
        })

    structure_official_coverage = (
        len(represented_official) / len(official_seq)
        if official_seq else 0.0
    )

    atom_sequence_agreement = (
        agreement_good / agreement_total
        if agreement_total else 0.0
    )

    n_official_positive = labels.count("1")

    represented_positive = sum(
        int(labels[i])
        for i in represented_official
    )

    positive_observable_fraction = (
        represented_positive / n_official_positive
        if n_official_positive else np.nan
    )

    mapping_qc_pass = (
        structure_official_coverage
        >= MIN_STRUCTURE_OFFICIAL_COVERAGE
        and
        atom_sequence_agreement
        >= MIN_ATOM_SEQUENCE_AGREEMENT
    )

    status = (
        "ok"
        if mapping_qc_pass
        else "mapping_qc_failed"
    )

    print(
        f"  observed={len(residues)} "
        f"official_cov={structure_official_coverage:.3f} "
        f"atom_seq_agree={atom_sequence_agreement:.3f} "
        f"positives={observed_positive}/{n_official_positive} "
        f"QC={mapping_qc_pass}"
    )

    qc_rows.append({
        "external_id": external_id,
        "structureId_apo": apo_id,
        "pdb_id": pdb_id,
        "auth_chain": auth_chain,
        "status": status,
        "n_observed_residues": len(residues),
        "n_labeled_observed": n_labeled_observed,
        "official_sequence_length": len(official_seq),
        "apo_sequence_length": len(apo_seq),
        "structure_official_coverage":
            structure_official_coverage,
        "atom_sequence_agreement":
            atom_sequence_agreement,
        "n_official_positive":
            n_official_positive,
        "n_observed_positive":
            observed_positive,
        "positive_observable_fraction":
            positive_observable_fraction,
        "mapping_qc_pass":
            mapping_qc_pass,
        "identity_step3":
            row["identity_aligned"],
        "official_coverage_step3":
            row["official_coverage"],
        "resolution_apo":
            row["resolution_apo"],
    })

    manifest_rows.append({
        "external_id": external_id,
        "structureId_apo": apo_id,
        "pdb_id": pdb_id,
        "auth_chain": auth_chain,
        "cif_path": str(cif),
        "mapping_qc_pass": mapping_qc_pass,
    })


long_df = pd.DataFrame(all_rows)
qc = pd.DataFrame(qc_rows)
manifest = pd.DataFrame(manifest_rows)

long_df.to_csv(OUT_LONG, index=False)
qc.to_csv(OUT_QC, index=False)
manifest.to_csv(OUT_MAN, index=False)


print("\n" + "=" * 100)
print("STEP 4 SUMMARY")
print("=" * 100)

print("selected proteins:", len(selected))
print("download failures:", download_fail)
print("parse/mapping failures:", parse_fail)

if len(qc):

    print("\nstatus:")
    print(qc["status"].value_counts(dropna=False))

    ok = qc[
        qc.get(
            "mapping_qc_pass",
            pd.Series(False, index=qc.index)
        ).fillna(False)
    ].copy()

    print("\nQC-passing proteins:", len(ok))

    if len(ok):

        print("\nstructure -> official coverage:")
        print(
            ok["structure_official_coverage"]
            .describe()
        )

        print("\nATOM sequence agreement:")
        print(
            ok["atom_sequence_agreement"]
            .describe()
        )

        print("\npositive observable fraction "
              "(reported, NOT used for selection):")

        print(
            ok["positive_observable_fraction"]
            .describe()
        )

        zero_pos = (
            ok["n_observed_positive"] == 0
        ).sum()

        print(
            "\nQC-passing proteins with "
            "zero observable positives:",
            int(zero_pos)
        )

print("\nLong label table rows:", len(long_df))

if len(long_df):

    labeled = long_df[
        long_df["label"].notna()
    ]

    print(
        "labeled observed residues:",
        len(labeled)
    )

    print(
        "positive labeled residues:",
        int(labeled["label"].sum())
    )

print("\nSaved:")
print(OUT_LONG)
print(OUT_QC)
print(OUT_MAN)

print("\nDONE")
