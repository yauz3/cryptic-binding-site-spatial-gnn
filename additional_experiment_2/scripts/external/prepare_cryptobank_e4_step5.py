#!/usr/bin/env python3

from pathlib import Path
import json
import shutil

import pandas as pd
from Bio.PDB import MMCIFParser, MMCIFIO, Select


ROOT = Path(
    "/path/to/cryptic_binding_site/"
    "cryptobench_gridfusion_suite_github_2/"
    "cryptobench_gridfusion_suite"
)

E4 = ROOT / "additional_experiment_2/external_data/cryptobank_e4"

QC = E4 / "E4_step4_mapping_qc.csv"
LONG = E4 / "E4_step4_residue_labels_long.csv"
RAW_CIF = E4 / "raw_cif"

OUT_STRUCT = E4 / "structures_chain_only"
OUT_LABELS = E4 / "cryptobank_e4_labels.json"
OUT_MANIFEST = E4 / "E4_step5_final_cohort.csv"

OUT_STRUCT.mkdir(parents=True, exist_ok=True)


class ChainOnly(Select):
    def __init__(self, chain_id):
        self.chain_id = str(chain_id)

    def accept_chain(self, chain):
        return int(str(chain.id) == self.chain_id)


THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "MSE":"M","SEC":"U","PYL":"O",
}


def parse_output(path):
    """
    Builder-equivalent observed-residue parser, used as final QC.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())

    out = []

    for chain in model:
        for res in chain:

            het, seq, icode = res.id

            if het.strip() not in {"", "H_MSE"}:
                continue

            name = res.get_resname().strip().upper()

            if name not in THREE_TO_ONE:
                continue

            ic = "" if icode in {" ", "", "?"} else icode.strip()

            key = f"{chain.id.strip()}_{seq}{ic}"

            out.append((key, THREE_TO_ONE[name]))

    return out


qc = pd.read_csv(QC)
long = pd.read_csv(LONG)

# String/bool robust conversion
pass_mask = qc["mapping_qc_pass"].astype(str).str.lower().eq("true")

final = qc[
    pass_mask &
    (pd.to_numeric(qc["n_observed_positive"], errors="coerce") > 0)
].copy()

print("=" * 100)
print("E4 STEP 5 — FINAL STRUCTURE/LABEL PREPARATION")
print("=" * 100)

print("Step-4 QC pass:", int(pass_mask.sum()))
print("ranking-evaluable:", len(final))
print("excluded zero-observable-positive:",
      int(pass_mask.sum()) - len(final))

assert len(final) == 283, (
    f"Expected 283 ranking-evaluable proteins, got {len(final)}"
)

labels = {}
manifest = []

parser = MMCIFParser(QUIET=True)

errors = []

for i, row in final.reset_index(drop=True).iterrows():

    external_id = str(row["external_id"]).upper()
    apo_id = str(row["structureId_apo"]).upper()
    pdb_id = str(row["pdb_id"]).upper()
    chain = str(row["auth_chain"])

    src = RAW_CIF / f"{pdb_id}.cif"

    # Builder lower-cases pid, so use lowercase filenames deterministically.
    pid = external_id.lower()
    dst = OUT_STRUCT / f"{pid}.cif"

    print(
        f"[{i+1:3d}/{len(final)}] "
        f"{external_id} <- {apo_id} chain={chain}"
    )

    if not src.exists():
        errors.append((external_id, "source CIF missing"))
        continue

    try:
        structure = parser.get_structure(pdb_id, str(src))

        available = [str(c.id) for c in next(structure.get_models())]

        if chain not in available:
            raise RuntimeError(
                f"chain {chain!r} absent; available={available}"
            )

        io = MMCIFIO()
        io.set_structure(structure)
        io.save(str(dst), select=ChainOnly(chain))

        obs = parse_output(dst)

        if not obs:
            raise RuntimeError("written chain-only CIF has zero residues")

        keys = [x[0] for x in obs]
        keyset = set(keys)

        # positives already mapped in Step 4
        p = long[
            (long["external_id"].astype(str).str.upper() == external_id) &
            (pd.to_numeric(long["label"], errors="coerce") == 1)
        ].copy()

        positive_keys = sorted(
            set(p["residue_key"].dropna().astype(str))
        )

        matched_positive = [
            k for k in positive_keys if k in keyset
        ]

        missing_positive = sorted(
            set(positive_keys) - keyset
        )

        if len(matched_positive) == 0:
            raise RuntimeError(
                f"no positive residue survived output CIF; "
                f"requested={len(positive_keys)}"
            )

        if missing_positive:
            raise RuntimeError(
                f"{len(missing_positive)} positive keys missing "
                f"after chain-only write: {missing_positive[:10]}"
            )

        # Verify only intended auth chain exists
        chains_written = sorted(
            {k.split("_", 1)[0] for k in keys}
        )

        if chains_written != [chain]:
            raise RuntimeError(
                f"unexpected chains after write: {chains_written}, "
                f"expected {[chain]}"
            )

        labels[pid] = [{
            "apo_chain": chain,
            "apo_pocket_selection": matched_positive,
        }]

        manifest.append({
            "pdb_id": pid,
            "external_id_original": external_id,
            "structureId_apo": apo_id,
            "source_pdb_id": pdb_id,
            "auth_chain": chain,
            "structure_file": str(dst),
            "n_residues": len(obs),
            "n_positive": len(matched_positive),
            "structure_official_coverage":
                row["structure_official_coverage"],
            "atom_sequence_agreement":
                row["atom_sequence_agreement"],
            "positive_observable_fraction":
                row["positive_observable_fraction"],
        })

    except Exception as e:
        errors.append((external_id, repr(e)))
        print("  [FAIL]", repr(e))


print("\n" + "=" * 100)
print("FINAL QC")
print("=" * 100)

print("prepared structures:", len(manifest))
print("label records:", len(labels))
print("errors:", len(errors))

if errors:
    for x in errors:
        print(x)

if errors:
    raise SystemExit("Step 5 preparation failed; fix errors before E4.")

if len(manifest) != 283 or len(labels) != 283:
    raise SystemExit(
        f"Expected 283 structures/labels, "
        f"got {len(manifest)}/{len(labels)}"
    )

Path(OUT_LABELS).write_text(
    json.dumps(labels, indent=2),
    encoding="utf-8"
)

m = pd.DataFrame(manifest)
m.to_csv(OUT_MANIFEST, index=False)

print("\nResidues:")
print(m["n_residues"].describe())

print("\nPositives:")
print(m["n_positive"].describe())

print("\nTotal residues:", int(m["n_residues"].sum()))
print("Total positives:", int(m["n_positive"].sum()))

print("\nSaved:")
print(OUT_STRUCT)
print(OUT_LABELS)
print(OUT_MANIFEST)

print("\nPASS — 283 chain-only external proteins ready.")
