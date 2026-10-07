from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from .common import (
    AA3_TO_1,
    clean_pdb_id,
    normalize_residue_token,
)


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def find_cif_file(
    cif_root: str | Path | None,
    pdb_id: str,
) -> Path | None:

    if not cif_root:
        return None

    root = Path(cif_root)
    pid = clean_pdb_id(pdb_id)

    for candidate in (
        root / f"{pid}.cif",
        root / f"{pid}.mmcif",
        root / f"{pid.upper()}.cif",
        root / f"{pid.upper()}.mmcif",
    ):
        if candidate.exists():
            return candidate

    return None


def parse_cif_residues(
    cif_path: str | Path,
    allowed_auth_chains: Sequence[str] | None = None,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Returns
    -------
    residues
        Residues in CryptoBench/auth namespace:
        e.g. AAA_1117

    aas
        AA1 sequence aligned one-to-one with residues.

    chain_map
        mmCIF label_asym_id -> auth_asym_id
        e.g. A -> AAA

    Important
    ---------
    Grid-Crypto PDB/grid residues use label-chain IDs but author residue
    numbering. CryptoBench labels use auth-chain IDs. Therefore residue
    numbers are preserved and only the chain namespace is remapped.
    """

    d = MMCIF2Dict(str(cif_path))

    group = _as_list(d.get("_atom_site.group_PDB"))
    label_chain = _as_list(d.get("_atom_site.label_asym_id"))
    auth_chain = _as_list(d.get("_atom_site.auth_asym_id"))
    auth_seq = _as_list(d.get("_atom_site.auth_seq_id"))
    comp = _as_list(d.get("_atom_site.label_comp_id"))
    ins = _as_list(d.get("_atom_site.pdbx_PDB_ins_code"))

    arrays = [
        group,
        label_chain,
        auth_chain,
        auth_seq,
        comp,
    ]

    n = min(len(x) for x in arrays if len(x))

    allow = set(str(x) for x in (allowed_auth_chains or []))

    chain_map: Dict[str, str] = {}

    residues: List[str] = []
    aas: List[str] = []
    seen = set()

    for i in range(n):

        if str(group[i]).upper() != "ATOM":
            continue

        lc = str(label_chain[i]).strip()
        ac = str(auth_chain[i]).strip()

        if ac in {".", "?", ""}:
            ac = lc

        if lc and ac:
            old = chain_map.get(lc)

            if old is not None and old != ac:
                raise RuntimeError(
                    f"{cif_path}: label chain {lc!r} maps to "
                    f"multiple auth chains: {old!r}, {ac!r}"
                )

            chain_map[lc] = ac

        if allow and ac not in allow:
            continue

        resi = str(auth_seq[i]).strip()

        if resi in {"", ".", "?"}:
            continue

        icode = ""

        if i < len(ins):
            x = str(ins[i]).strip()
            if x not in {"", ".", "?"}:
                icode = x

        raw_key = f"{ac}_{resi}{icode}"
        rk = normalize_residue_token(raw_key)

        if not rk or rk in seen:
            continue

        seen.add(rk)
        residues.append(rk)

        aa3 = str(comp[i]).strip().upper()
        aas.append(AA3_TO_1.get(aa3, "X"))

    return residues, aas, chain_map


def remap_residue_key(
    residue_key: str,
    label_to_auth: Dict[str, str],
) -> str:
    """
    Example:
        A_1117 + {"A": "AAA"} -> AAA_1117

    Residue number is deliberately unchanged.
    """

    r = normalize_residue_token(residue_key)

    if not r or "_" not in r:
        return r

    chain, resi = r.split("_", 1)

    auth_chain = label_to_auth.get(chain, chain)

    return normalize_residue_token(
        f"{auth_chain}_{resi}"
    )
