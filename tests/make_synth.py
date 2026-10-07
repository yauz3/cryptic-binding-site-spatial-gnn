#!/usr/bin/env python3
"""Synthetic mini-CryptoBench for smoke-testing the experiment scripts (NOT real data)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml

root = Path(sys.argv[1] if len(sys.argv) > 1 else "synth").resolve()
rng = np.random.default_rng(0)
AA = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
(root / "cifs").mkdir(parents=True, exist_ok=True)
(root / "features").mkdir(exist_ok=True)
(root / "pdb").mkdir(exist_ok=True)
D_ESM = 48

def walk(n):
    ca = np.zeros((n, 3)); d = rng.normal(size=3); d /= np.linalg.norm(d)
    for i in range(1, n):
        for _ in range(50):
            nd = d + rng.normal(scale=0.9, size=3) - 0.02 * ca[i - 1] / 3.8
            nd /= np.linalg.norm(nd)
            p = ca[i - 1] + 3.8 * nd
            if i < 3 or np.min(np.linalg.norm(ca[: i - 1] - p, axis=1)) > 4.0:
                break
        ca[i] = p; d = nd
    return ca

ids, dataset = [], {}
n_train, n_test = 48, 14
for k in range(n_train + n_test):
    pid = f"{rng.integers(1,9)}{''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), 3))}"
    while pid in ids:
        pid = f"{rng.integers(1,9)}{''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'), 3))}"
    ids.append(pid)
    n = int(rng.integers(70, 220))
    ca = walk(n)
    label_chain = "A"; auth_chain = rng.choice(["AAA", "A", "BBB"])
    start = int(rng.integers(-3, 400))
    resnum = start + np.arange(n)
    if n > 120:                       # introduce a numbering gap (missing loop)
        resnum[n // 2:] += 7
    comp = rng.choice(AA, size=n)
    # label: spatial patch around a random residue
    c = int(rng.integers(0, n))
    dist = np.linalg.norm(ca - ca[c], axis=1)
    y = dist < 8.5
    if y.sum() > 0.12 * n:
        y = dist <= np.sort(dist)[max(3, int(0.08 * n))]
    bfac = 20 + 15 * rng.random(n) + 4 * y + 4 * np.exp(-dist / 10)
    # atoms
    rows = []
    serial = 1
    for i in range(n):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        atoms = {"N": ca[i] + 1.46 * np.cross(u, [0, 0, 1.0]) / 1.0, "CA": ca[i],
                 "C": ca[i] - 1.52 * np.cross(u, [0, 1.0, 0]), "O": ca[i] + rng.normal(scale=1.3, size=3)}
        if comp[i] != "GLY":
            atoms["CB"] = ca[i] + 1.53 * u
            if rng.random() < 0.7:
                atoms["CG"] = ca[i] + 2.5 * u + rng.normal(scale=0.3, size=3)
        for an, xyz in atoms.items():
            rows.append(("ATOM", serial, an[0], an, ".", comp[i], label_chain, 1, i + 1, "?",
                         f"{xyz[0]:.3f}", f"{xyz[1]:.3f}", f"{xyz[2]:.3f}", "1.00", f"{bfac[i]+rng.normal():.2f}",
                         "?", str(resnum[i]), comp[i], auth_chain, an, 1))
            serial += 1
    # a hetero ligand-ish record that must be ignored
    rows.append(("HETATM", serial, "O", "O", ".", "HOH", "B", 2, ".", "?", "0.0", "0.0", "0.0", "1.00", "30.0",
                 "?", "900", "HOH", auth_chain, "O", 1))
    cols = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id",
            "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
            "B_iso_or_equiv", "pdbx_formal_charge", "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id",
            "pdbx_PDB_model_num"]
    with open(root / "cifs" / f"{pid}.cif", "w") as f:
        f.write(f"data_{pid.upper()}\n#\nloop_\n")
        for c_ in cols:
            f.write(f"_atom_site.{c_}\n")
        for r in rows:
            f.write(" ".join(str(x) for x in r) + "\n")
        f.write("#\n")
    truth = [f"{auth_chain}_{resnum[i]}" for i in range(n) if y[i]]
    dataset[pid] = [{"apo_chain": auth_chain, "apo_pocket_selection": truth, "holo_pdb_id": "xxxx"}]
    # grid pockets (label-chain namespace!) ; a few proteins get NO grid support
    prow = []
    no_support = (k % 11 == 5)
    npk = 0 if no_support else int(rng.integers(25, 60))
    for j in range(npk):
        gs = float(rng.choice([8, 10, 12, 16]))
        ctr = ca[int(rng.integers(0, n))] + rng.normal(scale=2.0, size=3)
        mem = np.flatnonzero(np.linalg.norm(ca - ctr, axis=1) < gs * 0.8)
        if len(mem) == 0:
            continue
        frac = y[mem].mean()
        feats = {f"f{q}_mean": rng.normal() + (1.2 * frac if q < 3 else 0.0) for q in range(10)}
        feats["gaff_energy"] = rng.standard_cauchy() * 50
        feats["sasa_total"] = np.nan if rng.random() < 0.2 else rng.random() * 100
        prow.append({"pdb_id": pid, "pdb_pocket": f"{pid}_{j}", "grid_size": gs, "pocket_name": f"p{j}",
                     "center_x": ctr[0], "center_y": ctr[1], "center_z": ctr[2],
                     "pocket_residues_list": ";".join(f"{label_chain}_{resnum[m]}" for m in mem),
                     "pocket_residues_count": len(mem), "cryptic_residues_count": int(y[mem].sum()),
                     "overlap_count": int(y[mem].sum()), "overlap_ratio": frac, "jaccard_index": frac / 2, **feats})
    if not prow:
        prow.append({"pdb_id": pid, "pdb_pocket": f"{pid}_0", "grid_size": 8.0, "pocket_name": "p0",
                     "center_x": 0, "center_y": 0, "center_z": 0, "pocket_residues_list": "Z_99999",
                     "pocket_residues_count": 1, "cryptic_residues_count": 0, "overlap_count": 0,
                     "overlap_ratio": 0, "jaccard_index": 0, **{f"f{q}_mean": 0.0 for q in range(10)},
                     "gaff_energy": 0.0, "sasa_total": 0.0})
    pd.DataFrame(prow).to_csv(root / "features" / f"{pid}_features.csv", index=False)
    # synthetic "ESM": noisy per-residue signal (deliberately NOT spatially smooth)
    X = rng.normal(size=(n, D_ESM)).astype(np.float32)
    X[:, :4] += (1.1 * y[:, None]).astype(np.float32)
    (root / "esm_raw").mkdir(exist_ok=True)
    np.save(root / "esm_raw" / f"{pid}.npy", X)

tr, te = ids[:n_train], ids[n_train:]
folds = {f"train-{i}": tr[i::4] for i in range(4)}
folds["test"] = te
(root / "folds.json").write_text(json.dumps(folds))
(root / "dataset.json").write_text(json.dumps(dataset))
cfg = yaml.safe_load(Path("config.yaml").read_text())
cfg["project"].update(dict(gridcrypto_root=str(root), folds_json=str(root / "folds.json"),
                           dataset_json=str(root / "dataset.json"), pdb_root=str(root / "pdb"),
                           cif_root=str(root / "cifs"), grid_features_dir=str(root / "features")))
cfg["work"] = dict(root=str(root / "work"), struct_cache=str(root / "work/cache_struct"),
                   esm_cache=str(root / "work/cache_esm"), results=str(root / "results"))
cfg["training"].update(dict(epochs=3, hidden_plm=32, hidden_struct=32, hidden_fusion=32, learning_rate=0.003, device="cpu"))
(root / "config.yaml").write_text(yaml.safe_dump(cfg))
print("synthetic root:", root, "| train", len(tr), "test", len(te))
