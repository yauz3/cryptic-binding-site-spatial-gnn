#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from cryptogrid.common import parse_pdb_observed_residues, clean_pdb_id

ap=argparse.ArgumentParser(); ap.add_argument("--results",required=True); ap.add_argument("--pdb-root",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
rows=[]
for fp in Path(args.results).rglob("*-preds.npy"):
    pid=clean_pdb_id(fp.name.split("-preds")[0]); arr=np.load(fp).reshape(-1)
    pdb=Path(args.pdb_root)/f"{pid}.pdb"; rk,_=parse_pdb_observed_residues(pdb)
    if len(rk)!=len(arr):
        print(f"[WARN] {pid}: residues={len(rk)} preds={len(arr)} skip"); continue
    rows += [{"method":"AE-PocketMiner","pdb_id":pid,"residue_key":r,"score":float(s)} for r,s in zip(rk,arr)]
pd.DataFrame(rows).to_csv(args.out,index=False); print("[DONE]",args.out,len(rows))
