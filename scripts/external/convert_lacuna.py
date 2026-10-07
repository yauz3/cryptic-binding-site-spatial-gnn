#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
import pandas as pd
from cryptogrid.common import clean_pdb_id, normalize_residue_token

def pockets_from_obj(o):
    if isinstance(o,list): return o
    if isinstance(o,dict):
        for k in ("pockets","ranked_pockets","candidates","sites"):
            if isinstance(o.get(k),list): return o[k]
    return []

def residues(p):
    for k in ("lining_residues","residues","lining_residue_ids","residue_ids"):
        v=p.get(k) if isinstance(p,dict) else None
        if v:
            if isinstance(v,str): v=re.split(r"[;,\s]+",v)
            out=[normalize_residue_token(x) for x in v]
            return [x for x in out if x]
    return []

ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
rows=[]
for fp in Path(args.root).rglob("pocket_report.json"):
    pid=clean_pdb_id(fp.parent.name); obj=json.loads(fp.read_text()); ps=pockets_from_obj(obj); score={}
    for rank,p in enumerate(ps,1):
        raw=None
        for k in ("score","rank_score","crypticity","druggability"):
            if isinstance(p,dict) and p.get(k) is not None:
                try: raw=float(p[k]); break
                except Exception: pass
        w=raw if raw is not None else 1.0/rank
        for r in residues(p): score[r]=max(score.get(r,float("-inf")),w)
    rows += [{"method":"Lacuna","pdb_id":pid,"residue_key":r,"score":s} for r,s in score.items()]
pd.DataFrame(rows).to_csv(args.out,index=False); print("[DONE]",args.out,len(rows))
