#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import requests

BASE="https://api.osf.io/v2"
def get_json(url):
    r=requests.get(url,timeout=60); r.raise_for_status(); return r.json()
def walk(url,outroot):
    data=get_json(url)
    for item in data.get("data",[]):
        attr=item.get("attributes",{}); kind=attr.get("kind"); name=attr.get("name"); links=item.get("links",{})
        if kind=="file":
            target=outroot/name; target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.stat().st_size>0: continue
            dl=links.get("download")
            if dl:
                print("[download]",target)
                with requests.get(dl,stream=True,timeout=120) as r:
                    r.raise_for_status()
                    with target.open("wb") as f:
                        for ch in r.iter_content(1024*1024):
                            if ch: f.write(ch)
        elif kind=="folder":
            sub=outroot/name; sub.mkdir(parents=True,exist_ok=True)
            rel=item.get("relationships",{}).get("files",{}).get("links",{}).get("related",{}).get("href")
            if rel: walk(rel,sub)
    nxt=data.get("links",{}).get("next")
    if nxt: walk(nxt,outroot)

ap=argparse.ArgumentParser(); ap.add_argument("--project",default="pz4a9"); ap.add_argument("--out",required=True); args=ap.parse_args()
providers=get_json(f"{BASE}/nodes/{args.project}/files/")
for prov in providers.get("data",[]):
    name=prov["attributes"]["name"]; rel=prov["relationships"]["files"]["links"]["related"]["href"]; walk(rel,Path(args.out)/name)
print("[DONE]")
