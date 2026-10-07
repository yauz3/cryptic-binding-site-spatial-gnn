#!/usr/bin/env python3
"""
Q1 EXPERIMENT E3 - Does better residue ranking become better pocket retrieval?

Every number in this project so far ranks residues. The question a user actually asks is
whether the true cryptic pocket is among the first few proposed pockets. This script converts
per-residue scores into ranked 3D pockets with one identical post-processing pipeline for
every method, then scores pocket retrieval with DCC and RRO.

Running every method through the same clustering is the whole point: pocket-level numbers from
different papers are not comparable, because the candidate generation, the success radius and
the definition of RRO all differ. We therefore never quote another paper's pocket numbers
beside ours; we re-derive them here from that method's own residue scores.

Inputs
------
--models        experiments in results/exp (ours) to evaluate
--predictions-dir  competitor CSVs (pdb_id, residue_key, score[, method]) from
                scripts/external/convert_comparators.py; each is treated as another method
--split         test (default) or oof

Post-processing is fixed a priori and identical for all methods: top 10% of residues by score,
single-linkage clustering at 8 A on CB atoms, clusters of at least 3 residues, pocket score =
mean of the 5 highest member scores, DCC success at 4 A, RRO success at Jaccard 0.5. The
sensitivity of the conclusion to these choices is reported by --sweep.

Outputs (under <results>/pocket_<split>/)
    pocket_per_protein.csv      every method x protein x metric
    table_pocket_absolute.csv/.tex
    table_pocket_paired.csv/.tex   paired against --reference, with permutation p and d_z
    sweep.csv                      (with --sweep) the same headline metrics over a grid of
                                   clustering parameters, to show the ordering is not an
                                   artefact of one setting
    summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cryptogrid.common import apo_chains, ensure_dir, load_dataset_json, normalize_residue_token
from cryptogrid.exp_common import (BASELINE, Store, canonical_frame, holm, list_experiments,
                                   load_preds, make_ctx, protein_slices, signflip_pvalue)
from cryptogrid.exp_stats import effect_sizes, fmt_p, label, macro_ci, write_table
from cryptogrid.pockets import POCKET_METRICS, evaluate_protein, predict_pockets, true_pockets

PRIMARY_POCKET = ["dcc_topN", "rro_success"]


def _sites_from_labels(store, pid, link_dist: float):
    """Fallback: split a protein's annotated residues into spatially connected sites.

    Used when the label source lists no explicit per-site selections for a protein, which is
    the case for an external benchmark built from a plain label vector. Sites are the
    single-linkage components of the positive residues at `link_dist`, which reproduces the
    per-site grouping of CryptoBench for the proteins where both are available.
    """
    y, keys = store.labels(pid)
    cb = store.geom(pid)["cb"].astype(np.float64)
    pos = np.flatnonzero((y > 0) & np.isfinite(cb).all(axis=1))
    if len(pos) == 0:
        return []
    if len(pos) == 1:
        return [[keys[pos[0]]]]
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    lab = fcluster(linkage(pdist(cb[pos]), method="single"), t=link_dist, criterion="distance")
    return [[keys[i] for i in pos[lab == l]] for l in np.unique(lab)]


def load_true_selections(ctx, store, labels_json=None, link_dist: float = 8.0):
    """{pdb_id: [list of residue-key lists]}, one entry per annotated cryptic site."""
    src = labels_json or ctx.cfg["project"].get("dataset_json")
    ds = {}
    if src and Path(src).exists():
        ds = load_dataset_json(src) if not labels_json else json.loads(Path(src).read_text())
    out, derived = {}, 0
    for pid in ctx.train_ids + ctx.test_ids:
        recs = ds.get(pid) or ds.get(pid.lower()) or []
        recs = recs if isinstance(recs, list) else [recs]
        chains = set(apo_chains(ds, pid)) if ds else set()
        sels = []
        for r in recs:
            keys = [normalize_residue_token(t) for t in r.get("apo_pocket_selection", [])]
            keys = [k for k in keys if k and (not chains or k.split("_", 1)[0] in chains)]
            if keys:
                sels.append(keys)
        if not sels:
            sels = _sites_from_labels(store, pid, link_dist)
            derived += bool(sels)
        out[pid] = sels
    if derived:
        print(f"[pocket] {derived} proteins had no per-site annotation in the label source; their sites "
              f"were derived by clustering the annotated residues at {link_dist:.0f} A (reported as such)")
    return out


def score_frames(ctx, models, predictions_dir, split):
    """{method label: (scores aligned to the canonical frame, coverage per protein)}."""
    ref = canonical_frame(ctx, split)
    out = {}
    for m in models:
        out[label(m)] = load_preds(ctx, m, split)["z"].to_numpy(dtype=float)
    d = Path(predictions_dir) if predictions_dir else None
    if d and d.is_dir():
        for f in sorted(d.glob("*.csv")):
            df = pd.read_csv(f, dtype={"pdb_id": str, "residue_key": str})
            if not {"pdb_id", "residue_key", "score"} <= set(df.columns):
                print(f"[skip] {f.name}: missing required columns")
                continue
            name = str(df["method"].iloc[0]) if "method" in df.columns and len(df) else f.stem
            df["pdb_id"] = df["pdb_id"].str.lower().str.strip()
            merged = ref.merge(df[["pdb_id", "residue_key", "score"]].drop_duplicates(["pdb_id", "residue_key"]),
                               on=["pdb_id", "residue_key"], how="left")
            out[name] = merged["score"].to_numpy(dtype=float)
    return out


def run(ctx, store, scores, truth, ids, params, min_coverage):
    rows = []
    sl = protein_slices(canonical_frame(ctx, params["split"])["pdb_id"].to_numpy())
    for name, z in scores.items():
        for pid, a, b in sl:
            s = z[a:b]
            cov = float(np.isfinite(s).mean())
            sels = truth.get(pid, [])
            if not sels:
                continue
            if cov < min_coverage:
                rows.append({"method": name, "pdb_id": pid, "coverage": cov, "evaluable": False})
                continue
            g = store.geom(pid)
            cb = g["cb"].astype(np.float64)
            keys = store.labels(pid)[1]
            pred = predict_pockets(s, cb, top_frac=params["top_frac"], link_dist=params["link_dist"],
                                   min_size=params["min_size"], score_top_k=params["score_top_k"])
            tp = true_pockets(keys, cb, sels)
            m = evaluate_protein(pred, tp, dcc_cut=params["dcc_cut"], rro_cut=params["rro_cut"])
            rows.append({"method": name, "pdb_id": pid, "coverage": cov, "evaluable": True, **m})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test", choices=["test", "oof"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--reference", default=BASELINE)
    ap.add_argument("--predictions-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--link-dist", type=float, default=8.0)
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--score-top-k", type=int, default=5)
    ap.add_argument("--dcc-cut", type=float, default=4.0)
    ap.add_argument("--rro-cut", type=float, default=0.5)
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--labels", default=None,
                    help="JSON of per-site selections (CryptoBench dataset.json schema). Default: the "
                         "dataset_json of the config; proteins absent from it get sites derived from "
                         "their label vector by spatial clustering.")
    ap.add_argument("--sweep", action="store_true", help="repeat over a grid of clustering settings")
    ap.add_argument("--n-perm", type=int, default=100000)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    store = Store(ctx)
    have = list_experiments(ctx, need_test=(args.split == "test"))
    models = [m for m in (args.models or have) if m in have]
    if args.reference in have and args.reference not in models:
        models = [args.reference] + models
    if not models:
        raise SystemExit("no model predictions found")
    out = ensure_dir(Path(args.out_dir) if args.out_dir
                     else Path(ctx.cfg["work"]["results"]) / f"pocket_{args.split}")
    ids = ctx.test_ids if args.split == "test" else ctx.train_ids
    truth = load_true_selections(ctx, store, args.labels, args.link_dist)
    n_sites = sum(len(truth.get(p, [])) for p in ids)
    scores = score_frames(ctx, models, args.predictions_dir, args.split)
    print(f"[pocket] {args.split}: {len(ids)} proteins, {n_sites} annotated pockets | "
          f"methods: {list(scores)}")

    params = dict(split=args.split, top_frac=args.top_frac, link_dist=args.link_dist,
                  min_size=args.min_size, score_top_k=args.score_top_k,
                  dcc_cut=args.dcc_cut, rro_cut=args.rro_cut)
    per = run(ctx, store, scores, truth, ids, params, args.min_coverage)
    if per.empty or "evaluable" not in per.columns:
        raise SystemExit("no protein had an annotated pocket and an evaluable score; check --labels "
                         "and the residue-key namespace of the competitor files")
    per.to_csv(out / "pocket_per_protein.csv", index=False)
    ev = per[per["evaluable"]]
    if ev.empty:
        raise SystemExit(f"every protein fell below --min-coverage {args.min_coverage}")

    rows = []
    for name in scores:
        g = ev[ev["method"] == name]
        if g.empty:
            continue
        r = {"Method": name, "Proteins": len(g),
             "Pockets/protein": f"{g['n_pred_pockets'].mean():.2f}"}
        for k in POCKET_METRICS:
            mean, lo, hi = macro_ci(g[k].to_numpy(dtype=float), args.n_boot)
            r[k] = f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
        rows.append(r)
    write_table(pd.DataFrame(rows), out / "table_pocket_absolute",
                caption=f"Pocket-level retrieval on the {args.split.upper()} split. Every method's "
                        "per-residue scores were converted into ranked pockets with one identical "
                        f"pipeline (top {args.top_frac:.0%} of residues, single-linkage at "
                        f"{args.link_dist:.0f}\\,\\AA, clusters of $\\geq${args.min_size} residues). "
                        f"DCC success at {args.dcc_cut:.0f}\\,\\AA; RRO is the Jaccard overlap with "
                        f"success at {args.rro_cut:.2f}. Values are means over proteins with 95\\% "
                        "bootstrap intervals.")

    ref_name = label(args.reference) if label(args.reference) in scores else list(scores)[0]
    base = ev[ev["method"] == ref_name].set_index("pdb_id")
    paired = []
    for name in scores:
        if name == ref_name:
            continue
        g = ev[ev["method"] == name].set_index("pdb_id")
        common = base.index.intersection(g.index)
        stats = []
        for k in POCKET_METRICS:
            d = (g.loc[common, k] - base.loc[common, k]).to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            if len(d) == 0:
                continue
            es = effect_sizes(d, n_boot=args.n_boot, seed=1)
            rng = np.random.default_rng(0)
            boot = d[rng.integers(0, len(d), size=(args.n_boot, len(d)))].mean(axis=1)
            stats.append({"method": name, "metric": k, "n_proteins": len(d),
                          "reference_mean": float(base.loc[common, k].mean()),
                          "method_mean": float(g.loc[common, k].mean()),
                          "mean_delta": float(d.mean()),
                          "ci95_low": float(np.percentile(boot, 2.5)),
                          "ci95_high": float(np.percentile(boot, 97.5)),
                          "p_signflip": signflip_pvalue(d, n_perm=args.n_perm, seed=2),
                          "cohen_dz": es["cohen_dz"],
                          "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()),
                          "n_equal": int((d == 0).sum())})
        prim = [i for i, x in enumerate(stats) if x["metric"] in PRIMARY_POCKET]
        adj = holm([stats[i]["p_signflip"] for i in prim])
        for x in stats:
            x["p_signflip_holm_primary"] = np.nan
        for i, a_ in zip(prim, adj):
            stats[i]["p_signflip_holm_primary"] = float(a_)
        paired += stats
    paired = pd.DataFrame(paired)
    if not paired.empty:
        paired["win/tie/loss"] = [f"{a}/{c}/{b}" for a, b, c in
                                  zip(paired["n_better"], paired["n_worse"], paired["n_equal"])]
        paired.to_csv(out / "table_pocket_paired_full.csv", index=False)
        slim = pd.DataFrame({
            "Method": paired["method"], "Metric": paired["metric"],
            "Reference": paired["reference_mean"], "Method value": paired["method_mean"],
            "Delta": paired["mean_delta"],
            "95% CI": [f"[{a:.4f}, {b:.4f}]" for a, b in zip(paired["ci95_low"], paired["ci95_high"])],
            "p (perm.)": paired["p_signflip"], "p Holm": paired["p_signflip_holm_primary"],
            "d_z": paired["cohen_dz"], "W/T/L": paired["win/tie/loss"]})
        write_table(slim, out / "table_pocket_paired", p_cols=["p (perm.)", "p Holm"],
                    caption=f"Pocket-level retrieval against {ref_name}, paired per protein. Holm "
                            f"adjustment within the prespecified pocket family {PRIMARY_POCKET}.")

    if args.sweep:
        grid = []
        for tf in (0.05, 0.10, 0.15):
            for ld in (6.0, 8.0, 10.0):
                for dc in (4.0, 6.0):
                    pr = dict(params, top_frac=tf, link_dist=ld, dcc_cut=dc)
                    e = run(ctx, store, scores, truth, ids, pr, args.min_coverage)
                    e = e[e["evaluable"]]
                    for name in scores:
                        g = e[e["method"] == name]
                        if g.empty:
                            continue
                        grid.append({"top_frac": tf, "link_dist": ld, "dcc_cut": dc, "method": name,
                                     **{k: float(g[k].mean()) for k in ("dcc_top1", "dcc_topN", "rro_success")}})
                    print(f"  [sweep] top_frac={tf} link={ld} dcc={dc}", flush=True)
        pd.DataFrame(grid).to_csv(out / "sweep.csv", index=False)

    lines = [f"# Pocket-level retrieval, {args.split} split", "",
             f"- {len(ids)} proteins, {n_sites} annotated cryptic pockets.",
             f"- Identical post-processing for every method: top {args.top_frac:.0%} of residues, "
             f"single-linkage at {args.link_dist:.0f} A, min {args.min_size} residues, pocket score = "
             f"mean of top {args.score_top_k}.",
             f"- DCC success at {args.dcc_cut:.0f} A; RRO = Jaccard, success at {args.rro_cut:.2f}.",
             f"- Reference: {ref_name}. Pocket primary family: {PRIMARY_POCKET}.", ""]
    for name in scores:
        g = ev[ev["method"] == name]
        if g.empty:
            continue
        lines.append(f"- **{name}** (n={len(g)}): DCC@1 {g['dcc_top1'].mean():.3f}, "
                     f"DCC@N {g['dcc_topN'].mean():.3f}, DCC@(N+2) {g['dcc_topN2'].mean():.3f}, "
                     f"RRO {g['rro'].mean():.3f}, RRO success {g['rro_success'].mean():.3f}")
    if not paired.empty:
        lines += ["", "| method | metric | ref | method | delta | 95% CI | p | d_z | W/T/L |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in paired.iterrows():
            lines.append(f"| {r['method']} | {r['metric']} | {r['reference_mean']:.4f} | "
                         f"{r['method_mean']:.4f} | {r['mean_delta']:+.4f} | "
                         f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {fmt_p(r['p_signflip'])} | "
                         f"{r['cohen_dz']:.3f} | {r['win/tie/loss']} |")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    for name in scores:
        g = ev[ev["method"] == name]
        if not g.empty:
            print(f"  {name:34s} DCC@1 {g['dcc_top1'].mean():.3f}  DCC@N {g['dcc_topN'].mean():.3f}  "
                  f"DCC@(N+2) {g['dcc_topN2'].mean():.3f}  RRO {g['rro'].mean():.3f}")
    print(f"\n[DONE] -> {out}")


if __name__ == "__main__":
    main()
