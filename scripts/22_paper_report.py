#!/usr/bin/env python3
"""
STEP 22 - Manuscript tables and figures for the FROZEN pilot model (results/exp).

Nothing is trained or selected here. OOF sections always run. TEST sections run only if the
preregistered unlock has happened (scripts/19 ... test --unlock-test) and the prediction files
still match the SHA-256 recorded at freeze time; otherwise they are skipped with a notice.
On test, only the frozen model x primary family {EF@1, BEDROC20} is confirmatory
(column `confirmatory`); every other test number is descriptive.

Outputs: results/paper/   (every table as .csv + booktabs .tex, every figure as .png 300 dpi + .pdf)
  T1  table1_performance_<split>          absolute performance, mean [95% CI over proteins]:
                                          macro AP/AUC/EF@1/EF@5/BEDROC20/50/80, macro+pooled MCC/F1,
                                          pooled AUC/AP (thresholds tuned on train OOF only)
  T2  table2_paired_vs_anchor_<split>     delta, 95% CI, sign-flip p, Holm, Wilcoxon p, t-test p,
                                          Cohen's d_z [CI], Hedges g_z, rank-biserial r,
                                          probability of superiority, win/tie/loss
  T3  table3_components_<split>           every candidate and control vs the anchor
      table3b_gnn_ablations_<split>       GNN ablations vs the full GNN (paired)
  T4  table4_grid_contribution_oof        nested re-stacking with / without Grid-derived inputs
  T5  table5_subgroups_<split>            Grid support, length, #site residues, chains, B-factors
  T6  table6_stack_weights                non-negative stack weights
  T7  table7_external_test                external predictors on identical residues (test, if provided)
  S   case_studies_<split>.csv            largest gains AND largest losses
  F1  fig1_enrichment_<split>             recall of site residues vs top-x % (macro, 95% band)
  F2  fig2_pooled_pr_roc_<split>
  F3  fig3_paired_scatter_<split>         per-protein anchor vs method + delta histogram
  F4  fig4_forest_components_<split>
  F5  fig5_forest_subgroups_<split>
  PAPER_NUMBERS.md                        key numbers + protocol statement for the Methods section
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from cryptogrid.common import ensure_dir
from cryptogrid.exp_common import (BASELINE, METRICS, PRIMARY, Store, canonical_frame, list_experiments, load_preds,
                                   make_ctx, per_protein_table, protein_slices, rank_metrics)
from cryptogrid.exp_stats import (GRID_DERIVED, METRIC_LABELS, crossfit_thresholds, enrichment_curve, fmt_p,
                                  full_paired, label, macro_ci, mcc_threshold, nested_stack, new_fig, pooled_compare,
                                  save_fig, test_gate, threshold_tables, write_table)

KEY = ["ap", "ef1", "bedroc20"]
FRACS = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20])


def thresholds_for(ctx, name, split):
    oof = load_preds(ctx, name, "oof")
    if split == "oof":
        return crossfit_thresholds(oof, ctx.fold_names)          # tuned on the other folds
    return mcc_threshold(oof["y_true"].to_numpy().astype(int), oof["z"].to_numpy(dtype=float))


def tidy(st, method, split, winner):
    st = st.copy()
    st.insert(0, "method", method)
    st.insert(0, "split", split)
    st["win/tie/loss"] = [f"{a}/{c}/{b}" for a, b, c in zip(st["n_better"], st["n_worse"], st["n_equal"])]
    st["confirmatory"] = (split == "test") & (method == winner) & st["metric"].isin(PRIMARY)
    return st


# ---------------------------------------------------------------------
# T1 / T2
# ---------------------------------------------------------------------

def main_tables(ctx, split, methods, winner, out, args):
    frames = {m: load_preds(ctx, m, split) for m in methods}
    pps = {m: per_protein_table(frames[m]) for m in methods}
    thr_pp, thr_pool = {}, {}
    for m in methods:
        thr_pp[m], thr_pool[m] = threshold_tables(frames[m], thresholds_for(ctx, m, split))
    rows = []
    for m in methods:
        r = {"Method": label(m), "n proteins": len(pps[m])}
        for k in METRICS:
            mean, lo, hi = macro_ci(pps[m][k].to_numpy(), args.n_boot)
            r[METRIC_LABELS[k]] = f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
        for k in ("mcc", "f1"):
            mean, lo, hi = macro_ci(thr_pp[m][k].to_numpy(), args.n_boot)
            r[f"macro {METRIC_LABELS[k]}"] = f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
        pc = pooled_compare(frames[BASELINE], frames[m], n_boot=2, seed=0).set_index("metric")
        r["pooled ROC-AUC"], r["pooled AP"] = pc.loc["pooled_auc", "new"], pc.loc["pooled_ap", "new"]
        r["pooled MCC"], r["pooled F1"] = thr_pool[m]["pooled_mcc"], thr_pool[m]["pooled_f1"]
        rows.append(r)
    write_table(pd.DataFrame(rows), out / f"table1_performance_{split}",
                caption=f"Residue-level performance on the {split.upper()} split. Macro metrics: mean over proteins "
                        "[95\\% bootstrap CI over proteins]. Thresholds for MCC/F1 were tuned on training OOF predictions only.")

    paired, pooled = [], []
    for m in methods:
        if m == BASELINE:
            continue
        st = full_paired(pps[BASELINE], pps[m], n_perm=args.n_perm, n_boot=args.n_boot)
        st2 = full_paired(thr_pp[BASELINE], thr_pp[m], metrics=["mcc", "f1"], n_perm=args.n_perm, n_boot=args.n_boot)
        paired.append(tidy(pd.concat([st, st2], ignore_index=True), m, split, winner))
        pc = pooled_compare(frames[BASELINE], frames[m], n_boot=args.n_boot_pooled)
        pc.insert(0, "method", m)
        pc.insert(0, "split", split)
        pooled.append(pc)
    paired = pd.concat(paired, ignore_index=True)
    paired.to_csv(out / f"table2_paired_vs_anchor_{split}_full.csv", index=False)
    pd.concat(pooled, ignore_index=True).to_csv(out / f"table2_pooled_delta_{split}.csv", index=False)
    slim = pd.DataFrame({
        "Method": [label(m) for m in paired["method"]], "Metric": [METRIC_LABELS[k] for k in paired["metric"]],
        "Anchor": paired["mean_base"], "Method value": paired["mean_new"], "Delta": paired["mean_delta"],
        "95% CI": [f"[{a:.4f}, {b:.4f}]" for a, b in zip(paired["ci95_low"], paired["ci95_high"])],
        "p (perm.)": paired["p_signflip"], "p Holm": paired["p_signflip_holm_primary"],
        "p (Wilcoxon)": paired["p_wilcoxon"], "d_z": paired["cohen_dz"],
        "d_z 95% CI": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(paired["dz_ci95_low"], paired["dz_ci95_high"])],
        "r_rb": paired["rank_biserial"], "W/T/L": paired["win/tie/loss"]})
    write_table(slim, out / f"table2_paired_vs_anchor_{split}", p_cols=["p (perm.)", "p Holm", "p (Wilcoxon)"],
                caption=f"Paired per-protein comparison with the pLM anchor on the {split.upper()} split. "
                        "p (perm.): two-sided sign-flip permutation test on the mean difference; Holm adjustment within "
                        "the prespecified primary family (EF@1\\%, BEDROC$_{20}$). $d_z$: paired Cohen's d; "
                        "$r_{rb}$: matched-pairs rank-biserial correlation.")
    return frames, pps, paired


# ---------------------------------------------------------------------
# T3 components / ablations + forest
# ---------------------------------------------------------------------

def components(ctx, split, winner, out, args, base_pp):
    exps = [e for e in list_experiments(ctx, need_test=(split == "test")) if e != BASELINE]
    pps, rows = {}, []
    for e in exps:
        pps[e] = per_protein_table(load_preds(ctx, e, split))
        st = full_paired(base_pp, pps[e], n_perm=args.n_perm_secondary, n_boot=args.n_boot)
        rows.append(tidy(st, e, split, winner))
    comp = pd.concat(rows, ignore_index=True)
    comp.to_csv(out / f"table3_components_{split}_full.csv", index=False)
    order = comp[comp["metric"] == "bedroc20"].sort_values("mean_delta", ascending=False)["method"].tolist()
    wide = []
    for e in order:
        s = comp[comp["method"] == e].set_index("metric")
        r = {"Model": label(e)}
        for k in KEY:
            r[f"d{METRIC_LABELS[k]}"] = s.loc[k, "mean_delta"]
            r[f"p {METRIC_LABELS[k]}"] = s.loc[k, "p_signflip"]
            r[f"d_z {METRIC_LABELS[k]}"] = s.loc[k, "cohen_dz"]
        wide.append(r)
    write_table(pd.DataFrame(wide), out / f"table3_components_{split}", p_cols=[f"p {METRIC_LABELS[k]}" for k in KEY],
                caption=f"All candidate models and controls versus the pLM anchor ({split.upper()}). "
                        "d: mean per-protein difference; p: sign-flip permutation test (unadjusted); $d_z$: paired Cohen's d.")
    if "gnn_anchor" in pps:
        ab = [tidy(full_paired(pps["gnn_anchor"], pps[e], n_perm=args.n_perm_secondary, n_boot=args.n_boot), e, split, winner)
              for e in exps if e.startswith("ablation_gnn")]
        if ab:
            ab = pd.concat(ab, ignore_index=True)
            ab.to_csv(out / f"table3b_gnn_ablations_{split}_full.csv", index=False)
            s = ab[ab["metric"].isin(KEY + ["auc"])]
            write_table(pd.DataFrame({"Ablation": [label(m) for m in s["method"]],
                                      "Metric": [METRIC_LABELS[k] for k in s["metric"]],
                                      "Full GNN": s["mean_base"].to_numpy(), "Ablated": s["mean_new"].to_numpy(),
                                      "Delta": s["mean_delta"].to_numpy(),
                                      "95% CI": [f"[{a:.4f}, {b:.4f}]" for a, b in zip(s["ci95_low"], s["ci95_high"])],
                                      "p (perm.)": s["p_signflip"].to_numpy(), "d_z": s["cohen_dz"].to_numpy()}),
                        out / f"table3b_gnn_ablations_{split}", p_cols=["p (perm.)"],
                        caption=f"Ablations of the anchored spatial GNN, paired against the full GNN ({split.upper()}).")
    fig, axes = new_fig(ncols=len(KEY), w=3.3, h=0.3 * len(order) + 1.3)
    for ax, k in zip(axes[0], KEY):
        s = comp[comp["metric"] == k].set_index("method").loc[order]
        yy = np.arange(len(order))
        ax.errorbar(s["mean_delta"], yy, xerr=[s["mean_delta"] - s["ci95_low"], s["ci95_high"] - s["mean_delta"]],
                    fmt="o", ms=3.5, lw=1, capsize=2, color="k")
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks(yy)
        ax.set_yticklabels([label(e) for e in order] if ax is axes[0][0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(f"$\\Delta$ {METRIC_LABELS[k]} vs anchor (95% CI)")
    save_fig(fig, out / f"fig4_forest_components_{split}")
    return comp


# ---------------------------------------------------------------------
# T4 marginal contribution of Grid-derived inputs (train OOF, nested)
# ---------------------------------------------------------------------

def grid_contribution(ctx, winner, out, args, base, base_pp):
    sel = ctx.exp_root / winner / "selection.json"
    if not sel.exists() or "inputs" not in json.loads(sel.read_text()):
        print("[T4] frozen model is not a stack - skipped")
        return None
    inputs = json.loads(sel.read_text())["inputs"]
    z0 = base["z"].to_numpy(dtype=float)
    y = base["y_true"].to_numpy(dtype=float)
    pid, fold = base["pdb_id"].to_numpy(), base["fold"].to_numpy()
    D = np.stack([load_preds(ctx, e, "oof")["z"].to_numpy(dtype=float) - z0 for e in inputs], axis=1)
    is_grid = np.array([any(g in e for g in GRID_DERIVED) for e in inputs])
    variants = {"all inputs (frozen stack definition)": np.ones(len(inputs), bool),
                "without Grid-derived inputs": ~is_grid,
                "Grid-derived inputs only": is_grid,
                "spatial GNN only": np.array([e == "gnn_anchor" for e in inputs])}
    if args.loo:
        for i, e in enumerate(inputs):
            m = np.ones(len(inputs), bool)
            m[i] = False
            variants[f"leave out: {label(e)}"] = m
    pps = {}
    for name, mask in variants.items():
        if not mask.any():
            continue
        t0 = time.time()
        zz = nested_stack(z0, D[:, mask], y, pid, fold, ctx.fold_names)
        pps[name] = per_protein_table(base.assign(z=zz))
        print(f"[T4] {name}: {int(mask.sum())} inputs, {time.time() - t0:.0f}s", flush=True)
    full = next(iter(pps))
    rows = []
    for name, pp in pps.items():
        a = full_paired(base_pp, pp, metrics=KEY, n_perm=args.n_perm_secondary, n_boot=args.n_boot).set_index("metric")
        b = full_paired(pps[full], pp, metrics=KEY, n_perm=args.n_perm_secondary, n_boot=args.n_boot).set_index("metric")
        for k in KEY:
            rows.append({"Stack variant": name, "Metric": METRIC_LABELS[k], "Value": a.loc[k, "mean_new"],
                         "Delta vs anchor": a.loc[k, "mean_delta"], "p vs anchor": a.loc[k, "p_signflip"],
                         "Delta vs full stack": b.loc[k, "mean_delta"],
                         "95% CI vs full": f"[{b.loc[k, 'ci95_low']:.4f}, {b.loc[k, 'ci95_high']:.4f}]",
                         "p vs full": b.loc[k, "p_signflip"], "d_z vs full": b.loc[k, "cohen_dz"]})
    t4 = pd.DataFrame(rows)
    write_table(t4, out / "table4_grid_contribution_oof", p_cols=["p vs anchor", "p vs full"],
                caption="Marginal contribution of Grid-Crypto-derived inputs: the anchored stack re-fitted with nested "
                        "cross-fitting on training OOF predictions for different input subsets.")
    (out / "table4_inputs.json").write_text(json.dumps({"inputs": inputs, "grid_derived": [e for e, g in zip(inputs, is_grid) if g]}, indent=2))
    return t4


# ---------------------------------------------------------------------
# T5 subgroups
# ---------------------------------------------------------------------

def protein_attributes(ctx, store, ids):
    names = store.struct_feature_names()
    mi = names.index("summary::pocket_membership_count") if "summary::pocket_membership_count" in names else None
    bi = store.geom_feature_names().index("b_available")
    rows = []
    for pid in ids:
        y, _ = store.labels(pid)
        g = store.geom(pid)
        sup = (np.nan_to_num(store.struct(pid)[:, mi], nan=0.0) > 0).mean() if mi is not None else np.nan
        rows.append({"pdb_id": pid, "n_residues": len(y), "n_site_residues": int(y.sum()), "grid_support_frac": float(sup),
                     "n_chains": int(len(np.unique(g["chain_index"]))), "has_bfactor": bool(g["X_geom"][0, bi] > 0)})
    return pd.DataFrame(rows).set_index("pdb_id")


def subgroups(split, attrs, base_pp, new_pp, winner, out, args):
    a = attrs.reindex(base_pp.index)

    def tert(col, name):
        q = a[col].quantile([1 / 3, 2 / 3]).to_numpy()
        lab = np.where(a[col] <= q[0], "low", np.where(a[col] <= q[1], "mid", "high"))
        return [(f"{name}: {l} tertile", lab == l) for l in ("low", "mid", "high")]

    groups = [("All proteins", np.ones(len(a), bool)),
              ("Grid support: none", (a["grid_support_frac"] == 0).to_numpy()),
              ("Grid support: any", (a["grid_support_frac"] > 0).to_numpy())]
    groups += tert("n_residues", "Length") + tert("n_site_residues", "Site size")
    groups += [("Chains: single", (a["n_chains"] == 1).to_numpy()), ("Chains: multiple", (a["n_chains"] > 1).to_numpy()),
               ("B-factors: available", a["has_bfactor"].to_numpy()), ("B-factors: not available", ~a["has_bfactor"].to_numpy())]
    rows = []
    for gname, mask in groups:
        if mask.sum() < 5:
            continue
        st = full_paired(base_pp[mask], new_pp[mask], metrics=KEY, n_perm=args.n_perm_secondary, n_boot=args.n_boot)
        for _, r in st.iterrows():
            rows.append({"Subgroup": gname, "n": int(mask.sum()), "Metric": METRIC_LABELS[r["metric"]], "metric": r["metric"],
                         "Anchor": r["mean_base"], "Method": r["mean_new"], "Delta": r["mean_delta"],
                         "ci_low": r["ci95_low"], "ci_high": r["ci95_high"], "p (perm.)": r["p_signflip"], "d_z": r["cohen_dz"]})
    sg = pd.DataFrame(rows)
    sg.to_csv(out / f"table5_subgroups_{split}_full.csv", index=False)
    t = sg.assign(**{"95% CI": [f"[{a_:.4f}, {b_:.4f}]" for a_, b_ in zip(sg["ci_low"], sg["ci_high"])]})
    write_table(t[["Subgroup", "n", "Metric", "Anchor", "Method", "Delta", "95% CI", "p (perm.)", "d_z"]],
                out / f"table5_subgroups_{split}", p_cols=["p (perm.)"],
                caption=f"Subgroup analysis of {label(winner)} versus the pLM anchor ({split.upper()}); exploratory, unadjusted p-values.")
    fig, axes = new_fig(ncols=len(KEY), w=3.3, h=0.3 * sg["Subgroup"].nunique() + 1.3)
    names = list(dict.fromkeys(sg["Subgroup"]))
    for ax, k in zip(axes[0], KEY):
        s = sg[sg["metric"] == k].set_index("Subgroup").loc[names]
        yy = np.arange(len(names))
        ax.errorbar(s["Delta"], yy, xerr=[s["Delta"] - s["ci_low"], s["ci_high"] - s["Delta"]], fmt="o", ms=3.5, lw=1,
                    capsize=2, color="k")
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{n} (n={int(s.loc[n, 'n'])})" for n in names] if ax is axes[0][0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(f"$\\Delta$ {METRIC_LABELS[k]} (95% CI)")
    save_fig(fig, out / f"fig5_forest_subgroups_{split}")


# ---------------------------------------------------------------------
# Figures 1-3, case studies
# ---------------------------------------------------------------------

def curve_figures(split, frames, pps, methods, winner, out):
    rng = np.random.default_rng(0)
    fig, axes = new_fig(ncols=1, w=4.0, h=3.3)
    ax = axes[0][0]
    for m in methods:
        E = enrichment_curve(frames[m], FRACS)
        boot = E[rng.integers(0, len(E), size=(1000, len(E)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        ax.plot(FRACS * 100, E.mean(axis=0), marker="o", ms=3, label=label(m))
        ax.fill_between(FRACS * 100, lo, hi, alpha=0.18)
    ax.plot(FRACS * 100, FRACS, color="grey", ls="--", lw=0.8, label="random")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("top x % of residues per protein")
    ax.set_ylabel("recall of site residues (macro)")
    ax.set_title(f"Early enrichment, {split.upper()}")
    ax.legend(fontsize=7)
    save_fig(fig, out / f"fig1_enrichment_{split}")

    fig, axes = new_fig(ncols=2, w=3.4, h=3.2)
    for m in methods:
        y, z = frames[m]["y_true"].to_numpy(), frames[m]["z"].to_numpy(dtype=float)
        p, r, _ = precision_recall_curve(y, z)
        f, t, _ = roc_curve(y, z)
        i1, i2 = np.linspace(0, len(p) - 1, 2000).astype(int), np.linspace(0, len(f) - 1, 2000).astype(int)
        axes[0][0].plot(r[i1], p[i1], lw=1.2, label=label(m))
        axes[0][1].plot(f[i2], t[i2], lw=1.2, label=label(m))
    axes[0][0].set_xlabel("recall (pooled residues)"); axes[0][0].set_ylabel("precision")
    axes[0][1].plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    axes[0][1].set_xlabel("false positive rate (pooled residues)"); axes[0][1].set_ylabel("true positive rate")
    axes[0][0].legend(fontsize=7)
    save_fig(fig, out / f"fig2_pooled_pr_roc_{split}")

    b, n = pps[BASELINE], pps[winner].reindex(pps[BASELINE].index)
    fig, axes = new_fig(ncols=3, w=3.1, h=3.0)
    for ax, k in zip(axes[0][:2], ("ap", "bedroc20")):
        ax.scatter(b[k], n[k], s=5, alpha=0.45, color="k", linewidths=0)
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
        ax.set_xlabel(f"{METRIC_LABELS[k]} - pLM anchor"); ax.set_ylabel(f"{METRIC_LABELS[k]} - {label(winner)}")
        ax.set_title(f"{int((n[k] > b[k]).sum())} better / {int((n[k] < b[k]).sum())} worse")
    d = (n["ap"] - b["ap"]).dropna()
    axes[0][2].hist(d, bins=40, color="0.4")
    axes[0][2].axvline(0, color="grey", ls="--", lw=0.8); axes[0][2].axvline(d.mean(), color="k", lw=1.2)
    axes[0][2].set_xlabel("per-protein $\\Delta$AP"); axes[0][2].set_ylabel("proteins")
    save_fig(fig, out / f"fig3_paired_scatter_{split}")

    cs = pd.DataFrame({"n_residues": b["n"], "n_site_residues": b["positives"]})
    for k in KEY:
        cs[f"{k}_anchor"], cs[f"{k}_method"], cs[f"delta_{k}"] = b[k], n[k], n[k] - b[k]
    cs = cs.sort_values("delta_ap")
    pd.concat([cs.tail(15).iloc[::-1].assign(group="largest gains"), cs.head(15).assign(group="largest losses")]
              ).reset_index().to_csv(out / f"case_studies_{split}.csv", index=False)


# ---------------------------------------------------------------------
# T7 external predictors (test only)
# ---------------------------------------------------------------------

def external_table(ctx, ext_dir, frames, winner, out, args):
    files = sorted(Path(ext_dir).glob("*.csv")) if Path(ext_dir).is_dir() else []
    if not files:
        print(f"[T7] no external prediction CSVs in {ext_dir} - skipped")
        return
    ref = canonical_frame(ctx, "test")
    rows = []
    for fp in files:
        ext = pd.read_csv(fp, dtype={"pdb_id": str, "residue_key": str})
        if not {"pdb_id", "residue_key", "score"} <= set(ext.columns):
            print(f"[T7] {fp.name}: needs columns pdb_id,residue_key,score - skipped")
            continue
        name = str(ext["method"].iloc[0]) if "method" in ext.columns and len(ext) else fp.stem
        ext["pdb_id"] = ext["pdb_id"].str.lower()
        m = ref.merge(ext[["pdb_id", "residue_key", "score"]].drop_duplicates(["pdb_id", "residue_key"]),
                      on=["pdb_id", "residue_key"], how="left")
        s, y = m["score"].to_numpy(dtype=float), ref["y_true"].to_numpy()
        zw = frames[winner]["z"].to_numpy(dtype=float)
        pe, pw, cov = [], [], []
        for pid, a, b in protein_slices(ref["pdb_id"].to_numpy()):
            ok = np.isfinite(s[a:b])
            cov.append(ok.mean())
            if ok.mean() < args.external_min_coverage or y[a:b][ok].sum() == 0:
                continue
            pe.append({"pdb_id": pid, **rank_metrics(y[a:b][ok], s[a:b][ok])})      # identical residues for both
            pw.append({"pdb_id": pid, **rank_metrics(y[a:b][ok], zw[a:b][ok])})
        if len(pe) < 5:
            print(f"[T7] {name}: only {len(pe)} evaluable proteins - skipped")
            continue
        st = full_paired(pd.DataFrame(pe).set_index("pdb_id"), pd.DataFrame(pw).set_index("pdb_id"),
                         n_perm=args.n_perm_secondary, n_boot=args.n_boot)
        for _, r in st.iterrows():
            rows.append({"External method": name, "n proteins": int(r["n_proteins"]), "mean coverage": float(np.mean(cov)),
                         "Metric": METRIC_LABELS[r["metric"]], "External": r["mean_base"], label(winner): r["mean_new"],
                         "Delta": r["mean_delta"], "95% CI": f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]",
                         "p (perm.)": r["p_signflip"], "d_z": r["cohen_dz"]})
    if rows:
        write_table(pd.DataFrame(rows), out / "table7_external_test", p_cols=["p (perm.)"],
                    caption="Comparison with external predictors on the CryptoBench test set, evaluated on identical residues "
                            "(proteins below the coverage threshold are excluded and counted). Descriptive.")


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--methods", nargs="*", default=None, help="extra models shown next to anchor and frozen model")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--loo", action="store_true", help="T4: also leave-one-input-out re-stacking (slow)")
    ap.add_argument("--skip-grid-contribution", action="store_true")
    ap.add_argument("--external-dir", default="external_predictions")
    ap.add_argument("--external-min-coverage", type=float, default=0.95)
    ap.add_argument("--n-perm", type=int, default=100000)
    ap.add_argument("--n-perm-secondary", type=int, default=20000)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-boot-pooled", type=int, default=300)
    args = ap.parse_args()

    ctx = make_ctx(args.config)
    store = Store(ctx)
    out = ensure_dir(Path(ctx.cfg["work"]["results"]) / "paper")
    gate = test_gate(ctx.exp_root)
    if gate["frozen"] is None:
        raise SystemExit("no frozen_selection.json - run scripts/19_stats_and_freeze.py oof first")
    winner = gate["frozen"]["winner"]
    have = set(list_experiments(ctx))
    methods = [BASELINE] + [m for m in (args.methods or ["gnn_anchor"]) if m in have and m not in (BASELINE, winner)] + [winner]
    splits = ["oof"]
    if args.no_test:
        note = "test sections disabled by --no-test"
    elif not gate["unlocked"]:
        note = "TEST NOT REPORTED: frozen model has not been unlocked yet (scripts/19 ... test --unlock-test)"
    elif not gate["hash_ok"]:
        note = "TEST NOT REPORTED: prediction files differ from the SHA-256 recorded at freeze time"
    else:
        splits.append("test")
        note = f"test reported; number of unlocks recorded: {gate['n_unlocks']}"
    print(f"[paper] frozen model = {winner} | models shown = {methods} | {note}")

    summary, t4 = {}, None
    for split in splits:
        frames, pps, paired = main_tables(ctx, split, methods, winner, out, args)
        comp = components(ctx, split, winner, out, args, pps[BASELINE])
        if split == "oof" and not args.skip_grid_contribution:
            t4 = grid_contribution(ctx, winner, out, args, frames[BASELINE], pps[BASELINE])
        ids = ctx.train_ids if split == "oof" else ctx.test_ids
        subgroups(split, protein_attributes(ctx, store, ids), pps[BASELINE], pps[winner].reindex(pps[BASELINE].index),
                  winner, out, args)
        curve_figures(split, frames, pps, methods, winner, out)
        if split == "test":
            external_table(ctx, args.external_dir, frames, winner, out, args)
        summary[split] = paired[paired["method"] == winner]

    sw = ctx.exp_root / winner / "stack_weights.csv"
    if sw.exists():
        w = pd.read_csv(sw)
        w["input"] = [label(x) for x in w["input"]]
        write_table(w, out / "table6_stack_weights", caption="Non-negative weights of the anchored stack (fitted on training OOF predictions).")

    lines = ["# Key numbers for the manuscript", "",
             f"- Frozen model: `{winner}` (frozen {gate['frozen']['frozen_at']}; criterion {gate['frozen']['criterion']}; "
             "selected on the 885 training proteins, official-fold OOF, nested selection).",
             f"- Test status: {note}.",
             "- Unit of analysis: protein. Primary test: paired two-sided sign-flip permutation test on the mean per-protein "
             f"difference ({args.n_perm} permutations); Holm adjustment within the primary family {PRIMARY}.",
             "- Effect size: paired Cohen's d_z = mean(diff)/SD(diff) over proteins, percentile-bootstrap 95% CI.",
             "- BEDROC50/80 are reported as secondary: alpha * (active fraction) is not << 1 at residue level.", ""]
    for split, s in summary.items():
        lines += [f"## {split.upper()} ({int(s['n_proteins'].iloc[0])} proteins): {label(winner)} vs pLM anchor", "",
                  "| metric | anchor | method | delta | 95% CI | p (perm.) | p Holm | d_z | W/T/L |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            lines.append(f"| {r['metric']} | {r['mean_base']:.4f} | {r['mean_new']:.4f} | {r['mean_delta']:+.4f} | "
                         f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {fmt_p(r['p_signflip'])} | "
                         f"{fmt_p(r['p_signflip_holm_primary'])} | {r['cohen_dz']:.3f} | {r['win/tie/loss']} |")
        lines.append("")
    if t4 is not None:
        try:
            body = t4.to_markdown(index=False, floatfmt=".4f")
        except ImportError:                               # to_markdown needs `tabulate`
            body = "```\n" + t4.to_string(index=False) + "\n```"
        lines += ["## Marginal contribution of Grid-derived stack inputs (train OOF, nested re-stacking)", "", body, ""]
    (out / "PAPER_NUMBERS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[DONE] tables and figures -> {out}\n{note}")


if __name__ == "__main__":
    main()
