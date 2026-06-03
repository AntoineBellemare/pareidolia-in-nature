"""v3 figure regeneration on sentence-pooled SigLIP-2 (no biome-tell filter).

Produces (writes to paper/figures/):
  fig2_biome_bars.png         — per-biome marginal + stratified Δ on iNat
  fig5_earth_map.png          — confidence-controlled world map (iNat, stratified)
  fig11_universals_analysis.png — Spec A / Semi / Universal breadth gradient
  fig_v2_controls.png         — 4-panel robustness atlas
                                  A. content vs structure (original vs word-shuffled)
                                  B. two-corpus replication (iNat ↔ Places365)
                                  C. anonymisation audit (200 motifs)
                                  D. four-model robustness
  fig9_crossmodel_correlation.png — per-biome Δ correlation across 4 models
  fig_taxon_combined.png      — taxon-stratified Δ on iNat (kept structure)
  fig4_taxon_facets.png       — per-iconic-taxon facets (supp)

All Δ values are sentence-pooled SigLIP-2 on full LLM-clean Berezkin abstracts.
No iteration-history panels (anonymisation stages, biome-tell filter audit).
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[2]  # project root
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
EMB_ALL = ROOT / "dataset/imagery/embeddings"
MAP = ROOT / "dataset/mapping_v2"
FIG = ROOT / "paper/figures"
FIG.mkdir(parents=True, exist_ok=True)

from make_phase2_figures import short_biome, biome_color, sig_stars


def bh_q(p):
    p = np.asarray(p, float); valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if not valid.any(): return q
    pv = p[valid]; n = len(pv)
    order = np.argsort(pv); ranked = pv[order]
    qv = ranked * n / (np.arange(n) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0, 1)
    unranked = np.empty_like(qv); unranked[order] = qv
    q[valid] = unranked; return q


# --------------------------------------------------------------------------- #
# FIG 2 — Two-corpus replication: iNat (left) + Places365 (right), marginal Δ
# --------------------------------------------------------------------------- #
def fig2_biome_bars():
    inat = pd.read_csv(EMB / "inat_basic/v3_biome_test_sentpool_resid.csv")
    # Use stratified Δ as the iNat headline (delta_strat / p_strat)
    inat["delta_main"] = inat["delta_strat"]
    inat["p_main"] = inat["p_strat"]
    inat["q"] = bh_q(inat["p_main"].values)
    p365 = pd.read_csv(EMB / "places365_strict/v3_biome_test_sentpool_resid.csv")
    p365["delta_main"] = p365["delta"]  # P365 has no taxa to stratify on
    p365["p_main"] = p365["p_one_sided"]
    p365["q"] = bh_q(p365["p_main"].values)

    # Shared biome ordering: by iNat stratified Δ (largest at top)
    order = inat.sort_values("delta_main", ascending=False)["biome"].tolist()
    inat_o = inat.set_index("biome").reindex(order).reset_index()

    # Build P365 frame aligned to the same biome order; missing biomes leave a gap
    p365_idx = p365.set_index("biome")
    p365_o = p365_idx.reindex(order).reset_index()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 7.4),
                                    gridspec_kw={"wspace": 0.50})
    fig.patch.set_facecolor("white")
    for ax in (axL, axR): ax.set_facecolor("white")

    y = np.arange(len(order))
    colors = [biome_color(b) for b in order]

    # Independent x-limits per panel — each shows its own dynamic range
    xmax_L = max(abs(inat_o["delta_main"]).max(), 1e-7) * 1000 * 1.35
    xmax_R = max(abs(p365_o["delta_main"].fillna(0)).max(), 1e-7) * 1000 * 1.35

    # LEFT — iNaturalist species (stratified Δ within iconic-taxon)
    axL.barh(y, inat_o["delta_main"]*1000, color=colors,
              edgecolor="#222", lw=0.5)
    axL.axvline(0, color="#666", lw=0.6)
    # Tight xlim: just enough room for sig stars on the longest bar
    inat_max = max(abs(inat_o["delta_main"]).max()*1000, 0.1)
    axL.set_xlim(-inat_max*1.18, inat_max*1.18)
    for i, r in inat_o.iterrows():
        s = sig_stars(r["p_main"])
        fdr = (not pd.isna(r["q"])) and (r["q"] < 0.05)
        if s:
            x = r["delta_main"]*1000
            axL.text(x + (inat_max*0.025 if x>=0 else -inat_max*0.025), i, s,
                     color="#b8860b" if fdr else "#888",
                     fontsize=10, fontweight="bold" if fdr else "normal",
                     va="center", ha="left" if x>=0 else "right")
    axL.set_yticks(y)
    axL.set_yticklabels([short_biome(b) for b in order],
                        color="#222", fontsize=9)
    axL.set_xlabel(r"within-iconic-taxon stratified $\Delta$ ($\times 10^{-3}$)",
                   color="#222", fontsize=9.5)
    axL.set_title(
        f"A. iNaturalist species photos  ·  47,478 imgs · 2,158 motifs\n"
        f"stratified $\\mu\\Delta$ = "
        f"{inat['delta_main'].mean()*1000:+.3f} $\\times 10^{{-3}}$",
        color="#111", fontsize=10.5, loc="left", pad=14)
    axL.tick_params(colors="#222", labelsize=8)
    for sp in axL.spines.values(): sp.set_color("#bbb")
    axL.spines["top"].set_visible(False); axL.spines["right"].set_visible(False)

    # RIGHT — Places365 landscape scenes (marginal — no taxa to stratify on)
    deltas = p365_o["delta_main"].fillna(np.nan).values * 1000
    bars_y = y[~np.isnan(deltas)]
    bars_v = deltas[~np.isnan(deltas)]
    bars_c = [colors[i] for i in range(len(colors)) if not np.isnan(deltas[i])]
    axR.barh(bars_y, bars_v, color=bars_c, edgecolor="#222", lw=0.5)
    axR.axvline(0, color="#666", lw=0.6)
    p365_max = max(abs(p365_o["delta_main"].fillna(0)).max()*1000, 0.1)
    axR.set_xlim(-p365_max*1.18, p365_max*1.18)
    for i, r in p365_o.iterrows():
        if pd.isna(r["delta_main"]):
            axR.text(0, i, "— not in Places365 —",
                      color="#aaa", fontsize=7.5, va="center", ha="center",
                      style="italic")
            continue
        s = sig_stars(r["p_main"])
        fdr = (not pd.isna(r["q"])) and (r["q"] < 0.05)
        if s:
            x = r["delta_main"]*1000
            axR.text(x + (p365_max*0.025 if x>=0 else -p365_max*0.025), i, s,
                     color="#b8860b" if fdr else "#888",
                     fontsize=10, fontweight="bold" if fdr else "normal",
                     va="center", ha="left" if x>=0 else "right")
    axR.set_yticks(y); axR.set_yticklabels([])
    axR.set_xlabel(r"marginal residualised $\Delta$ ($\times 10^{-3}$)",
                   color="#222", fontsize=9.5)
    axR.set_title(
        f"B. Places365 landscape scenes  ·  1,675 imgs · 2,158 motifs\n"
        f"marginal $\\mu\\Delta$ = "
        f"{p365['delta_main'].mean()*1000:+.3f} $\\times 10^{{-3}}$",
        color="#111", fontsize=10.5, loc="left", pad=14)
    axR.tick_params(colors="#222", labelsize=8)
    for sp in axR.spines.values(): sp.set_color("#bbb")
    axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)
    # Invert y so largest-Δ biome sits on top
    for ax in (axL, axR): ax.invert_yaxis()

    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.18, right=0.97)
    out = FIG / "fig2_biome_bars.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# FIG 5 — world map of effect strength, iNat stratified
# --------------------------------------------------------------------------- #
def fig5_earth_map():
    """Two-panel earth map: full LLM-clean (left, matches fig2) and
    Spec A subset (right, breadth-restricted). Both use YlOrRd
    sequential colormap (most stratified Δ values are positive); the
    rare negative cells are rendered in muted slate-blue. Negative-Δ
    and sub-confidence cells are clipped to a light grey on the map
    body but their colour bar stub is kept for clarity."""
    import geopandas as gpd

    MIN_MOTIFS = 50
    MIN_TRADITIONS = 10
    MIN_IMAGES = 50

    # Full LLM-clean csv: built-in stratified Δ across 14 biomes
    df_full = pd.read_csv(
        EMB / "inat_basic/v3_biome_test_sentpool_resid.csv")
    trad = pd.read_parquet(MAP / "traditions.parquet")
    n_trad_per_biome = trad.groupby("biome_wwf").size().to_dict()

    def build_biome_data(df):
        bd = {}
        for _, r in df.iterrows():
            b = r["biome"]
            bd[b] = {
                "delta": float(r["delta_strat"])
                  if not pd.isna(r["delta_strat"]) else float(r["delta"]),
                "p": float(r["p_strat"]) if not pd.isna(r["p_strat"])
                       else float(r["p_one_sided"]),
                "n_imgs": int(r["n_imgs"]),
                "n_motifs": int(r["n_motifs_in_biome"]),
                "n_trad": int(n_trad_per_biome.get(b, 0)),
            }
        return bd

    bd_full = build_biome_data(df_full)
    # Spec A subset: read breadth csv, filter to breadth == 'SpecA'
    br = pd.read_csv(EMB / "v3_breadth_sentpool_iNat.csv")
    df_specA = br[br["breadth"] == "SpecA"].rename(
        columns={"p_one_sided": "p_one_sided"})
    bd_spec = build_biome_data(df_specA)

    shp = ROOT / "raw_downloads/Ecoregions2017/Ecoregions2017.shp"
    eco = gpd.read_file(shp).to_crs("EPSG:4326")

    # Shared vmax across panels so colors are comparable
    shown_vals = []
    for bd in [bd_full, bd_spec]:
        for v in bd.values():
            if (v["n_trad"] >= MIN_TRADITIONS
                  and v["n_motifs"] >= MIN_MOTIFS
                  and v["n_imgs"] >= MIN_IMAGES
                  and v["delta"] > 0):
                shown_vals.append(v["delta"])
    vmax = float(max(shown_vals)) * 1.05 if shown_vals else 0.0015

    fig = plt.figure(figsize=(19, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[40, 40, 1.0],
                          wspace=0.05)
    ax_full = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    fig.patch.set_facecolor("white")
    for ax in (ax_full, ax_spec): ax.set_facecolor("#f4f6f8")

    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    NEG_COL = "#7a98a6"        # muted slate-blue for negative cells
    GREY = "#d6d9de"           # light grey for sub-confidence cells

    def render_panel(ax, bd, title):
        for biome_name, group in eco.groupby("BIOME_NAME"):
            info = bd.get(biome_name)
            if (info is None
                or info["n_trad"] < MIN_TRADITIONS
                or info["n_motifs"] < MIN_MOTIFS
                or info["n_imgs"] < MIN_IMAGES):
                color = GREY
            elif info["delta"] < 0:
                color = NEG_COL
            else:
                base = np.array(cmap(norm(info["delta"])))
                confidence = 1.0 - min(info["p"], 0.5)  # ∈ [0.5, 1.0]
                bg = np.array(mcolors.to_rgba("#ffffff"))
                base = base * confidence + bg * (1 - confidence)
                color = tuple(base)
            group.plot(ax=ax, color=color, edgecolor="#ffffff",
                        linewidth=0.15)
        ax.scatter(trad["lon"], trad["lat"], s=3.0, color="#222",
                    alpha=0.50, edgecolors="none", zorder=4)
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#ccc")
        ax.set_title(title, color="#111", fontsize=11.5, pad=6)

    render_panel(ax_full, bd_full,
                  "A. Full LLM-clean corpus  ·  stratified $\\Delta$  "
                  "(matches Fig 2)")
    render_panel(ax_spec, bd_spec,
                  "B. Spec A subset (motifs touching $\\leq 3$ biomes)  ·  "
                  "stratified $\\Delta$")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="vertical")
    cbar.set_label(r"stratified $\Delta$", color="#222",
                    labelpad=8, fontsize=9)
    cbar.ax.tick_params(color="#666", labelsize=7.5)
    cbar.outline.set_edgecolor("#bbb")

    # Compact legend strip below the maps
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=NEG_COL, edgecolor="#666",
              label=r"negative $\Delta$"),
        Patch(facecolor=GREY, edgecolor="#666",
              label="sub-confidence"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
                ncol=2, frameon=False, fontsize=8,
                bbox_to_anchor=(0.42, -0.04))

    out = FIG / "fig5_earth_map.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# FIG 11 — breadth gradient: Spec A → Semi → Universal
# --------------------------------------------------------------------------- #
def fig11_breadth():
    br = pd.read_csv(EMB / "v3_breadth_sentpool_iNat.csv")

    fig = plt.figure(figsize=(13.5, 8))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.55)
    ax_summary = fig.add_subplot(gs[0])
    ax_per = fig.add_subplot(gs[1])

    # Top panel: stratified μΔ across 3 breadth groups (stratified only)
    ax_summary.set_facecolor("white")
    groups = ["SpecA", "Semi", "Universal"]
    glabel = {"SpecA": "Spec A\n(≤ 3 biomes)",
              "Semi": "Semi-universal\n(4–7 biomes)",
              "Universal": "Universal\n(≥ 8 biomes)"}
    x = np.arange(len(groups))
    means_s = []
    for g in groups:
        sub = br[br["breadth"] == g]
        means_s.append(sub["delta_strat"].mean()*1000)
    bar_cols = ["#7cbe5e", "#5b8db8", "#cf6f3f"]
    ax_summary.bar(x, means_s, 0.55, color=bar_cols,
                    edgecolor="#222", lw=0.5)
    for i, ms in enumerate(means_s):
        ax_summary.text(i, ms + 0.02, f"{ms:+.2f}",
                         color="#111", ha="center", va="bottom",
                         fontsize=11, fontweight="bold")
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels([glabel[g] for g in groups],
                                color="#222", fontsize=10)
    ax_summary.axhline(0, color="#666", lw=0.5)
    ax_summary.set_ylabel(r"stratified $\mu\Delta$ across 14 biomes "
                          r"($\times 10^{-3}$)",
                          color="#222", fontsize=9.5)
    ax_summary.set_title(
        "A. Breadth gradient   ·   specific motifs couple tightest, universal motifs fade",
        color="#111", fontsize=11, loc="left", fontweight="bold", pad=14)
    ax_summary.tick_params(colors="#222", labelsize=8)
    for sp in ax_summary.spines.values(): sp.set_color("#bbb")
    ax_summary.spines["top"].set_visible(False)
    ax_summary.spines["right"].set_visible(False)

    # Bottom panel: per-biome stratified Δ, faceted by breadth group
    ax_per.set_facecolor("white")
    biome_order = (br.groupby("biome")["delta_strat"].mean()
                    .sort_values(ascending=False).index.tolist())
    bw2 = 0.27
    colors_g = {"SpecA": "#7cbe5e", "Semi": "#5b8db8", "Universal": "#cf6f3f"}
    for i, g in enumerate(groups):
        sub = br[br["breadth"] == g].set_index("biome")
        vals = [sub.loc[b, "delta_strat"]*1000 if b in sub.index else 0
                 for b in biome_order]
        xpos = np.arange(len(biome_order)) + (i - 1)*bw2
        ax_per.bar(xpos, vals, bw2, color=colors_g[g], label=g,
                    edgecolor="#222", lw=0.4)
    ax_per.axhline(0, color="#666", lw=0.5)
    ax_per.set_xticks(np.arange(len(biome_order)))
    ax_per.set_xticklabels([short_biome(b) for b in biome_order],
                            color="#222", rotation=40, ha="right",
                            fontsize=8)
    ax_per.set_ylabel(r"stratified $\Delta$ ($\times 10^{-3}$)",
                      color="#222", fontsize=9.5)
    ax_per.set_title(
        "B. Per-biome stratified $\\Delta$, by motif breadth",
        color="#111", fontsize=11, loc="left", fontweight="bold", pad=14)
    ax_per.legend(facecolor="white", edgecolor="#aaa",
                   fontsize=9, loc="upper right")
    ax_per.tick_params(colors="#222", labelsize=8)
    for sp in ax_per.spines.values(): sp.set_color("#bbb")
    ax_per.spines["top"].set_visible(False)
    ax_per.spines["right"].set_visible(False)

    out = FIG / "fig11_universals_analysis.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# FIG 9 — cross-model correlation, per-biome stratified Δ
# --------------------------------------------------------------------------- #
def fig9_crossmodel():
    sig = pd.read_csv(EMB / "inat_basic/v3_biome_test_sentpool_resid.csv")[
        ["biome", "delta_strat"]].rename(columns={"delta_strat": "SigLIP-2 (sent-pool)"})
    mc = pd.read_csv(EMB_ALL / "mclip/biome_test_llm_clean_stratified.csv")[
        ["biome", "delta_strat"]].rename(columns={"delta_strat": "M-CLIP"})
    la = pd.read_csv(EMB_ALL / "openclip_laion2b/biome_test_llm_clean_stratified.csv")[
        ["biome", "delta_strat"]].rename(columns={"delta_strat": "OpenCLIP-LAION-2B"})
    oa = pd.read_csv(EMB_ALL / "openclip_openai/biome_test_llm_clean_stratified.csv")[
        ["biome", "delta_strat"]].rename(columns={"delta_strat": "OpenCLIP-OpenAI"})
    df = sig.merge(mc, on="biome").merge(la, on="biome").merge(oa, on="biome")

    models = ["SigLIP-2 (sent-pool)", "M-CLIP",
              "OpenCLIP-LAION-2B", "OpenCLIP-OpenAI"]
    n = len(models)
    fig, axes = plt.subplots(n, n, figsize=(11, 11))
    fig.patch.set_facecolor("white")
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.set_facecolor("white")
            if i == j:
                ax.text(0.5, 0.5, models[i], ha="center", va="center",
                         fontsize=9, color="#222", fontweight="bold",
                         transform=ax.transAxes,
                         bbox=dict(boxstyle="round,pad=0.4", fc="#eef3f8",
                                    ec="#aabacf"))
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(False)
                continue
            x = df[models[j]].values * 1000
            y = df[models[i]].values * 1000
            colors = [biome_color(b) for b in df["biome"]]
            ax.scatter(x, y, c=colors, s=40, edgecolor="#222", lw=0.4)
            lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
            pad = (hi - lo) * 0.1
            ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--",
                     color="#999", lw=0.7)
            ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)
            ax.axhline(0, color="#ccc", lw=0.4)
            ax.axvline(0, color="#ccc", lw=0.4)
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.04, 0.94, f"r = {r:+.2f}", transform=ax.transAxes,
                     fontsize=8.5, color="#222", fontweight="bold",
                     va="top", ha="left",
                     bbox=dict(boxstyle="round,pad=0.25",
                                fc="#f8f8f8", ec="#aaa"))
            if i == n-1:
                ax.set_xlabel(f"$\\Delta$ ×10⁻³", fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"$\\Delta$ ×10⁻³", fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
            for sp in ax.spines.values(): sp.set_color("#bbb")
    fig.subplots_adjust(top=0.98, hspace=0.15, wspace=0.15,
                        left=0.07, right=0.97, bottom=0.07)
    out = FIG / "fig9_crossmodel_correlation.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# FIG v2 controls — 4-panel robustness atlas
# --------------------------------------------------------------------------- #
def fig_v2_controls():
    fig = plt.figure(figsize=(15, 14.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1],
                          hspace=1.05, wspace=0.32)
    fig.patch.set_facecolor("white")
    ax_A = fig.add_subplot(gs[0, :])
    ax_B = fig.add_subplot(gs[1, 0])
    ax_C = fig.add_subplot(gs[1, 1])
    ax_D = fig.add_subplot(gs[2, :])

    # ---- Panel A — content vs structure (per-biome) on STRATIFIED Δ ----
    headline = pd.read_csv(
        EMB / "inat_basic/v3_biome_test_sentpool_resid.csv"
    )[["biome", "delta_strat"]].rename(
        columns={"delta_strat": "Original sentence-pool"})
    shuf_csv = EMB / "v2_R5a_sentpool_shuffled_biome_test_stratified.csv"
    wiki_csv = EMB / "v2_R5b_wiki_biome_test_stratified.csv"
    shuf = pd.read_csv(shuf_csv)[["biome", "delta_strat"]].rename(
        columns={"delta_strat": "Word-shuffled sentence-pool"})
    wiki = pd.read_csv(wiki_csv)[["biome", "delta_strat"]].rename(
        columns={"delta_strat": "Encyclopedic null (Wikipedia)"})
    df = headline.merge(shuf, on="biome").merge(wiki, on="biome", how="left")
    biomes = df.sort_values("Original sentence-pool", ascending=False)["biome"].tolist()
    df = df.set_index("biome").reindex(biomes)
    cols = ["Original sentence-pool", "Word-shuffled sentence-pool",
            "Encyclopedic null (Wikipedia)"]

    ax_A.set_facecolor("white")
    x = np.arange(len(df)); bw = 0.27
    palette = ["#2f6391", "#9bbdd6", "#bbbbbb"]
    for i, c in enumerate(cols):
        ax_A.bar(x + (i-1)*bw, df[c].values*1000, bw, label=c,
                  color=palette[i], edgecolor="#222", lw=0.4)
    ax_A.axhline(0, color="#888", lw=0.6)
    ax_A.set_xticks(x); ax_A.set_xticklabels(
        [short_biome(b) for b in df.index], rotation=45, ha="right", fontsize=8)
    ax_A.set_ylabel(r"stratified $\Delta$ ($\times 10^{-3}$)", fontsize=9)
    ax_A.set_title(
        "A. Content vs structure   ·   Word-shuffled motifs ≈ original "
        "(bag-of-words sufficient); encyclopedic null collapses to ~0",
        loc="left", fontsize=10.5, fontweight="bold", pad=12)
    ax_A.legend(fontsize=8, loc="upper right", frameon=True)
    ax_A.tick_params(labelsize=8)
    for sp in ax_A.spines.values(): sp.set_color("#aaa")
    ax_A.spines["top"].set_visible(False); ax_A.spines["right"].set_visible(False)

    # ---- Panel B — two-corpus replication scatter ----
    # iNat stratified (the headline statistic) vs Places365 marginal (P365
    # has no taxa to stratify on).
    inat = pd.read_csv(EMB / "inat_basic/v3_biome_test_sentpool_resid.csv")[
        ["biome", "delta_strat"]].rename(columns={"delta_strat": "delta_inat"})
    p365 = pd.read_csv(EMB / "places365_strict/v3_biome_test_sentpool_resid.csv")[
        ["biome", "delta"]].rename(columns={"delta": "delta_p365"})
    rep = inat.merge(p365, on="biome")  # 11 shared biomes
    ax_B.set_facecolor("white")
    xs = rep["delta_inat"].values * 1000
    ys = rep["delta_p365"].values * 1000
    colors = [biome_color(b) for b in rep["biome"]]
    ax_B.scatter(xs, ys, c=colors, s=70, edgecolor="#222", lw=0.5)
    lo = min(xs.min(), ys.min()); hi = max(xs.max(), ys.max())
    pad = (hi - lo) * 0.15
    ax_B.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="#999", lw=0.7,
               label="y = x")
    ax_B.set_xlim(lo-pad, hi+pad); ax_B.set_ylim(lo-pad, hi+pad)
    ax_B.axhline(0, color="#ccc", lw=0.4); ax_B.axvline(0, color="#ccc", lw=0.4)
    r = np.corrcoef(xs, ys)[0, 1]
    ax_B.text(0.04, 0.94,
              f"Pearson r = {r:+.2f}\n"
              f"$\\mu\\Delta_{{iNat,strat}}$ = {xs.mean():+.2f}\n"
              f"$\\mu\\Delta_{{P365,marg}}$ = {ys.mean():+.2f}",
              transform=ax_B.transAxes, fontsize=8.5, color="#222",
              fontweight="bold", va="top", ha="left",
              bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#aaa"))
    ax_B.set_xlabel(r"iNaturalist (stratified): $\Delta$ ($\times 10^{-3}$)",
                    fontsize=9)
    ax_B.set_ylabel(r"Places365 (marginal): $\Delta$ ($\times 10^{-3}$)",
                    fontsize=9)
    ax_B.set_title(
        "B. Two-corpus replication   ·   "
        "alignment holds on independent landscape photos",
        loc="left", fontsize=10.5, fontweight="bold", pad=12)
    ax_B.tick_params(labelsize=8)
    for sp in ax_B.spines.values(): sp.set_color("#aaa")
    ax_B.spines["top"].set_visible(False); ax_B.spines["right"].set_visible(False)

    # ---- Panel C — anonymisation audit (200 motifs, unchanged) ----
    ax_C.set_facecolor("white")
    df_aud = pd.read_csv(
        ROOT / "dataset/imagery/figures/v2_R7_audit_perrow.csv")
    total = len(df_aud)
    rows = [
        ("biome_word",       "Explicit biome word",                "#3a9d50"),
        ("place_residue",    "Place / ethnonym residue",           "#3a9d50"),
        ("species_residue",  "Species residue (genus / common)",   "#d9a73a"),
        ("generic_landform", "Generic landform (kept by design)",  "#9999bb"),
        ("activity",         "Activity vocabulary (kept by design)", "#9999bb"),
    ]
    labels, vals, cols2 = [], [], []
    for k, name, color in rows:
        n_v = int(df_aud[k].sum())
        labels.append(f"{name}\n({n_v}/{total})")
        vals.append(100 * n_v / total)
        cols2.append(color)
    y = np.arange(len(labels))
    ax_C.barh(y, vals, color=cols2, edgecolor="#222", lw=0.5)
    for i, v in enumerate(vals):
        ax_C.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    ax_C.set_yticks(y); ax_C.set_yticklabels(labels, fontsize=8)
    ax_C.invert_yaxis(); ax_C.set_xlim(0, 110)
    ax_C.set_xlabel("motifs with residue (%, n=200)", fontsize=8)
    ax_C.set_title("C. Anonymisation audit on 200 random motifs",
                    loc="left", fontsize=10.5, fontweight="bold", pad=12)
    ax_C.tick_params(labelsize=8)
    for sp in ax_C.spines.values(): sp.set_color("#aaa")
    ax_C.spines["top"].set_visible(False); ax_C.spines["right"].set_visible(False)

    # ---- Panel D — cross-model headline (stratified only) ----
    sig_m = pd.read_csv(EMB / "inat_basic/v3_biome_test_sentpool_resid.csv")
    sig_st = sig_m["delta_strat"].mean()*1000
    mc = pd.read_csv(EMB_ALL / "mclip/biome_test_llm_clean_stratified.csv")
    la = pd.read_csv(EMB_ALL / "openclip_laion2b/biome_test_llm_clean_stratified.csv")
    oa = pd.read_csv(EMB_ALL / "openclip_openai/biome_test_llm_clean_stratified.csv")
    mc_st = mc["delta_strat"].mean()*1000
    la_st = la["delta_strat"].mean()*1000
    oa_st = oa["delta_strat"].mean()*1000
    rows = [
        ("SigLIP-2 (sent-pool)", sig_st),
        ("M-CLIP (XLM-R + ViT)", mc_st),
        ("OpenCLIP-LAION-2B",     la_st),
        ("OpenCLIP-OpenAI",       oa_st),
    ]
    rows = sorted(rows, key=lambda r: -r[1])
    labels = [n for (n, _) in rows]
    vals_s = [s for (_, s) in rows]
    ax_D.set_facecolor("white")
    y = np.arange(len(labels))
    ax_D.barh(y, vals_s, 0.55, color="#d9a73a",
               edgecolor="#222", lw=0.5)
    for i, s in enumerate(vals_s):
        ax_D.text(s + 0.015 if s>0 else s - 0.015, i, f"{s:+.2f}",
                   va="center", ha="left" if s>0 else "right",
                   fontsize=10, color="#111", fontweight="bold")
    ax_D.axvline(0, color="#888", lw=0.6)
    ax_D.set_yticks(y); ax_D.set_yticklabels(labels, fontsize=9.5)
    ax_D.invert_yaxis()
    ax_D.set_xlabel(
        r"stratified $\mu\Delta$ across 14 biomes ($\times 10^{-3}$), "
        r"LLM-clean motif text",
        fontsize=9)
    ax_D.set_title(
        "D. Four-model robustness   ·   stratified $\\Delta$ is positive across all four "
        "vision-language models",
        loc="left", fontsize=10.5, fontweight="bold", pad=12)
    ax_D.tick_params(labelsize=8)
    for sp in ax_D.spines.values(): sp.set_color("#aaa")
    ax_D.spines["top"].set_visible(False); ax_D.spines["right"].set_visible(False)

    out = FIG / "fig_v2_controls.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# FIG S — Biome-tell tests (high-tell vs low-tell split + Glottolog swap null)
# --------------------------------------------------------------------------- #
def figS_biome_tell():
    split = pd.read_csv(EMB / "v3_biome_tell_split.csv")
    swap = pd.read_csv(EMB / "v3_glottolog_swap_null.csv")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 7),
                                    gridspec_kw={"wspace": 0.65})
    fig.patch.set_facecolor("white")

    # ---- Panel A — per-biome stratified Δ on high-tell vs low-tell halves
    axA.set_facecolor("white")
    biome_order = (split[split.half == "high_tell"]
                    .sort_values("delta_strat", ascending=False)["biome"]
                    .tolist())
    y = np.arange(len(biome_order))
    bw = 0.38
    low = split[split.half == "low_tell"].set_index("biome").reindex(biome_order)
    high = split[split.half == "high_tell"].set_index("biome").reindex(biome_order)
    axA.barh(y - bw/2, high["delta_strat"]*1000, bw,
              color="#d9a73a", edgecolor="#222", lw=0.4,
              label=r"high-tell (text-side biome-recoverable)")
    axA.barh(y + bw/2, low["delta_strat"]*1000, bw,
              color="#5b8db8", edgecolor="#222", lw=0.4,
              label=r"low-tell (text-side biome-uninformative)")
    axA.axvline(0, color="#666", lw=0.6)

    # Add significance markers (stars) for each bar
    xmax_A = max(abs(high["delta_strat"]).max(),
                  abs(low["delta_strat"]).max()) * 1000
    for i, b in enumerate(biome_order):
        for half, off, color_t in [(high, -bw/2, "#a47b1f"),
                                     (low, +bw/2, "#2f6391")]:
            r = half.loc[b]
            s = sig_stars(r["p_strat"])
            if not s: continue
            x = r["delta_strat"] * 1000
            axA.text(x + (xmax_A*0.02 if x >= 0 else -xmax_A*0.02),
                      i + off, s,
                      color=color_t, fontsize=9, fontweight="bold",
                      va="center", ha="left" if x >= 0 else "right")

    axA.set_yticks(y)
    axA.set_yticklabels([short_biome(b) for b in biome_order],
                        color="#222", fontsize=8.5)
    axA.invert_yaxis()
    axA.set_xlabel(r"stratified $\Delta$ ($\times 10^{-3}$)",
                    color="#222", fontsize=9.5)
    n_sig_high = int((high["p_strat"] < 0.05).sum())
    n_sig_low = int((low["p_strat"] < 0.05).sum())
    axA.set_title(
        f"A. High-tell vs low-tell $\\Delta$ split\n"
        f"high-tell $\\mu\\Delta$ = "
        f"{high['delta_strat'].mean()*1000:+.3f} "
        f"({n_sig_high}/14 sig p<.05)   ·   "
        f"low-tell $\\mu\\Delta$ = "
        f"{low['delta_strat'].mean()*1000:+.3f} "
        f"({n_sig_low}/14 sig p<.05)",
        color="#111", fontsize=10.5, fontweight="bold", loc="left", pad=10)
    axA.legend(loc="lower right", fontsize=8.5, frameon=True)
    axA.tick_params(colors="#222", labelsize=8)
    for sp in axA.spines.values(): sp.set_color("#bbb")
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    # widen xlim to leave room for stars
    axA.set_xlim(axA.get_xlim()[0] * 1.15, axA.get_xlim()[1] * 1.15)

    # ---- Panel B — Glottolog within-macroarea biome-swap null
    axB.set_facecolor("white")
    swap_sorted = swap.sort_values("delta_strat_observed", ascending=False).reset_index(drop=True)
    yb = np.arange(len(swap_sorted))
    # Plot observed Δ as colored bars
    colors_b = [biome_color(b) for b in swap_sorted["biome"]]
    # Error bar (1.96 * null_std on each side, around null_mean)
    null_mean = swap_sorted["null_mean"].values * 1000
    null_std = swap_sorted["null_std"].values * 1000
    obs = swap_sorted["delta_strat_observed"].values * 1000
    # Light grey horizontal strip = null distribution (mean ± 1.96σ)
    axB.barh(yb, 2 * 1.96 * null_std,
              left=null_mean - 1.96 * null_std,
              color="#dddddd", edgecolor="#aaa", lw=0.3,
              label="swap null, 95% range")
    # Observed Δ markers
    for i, (o, c) in enumerate(zip(obs, colors_b)):
        sig = swap_sorted.iloc[i]["p_swap_null"] < 0.05
        axB.scatter(o, yb[i], s=120 if sig else 60,
                     c=c, edgecolors="#111",
                     linewidths=1.0 if sig else 0.5,
                     marker="o" if sig else "s",
                     zorder=4)
    axB.axvline(0, color="#666", lw=0.5)
    axB.set_yticks(yb)
    axB.set_yticklabels([short_biome(b) for b in swap_sorted["biome"]],
                        color="#222", fontsize=8.5)
    axB.invert_yaxis()
    axB.set_xlabel(r"stratified $\Delta$ ($\times 10^{-3}$)",
                    color="#222", fontsize=9.5)
    axB.set_title(
        "B. Within-Glottolog-macroarea biome-swap null\n"
        "6 / 14 biomes survive at $p < .05$ (circle = sig, "
        "square = n.s.)",
        color="#111", fontsize=11, fontweight="bold", loc="left", pad=10)
    axB.legend(loc="lower right", fontsize=8, frameon=True)
    axB.tick_params(colors="#222", labelsize=8)
    for sp in axB.spines.values(): sp.set_color("#bbb")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    out = FIG / "figS_biome_tell.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Run all
# --------------------------------------------------------------------------- #
def main():
    print("== building v3 figures ==", flush=True)
    fig2_biome_bars()
    fig11_breadth()
    fig9_crossmodel()
    fig_v2_controls()
    fig5_earth_map()
    figS_biome_tell()
    print("== done ==", flush=True)


if __name__ == "__main__":
    main()
