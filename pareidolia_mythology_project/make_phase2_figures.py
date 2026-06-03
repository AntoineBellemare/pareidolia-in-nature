"""
make_phase2_figures.py — render the headline figures from the phase-2 run.

Reads:
  dataset/mapping_v2/traditions.parquet
  dataset/imagery/inaturalist/manifest.parquet
  dataset/imagery/embeddings/biome_test_all.csv
  dataset/imagery/embeddings/biome_test_all_byTaxon.csv
  dataset/imagery/embeddings/biome_test_all_creature_like.csv
  dataset/imagery/embeddings/tradition_test_all.csv
  dataset/imagery/embeddings/img_paths.parquet  (for taxon balance)

Writes PNGs to dataset/imagery/figures/.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent
MAP = ROOT / "dataset/mapping_v2"
IMG = ROOT / "dataset/imagery"
EMB = IMG / "embeddings"
FIG = IMG / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 14 WWF biomes — a colour-blind-friendly cycle
BIOME_COLORS = {
    "Tropical & Subtropical Moist Broadleaf Forests": "#0d7a3a",
    "Tropical & Subtropical Dry Broadleaf Forests": "#7cae3b",
    "Tropical & Subtropical Coniferous Forests": "#3b8eae",
    "Temperate Broadleaf & Mixed Forests": "#46b06b",
    "Temperate Conifer Forests": "#15677a",
    "Boreal Forests/Taiga": "#2d4f7a",
    "Tropical & Subtropical Grasslands, Savannas & Shrublands": "#e0a83a",
    "Temperate Grasslands, Savannas & Shrublands": "#c2914b",
    "Flooded Grasslands & Savannas": "#5db7c3",
    "Montane Grasslands & Shrublands": "#9e6cb4",
    "Tundra": "#b9c6cf",
    "Mediterranean Forests, Woodlands & Scrub": "#c9603a",
    "Deserts & Xeric Shrublands": "#e7c878",
    "Mangroves": "#357055",
}


def biome_color(b):
    return BIOME_COLORS.get(b, "#888888")


def short_biome(b):
    if not isinstance(b, str):
        return str(b)
    return (b
            .replace("Tropical & Subtropical ", "Trop. ")
            .replace("Temperate ", "Temp. ")
            .replace(" & ", " & ")
            .replace("Mixed Forests", "Mixed Fst")
            .replace("Broadleaf Forests", "Broadleaf Fst")
            .replace("Conifer Forests", "Conifer Fst")
            .replace("Coniferous Forests", "Coniferous Fst")
            .replace("Forests/Taiga", "Fst/Taiga")
            .replace(", Savannas & Shrublands", ", Sav. & Shrub.")
            .replace(" & Shrublands", " & Shrub.")
            .replace(" & Scrub", " & Scrub")
            .replace("Woodlands & Scrub", "Woodl. & Scrub")
            .replace("Xeric Shrublands", "Xeric Shrub.")
            )


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# --------------------------------------------------------------------------- #
# FIG 1 — world map of 958 traditions, coloured by WWF biome
# --------------------------------------------------------------------------- #

def fig1_world_traditions():
    trad = pd.read_parquet(MAP / "traditions.parquet")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_facecolor("white")

    # Continent layer: WWF Ecoregions polygons coloured by their biome.
    # This shows the geographic distribution of biomes, so the reader can
    # see how Berezkin traditions (overlaid as points) sample each biome's
    # actual geographic extent. Polygons are rendered with low alpha so the
    # tradition dots remain clearly visible.
    try:
        import geopandas as gpd
        shp = ROOT / "raw_downloads/Ecoregions2017/Ecoregions2017.shp"
        if shp.exists():
            eco = gpd.read_file(shp).to_crs("EPSG:4326")
            # Polygons coloured by their WWF biome
            eco["_color"] = eco["BIOME_NAME"].apply(
                lambda b: biome_color(b) if isinstance(b, str) else "#dddddd"
            )
            eco.plot(ax=ax, color=eco["_color"], alpha=0.45,
                     edgecolor="white", linewidth=0.08, zorder=0)
            # Faint coastline / continent outline on top so the geography
            # reads clearly without dominating the biome colours
            eco.dissolve().boundary.plot(
                ax=ax, color="#888", linewidth=0.4, zorder=1, alpha=0.55,
            )
    except Exception as e:
        print(f"  [fig1] continent shapefile unavailable: {e}")

    biomes_sorted = (
        trad.groupby("biome_wwf").size().sort_values(ascending=False).index
    )
    for b in biomes_sorted:
        sub = trad[trad["biome_wwf"] == b]
        ax.scatter(sub["lon"], sub["lat"], s=16, alpha=0.95,
                   c=biome_color(b),
                   label=f"{short_biome(b)} (n={len(sub)})",
                   edgecolors="#222", linewidth=0.45, zorder=3)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-60, 91, 30))
    ax.tick_params(colors="#444", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#aaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("longitude", color="#222", fontsize=9)
    ax.set_ylabel("latitude", color="#222", fontsize=9)
    leg = ax.legend(loc="lower left", fontsize=7, frameon=True,
                    facecolor="white", edgecolor="#bbb",
                    labelcolor="#222", ncol=2, columnspacing=0.6,
                    handletextpad=0.3, borderpad=0.4)
    leg.get_frame().set_linewidth(0.6)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    out = FIG / "fig1_world_traditions.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# FIG 2 — Headline biome-test bar chart with significance stars
# --------------------------------------------------------------------------- #

def fig2_biome_test():
    df = pd.read_csv(EMB / "biome_test_all.csv").sort_values("delta")
    # drop the N/A biome row from display
    df = df[df["biome"] != "N/A"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor("#11141a")
    colors = [biome_color(b) for b in df["biome"]]
    bars = ax.barh(df["biome"].map(short_biome), df["delta"], color=colors,
                   edgecolor="#222831", linewidth=0.6)
    # annotate with n & p
    for i, r in df.iterrows():
        s = sig_stars(r["p_one_sided"])
        txt = f"n={int(r['n_imgs'])}  p={r['p_one_sided']:.3f} {s}"
        x = r["delta"]
        offset = 0.0002 if x >= 0 else -0.0002
        ax.text(x + offset, i, txt,
                ha="left" if x >= 0 else "right",
                va="center", fontsize=8, color="#eaeaea")
    ax.axvline(0, color="#aaa", lw=0.8)
    ax.set_xlabel("Δ = mean cosine sim(own biome's mythemes) "
                  "− mean(other biomes')",
                  color="#bbbbbb")
    ax.set_title("Biome-level signal: tropical/montane biomes' images sit closer "
                 "to their own mythemes (1000-perm test, * p<.05  ** p<.01  *** p<.001)",
                 color="#eeeeee", fontsize=11)
    ax.tick_params(colors="#dddddd")
    for s in ax.spines.values():
        s.set_color("#444")
    fig.patch.set_facecolor("#0c0d11")
    fig.tight_layout()
    out = FIG / "fig2_biome_test_headline.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# FIG 3 — biome × iconic_taxon Δ heatmap (taxon-stratified test)
# --------------------------------------------------------------------------- #

def fig3_taxon_heatmap():
    df = pd.read_csv(EMB / "biome_test_all_byTaxon.csv")
    df = df[~df["biome"].isin(["N/A"])]
    # restrict to taxa with >1 biome tested
    taxa_keep = df.groupby("taxon_group").size()
    taxa_keep = taxa_keep[taxa_keep >= 5].index.tolist()
    df = df[df["taxon_group"].isin(taxa_keep)]
    # order biomes by their "all" delta
    all_order = (df[df["taxon_group"] == "all"]
                 .sort_values("delta", ascending=False)["biome"]
                 .tolist())
    extras = [b for b in df["biome"].unique() if b not in all_order]
    biome_order = all_order + extras
    # order taxa by their mean delta
    taxon_order = (df.groupby("taxon_group")["delta"].mean()
                   .sort_values(ascending=False).index.tolist())
    if "all" in taxon_order:
        taxon_order.remove("all"); taxon_order = ["all"] + taxon_order
    piv_delta = df.pivot_table(index="biome", columns="taxon_group",
                               values="delta").reindex(index=biome_order,
                                                       columns=taxon_order)
    piv_p = df.pivot_table(index="biome", columns="taxon_group",
                           values="p_one_sided").reindex(index=biome_order,
                                                          columns=taxon_order)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    vmax = float(np.nanmax(np.abs(piv_delta.values)))
    im = ax.imshow(piv_delta.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(taxon_order)))
    ax.set_xticklabels(taxon_order, rotation=35, ha="right", color="#dddddd")
    ax.set_yticks(range(len(biome_order)))
    ax.set_yticklabels([short_biome(b) for b in biome_order], color="#dddddd")
    # annotate with stars where significant
    for i in range(piv_delta.shape[0]):
        for j in range(piv_delta.shape[1]):
            d = piv_delta.values[i, j]
            p = piv_p.values[i, j]
            if np.isnan(d):
                ax.text(j, i, "—", ha="center", va="center",
                        color="#444", fontsize=8)
                continue
            s = sig_stars(p) if not np.isnan(p) else ""
            ax.text(j, i, s, ha="center", va="center",
                    color="#000" if abs(d) > vmax*0.5 else "#eee",
                    fontsize=9, fontweight="bold")
    ax.set_title("Δ by biome × iconic_taxon. Red = own-biome motifs are closer "
                 "than other-biome motifs (the hypothesis). "
                 "Stars: * p<.05  ** p<.01  *** p<.001",
                 color="#eeeeee", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Δ (cosine sim diff)", color="#bbbbbb")
    cbar.ax.yaxis.set_tick_params(color="#bbb")
    cbar.outline.set_edgecolor("#444")
    plt.setp(cbar.ax.get_yticklabels(), color="#bbbbbb")
    ax.tick_params(colors="#dddddd")
    fig.patch.set_facecolor("#0c0d11")
    fig.tight_layout()
    out = FIG / "fig3_biome_x_taxon_delta.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# FIG 4 — per-tradition enrichment world map
# --------------------------------------------------------------------------- #

def fig4_tradition_map():
    # prefer the residualised per-tradition test, with siglip2-large if available
    candidates = [
        EMB / "siglip2-large/tradition_test_all_resid.csv",
        EMB / "tradition_test_all_resid.csv",
        EMB / "tradition_test_all.csv",
    ]
    chosen = next((p for p in candidates if p.exists()), None)
    df = pd.read_csv(chosen)
    is_resid = "_resid" in chosen.name
    trad = pd.read_parquet(MAP / "traditions.parquet")
    df = df.merge(trad[["oid", "lat", "lon"]].rename(columns={"oid": "tradition_oid"}),
                  on="tradition_oid", how="left")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_facecolor("#11141a")
    for x in range(-180, 181, 60):
        ax.axvline(x, color="#222831", lw=0.4, zorder=0)
    for y in range(-60, 91, 30):
        ax.axhline(y, color="#222831", lw=0.4, zorder=0)
    # baseline scatter
    e = df["top50_enrichment"].clip(0, 10)
    sc = ax.scatter(df["lon"], df["lat"], c=e, cmap="magma_r",
                    s=15 + e * 4, alpha=0.85, edgecolors="none",
                    vmin=0, vmax=10, zorder=2)
    # top 10 labels
    top10 = df.sort_values("top50_enrichment", ascending=False).head(10)
    for _, r in top10.iterrows():
        ax.annotate(r["group_Berezkin"][:18], (r["lon"], r["lat"]),
                    xytext=(6, 4), textcoords="offset points",
                    color="#ffd47a", fontsize=8,
                    path_effects=[],
                    arrowprops=None)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-60, 91, 30))
    ax.tick_params(colors="#bbbbbb")
    for s in ax.spines.values():
        s.set_color("#444")
    ax.set_title(("RESIDUALISED " if is_resid else "")
                 + "Per-tradition image→myth enrichment over chance "
                 "(top-50 nearest motifs). Bigger/yellower = stronger.",
                 color="#eeeeee", fontsize=11)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("enrichment over chance (clipped at 10×)", color="#bbbbbb")
    plt.setp(cbar.ax.get_yticklabels(), color="#bbbbbb")
    cbar.outline.set_edgecolor("#444")
    fig.patch.set_facecolor("#0c0d11")
    fig.tight_layout()
    out = FIG / "fig4_tradition_enrichment_map.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# FIG 5 — biome × iconic_taxon image count heatmap (data balance)
# --------------------------------------------------------------------------- #

def fig5_taxon_balance():
    img_meta = pd.read_parquet(EMB / "img_paths.parquet")
    img_meta["use_biome"] = img_meta["photo_biome_wwf"].fillna(
        img_meta["tradition_biome_wwf"])
    ct = pd.crosstab(img_meta["use_biome"], img_meta["iconic_taxon"])
    ct = ct.drop(index=[i for i in ct.index if not isinstance(i, str)
                        or i in ("N/A","NaN")], errors="ignore")
    keep = ct.sum().sort_values(ascending=False).head(11).index
    ct = ct[keep]
    # log scale for the colour but show real counts in text
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    im = ax.imshow(np.log10(ct.values.astype(float) + 1), cmap="cividis",
                   aspect="auto")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=35, ha="right", color="#dddddd")
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels([short_biome(b) for b in ct.index], color="#dddddd")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            n = ct.values[i, j]
            ax.text(j, i, f"{int(n):,}" if n > 0 else "·",
                    ha="center", va="center", fontsize=8,
                    color="#fff" if n > 100 else "#888")
    ax.set_title("Image counts per WWF biome × iNat iconic_taxon "
                 "(colour = log10 count)",
                 color="#eeeeee", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log10(n+1)", color="#bbbbbb")
    plt.setp(cbar.ax.get_yticklabels(), color="#bbbbbb")
    cbar.outline.set_edgecolor("#444")
    ax.tick_params(colors="#dddddd")
    fig.patch.set_facecolor("#0c0d11")
    fig.tight_layout()
    out = FIG / "fig5_taxon_balance.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# FIG 6 — creature-like vs all comparison
# --------------------------------------------------------------------------- #

def fig6_creature_like_robustness():
    a = pd.read_csv(EMB / "biome_test_all.csv")[["biome","delta","p_one_sided","n_imgs"]]
    c = pd.read_csv(EMB / "biome_test_all_creature_like.csv")[["biome","delta","p_one_sided","n_imgs"]]
    a = a.rename(columns={"delta":"delta_all","p_one_sided":"p_all","n_imgs":"n_all"})
    c = c.rename(columns={"delta":"delta_creature","p_one_sided":"p_creature","n_imgs":"n_creature"})
    df = a.merge(c, on="biome")
    df = df[df["biome"].isin([b for b in df["biome"] if isinstance(b, str)
                              and b != "N/A"])]
    df = df.sort_values("delta_all")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#11141a")
    ax.barh(y - 0.2, df["delta_all"], height=0.36,
            color="#4ea36f", label="all taxa")
    ax.barh(y + 0.2, df["delta_creature"], height=0.36,
            color="#e09a2c", label="creature-like (Aves+Mammalia+Reptilia+Amphibia)")
    ax.axvline(0, color="#aaa", lw=0.8)
    # Significance stars at the tip of each bar
    for i, r in df.reset_index(drop=True).iterrows():
        sa, sc = sig_stars(r["p_all"]), sig_stars(r["p_creature"])
        if sa:
            x = r["delta_all"]
            ax.text(x + (0.00015 if x >= 0 else -0.00015), i - 0.2, sa,
                    color="#a2e9c2", fontsize=11, fontweight="bold",
                    va="center", ha="left" if x >= 0 else "right")
        if sc:
            x = r["delta_creature"]
            ax.text(x + (0.00015 if x >= 0 else -0.00015), i + 0.2, sc,
                    color="#ffcf80", fontsize=11, fontweight="bold",
                    va="center", ha="left" if x >= 0 else "right")
    ax.set_yticks(y); ax.set_yticklabels([short_biome(b) for b in df["biome"]],
                                         color="#dddddd")
    ax.set_xlabel("Δ = sim(own biome motifs) − sim(other biome motifs)   "
                  "·  stars: * p<.05  ** p<.01  *** p<.001",
                  color="#bbbbbb")
    ax.set_title("Robustness: 'creature-like' filter preserves the tropical-positive pattern",
                 color="#eeeeee", fontsize=11)
    ax.tick_params(colors="#dddddd")
    for s in ax.spines.values():
        s.set_color("#444")
    leg = ax.legend(loc="lower right", frameon=False, labelcolor="#dddddd")
    fig.patch.set_facecolor("#0c0d11")
    fig.tight_layout()
    out = FIG / "fig6_creature_like_robustness.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# FIG 8 — Faceted: plants vs animals vs vertebrates vs insects
# --------------------------------------------------------------------------- #

def fig8_taxon_groups_compared():
    """Side-by-side comparison of which taxa drive the effect."""
    csv_files = [
        ("biome_test_all.csv",            "all taxa",         "#4ea36f"),
        ("biome_test_all_plants_only.csv","plants only",      "#7cbe5e"),
        ("biome_test_all_animals_only.csv","animals (no plants/fungi)","#cf6f3f"),
        ("biome_test_all_vertebrates.csv","vertebrates",      "#d4983c"),
        ("biome_test_all_Insecta.csv",    "insects only",     "#a96cb0"),
        ("biome_test_all_Aves.csv",       "birds only",       "#76b6e5"),
        ("biome_test_all_Mammalia.csv",   "mammals only",     "#d04f6f"),
        ("biome_test_all_Reptilia.csv",   "reptiles only",    "#6abc8f"),
    ]
    dfs = []
    for path, label, color in csv_files:
        f = EMB / path
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d["biome"].apply(lambda x: isinstance(x, str)))
              & (d["biome"] != "N/A")][["biome","delta","p_one_sided","n_imgs"]].copy()
        d["facet"] = label
        d["color"] = color
        dfs.append(d)
    all_df = pd.concat(dfs, ignore_index=True)
    # order biomes by the "all" delta
    head_order = (all_df[all_df["facet"] == "all taxa"]
                  .sort_values("delta")["biome"].tolist())
    facets = [f for _, f, _ in csv_files
              if f in all_df["facet"].unique()]
    color_map = {f: c for _, f, c in csv_files}

    n_facets = len(facets)
    cols = 4
    rows = (n_facets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 2.5 + 4*rows),
                             sharey=True)
    axes = np.array(axes).reshape(rows, cols)

    for k, facet in enumerate(facets):
        ax = axes[k // cols, k % cols]
        ax.set_facecolor("#11141a")
        sub = all_df[all_df["facet"] == facet].set_index("biome").reindex(head_order)
        ypos = np.arange(len(sub))
        ax.barh(ypos, sub["delta"], color=color_map[facet],
                edgecolor="#222831", linewidth=0.5)
        ax.axvline(0, color="#aaa", lw=0.6)
        for i, (b, r) in enumerate(sub.iterrows()):
            if pd.notna(r["p_one_sided"]):
                s = sig_stars(r["p_one_sided"])
                if s:
                    x = r["delta"]
                    ax.text(x + (0.0001 if x >= 0 else -0.0001), i, s,
                            color="#ffeaa7", fontsize=9, fontweight="bold",
                            va="center", ha="left" if x >= 0 else "right")
        ax.set_title(f"{facet}", color="#eeeeee", fontsize=11)
        ax.tick_params(colors="#dddddd", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#444")
        if k % cols == 0:
            ax.set_yticks(ypos)
            ax.set_yticklabels([short_biome(b) for b in head_order],
                               fontsize=8, color="#dddddd")
        ax.axvline(0, color="#aaa", lw=0.4)

    # hide unused subplots
    for k in range(len(facets), rows * cols):
        axes[k // cols, k % cols].axis("off")

    fig.suptitle("Δ per biome, broken down by taxon group of the images   "
                 "·  positive = images closer to own biome's mythemes   "
                 "·  stars: * p<.05  ** p<.01  *** p<.001",
                 color="#eeeeee", fontsize=12, y=0.998)
    fig.patch.set_facecolor("#0c0d11")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "fig8_taxon_groups_compared.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# FIG 9 — Mean Δ across biomes, per taxon group (one bar per group)
# --------------------------------------------------------------------------- #

def fig9_taxon_summary():
    df = pd.read_csv(EMB / "biome_test_all_byTaxon.csv")
    df = df[df["biome"].apply(lambda x: isinstance(x, str)) & (df["biome"] != "N/A")]
    # Restrict to "real" biomes
    summary = (df.groupby("taxon_group")
               .agg(mean_delta=("delta", "mean"),
                    n_biomes=("delta", "size"),
                    pct_significant=("p_one_sided",
                                     lambda x: 100 * (x < 0.05).mean()),
                    pct_positive=("delta", lambda x: 100 * (x > 0).mean()))
               .reset_index()
               .sort_values("mean_delta", ascending=True))
    # Drop "all" from this taxon-only summary to keep apples-to-apples
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor("#11141a")
    colors = ["#4ea36f" if t == "all" else "#9fcdab" if d > 0 else "#cf8a7f"
              for t, d in zip(summary["taxon_group"], summary["mean_delta"])]
    y = np.arange(len(summary))
    ax.barh(y, summary["mean_delta"], color=colors,
            edgecolor="#222831", linewidth=0.5)
    ax.axvline(0, color="#aaa", lw=0.8)
    for i, r in summary.reset_index(drop=True).iterrows():
        x = r["mean_delta"]
        txt = f"  {r['pct_significant']:.0f}% sig.  ·  {r['pct_positive']:.0f}% positive  ·  {int(r['n_biomes'])} biomes"
        ax.text(x + (0.00003 if x >= 0 else -0.00003), i, txt,
                ha="left" if x >= 0 else "right", va="center",
                fontsize=9, color="#eaeaea")
    ax.set_yticks(y); ax.set_yticklabels(summary["taxon_group"], color="#dddddd")
    ax.set_xlabel("mean Δ across biomes (cosine-sim difference)", color="#bbbbbb")
    ax.set_title("Effect strength by image taxon  ·  reptiles, insects, mammals, "
                 "arachnids drive the signal; aves/plantae are flat",
                 color="#eeeeee", fontsize=11)
    ax.tick_params(colors="#dddddd")
    for s in ax.spines.values():
        s.set_color("#444")
    fig.patch.set_facecolor("#0c0d11")
    fig.tight_layout()
    out = FIG / "fig9_taxon_summary.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# FIG 7 — overview map: image points coloured by photo's WWF biome
# --------------------------------------------------------------------------- #

def fig7_image_coverage():
    m = pd.read_parquet(IMG / "inaturalist/manifest.parquet")
    m["use_biome"] = m["photo_biome_wwf"].fillna(m["tradition_biome_wwf"])
    # sample down for plotting
    s = m.sample(min(20000, len(m)), random_state=1)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_facecolor("#11141a")
    for x in range(-180, 181, 60):
        ax.axvline(x, color="#222831", lw=0.4, zorder=0)
    for y in range(-60, 91, 30):
        ax.axhline(y, color="#222831", lw=0.4, zorder=0)
    biomes_sorted = (
        s.groupby("use_biome").size().sort_values(ascending=False).index.tolist()
    )
    for b in biomes_sorted:
        sub = s[s["use_biome"] == b]
        if not isinstance(b, str): continue
        ax.scatter(sub["lon"], sub["lat"], s=2.5, alpha=0.55,
                   c=biome_color(b), edgecolors="none",
                   label=f"{short_biome(b)} (n={(m['use_biome']==b).sum()})",
                   zorder=2)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-60, 91, 30))
    ax.tick_params(colors="#bbbbbb")
    for s_ in ax.spines.values():
        s_.set_color("#444")
    ax.set_title(f"47,900 iNaturalist photos, plotted at their own coords, "
                 "colored by WWF biome (sampled to 20k for clarity)",
                 color="#eeeeee", fontsize=11)
    leg = ax.legend(loc="lower left", fontsize=7, frameon=False,
                    labelcolor="#dddddd", ncol=2, columnspacing=0.6,
                    handletextpad=0.3, borderpad=0.2)
    fig.patch.set_facecolor("#0c0d11")
    fig.tight_layout()
    out = FIG / "fig7_image_coverage.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


def main():
    fig1_world_traditions()
    fig2_biome_test()
    fig3_taxon_heatmap()
    fig4_tradition_map()
    fig5_taxon_balance()
    fig6_creature_like_robustness()
    fig7_image_coverage()
    fig8_taxon_groups_compared()
    fig9_taxon_summary()
    print(f"\nAll figures in: {FIG}")


if __name__ == "__main__":
    main()
