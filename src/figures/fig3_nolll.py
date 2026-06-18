"""Fig 3 (method convergence): biome-mythology geography under both strategies.
Left = remove the names (LLM-strip, stratified Delta, perm null); right = hold
the names constant (raw myths, discrete matched null). Each panel paints only
its significant biomes; the same regions light up.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
LAD = EMB / "ladder"
MAP = ROOT / "dataset/mapping_v2"
OUT = ROOT / "paper/figures/fig3_nolll.png"
MIN_TRAD = 10


def main():
    import geopandas as gpd
    llm = pd.read_csv(EMB / "inat_basic/v3_biome_test_sentpool_resid.csv").set_index("biome")
    mat = pd.read_csv(LAD / "nolll_perbiome_inat.csv").set_index("biome")
    trad = pd.read_parquet(MAP / "traditions.parquet")
    n_trad = trad.groupby("biome_wwf").size().to_dict()
    eco = gpd.read_file(ROOT / "raw_downloads/Ecoregions2017/Ecoregions2017.shp").to_crs("EPSG:4326")

    cmap = plt.get_cmap("YlOrRd"); GREY = "#d6d9de"; NEG = "#7a98a6"

    fig = plt.figure(figsize=(19, 6.7))
    gs = fig.add_gridspec(1, 2, wspace=0.04)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    fig.patch.set_facecolor("white")
    for ax in (axA, axB):
        ax.set_facecolor("#f4f6f8")

    def render(ax, df, dcol, pcol, title):
        shown = [b for b in df.index if n_trad.get(b, 0) >= MIN_TRAD
                 and df.loc[b, dcol] > 0 and df.loc[b, pcol] < 0.05]
        vmax = float(df.loc[shown, dcol].max()) * 1.05 if shown else 1.0
        norm = mcolors.Normalize(0, vmax)
        for bname, grp in eco.groupby("BIOME_NAME"):
            if bname not in df.index or n_trad.get(bname, 0) < MIN_TRAD:
                color = GREY
            elif df.loc[bname, dcol] < 0:
                color = NEG
            elif df.loc[bname, pcol] < 0.05:
                color = tuple(cmap(norm(df.loc[bname, dcol])))
            else:
                color = GREY
            grp.plot(ax=ax, color=color, edgecolor="#ffffff", linewidth=0.15)
        ax.scatter(trad["lon"], trad["lat"], s=3.0, color="#222", alpha=0.5,
                   edgecolors="none", zorder=4)
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#ccc")
        ax.set_title(f"{title}  ({len(shown)} biomes $p<.05$)", fontsize=11.5, pad=6)

    render(axA, llm, "delta_strat", "p_strat", "A. Remove the names (LLM-strip, stratified)")
    render(axB, mat, "delta_raw", "p_matched", "B. Hold the names constant (raw + matched null)")

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#fdae61", label="aligned (warmer = stronger, within panel)"),
                        Patch(facecolor=NEG, label=r"negative $\Delta$"),
                        Patch(facecolor=GREY, label="not significant / under-sampled")],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Same biome geography from two independent methods", fontsize=12, y=1.0)
    fig.savefig(OUT, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
