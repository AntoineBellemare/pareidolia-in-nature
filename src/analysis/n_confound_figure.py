"""Is the geographic effect (fig5 earth map) driven by how many myths fall
in a region?

The earth map paints each Berezkin tradition with its BIOME's stratified Δ
(delta_strat), greying out biomes below a confidence floor (>=50 motifs,
>=10 traditions, >=50 images). So "more myths per region -> stronger effect"
reduces to: is per-biome Δ correlated with that biome's myth/tradition count?

Panel A: marginal Δ vs sampling size — the confound, if present, lives here.
Panel B: stratified Δ (the statistic fig5 actually shows) vs sampling size.

A flat Panel B = the headline geography is NOT a sampling-size artefact.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for shared utils
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
MAP = ROOT / "dataset/mapping_v2"
OUT = ROOT / "paper/figures/figS_n_confound.png"

from make_phase2_figures import short_biome, biome_color

MIN_MOTIFS, MIN_TRAD, MIN_IMGS = 50, 10, 50


def main():
    d = pd.read_csv(EMB / "inat_basic/v3_biome_test_sentpool_resid.csv")
    trad = pd.read_parquet(MAP / "traditions.parquet")
    n_trad = trad.groupby("biome_wwf").size().to_dict()
    d["n_trad"] = d["biome"].map(n_trad).fillna(0).astype(int)
    # confidence floor that fig5 applies
    d["shown"] = ((d["n_motifs_in_biome"] >= MIN_MOTIFS)
                  & (d["n_trad"] >= MIN_TRAD)
                  & (d["n_imgs"] >= MIN_IMGS))

    xvar = "n_trad"                     # "number of myth traditions in the region"
    x = d[xvar].values.astype(float)
    dm = d["delta"].values * 1000       # marginal Δ
    ds = d["delta_strat"].values * 1000 # stratified Δ (fig5 statistic)

    def corr(xv, yv):
        r, p = stats.pearsonr(xv, yv)
        rho, ps = stats.spearmanr(xv, yv)
        return r, p, rho, ps

    # full 14 biomes and the shown (confidence-passing) subset
    sh = d["shown"].values
    print("=== Δ vs number of traditions per biome ===")
    for name, yv in [("marginal", dm), ("stratified", ds)]:
        r, p, rho, ps = corr(x, yv)
        rs, psr, rhos, pss = corr(x[sh], yv[sh])
        print(f"{name:11s}: all14  Pearson r={r:+.3f} p={p:.3f} | "
              f"Spearman ρ={rho:+.3f} p={ps:.3f}")
        print(f"{'':11s}  shown  Pearson r={rs:+.3f} p={psr:.3f} | "
              f"Spearman ρ={rhos:+.3f} p={pss:.3f}  (n={sh.sum()})")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    fig.patch.set_facecolor("white")

    for ax, yv, ttl, lett in [(axes[0], dm, "Marginal $\\Delta$", "A"),
                              (axes[1], ds, "Stratified $\\Delta$  (fig. 5 statistic)", "B")]:
        ax.set_facecolor("white")
        # full-sample fit
        r, p, rho, ps = corr(x, yv)
        rs, psr, rhos, pss = corr(x[sh], yv[sh])
        # scatter, grey = greyed-out on the map, coloured = shown
        for i in range(len(d)):
            b = d["biome"].iloc[i]
            col = biome_color(b) if sh[i] else "#cccccc"
            ax.scatter(x[i], yv[i], s=90, c=[col],
                       edgecolors="#333" if sh[i] else "#bbb",
                       lw=0.8, zorder=3, alpha=0.95)
        # regression line over the shown subset (what the map uses)
        if sh.sum() >= 3:
            xs = x[sh]
            m, c = np.polyfit(xs, yv[sh], 1)
            xx = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xx, m*xx + c, "--", color="#444", lw=1.4, zorder=2)
        ax.axhline(0, color="#999", lw=0.7)
        # label a few extreme biomes
        for i in range(len(d)):
            if not sh[i]:
                continue
            if (yv[i] == yv[sh].max() or yv[i] == yv[sh].min()
                    or x[i] == x[sh].max()):
                ax.annotate(short_biome(d["biome"].iloc[i]),
                            (x[i], yv[i]), fontsize=7, color="#333",
                            xytext=(6, 4), textcoords="offset points")
        ax.set_xlabel("number of myth traditions in the biome", fontsize=10)
        ax.set_ylabel(r"$\Delta$ ($\times 10^{-3}$)", fontsize=10)
        ax.set_title(f"{lett}. {ttl}\n"
                     f"shown biomes: r={rs:+.2f} (p={psr:.2f}), "
                     f"$\\rho$={rhos:+.2f} (p={pss:.2f})",
                     fontsize=11, fontweight="bold", loc="left")
        for s in ax.spines.values(): s.set_color("#ccc")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                  markeredgecolor="#333", markersize=9, label="shown on map (passes confidence floor)"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#ccc",
                  markeredgecolor="#bbb", markersize=9, label="greyed out on map (under-sampled)")]
    axes[0].legend(handles=leg, fontsize=8, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
