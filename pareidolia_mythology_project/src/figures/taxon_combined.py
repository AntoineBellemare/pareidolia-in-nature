"""Three-panel taxon × biome figure.

A. Heatmap with class words PRESERVED (mammal/bird/plant/fish/...).
B. Heatmap with class words COLLAPSED (animal, plant).
C. Per-taxon double-violin showing how the distribution of biome cells
   shifts under the collapse: faded violin = preserved, solid violin =
   collapsed. Wilcoxon p-values report whether each taxon's collapsed
   biome-cell distribution differs from zero.

Output: paper/figures/fig_taxon_combined.png
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]  # project root
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
OUT = ROOT / "paper/figures/fig_taxon_combined.png"

from make_phase2_figures import short_biome, sig_stars

MIN_MOTIFS = 10
TAXON_ORDER = ["all", "Plantae", "Fungi", "Animalia", "Mammalia",
                "Aves", "Reptilia", "Amphibia", "Actinopterygii",
                "Insecta", "Arachnida", "Mollusca"]

TAXON_COLOR = {
    "all": "#888888",
    "Plantae": "#7cbe5e", "Fungi": "#c2914b",
    "Animalia": "#cf6f3f", "Mammalia": "#d04f6f", "Aves": "#76b6e5",
    "Reptilia": "#6abc8f", "Amphibia": "#4ea36f",
    "Actinopterygii": "#2d8fb3", "Insecta": "#a96cb0",
    "Arachnida": "#9e6cb4", "Mollusca": "#8a6240",
}


def bh(p):
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


def prep(csv_path, biome_order=None):
    df = pd.read_csv(csv_path)
    if "n_motifs_in_biome_specific" in df.columns:
        df = df.rename(columns={"n_motifs_in_biome_specific": "n_mot"})
    df = df[df["biome"].apply(lambda x: isinstance(x, str))
            & (df["biome"] != "N/A")]
    df = df[df["n_mot"] >= MIN_MOTIFS]
    df["q"] = (df.groupby("taxon_group")["p_one_sided"]
               .transform(lambda x: bh(x.values)))
    if biome_order is None:
        biome_order = (df[df["taxon_group"] == "all"]
                        .sort_values("delta", ascending=False)
                        ["biome"].tolist())
    taxa = [t for t in TAXON_ORDER if t in df["taxon_group"].unique()]
    delta = np.full((len(biome_order), len(taxa)), np.nan)
    pval = np.full_like(delta, np.nan)
    qval = np.full_like(delta, np.nan)
    bi = {b: i for i, b in enumerate(biome_order)}
    ti = {t: i for i, t in enumerate(taxa)}
    for _, r in df.iterrows():
        if r["biome"] in bi and r["taxon_group"] in ti:
            i, j = bi[r["biome"]], ti[r["taxon_group"]]
            delta[i, j] = r["delta"]; pval[i, j] = r["p_one_sided"]
            qval[i, j] = r["q"]
    return delta, pval, qval, biome_order, taxa, df


def render_heatmap(ax, delta, pval, qval, biomes, taxa, vmax, title,
                    label_yticks, ax_cb=None):
    ax.set_facecolor("white")
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    im = ax.imshow(delta, cmap=cmap, norm=norm, aspect="auto",
                    interpolation="nearest")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            d = delta[i, j]; p = pval[i, j]; q = qval[i, j]
            if np.isnan(d):
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                            facecolor="#e5e7ea",
                                            edgecolor="white", lw=0.4))
                ax.text(j, i, "—", color="#888", fontsize=6.5,
                         ha="center", va="center"); continue
            r2, g2, b2, _ = cmap(norm(d))
            luma = 0.299 * r2 + 0.587 * g2 + 0.114 * b2
            tc = "#11141a" if luma > 0.55 else "#f4f4f4"
            s = sig_stars(p) if not np.isnan(p) else ""
            fdr = (not np.isnan(q)) and (q < 0.05)
            label = f"{d*1000:+.2f}{s}" if s else f"{d*1000:+.2f}"
            ax.text(j, i, label, color=tc, fontsize=6.5,
                     ha="center", va="center",
                     fontweight="bold" if (s and fdr) else "normal")
    ax.set_xticks(range(len(taxa)))
    ax.set_xticklabels(taxa, color="#222", fontsize=8.5, rotation=0)
    ax.tick_params(axis="x", which="both", length=0, pad=4, top=True,
                    labeltop=True, bottom=False, labelbottom=False)
    if label_yticks:
        ax.set_yticks(range(len(biomes)))
        ax.set_yticklabels([short_biome(b) for b in biomes],
                            color="#222", fontsize=8.5)
    else:
        ax.set_yticks([]); ax.set_yticklabels([])
    ax.set_xlim(-0.5, len(taxa) - 0.5)
    ax.set_ylim(len(biomes) - 0.5, -0.5)
    if "all" in taxa:
        j = taxa.index("all")
        ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(biomes),
                                    fill=False, edgecolor="#d9a73a",
                                    lw=1.8))
    for sp in ax.spines.values(): sp.set_color("#888")
    ax.set_title(title, color="#111", fontsize=11, fontweight="bold",
                  loc="left", pad=8)
    if ax_cb is not None:
        cb = plt.colorbar(im, cax=ax_cb)
        cb.set_label(r"$\Delta$ × $10^{-3}$", color="#222",
                      labelpad=6, fontsize=9)
        cb.ax.yaxis.set_tick_params(color="#444", labelsize=8)
        cb.outline.set_edgecolor("#888")
        plt.setp(cb.ax.get_yticklabels(), color="#222")


def render_double_violin(ax, df_pres, df_coll, taxa):
    ax.set_facecolor("white")
    # Order taxa by descending collapsed mean (most-robust first)
    coll_means = {t: df_coll[df_coll.taxon_group == t]["delta"].mean()
                   for t in taxa}
    taxa_sorted = sorted(taxa, key=lambda t: -coll_means.get(t, -np.inf))

    positions = np.arange(len(taxa_sorted))
    width_pres = 0.8
    width_coll = 0.55

    for i, t in enumerate(taxa_sorted):
        vals_p = df_pres[df_pres.taxon_group == t]["delta"].values * 1000
        vals_c = df_coll[df_coll.taxon_group == t]["delta"].values * 1000
        color = TAXON_COLOR.get(t, "#888888")

        # PRESERVED — faded violin (wide)
        if len(vals_p) >= 3:
            vp = ax.violinplot([vals_p], positions=[i], widths=width_pres,
                                showmeans=False, showmedians=False,
                                showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(color); b.set_alpha(0.18)
                b.set_edgecolor("#999"); b.set_linewidth(0.5)
        # COLLAPSED — solid violin (narrow, on top)
        if len(vals_c) >= 3:
            vc = ax.violinplot([vals_c], positions=[i], widths=width_coll,
                                showmeans=False, showmedians=False,
                                showextrema=False)
            for b in vc["bodies"]:
                b.set_facecolor(color); b.set_alpha(0.85)
                b.set_edgecolor("#222"); b.set_linewidth(0.7)

        # Scatter points
        ax.scatter([i - 0.20] * len(vals_p), vals_p, s=14,
                    color=color, alpha=0.55, edgecolor="none", zorder=2)
        ax.scatter([i + 0.05] * len(vals_c), vals_c, s=14,
                    color="#111", alpha=0.85, edgecolor="none", zorder=3)

        # Mean markers
        ax.scatter([i - 0.20], [np.mean(vals_p)] if len(vals_p) else [],
                    s=70, marker="_", color="#444", lw=2, zorder=4)
        ax.scatter([i + 0.05], [np.mean(vals_c)] if len(vals_c) else [],
                    s=70, marker="_", color="#111", lw=2.4, zorder=5)

        # Per-taxon Wilcoxon p (collapsed vs 0)
        if len(vals_c) >= 4:
            try:
                _, p_w = stats.wilcoxon(vals_c, alternative="greater")
                pstr = f"p={p_w:.2f}"
            except Exception:
                pstr = ""
        else:
            pstr = ""
        # Annotate mean & p below
        ax.annotate(
            f"$\\mu_P$={np.mean(vals_p)*1:+.2f}\n"
            f"$\\mu_C$={np.mean(vals_c)*1:+.2f}\n{pstr}",
            xy=(i, ax.get_ylim()[0]), xytext=(i, -2.6),
            ha="center", va="top", fontsize=7, color="#222",
            annotation_clip=False)

    ax.axhline(0, color="#666", lw=0.6, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(taxa_sorted, color="#222", fontsize=9, rotation=0)
    ax.set_xlim(-0.6, len(taxa_sorted) - 0.4)
    ax.set_ylabel(r"per-biome $\Delta$ ($\times 10^{-3}$)",
                  color="#222", fontsize=9.5)
    ax.tick_params(colors="#222", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#bbb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "C. Distribution shift under class-word collapse, per taxon "
        "(faded = preserved, solid = collapsed)",
        color="#111", fontsize=11, fontweight="bold", loc="left", pad=10)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#888", alpha=0.25, label="preserved (Panel A)"),
        Patch(facecolor="#888", alpha=0.85, label="collapsed (Panel B)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8.5,
               frameon=True)


def main():
    csv_left = EMB / "v3_byTaxon_sentpool_iNat.csv"
    csv_right = EMB / "v3_byTaxon_sentpool_collapsed_iNat.csv"

    d_l, p_l, q_l, biomes, taxa_l, df_l = prep(csv_left)
    d_r, p_r, q_r, _, taxa_r, df_r = prep(csv_right, biome_order=biomes)

    vmax = float(np.nanpercentile(np.abs(np.concatenate(
        [d_l.ravel(), d_r.ravel()])), 99))

    # Layout: 2 rows (heatmaps top, violin bottom). Top row has 3 cols
    # (heatmap, heatmap, colorbar). Bottom row is a single full-width
    # violin panel.
    fig = plt.figure(figsize=(
        0.85 * (len(taxa_l) * 2) + 5,
        0.50 * len(biomes) + 6.5
    ))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[0.50 * len(biomes) + 1.5, 5.0],
        width_ratios=[len(taxa_l), len(taxa_r), 0.6],
        hspace=0.32, wspace=0.04,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    ax_cb = fig.add_subplot(gs[0, 2])
    axC = fig.add_subplot(gs[1, :])

    render_heatmap(axA, d_l, p_l, q_l, biomes, taxa_l, vmax,
                    "A. Class words preserved   "
                    "(mammal, bird, fish, plant, …)",
                    label_yticks=True, ax_cb=None)
    render_heatmap(axB, d_r, p_r, q_r, biomes, taxa_r, vmax,
                    "B. Class words collapsed   "
                    "(animal, plant)",
                    label_yticks=False, ax_cb=ax_cb)

    # Intersection of taxa across both encodings (drop any taxon only
    # in one), so the violin compares matched distributions.
    taxa_both = [t for t in taxa_l if t in taxa_r]
    render_double_violin(axC, df_l, df_r, taxa_both)

    fig.subplots_adjust(top=0.94, bottom=0.10, left=0.10, right=0.96)
    fig.savefig(OUT, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
