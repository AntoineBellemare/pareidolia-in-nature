"""Re-render fig4_taxon_facets.png using the polished make_facets_grid from
make_paper_figures_fdr.py, with the final config (abstracts × hypv4 ×
biome-tell filter × Spec A on iNat)."""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
OUT = ROOT / "paper/figures/fig4_taxon_facets.png"

import numpy as np
import matplotlib.pyplot as plt
from _fdr_helpers import (
    bh_qvalues, TAXON_ORDER, TAXON_LABEL_LONG, TAXON_COLOR, stars_with_q,
)
from make_phase2_figures import short_biome


def make_facets_grid_white(df, taxa, biomes, suptitle, out_path):
    """White-background facets grid (paper version of make_facets_grid)."""
    cols = 4
    rows = (len(taxa) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(5.0 * cols,
                                       0.40 * len(biomes) * rows + 1.6),
                              sharey=True)
    axes = np.array(axes).reshape(rows, cols)
    vmax = float(np.nanmax(np.abs(df["delta"].values))) * 1.18

    fig.patch.set_facecolor("white")
    for k, taxon in enumerate(taxa):
        ax = axes[k // cols, k % cols]
        ax.set_facecolor("white")
        sub = (df[df["taxon_group"] == taxon]
                .set_index("biome").reindex(biomes))
        y = np.arange(len(biomes))
        color = TAXON_COLOR.get(taxon, "#888")
        ax.barh(y, sub["delta"].fillna(0), color=color,
                edgecolor="#222", lw=0.45)
        ax.axvline(0, color="#666", lw=0.5)
        ax.set_xlim(-vmax, vmax)
        for i, (b, r) in enumerate(sub.iterrows()):
            p = r.get("p_one_sided")
            q = r.get("q_value", np.nan)
            if pd.isna(p): continue
            star, fdr_pass = stars_with_q(p, q)
            if not star: continue
            x = r["delta"]
            tx = x + (vmax * 0.025 if x >= 0 else -vmax * 0.025)
            color_star = "#b8860b" if fdr_pass else "#888"
            weight = "bold" if fdr_pass else "normal"
            ax.text(tx, i, star,
                    color=color_star, fontsize=8.5, fontweight=weight,
                    va="center", ha="left" if x >= 0 else "right")
        ax.set_title(TAXON_LABEL_LONG.get(taxon, taxon),
                     color="#111", fontsize=10.5, fontweight="bold")
        ax.tick_params(colors="#222", labelsize=7.5)
        for sp in ax.spines.values(): sp.set_color("#bbb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if k % cols == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([short_biome(b) for b in biomes],
                               color="#222", fontsize=7.5)
    for k in range(len(taxa), rows * cols):
        axes[k // cols, k % cols].axis("off")

    # suptitle removed — handled by paper.tex caption
    legend_txt = ("Stars: * p<.05  ** p<.01  *** p<.001    ·    "
                   "GOLD bold = survives Benjamini–Hochberg FDR q<.05  "
                   "(grey = nominal p only)")
    fig.text(0.5, 0.005, legend_txt, color="#555", fontsize=9, ha="center")
    fig.tight_layout(rect=(0, 0.018, 1, 0.96))
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    df = pd.read_csv(EMB / "v3_byTaxon_sentpool_iNat.csv")
    df = df[df["biome"].apply(lambda x: isinstance(x, str))
            & (df["biome"] != "N/A")].copy()
    if "n_mot" in df.columns and "n_motifs_in_biome" not in df.columns:
        df = df.rename(columns={"n_mot": "n_motifs_in_biome"})
    df["q_value"] = bh_qvalues(df["p_one_sided"].values)

    taxa = [t for t in TAXON_ORDER if t in df["taxon_group"].unique()]
    base = (df[df["taxon_group"] == "all"]
            .sort_values("delta", ascending=False))
    biomes = base["biome"].tolist()
    suptitle = (
        "Biome × Taxon — Mythology–imagery alignment   ·   full LLM-clean "
        "corpus, BH-FDR corrected\n"
        "Sentence-pooled SigLIP-2 (English-anonymised Berezkin abstracts).   "
        "iNaturalist photos, n = 47,478."
    )
    make_facets_grid_white(df, taxa, biomes, suptitle, OUT)


if __name__ == "__main__":
    main()
