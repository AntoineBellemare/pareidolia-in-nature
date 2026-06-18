"""Argument roadmap: the three independent strategies that locate the
biome-mythology coupling. A visual table-of-contents for the paper.

  (1) Remove the names    -- anonymise the text, the alignment survives
  (2) Hold the names constant -- keep raw myths, control identity statistically
  (3) Recover from the geometry -- biome emerges unsupervised from myth x image

Outputs paper/figures/fig_roadmap.png
"""
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig_roadmap.png"


def box(ax, x, y, w, h, fc, ec, title, body, result, figs, title_c="#fff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.6, edgecolor=ec, facecolor="#ffffff", zorder=2))
    # header strip
    ax.add_patch(FancyBboxPatch((x, y + h - 0.072), w, 0.072,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0, facecolor=ec, zorder=3))
    ax.text(x + w / 2, y + h - 0.036, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=title_c, zorder=4)
    ax.text(x + w / 2, y + h - 0.092, figs, ha="center", va="center", fontsize=8,
            style="italic", color=ec, zorder=4)
    body_w = "\n".join(textwrap.wrap(body, width=42))
    ax.text(x + w / 2, y + h - 0.125, body_w, ha="center", va="top", fontsize=8.5,
            color="#222", zorder=4)
    ax.add_patch(FancyBboxPatch((x + 0.012, y + 0.012), w - 0.024, 0.155,
                                boxstyle="round,pad=0.008,rounding_size=0.015",
                                linewidth=0, facecolor=fc, alpha=0.55, zorder=3))
    ax.text(x + w / 2, y + 0.09, result, ha="center", va="center", fontsize=8.5,
            color="#111", fontweight="bold", zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(14, 8.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # shared input
    ax.add_patch(FancyBboxPatch((0.18, 0.86), 0.64, 0.10,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.6, edgecolor="#444", facecolor="#f2f4f6", zorder=2))
    ax.text(0.5, 0.91, "Cross-cultural mythology meets biome imagery",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#111")
    ax.text(0.5, 0.875, "2,158 Berezkin folk-motifs over 14 WWF biomes, aligned to "
            "iNaturalist species photos and Places365 scenes via sentence-pooled SigLIP-2",
            ha="center", va="center", fontsize=8.8, color="#333")

    cols = [(0.035, "#2e7d54", "#d7efe2"), (0.365, "#c4632e", "#f6e3d6"),
            (0.695, "#5a4b9c", "#e4def2")]
    w, h, ytop = 0.27, 0.52, 0.30
    # arrows from input to each lane
    for x0, ec, _ in cols:
        ax.add_patch(FancyArrowPatch((0.5, 0.855), (x0 + w / 2, ytop + h + 0.005),
                                     arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.4, color="#888", zorder=1))

    box(ax, cols[0][0], ytop, w, h, cols[0][2], cols[0][1],
        "1  ·  Remove the names",
        "Anonymise the text: every species $\\rightarrow$ class word, place and "
        "ethnonym $\\rightarrow$ placeholder, biome words dropped. Score the cleaned "
        "text with the within-iconic-taxon stratified $\\Delta$.",
        "Anonymised mythology still aligns with\nbiome imagery: 8/14 biomes (FDR),\n"
        "two corpora, four models; a breadth\ngradient; survives collapse to\n"
        "“animal”/“plant”.",
        "Figs 3, 4, 6, 7")
    box(ax, cols[1][0], ytop, w, h, cols[1][2], cols[1][1],
        "2  ·  Hold the names constant",
        "Keep the raw myths intact. A matched-permutation null shuffles biome only "
        "among motifs with the same species + place + ethnonym identity content.",
        "The alignment exceeds what naming\ndetermines: 6/14 biomes survive\n"
        "holding all identity constant (FDR),\nconverging with the geography\ncontrol.",
        "Fig 5")
    box(ax, cols[2][0], ytop, w, h, cols[2][2], cols[2][1],
        "3  ·  Read it off the geometry",
        "Describe each myth by its similarity profile across all 46,481 images. "
        "Biome labels are never used to build this space.",
        "Biome is recoverable unsupervised:\nown biome ranks in the 64th\npercentile, "
        "and all 9 well-sampled\nbiomes decode above chance (FDR).",
        "Fig 8")

    # convergence footer
    ax.add_patch(FancyBboxPatch((0.10, 0.045), 0.80, 0.135,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.6, edgecolor="#444",
                                facecolor="#fbf7ef", zorder=2))
    for x0, ec, _ in cols:
        ax.add_patch(FancyArrowPatch((x0 + w / 2, ytop - 0.005), (0.5, 0.185),
                                     arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.4, color="#888", zorder=1))
    ax.text(0.5, 0.135, "Three independent strategies converge",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#111")
    ax.text(0.5, 0.085, "Removing the names, holding them constant, and recovering biome "
            "from the geometry all locate the same\nbiome–mythology coupling, carried "
            "by lexical-thematic content rather than by the species a myth happens to name.",
            ha="center", va="center", fontsize=9.2, color="#333")

    fig.savefig(OUT, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
