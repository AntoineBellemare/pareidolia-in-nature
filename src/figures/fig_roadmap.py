"""Argument roadmap: the three independent strategies that locate the
biome-mythology coupling. A visual table-of-contents for the paper.

  (1) Remove the names        -- anonymise the text; the alignment survives
  (2) Hold the names constant -- keep raw myths; control identity statistically
  (3) Read it off the geometry-- biome emerges unsupervised from myth x image

Outputs paper/figures/fig_roadmap.png
"""
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig_roadmap.png"

HH = 0.066          # header-strip height
BODY_W = 38         # wrap width for the body paragraph
RES_W = 32          # wrap width for the result strip
LINE = 0.030        # line spacing (axis fraction)


def box(ax, x, y, w, h, accent, light, title, figs, body, result):
    # card
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.018",
                                linewidth=1.5, edgecolor=accent, facecolor="#ffffff",
                                zorder=2))
    # coloured header strip + title
    ax.add_patch(FancyBboxPatch((x, y + h - HH), w, HH,
                                boxstyle="round,pad=0.010,rounding_size=0.018",
                                linewidth=0, facecolor=accent, zorder=3))
    ax.text(x + w / 2, y + h - HH / 2, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#ffffff", zorder=4)
    # figure pointers
    ax.text(x + w / 2, y + h - HH - 0.026, figs, ha="center", va="center",
            fontsize=8, style="italic", color=accent, zorder=4)
    # body paragraph (top-aligned, under the figs line)
    ty = y + h - HH - 0.058
    for ln in textwrap.wrap(body, BODY_W):
        ax.text(x + w / 2, ty, ln, ha="center", va="top", fontsize=8.7,
                color="#222", zorder=4)
        ty -= LINE
    # result strip anchored at the bottom, sized to its line count
    res_lines = textwrap.wrap(result, RES_W)
    strip_h = 0.040 + LINE * len(res_lines)
    ax.add_patch(FancyBboxPatch((x + 0.012, y + 0.014), w - 0.024, strip_h,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=0, facecolor=light, zorder=3))
    ry = y + 0.014 + strip_h - 0.022
    for ln in res_lines:
        ax.text(x + w / 2, ry, ln, ha="center", va="center", fontsize=8.6,
                color="#111", fontweight="bold", zorder=4)
        ry -= LINE


def main():
    fig, ax = plt.subplots(figsize=(14, 8.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # shared input
    ax.add_patch(FancyBboxPatch((0.14, 0.850), 0.72, 0.125,
                                boxstyle="round,pad=0.010,rounding_size=0.018",
                                linewidth=1.5, edgecolor="#444", facecolor="#f2f4f6",
                                zorder=2))
    ax.text(0.5, 0.943, "Cross-cultural mythology meets biome imagery",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#111")
    ax.text(0.5, 0.906, "2,158 Berezkin folk-motifs over 14 WWF biomes, aligned to",
            ha="center", va="center", fontsize=8.7, color="#333")
    ax.text(0.5, 0.879, "iNaturalist photos and Places365 scenes (sentence-pooled SigLIP-2)",
            ha="center", va="center", fontsize=8.7, color="#333")

    cols = [(0.035, "#2e7d54", "#dcefe4"), (0.365, "#c4632e", "#f6e2d4"),
            (0.695, "#5a4b9c", "#e6e0f2")]
    w, h, ytop = 0.27, 0.41, 0.315

    # arrows: input -> each lane
    for x0, accent, _ in cols:
        ax.add_patch(FancyArrowPatch((0.5, 0.848), (x0 + w / 2, ytop + h + 0.004),
                                     arrowstyle="-|>", mutation_scale=15,
                                     linewidth=1.3, color="#999", zorder=1))

    box(ax, cols[0][0], ytop, w, h, cols[0][1], cols[0][2],
        "1  ·  Remove the names", "Figs 3, 4, 6, 7",
        "Strip every species, place, and people from the myth, then "
        "re-score what remains with the stratified Δ.",
        "Anonymised myth still aligns: 8/14 biomes (FDR), two corpora, "
        "four models.")
    box(ax, cols[1][0], ytop, w, h, cols[1][1], cols[1][2],
        "2  ·  Hold the names constant", "Fig 5  ·  S6",
        "Keep the raw myth. Shuffle the biome label only among myths "
        "with the same species, place, and people.",
        "Alignment exceeds naming: 6/14 biomes survive (FDR), converging "
        "with strategy 1.")
    box(ax, cols[2][0], ytop, w, h, cols[2][1], cols[2][2],
        "3  ·  Read it off the geometry", "Fig 8",
        "Describe each myth by its similarity to all 46,481 images. "
        "Biome labels are never used to build this space.",
        "Biome is recoverable unsupervised: 9/9 well-sampled biomes "
        "decode above chance (FDR).")

    # convergence footer
    ax.add_patch(FancyBboxPatch((0.13, 0.045), 0.74, 0.135,
                                boxstyle="round,pad=0.010,rounding_size=0.018",
                                linewidth=1.5, edgecolor="#444", facecolor="#fbf7ef",
                                zorder=2))
    for x0, accent, _ in cols:
        ax.add_patch(FancyArrowPatch((x0 + w / 2, ytop - 0.004), (0.5, 0.182),
                                     arrowstyle="-|>", mutation_scale=15,
                                     linewidth=1.3, color="#999", zorder=1))
    ax.text(0.5, 0.137, "Three independent strategies converge",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#111")
    ax.text(0.5, 0.087, "Removing the names, holding them constant, and reading the "
            "geometry all locate the same biome–mythology coupling—\ncarried by "
            "lexical-thematic content, not by the species a myth happens to name.",
            ha="center", va="center", fontsize=9.2, color="#333")

    fig.savefig(OUT, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
