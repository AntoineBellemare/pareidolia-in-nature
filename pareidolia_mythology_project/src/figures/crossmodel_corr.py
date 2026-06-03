"""_fig9_crossmodel_v2.py — clean cross-model robustness figure.

Layout (3 rows × 3 cols, generous spacing):
  ROW 1 — Russian v7 abstracts: SigLIP-2 panel | M-CLIP panel | scatter
  ROW 2 — English v4 oneliners: SigLIP-2 panel | OpenCLIP-laion2b | OpenCLIP-openai
  ROW 3 — scatter row for oneliners: SigLIP vs laion2b | SigLIP vs openai | (empty / legend)

Cleanliness fixes vs v1:
  • No annotation text inside scatter plots (was overlapping with subplots
    in v1). Instead: short biome codes via short_biome().
  • Each panel has its own axes; scatter panels never share row with bar
    panels of different aspect.
  • Common biome ordering across all bar panels (Spec A iNat v6 ordering)
    so the same biome is at the same y-row.
  • Larger fig + generous wspace/hspace.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]  # project root
EMB = ROOT / "dataset/imagery/embeddings"
SIGLIP = EMB / "siglip2-large"
OUT = ROOT / "dataset/imagery/figures/headlines_final_russian/fig9_crossmodel_correlation.png"

from make_phase2_figures import short_biome, biome_color, sig_stars

MIN_MOTIFS = 10


def load_spec(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "n_motifs_in_biome_specific" in df.columns:
        df = df.rename(columns={"n_motifs_in_biome_specific": "n_motifs_in_biome"})
    df = df[df["biome"].apply(lambda x: isinstance(x, str)) & (df["biome"] != "N/A")]
    df = df[df["n_motifs_in_biome"] >= MIN_MOTIFS]
    return df.copy()


def bar_panel(ax, df, title, biome_order=None, xmax=None):
    ax.set_facecolor("white")
    if biome_order is not None:
        df = df.set_index("biome").reindex(biome_order).reset_index()
        df = df.dropna(subset=["delta"])
    else:
        df = df.sort_values("delta")
    y = np.arange(len(df))
    cols = [biome_color(b) for b in df["biome"]]
    ax.axvline(0, color="#777", lw=0.8, zorder=1)
    ax.grid(axis="x", linestyle="--", alpha=0.30, color="#cccccc", zorder=0)
    for yi, c, d in zip(y, cols, df["delta"]):
        ax.barh(yi, d, color=c, edgecolor=c, lw=0.6, alpha=0.92, height=0.72,
                zorder=2)
    if xmax is None:
        xmax = max(abs(df["delta"]).max() * 1.25, 1e-7)
    for i, (_, r) in enumerate(df.reset_index(drop=True).iterrows()):
        s = sig_stars(r["p_one_sided"])
        if s:
            x = r["delta"]
            ax.text(x + (xmax * 0.015 if x >= 0 else -xmax * 0.015), i, s,
                    color="#a36b00", fontsize=9, fontweight="bold",
                    va="center", ha="left" if x >= 0 else "right", zorder=3)
    ax.set_xlim(-xmax, xmax * 1.10)
    ax.set_yticks(y)
    ax.set_yticklabels([short_biome(b) for b in df["biome"]],
                        color="#222", fontsize=8.5)
    for tick, b in zip(ax.get_yticklabels(), df["biome"]):
        tick.set_bbox(dict(facecolor=biome_color(b), edgecolor="none",
                            pad=2.0, alpha=0.18))
    ax.set_title(title, color="#111", fontsize=10.5, fontweight="bold",
                 loc="left", pad=6)
    ax.tick_params(colors="#444", labelsize=7.5)
    for sp in ax.spines.values(): sp.set_color("#aaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def scatter_panel(ax, df_a, df_b, label_a, label_b):
    ax.set_facecolor("white")
    j = df_a.merge(df_b, on="biome", suffixes=("_a", "_b"))
    if j.empty:
        ax.set_title("(no overlap)", color="#888"); return
    cols = [biome_color(b) for b in j["biome"]]
    # 45° guideline + axes
    lo = float(min(j["delta_a"].min(), j["delta_b"].min(), 0))
    hi = float(max(j["delta_a"].max(), j["delta_b"].max())) * 1.15
    pad = (hi - lo) * 0.15
    lo -= pad; hi += pad
    ax.plot([lo, hi], [lo, hi], "--", color="#999", lw=0.8, zorder=1)
    ax.axhline(0, color="#cccccc", lw=0.6, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.6, zorder=1)
    ax.scatter(j["delta_a"], j["delta_b"], s=120, c=cols,
                edgecolor="#222", linewidth=0.6, zorder=3, alpha=0.95)
    for i, (_, r) in enumerate(j.iterrows()):
        offset_y = 14 if i % 2 == 0 else -14
        va = "bottom" if offset_y > 0 else "top"
        ax.annotate(short_biome(r["biome"]),
                    (r["delta_a"], r["delta_b"]),
                    xytext=(0, offset_y), textcoords="offset points",
                    color="#444", fontsize=6.8,
                    ha="center", va=va)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    if len(j) >= 3:
        rho, p_rho = stats.spearmanr(j["delta_a"], j["delta_b"])
        r_p, p_r = stats.pearsonr(j["delta_a"], j["delta_b"])
        ttl = (f"Spearman ρ={rho:+.2f} (p={p_rho:.3f})\n"
               f"Pearson r={r_p:+.2f} (p={p_r:.3f})")
    else:
        ttl = "(too few biomes)"
    ax.set_xlabel(f"Δ — {label_a}", color="#222", fontsize=9)
    ax.set_ylabel(f"Δ — {label_b}", color="#222", fontsize=9)
    ax.tick_params(colors="#444", labelsize=7.5)
    ax.set_title(ttl, color="#111", fontsize=9.5, fontweight="bold", pad=6)
    for sp in ax.spines.values(): sp.set_color("#aaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    siglip_abs  = load_spec(SIGLIP / "specA_iNatBasicxabstracts_ru_hypv7_filt.csv")
    mclip_abs   = load_spec(EMB / "mclip/specA_iNatxabstracts_ru_hypv7_filt.csv")
    siglip_one  = load_spec(SIGLIP / "specA_iNatBasicxoneliners_hypv4_filt.csv")
    oc_laion    = load_spec(EMB / "openclip_laion2b/specA_iNatxoneliners_hypv4_filt.csv")
    oc_openai   = load_spec(EMB / "openclip_openai/specA_iNatxoneliners_hypv4_filt.csv")

    # Use SigLIP-2 abstracts ordering as canonical (ascending by Δ for nice bar plot)
    abs_order = siglip_abs.sort_values("delta")["biome"].tolist()
    one_order = siglip_one.sort_values("delta")["biome"].tolist()

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.45,
                          height_ratios=[1, 1, 1.05])

    # === ROW 1: abstracts (bars + bars + scatter) ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    # common xmax for the two abstract panels
    xmax_abs = max(abs(siglip_abs["delta"]).max(),
                    abs(mclip_abs["delta"]).max()) * 1.20
    bar_panel(ax1, siglip_abs, "SigLIP-2-large  ·  Russian v7 abstracts",
              biome_order=abs_order, xmax=xmax_abs)
    bar_panel(ax2, mclip_abs,  "M-CLIP (XLM-R + ViT-L/14)  ·  Russian v7 abstracts",
              biome_order=abs_order, xmax=xmax_abs)
    scatter_panel(ax3, siglip_abs, mclip_abs, "SigLIP-2 abstracts", "M-CLIP abstracts")

    # === ROW 2: oneliner bars ===
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    xmax_one = max(abs(siglip_one["delta"]).max(),
                    abs(oc_laion["delta"]).max(),
                    abs(oc_openai["delta"]).max()) * 1.20
    bar_panel(ax4, siglip_one,  "SigLIP-2-large  ·  English v4 oneliners",
              biome_order=one_order, xmax=xmax_one)
    bar_panel(ax5, oc_laion,    "OpenCLIP-ViT-L/14 (LAION-2B)  ·  English v4 oneliners",
              biome_order=one_order, xmax=xmax_one)
    bar_panel(ax6, oc_openai,   "OpenCLIP-ViT-L/14 (OpenAI)  ·  English v4 oneliners",
              biome_order=one_order, xmax=xmax_one)

    # === ROW 3: oneliner scatters (only two; right slot left empty) ===
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax_empty = fig.add_subplot(gs[2, 2])
    scatter_panel(ax7, siglip_one, oc_laion,  "SigLIP-2 oneliners", "OpenCLIP-LAION-2B")
    scatter_panel(ax8, siglip_one, oc_openai, "SigLIP-2 oneliners", "OpenCLIP-OpenAI")
    ax_empty.axis("off")

    fig.savefig(OUT, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
