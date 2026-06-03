"""
make_paper_figures_fdr.py — FDR-corrected versions of the four taxon paper
figures, plus a Spec A version of the facets-grid layout that the user
screenshotted.

For each (text_variant, control) combination we produce:
  - facets-grid 3x4 layout (one panel per taxon), Δ per biome with stars
  - taxon-matrix layout (heatmap rows=biome, cols=taxon)

Stars are computed using Benjamini-Hochberg FDR correction across all cells
in the figure (i.e. across all biome × taxon combinations).  Light stars
remain p-value based; bold stars indicate q < 0.05 after BH-FDR.

Outputs (eight files):
  fig_paper_taxon_ONELINERS_fdr.png
  fig_paper_taxon_ABSTRACTS_fdr.png
  fig_paper_taxon_ONELINERS_specA.png        (facets-grid version, new)
  fig_paper_taxon_ABSTRACTS_specA.png        (facets-grid version, new)
  fig_paper_taxon_ONELINERS_specA_fdr.png    (facets-grid + FDR)
  fig_paper_taxon_ABSTRACTS_specA_fdr.png    (facets-grid + FDR)
  fig_paper_taxon_matrix_ONELINERS_fdr.png   (matrix + FDR)
  fig_paper_taxon_matrix_ABSTRACTS_fdr.png   (matrix + FDR)
  fig_paper_taxon_matrix_ONELINERS_specA_fdr.png
  fig_paper_taxon_matrix_ABSTRACTS_specA_fdr.png
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
FIG = ROOT / "dataset/imagery/figures"

from make_phase2_figures import short_biome, sig_stars


# ----- Benjamini-Hochberg FDR ------------------------------------------------
def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg q-values. NaN p-values yield NaN q-values."""
    p = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if not valid.any():
        return q
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    qv = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0, 1)
    unranked = np.empty_like(qv)
    unranked[order] = qv
    q[valid] = unranked
    return q


def stars_with_q(p, q):
    """Return (text, fdr_pass) where text is the asterisk string and fdr_pass
    indicates BH q < 0.05."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "", False
    base = sig_stars(p)
    fdr = (not np.isnan(q)) and (q < 0.05)
    return base, fdr


# Taxon ordering / labels
TAXON_ORDER = [
    "all", "Plantae", "Fungi", "Animalia",
    "Mammalia", "Aves", "Reptilia", "Amphibia",
    "Actinopterygii", "Insecta", "Arachnida", "Mollusca",
]
TAXON_COLOR = {
    "all":            "#cccccc",
    "Plantae":        "#7cbe5e",
    "Fungi":          "#c2914b",
    "Animalia":       "#cf6f3f",
    "Mammalia":       "#d04f6f",
    "Aves":           "#76b6e5",
    "Reptilia":       "#6abc8f",
    "Amphibia":       "#4ea36f",
    "Actinopterygii": "#2d8fb3",
    "Insecta":        "#a96cb0",
    "Arachnida":      "#9e6cb4",
    "Mollusca":       "#8a6240",
}
TAXON_LABEL_LONG = {
    "all":            "ALL TAXA",
    "Plantae":        "Plantae",
    "Fungi":          "Fungi",
    "Animalia":       "Animalia (other)",
    "Mammalia":       "Mammalia",
    "Aves":           "Aves",
    "Reptilia":       "Reptilia",
    "Amphibia":       "Amphibia",
    "Actinopterygii": "Actinopterygii (fish)",
    "Insecta":        "Insecta",
    "Arachnida":      "Arachnida",
    "Mollusca":       "Mollusca",
}
TAXON_LABEL_SHORT = {
    "all":            "ALL", "Plantae": "Plantae", "Fungi": "Fungi",
    "Animalia": "Animalia\n(other)", "Mammalia": "Mammalia", "Aves": "Aves",
    "Reptilia": "Reptilia", "Amphibia": "Amphibia",
    "Actinopterygii": "Fish\n(Actinopt.)",
    "Insecta": "Insecta", "Arachnida": "Arachnida", "Mollusca": "Mollusca",
}


def load_byTaxon(text_variant: str, control: str) -> pd.DataFrame:
    """control ∈ {'resid', 'specA'}"""
    if control == "resid":
        csv = EMB / f"biome_test_{text_variant}_byTaxon_resid.csv"
    elif control == "specA":
        # specA uses 'oneliners'/'abstracts' naming
        slug = "oneliners" if text_variant == "all" else text_variant
        csv = EMB / f"specA_byTaxon_iNatx{slug}.csv"
    else:
        raise ValueError(control)
    df = pd.read_csv(csv)
    df = df[df["biome"].apply(lambda x: isinstance(x, str))
            & (df["biome"] != "N/A")].copy()
    # Compute BH q-values across all valid p
    df["q_value"] = bh_qvalues(df["p_one_sided"].values)
    return df


# --------------------------------------------------------------------------- #
#                          FACETS-GRID FIGURE                                 #
# --------------------------------------------------------------------------- #
def make_facets_grid(df: pd.DataFrame, taxa, biomes, suptitle, out_path,
                     show_fdr_legend=True):
    """Builds the 3×4 facets-grid figure matching fig_paper_taxon_*.png."""
    cols = 4
    rows = (len(taxa) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(5.0 * cols, 0.40 * len(biomes) * rows + 1.5),
                              sharey=True)
    axes = np.array(axes).reshape(rows, cols)
    vmax = float(np.nanmax(np.abs(df["delta"].values))) * 1.18

    for k, taxon in enumerate(taxa):
        ax = axes[k // cols, k % cols]
        ax.set_facecolor("#11141a")
        sub = df[df["taxon_group"] == taxon].set_index("biome").reindex(biomes)
        y = np.arange(len(biomes))
        color = TAXON_COLOR.get(taxon, "#888")
        ax.barh(y, sub["delta"].fillna(0), color=color,
                edgecolor="#222831", lw=0.45)
        ax.axvline(0, color="#aaa", lw=0.5)
        ax.set_xlim(-vmax, vmax)
        for i, (b, r) in enumerate(sub.iterrows()):
            p = r.get("p_one_sided")
            q = r.get("q_value", np.nan)
            if pd.isna(p): continue
            star, fdr_pass = stars_with_q(p, q)
            if not star: continue
            x = r["delta"]
            tx = x + (vmax * 0.025 if x >= 0 else -vmax * 0.025)
            color_star = "#ffe680" if fdr_pass else "#7a7a7a"
            weight = "bold" if fdr_pass else "normal"
            ax.text(tx, i, star,
                    color=color_star, fontsize=8.5, fontweight=weight,
                    va="center", ha="left" if x >= 0 else "right")
        ax.set_title(TAXON_LABEL_LONG.get(taxon, taxon),
                     color="#eeeeee", fontsize=10.5)
        ax.tick_params(colors="#dddddd", labelsize=7.5)
        for s in ax.spines.values(): s.set_color("#444")
        if k % cols == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([short_biome(b) for b in biomes],
                               color="#dddddd", fontsize=7.5)
    for k in range(len(taxa), rows * cols):
        axes[k // cols, k % cols].axis("off")

    fig.suptitle(suptitle, color="#eeeeee", fontsize=12, y=0.998)
    if show_fdr_legend:
        # tiny legend block at bottom
        legend_txt = ("Stars: * p<.05  ** p<.01  *** p<.001    ·    "
                      "GOLD bold = survives Benjamini-Hochberg FDR q<.05  "
                      "(grey = nominal p only)")
        fig.text(0.5, 0.005, legend_txt, color="#bbb", fontsize=9, ha="center")
    fig.patch.set_facecolor("#0c0d11")
    fig.tight_layout(rect=(0, 0.018, 1, 0.96))
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
#                          TAXON MATRIX FIGURE                                #
# --------------------------------------------------------------------------- #
def _matrix_arrays(df, taxa, biomes):
    delta = np.full((len(biomes), len(taxa)), np.nan)
    pval = np.full((len(biomes), len(taxa)), np.nan)
    qval = np.full((len(biomes), len(taxa)), np.nan)
    bi = {b: i for i, b in enumerate(biomes)}
    ti = {t: i for i, t in enumerate(taxa)}
    for _, r in df.iterrows():
        if r["biome"] not in bi or r["taxon_group"] not in ti: continue
        i, j = bi[r["biome"]], ti[r["taxon_group"]]
        delta[i, j] = r["delta"]
        pval[i, j] = r["p_one_sided"]
        qval[i, j] = r.get("q_value", np.nan)
    return delta, pval, qval


def make_matrix(df, taxa, biomes, suptitle, out_path, control_label):
    delta, pval, qval = _matrix_arrays(df, taxa, biomes)
    abs_max = np.nanpercentile(np.abs(delta), 99)
    if not np.isfinite(abs_max) or abs_max == 0:
        abs_max = float(np.nanmax(np.abs(delta)) or 0.003)
    vmax = float(abs_max)

    base = df[df["taxon_group"] == "all"].set_index("biome").reindex(biomes)

    fig = plt.figure(figsize=(0.85 * len(taxa) + 6.8, 0.55 * len(biomes) + 3.3))
    fig.patch.set_facecolor("#0c0d11")
    gs = fig.add_gridspec(1, 3, width_ratios=[len(taxa) * 0.6, 0.5, 0.18],
                          wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    ax_cb = fig.add_subplot(gs[2])
    ax.set_facecolor("#11141a")
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    im = ax.imshow(delta, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            if np.isnan(delta[i, j]):
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                           facecolor="#2a2e36",
                                           edgecolor="#0c0d11", lw=0.4))
                ax.text(j, i, "—", color="#666", fontsize=9,
                        ha="center", va="center")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            d = delta[i, j]; p = pval[i, j]; q = qval[i, j]
            if np.isnan(d): continue
            r, g, b, _ = cmap(norm(d))
            luma = 0.299*r + 0.587*g + 0.114*b
            txt_color = "#11141a" if luma > 0.55 else "#f4f4f4"
            star, fdr_pass = stars_with_q(p, q)
            value = f"{d*1000:+.2f}"
            if star:
                if fdr_pass:
                    label = f"{value}{star}"
                    weight = "bold"
                else:
                    label = f"{value}{star.lower()}"  # de-emphasise
                    weight = "normal"
            else:
                label = value
                weight = "normal"
            ax.text(j, i, label, color=txt_color, fontsize=7.8,
                    ha="center", va="center", fontweight=weight)
    ax.set_xticks(range(len(taxa)))
    ax.set_xticklabels([TAXON_LABEL_SHORT[t] for t in taxa],
                       color="#dddddd", fontsize=9)
    ax.tick_params(axis="x", which="both", length=0, pad=4, top=True,
                   labeltop=True, bottom=False, labelbottom=False)
    ax.set_yticks(range(len(biomes)))
    ax.set_yticklabels([short_biome(b) for b in biomes],
                       color="#dddddd", fontsize=9)
    for s in ax.spines.values(): s.set_color("#444")
    ax.set_xlim(-0.5, len(taxa) - 0.5)
    ax.set_ylim(len(biomes) - 0.5, -0.5)
    for j in range(len(taxa) + 1):
        ax.axvline(j-0.5, color="#0c0d11", lw=0.4)
    for i in range(len(biomes) + 1):
        ax.axhline(i-0.5, color="#0c0d11", lw=0.4)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, len(biomes),
                               fill=False, edgecolor="#f4d35e", lw=1.8))

    ax_bar.set_facecolor("#11141a")
    all_d = base["delta"].values
    all_p = base["p_one_sided"].values
    all_q = base["q_value"].values if "q_value" in base.columns else np.full_like(all_p, np.nan)
    y = np.arange(len(biomes))
    colors = [cmap(norm(d)) for d in all_d]
    ax_bar.barh(y, all_d, color=colors, edgecolor="#222831", lw=0.4)
    ax_bar.axvline(0, color="#aaa", lw=0.5)
    for i, (d, p, q) in enumerate(zip(all_d, all_p, all_q)):
        if pd.isna(p) or pd.isna(d): continue
        star, fdr_pass = stars_with_q(p, q)
        if not star: continue
        ax_bar.text(d + (vmax*0.05 if d>=0 else -vmax*0.05), i, star,
                    color="#ffe680" if fdr_pass else "#7a7a7a",
                    fontsize=9,
                    fontweight="bold" if fdr_pass else "normal",
                    va="center", ha="left" if d>=0 else "right")
    ax_bar.set_ylim(len(biomes)-0.5, -0.5)
    ax_bar.set_yticks([])
    ax_bar.set_xlim(-vmax*1.3, vmax*1.3)
    ax_bar.tick_params(colors="#bbb", labelsize=7.5)
    for s in ax_bar.spines.values(): s.set_color("#444")
    ax_bar.set_title("Δ (all taxa)", color="#eeeeee", fontsize=9, pad=4)

    cb = fig.colorbar(im, cax=ax_cb)
    cb.ax.yaxis.set_ticks_position("right")
    cb.set_label(f"residualised Δ similarity, {control_label}", color="#bbb",
                 labelpad=8)
    cb.ax.yaxis.set_tick_params(color="#bbb", labelsize=8)
    cb.outline.set_edgecolor("#444")
    plt.setp(cb.ax.get_yticklabels(), color="#bbb")

    fig.suptitle(suptitle, color="#eeeeee", fontsize=12, y=0.995)
    fig.text(0.5, 0.005,
             "GOLD bold = survives Benjamini-Hochberg FDR q<.05  ·  "
             "grey lowercase = nominal p only",
             color="#bbb", fontsize=9, ha="center")
    fig.subplots_adjust(top=0.90, left=0.20, right=0.95, bottom=0.04)
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
def main():
    for text_variant, slug, text_label in [
        ("all",       "ONELINERS", "Berezkin one-line motif titles"),
        ("abstracts", "ABSTRACTS", "Full Berezkin abstracts (multilingual SigLIP-2)"),
    ]:
        # ---- residualised-only ----
        df_r = load_byTaxon(text_variant, "resid")
        taxa_r = [t for t in TAXON_ORDER if t in df_r["taxon_group"].unique()]
        base_r = (df_r[df_r["taxon_group"] == "all"]
                  .sort_values("delta", ascending=False))
        biomes_r = base_r["biome"].tolist()

        title_r_facets = (
            "Biome × Taxon — Mythology–imagery alignment   ·   "
            "residualised Δ, BH-FDR corrected\n"
            f"Text source: {text_label}.   iNaturalist photos, n=47,900.")
        make_facets_grid(
            df_r, taxa_r, biomes_r, title_r_facets,
            FIG / f"fig_paper_taxon_{slug}_fdr.png")

        title_r_mat = (
            f"Biome × Taxon matrix — residualised + BH-FDR.   "
            f"Text source: {text_label}.   "
            f"Cell = Δ×1000.   Yellow box = ALL taxa.")
        make_matrix(df_r, taxa_r, biomes_r, title_r_mat,
                    FIG / f"fig_paper_taxon_matrix_{slug}_fdr.png",
                    "residualised")

        # ---- Spec A ----
        df_a = load_byTaxon(text_variant, "specA")
        taxa_a = [t for t in TAXON_ORDER if t in df_a["taxon_group"].unique()]
        base_a = (df_a[df_a["taxon_group"] == "all"]
                  .sort_values("delta", ascending=False))
        biomes_a = base_a["biome"].tolist()

        # Spec A facets-grid (new — equivalent of the screenshotted figure)
        title_a_facets = (
            "Biome × Taxon — Mythology–imagery alignment  ·  "
            "Spec A (universals dropped, ≥3 own-traditions), BH-FDR corrected\n"
            f"Text source: {text_label}.   iNaturalist photos, n=47,900.")
        make_facets_grid(
            df_a, taxa_a, biomes_a, title_a_facets,
            FIG / f"fig_paper_taxon_{slug}_specA_fdr.png")

        # Also a non-FDR Spec A facets-grid for completeness (no fdr suffix)
        make_facets_grid(
            df_a, taxa_a, biomes_a,
            "Biome × Taxon — Spec A (universals dropped). "
            f"Text: {text_label}.",
            FIG / f"fig_paper_taxon_{slug}_specA.png")

        title_a_mat = (
            f"Biome × Taxon matrix — Spec A + BH-FDR.   "
            f"Text source: {text_label}.   "
            f"Cell = Δ×1000.   Yellow box = ALL taxa.")
        make_matrix(df_a, taxa_a, biomes_a, title_a_mat,
                    FIG / f"fig_paper_taxon_matrix_{slug}_specA_fdr.png",
                    "Spec A")

        # ---- Quick FDR summary text ----
        n_cells_r = int(df_r["q_value"].notna().sum())
        n_q05_r = int((df_r["q_value"] < 0.05).sum())
        n_cells_a = int(df_a["q_value"].notna().sum())
        n_q05_a = int((df_a["q_value"] < 0.05).sum())
        print(f"\n  {text_variant}:  resid survives FDR q<.05 in {n_q05_r}/{n_cells_r} cells   ·   "
              f"specA survives FDR q<.05 in {n_q05_a}/{n_cells_a} cells")


if __name__ == "__main__":
    main()
