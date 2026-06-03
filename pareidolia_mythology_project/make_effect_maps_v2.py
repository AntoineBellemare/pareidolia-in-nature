"""
make_effect_maps_v2.py — controlled-confidence world maps.

Improvements over make_effect_maps.py:
  - GREY OUT biomes with insufficient sample size (configurable thresholds)
  - Alpha (saturation) scales with statistical confidence: 1 − p_value
  - Show small marginal panel listing each biome's n_traditions and n_motifs
    so the reader sees the data behind every color
  - Tradition density dots overlaid (already)

Output:
  fig60_effect_map_headline_confidence.png
  fig61_effect_map_yfcc_filtered_confidence.png
  fig62_effect_maps_by_taxon_confidence.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parent
MAP = ROOT / "dataset/mapping_v2"
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
SHP = ROOT / "raw_downloads/Ecoregions2017/Ecoregions2017.shp"
FIG = ROOT / "dataset/imagery/figures"

from make_phase2_figures import short_biome


# Thresholds for "trustworthy enough to show"
MIN_TRADITIONS = 10   # biomes with fewer get greyed
MIN_MOTIFS = 50       # biomes with fewer "own" motifs get greyed
MIN_IMAGES = 30       # biomes with fewer images get greyed


def dark(fig, ax):
    fig.patch.set_facecolor("#0c0d11")
    ax.set_facecolor("#0c1422")


def per_biome_data(csv_path: Path):
    """Returns dict biome -> {'delta', 'p', 'n_imgs', 'n_motifs', 'n_trad'}.

    Handles both residualised CSVs (column 'n_motifs_in_biome') and Spec A
    CSVs (column 'n_motifs_in_biome_specific')."""
    df = pd.read_csv(csv_path)
    df = df[df["biome"].apply(lambda x: isinstance(x, str))]
    trad = pd.read_parquet(MAP / "traditions.parquet")
    n_trad_per_biome = trad.groupby("biome_wwf").size().to_dict()
    out = {}
    for _, r in df.iterrows():
        b = r["biome"]
        # n_motifs column might be named differently in Spec A files
        n_motifs = None
        for col in ("n_motifs_in_biome", "n_motifs_in_biome_specific"):
            if col in r and pd.notna(r[col]):
                n_motifs = int(r[col])
                break
        out[b] = {
            "delta": float(r["delta"]),
            "p": float(r["p_one_sided"]),
            "n_imgs": int(r["n_imgs"]),
            "n_motifs": n_motifs,
            "n_trad": int(n_trad_per_biome.get(b, 0)),
        }
    return out


def render_map(ax, eco, biome_data: dict, traditions: pd.DataFrame,
               title: str, vmax: float | None = None,
               show_traditions: bool = True,
               require_min_trad: int = MIN_TRADITIONS,
               require_min_motifs: int = MIN_MOTIFS,
               require_min_imgs: int = MIN_IMAGES):
    if vmax is None:
        vals = [v["delta"] for v in biome_data.values()]
        vmax = max(abs(min(vals)), abs(max(vals))) * 1.05 if vals else 0.003

    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    GREY = "#383d49"  # for "not enough data" cells

    # Color polygons
    for biome_name, group in eco.groupby("BIOME_NAME"):
        info = biome_data.get(biome_name)
        if info is None:
            color = GREY
        elif (info["n_trad"] < require_min_trad
              or (info.get("n_motifs") is not None
                  and info["n_motifs"] < require_min_motifs)
              or info["n_imgs"] < require_min_imgs):
            color = GREY
        else:
            base = np.array(cmap(norm(info["delta"])))
            # alpha scales with confidence: 1 - p (so sig cells are saturated)
            confidence = 1.0 - min(info["p"], 0.5)  # range [0.5, 1.0]
            # blend toward gray bg when low confidence
            bg = np.array(mcolors.to_rgba("#1e2530"))
            base = base * confidence + bg * (1 - confidence)
            color = tuple(base)
        group.plot(ax=ax, color=color, edgecolor="#0c0d11", linewidth=0.1)

    if show_traditions and traditions is not None:
        ax.scatter(traditions["lon"], traditions["lat"],
                   s=2.5, color="#ffffff", alpha=0.5,
                   edgecolors="none", zorder=4)

    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color="#eeeeee", fontsize=11)
    for s in ax.spines.values(): s.set_color("#222")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def make_legend_table(ax, biome_data, biomes_sorted_by_delta):
    """Right-hand companion table: per biome n_trad, n_motifs, Δ, p."""
    ax.axis("off")
    ax.set_facecolor("#0c0d11")
    rows = []
    for b in biomes_sorted_by_delta:
        info = biome_data.get(b, {})
        rows.append([
            short_biome(b)[:30],
            str(info.get("n_trad", "—")),
            str(info.get("n_motifs", "—")),
            str(info.get("n_imgs", "—")),
            f"{info.get('delta', 0):+.4f}" if "delta" in info else "—",
            f"{info.get('p', 1):.3f}" if "p" in info else "—",
            "***" if info.get("p", 1) < 0.001 else
            "**"  if info.get("p", 1) < 0.01  else
            "*"   if info.get("p", 1) < 0.05  else "",
            # Reliability flag (n_motifs may be None for Spec A; treat as missing)
            "OK" if (info.get("n_trad", 0) >= MIN_TRADITIONS
                     and (info.get("n_motifs") is None
                          or info.get("n_motifs") >= MIN_MOTIFS)
                     and info.get("n_imgs", 0) >= MIN_IMAGES) else "—LOW—",
        ])
    headers = ["biome", "n_trad", "n_motifs", "n_imgs", "Δ", "p", "", "shown?"]
    table = ax.table(cellText=rows, colLabels=headers,
                     loc="center", cellLoc="left",
                     colWidths=[0.36, 0.07, 0.08, 0.08, 0.10, 0.07, 0.06, 0.10])
    table.auto_set_font_size(False); table.set_fontsize(8)
    table.scale(1, 1.3)
    # Style
    for k, cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if k[0] == 0:
            cell.set_facecolor("#222b34")
            cell.set_text_props(color="#9cf", fontweight="bold")
        else:
            cell.set_facecolor("#11141a")
            cell.set_text_props(color="#dddddd")
        # Highlight LOW rows
        if k[0] > 0 and k[1] == 7:
            txt = rows[k[0] - 1][7]
            if txt == "—LOW—":
                for col in range(8):
                    table[(k[0], col)].set_facecolor("#2a1c1c")
                    table[(k[0], col)].set_text_props(color="#cf8a7f")
    ax.set_title("data per biome (LOW = greyed on map)",
                 color="#eeeeee", fontsize=10, loc="left", pad=4)


def _make_fig_with_table(eco, trad, biome_data, title, vmax, out_path):
    """Common layout: wide map | small colorbar gutter | side table.
    Use explicit gridspec so the colorbar lives in its own column and won't
    overlap the table."""
    # 5 columns: [ map | gap1 | colorbar | gap2 | table ]
    # Colorbar tick-labels are moved to the LEFT side so they sit against gap1
    # (the map side) rather than spilling into the table on the right.
    fig = plt.figure(figsize=(24, 7.8))
    gs = fig.add_gridspec(1, 5,
                          width_ratios=[2.6, 0.18, 0.10, 0.40, 1.55],
                          wspace=0.0)
    ax_map = fig.add_subplot(gs[0])
    fig.add_subplot(gs[1]).axis("off")
    ax_cbar = fig.add_subplot(gs[2])
    fig.add_subplot(gs[3]).axis("off")
    ax_tbl = fig.add_subplot(gs[4])
    dark(fig, ax_map)
    fig.patch.set_facecolor("#0c0d11")

    sm = render_map(ax_map, eco, biome_data, trad,
                    title=title, vmax=vmax)
    cbar = fig.colorbar(sm, cax=ax_cbar)
    # Move ticks + label to the LEFT of the bar — no more overlap with table
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label("Δ residualised", color="#bbb", labelpad=8)
    cbar.ax.yaxis.set_tick_params(color="#bbb", labelsize=8)
    cbar.outline.set_edgecolor("#444")
    plt.setp(cbar.ax.get_yticklabels(), color="#bbb")

    sorted_biomes = sorted(biome_data.keys(),
                           key=lambda b: -biome_data[b]["delta"])
    make_legend_table(ax_tbl, biome_data, sorted_biomes)

    # No tight_layout — gridspec already gives us the right placement, and
    # tight_layout was shrinking the gap columns and re-creating the overlap.
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def fig60_headline_with_confidence():
    import geopandas as gpd
    eco = gpd.read_file(SHP).to_crs("EPSG:4326")
    trad = pd.read_parquet(MAP / "traditions.parquet")
    csv = EMB / "combined/biome_test_abstracts_resid.csv"
    if not csv.exists(): print("missing"); return
    bd = per_biome_data(csv)
    _make_fig_with_table(
        eco, trad, bd,
        title="HEADLINE  ·  combined imagery × full Berezkin abstracts × siglip2-large\n"
              "(grey = data too sparse to trust; saturation = (1−p), tradition dots overlaid)",
        vmax=0.0014,
        out_path=FIG / "fig60_effect_map_headline_confidence.png")


def fig61_yfcc_with_confidence():
    import geopandas as gpd
    eco = gpd.read_file(SHP).to_crs("EPSG:4326")
    trad = pd.read_parquet(MAP / "traditions.parquet")
    csv = EMB / "yfcc_filtered/biome_test_abstracts_resid.csv"
    if not csv.exists(): print("missing"); return
    bd = per_biome_data(csv)
    _make_fig_with_table(
        eco, trad, bd,
        title="YFCC-FILTERED landscapes × full Berezkin abstracts × siglip2-large\n"
              "(grey = data too sparse to trust; saturation = (1−p), tradition dots overlaid)",
        vmax=0.0025,
        out_path=FIG / "fig61_effect_map_yfcc_filtered_confidence.png")


def fig62_per_taxon_with_confidence():
    """8-panel grid, greyed for low-confidence."""
    import geopandas as gpd
    eco = gpd.read_file(SHP).to_crs("EPSG:4326")
    trad = pd.read_parquet(MAP / "traditions.parquet")
    n_trad_per_biome = trad.groupby("biome_wwf").size().to_dict()

    csv = EMB / "biome_test_all_byTaxon_resid.csv"
    if not csv.exists():
        print("missing byTaxon"); return
    df = pd.read_csv(csv)
    df = df[df["biome"].apply(lambda x: isinstance(x, str)) & (df["biome"] != "N/A")]

    taxa = ["all", "Insecta", "Mammalia", "Reptilia", "Arachnida",
            "Plantae", "Aves", "Amphibia"]
    taxa = [t for t in taxa if t in df["taxon_group"].unique()]
    vmax = float(np.nanmax(np.abs(df["delta"].values)))

    n = len(taxa); cols = 2; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.5*cols, 4.0*rows))
    fig.patch.set_facecolor("#0c0d11")
    axes = np.array(axes).reshape(rows, cols)

    for k, taxon in enumerate(taxa):
        ax = axes[k // cols, k % cols]
        dark(fig, ax)
        sub = df[df["taxon_group"] == taxon]
        bd = {}
        for _, r in sub.iterrows():
            bd[r["biome"]] = {
                "delta": float(r["delta"]),
                "p": float(r["p_one_sided"]),
                "n_imgs": int(r["n_imgs"]),
                "n_motifs": int(r["n_motifs_in_biome"]) if "n_motifs_in_biome" in r else None,
                "n_trad": int(n_trad_per_biome.get(r["biome"], 0)),
            }
        n_shown = sum(1 for b, info in bd.items()
                      if info["n_trad"] >= MIN_TRADITIONS
                      and (info["n_motifs"] is None or info["n_motifs"] >= MIN_MOTIFS)
                      and info["n_imgs"] >= MIN_IMAGES)
        render_map(ax, eco, bd, traditions=None,
                   title=f"{taxon}  ·  {n_shown}/{len(bd)} biomes shown",
                   vmax=vmax, show_traditions=False)
    fig.suptitle("Per-taxon residualised Δ across WWF biomes — saturation = (1−p), "
                 "grey = data too sparse",
                 color="#eeeeee", fontsize=13, y=0.998)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("RdBu_r"),
                                norm=mcolors.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.10, 0.015, 0.80])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Δ residualised", color="#bbb")
    cbar.ax.yaxis.set_tick_params(color="#bbb")
    cbar.outline.set_edgecolor("#444")
    plt.setp(cbar.ax.get_yticklabels(), color="#bbb")
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))
    out = FIG / "fig62_effect_maps_by_taxon_confidence.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


def main():
    fig60_headline_with_confidence()
    fig61_yfcc_with_confidence()
    fig62_per_taxon_with_confidence()


if __name__ == "__main__":
    main()
