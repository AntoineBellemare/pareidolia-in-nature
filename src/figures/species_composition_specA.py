"""_fig_species_composition_specA.py — S6b.

Same as figS_species_per_biome.png BUT restricted to Spec A motifs
(motifs touching ≤ 3 biomes AND having ≥ 3 own-traditions in this
biome). This makes the heatmap actually discriminate biomes — the
current full-corpus version is uniformly red because universal motifs
appear in every biome's motif-set.

Read as: "of the BIOME-SPECIFIC motifs that anchor biome b, what
fraction mention at least one species/plant of category c in their raw
Russian abstract".

Output:
  dataset/imagery/figures/headlines_final_russian/figS_species_per_biome_specA.png
  dataset/imagery/figures/headlines_final_russian/species_per_biome_counts_specA.csv
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[2]  # project root
MAP = ROOT / "dataset/mapping_v2"
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
OUT = ROOT / "dataset/imagery/figures/headlines_final_russian"

from make_phase2_figures import short_biome


CAT_ORDER = ["птица", "млекопитающее", "рептилия", "амфибия", "рыба",
             "насекомое", "паукообразное", "моллюск", "ракообразное",
             "червь", "дерево", "растение", "цветок", "животное"]
CAT_LABEL = {
    "птица": "Birds",
    "млекопитающее": "Mammals",
    "рептилия": "Reptiles",
    "амфибия": "Amphibians",
    "рыба": "Fish",
    "насекомое": "Insects",
    "паукообразное": "Arachnids",
    "моллюск": "Molluscs",
    "ракообразное": "Crustaceans",
    "червь": "Worms",
    "дерево": "Trees",
    "растение": "Plants",
    "цветок": "Flowers",
    "животное": "Animal (other)",
}


def main():
    print("Loading abstracts + v6 hypernym …", flush=True)
    abs_raw = pd.read_parquet(MAP / "motif_abstracts.parquet").fillna("")
    grouped = (abs_raw.groupby("motif_id")
               .apply(lambda g: " ".join(g["abstract_ru"].tolist()),
                      include_groups=False)
               .reset_index(name="raw_text"))
    motifs_full = pd.read_parquet(MAP / "motifs.parquet")[
        ["motif_id", "name_en", "description_en"]].fillna("")
    motifs_full = motifs_full.merge(grouped, on="motif_id", how="left")
    motifs_full["raw_text"] = motifs_full["raw_text"].fillna("")

    # === Motif → biome footprint AND own-tradition counts ===
    tm = pd.read_parquet(MAP / "tradition_motif.parquet")
    trad = pd.read_parquet(MAP / "traditions.parquet")
    tm = tm.copy()
    tm["biome_wwf"] = tm["oid"].map(trad.set_index("oid")["biome_wwf"])
    motif_to_biomes = (tm.dropna(subset=["biome_wwf"])
                         .groupby("motif_id")["biome_wwf"]
                         .apply(lambda s: set(s))
                         .to_dict())
    cnt = (tm.dropna(subset=["biome_wwf"])
             .groupby(["motif_id", "biome_wwf"]).size()
             .unstack(fill_value=0))
    motif_biome_owntrads = {mid: dict(row.items()) for mid, row in cnt.iterrows()}

    # === Spec A: motif breadth ≤ 3 ===
    specA_motifs = {mid for mid, bs in motif_to_biomes.items() if len(bs) <= 3}
    print(f"Spec A motifs (≤3 biomes): {len(specA_motifs)} / {len(motif_to_biomes)}",
          flush=True)

    # === v6 hypernym map → which words map to which class ===
    v6 = pd.read_csv(EMB / "hypernym_v6_ru.csv")
    cls2words = (v6.dropna(subset=["word_ru", "class_ru"])
                  .groupby("class_ru")["word_ru"]
                  .apply(lambda s: sorted(set(w.lower() for w in s
                                               if isinstance(w, str) and w)))
                  .to_dict())
    cls2pat = {cls: re.compile(r"\b(" + "|".join(re.escape(w) for w in words)
                                + r")\b", re.IGNORECASE)
               for cls, words in cls2words.items() if words}
    cats = [c for c in CAT_ORDER if c in cls2pat]
    print(f"Categories: {[CAT_LABEL[c] for c in cats]}", flush=True)

    # === Tag each motif with the categories its raw text mentions ===
    print("Tagging each motif with category mentions …", flush=True)
    motif_has_cat: dict[str, set[str]] = {}
    for _, r in motifs_full.iterrows():
        text = r["raw_text"].lower()
        if not text:
            motif_has_cat[r["motif_id"]] = set()
            continue
        motif_has_cat[r["motif_id"]] = {
            cat for cat in cats if cls2pat[cat].search(text)
        }

    # === For each biome, % of its Spec A motifs (with ≥3 own-trads in this biome)
    # that mention each category ===
    print("Aggregating per-biome category share within Spec A subset …",
          flush=True)
    biomes_set = sorted({b for s in motif_to_biomes.values() for b in s
                          if isinstance(b, str) and b != "N/A"})

    rows_all = []
    rows_spec = []
    for biome in biomes_set:
        # ALL motifs touching this biome
        all_in_biome = {mid for mid, bs in motif_to_biomes.items() if biome in bs}
        # Spec A in this biome: breadth ≤ 3 AND ≥ 3 own-trads in this biome
        spec_in_biome = {mid for mid in all_in_biome
                          if mid in specA_motifs
                          and motif_biome_owntrads.get(mid, {}).get(biome, 0) >= 3}
        for cat in cats:
            n_with_cat_spec = sum(
                1 for mid in spec_in_biome if cat in motif_has_cat.get(mid, set()))
            n_with_cat_all = sum(
                1 for mid in all_in_biome if cat in motif_has_cat.get(mid, set()))
            rows_spec.append({
                "biome": biome,
                "category_ru": cat,
                "category_en": CAT_LABEL[cat],
                "n_motifs_specA_in_biome": len(spec_in_biome),
                "n_motifs_specA_with_cat": n_with_cat_spec,
                "pct_specA_with_cat": (
                    100.0 * n_with_cat_spec / len(spec_in_biome)
                    if spec_in_biome else 0.0),
            })
            rows_all.append({
                "biome": biome,
                "category_ru": cat,
                "category_en": CAT_LABEL[cat],
                "n_motifs_all_in_biome": len(all_in_biome),
                "n_motifs_all_with_cat": n_with_cat_all,
                "pct_all_with_cat": (
                    100.0 * n_with_cat_all / len(all_in_biome)
                    if all_in_biome else 0.0),
            })

    df_spec = pd.DataFrame(rows_spec)
    df_all = pd.DataFrame(rows_all)
    df = df_spec.merge(
        df_all[["biome", "category_ru", "n_motifs_all_in_biome",
                "n_motifs_all_with_cat", "pct_all_with_cat"]],
        on=["biome", "category_ru"])
    # Also: difference (Spec A % minus universal %), reveals over-/under-rep
    df["pct_diff_specA_minus_all"] = df["pct_specA_with_cat"] - df["pct_all_with_cat"]
    df.to_csv(OUT / "species_per_biome_counts_specA.csv", index=False,
              encoding="utf-8")
    print(f"  saved CSV ({len(df)} rows)", flush=True)

    # === Heatmap: rows = biomes (sorted by Spec A motif count), cols = cats ===
    biomes_ord = (df.groupby("biome")["n_motifs_specA_in_biome"]
                  .first().sort_values(ascending=False).index.tolist())
    mat_spec = np.zeros((len(biomes_ord), len(cats)))
    mat_all = np.zeros_like(mat_spec)
    mat_diff = np.zeros_like(mat_spec)
    n_spec_by_biome = []
    for i, b in enumerate(biomes_ord):
        sub_b = df[df["biome"] == b]
        n_spec_by_biome.append(int(sub_b["n_motifs_specA_in_biome"].iloc[0]))
        for j, c in enumerate(cats):
            row = sub_b[sub_b["category_ru"] == c]
            if not row.empty:
                mat_spec[i, j] = row.iloc[0]["pct_specA_with_cat"]
                mat_all[i, j] = row.iloc[0]["pct_all_with_cat"]
                mat_diff[i, j] = row.iloc[0]["pct_diff_specA_minus_all"]

    # === Plot 2 panels: Spec A % + Difference (omit the all-motifs panel,
    # which is uniformly red and uninformative) ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.42 * len(biomes_ord) + 3.5))
    fig.patch.set_facecolor("white")

    def _heat(ax, mat, title, cmap_name, vmin, vmax, fmt="{:.0f}",
              show_y=True):
        ax.set_facecolor("white")
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto",
                       interpolation="nearest")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                r2, g2, b2, _ = cmap(norm(v))
                luma = 0.299*r2 + 0.587*g2 + 0.114*b2
                tc = "#11141a" if luma > 0.55 else "#f4f4f4"
                ax.text(j, i, fmt.format(v), color=tc, fontsize=7.8,
                        ha="center", va="center")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([CAT_LABEL[c] for c in cats],
                           rotation=45, ha="right",
                           color="#222", fontsize=9)
        if show_y:
            ax.set_yticks(range(len(biomes_ord)))
            ax.set_yticklabels([f"{short_biome(b)}  (n={n_spec_by_biome[i]})"
                                for i, b in enumerate(biomes_ord)],
                               color="#222", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.tick_params(colors="#444")
        for sp in ax.spines.values(): sp.set_color("#aaa")
        ax.set_title(title, color="#111", fontsize=10.5, fontweight="bold",
                     loc="left", pad=8)
        return im

    im1 = _heat(axes[0], mat_spec,
                "Spec A subset (≤3 biomes, ≥3 own-trads per biome)\n"
                "% biome-specific motifs mentioning ≥1 word of category",
                "YlOrRd", 0, max(80, mat_spec.max()),
                show_y=True)
    abs_diff_max = max(20, float(np.percentile(np.abs(mat_diff), 95)))
    im2 = _heat(axes[1], mat_diff,
                "Difference (Spec A − all-in-biome)\n"
                "+ : category over-represented in biome-specific vs universals",
                "RdBu_r", -abs_diff_max, abs_diff_max,
                fmt="{:+.0f}", show_y=False)

    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.55, pad=0.02)
    cbar1.set_label("%", color="#444"); cbar1.ax.tick_params(colors="#444")
    cbar1.outline.set_edgecolor("#aaa")
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.55, pad=0.02)
    cbar2.set_label("Δ %", color="#444"); cbar2.ax.tick_params(colors="#444")
    cbar2.outline.set_edgecolor("#aaa")

    fig.tight_layout()
    out_png = OUT / "figS_species_per_biome_specA.png"
    fig.savefig(out_png, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
