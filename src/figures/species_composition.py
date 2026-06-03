"""_fig_species_composition_per_biome.py — supp-mat figure.

For each WWF biome × each species/plant CATEGORY, compute the % of motifs
whose RAW (pre-anon) Russian abstract mentions at least one species of
that category. Shows the ecological "fingerprint" of each biome's
mythology and clarifies exactly what the v4/v5/v6 hypernyms anonymise.

Output:
  dataset/imagery/figures/headlines_final_russian/figS_species_per_biome.png
  dataset/imagery/figures/headlines_final_russian/species_per_biome_counts.csv
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

from make_phase2_figures import short_biome, biome_color


def main():
    print("Loading abstracts + v6 hypernym …")
    abs_raw = pd.read_parquet(MAP / "motif_abstracts.parquet").fillna("")
    # Build raw abstract_combined per motif (no anon, but no prefix either — we
    # just want the text content for word-counting)
    grouped = (abs_raw.groupby("motif_id")
               .apply(lambda g: " ".join(g["abstract_ru"].tolist()),
                      include_groups=False)
               .reset_index(name="raw_text"))
    motifs_full = pd.read_parquet(MAP / "motifs.parquet")[
        ["motif_id", "name_en", "description_en"]].fillna("")
    motifs_full = motifs_full.merge(grouped, on="motif_id", how="left")
    motifs_full["raw_text"] = motifs_full["raw_text"].fillna("")

    # Motif → biome footprint (count of own-trads per biome)
    tm = pd.read_parquet(MAP / "tradition_motif.parquet")
    trad = pd.read_parquet(MAP / "traditions.parquet")
    tm = tm.copy()
    tm["biome_wwf"] = tm["oid"].map(trad.set_index("oid")["biome_wwf"])
    motif_biomes_long = tm[["motif_id", "biome_wwf"]].dropna()
    motif_biomes_long = motif_biomes_long.drop_duplicates()

    # v6 hypernym map → which words map to which class
    v6 = pd.read_csv(EMB / "hypernym_v6_ru.csv")
    # Group all word_ru forms by class target
    cls2words = (v6.dropna(subset=["word_ru", "class_ru"])
                  .groupby("class_ru")["word_ru"]
                  .apply(lambda s: sorted(set(w.lower() for w in s if isinstance(w, str) and w)))
                  .to_dict())
    # Build compiled per-class regex (single big alternation)
    cls2pat = {}
    for cls, words in cls2words.items():
        if not words: continue
        cls2pat[cls] = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in words) + r")\b",
            re.IGNORECASE)

    # Categories we want to show in the figure, in a stable order
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
    cats = [c for c in CAT_ORDER if c in cls2pat]
    print(f"Categories: {[CAT_LABEL[c] for c in cats]}")

    # For each motif, which categories does its raw text mention?
    print("Tagging each motif with category mentions …")
    motif_has_cat: dict[str, set[str]] = {}
    motif_cat_mentions: list[dict] = []
    for _, r in motifs_full.iterrows():
        text = r["raw_text"].lower()
        if not text:
            motif_has_cat[r["motif_id"]] = set()
            continue
        hits = set()
        cat_n_mentions = {}
        for cat in cats:
            n = len(cls2pat[cat].findall(text))
            if n > 0:
                hits.add(cat)
                cat_n_mentions[cat] = n
        motif_has_cat[r["motif_id"]] = hits
        if cat_n_mentions:
            motif_cat_mentions.append({"motif_id": r["motif_id"],
                                        **{f"n_{c}": cat_n_mentions.get(c, 0)
                                           for c in cats}})

    # For each biome, % of its motifs (any tradition) that mention each cat
    print("Aggregating per-biome category share …")
    biome_motif_sets: dict[str, set[str]] = {}
    for _, r in motif_biomes_long.iterrows():
        biome_motif_sets.setdefault(r["biome_wwf"], set()).add(r["motif_id"])

    rows = []
    for biome, mids in biome_motif_sets.items():
        if not isinstance(biome, str) or biome in ("N/A", ""):
            continue
        n_motifs = len(mids)
        if n_motifs < 5:
            continue
        for cat in cats:
            n_with_cat = sum(1 for mid in mids if cat in motif_has_cat.get(mid, set()))
            rows.append({
                "biome": biome,
                "category_ru": cat,
                "category_en": CAT_LABEL[cat],
                "n_motifs_in_biome": n_motifs,
                "n_motifs_with_cat": n_with_cat,
                "pct_motifs_with_cat": 100.0 * n_with_cat / n_motifs,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "species_per_biome_counts.csv", index=False, encoding="utf-8")
    print(f"  saved CSV ({len(df)} rows)")

    # === Heatmap figure ===
    biomes = sorted(set(df["biome"].tolist()))
    matrix = np.zeros((len(biomes), len(cats)))
    for i, b in enumerate(biomes):
        for j, c in enumerate(cats):
            sel = df[(df["biome"] == b) & (df["category_ru"] == c)]
            if not sel.empty:
                matrix[i, j] = sel.iloc[0]["pct_motifs_with_cat"]
    # Sort biomes by total ecological-content share (for nicer reading order)
    totals = matrix.sum(axis=1)
    order = np.argsort(-totals)
    biomes_ord = [biomes[i] for i in order]
    matrix_ord = matrix[order]

    fig, ax = plt.subplots(figsize=(0.85*len(cats)+5, 0.42*len(biomes_ord)+3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=0, vmax=min(80, matrix_ord.max()))
    im = ax.imshow(matrix_ord, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")
    for i in range(matrix_ord.shape[0]):
        for j in range(matrix_ord.shape[1]):
            v = matrix_ord[i, j]
            r2, g2, b2, _ = cmap(norm(v))
            luma = 0.299*r2 + 0.587*g2 + 0.114*b2
            tc = "#11141a" if luma > 0.55 else "#f4f4f4"
            ax.text(j, i, f"{v:.0f}", color=tc, fontsize=8,
                    ha="center", va="center")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([CAT_LABEL[c] for c in cats],
                        rotation=45, ha="right", color="#222", fontsize=9)
    ax.set_yticks(range(len(biomes_ord)))
    ax.set_yticklabels([short_biome(b) for b in biomes_ord],
                        color="#222", fontsize=9)
    ax.tick_params(colors="#444")
    for sp in ax.spines.values(): sp.set_color("#aaa")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("% motifs in biome mentioning ≥1 word of this category\n"
                   "(raw Russian abstracts, pre-anonymisation)",
                   color="#444", fontsize=9)
    cbar.ax.tick_params(colors="#444")
    cbar.outline.set_edgecolor("#aaa")
    fig.tight_layout()
    out_png = OUT / "figS_species_per_biome.png"
    fig.savefig(out_png, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
