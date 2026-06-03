# Phase 2 — Imagery × mythology: headline results

Run timestamp: 2026-05-28
Model: `google/siglip-base-patch16-224` (768-d, frozen, on RTX 4090)
Pipeline scripts: `inat_bulk_sample.py` → `inat_tag_image_biome.py` →
`inat_download_images.py` → `embed_global.py`

## Inputs

- **Spine (v2):** 958 traditions × 2,564 motifs × 83,565 present (tradition, motif)
  pairs, with WWF Ecoregions2017 biomes (14 biomes, 428 ecoregions, 8 realms),
  documentation-effort controls, language family.
- **Imagery:** 47,900 iNaturalist research-grade photos (CC0 / CC-BY / CC-BY-NC),
  one bbox per tradition centroid, 50 sampled per tradition, dual-tagged with
  the photo's *own* WWF ecoregion (primary link) and the tradition's biome
  (secondary). 13 iconic taxa, 21,553 unique species.

## Test

For each biome **B**:

> Δ = mean cosine sim(images-from-B, mythemes-in-traditions-from-B)
>     − mean cosine sim(images-from-B, mythemes-from-other-biomes)

Null: shuffle the motif → biome assignment, 1000 permutations.

## Headline biome result (all images, all motifs)

| biome | n imgs | Δ | p (1-sided) |
|---|---:|---:|---:|
| **Tropical & Subtropical Coniferous Forests** | 1,109 | **+0.0036** | **0.001** |
| **Tropical & Subtropical Moist Broadleaf Forests** | 14,379 | **+0.0034** | **0.001** |
| **Montane Grasslands & Shrublands** | 1,438 | **+0.0025** | **0.007** |
| **Tropical & Subtropical Grasslands, Savannas & Shrublands** | 5,633 | **+0.0020** | **0.020** |
| Mangroves | 250 | +0.0019 | 0.126 |
| Tropical Dry Broadleaf Forests | 2,304 | +0.0013 | 0.098 |
| Deserts & Xeric Shrublands | 3,711 | +0.0002 | 0.418 |
| Temperate Broadleaf & Mixed Forests | 5,309 | −0.0001 | 0.527 |
| Boreal Forests/Taiga | 2,702 | −0.0002 | 0.591 |
| Temperate Grasslands, Savannas & Shrublands | 3,049 | −0.0007 | 0.748 |
| Tundra | 2,001 | −0.0014 | 0.912 |
| Mediterranean Forests, Woodlands & Scrub | 1,902 | −0.0014 | 0.927 |
| Temperate Conifer Forests | 3,418 | −0.0018 | 0.970 |
| Flooded Grasslands & Savannas | 550 | −0.0019 | 0.952 |

→ `dataset/imagery/embeddings/biome_test_all.csv`

**Reading:** in **tropical and montane** biomes, the SigLIP embedding of a
biome's iNaturalist photos lies measurably closer to that biome's mythemes
than to other biomes' mythemes. The effect inverts (negative Δ) in
**temperate, boreal, Mediterranean, and tundra** biomes.

This **matches the phase-1 text-only result (fig12):** terrestrial pareidolic
motifs concentrate in feature-rich tropical biomes (R² = 0.056, p = 0.0005,
n = 87 traditions). The image side now reproduces the same gradient.

## Robustness 1 — Per-taxon stratification (15 biomes × 11 taxa)

→ `dataset/imagery/embeddings/biome_test_all_byTaxon.csv`

**Significant (p<0.05) cells, sorted by Δ — top 15:**

| taxon | biome | n | Δ | p |
|---|---|---:|---:|---:|
| Amphibia | Tropical Coniferous Forests | 11 | +0.0054 | 0.000 |
| Mammalia | Montane Grasslands & Shrublands | 114 | +0.0051 | 0.000 |
| Amphibia | Tropical Moist Broadleaf Forests | 460 | +0.0049 | 0.000 |
| Reptilia | Tropical Coniferous Forests | 44 | +0.0047 | 0.000 |
| Plantae | Tropical Moist Broadleaf Forests | 2,781 | +0.0044 | 0.000 |
| Arachnida | Tropical Moist Broadleaf Forests | 345 | +0.0043 | 0.000 |
| Mollusca | Tropical Moist Broadleaf Forests | 250 | +0.0042 | 0.000 |
| Reptilia | Tropical Moist Broadleaf Forests | 957 | +0.0042 | 0.000 |
| Insecta | Tropical Moist Broadleaf Forests | 3,973 | +0.0042 | 0.000 |
| Fungi | Tropical Coniferous Forests | 27 | +0.0041 | 0.003 |
| Plantae | Tropical Coniferous Forests | 373 | +0.0040 | 0.001 |
| Fungi | Tropical Moist Broadleaf Forests | 95 | +0.0039 | 0.000 |
| Insecta | Mangroves | 45 | +0.0039 | 0.010 |
| Insecta | Tropical Coniferous Forests | 271 | +0.0039 | 0.002 |
| Animalia | Tropical Moist Broadleaf Forests | 382 | +0.0038 | 0.000 |

**Of 38 significant cells, all 38 are in tropical or montane biomes.** Mediterranean
is negative for every taxon tested (11/11). Temperate Conifer is negative for every
taxon (11/11). This is not driven by one over-represented taxon.

## Robustness 2 — "Creature-like" filter (Aves + Mammalia + Reptilia + Amphibia)

→ `dataset/imagery/embeddings/biome_test_all_creature_like.csv`

Restricting to vertebrate creatures only (drops 32% plants and 21% insects) — the
pattern is preserved with the same 4 tropical/montane biomes significant:

| biome | n imgs | Δ | p |
|---|---:|---:|---:|
| Tropical Coniferous Forests | 423 | +0.0031 | 0.009 |
| Tropical Moist Broadleaf Forests | 5,539 | +0.0024 | 0.018 |
| Montane Grasslands & Shrublands | 640 | +0.0023 | 0.025 |
| Tropical Grasslands, Savannas | 3,464 | +0.0018 | 0.039 |
| Mediterranean | 512 | −0.0024 | 0.982 |
| Temperate Conifer Forests | 704 | −0.0012 | 0.879 |

## Robustness 3 — Per-tradition

→ `dataset/imagery/embeddings/tradition_test_all.csv`

For each tradition with images, count how many of *its* mythemes appear in the
top-50 nearest neighbours of *its* image set. Baseline (chance) = (own motifs / 2,564) × 50.

**Mean enrichment over chance: 1.05× across 958 traditions.**

The top 20 traditions show **6× – 25× enrichment** and are essentially all from
tropical and Pacific cultures (Tujia, Lampung Malay, Tarascan, Aceh, Nahua,
Wallis/Futuna, Mangareva, Loyalty, Piaroa…) — exactly matching the
biome-level pattern.

## Caveats

1. **Effect sizes are small in absolute cosine units** (~0.003 on a baseline of
   −0.035 to −0.040). The signal is statistically robust but the *magnitude*
   is what you'd expect from a single visual model on a multi-causal cultural
   process. Don't over-claim.
2. **Pareidolia filter not yet applied** — these results use ALL 2,564 motifs,
   not the LLM-classified "perception" / "metamorphosis" subset (classifier
   queued to run overnight, `classify_pareidolia_ollama.py`). When the strict
   pareidolia subset is ready, re-run with `--motif-filter perception` or
   `--motif-filter perceptual_broad`.
3. **The temperate/cold negative is real but ambiguous.** Possibilities:
   (a) Berezkin temperate traditions are heavily documented by Western
   ethnographers and may carry import/diffusion content; (b) iNaturalist
   over-represents temperate Western photography of unusual species; (c) cold
   biomes have less visual complexity to project onto. Distinguishing these
   needs more work.
4. **Coverage:** each tradition is one Berezkin centroid; large cultural areas
   (Sami, Aboriginal Australian) get one point. The 0.5°–5° adaptive bbox is a
   reasonable proxy but not the full geographic extent of the tradition.
5. **Visual model is `siglip-base`** (203M params). A larger model
   (`siglip-large` or `siglip2-giga`) would likely sharpen the signal.

## Outputs

```
dataset/imagery/embeddings/
  img_emb.npy                     (47900, 768) float32
  motif_emb_all.npy               (2564, 768) float32
  img_paths.parquet               aligned to img_emb
  motif_meta_all.parquet
  taxon_balance.csv               biome × iconic_taxon counts
  biome_test_all.csv              the headline biome test
  biome_test_all_byTaxon.csv      stratified, 38 significant cells
  biome_test_all_creature_like.csv  vertebrate-only robustness
  tradition_test_all.csv          per-tradition top-50 enrichment
```

## What's queued next

1. **LLM classifier overnight** → re-run all four tests with
   `--motif-filter perception` and `--motif-filter perceptual_broad`.
2. Same tests with `siglip-large` or `siglip2`.
3. Per-realm aggregation (Neotropic vs Palearctic vs Afrotropic) — coarser unit,
   often more statistical power.
4. Held-out cross-validation: predict each tradition's mythemes from images
   alone, score top-K.
