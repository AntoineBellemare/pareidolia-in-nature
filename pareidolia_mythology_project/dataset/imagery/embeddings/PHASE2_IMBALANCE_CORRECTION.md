# Phase 2 — corrected interpretation after imbalance controls

Following a sceptical question about whether the headline fig 2 result could
be an artifact of category imbalance or text-side bias, we ran three
controls (`robustness_imbalance.py`). The results are below; they substantially
weaken (but do not eliminate) the headline claim.

## What we tested

The headline test (fig 2) computed, per biome **B**:

    Δ_B = mean_sim(images-in-B, motifs-in-B's-traditions)
        − mean_sim(images-in-B, motifs-in-OTHER-biomes)

Significance was assessed via a permutation null that **shuffles motif → biome
membership** while holding the image set fixed. This controls for the *count*
of motifs assigned to each biome, but it does NOT control for:

- the **intrinsic image-likeness of a biome's motifs**: if tropical motifs
  use more visually-grounded vocabulary ("jaguar", "snake", "palm") than
  temperate motifs ("hero", "valkyrie", "elf-king"), tropical motifs would
  score higher against *any* nature image, irrespective of biome.
- the **image-side biome label**: maybe the effect would survive even if
  we randomly reassigned images to biomes.

## Controls

**(A) Motif-residualised biome test.** Subtract each motif's grand-mean
similarity over all images (`per_motif_grand = sims.mean(axis=0)`) before
computing Δ. This zeroes out any per-motif "image-likeness" advantage and
isolates the image-biome × motif-biome interaction.

**(B) Intrinsic image-likeness per biome.** For each biome, compute the mean
similarity of its motifs to the FULL image set (not just its own). If
tropical biomes' motifs are intrinsically higher, that's a text-side bias.

**(C) Image-side permutation null.** Shuffle each image's biome label and
re-run the headline test. If most of the observed Δ is reproduced under
shuffled labels, the effect lives in the motif-side bias not in the
biome-image match.

## Result

| biome | original Δ | text bias (B) | **residualised Δ (A)** | p_resid |
|---|---:|---:|---:|---:|
| Tropical Moist Broadleaf Fst | +0.0034 ⭐ | +0.0022 | **+0.0013** | **0.000** |
| Tropical Dry Broadleaf Fst | +0.0013 | +0.0007 | **+0.0006** | **0.000** |
| Temperate Broadleaf & Mixed Fst | −0.0001 | −0.0007 | **+0.0006** | **0.000** |
| Tropical Grasslands, Savannas | +0.0020 ⭐ | +0.0015 | **+0.0005** | **0.003** |
| Deserts & Xeric Shrublands | +0.0002 | −0.0003 | **+0.0005** | **0.000** |
| Montane Grasslands & Shrublands | +0.0025 ⭐ | +0.0021 | **+0.0004** | **0.002** |
| Temperate Grasslands, Savannas | −0.0007 | −0.0011 | **+0.0004** | **0.000** |
| Mediterranean Fst, Woodlands | −0.0014 | −0.0019 | **+0.0004** | **0.000** |
| Mangroves | +0.0019 | +0.0016 | +0.0004 | 0.134 |
| Boreal Fst/Taiga | −0.0002 | −0.0005 | +0.0003 | 0.070 |
| Temperate Conifer Fst | −0.0018 | −0.0019 | +0.0001 | 0.321 |
| Tropical Coniferous Fst | +0.0036 ⭐ | **+0.0035** | +0.0001 | **0.258 n.s.** |
| Tundra | −0.0014 | −0.0013 | −0.0001 | 0.658 |
| Flooded Grasslands & Savannas | −0.0019 | −0.0016 | −0.0003 | 0.899 |

⭐ = was claimed significant in the original fig 2.

The **image-side shuffle null** (control C) returned `null_mean ≈ text_bias`
for every biome, confirming the diagnosis: most of what the original test
measured was the per-biome motif-side baseline, not a biome-image match.

## Corrected interpretation

1. **Tropical Coniferous Forests' significance was an artifact.** Mesoamerican
   mythology's distinctive concrete vocabulary (opossum, jaguar, corn, fire-
   stealing animals, monstrous puma) makes its motifs intrinsically more
   image-like. After removing this baseline, the geographic-match effect is
   not significant (p=0.258).

2. **Tropical Moist Broadleaf Forest is the only biome with a large surviving
   effect.** Δ_residual = +0.0013, p<0.001 — about 40% of the original Δ
   remains. This is the cleanest evidence for the hypothesis in the imagery
   data, and it specifically reproduces the Amazon-as-hotspot finding from
   the phase-1 text result.

3. **Smaller positive residual effects emerge in non-tropical biomes that
   were previously called negative**: Temperate Broadleaf, Mediterranean,
   Temperate Grasslands, Deserts all show statistically significant residual
   Δ around +0.0004. After de-biasing, the picture is NOT "tropics vs cold" —
   it's "Tropical Moist Broadleaf clearly stands out, with smaller signals in
   several other biomes."

4. **No biome shows a significant negative residualised Δ.** Tundra and
   Flooded Grasslands are flat. This means we no longer have evidence that
   "temperate cultures' mythology is uniquely UN-pareidolic" — that part of
   the original story was the text-bias mirror image of the tropical-positive
   inflation.

5. **The hypothesis is supported but weaker than initially claimed.** The
   image × biome interaction exists (the residual is highly significant in
   one biome and modestly in several others) but the absolute magnitudes are
   ~0.0001–0.0013 cosine, ~25–40% of what fig 2 implied.

## Update to fig 2 caption

A revised fig 2 should show **delta_residualised** with p_residualised stars,
not the original delta. See `fig10_imbalance_controls.png` panel 2 (top right)
for the corrected version.

## What's not yet ruled out

- **Geographic clustering of the iNat photographer base** within tropics.
  Tropical photos may come from a narrower distribution of contributors who
  share visual style.
- **SigLIP training bias toward particular biomes.** The model has seen
  more web-tagged tropical imagery than tundra imagery; the cosine geometry
  is not uniform across biomes.
- **The motif descriptions are Berezkin's short summaries, not the full
  ~70k-page abstracts.** A richer text input might recover stronger
  per-biome interactions.

## Recommended actions

1. **Replace fig 2 with the residualised version** in any communication
   beyond this repo.
2. **Re-run with the LLM pareidolia classifier** (`--motif-filter perception`)
   when the overnight Ollama run completes — if the residual effect grows
   stronger when restricted to genuinely pareidolic motifs (vs all mythology),
   that's a critical hypothesis test.
3. **Re-run with a larger SigLIP** (`siglip-large` or `siglip2`) to test
   whether a more capable model amplifies the residual interaction.
4. **Add per-realm aggregation** (Neotropic, Palearctic, etc.) as a coarser
   alternative to the 14 biomes — fewer cells, more power per cell.

## Files

- `dataset/imagery/embeddings/biome_test_residualised.csv`
- `dataset/imagery/embeddings/biome_image_likeness.csv`
- `dataset/imagery/embeddings/biome_image_shuffle_null.csv`
- `dataset/imagery/figures/fig10_imbalance_controls.png`
