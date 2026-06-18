# Reproducing the v3 paper

This document lists every script in pipeline order, what it produces,
and how to run it. All commands assume the project root as working
directory.

## What's in the repo, what isn't

The repo includes:

- All code (33 scripts under `src/` plus 4 shared utility modules at root).
- The Berezkin spine (traditions, motifs, tradition–motif parquets) and the raw Russian abstracts.
- The LLM-clean English motif text used as input to the v3 embeddings.
- The three v3 motif-text embeddings: sentence-pooled, word-shuffled (control), and class-word-collapsed.
- The Places365 image embeddings + paths.
- All v3 result CSVs (per-biome Δ, breadth, per-taxon, biome-tell, Glottolog swap, two-corpus).
- The paper (LaTeX + bib + PDF + all figures + the methods diagram).

The repo does NOT include (gitignored, too big or external):

- **iNat image embeddings** (`dataset/imagery/embeddings/siglip2-large/inat_basic/img_emb.npy`, 182 MB) — exceeds GitHub's per-file limit. Regenerate by running `src/embedding/embed_images.py` after downloading iNat images, or contact authors for the file.
- **iNat / YFCC / Places365 raw image binaries** — re-download from their original sources via the scripts in `src/acquisition/`.
- **WWF Terrestrial Ecoregions 2017 shapefile** — re-download from [WWF](https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world).
- **Berezkin HTML cache** — already scraped into parquets; if you want to re-scrape, use `src/acquisition/scrape_berezkin.py`.

## Three reproduction tiers

### Tier 1 — Regenerate the PDF figures and paper (5 minutes, no GPU)

Everything needed is already in the repo. Just run the figure scripts
and compile the PDF:

```bash
python src/figures/make_v3_figures.py
python src/figures/taxon_combined.py
python src/figures/taxon_facets.py
cd paper && tectonic paper.tex
```

### Tier 2 — Recompute all Δ analyses (15 minutes, no GPU)

Uses the shipped motif embeddings + Places365 image embeddings. The iNat
img_emb.npy is required and is not in the repo — see "What's in the
repo" above. Once you have it:

```bash
python src/analysis/recompute_all.py
python src/analysis/per_taxon.py
python src/analysis/stratified_baselines.py
python src/analysis/biome_tell.py
python src/analysis/glottolog_swap.py
python src/analysis/word_shuffle.py
python src/analysis/crossmodel.py
# ...then Tier 1 to regenerate the figures and PDF.
```

### Tier 3 — Full reproduction from raw inputs (several hours, GPU recommended)

Requires raw iNat / Places365 image downloads, the WWF shapefile, and
SigLIP-2 + OpenCLIP + M-CLIP embedding passes. Phase-by-phase below.

## Phase 1 — Data acquisition

Run once. These scripts download and stage raw inputs.

| Script | Purpose |
|---|---|
| `src/acquisition/scrape_berezkin.py` | Fetch Berezkin catalogue motif pages from `ruthenia.ru`, parse abstracts and citations, write to `dataset/mapping_v2/`. |
| `src/acquisition/wwf_join.py` | Spatial join tradition and image coordinates to WWF Terrestrial Ecoregions (requires the WWF shapefile in `raw_downloads/Ecoregions2017/`). |
| `src/acquisition/inat_bulk_sample.py` + `src/acquisition/inat_download_images.py` | Sample and download iNaturalist Open Data observations. |
| `src/acquisition/inat_basic_filter.py` | Drop non-nature content via SigLIP-2 zero-shot scoring against negative prompts. |
| `src/acquisition/inat_tag_image_biome.py` | Spatial-join iNat image coordinates to WWF biomes. |
| `src/acquisition/pull_places365_biomes.py` | Download Places365 scenes matching the WWF biome categories, with the strict-landscape SigLIP-2 filter. |
| `src/acquisition/yfcc_download_images.py` + `yfcc_filter_landscape.py` + `yfcc_tag_and_link.py` | YFCC100M Flickr landscape pipeline (used as an additional supplementary corpus). |

## Phase 2 — Build canonical spine and unified image manifest

| Script | Purpose |
|---|---|
| `src/pipeline/build_spine.py` | Build canonical `traditions.parquet`, `motifs.parquet`, and `tradition_motif.parquet` under `dataset/mapping_v2/`. |
| `src/pipeline/build_unified_manifest.py` | Concatenate iNat, YFCC, and Places365 per-source image manifests into one unified manifest under `dataset/imagery/`. |

## Phase 3 — Anonymisation and embedding

The anonymisation pipeline runs as two LLM passes (Gemini 3.5 Flash)
against the raw Russian Berezkin abstracts. The output, the LLM-clean
English motif text, lives at
`dataset/analysis/llm_rewrite_specA_gemini_pass2.csv`. The LLM-pass
scripts themselves live in `_archive/llm_dev/` because they wrap an
external API call and are not part of the offline reproducible
pipeline; the cleaned text is included in the dataset.

| Script | Purpose |
|---|---|
| `src/embedding/sentence_pooled_siglip.py` | Sentence-pool SigLIP-2-large text embeddings over each motif's LLM-clean abstract. Writes `motif_emb_llm_pass2_abstract_sentpooled.npy`. |
| `src/embedding/embed_images.py` | Embed iNat / Places365 / YFCC images with SigLIP-2-large. Writes per-corpus `img_emb.npy`. |
| `src/embedding/embed_openclip.py` | Embed the same LLM-clean text with OpenCLIP-LAION-2B / OpenCLIP-OpenAI / M-CLIP for the cross-model panel. |
| `src/embedding/class_word_collapse.py` | Re-embed the LLM-clean text after collapsing every animal-kingdom class word to "animal" and every plant-kingdom class word to "plant". Used for the per-taxon collapse control. |

## Phase 4 — Δ analysis

| Script | Purpose |
|---|---|
| `src/analysis/recompute_all.py` | Master orchestrator: recompute headline residualised + stratified Δ on iNat and Places365, breadth gradient, per-taxon decomposition. Writes the v3 CSVs under `dataset/imagery/embeddings/siglip2-large/`. |
| `src/analysis/per_taxon.py` | Per-(biome × iconic-taxon) Δ decomposition. Writes `v3_byTaxon_sentpool_iNat.csv`. |
| `src/analysis/stratified_baselines.py` | Within-iconic-taxon stratified versions of the word-shuffle and encyclopedic-null baselines. |
| `src/analysis/biome_tell.py` | High-tell vs low-tell median-split on biome-tell z-score; outputs `v3_biome_tell_split.csv` and `v3_glottolog_swap_null.csv`. |
| `src/analysis/word_shuffle.py` | Bag-of-words sentence-pooled shuffle null. |
| `src/analysis/crossmodel.py` | Compute stratified Δ on M-CLIP, OpenCLIP-LAION-2B, OpenCLIP-OpenAI for the cross-model panel. |
| `src/analysis/realm_block_null.py` | Block-permutation null by WWF biogeographic realm. |
| `src/analysis/glottolog_swap.py` | Within-Glottolog-macroarea biome-swap null (used in the main-text "Cultural-geographic autocorrelation" subsection). |
| `src/analysis/ladder_embed.py` | Identity-naming decomposition (paper §S6): embed the full raw-Russian myth and three *separated* baselines — species, place, and ethnonym bags read from `_entity_extraction/full/` — sentence-pooled with SigLIP-2. Writes `ladder/emb_*.npy` + `ladder/manifest.parquet`. Requires a GPU. |
| `src/analysis/ladder_stats.py` | Per-biome decomposition (full vs each baseline), species-matched permutation null (K-means blocks on the species embedding), and species-subspace projection. Writes `ladder/stats_decomposition.csv` and `ladder/stats_species_matched_null.csv`. |
| `src/analysis/ladder_stats_extra.py` | Place-, ethnonym-, and joint (species+place+ethnonym)-matched permutation nulls on the marginal Δ_full. Writes the remaining `ladder/stats_*_matched_null.csv` and `ladder/stats_matched_null_summary.csv`. |
| `src/analysis/ladder_stats_stratified.py` | The **headline** ladder battery: the entire decomposition + all four matched-permutation nulls + species-subspace projection recomputed on the **within-iconic-taxon stratified** Δ (not the marginal). This is what the main-text identity-naming subsection reports. Writes `ladder/stats_*_strat.csv`. |
| `src/analysis/myth_image_umap.py` | Unsupervised biome-recovery geometry: residualised myth×image cosine (2,158 × 46,481) → PCA-50 → UMAP. Writes `umap_pca50.npy`, `umap_xy.npy`, `umap_biome_retrievability.csv`, and `figS_myth_image_umap.png`. |
| `src/analysis/myth_image_umap2.py` | Three-way characterisation (biome / cultural macro-area / content-taxon) of the same geometry via kNN + linear probe vs a shuffled null. Writes `umap_retrievability3.csv` and `figS_myth_umap_3way.png`. |
| `src/analysis/n_confound_figure.py` | Sampling-size robustness check for the earth map: per-biome marginal vs stratified Δ against the number of traditions per biome. Writes `paper/figures/figS_n_confound.png`. |

## Phase 5 — Figure generation

| Script | Output figure |
|---|---|
| `src/figures/make_v3_figures.py` | Master entry-point — builds fig 2 (two-corpus biome bars), fig 5 (earth map), fig 11 (breadth gradient), fig 9 (cross-model 4×4), fig_v2_controls (robustness atlas, now in supp), figS_biome_tell (supp biome-tell tests). |
| `src/figures/taxon_combined.py` | fig_taxon_combined.png — three-panel taxon decomposition (preserved heatmap + collapsed heatmap + per-taxon double-violin). |
| `src/figures/taxon_facets.py` | fig4_taxon_facets.png — per-taxon facet grid (supplementary). |
| `src/figures/breadth_universals.py` | Standalone breadth-gradient figure helper (called by `make_v3_figures.py`). |
| `src/figures/crossmodel_corr.py` | Cross-model correlation helper. |
| `src/figures/species_composition.py` + `species_composition_specA.py` | Supplementary fig S5 (raw ecological composition per biome, Spec A view). |
| `src/figures/bulletproof_table.py` | Generate supplementary tables. |
| `src/analysis/ladder_figure.py` | fig_identity_naming.png — identity-naming decomposition (main text), built from the **stratified** battery. Reads the `ladder/stats_*_strat.csv` tables; pass the stratified species-subspace projection μΔ values printed by `ladder_stats_stratified.py` as CLI args for panel C (e.g. `... 0.416 0.247 0.195 0.274`). |
| `src/analysis/biome_recovery_figure.py` | fig_biome_recovery.png — unsupervised biome recovery (main text): own-biome retrieval curve, taxon-stratified biome×biome matrix, and per-biome decodability. Reads `umap_pca50.npy` (run `myth_image_umap2.py` first). |

## Phase 6 — Paper compilation

```bash
cd paper
tectonic paper.tex
```

The PDF lands at `paper/paper.pdf`. Replace `tectonic` with any modern
LaTeX engine (xelatex / lualatex / pdflatex with bibtex).

## End-to-end minimum reproduction from cleaned text and embeddings

If you have the cleaned motif text and SigLIP-2 image embeddings already
(they ship with the dataset spine), the minimum pipeline to regenerate
the paper is:

```bash
# Embeddings (sentence-pool the motif text)
python src/embedding/sentence_pooled_siglip.py

# Re-embedding for the class-word collapse control
python src/embedding/class_word_collapse.py

# All Δ analyses
python src/analysis/recompute_all.py
python src/analysis/per_taxon.py
python src/analysis/stratified_baselines.py
python src/analysis/biome_tell.py
python src/analysis/glottolog_swap.py
python src/analysis/word_shuffle.py
python src/analysis/crossmodel.py

# All figures
python src/figures/make_v3_figures.py
python src/figures/taxon_combined.py
python src/figures/taxon_facets.py

# Identity-naming decomposition (main-text fig_identity_naming.png) — needs the
# entity extractions in _entity_extraction/full/ (shipped) and a GPU for the embed
python src/analysis/ladder_embed.py
python src/analysis/ladder_stats.py             # marginal-frame battery (context)
python src/analysis/ladder_stats_extra.py
python src/analysis/ladder_stats_stratified.py  # headline stratified battery
python src/analysis/ladder_figure.py 0.416 0.247 0.195 0.274

# Unsupervised biome recovery (main-text fig_biome_recovery.png)
python src/analysis/myth_image_umap.py
python src/analysis/myth_image_umap2.py
python src/analysis/biome_recovery_figure.py

# Compile PDF
cd paper && tectonic paper.tex
```

## Shared utilities at the project root

`src/` scripts import four shared utility modules that live at the
project root:

- `motif_specificity_controls.py` — `biome_motif_membership_count()` and Spec A control helpers.
- `make_phase2_figures.py` — colour palettes, `short_biome()`, significance star helpers.
- `specA_paper_runs.py` — orchestration helpers for Spec A test variants.
- `make_effect_maps_v2.py` — colormap utilities and confidence-weighting for earth maps.

Each script in `src/` begins with a small `sys.path` bootstrap that
inserts the project root onto the import path, so these utilities
resolve unchanged.

## Archive contents

Anything in `_archive/` is not part of the v3 reproduction set but is
kept locally for reference:

- `legacy_scripts/` — superseded v1 / v2 / v4 / v5 / v6 / v8 hypernym
  variants, older orchestrators, older figure generators, and the
  pareidolia analysis that was removed from v3.
- `audit/` — one-off audit batches and the residue-counting helpers.
- `exploratory/` — `_check_*`, `_inspect_*`, `_investigate_*`,
  `_probe_*`, `_sample_*` diagnostic scripts.
- `llm_dev/` — LLM-development helpers (Gemini API wrappers, NLLB
  translation experiments).
- `working_logs/` — run logs and shell orchestration from earlier
  iterations.
- `mythology_art/` — recursive Commons crawler and Met / Smithsonian
  pulls for the mythology-art follow-up corpus (under construction; not
  used in v3 paper).
