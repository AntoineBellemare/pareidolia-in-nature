# The Imagination of Nature

### Mythology as the Cultural Recording Layer of Perceptual Coupling

A study testing whether the mythologies of cultures inhabiting different
biomes carry a recoverable visual signature of those biomes, **even after
every species name, place name, ethnonym, and biome word is stripped from
the text**.

The pattern is consistent with perceptual coupling at the lexical-thematic
level: the residue mythology carries of its biome is not a catalogue of
names but the schemas of attention, activity, and form that the biome
made available to imagination.

---

## What the study finds

Anonymised motif text from Berezkin's catalogue of 958 cross-cultural
folk traditions aligns above shuffle-permutation chance with images of
the biomes those traditions inhabit, on two independent visual corpora
(iNaturalist species photographs and Places365 landscape scenes) and
across four vision–language model architectures (SigLIP-2, M-CLIP,
OpenCLIP-LAION-2B, OpenCLIP-OpenAI). The alignment **decays monotonically
with motif breadth** — strongest where the mythology is biome-specific,
fading toward zero where it has propagated across most of the world.

The breadth gradient is the load-bearing signature. Biome-specific motifs
stabilised in encounter with a particular ecology and carry that
encounter forward in their narrative substance. Universal motifs
propagated only by detaching from any single biome, and the coupling
fades in them precisely as a lineage from Schelling and Bateson through
contemporary 4E cognitive science would predict.

A class-word collapse control — in which every animal-kingdom class word
in the cleaned text is further reduced to `animal` and every
plant-kingdom word to `plant` — attenuates the per-taxon alignment for
mammals, reptiles, and amphibians (whose mythological presence is built
mostly from naming) but leaves the alignment for plants, fungi, birds,
and insects largely intact (their mythological presence is built more
from activities and contexts than from names).

A within-Glottolog-macroarea biome-swap null, holding cultural region
constant while shuffling biome assignment, attenuates the effect by
roughly half but does not eliminate it. Geographic and cultural
autocorrelation contribute to the alignment but do not exhaust it.

---

## Headline figures

| | |
|---|---|
| **Fig 2** — Two-corpus replication of biome × mythology alignment on iNaturalist (stratified) and Places365 (marginal). | ![fig2](paper/figures/fig2_biome_bars.png) |
| **Fig 5** — Per-biome stratified Δ projected onto a world map, full LLM-clean corpus (A) and Spec A subset (B). | ![fig5](paper/figures/fig5_earth_map.png) |
| **Fig 11** — Breadth gradient: stratified μΔ across biomes falls from +0.58 ×10⁻³ in biome-specific motifs to +0.12 ×10⁻³ in universals. | ![fig11](paper/figures/fig11_universals_analysis.png) |

---

## How to reproduce

The full pipeline runs in six phases, documented in
[`REPRODUCE.md`](REPRODUCE.md). The minimum end-to-end reproduction
from the shipped cleaned text and image embeddings is:

```bash
# Phase 3 — sentence-pool the motif text, plus the class-word collapse re-embed
python src/embedding/sentence_pooled_siglip.py
python src/embedding/class_word_collapse.py

# Phase 4 — Δ analyses
python src/analysis/recompute_all.py
python src/analysis/per_taxon.py
python src/analysis/stratified_baselines.py
python src/analysis/biome_tell.py
python src/analysis/glottolog_swap.py
python src/analysis/word_shuffle.py
python src/analysis/crossmodel.py

# Phase 5 — all paper figures
python src/figures/make_v3_figures.py
python src/figures/taxon_combined.py
python src/figures/taxon_facets.py

# Phase 6 — compile the PDF
cd paper && tectonic paper.tex
```

If you want to reproduce from scratch (no cleaned text, no embeddings),
run the data acquisition scripts under `src/acquisition/` and the
build-spine + unified-manifest scripts under `src/pipeline/` first.

---

## Repository layout

```
pareidolia_mythology_project/
├── README.md              ← you are here
├── REPRODUCE.md           ← step-by-step pipeline
│
├── paper/                 ← LaTeX manuscript + figures + PDF
├── dataset/               ← canonical data spine + embeddings + manifests
├── raw_downloads/         ← Berezkin HTML cache, WWF Ecoregions shapefile
│
├── src/                   ← all paper code, organised by pipeline phase
│   ├── acquisition/       ← Berezkin scraper, iNat / YFCC / Places365 downloaders, WWF join
│   ├── pipeline/          ← spine + unified image manifest construction
│   ├── embedding/         ← sentence-pooled SigLIP-2, cross-model, class-word collapse
│   ├── analysis/          ← Δ tests, breadth, per-taxon, biome-tell, swap null
│   └── figures/           ← paper figure generators
│
├── motif_specificity_controls.py   ← shared utility modules at root
├── make_phase2_figures.py          ←   imported by 30+ scripts in src/
├── specA_paper_runs.py             ←   under their original names
├── make_effect_maps_v2.py          ←   to keep the import surface stable
│
└── _archive/              ← legacy / superseded scripts kept locally, .gitignored
```

The four shared utility modules live at the project root because they
are imported by 30+ scripts under `src/`. Each `src/` script begins with
a small `sys.path` bootstrap that inserts the project root onto the
import path, so `from motif_specificity_controls import ...` resolves
unchanged.

---

## Heavy data and the .gitignore

The image-side embeddings (`dataset/imagery/embeddings/*.npy`), iNat /
YFCC / Places365 image binaries, the Berezkin HTML cache, and the WWF
Ecoregions shapefile are not version-controlled. See `.gitignore` for
the precise exclusions. The canonical motif text
(`dataset/analysis/llm_rewrite_specA_gemini_pass2.csv`) and the
tradition / motif spine parquets (`dataset/mapping_v2/`) are tracked.

---

## Author

**Antoine Bellemare-Pepin** — Department of Psychology, Université de Montréal  
**Mar Estarellas** — Department of Psychiatry, McGill University

## Citation

> Bellemare-Pepin A, Estarellas M. *The Imagination of Nature:
> Mythology as the Cultural Recording Layer of Perceptual Coupling*.
> Preprint, 2026.

## Acknowledgement

This project uses Yu. E. Berezkin's *Analytical Catalogue of World
Mythology and Folklore* (ruthenia.ru), the WWF Terrestrial Ecoregions
of the World, iNaturalist Open Data, Places365, and the YFCC100M
dataset.
