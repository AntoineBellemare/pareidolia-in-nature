# What data do we actually have, and what can it answer?

## The data we have RIGHT NOW (real, on disk, queryable)

Everything below was built from the **Berezkin Analytical Catalogue of World
Mythology and Folklore**, obtained from the open `mythology-queries` GitHub
repository, then enriched with **Köppen-Geiger climate data** (real climate
observations) for biome classification.

| File | What it is | Rows |
|---|---|---|
| `traditions.parquet` | every mythological tradition with lat/lon, Köppen code, biome | 926 |
| `motifs.parquet` | every mytheme with name, description, and a pareidolic flag | 2,138 |
| `tradition_motif.parquet` | which motifs are present in which tradition (long format) | 68,837 |

This is a **complete, geo-located, machine-readable map of world mythology**:
926 cultures, each pinned to a location and climate, each with the full list
of the mythological motifs it contains, and each motif described in plain
language. That is a substantial, real dataset on its own.

## The crucial distinction: this data is TEXT/STRUCTURE only

Here is the honest answer to your question about imagery vs. text.

**What we have is entirely on the TEXT + STRUCTURE side.** It tells us, for
every culture, *what stories and figures exist and where on Earth they are*.
It contains **zero images** — no landscape photos, no mythological art. The
imagery pillars (iNaturalist landscapes, museum art) are still un-run scripts;
nothing has been downloaded.

So the dataset as it stands today can answer **text/geography questions**, and
it **cannot yet answer imagery questions**. Concretely:

### Questions this data CAN answer now (text + geography)
- Which mythemes are universal vs. region-specific? (e.g., "Trickster-fox" in
  339 traditions; "Figure on lunar disc" in 337)
- Do "seeing-a-figure-in-nature" motifs cluster geographically or by climate?
  **Yes — and we measured it** (see results below).
- For any biome or coordinate, what mythological figures should we expect?
  (This is what makes the imagery step possible later — it gives the *targets*.)
- How does motif richness vary with latitude? (a documentation-bias check)
- Which specific motifs are the most directly "pareidolic" and therefore the
  best test cases to later hunt for in actual images?

### Questions this data CANNOT answer yet (need the imagery pillar)
- Does a Norwegian fjord *look like* a Norse serpent to a vision model? — needs
  landscape images + mythological art images, then the embedding comparison in
  `embed_and_analyze.py`.
- Do humans from different cultures perceive different creatures in the same
  landscape? — needs the human-study pillar (not built yet).
- Can we *reconstruct* a region's mythology from its imagery? — the headline
  question; needs both imagery pillars joined to this spine.

In short: **we have built the skeleton and the target list. We have not yet
collected the pictures.** The text data is what makes the eventual imagery
work answerable, because it tells the vision models what to look for and where.

## Preliminary RESULTS (text side only — see figures/)

We flagged motifs whose descriptions involve *perceiving a being or figure in
a natural feature* (moon, sun, stars, clouds, rocks, mountains, rainbows, sky,
shadows). **299 of ~1,800 named motifs (16%) are pareidolic in exactly your
hypothesized sense** — and they are present in hundreds of cultures each.

**Headline preliminary finding:** the prevalence of these pareidolic mythemes
varies systematically by biome (`fig2`). Tropical biomes carry significantly
*more* figure-in-nature motifs (tropical moist forest 22.7%, tropical savanna
21.6%) than the global mean (17.4%), while deserts and cold continental forests
carry significantly fewer (~13%). The confidence intervals don't overlap the
mean for the extremes, so this is a real pattern in the data, not noise.

This is genuinely suggestive for your hypothesis: *the kind of environment a
culture lives in is associated with how much of its mythology is built from
seeing-figures-in-nature.* But read the caveats.

### Figures
- `fig1_world_traditions.png` — all 926 traditions on a world map, colored by
  biome. (Continental shapes emerge from the points; coverage is global.)
- `fig2_pareidolic_by_biome.png` — **the key result**: pareidolic-motif
  prevalence per biome with Wilson 95% CIs.
- `fig3_top_pareidolic.png` — the most widespread figure-in-nature mythemes
  ("Figure on lunar disc", "Man in the Moon", "Stars are people", ...).
- `fig4_motif_richness.png` — motif richness vs latitude, a check on whether
  some regions are simply better documented (a real confound — see below).

## Caveats you must keep in mind

1. **No imagery yet.** Every result here is from text/structure. The pareidolia
   claim is currently about *how mythemes are described*, not about whether
   landscapes actually look like the creatures.

2. **The pareidolic flag is keyword-based.** It catches "moon", "rock", "star",
   etc. in descriptions. It will have false positives (a motif mentioning the
   moon without any figure-seeing) and false negatives. A human pass over the
   299 flagged motifs would sharpen this a lot.

3. **Biome = climate proxy, not ecoregion polygon.** We could not reach the WWF
   shapefile from this environment (the host isn't network-whitelisted here), so
   biomes come from Köppen-Geiger climate classification instead. This uses real
   climate data and is far better than a latitude guess, but it is not identical
   to the WWF vegetation/ecoregion map. The provided `wwf_join.py` does the true
   point-in-polygon join the moment you have the shapefile locally — run it and
   the biome column upgrades in place.

4. **Documentation bias is a live confound.** Tropical/Amazonian and Siberian
   societies may have more (or fewer) recorded motifs for reasons of
   ethnographic history, not mythology. `fig4` is the first look at this; control
   for total-motif-count before over-interpreting `fig2`.

5. **68 traditions got "unknown" biome** because their coordinates fall on
   coastline/water cells in the climate grid. The WWF join or a nearest-land
   snap will recover most of them.

## What unlocks the imagery questions

To move from "text only" to actually testing the visual hypothesis:
1. Run `pilot_collect.py` (collects Norse texts + art + nature images).
2. Run `embed_and_analyze.py` (embeds and runs the landscape↔myth signal check).
Both need network access to iNaturalist / the Met / Wikidata, which your own
machine has and this sandbox does not.
