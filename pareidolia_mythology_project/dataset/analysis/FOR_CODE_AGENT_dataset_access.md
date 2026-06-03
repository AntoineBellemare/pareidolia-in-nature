# For the code agent: getting the FULL original Berezkin dataset

## What we currently have (and its limit)

We are working from the `macleginn/mythology-queries` GitHub repo, which contains the
COMPILED MATRIX only:
- `berezkin_new.csv` — 926 traditions × 2,138 motifs, presence/absence (0/1), plus
  metadata columns: lat, lon, area codes, and language family/subfamily (lang1/lang2/lang3).
- motif names + one-line descriptions (`new_descriptions.json`).

What this does NOT include, and what "the full original dataset" refers to:
- The ~70,000 individual TEXT ABSTRACTS (one-paragraph summaries of each specific myth,
  per source) that sit behind each presence cell.
- The SOURCE CITATIONS (which of ~6,500 books/articles each abstract came from).
- The number of tellings supporting each cell (we only have binary present/absent).
- The full motif hierarchy and cross-references.

## Where to get the full version (in priority order)

### 1. Harvard Dataverse — "Replication Data for Electronic Analytical Catalogue" (BEST)
Boston University's folklore guide references a published, citable replication dataset:
  - Search Harvard Dataverse for: "Berezkin Electronic Analytical Catalogue replication"
    or "Berezkin folklore motifs dataset".
  - This is the cleanest machine-readable distribution with a DOI; it is what recent
    quantitative-mythology papers (d'Huy, Tehrani, the 2025 bioRxiv "Worldwide patterns in
    mythology echo the human expansion out of Africa") build on.
  - The 2025 bioRxiv paper's supplementary materials may also contain a processed 781- or
    926-tradition table with their environmental/genetic distance matrices — worth grabbing
    for direct comparison with our partial-Mantel numbers.
  AGENT TASK: locate the Dataverse record, download the .tab/.csv files, and diff the
  motif/tradition coverage against our berezkin_new.csv (we have 926×2138; the catalogue
  has grown to ~958 societies × ~2,564 motifs, so the published set may be larger/newer).

### 2. The primary source — ruthenia.ru (most complete, hardest to parse)
  - URL: http://www.ruthenia.ru/folklore/berezkin
  - This is Berezkin & Duvakin's live catalogue: ~70,000 abstracts, in RUSSIAN, as HTML
    pages organized by motif. It has the actual abstract texts and citations.
  - AGENT TASK: this needs scraping + Russian-language handling. Each motif page lists the
    traditions and the abstract text. Plan: (a) enumerate motif pages, (b) scrape abstracts
    + source refs, (c) machine-translate RU→EN for downstream NLP, (d) key everything by the
    same motif IDs we already use so it joins to our matrix. Respect robots.txt and rate.

### 3. mythologydatabase.com (English front-end, partly paywalled)
  - Searchable English interface over the Berezkin catalogue + ATU index + ~200k narratives,
    with geospatial search. Full search requires a paid/login tier.
  - AGENT TASK: check whether their terms permit bulk export or API access for research; if
    so, it's the easiest English-language route to the abstracts.

### 4. Contact the authors
  - For research use, Berezkin/Duvakin (Kunstkamera / Museum of Anthropology & Ethnography,
    St. Petersburg) have historically shared the catalogue's accompanying data files for
    academic projects. A direct request may get the complete abstract+citation files that
    are "not yet available online" (noted in the Soqotri-narratives paper).

## Why the full dataset matters for THIS project

- The text ABSTRACTS would let us replace the keyword/curation heuristics for "perceptual"
  motifs with actual NLP over the myth descriptions — far more reliable than motif-name
  matching, and it would likely recover more terrestrial pareidolic content than the 10
  motifs we could find from names alone.
- The CITATIONS give a documentation-effort measure per tradition, letting us properly
  control the ethnographic-bias confound (currently only sanity-checked via fig4).
- Per-cell telling COUNTS would turn binary presence into weighted presence, sharpening
  every distance matrix.
- The newer/larger version (958×2564) adds traditions and motifs, increasing power for the
  underpowered terrestrial tests.

## Sanity checks for the agent after downloading

1. Confirm tradition coordinates and language families match ours for overlapping rows
   (our spine is validated; use it as ground truth for the join).
2. Re-key any new motifs to our `motif_list.json` scheme where possible; flag new IDs.
3. Re-run `environment_vs_ancestry.py`, `curate_v2.py`, and `ancestry_and_terrestrial.py`
   pointing at the fuller matrix — results should be consistent and better-powered.
4. If abstracts are obtained: build an NLP "perceptual motif" classifier over abstract text
   (figure-perceived-in-natural-feature) to replace the regex curation, then re-run the
   climate-analog and 3-way Mantel tests on the improved labels.

## Note on what is NOT blocked for you (but was for me)

I (in this sandbox) could only reach GitHub and PyPI. Your code agent on a normal network
can reach Harvard Dataverse, ruthenia.ru, the museum APIs, iNaturalist/AWS, HuggingFace,
and Kaggle — so all of the above, plus the imagery datasets in RESULTS_AND_DATASETS.md, are
available to it. The pipeline scripts (pilot_collect.py, embed_and_analyze.py) are written
to run as-is once it has network access.
