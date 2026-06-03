# Methods figure — design specification (v2, visual-first)

## What this figure must communicate

The pipeline takes a raw Russian motif, strips it of every biome-anchoring
word, embeds it, and tests its alignment against biome imagery on the
image side controlled by iconic taxon. The reader should grasp this in
**three seconds of looking**, without reading any body text in the
figure.

## Constraints

- **Total figure body text under 80 words.** If a stage needs more than
  one short label plus its example, the stage is overweighted.
- **Every stage shows the example transforming visibly.** Stages where
  the example doesn't change carry no information.
- **Show, don't tell.** Use icons, glyphs, swatches, and tiny
  thumbnails. Reserve words for one-line stage labels and one-line
  example deltas.

## Layout

A single horizontal pipeline, five panels left-to-right, plus a
short branch off Stage 4 to the image side.

```
[Stage 0 raw]→[Stage 1 anonymise]→[Stage 2 refine]→[Stage 3 embed]→[Stage 4 stratify]
                                                                            │
                                                                            ↓
                                                                  [iconic-taxon image strip]
```

No outer figure title. Panel headers only. White background.

## Per-stage content

For each stage: **one icon, one 3–4-word label, the example in
present form**. Nothing else.

### Stage 0 — Raw input
- Icon: opened book / catalogue glyph.
- Label: "Berezkin motif (RU)"
- Example (Cyrillic, 1 line): the same short Russian sentence as below,
  with three words rendered in red, no underline, no boxes:
  - **медведь** (bear → biome-anchor noun)
  - **шаман** (shaman → cultural anchor)
  - one place/ethnonym in brackets, e.g. **[Маккензи]**

### Stage 1 — LLM anonymisation
- Icon: small Gemini / LLM glyph (a stylised AI chip or asterisk).
- Label: "Class word + placeholder"
- Example (English, 1 line) — same sentence translated, with the
  three red tokens **replaced and visibly recoloured to blue**:
  - **bear** → **mammal**
  - **[Маккензи]** → **[people]**
  - (any biome word, e.g. taiga, simply *gone* — render the dropped
    word with a strike-through and faded grey at half-opacity)

### Stage 2 — Anchor refinement
- Icon: same LLM glyph but with a small "pass 2" badge.
- Label: "Cultural anchors → generic"
- Example: only one further swap, but it must be visible:
  - **shaman** → **character** (or "ruler" / "holy person", pick the
    one in your example)
  - Render the new token in a third colour, e.g. **purple**, so the
    reader sees this stage doing something distinct from Stage 1.
- The rest of the sentence is unchanged from Stage 1.

### Stage 3 — Sentence-pooled SigLIP-2 embedding
- This is where the figure becomes a diagram, not a text caption.
- Take the Stage 2 sentence and **physically split it into 3 small
  rounded rectangles**, each containing one short clause:
  - "[people]: Woman gathers berries."
  - "Steps three times in mammal dung."
  - "Calls the character."
- Each rectangle has an arrow into a single SigLIP-2 chip icon (a
  small rounded box labelled "SigLIP-2 / 64 tok"), and the chip
  produces a row of three small vector glyphs (think tiny coloured
  bars or dots, 2-3 mm tall).
- The three vector glyphs flow into a "mean ⊕" symbol producing
  a single fatter vector glyph, the motif vector.
- Label under the chip: "Mean-pool → motif vector"
- No further text.

### Stage 4 — Image-side stratification
- Stage 4 is a vertical strip *coming off* the motif vector.
- The motif vector arrow hits a column of 4-5 small iconic-taxon
  rows. Each row has a tiny representative thumbnail/icon:
  - 🐻 Mammalia
  - 🦅 Aves
  - 🌿 Plantae
  - 🦎 Reptilia
  - 🐸 Amphibia
- Each row has a small Δ symbol next to it. A bracket joins the
  Δ symbols on the right and points to a final "Δ̄ uniform mean"
  glyph.
- Label under the column: "Δ per taxon → uniform mean"
- No other text.

## Colour code (shared across the figure)

- Red: biome-anchor tokens that the pipeline targets (Stage 0
  highlight).
- Blue: their generic replacements after LLM Pass 1.
- Purple: the additional replacements at LLM Pass 2.
- Grey strikethrough: tokens dropped (e.g. biome words).
- Earth-tone backgrounds for stage panels, getting progressively
  warmer left-to-right (cool grey → tan → terracotta), so the eye
  travels naturally across the pipeline.

## What to avoid

- No paragraphs of rule text inside the stage panels. Reserve them
  for the caption.
- No counts or sample sizes inside the panels — they go in the
  caption.
- No repetition: each stage must show a *visibly different*
  example state. If a stage doesn't move the example, the stage
  doesn't belong here.
- No "Δ computed in each stratum, averaged uniformly. (14 biomes ×
  11 taxa = 154 strata)" body text — that level of detail goes in
  the caption.
- No outer figure title.

## Reference example to thread through all stages

Use this single sentence as the example through Stages 0–3.

- **Stage 0 (RU)**: «Маккензи: женщина собирает ягоды, трижды
  наступает в **медвежий** помёт, зовёт **шамана**.»
- **Stage 1 (EN, anonymised)**: "[people]: Woman gathers berries,
  steps three times in **mammal** dung, calls the **shaman**."
- **Stage 2 (EN, refined)**: "[people]: Woman gathers berries,
  steps three times in **mammal** dung, calls the **character**."
- **Stage 3**: the Stage 2 sentence broken into three clauses,
  each embedded, mean-pooled.
- **Stage 4**: that motif vector compared against per-taxon iNat
  image rows.

The change is visible at *every* stage, and the reader can follow
one token's journey from "медведь" (red) to "mammal" (blue) to
unchanged → embedded → mean-pooled → tested against animal photos.

## Caption (to be written in LaTeX, not in the figure)

Two short sentences: "Pipeline schematic. A motif's Russian
abstract is translated and anonymised under noun-targeted rules,
sentence-pooled by SigLIP-2 into a single text vector, then
compared against biome imagery within each iconic-taxon stratum,
with Δ averaged uniformly across strata to control for image-side
taxon composition."
