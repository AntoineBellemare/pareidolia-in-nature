# Principled pareidolia classification — what changed

## The problem with how we defined it before

The original `pareidolic` flag was a keyword match: a motif counted if its text mentioned
moon/sun/star/rock/mountain/etc. That conflates three different things. The new classifier
(`classify_pareidolia.py`) separates them into four classes:

- **PERCEPTION** — a figure is *perceived in / read into* an ambiguous natural feature
  ("a figure is seen in the moon", "the rock looks like a man"). **This is true pareidolia.**
- **METAMORPHOSIS** — a being *became / was turned into* a feature ("turned to stone",
  "became a constellation"). Pareidolia-adjacent etiology — often attached to a striking
  real feature, but the text asserts transformation, not perception.
- **MENTION** — a feature appears but no figure is perceived and nothing becomes it
  ("several suns burn the earth", "theft of the sun"). **Not pareidolia.**
- **NONE** — no celestial/landscape feature centrally involved.

## What the classification revealed (rule engine, on names+descriptions)

| class | count | celestial | terrestrial |
|---|---:|---:|---:|
| PERCEPTION (true pareidolia) | 30 | 29 | 1 |
| METAMORPHOSIS | 46 | 20 | 26 |
| MENTION | 512 | — | — |
| NONE | 1,550 | — | — |

**The headline correction: the old flag was mostly wrong about pareidolia.**
Of the 299 motifs the old keyword flag called "pareidolic":
- only **24 are actually PERCEPTION** (true pareidolia),
- 22 are METAMORPHOSIS,
- **251 (84%) are mere MENTION** — false positives.

So every earlier figure that used the broad `pareidolic` flag (figs 2, 5, 6, 7) was really
measuring "sky/landscape-themed motifs," not pareidolia. The stricter `perceptual_v2` set
(figs 10, 11) was much closer and lines up with the new PERCEPTION class.

## Two conceptual corrections this gives us

1. **True perception is almost entirely celestial (29 of 30).** Confirms, even more
   starkly, that the catalogue records "figure perceived in nature" essentially only for
   the sky. People do see figures in rocks and clouds, but that lore is local and doesn't
   become a transmitted cross-cultural motif.

2. **The terrestrial signal is METAMORPHOSIS, not PERCEPTION (26 terrestrial vs 1).**
   This refines the fig12 result. The land-based pareidolia in this dataset is not
   "people see a giant in the cliff" — it is "a being was turned into this cliff/tree/rock"
   ("Transformed into stone", "Waves turn into mountains", "Tree turns to rocks",
   "Tree is a person"). That is a different mechanism: an etiological story explaining a
   striking real feature, which *may* be triggered by resemblance but is narrated as
   transformation. So the terrestrial hypothesis, as testable in text, is really about
   **metamorphosis etiology clustering near dramatic landscapes** — and whether the feature
   actually resembles the being is exactly the question only images can answer.

## Caveats on the classifier itself
- This is the RULE engine: transparent regex over name + one-line description. It is much
  better than the old flag (it separates perception/metamorphosis/mention) but still cannot
  read nuance, and a one-line description is thin evidence.
- An LLM engine is included in the same script (`--engine llm`) but could not run here (no
  API key in this sandbox). It sends each entry to Claude with the explicit 4-class rubric
  and returns class + subtype + confidence + reason. **Run it with an API key, and ideally
  on the full ~70k abstracts** (`--engine llm --text-col abstract`) — that is the version
  that will give a trustworthy, validated labelling and likely recover more genuine
  terrestrial perception cases than the 1 the names exposed.

## How to use this going forward
- `dataset/analysis/motif_pareidolia_classified.csv` has every motif with `pc_class`,
  `pc_subtype`, `pc_engine`. Use `pc_class == "PERCEPTION"` as the strict pareidolia set and
  `pc_class in {PERCEPTION, METAMORPHOSIS}` as the broad "environment-figure" set.
- Re-run the climate-analog and 3-way Mantel analyses keyed on `pc_class` instead of the old
  flag to get cleaner versions of figs 10–12. (The conclusions should hold or sharpen, since
  we are removing 251 mention-only false positives.)
- When the abstracts arrive, re-run with the LLM engine and re-do everything on those labels.
