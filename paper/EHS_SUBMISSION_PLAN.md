# EHS submission plan — *The Imagination of Nature*

Target: **Evolutionary Human Sciences** (Cambridge UP), Original Research Article.
APC-free for UdeM/McGill corresponding authors under the CRKN–Cambridge agreement,
which covers **"all Cambridge hybrid and gold journals"** (EHS is fully gold; list APC USD 3,450).

**Status: plan only. No changes have been made to `paper.tex`.**

> **Deadline pressure.** The CRKN–Cambridge term ends **31 Dec 2026** and coverage
> attaches at **acceptance**, not submission. Everything below is editing, not analysis —
> it does not gate on new results.

---

## 0. Verdict in one line

Sixteen items, **none of which is an analysis**. No new statistics, no reanalysis, no
reframing that could break a claim. This is a day or two of editing against ~£1,400 at RSOS.

---

## 1. Hard gates — submission will not reach review without these

| # | Item | State |
|---|------|-------|
| 1 | **Conflicts of Interest statement** — *"If authors do not include this, their submission will not proceed to peer review."* | ❌ missing |
| 2 | **Author Contributions statement** | ❌ missing |
| 3 | **Financial Support statement** — if unfunded, must use their exact wording: *"This research received no specific grant from any funding agency, commercial or not-for-profit sectors."* | ❌ missing |
| 4 | **Graphical abstract** — required of all submissions. PNG, 900px on longest edge. May be an illustration; needs no original data. | ❌ missing |
| 5 | **ORCID** for corresponding author | ❌ to confirm |

All five statements go **at the end of the manuscript, before References**, in this order:
Acknowledgements → Author Contributions → Financial Support → Conflicts of Interest →
Research Transparency and Reproducibility.

Item 5 (Research Transparency) is **half-done**: `\section*{Data and code availability}`
already exists at **line 1618** and needs recasting under their heading, with explicit URLs
for the dataset and code.

---

## 2. Word budget — how to reach 8,000

Limit is **8,000 words excluding abstract, tables, figures and references**.
Measured on exactly that basis, the paper is at **9,359** → **cut 1,360**.

Strategy per your call: trim the intro lightly (keep the humanities lineage),
tighten prose globally without losing arguments, push Methods detail to supplementary.

| Donor | Now | Cut | After | How |
|---|---:|---:|---:|---|
| **Methods → supplementary** | 2,065 | −600 | 1,465 | Move null-construction detail, per-model specifics, and statistical boilerplate to a new **S7**. Keep a tight, complete Methods in main text (EHS mandates the section, not its depth). Costs zero argument. |
| **Discussion tightening** | 2,692 | −360 | 2,332 | *"What kind of coupling: the lexical-thematic level"* (line 1025) is **1,145 words — the largest subsection in the paper** and carries most of this. |
| **Results tightening** | 3,358 | −250 | 3,108 | Prose compression only; **no result removed**. *"Per-taxon decomposition"* (line 672, 769 w) is the largest and densest. |
| **Introduction** | 1,184 | −150 | 1,034 | Light trim only. **Keep** the Schelling→Jung→Hillman→Corbin lineage and the cultural-attraction reframe. |
| **Total** | **9,359** | **−1,360** | **7,999** | ✅ under 8,000 |

### Section word counts (measured, for reference)

| Section | Line | Words | Figs |
|---|---:|---:|---:|
| Introduction | 109 | 1,184 | 0 |
| Results | 252 | 3,358 | 9 |
| Discussion | 905 | 2,692 | 0 |
| Materials and Methods | 1230 | 2,065 | 1 |
| Data and code availability | 1618 | 58 | 0 |

Largest subsections, i.e. the natural donors:
`What kind of coupling` 1,145 · `Per-taxon decomposition` 769 · `Biome-specific mythology aligns` 673 ·
`The alignment is not reducible to identity naming` 634 · `Toward a falsifiable enactivist mythography` 555.

---

## 3. Display items — 10 → 8

Cap is **8 figures or tables**. Main text currently has **10 figures**.
Nothing needs to be lost:

| Action | Figure | Rationale |
|---|---|---|
| **→ Supplementary** | `fig1_world_traditions.png` (line 281, "Fig 2") | Descriptive corpus map. **No claim rests on it** — the 958-tradition geocoding is fully stated in text. Cheapest possible slot. |
| **Merge into one 2-panel item** | `fig_biome_recovery.png` (864) + `fig_biome_recovery_p365.png` (888) | Both are strategy ③. Merging keeps **both corpora visible** and costs one slot instead of deleting the Places365 replication. Better than dropping it. |

Net: 10 → 8. ✅

**Alternate** if you'd rather keep Fig 9 standalone: move `methods.png` (1344) to supplementary
instead. Noting you already chose earlier this project to keep the methods figure, so it is
listed second.

Full main-text figure inventory:

| # | File | Line | Role |
|---|---|---:|---|
| 1 | `fig_roadmap.png` | 256 | three strategies — keep |
| 2 | `fig1_world_traditions.png` | 281 | corpus map — **move** |
| 3 | `fig2_biome_bars.png` | 299 | two-corpus alignment — keep |
| 4 | `fig5_earth_map.png` | 329 | geography — keep |
| 5 | `fig_identity_naming.png` | 569 | strategy ② — keep |
| 6 | `fig11_universals_analysis.png` | 603 | **breadth gradient — load-bearing, keep** |
| 7 | `fig_taxon_combined.png` | 676 | per-taxon — keep |
| 8 | `fig_biome_recovery.png` | 864 | strategy ③ iNat — **merge** |
| 9 | `fig_biome_recovery_p365.png` | 888 | strategy ③ P365 — **merge** |
| 10 | `methods.png` | 1344 | pipeline — keep |

---

## 4. Abstract — 323 → 200 words

A **−123 word (−38%) rewrite**, not a trim. Their spec: *summarize the background,
findings, and implications.*

Keep, in priority order:
1. The question (one sentence).
2. Corpus + method: 958 traditions, 14 WWF biomes, sentence-pooled SigLIP-2, iNaturalist + Places365.
3. The three convergent strategies **as a clause, not a list** — this is where most of the 123 words are.
4. The breadth gradient (the load-bearing signature).
5. One implication sentence.

Cut: the bracketed per-strategy statistics (8/14, 6/14, ρ=0.75, 64th/61st percentile,
9/9, 7/7) — these belong in Results, and they are the densest text in the abstract.

---

## 5. Structural jobs

### 5a. Section reorder — **required**
EHS mandates *introduction, methods, results and discussion*. The paper is PNAS-style:

```
now:  Introduction(109) → Results(252) → Discussion(905) → Materials and Methods(1230)
EHS:  Introduction      → Methods      → Results         → Discussion
```

Methods is a ~390-line block (1230–1617) that moves up. Mechanical, but it will need
transition sentences rewritten at both seams, and forward-references in Results
("as described below") inverted.

### 5b. APA restyle
`\bibliographystyle{plainnat}` (line 31) → APA. Alphabetical by first author surname.
In-text: (Smith, 2012) / (Smith and Wright, 2013) / (Smith *et al.*).
**References are not a problem: 40 cited against a 75 cap.**

### 5c. Line numbers
`lineno` is not loaded. EHS: *"Please include continuous line numbers."*

### 5d. AI declaration — the one substantive item
Your Gemini anonymisation pipeline falls squarely under their policy trigger:
*"used it to … analyse or extract insights from data or other materials"* → **must be declared
and described in the Methods section**. Required content:

- name **and version** of the tool
- date(s) used, to the extent reasonably possible
- how the tool can be accessed by others
- a **full description** of how it was used
- citations to any third-party material in its output

This applies to the Gemini rewrite/anonymisation pass. SigLIP-2, M-CLIP and OpenCLIP are
analysis models already described in Methods — extend that description to meet the same bar.

### 5e. Foreign quotations
7 lines contain Cyrillic (медведь, тундра, млекопитающее …). EHS: *"Foreign quotations and
phrases should be followed by a translation."* Add glosses.

### 5f. Title — 13 words → ≤12 preferred
Current: *"The Imagination of Nature: Mythology as the Cultural Recording Layer of Perceptual Coupling"* (13).

- Minimal fix (12): *"The Imagination of Nature: Mythology as a Recording Layer of Perceptual Coupling"*
- Register fix (9), better for this desk: *"Ecology and the content of myth across 958 traditions"*

---

## 6. New artifacts to produce

| Item | Spec |
|---|---|
| Graphical abstract | PNG, 900px longest edge. Illustration is fine. The roadmap figure is the obvious source. |
| Social media summary | ≤120 characters. Final draft only. |
| Keywords | ≥3, at submission. |
| Alt-text | Required for **every** figure (WCAG 2.1 AA), via the Accessibility Descriptions Submission Form. Machine-generated if not supplied — accuracy remains the author's responsibility. |
| Title page | Title, authors, affiliations **with country**, corresponding author asterisked + email, **word count**. |
| Files | LaTeX sources **and** PDF, both at review and at export. Overleaf template available. |

---

## 7. Already satisfied — do not spend time here

- ✅ **References: 40 cited vs. 75 cap.** (The claim that this cap "forces the humanities lineage out, ~40% of the theoretical apparatus" was **wrong** — it is not binding at all.)
- ✅ **Footnotes: 0.** EHS accepts none.
- ✅ Acknowledgements section exists (1628).
- ✅ Data/code availability exists (1618) — recast, don't write from scratch.
- ✅ Supplementary already uses the `S`-prefix convention (S1–S6).
- ✅ Figures already embedded in-place with legends.
- ✅ LaTeX accepted.
- ✅ CC-BY by default; authors retain copyright.
- ✅ Preprint deposition is **not** prior publication — the repo/preprint is safe.

---

## 8. Editorial positioning (free, high-leverage)

Not required by the guidelines, but this is what converts a moderate desk-reject risk to a low one:

- **State the breadth gradient in cultural-evolutionary terms.** It already *is* a
  differential-retention result: ecological affordances as attractors; universals as motifs
  whose stability is independent of local ecology. Same analysis, recognizable vocabulary.
- **Cite Bromham & Yaxley (2023, EHS 5:e27) head-on.** It is *this journal's own* paper on the
  Galton's-problem confound that your within-Glottolog-macroarea biome-swap null exists to
  answer. Highest-leverage citation available.
- **Pre-empt the construct-validity objection in the Discussion.** The alignment is measured
  against *Berezkin's English-language analytical summaries* — a 20th-century folklorist's
  paraphrase, not a tradition's own words. LLM anonymisation and class-word collapse address
  **vocabulary** leakage, not describer-selection bias. The breadth gradient is the best
  defence; foreground it.

---

## 9. Provenance

Verified directly against the publishers (July 2026):

- CRKN read-and-publish partners: Cambridge, Canadian Science Publishing, Elsevier, IOP, OUP,
  PLOS, **Royal Society of Chemistry**, SAGE, Wiley. **The Royal Society (London) is not among
  them** — the earlier "free at UdeM/McGill" reading of Proc B / RSOS / RSIF rested on a
  publisher **name collision**.
- CRKN–Cambridge explicitly covers **gold** titles → EHS is genuinely free.
- Royal Society Subscribe-to-Open 2026 covers eight subscription titles free to any author;
  **RSOS and Open Biology are excluded** and still charge (~£1,400) — which is why the
  best-fitting Royal Society venue is the one that costs money.

Paper metrics measured from `paper.tex` / `paper.pdf` at commit `2617c6b`.
Not independently re-checked: EHS internal review timelines, editor identities.
