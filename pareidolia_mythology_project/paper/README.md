# Paper folder

Submission-ready LaTeX draft of *"Mythology aligns with biome"*.

## Contents

| file | purpose |
|---|---|
| `paper.tex` | Main manuscript: abstract → intro → methods → results → discussion → appendix |
| `references.bib` | BibTeX bibliography (Schelling → Bateson → 4E → Berezkin → SigLIP) |
| `figures/` | All 13 figures the paper references (auto-copied from `dataset/imagery/figures/headlines_final_russian/`) |
| `METHODS_anonymisation.md` | Stand-alone methods supplement (in the figures folder) |
| `SUMMARY.md` | Numerical summary with all Spec A / per-taxon / cross-model tables (in the figures folder) |

## Compile

Standard LaTeX with BibTeX:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Requires:
- `babel` with `english` + `russian` (for the inline Cyrillic examples in §2.4)
- `fontenc` with `T1, T2A`
- `siunitx`, `natbib`, `booktabs`, `graphicx`, `hyperref`, `microtype`

## Figure list (13 figures, 7 main + 6 supplementary)

**Main (in order of appearance):**
- `fig2_biome_bars.png` — headline (Fig 2)
- `fig3_taxon_matrix.png` — taxon × biome matrix (Fig 3)
- `fig4_taxon_facets.png` — per-taxon facets (Fig 4)
- `fig11_universals_analysis.png` — coupling-vs-projection breadth split (Fig 5)
- `fig6_naming_progression.png` — anonymisation ladder (Fig 6)
- `fig10_taxon_violins.png` — per-taxon distribution (Fig 7)
- `fig9_crossmodel_correlation.png` — cross-model robustness (Fig 8)

**Supplementary (Appendix A):**
- `fig1_world_traditions.png` — corpus map (S1)
- `fig5_earth_map.png` — Δ on world map (S2)
- `fig7_places365_supplementary.png` — Places365 progression (S3)
- `fig11_full_corpus_residualised.png` — all 14 biomes whole-corpus (S4)
- `fig8_pareidolia_classification.png` — Gemma classification (S5)
- `figS_species_per_biome.png` — ecological composition per biome (S6)

## Target venues

1. **Nature Human Behaviour** (top choice) — interdisciplinary cognitive-anthropology + ML methods.
2. **PNAS Anthropology section** — strong alternate.
3. **Proc. Roy. Soc. B** — fits the cognitive-ecology framing.
4. **Phil. Trans. Roy. Soc. B** — thematic-issue option (e.g. "Ecology of mind").

## What's still TODO before submission

- [ ] Author list + affiliations
- [ ] Acknowledgements (funding, data access)
- [ ] Repository URL for data + code release
- [ ] Final pass on figure aesthetics for journal style
- [ ] Cover letter
- [ ] Reviewer-anticipation memo (one-page internal: list each plausible leak vector and how the paper addresses it)
