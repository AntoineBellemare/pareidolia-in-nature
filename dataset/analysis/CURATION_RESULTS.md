# Curation + hardened climate-analog: what happened and what we found

You asked for two things: (1) curate the pareidolic set to separate true "looks-like"
pareidolia from "lives-here" fauna, then re-run the climate-analog test, and (2) run the
celestial-vs-terrestrial gradient. Here's the honest account of both, including a wrong
turn I corrected.

## The wrong turn (and the fix)

My first curation pass was too aggressive. It used an EXCLUDE keyword list to throw out
"fauna/material-culture" motifs, but stray words tripped it — e.g. it discarded clearly
pareidolic motifs like "The old sun" and "Several suns burn the earth" because they
contained "earth"/"burn". That cut the set from 299 to 42, and worse, dumped ~257
genuinely-perceptual motifs into the *comparison* bucket, contaminating it. The result
looked like the dissociation had collapsed (+7.7% vs +6.6%) — but that was an artifact of
the bad curation, not a real finding.

I rebuilt it with a **positive-only rule**: a motif is "perceptual pareidolic" if its text
contains a natural-feature noun (moon, star, rock, cloud...) AND an explicit perception
relation (seen-in, figure, imprint, is-a-being, turns-into-a-constellation...). No fragile
exclusions. That yields a small, clean, unambiguous set.

## Finding 1: perceptual mythology is overwhelmingly CELESTIAL

The clean perceptual set is **33 motifs — 31 celestial, 2 terrestrial.** Examples:
"Figure on lunar disc", "Man in the Moon", "The Moon rabbit", "Water-carrier in the Moon",
"The burned-up persons turn into a constellation", "Eclipses: a predator animal."

This is itself a real result, and it answers your celestial-vs-terrestrial question in an
unexpected way: **we can't run that gradient test, because the catalogue barely contains
terrestrial perceptual motifs.** When humans across cultures record "a figure perceived in
nature," it is almost always a figure in the *sky* (moon, stars, sun, eclipse), not in
rocks or clouds. The sky is the dominant pareidolic canvas in the mythological record.

Why this makes sense: the sky is a shared, ever-present, high-contrast screen with stable
recurring patterns (lunar maria, constellations) that everyone on Earth sees. Rock/cloud
pareidolia is intensely local and ephemeral, so it rarely crystallizes into a named,
transmitted motif. (Note: this is a statement about the *Berezkin motif catalogue*, which
indexes cross-culturally recurring mythemes — local one-off "this cliff looks like a
giant" lore is exactly what such a catalogue would miss. So the imagery stage, which can
probe local landscape resemblance directly, is the right tool for the terrestrial side
that the text record can't capture.)

## Finding 2: the hardened climate-analog test is STRONGER, not weaker

Re-running the climate-analog natural experiment with the clean celestial-perceptual set
against a *disjoint* bucket of all other motifs (fixing the v1 contamination):

| motif set | analog (far + same climate) | control (far + diff climate) | lift | p |
|---|---:|---:|---:|---:|
| **perceptual (clean, celestial)** | 0.0863 | 0.0715 | **+20.8%** | 0.0002 |
| all other motifs | 0.0304 | 0.0290 | +4.9% | 0.0002 |

`fig10_climate_analog_clean.png`.

The dissociation is now **sharper than the original** (+20.8% vs +4.9%, a ~4× difference in
lift). Two cultures >5,000 km apart, with no possible contact, share *celestial-perceptual*
mythemes ~21% more when they live in the same climate; for all other motifs the same-climate
effect is only ~5%. Proper curation concentrated the signal instead of diluting it — which
is exactly what should happen if the perceptual motifs are the real carriers of the
environment effect.

(Note the "all other motifs" lift is now +4.9% and significant, vs ~−1% before. That's
because the disjoint bucket here still contains the ~266 keyword-pareidolic-but-not-clean
motifs — many of which are environment-influenced fauna/celestial-adjacent motifs. The
contrast between the clean set and the rest is the finding; the small residual lift in
"all other" is consistent with the fauna/materials environment channel we flagged.)

## Where this leaves the text-only program

The interpretation has tightened considerably:
- The environment effect in mythology is **concentrated in celestial-perceptual motifs** —
  figures seen in the moon, stars, sun, eclipses.
- These motifs converge across unconnected cultures in matched climates (~21% lift), far
  more than ordinary mythology (~5%).
- The text record cannot test *terrestrial* (landscape) pareidolia because it barely
  records it — which is precisely the gap the imagery stage fills.

So the imagery work now has a crisp division of labor:
- The **text data already establishes** the celestial pareidolia signal quantitatively.
- The **imagery data is needed** to test the *terrestrial* hypothesis — does a fjord look
  like a serpent, a rock formation like a troll — which is unrecoverable from the catalogue
  precisely because that lore is local and rarely crystallizes into a transmitted motif.

This is a cleaner project framing than we started with: text handles sky, images handle land.

## Caveats
- The clean perceptual set is small (33). The +20.8% lift is significant (p=0.0002,
  permutation) but rests on relatively few motif columns; treat the effect size as
  indicative.
- "Climate-analog" = same Köppen class + within 5° absolute latitude; tighten with
  WorldClim bioclim vectors.
- Eclipse/celestial motifs may also diffuse via shared astronomical events; the >5,000 km
  contact exclusion mitigates but does not eliminate deep-time common ancestry.
- The celestial-dominance finding is about the Berezkin catalogue specifically; it reflects
  what gets recorded as a cross-cultural mytheme, not necessarily what individuals perceive.
