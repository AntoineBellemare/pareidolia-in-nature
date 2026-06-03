"""R6b — Glottolog macro-area block permutation null.

Block-permute motif-to-biome assignments WITHIN Glottolog macro-area
(Africa, Australia, Eurasia, North America, Papunesia, South America).
This controls for cultural and linguistic autocorrelation more directly
than the WWF biogeographic-realm block of R6, because Glottolog
macro-areas are the standard grouping unit in cross-linguistic and
cross-cultural research.

Mapping: each Berezkin tradition is assigned a macro-area from its
(lat, lon) coordinates. A motif's macro-area set is the union over all
its traditions. The null permutes motif assignments only among biomes
that contain at least one tradition from the same macro-area.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
MAP = ROOT / "dataset/mapping_v2"
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
OUT = EMB / "v2_R6b_glottolog_macroarea_null.csv"

from motif_specificity_controls import biome_motif_membership_count

N_PERMS = 1000


def macroarea_from_coords(lat, lon):
    """Approximate Glottolog macro-area from coordinates. Returns one of
    Africa, Australia, Eurasia, North America, Papunesia, South America,
    or None if ambiguous / out of range."""
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    # Australia first (clear continental boundary)
    if -45 <= lat <= -10 and 110 <= lon <= 155:
        return "Australia"
    # Papunesia: New Guinea, Indonesia, Philippines, Pacific islands
    if -15 <= lat <= 25 and 95 <= lon <= 180:
        return "Papunesia"
    if -45 <= lat <= 0 and 155 <= lon <= 180:
        return "Papunesia"
    if -25 <= lat <= -10 and 155 <= lon <= 180:
        return "Papunesia"
    # North America
    if 15 <= lat <= 75 and -170 <= lon <= -50:
        return "North America"
    if 50 <= lat <= 80 and -180 <= lon <= -50:
        return "North America"  # Alaska, Greenland
    # South America
    if -56 <= lat <= 15 and -85 <= lon <= -34:
        return "South America"
    # Africa
    if -35 <= lat <= 35 and -20 <= lon <= 55:
        return "Africa"
    # Eurasia (default for remaining northern landmass)
    if lat > 25 and lon >= -15:
        return "Eurasia"
    if 5 <= lat <= 25 and 25 <= lon <= 100:
        return "Eurasia"
    return None


def main():
    print("loading traditions ...", flush=True)
    trad = pd.read_parquet(MAP / "traditions.parquet")
    trad["macroarea"] = trad.apply(
        lambda r: macroarea_from_coords(r["lat"], r["lon"]), axis=1)
    print("macroarea distribution:")
    print(trad["macroarea"].value_counts(dropna=False))
    print(f"\nunmapped traditions: {trad['macroarea'].isna().sum()}")

    # Build motif -> set(macroareas)
    print("\nbuilding motif macroarea sets ...", flush=True)
    tm = pd.read_parquet(MAP / "tradition_motif.parquet")
    trad_macro = trad.set_index("oid")["macroarea"].to_dict()
    motif_macros = {}
    for mid, sub in tm.groupby("motif_id"):
        s = set()
        for oid in sub["oid"]:
            m = trad_macro.get(oid)
            if isinstance(m, str):
                s.add(m)
        motif_macros[str(mid)] = s

    # Load motif-biome assignments and motif embeddings (after SigLIP filter)
    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet")
    img_meta = img_meta.reset_index(drop=True)
    img_biome = img_meta["photo_biome_wwf"].fillna("").values

    motif_emb = np.load(EMB / "motif_emb_llm_pass2_abstract.npy")
    motif_meta = pd.read_parquet(EMB / "motif_meta_llm_pass2.parquet")
    keep_ids = set(pd.read_csv(EMB / "biome_tell_filter_siglip.csv")
                    .query("keep_in_filtered")["motif_id"].astype(str))
    mask = motif_meta["motif_id"].astype(str).isin(keep_ids).values
    motif_emb = motif_emb[mask]
    motif_meta = motif_meta[mask].reset_index(drop=True)
    motif_ids = motif_meta["motif_id"].astype(str).tolist()

    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    in_B = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
    for i, mid in enumerate(motif_ids):
        a = mb_set.get(mid, set())
        for j, b in enumerate(biomes):
            if b in a:
                in_B[i, j] = True

    # Build biome -> set(macroareas) by aggregating macro-areas of all
    # traditions assigned to that biome (via their tm rows).
    biome_macros = {b: set() for b in biomes}
    for oid, row in trad.set_index("oid").iterrows():
        b = row["biome_wwf"]
        m = row["macroarea"]
        if isinstance(b, str) and isinstance(m, str):
            biome_macros.setdefault(b, set()).add(m)

    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)

    rng = np.random.default_rng(42)

    rows = []
    for j, b in enumerate(biomes):
        b_imgs = img_biome == b
        if b_imgs.sum() < 5:
            continue
        in_b = in_B[:, j]
        if in_b.sum() < 5 or (~in_b).sum() < 5:
            continue
        per = sims[b_imgs].mean(axis=0)
        delta_obs = float(per[in_b].mean() - per[~in_b].mean())

        # Identify motifs eligible to land in biome b under
        # macro-area-blocked permutation: a motif is eligible if at least
        # one of its macro-areas is also a macro-area containing the biome
        b_macros = biome_macros.get(b, set())
        eligible = np.zeros(len(motif_ids), dtype=bool)
        for i, mid in enumerate(motif_ids):
            mm = motif_macros.get(mid, set())
            if mm & b_macros:
                eligible[i] = True
        n_eligible = int(eligible.sum())
        n_in = int(in_b.sum())
        if n_eligible < n_in or len(b_macros) == 0:
            # Fall back to global permutation (single-macroarea biome)
            eligible_idx = np.arange(len(motif_ids))
            note = "global_fallback"
        else:
            eligible_idx = np.where(eligible)[0]
            note = "block"

        null = np.empty(N_PERMS)
        for k in range(N_PERMS):
            chosen = rng.choice(eligible_idx, size=n_in, replace=False)
            shuf = np.zeros(len(motif_ids), dtype=bool)
            shuf[chosen] = True
            null[k] = per[shuf].mean() - per[~shuf].mean()
        p = float((null >= delta_obs).mean())

        rows.append({
            "biome": b,
            "biome_macroareas": ",".join(sorted(b_macros)),
            "n_imgs": int(b_imgs.sum()),
            "n_motifs_in_biome": n_in,
            "n_eligible_motifs": n_eligible,
            "delta": delta_obs,
            "p_glottolog_block": p,
            "null_type": note,
        })

    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}\n")
    print(df.to_string(index=False))
    sig = int((df["p_glottolog_block"] < 0.05).sum())
    print(f"\nμΔ = {df['delta'].mean()*1000:+.3f} ×10⁻³")
    print(f"sig biomes under Glottolog macro-area null: {sig}/{len(df)}")
    n_real_block = (df["null_type"] == "block").sum()
    print(f"biomes with a real macro-area block test: {n_real_block}")


if __name__ == "__main__":
    main()
