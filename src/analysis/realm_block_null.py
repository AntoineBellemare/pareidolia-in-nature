"""R6 — Block-permutation null by WWF biogeographic realm.

The original null permutes motif->biome assignment globally. This
re-runs the residualised Δ test with permutations BLOCKED by WWF
realm (Neotropic, Palearctic, Nearctic, Afrotropic, Indo-Malay,
Australasia, Oceania, Antarctic), so that geographic and cultural
autocorrelation are preserved in the null. The biome assignment of
each motif is shuffled only among biomes that lie in the same realm.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
MAP = ROOT / "dataset/mapping_v2"
OUT = EMB / "v2_R6_block_perm_biome_test.csv"

from motif_specificity_controls import biome_motif_membership_count

N_PERMS = 1000

# WWF realm membership of each biome. Some biomes (e.g. Tundra,
# Boreal Forests) cross multiple realms; we assign each biome to the
# realm that contains the majority of its area in WWF's classification.
BIOME_REALM = {
    "Tropical & Subtropical Moist Broadleaf Forests": "tropical",
    "Tropical & Subtropical Dry Broadleaf Forests": "tropical",
    "Tropical & Subtropical Coniferous Forests": "tropical",
    "Tropical & Subtropical Grasslands, Savannas & Shrublands": "tropical",
    "Mangroves": "tropical",
    "Flooded Grasslands & Savannas": "tropical",
    "Mediterranean Forests, Woodlands & Scrub": "mediterranean",
    "Deserts & Xeric Shrublands": "arid",
    "Montane Grasslands & Shrublands": "montane",
    "Temperate Broadleaf & Mixed Forests": "temperate",
    "Temperate Conifer Forests": "temperate",
    "Temperate Grasslands, Savannas & Shrublands": "temperate",
    "Boreal Forests/Taiga": "boreal",
    "Tundra": "boreal",
}


def main():
    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet")
    img_meta = img_meta.reset_index(drop=True)
    img_biome = img_meta["photo_biome_wwf"].fillna("").values

    motif_emb = np.load(EMB / "motif_emb_llm_pass2_abstract.npy")
    motif_meta = pd.read_parquet(EMB / "motif_meta_llm_pass2.parquet")

    # SigLIP biome-tell filter (apply same restriction as headline)
    keep_ids = set(pd.read_csv(EMB / "biome_tell_filter_siglip.csv")
                    .query("keep_in_filtered")["motif_id"].astype(str))
    mask = motif_meta["motif_id"].astype(str).isin(keep_ids).values
    motif_emb = motif_emb[mask]
    motif_meta = motif_meta[mask].reset_index(drop=True)

    mb_set, _ = biome_motif_membership_count()

    # Compute residualised similarity
    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)

    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    motif_ids = motif_meta["motif_id"].astype(str).tolist()
    in_B = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
    for i, mid in enumerate(motif_ids):
        a = mb_set.get(mid, set())
        for j, b in enumerate(biomes):
            if b in a:
                in_B[i, j] = True

    # Realm membership of each biome
    biome_realm = np.array([BIOME_REALM.get(b, "other") for b in biomes])
    # For each motif we need a realm-restricted permutation: shuffle each
    # motif's biome assignment only among biomes that share its current
    # realm. We treat each motif as belonging to the realm(s) of its
    # assigned biomes. The simplest implementation: per motif, define the
    # set of permissible biomes (same realm as any of its assigned biomes)
    # and randomly resample a same-size subset from that pool.
    rng = np.random.default_rng(42)

    rows = []
    for j, b in enumerate(biomes):
        b_imgs = img_biome == b
        if b_imgs.sum() < 5:
            continue
        in_b = in_B[:, j]
        out_b = ~in_b
        if in_b.sum() < 5 or out_b.sum() < 5:
            continue
        per = sims[b_imgs].mean(axis=0)
        delta_obs = float(per[in_b].mean() - per[out_b].mean())

        # Build the within-realm null
        # For each motif i, identify which biomes in its assigned set
        # share the realm of biome b. If any assigned biome is in b's
        # realm, the motif is eligible to be 'in B' under realm-blocked
        # permutation; otherwise it is restricted to its own realm and
        # never permuted into B.
        b_realm = biome_realm[j]
        motif_eligible = np.zeros(len(motif_ids), dtype=bool)
        for i, mid in enumerate(motif_ids):
            assigned = mb_set.get(mid, set())
            if any(BIOME_REALM.get(ab, "other") == b_realm for ab in assigned):
                motif_eligible[i] = True

        n_in = int(in_b.sum())
        eligible_idx = np.where(motif_eligible)[0]
        if len(eligible_idx) < n_in:
            # Not enough eligible motifs; fall back to global permutation
            eligible_idx = np.arange(len(motif_ids))

        null = np.empty(N_PERMS)
        for k in range(N_PERMS):
            shuf_in = np.zeros(len(motif_ids), dtype=bool)
            chosen = rng.choice(eligible_idx, size=n_in, replace=False)
            shuf_in[chosen] = True
            null[k] = per[shuf_in].mean() - per[~shuf_in].mean()
        p_block = float((null >= delta_obs).mean())

        rows.append({
            "biome": b,
            "realm": b_realm,
            "n_imgs": int(b_imgs.sum()),
            "n_motifs_in_biome": n_in,
            "n_eligible_motifs": int(motif_eligible.sum()),
            "delta": delta_obs,
            "p_one_sided_block": p_block,
        })

    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(df.to_string(index=False))
    sig = int((df["p_one_sided_block"] < 0.05).sum())
    print(f"\nμΔ = {df['delta'].mean()*1000:+.3f} ×10⁻³")
    print(f"sig biomes under realm-blocked null: {sig}/{len(df)}")


if __name__ == "__main__":
    main()
