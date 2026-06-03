"""Compute per-image-iconic-taxon Δ on sentence-pooled SigLIP-2, full LLM-clean
corpus. Output: v3_byTaxon_sentpool_iNat.csv with rows (biome, taxon_group,
delta, p_one_sided, n_imgs, n_mot).

taxon_group ∈ {"all", "Plantae", "Fungi", "Animalia", "Mammalia", "Aves",
"Reptilia", "Amphibia", "Actinopterygii", "Insecta", "Arachnida", "Mollusca"}.

For each (biome b, taxon t):
  imgs = iNat images with biome == b AND (t == "all" OR iconic_taxon == t)
  per_motif_mean_sim = sims[imgs].mean(axis=0)
  Δ = mean(per_motif[in_b motifs]) − mean(per_motif[other motifs])
  p = one-sided permutation null over motif labels (1000 perms).
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"

from motif_specificity_controls import biome_motif_membership_count

N_PERMS = 1000
MIN_IMGS = 10  # was 20; lower threshold recovers more biome×taxon cells
TAXON_ORDER = ["all", "Plantae", "Fungi", "Animalia", "Mammalia", "Aves",
               "Reptilia", "Amphibia", "Actinopterygii", "Insecta",
               "Arachnida", "Mollusca"]


def main():
    motif_emb = np.load(EMB / "motif_emb_llm_pass2_abstract_sentpooled.npy")
    meta = pd.read_parquet(EMB / "motif_meta_llm_pass2_sentpooled.parquet")
    valid = meta["valid"].values
    motif_emb = motif_emb[valid]
    meta = meta[valid].reset_index(drop=True)
    motif_ids = meta["motif_id"].astype(str).tolist()

    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet"
                                ).reset_index(drop=True)

    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    in_B = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
    for i, mid in enumerate(motif_ids):
        for j, b in enumerate(biomes):
            if b in mb_set.get(mid, set()):
                in_B[i, j] = True

    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)

    img_biome = img_meta["photo_biome_wwf"].fillna("").values
    img_taxon = img_meta["iconic_taxon"].fillna("").values

    rng = np.random.default_rng(42)
    rows = []
    print(f"computing {len(biomes)} biomes × {len(TAXON_ORDER)} taxa = "
          f"{len(biomes)*len(TAXON_ORDER)} cells", flush=True)
    for j, b in enumerate(biomes):
        in_b = in_B[:, j]
        if in_b.sum() < 5 or (~in_b).sum() < 5:
            continue
        for t in TAXON_ORDER:
            if t == "all":
                mask = img_biome == b
            else:
                mask = (img_biome == b) & (img_taxon == t)
            if mask.sum() < MIN_IMGS:
                continue
            per = sims[mask].mean(axis=0)
            d = float(per[in_b].mean() - per[~in_b].mean())
            null = np.empty(N_PERMS)
            for k in range(N_PERMS):
                shuf = rng.permutation(in_b)
                null[k] = per[shuf].mean() - per[~shuf].mean()
            p = float((null >= d).mean())
            rows.append({
                "biome": b, "taxon_group": t,
                "delta": d, "p_one_sided": p,
                "n_imgs": int(mask.sum()),
                "n_mot": int(in_b.sum()),
            })
        print(f"  done biome {b[:40]}", flush=True)
    df = pd.DataFrame(rows)
    out = EMB / "v3_byTaxon_sentpool_iNat.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} cells)", flush=True)


if __name__ == "__main__":
    main()
