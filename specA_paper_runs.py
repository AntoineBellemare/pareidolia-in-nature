"""
specA_paper_runs.py — produce the Spec A (residualised + drop universals + ≥3 own-traditions)
runs needed for the paper-headline figures.

Need:
  iNat   × oneliners                 (already exists as specA_iNatxoneliners.csv)
  iNat   × abstracts                 (NEW)
  YFCC-f × oneliners                 (NEW)
  YFCC-f × abstracts                 (already exists)
  iNat   × oneliners  × byTaxon      (NEW)
  iNat   × abstracts  × byTaxon      (NEW)

Spec A definition: motif counted as "in biome B" only if it appears in ≤3 biomes
total AND in ≥3 traditions of B. Universals (>3 biomes) are excluded entirely.
Residualisation is applied (per-motif grand mean subtracted from sims).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MAP = ROOT / "dataset/mapping_v2"
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"

from motif_specificity_controls import (
    biome_motif_membership_count, control_A,
)


def control_A_byTaxon(img_emb_path, img_meta_path, motif_emb_path, motif_meta_path,
                      max_biomes=3, min_in_biome=3, n_perms=1000):
    """Same as control_A but stratified by iconic_taxon of the image."""
    img_emb = np.load(img_emb_path)
    img_meta = pd.read_parquet(img_meta_path).reset_index(drop=True)
    motif_emb = np.load(motif_emb_path)
    motif_meta = pd.read_parquet(motif_meta_path)
    motif_to_biomes_set, motif_to_count = biome_motif_membership_count()

    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)  # residualise
    motif_ids = motif_meta["motif_id"].tolist()
    biomes = sorted({b for s in motif_to_biomes_set.values() for b in s})

    is_in_biome = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
    for i, mid in enumerate(motif_ids):
        my_biomes = motif_to_biomes_set.get(mid, set())
        if len(my_biomes) > max_biomes:
            continue
        per_biome_n = motif_to_count.get(mid, {})
        for j, b in enumerate(biomes):
            if b in my_biomes and per_biome_n.get(b, 0) >= min_in_biome:
                is_in_biome[i, j] = True
    passed_any = is_in_biome.any(axis=1)

    use_biome = img_meta.get("photo_biome_wwf")
    if use_biome is None:
        use_biome = img_meta["tradition_biome_wwf"]
    use_biome = use_biome.fillna(img_meta.get("tradition_biome_wwf", "")).values

    taxa = ["all", "Plantae", "Fungi", "Animalia",
            "Mammalia", "Aves", "Reptilia", "Amphibia",
            "Actinopterygii", "Insecta", "Arachnida", "Mollusca"]
    iconic = img_meta["iconic_taxon"].fillna("").values if "iconic_taxon" in img_meta.columns else None

    rng = np.random.default_rng(42)
    rows = []
    for taxon in taxa:
        if taxon == "all":
            taxon_mask = np.ones(len(img_meta), dtype=bool)
        else:
            if iconic is None: continue
            taxon_mask = (iconic == taxon)
        if taxon_mask.sum() < 30:
            continue
        print(f"  taxon={taxon}  n_imgs={taxon_mask.sum()}")
        for j, b in enumerate(biomes):
            b_imgs = (use_biome == b) & taxon_mask
            if b_imgs.sum() < 5: continue
            b_motifs = is_in_biome[:, j]
            if b_motifs.sum() < 5: continue
            b_other = passed_any & (~b_motifs)
            if b_other.sum() < 5: continue
            per = sims[b_imgs].mean(axis=0)
            mean_own = float(per[b_motifs].mean())
            mean_oth = float(per[b_other].mean())
            delta = mean_own - mean_oth

            idx_passed = np.where(passed_any)[0]
            passed_in_b_mask = b_motifs[idx_passed]
            null = np.empty(n_perms)
            for k in range(n_perms):
                shuf = rng.permutation(passed_in_b_mask)
                in_idx = idx_passed[shuf]
                out_idx = idx_passed[~shuf]
                null[k] = per[in_idx].mean() - per[out_idx].mean()
            p = float((null >= delta).mean())
            rows.append({
                "taxon_group": taxon, "biome": b,
                "n_imgs": int(b_imgs.sum()),
                "n_motifs_in_biome_specific": int(b_motifs.sum()),
                "n_motifs_other_specific": int(b_other.sum()),
                "delta": delta, "p_one_sided": p,
            })
    return pd.DataFrame(rows)


def main():
    # === missing biome-level Spec A pairs ===
    missing = [
        ("iNat × abstracts",
         EMB / "img_emb.npy", EMB / "img_paths.parquet",
         EMB / "motif_emb_abstracts.npy", EMB / "motif_meta_abstracts.parquet",
         EMB / "specA_iNatxabstracts.csv"),
        ("YFCC-filtered × oneliners",
         EMB / "yfcc_filtered/img_emb.npy",
         EMB / "yfcc_filtered/img_paths.parquet",
         EMB / "motif_emb_all.npy", EMB / "motif_meta_all.parquet",
         EMB / "specA_YFCCfilteredxoneliners.csv"),
    ]
    for label, ie, im, me, mm, out in missing:
        if out.exists():
            print(f"  exists, skipping: {out.name}"); continue
        print(f"\n== Spec A: {label} ==")
        df = control_A(ie, im, me, mm, label=label)
        df.to_csv(out, index=False)
        print(df.to_string(index=False))
        print(f"wrote {out}")

    # === byTaxon Spec A ===
    for label, motif_emb, motif_meta, out in [
        ("iNat × oneliners × byTaxon",
         EMB / "motif_emb_all.npy", EMB / "motif_meta_all.parquet",
         EMB / "specA_byTaxon_iNatxoneliners.csv"),
        ("iNat × abstracts × byTaxon",
         EMB / "motif_emb_abstracts.npy", EMB / "motif_meta_abstracts.parquet",
         EMB / "specA_byTaxon_iNatxabstracts.csv"),
    ]:
        if out.exists():
            print(f"  exists, skipping: {out.name}"); continue
        print(f"\n== Spec A byTaxon: {label} ==")
        df = control_A_byTaxon(
            EMB / "img_emb.npy", EMB / "img_paths.parquet",
            motif_emb, motif_meta,
        )
        df.to_csv(out, index=False)
        print(df.groupby("taxon_group")["delta"].agg(["mean", "size"]).round(5).to_string())
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
