"""v3 recompute: all Δ-dependent outputs using sentence-pooled SigLIP-2
embeddings (mean pooling), full LLM-clean corpus (no biome-tell filter).

Outputs:
  inat_basic/v3_biome_test_sentpool_marg_resid.csv  (marginal Δ on iNat)
  taxon_stratified_sentpool_iNat.csv                (stratified Δ on iNat)
  places365_strict/v3_biome_test_sentpool_marg_resid.csv  (Places365 marginal)
  v3_byTaxon_sentpool_iNat.csv                      (per-taxon decomposition)
  v3_breadth_sentpool_iNat.csv                      (Spec A / semi-univ / univ)
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

from motif_specificity_controls import biome_motif_membership_count

N_PERMS = 1000
MIN_BIOME_IMGS = 50
MIN_TAXON_IMGS = 20


def residualised_test(motif_emb, motif_ids, img_emb, img_meta,
                       n_perms=N_PERMS, stratify_taxon=False):
    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    in_B = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
    for i, mid in enumerate(motif_ids):
        for j, b in enumerate(biomes):
            if b in mb_set.get(mid, set()):
                in_B[i, j] = True
    img_biome = img_meta["photo_biome_wwf"].fillna("").values
    img_taxon = (img_meta["iconic_taxon"].fillna("").values
                  if "iconic_taxon" in img_meta.columns else None)
    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)

    rng = np.random.default_rng(42)
    rows = []
    strata = (sorted([t for t in pd.Series(img_taxon).unique()
                       if t and t != "N/A"])
               if stratify_taxon and img_taxon is not None else None)
    for j, b in enumerate(biomes):
        b_imgs = img_biome == b
        if b_imgs.sum() < MIN_BIOME_IMGS:
            continue
        in_b = in_B[:, j]
        if in_b.sum() < 5 or (~in_b).sum() < 5:
            continue
        per = sims[b_imgs].mean(axis=0)
        d_marg = float(per[in_b].mean() - per[~in_b].mean())
        null_marg = np.empty(n_perms)
        for k in range(n_perms):
            shuf = rng.permutation(in_b)
            null_marg[k] = per[shuf].mean() - per[~shuf].mean()
        p_marg = float((null_marg >= d_marg).mean())
        row = {"biome": b, "n_imgs": int(b_imgs.sum()),
               "n_motifs_in_biome": int(in_b.sum()),
               "delta": d_marg, "p_one_sided": p_marg}
        if strata is not None:
            d_strata = []
            for t in strata:
                tm = b_imgs & (img_taxon == t)
                if tm.sum() < MIN_TAXON_IMGS:
                    continue
                per_t = sims[tm].mean(axis=0)
                d_strata.append(
                    per_t[in_b].mean() - per_t[~in_b].mean())
            if len(d_strata) >= 2:
                d_strat = float(np.mean(d_strata))
                null_strat = np.empty(n_perms)
                for k in range(n_perms):
                    shuf = rng.permutation(in_b)
                    ds = []
                    for t in strata:
                        tm = b_imgs & (img_taxon == t)
                        if tm.sum() < MIN_TAXON_IMGS:
                            continue
                        per_t = sims[tm].mean(axis=0)
                        ds.append(
                            per_t[shuf].mean() - per_t[~shuf].mean())
                    null_strat[k] = float(np.mean(ds))
                row["delta_strat"] = d_strat
                row["p_strat"] = float((null_strat >= d_strat).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("delta", ascending=False)


def main():
    # Load sentence-pooled embeddings
    motif_emb = np.load(EMB / "motif_emb_llm_pass2_abstract_sentpooled.npy")
    meta = pd.read_parquet(
        EMB / "motif_meta_llm_pass2_sentpooled.parquet")
    valid = meta["valid"].values
    motif_emb = motif_emb[valid]
    meta = meta[valid].reset_index(drop=True)
    motif_ids = meta["motif_id"].astype(str).tolist()
    print(f"sentence-pooled motifs (no biome-tell filter): {len(motif_ids)}",
          flush=True)

    # ============ iNat ============
    print("\n=== iNat: marginal + stratified ===", flush=True)
    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(
        EMB / "inat_basic/img_paths.parquet").reset_index(drop=True)
    df = residualised_test(motif_emb, motif_ids, img_emb, img_meta,
                            stratify_taxon=True)
    out = EMB / "inat_basic/v3_biome_test_sentpool_resid.csv"
    df.to_csv(out, index=False)
    print(f"  μΔ marg = {df['delta'].mean()*1000:+.3f}, "
          f"μΔ strat = {df['delta_strat'].mean()*1000:+.3f}", flush=True)
    print(f"  wrote {out}", flush=True)

    # ============ Places365 ============
    print("\n=== Places365: marginal only ===", flush=True)
    p365_emb = np.load(EMB / "places365_strict/img_emb.npy")
    p365_meta = pd.read_parquet(
        EMB / "places365_strict/img_paths.parquet").reset_index(drop=True)
    df_p = residualised_test(motif_emb, motif_ids, p365_emb, p365_meta,
                              stratify_taxon=False)
    out = EMB / "places365_strict/v3_biome_test_sentpool_resid.csv"
    df_p.to_csv(out, index=False)
    print(f"  μΔ marg = {df_p['delta'].mean()*1000:+.3f}", flush=True)

    # ============ Per-taxon decomposition (text-side) ============
    # Same per-text-taxon decomp logic as before, but on sentence-pooled
    # embeddings. Need iconic-taxon labels on motif side from the
    # taxon_labels file used previously.
    print("\n=== Per-taxon decomposition ===", flush=True)
    taxon_csv = ROOT / "dataset/analysis/motif_taxon_groups.csv"
    if taxon_csv.exists():
        tg = pd.read_csv(taxon_csv)
        tg["motif_id"] = tg["motif_id"].astype(str)
        meta_w_tax = meta.merge(tg, on="motif_id", how="left")
        per_taxon_rows = []
        mb_set, _ = biome_motif_membership_count()
        biomes = sorted({b for s in mb_set.values() for b in s
                         if isinstance(b, str) and b != "N/A"})
        for tg_val in (["all"] +
                        sorted(meta_w_tax["taxon_group"].dropna().unique().tolist())):
            if tg_val == "all":
                mask = np.ones(len(meta_w_tax), dtype=bool)
            else:
                mask = (meta_w_tax["taxon_group"] == tg_val).values
            if mask.sum() < 30:
                continue
            sub_emb = motif_emb[mask]
            sub_ids = [m for i, m in enumerate(motif_ids) if mask[i]]
            df_t = residualised_test(sub_emb, sub_ids, img_emb, img_meta,
                                       stratify_taxon=False)
            df_t["taxon_group"] = tg_val
            per_taxon_rows.append(df_t)
        if per_taxon_rows:
            df_pt = pd.concat(per_taxon_rows, ignore_index=True)
            out = EMB / "v3_byTaxon_sentpool_iNat.csv"
            df_pt.to_csv(out, index=False)
            print(f"  wrote {out}", flush=True)
    else:
        print(f"  [warn] {taxon_csv} not found, skipping per-taxon")

    # ============ Breadth-stratified ============
    print("\n=== Breadth-stratified (Spec A / semi-univ / universal) ===",
          flush=True)
    mb_set, _ = biome_motif_membership_count()
    breadth = []
    for mid in motif_ids:
        n = len({b for b in mb_set.get(mid, set())
                 if isinstance(b, str) and b != "N/A"})
        if n <= 3:
            breadth.append("SpecA")
        elif n <= 7:
            breadth.append("Semi")
        else:
            breadth.append("Universal")
    breadth = np.array(breadth)
    breadth_rows = []
    for grp in ["SpecA", "Semi", "Universal"]:
        mask = breadth == grp
        if mask.sum() < 20:
            continue
        sub_emb = motif_emb[mask]
        sub_ids = [m for i, m in enumerate(motif_ids) if mask[i]]
        df_b = residualised_test(sub_emb, sub_ids, img_emb, img_meta,
                                   stratify_taxon=True)
        df_b["breadth"] = grp
        breadth_rows.append(df_b)
        print(f"  {grp}: {mask.sum()} motifs, "
              f"μΔ marg = {df_b['delta'].mean()*1000:+.3f}, "
              f"μΔ strat = {df_b['delta_strat'].mean()*1000:+.3f}",
              flush=True)
    if breadth_rows:
        df_br = pd.concat(breadth_rows, ignore_index=True)
        out = EMB / "v3_breadth_sentpool_iNat.csv"
        df_br.to_csv(out, index=False)
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
