"""Two biome-tell analyses requested by the v3 review.

(1) HIGH-TELL vs LOW-TELL split. Compute biome-tell z-score for each motif
    on sentence-pooled SigLIP-2 embeddings, split motifs at median z,
    and run the within-iconic-taxon stratified Δ test on each half. If
    the low-tell half still gives positive Δ, the alignment is not only
    explained by biome-correlated lexical content recoverable from text
    alone.

(2) WITHIN-GLOTTOLOG-MACROAREA BIOME-SWAP NULL. For each tradition,
    randomly reassign its motifs to another tradition in the same
    Glottolog macro-area but a DIFFERENT biome, and recompute Δ. This
    is a stronger version of the macro-area block-permutation null: it
    actively swaps biome labels while preserving cultural geography. If
    Δ collapses, biome assignment is doing the work; if Δ holds, the
    motif content itself carries biome-correlated signal independent
    of which biome we say it belongs to.

Outputs:
  v3_biome_tell_split.csv             — per-biome Δ for high-tell & low-tell halves
  v3_glottolog_swap_null.csv          — per-biome observed Δ + p_value under swap null
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
from glottolog_swap import macroarea_from_coords

N_PERMS = 1000
MIN_BIOME_IMGS = 50
MIN_TAXON_IMGS = 20


def stratified_test(motif_emb, motif_ids, img_emb, img_meta, n_perms=N_PERMS):
    """Within-iconic-taxon stratified Δ test, same definition as v3."""
    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    in_B = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
    for i, mid in enumerate(motif_ids):
        for j, b in enumerate(biomes):
            if b in mb_set.get(mid, set()):
                in_B[i, j] = True
    img_biome = img_meta["photo_biome_wwf"].fillna("").values
    img_taxon = img_meta["iconic_taxon"].fillna("").values
    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)

    rng = np.random.default_rng(42)
    strata = sorted([t for t in pd.Series(img_taxon).unique()
                     if t and t != "N/A"])
    rows = []
    for j, b in enumerate(biomes):
        b_imgs = img_biome == b
        if b_imgs.sum() < MIN_BIOME_IMGS:
            continue
        in_b = in_B[:, j]
        if in_b.sum() < 5 or (~in_b).sum() < 5:
            continue
        per = sims[b_imgs].mean(axis=0)
        d_marg = float(per[in_b].mean() - per[~in_b].mean())
        # Stratified
        d_strata = []
        for t in strata:
            tm = b_imgs & (img_taxon == t)
            if tm.sum() < MIN_TAXON_IMGS:
                continue
            per_t = sims[tm].mean(axis=0)
            d_strata.append(per_t[in_b].mean() - per_t[~in_b].mean())
        if len(d_strata) < 2:
            d_strat, p_strat = np.nan, np.nan
        else:
            d_strat = float(np.mean(d_strata))
            null = np.empty(n_perms)
            for k in range(n_perms):
                shuf = rng.permutation(in_b)
                ds = []
                for t in strata:
                    tm = b_imgs & (img_taxon == t)
                    if tm.sum() < MIN_TAXON_IMGS:
                        continue
                    per_t = sims[tm].mean(axis=0)
                    ds.append(per_t[shuf].mean() - per_t[~shuf].mean())
                null[k] = float(np.mean(ds))
            p_strat = float((null >= d_strat).mean())
        rows.append({"biome": b, "n_imgs": int(b_imgs.sum()),
                     "n_motifs_in_biome": int(in_b.sum()),
                     "delta_marg": d_marg,
                     "delta_strat": d_strat, "p_strat": p_strat})
    return pd.DataFrame(rows)


def compute_biome_tell(motif_emb, motif_ids, img_emb, img_meta):
    """Per-motif biome-tell z-score on the sentence-pooled embedding.

    For each motif M with biome assignment set B_M, compute the gap
    between max similarity to assigned biomes' visual prototypes and
    mean similarity to non-assigned biomes' visual prototypes,
    normalised by the std of off-target similarities."""
    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    img_biome = img_meta["photo_biome_wwf"].fillna("").values
    # Build per-biome visual prototype (mean image embedding)
    proto = np.zeros((len(biomes), img_emb.shape[1]), dtype=np.float32)
    for j, b in enumerate(biomes):
        mask = img_biome == b
        if mask.sum() == 0: continue
        v = img_emb[mask].mean(axis=0)
        proto[j] = v / (np.linalg.norm(v) + 1e-12)
    sims = motif_emb @ proto.T  # (n_motifs, n_biomes)
    z = np.full(len(motif_ids), np.nan)
    for i, mid in enumerate(motif_ids):
        assigned = mb_set.get(mid, set())
        in_idx = [j for j, b in enumerate(biomes) if b in assigned]
        out_idx = [j for j, b in enumerate(biomes) if b not in assigned]
        if not in_idx or len(out_idx) < 2:
            continue
        s_in = sims[i, in_idx].max()
        s_out = sims[i, out_idx]
        z[i] = (s_in - s_out.mean()) / (s_out.std() + 1e-12)
    return z


def main():
    # Load sentence-pooled motif embeddings + meta + iNat images
    motif_emb = np.load(EMB / "motif_emb_llm_pass2_abstract_sentpooled.npy")
    meta = pd.read_parquet(EMB / "motif_meta_llm_pass2_sentpooled.parquet")
    valid = meta["valid"].values
    motif_emb_v = motif_emb[valid]
    meta_v = meta[valid].reset_index(drop=True)
    motif_ids = meta_v["motif_id"].astype(str).tolist()

    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet"
                                ).reset_index(drop=True)

    # ============ Test 1 — high-tell vs low-tell split ============
    print("=== biome-tell z-score per motif ===", flush=True)
    z = compute_biome_tell(motif_emb_v, motif_ids, img_emb, img_meta)
    finite = ~np.isnan(z)
    z_use = z[finite]
    motif_emb_use = motif_emb_v[finite]
    motif_ids_use = [motif_ids[i] for i in range(len(motif_ids)) if finite[i]]
    z_med = float(np.median(z_use))
    print(f"  n motifs with tell: {len(z_use)}, median z = {z_med:+.3f}",
          flush=True)

    rows_all = []
    for half_name, mask in [("low_tell", z_use < z_med),
                              ("high_tell", z_use >= z_med)]:
        print(f"\n  computing stratified Δ on {half_name} half "
              f"(n={int(mask.sum())} motifs) ...", flush=True)
        sub_emb = motif_emb_use[mask]
        sub_ids = [motif_ids_use[i] for i in range(len(motif_ids_use))
                    if mask[i]]
        df = stratified_test(sub_emb, sub_ids, img_emb, img_meta)
        df["half"] = half_name
        rows_all.append(df)
        print(f"    μΔ marg = {df['delta_marg'].mean()*1000:+.3f}, "
              f"μΔ strat = {df['delta_strat'].mean()*1000:+.3f}",
              flush=True)
    df_split = pd.concat(rows_all, ignore_index=True)
    out1 = EMB / "v3_biome_tell_split.csv"
    df_split.to_csv(out1, index=False)
    print(f"\n  wrote {out1}", flush=True)

    # ============ Test 2 — within-Glottolog biome-swap null ============
    print("\n\n=== within-Glottolog-macroarea biome-swap null ===", flush=True)
    trad = pd.read_parquet(MAP / "traditions.parquet")
    trad["macroarea"] = trad.apply(
        lambda r: macroarea_from_coords(r["lat"], r["lon"]), axis=1)
    print("  macroarea distribution:")
    print(trad["macroarea"].value_counts(dropna=False))

    # Build motif -> set(traditions) and tradition -> (biome, macroarea)
    tm = pd.read_parquet(MAP / "tradition_motif.parquet")
    trad_lookup = trad.set_index("oid")[["biome_wwf", "macroarea"]].to_dict("index")

    # Build motif -> set(biomes) under random within-macroarea swap
    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})

    # First compute the observed stratified Δ on the full sentpool corpus
    print("\n  observed stratified Δ (sanity check) ...", flush=True)
    obs_df = stratified_test(motif_emb_v, motif_ids, img_emb, img_meta,
                              n_perms=200)  # we'll override p with swap null
    obs_strat = obs_df.set_index("biome")["delta_strat"].to_dict()

    # Group traditions by macro-area, then within each macro-area group by biome
    rng = np.random.default_rng(0)
    macro_to_trads = trad.dropna(subset=["macroarea"]).groupby("macroarea")["oid"].apply(list).to_dict()
    macro_to_biome_to_trads = {}
    for ma, oids in macro_to_trads.items():
        biome_groups = {}
        for oid in oids:
            b = trad_lookup.get(oid, {}).get("biome_wwf")
            if isinstance(b, str) and b != "N/A":
                biome_groups.setdefault(b, []).append(oid)
        macro_to_biome_to_trads[ma] = biome_groups

    # Per-motif tradition list
    motif_trads = {mid: list(sub["oid"]) for mid, sub in tm.groupby("motif_id")}

    # Run N_PERMS swap permutations
    print(f"\n  running {N_PERMS} within-macroarea biome swaps ...", flush=True)
    null_deltas_per_biome = {b: [] for b in biomes}
    for k in range(N_PERMS):
        # For each tradition, decide a new biome by sampling another biome
        # within the same macro-area; otherwise keep original
        trad_new_biome = {}
        for ma, biome_groups in macro_to_biome_to_trads.items():
            biomes_here = [b for b in biome_groups.keys()]
            if len(biomes_here) < 2:
                for oid in macro_to_trads.get(ma, []):
                    trad_new_biome[oid] = trad_lookup.get(oid, {}).get("biome_wwf")
                continue
            for oid in macro_to_trads.get(ma, []):
                orig_b = trad_lookup.get(oid, {}).get("biome_wwf")
                # Sample a different biome from this macro-area
                others = [b for b in biomes_here if b != orig_b]
                trad_new_biome[oid] = (rng.choice(others)
                                          if others else orig_b)

        # Rebuild motif -> set(biomes) under swap
        in_B_perm = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
        for i, mid in enumerate(motif_ids):
            new_bs = set()
            for oid in motif_trads.get(mid, []):
                nb = trad_new_biome.get(oid)
                if isinstance(nb, str): new_bs.add(nb)
            for j, b in enumerate(biomes):
                if b in new_bs:
                    in_B_perm[i, j] = True

        # Compute stratified Δ for each biome under this permutation
        img_biome = img_meta["photo_biome_wwf"].fillna("").values
        img_taxon = img_meta["iconic_taxon"].fillna("").values
        sims = img_emb @ motif_emb_v.T
        sims = sims - sims.mean(axis=0, keepdims=True)
        strata = sorted([t for t in pd.Series(img_taxon).unique()
                          if t and t != "N/A"])
        for j, b in enumerate(biomes):
            in_b = in_B_perm[:, j]
            if in_b.sum() < 5 or (~in_b).sum() < 5:
                null_deltas_per_biome[b].append(np.nan); continue
            b_imgs = img_biome == b
            if b_imgs.sum() < MIN_BIOME_IMGS:
                null_deltas_per_biome[b].append(np.nan); continue
            d_strata = []
            for t in strata:
                tm_mask = b_imgs & (img_taxon == t)
                if tm_mask.sum() < MIN_TAXON_IMGS:
                    continue
                per_t = sims[tm_mask].mean(axis=0)
                d_strata.append(per_t[in_b].mean() - per_t[~in_b].mean())
            if len(d_strata) < 2:
                null_deltas_per_biome[b].append(np.nan); continue
            null_deltas_per_biome[b].append(float(np.mean(d_strata)))
        if (k+1) % 100 == 0:
            print(f"    swap perm {k+1}/{N_PERMS}", flush=True)

    rows_swap = []
    for b in biomes:
        obs_d = obs_strat.get(b, np.nan)
        nulls = np.array([x for x in null_deltas_per_biome[b]
                           if not np.isnan(x)])
        if len(nulls) < 50 or np.isnan(obs_d):
            continue
        p = float((nulls >= obs_d).mean())
        rows_swap.append({"biome": b, "delta_strat_observed": obs_d,
                           "null_mean": float(np.mean(nulls)),
                           "null_std": float(np.std(nulls)),
                           "n_null_valid": len(nulls),
                           "p_swap_null": p})
    df_swap = pd.DataFrame(rows_swap).sort_values(
        "delta_strat_observed", ascending=False)
    out2 = EMB / "v3_glottolog_swap_null.csv"
    df_swap.to_csv(out2, index=False)
    print(f"\n  wrote {out2}", flush=True)
    sig = int((df_swap["p_swap_null"] < 0.05).sum())
    print(f"  biomes significant at p<.05 under swap null: "
          f"{sig}/{len(df_swap)}", flush=True)


if __name__ == "__main__":
    main()
