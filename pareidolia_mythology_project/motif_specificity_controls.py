"""
motif_specificity_controls.py — three principled controls for universal-motif
dilution in the biome × motif residualised test.

The default biome test treats a motif as "in biome B" if any tradition of B
carries it. This means universal motifs (e.g. "Trickster-fox", in 14/14 biomes)
contribute equally to every biome's "own" pile — adding noise but no bias.
We test whether the biome × image-biome interaction is STRONGER when we focus
on motifs that are concentrated in a small number of biomes.

Three controls:

  (A) Specificity-thresholded test:
      Only count motif as "in B" if:
        - it appears in ≤3 biomes total, AND
        - it appears in ≥3 traditions of B.
      Then run the same residualised biome test. Result CSV labelled "_specA".

  (B) TF-IDF weighted test:
      Soft weighting w(m, b) = in_b / (in_b + out_b).
      Δ_b = weighted-mean(sim(images_b, motifs_b), weights=w[:, b])
          − weighted-mean(sim(images_b, motifs_b), weights=1−w[:, b]).
      Result CSV labelled "_specB".

  (C) Breadth-stratified test:
      Run the default test 5 times, once per breadth bucket
      k ∈ {1, 2, 3, 4-6, 7+}. Report Δ per (biome, bucket).
      If Δ grows as k decreases (motif gets more biome-specific), the
      geographic interaction is real and dominated by specific motifs.
      Result CSV labelled "_specC_breadth".

Runs on:
  - iNat × oneliners × siglip2-large (baseline)
  - YFCC-filtered × abstracts × siglip2-large (best landscape run)

Output figures:
  fig50_specificity_thresholded.png
  fig51_tfidf_weighted.png
  fig52_breadth_stratified.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MAP = ROOT / "dataset/mapping_v2"
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"


# --------------------------------------------------------------------------- #
# Shared loading
# --------------------------------------------------------------------------- #

def load_run(img_emb_path, img_meta_path, motif_emb_path, motif_meta_path):
    img_emb = np.load(img_emb_path)
    img_meta = pd.read_parquet(img_meta_path).reset_index(drop=True)
    motif_emb = np.load(motif_emb_path)
    motif_meta = pd.read_parquet(motif_meta_path)
    return img_emb, img_meta, motif_emb, motif_meta


def biome_motif_membership_count():
    """Returns:
       motif_to_biomes_set : dict motif_id -> set(biomes)
       motif_to_biome_traditions : dict motif_id -> dict(biome -> n_traditions)
    """
    tm = pd.read_parquet(MAP / "tradition_motif.parquet")
    trad = pd.read_parquet(MAP / "traditions.parquet")
    biome_lookup = trad.set_index("oid")["biome_wwf"].to_dict()
    tm = tm.copy()
    tm["biome_wwf"] = tm["oid"].map(biome_lookup)
    grp = tm.groupby(["motif_id", "biome_wwf"]).size().reset_index(name="n")
    motif_to_biomes_set = {}
    motif_to_biome_traditions = {}
    for mid, sub in grp.groupby("motif_id"):
        biomes = set(b for b in sub["biome_wwf"] if isinstance(b, str))
        motif_to_biomes_set[mid] = biomes
        motif_to_biome_traditions[mid] = {
            r["biome_wwf"]: int(r["n"]) for _, r in sub.iterrows()
            if isinstance(r["biome_wwf"], str)
        }
    return motif_to_biomes_set, motif_to_biome_traditions


# --------------------------------------------------------------------------- #
# (A) Specificity-thresholded test
# --------------------------------------------------------------------------- #
def control_A(img_emb_path, img_meta_path, motif_emb_path, motif_meta_path,
              max_biomes=3, min_in_biome=3,
              n_perms=1000, label=""):
    img_emb, img_meta, motif_emb, motif_meta = load_run(
        img_emb_path, img_meta_path, motif_emb_path, motif_meta_path)
    motif_to_biomes_set, motif_to_count = biome_motif_membership_count()

    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)  # residualise
    motif_ids = motif_meta["motif_id"].tolist()
    biomes = sorted({b for s in motif_to_biomes_set.values() for b in s})

    # is_in_biome with the SPECIFICITY THRESHOLD applied
    is_in_biome = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
    for i, mid in enumerate(motif_ids):
        my_biomes = motif_to_biomes_set.get(mid, set())
        if len(my_biomes) > max_biomes:
            continue  # too universal — exclude entirely
        per_biome_n = motif_to_count.get(mid, {})
        for j, b in enumerate(biomes):
            if b in my_biomes and per_biome_n.get(b, 0) >= min_in_biome:
                is_in_biome[i, j] = True

    # Per-image biome (photo biome or fallback)
    use_biome = img_meta.get("photo_biome_wwf")
    if use_biome is None:
        use_biome = img_meta["tradition_biome_wwf"]
    use_biome = use_biome.fillna(img_meta.get("tradition_biome_wwf", "")).values

    rng = np.random.default_rng(42)
    rows = []
    for j, b in enumerate(biomes):
        b_imgs = use_biome == b
        if b_imgs.sum() < 5: continue
        b_motifs = is_in_biome[:, j]
        if b_motifs.sum() < 5: continue
        per = sims[b_imgs].mean(axis=0)
        n_own = int(b_motifs.sum())
        M_used = int(is_in_biome.any(axis=1).sum())  # consider motif used for ANY biome
        # We only restrict the "own" side; "other" remains all OTHER non-own motifs
        # that meet the criterion in some other biome.
        # Build mask for "other": passed-threshold motifs that are NOT in b.
        passed_any = is_in_biome.any(axis=1)
        b_other = passed_any & (~b_motifs)
        n_other = int(b_other.sum())
        if n_other < 5: continue
        mean_own = float(per[b_motifs].mean())
        mean_oth = float(per[b_other].mean())
        delta = mean_own - mean_oth

        # Permutation null: shuffle "in-biome b" labels among passed motifs
        idx_passed = np.where(passed_any)[0]
        passed_in_b_mask = b_motifs[idx_passed]
        null = np.empty(n_perms)
        for k in range(n_perms):
            shuf = rng.permutation(passed_in_b_mask)
            in_b_idx = idx_passed[shuf]
            not_in_b_idx = idx_passed[~shuf]
            null[k] = per[in_b_idx].mean() - per[not_in_b_idx].mean()
        p = float((null >= delta).mean())
        rows.append({"biome": b, "n_imgs": int(b_imgs.sum()),
                     "n_motifs_in_biome_specific": n_own,
                     "n_motifs_other_specific": n_other,
                     "delta": delta, "p_one_sided": p})
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    return df


# --------------------------------------------------------------------------- #
# (B) TF-IDF weighted test
# --------------------------------------------------------------------------- #
def control_B(img_emb_path, img_meta_path, motif_emb_path, motif_meta_path,
              n_perms=1000, label=""):
    img_emb, img_meta, motif_emb, motif_meta = load_run(
        img_emb_path, img_meta_path, motif_emb_path, motif_meta_path)
    motif_to_biomes_set, motif_to_count = biome_motif_membership_count()

    # Per biome: how many traditions are in it?
    trad = pd.read_parquet(MAP / "traditions.parquet")
    biome_size = trad.groupby("biome_wwf").size().to_dict()  # biome -> n_traditions
    total_trad = trad["oid"].nunique()

    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)
    motif_ids = motif_meta["motif_id"].tolist()
    biomes = sorted({b for s in motif_to_biomes_set.values() for b in s})
    biome_to_idx = {b: i for i, b in enumerate(biomes)}

    # Build weight matrix W[i, j] = in-biome rate / (in + out)
    W = np.zeros((len(motif_ids), len(biomes)), dtype=np.float32)
    for i, mid in enumerate(motif_ids):
        cnt = motif_to_count.get(mid, {})
        total_in = sum(cnt.values())
        for j, b in enumerate(biomes):
            in_b = cnt.get(b, 0)
            out_b = total_in - in_b
            # Rates relative to biome population
            in_rate = in_b / max(biome_size.get(b, 1), 1)
            other_pop = max(total_trad - biome_size.get(b, 0), 1)
            out_rate = out_b / other_pop
            denom = in_rate + out_rate
            W[i, j] = in_rate / denom if denom > 0 else 0.5

    use_biome = img_meta.get("photo_biome_wwf")
    if use_biome is None:
        use_biome = img_meta["tradition_biome_wwf"]
    use_biome = use_biome.fillna(img_meta.get("tradition_biome_wwf", "")).values

    rng = np.random.default_rng(42)
    rows = []
    for j, b in enumerate(biomes):
        b_imgs = use_biome == b
        if b_imgs.sum() < 5: continue
        per = sims[b_imgs].mean(axis=0)
        w_in = W[:, j]
        w_out = 1.0 - w_in
        if w_in.sum() < 5 or w_out.sum() < 5: continue
        mean_own = float((per * w_in).sum() / w_in.sum())
        mean_oth = float((per * w_out).sum() / w_out.sum())
        delta = mean_own - mean_oth

        # Permutation null: shuffle the weights
        null = np.empty(n_perms)
        for k in range(n_perms):
            shuf = rng.permutation(w_in)
            shuf_out = 1.0 - shuf
            null[k] = ((per * shuf).sum() / shuf.sum()
                       - (per * shuf_out).sum() / shuf_out.sum())
        p = float((null >= delta).mean())
        rows.append({"biome": b, "n_imgs": int(b_imgs.sum()),
                     "sum_weight_in": float(w_in.sum()),
                     "delta": delta, "p_one_sided": p})
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    return df


# --------------------------------------------------------------------------- #
# (C) Breadth-stratified test
# --------------------------------------------------------------------------- #
def control_C(img_emb_path, img_meta_path, motif_emb_path, motif_meta_path,
              n_perms=1000, label=""):
    img_emb, img_meta, motif_emb, motif_meta = load_run(
        img_emb_path, img_meta_path, motif_emb_path, motif_meta_path)
    motif_to_biomes_set, _ = biome_motif_membership_count()

    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)
    motif_ids = motif_meta["motif_id"].tolist()
    biomes = sorted({b for s in motif_to_biomes_set.values() for b in s})

    # Compute breadth per motif
    motif_breadth = np.array([
        len(motif_to_biomes_set.get(mid, set())) for mid in motif_ids
    ])
    print(f"breadth distribution: "
          f"{pd.Series(motif_breadth).value_counts().sort_index().to_dict()}")

    use_biome = img_meta.get("photo_biome_wwf")
    if use_biome is None:
        use_biome = img_meta["tradition_biome_wwf"]
    use_biome = use_biome.fillna(img_meta.get("tradition_biome_wwf", "")).values

    # Define buckets
    buckets = [("1", lambda k: k == 1),
               ("2", lambda k: k == 2),
               ("3", lambda k: k == 3),
               ("4-6", lambda k: 4 <= k <= 6),
               ("7+", lambda k: k >= 7)]

    is_in_biome = np.array([
        [b in motif_to_biomes_set.get(mid, set()) for b in biomes]
        for mid in motif_ids
    ])
    rng = np.random.default_rng(42)
    rows = []
    for bucket_name, predicate in buckets:
        bucket_mask = np.array([predicate(motif_breadth[i])
                                for i in range(len(motif_ids))])
        if bucket_mask.sum() < 10: continue
        # restricted-to-bucket biome test
        for j, b in enumerate(biomes):
            b_imgs = use_biome == b
            if b_imgs.sum() < 5: continue
            in_b = is_in_biome[:, j] & bucket_mask
            out_b = (~is_in_biome[:, j]) & bucket_mask
            if in_b.sum() < 5 or out_b.sum() < 5: continue
            per = sims[b_imgs].mean(axis=0)
            mean_own = float(per[in_b].mean())
            mean_oth = float(per[out_b].mean())
            delta = mean_own - mean_oth

            # Null: shuffle motif-biome assignment WITHIN the bucket
            bucket_idx = np.where(bucket_mask)[0]
            in_b_in_bucket = is_in_biome[bucket_idx, j]
            null = np.empty(n_perms)
            for k in range(n_perms):
                shuf = rng.permutation(in_b_in_bucket)
                in_idx = bucket_idx[shuf]
                out_idx = bucket_idx[~shuf]
                null[k] = per[in_idx].mean() - per[out_idx].mean()
            p = float((null >= delta).mean())
            rows.append({"bucket": bucket_name, "biome": b,
                         "n_imgs": int(b_imgs.sum()),
                         "n_motifs_in_bucket": int(bucket_mask.sum()),
                         "n_in_b": int(in_b.sum()), "n_out_b": int(out_b.sum()),
                         "delta": delta, "p_one_sided": p})
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------- #
# Run + save
# --------------------------------------------------------------------------- #

RUNS = [
    # default text: one-line descriptions
    ("iNat × oneliners",
        EMB / "img_emb.npy",
        EMB / "img_paths.parquet",
        EMB / "motif_emb_all.npy",
        EMB / "motif_meta_all.parquet"),
    ("YFCC-filtered × abstracts",
        EMB / "yfcc_filtered/img_emb.npy",
        EMB / "yfcc_filtered/img_paths.parquet",
        EMB / "motif_emb_abstracts.npy",
        EMB / "motif_meta_abstracts.parquet"),
    # naming-controlled text: hypernymed
    ("iNat × HYPERNYMED",
        EMB / "img_emb.npy",
        EMB / "img_paths.parquet",
        EMB / "motif_emb_hypernymed.npy",
        EMB / "motif_meta_hypernymed.parquet"),
    ("YFCC-filtered × HYPERNYMED",
        EMB / "yfcc_filtered/img_emb.npy",
        EMB / "yfcc_filtered/img_paths.parquet",
        EMB / "motif_emb_hypernymed.npy",
        EMB / "motif_meta_hypernymed.parquet"),
]


def main():
    for label, *paths in RUNS:
        if not all(Path(p).exists() for p in paths):
            print(f"SKIP {label}: missing files"); continue
        print(f"\n========== {label} ==========")
        print("\n[A] Specificity-thresholded:")
        df_a = control_A(*paths, label=label)
        out_a = EMB / f"specA_{label.replace(' ','').replace('×','x').replace('-','')}.csv"
        df_a.to_csv(out_a, index=False)
        print(df_a.to_string(index=False))
        print(f"  -> {out_a}")

        print("\n[B] TF-IDF weighted:")
        df_b = control_B(*paths, label=label)
        out_b = EMB / f"specB_{label.replace(' ','').replace('×','x').replace('-','')}.csv"
        df_b.to_csv(out_b, index=False)
        print(df_b.to_string(index=False))
        print(f"  -> {out_b}")

        print("\n[C] Breadth-stratified:")
        df_c = control_C(*paths, label=label)
        out_c = EMB / f"specC_breadth_{label.replace(' ','').replace('×','x').replace('-','')}.csv"
        df_c.to_csv(out_c, index=False)
        # Summary: mean Δ per bucket
        summ = df_c.groupby("bucket").agg(
            mean_delta=("delta","mean"),
            n_cells=("delta","size"),
            pct_sig=("p_one_sided", lambda x: 100*(x<0.05).mean()),
            pct_pos=("delta", lambda x: 100*(x>0).mean()),
        ).round(5)
        print(summ.to_string())
        print(f"  -> {out_c}")


if __name__ == "__main__":
    main()
