"""Identity-naming ladder on the WITHIN-ICONIC-TAXON STRATIFIED Delta.

Same battery as ladder_stats.py + ladder_stats_extra.py, but every Delta is
the headline stratified statistic rather than the marginal one. For biome b
the per-motif score is the uniform mean, over iconic taxa t present in b's
images (>=20 imgs/stratum), of the motif's mean residualised similarity to
b's taxon-t images. Because both the over-motif and over-taxon averages are
uniform they commute, so substituting this stratified per-motif score for
`sims[bm].mean(0)` makes the base Delta, the matched-permutation nulls, and
the species-subspace projection all stratified with no other change.

This answers: does the identity-naming decomposition survive on the same
taxon-confound-controlled statistic the headline uses (not just the
marginal Delta)?

Outputs: dataset/.../ladder/stats_*_strat.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for shared utils
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
LAD = EMB / "ladder"
from motif_specificity_controls import biome_motif_membership_count

N_PERMS = 1000
N_BOOT = 1000
MIN_BIOME_IMGS = 50
MIN_TAXON_IMGS = 20      # matches the headline per biome-taxon stratum floor
rng = np.random.default_rng(42)


def load():
    man = pd.read_parquet(LAD / "manifest.parquet")
    motif_ids = man["motif_id"].astype(str).tolist()
    embs = {n: np.load(LAD / f"emb_{n}.npy")
            for n in ["full", "species", "place", "ethnonym"]}
    valids = {n: man[f"valid_{n}"].values
              for n in ["full", "species", "place", "ethnonym"]}
    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(
        EMB / "inat_basic/img_paths.parquet").reset_index(drop=True)
    return motif_ids, embs, valids, img_emb, img_meta


def biome_membership(motif_ids):
    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    in_B = np.zeros((len(motif_ids), len(biomes)), bool)
    for i, mid in enumerate(motif_ids):
        for j, b in enumerate(biomes):
            if b in mb_set.get(mid, set()):
                in_B[i, j] = True
    return biomes, in_B


def strat_per(sims, bm, taxon, min_t=MIN_TAXON_IMGS):
    """Taxon-stratified per-motif score: uniform mean over iconic taxa of the
    per-motif mean similarity to (biome AND taxon) images. None if no stratum
    clears the floor."""
    cols = []
    for t in [x for x in np.unique(taxon[bm]) if x and x != "N/A"]:
        m = bm & (taxon == t)
        if m.sum() >= min_t:
            cols.append(sims[m].mean(0))            # (n_motif,)
    if not cols:
        return None
    return np.mean(np.vstack(cols), axis=0)         # (n_motif,)


def per_biome_delta(motif_emb, in_B, biomes, img_emb, img_biome, taxon,
                    valid_mask, perm=N_PERMS, boot=N_BOOT):
    """Stratified Delta per biome with permutation-p and bootstrap CI on muDelta."""
    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(0, keepdims=True)
    rows = []
    for j, b in enumerate(biomes):
        bm = img_biome == b
        if bm.sum() < MIN_BIOME_IMGS:
            continue
        per = strat_per(sims, bm, taxon)
        if per is None:
            continue
        in_b = in_B[:, j] & valid_mask
        out_b = (~in_B[:, j]) & valid_mask
        if in_b.sum() < 5 or out_b.sum() < 5:
            continue
        d = float(per[in_b].mean() - per[out_b].mean())
        idx = np.where(valid_mask)[0]
        lab = in_B[idx, j]
        null = np.empty(perm)
        for k in range(perm):
            sh = rng.permutation(lab)
            null[k] = per[idx[sh]].mean() - per[idx[~sh]].mean()
        p = float((null >= d).mean())
        rows.append({"biome": b, "delta": d, "p": p, "n_in": int(in_b.sum())})
    df = pd.DataFrame(rows)
    if len(df):
        muboot = [np.mean(rng.choice(df["delta"].values, len(df), replace=True))
                  for _ in range(boot)]
        ci = (float(np.percentile(muboot, 2.5)), float(np.percentile(muboot, 97.5)))
    else:
        ci = (np.nan, np.nan)
    return df, ci


def matched_null(full_emb, block_emb, in_B, biomes, img_emb, img_biome, taxon,
                 valid, K=60, perm=N_PERMS):
    """Stratified Delta_full vs a null that shuffles biome only within
    identity-similarity K-means blocks."""
    from sklearn.cluster import KMeans
    idx = np.where(valid)[0]
    km = KMeans(n_clusters=min(K, len(idx) // 5), n_init=4,
                random_state=0).fit(block_emb[idx])
    block = km.labels_
    sims = img_emb @ full_emb.T
    sims = sims - sims.mean(0, keepdims=True)
    rows = []
    for j, b in enumerate(biomes):
        bm = img_biome == b
        if bm.sum() < MIN_BIOME_IMGS:
            continue
        per = strat_per(sims, bm, taxon)
        if per is None:
            continue
        lab = in_B[idx, j]
        if lab.sum() < 5 or (~lab).sum() < 5:
            continue
        d = float(per[idx[lab]].mean() - per[idx[~lab]].mean())
        null = np.empty(perm)
        for k in range(perm):
            sh = lab.copy()
            for bl in np.unique(block):
                m = block == bl
                sh[m] = rng.permutation(lab[m])
            null[k] = per[idx[sh]].mean() - per[idx[~sh]].mean()
        rows.append({"biome": b, "delta_obs": d,
                     "null_mean": float(null.mean()),
                     "p": float((null >= d).mean())})
    return pd.DataFrame(rows)


def project_out(full_emb, species_emb, valid, n_comp=10):
    idx = np.where(valid)[0]
    S = species_emb[idx]
    S = S - S.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    V = Vt[:n_comp].T
    fp = full_emb - full_emb @ V @ V.T
    return fp / (np.linalg.norm(fp, axis=1, keepdims=True) + 1e-12)


def main():
    motif_ids, embs, valids, img_emb, img_meta = load()
    biomes, in_B = biome_membership(motif_ids)
    img_biome = img_meta["photo_biome_wwf"].fillna("").values
    taxon = img_meta["iconic_taxon"].fillna("").values
    print(f"motifs={len(motif_ids)} biomes={len(biomes)} "
          f"taxa={len([t for t in np.unique(taxon) if t and t!='N/A'])}", flush=True)

    # 1+2. Base stratified Delta per text set + decomposition
    print("\n=== Base STRATIFIED Delta per text set ===", flush=True)
    base = {}
    for n in ["full", "species", "place", "ethnonym"]:
        df, ci = per_biome_delta(embs[n], in_B, biomes, img_emb, img_biome,
                                 taxon, valids[n])
        base[n] = df.set_index("biome")
        print(f"  {n:9s} muDelta={df['delta'].mean()*1000:+.3f}e-3  "
              f"95%CI[{ci[0]*1000:+.3f},{ci[1]*1000:+.3f}]  "
              f"sig {int((df['p']<0.05).sum())}/{len(df)}", flush=True)
    tab = base["full"][["delta"]].rename(columns={"delta": "delta_full"})
    for n in ["species", "place", "ethnonym"]:
        tab = tab.join(base[n][["delta"]].rename(columns={"delta": f"delta_{n}"}))
    tab.to_csv(LAD / "stats_decomposition_strat.csv")

    # 3. Matched-permutation nulls (species / ethnonym / place / joint), stratified
    def l2(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    joint = np.concatenate([l2(embs["species"]), l2(embs["place"]),
                            l2(embs["ethnonym"])], axis=1)
    tests = [
        ("species",  embs["species"], valids["full"] & valids["species"]),
        ("ethnonym", embs["ethnonym"], valids["full"] & valids["ethnonym"]),
        ("place",    embs["place"],   valids["full"] & valids["place"]),
        ("joint",    joint, valids["full"] & valids["species"]
         & valids["place"] & valids["ethnonym"]),
    ]
    print("\n=== STRATIFIED matched-permutation nulls (Delta_full) ===", flush=True)
    summary = []
    for name, blk, mask in tests:
        df = matched_null(embs["full"], blk, in_B, biomes, img_emb, img_biome,
                          taxon, mask)
        df.to_csv(LAD / f"stats_{name}_matched_null_strat.csv", index=False)
        nsig = int((df["p"] < 0.05).sum())
        print(f"  {name:9s} obs muDelta={df['delta_obs'].mean()*1000:+.3f}  "
              f"null mu={df['null_mean'].mean()*1000:+.3f}  "
              f"survive p<.05: {nsig}/{len(df)}", flush=True)
        summary.append({"null": name, "obs": df["delta_obs"].mean() * 1000,
                        "null_mean": df["null_mean"].mean() * 1000,
                        "n_sig": nsig, "n": len(df)})
    pd.DataFrame(summary).to_csv(LAD / "stats_matched_null_summary_strat.csv",
                                 index=False)

    # 4. Species-subspace projection, stratified
    print("\n=== STRATIFIED species-subspace projection (Delta_full) ===", flush=True)
    for nc in [5, 10, 20]:
        fp = project_out(embs["full"], embs["species"],
                         valids["full"] & valids["species"], n_comp=nc)
        df, ci = per_biome_delta(fp, in_B, biomes, img_emb, img_biome, taxon,
                                 valids["full"] & valids["species"],
                                 perm=500, boot=300)
        print(f"  proj {nc:2d} comps: muDelta={df['delta'].mean()*1000:+.3f}e-3  "
              f"95%CI[{ci[0]*1000:+.3f},{ci[1]*1000:+.3f}]  "
              f"sig {int((df['p']<0.05).sum())}/{len(df)}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
