"""Ladder rung-1 robust statistics battery.

Loads the four sentence-pooled embeddings (full Russian myth + species /
place / ethnonym baselines) and runs:

 1. Base residualised marginal Δ per biome for each text set, with
    permutation p-values and bootstrap CIs on μΔ.
 2. Per-biome decomposition table (full vs each baseline) + across-biome
    partial correlations.
 3. Species-matched permutation null on Δ_full: biome labels shuffled
    only among species-similar motifs (K-means blocks on species
    embeddings). Tests whether full-myth alignment exceeds what species
    naming alone determines.
 4. Independent triangulation — embedding-space species-subspace
    projection: project the species direction out of the full-myth
    embeddings and recompute Δ. If it survives, the alignment is not
    reducible to the species axis.

Outputs: dataset/imagery/embeddings/siglip2-large/ladder/stats_*.csv
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


def per_biome_delta(motif_emb, in_B, biomes, img_emb, img_biome,
                     valid_mask, perm=N_PERMS, boot=N_BOOT):
    """Residualised marginal Δ per biome with perm-p and bootstrap CI."""
    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(0, keepdims=True)
    rows = []
    for j, b in enumerate(biomes):
        bm = img_biome == b
        if bm.sum() < MIN_BIOME_IMGS:
            continue
        in_b = in_B[:, j] & valid_mask
        out_b = (~in_B[:, j]) & valid_mask
        if in_b.sum() < 5 or out_b.sum() < 5:
            continue
        per = sims[bm].mean(0)
        d = float(per[in_b].mean() - per[out_b].mean())
        # permutation over labels among valid motifs
        idx = np.where(valid_mask)[0]
        lab = in_B[idx, j]
        null = np.empty(perm)
        for k in range(perm):
            sh = rng.permutation(lab)
            null[k] = per[idx[sh]].mean() - per[idx[~sh]].mean()
        p = float((null >= d).mean())
        rows.append({"biome": b, "delta": d, "p": p,
                      "n_in": int(in_b.sum())})
    df = pd.DataFrame(rows)
    # bootstrap CI on μΔ across biomes
    if len(df):
        muboot = [np.mean(rng.choice(df["delta"].values, len(df), replace=True))
                  for _ in range(boot)]
        ci = (float(np.percentile(muboot, 2.5)),
              float(np.percentile(muboot, 97.5)))
    else:
        ci = (np.nan, np.nan)
    return df, ci


def species_matched_null(full_emb, species_emb, in_B, biomes, img_emb,
                          img_biome, valid, K=60, perm=N_PERMS):
    """Permute biome membership only within species-similarity blocks."""
    from sklearn.cluster import KMeans
    idx = np.where(valid)[0]
    sp = species_emb[idx]
    km = KMeans(n_clusters=min(K, len(idx)//5), n_init=4, random_state=0).fit(sp)
    block = km.labels_  # block per valid motif
    sims = img_emb @ full_emb.T
    sims = sims - sims.mean(0, keepdims=True)
    rows = []
    for j, b in enumerate(biomes):
        bm = img_biome == b
        if bm.sum() < MIN_BIOME_IMGS:
            continue
        lab = in_B[idx, j]
        if lab.sum() < 5 or (~lab).sum() < 5:
            continue
        per = sims[bm].mean(0)
        d = float(per[idx[lab]].mean() - per[idx[~lab]].mean())
        # within-block permutation of labels
        null = np.empty(perm)
        for k in range(perm):
            sh = lab.copy()
            for bl in np.unique(block):
                m = block == bl
                sh[m] = rng.permutation(lab[m])
            null[k] = per[idx[sh]].mean() - per[idx[~sh]].mean()
        rows.append({"biome": b, "delta_obs": d,
                      "null_mean": float(null.mean()),
                      "p_matched": float((null >= d).mean())})
    return pd.DataFrame(rows)


def project_out(full_emb, species_emb, valid, n_comp=10):
    """Remove top species-subspace directions from full embeddings."""
    idx = np.where(valid)[0]
    S = species_emb[idx]
    S = S - S.mean(0, keepdims=True)
    # top principal directions of species embeddings
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    V = Vt[:n_comp].T  # dim x n_comp
    proj = full_emb @ V @ V.T
    fp = full_emb - proj
    fp = fp / (np.linalg.norm(fp, axis=1, keepdims=True) + 1e-12)
    return fp


def main():
    motif_ids, embs, valids, img_emb, img_meta = load()
    biomes, in_B = biome_membership(motif_ids)
    img_biome = img_meta["photo_biome_wwf"].fillna("").values
    print(f"motifs={len(motif_ids)} biomes={len(biomes)}", flush=True)

    # 1+2. Base Δ per set
    print("\n=== Base Δ per text set ===", flush=True)
    base = {}
    for n in ["full", "species", "place", "ethnonym"]:
        df, ci = per_biome_delta(embs[n], in_B, biomes, img_emb, img_biome,
                                  valids[n])
        base[n] = df.set_index("biome")
        print(f"  {n:9s} μΔ={df['delta'].mean()*1000:+.3f}×10⁻³  "
              f"95%CI[{ci[0]*1000:+.3f},{ci[1]*1000:+.3f}]  "
              f"sig {int((df['p']<0.05).sum())}/{len(df)}", flush=True)

    # Decomposition table + partial correlations
    tab = base["full"][["delta"]].rename(columns={"delta": "delta_full"})
    for n in ["species", "place", "ethnonym"]:
        tab = tab.join(base[n][["delta"]].rename(columns={"delta": f"delta_{n}"}))
    tab.to_csv(LAD / "stats_decomposition.csv")
    print("\n=== Across-biome partial structure ===", flush=True)
    t = tab.dropna()
    for n in ["species", "place", "ethnonym"]:
        r = np.corrcoef(t["delta_full"], t[f"delta_{n}"])[0, 1]
        # residual of full after regressing on baseline
        x = np.c_[np.ones(len(t)), t[f"delta_{n}"].values]
        beta = np.linalg.lstsq(x, t["delta_full"].values, rcond=None)[0]
        res = t["delta_full"].values - x @ beta
        print(f"  full vs {n:9s}: r={r:+.2f}  residual μΔ={res.mean()*1000:+.3f}×10⁻³", flush=True)

    # 3. Species-matched permutation null
    print("\n=== Species-matched permutation null (Δ_full) ===", flush=True)
    mn = species_matched_null(embs["full"], embs["species"], in_B, biomes,
                               img_emb, img_biome,
                               valids["full"] & valids["species"])
    mn.to_csv(LAD / "stats_species_matched_null.csv", index=False)
    nsig = int((mn["p_matched"] < 0.05).sum())
    print(f"  observed μΔ_full={mn['delta_obs'].mean()*1000:+.3f}  "
          f"matched-null μ={mn['null_mean'].mean()*1000:+.3f}  "
          f"biomes surviving p<.05: {nsig}/{len(mn)}", flush=True)

    # 4. Embedding-space species-subspace projection (triangulation)
    print("\n=== Species-subspace projection (Δ_full after removing species axis) ===", flush=True)
    for nc in [5, 10, 20]:
        fp = project_out(embs["full"], embs["species"],
                          valids["full"] & valids["species"], n_comp=nc)
        df, ci = per_biome_delta(fp, in_B, biomes, img_emb, img_biome,
                                  valids["full"] & valids["species"],
                                  perm=500, boot=300)
        print(f"  proj {nc:2d} comps: μΔ={df['delta'].mean()*1000:+.3f}×10⁻³  "
              f"95%CI[{ci[0]*1000:+.3f},{ci[1]*1000:+.3f}]  "
              f"sig {int((df['p']<0.05).sum())}/{len(df)}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
