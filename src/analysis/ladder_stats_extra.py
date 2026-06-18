"""Complete the matched-null ladder: place-matched, ethnonym-matched, and
the strongest joint (species+place+ethnonym)-matched permutation null on
Δ_full. If the full-myth alignment survives the joint null, it exceeds
what ALL THREE identity-naming classes jointly determine.
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
MIN_BIOME_IMGS = 50
rng = np.random.default_rng(7)


def matched_null(full_emb, block_emb, in_B, biomes, img_emb, img_biome,
                  valid, K=60, perm=N_PERMS):
    from sklearn.cluster import KMeans
    idx = np.where(valid)[0]
    km = KMeans(n_clusters=min(K, len(idx)//5), n_init=4,
                random_state=0).fit(block_emb[idx])
    block = km.labels_
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


def main():
    man = pd.read_parquet(LAD / "manifest.parquet")
    motif_ids = man["motif_id"].astype(str).tolist()
    full = np.load(LAD / "emb_full.npy")
    sp = np.load(LAD / "emb_species.npy")
    pl = np.load(LAD / "emb_place.npy")
    et = np.load(LAD / "emb_ethnonym.npy")
    v = {n: man[f"valid_{n}"].values for n in
         ["full", "species", "place", "ethnonym"]}
    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet").reset_index(drop=True)
    img_biome = img_meta["photo_biome_wwf"].fillna("").values

    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    in_B = np.zeros((len(motif_ids), len(biomes)), bool)
    for i, mid in enumerate(motif_ids):
        for j, b in enumerate(biomes):
            if b in mb_set.get(mid, set()):
                in_B[i, j] = True

    # joint block embedding: concat of the three baselines (L2 each)
    def l2(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    joint = np.concatenate([l2(sp), l2(pl), l2(et)], axis=1)

    tests = [
        ("place",    pl, v["full"] & v["place"]),
        ("ethnonym", et, v["full"] & v["ethnonym"]),
        ("joint",    joint, v["full"] & v["species"] & v["place"] & v["ethnonym"]),
    ]
    summary = []
    for name, blk, mask in tests:
        print(f"=== {name}-matched null ===", flush=True)
        df = matched_null(full, blk, in_B, biomes, img_emb, img_biome, mask)
        df.to_csv(LAD / f"stats_{name}_matched_null.csv", index=False)
        nsig = int((df["p"] < 0.05).sum())
        print(f"  observed μΔ={df['delta_obs'].mean()*1000:+.3f}  "
              f"null μ={df['null_mean'].mean()*1000:+.3f}  "
              f"survive p<.05: {nsig}/{len(df)}", flush=True)
        summary.append({"null": name, "obs": df['delta_obs'].mean()*1000,
                        "null_mean": df['null_mean'].mean()*1000,
                        "n_sig": nsig, "n": len(df)})
    pd.DataFrame(summary).to_csv(LAD / "stats_matched_null_summary.csv", index=False)
    print("\nSUMMARY:")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
