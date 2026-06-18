"""Control for fig_biome_recovery: is the unsupervised biome recovery driven
by the species / taxon channel?

The affinity vectors already use the ANONYMISED LLM-clean text (species names
replaced by class words), and Panel A's retrieval is taxon-stratified. As the
strongest text-side control we re-run both diagnostics on the CLASS-WORD-
COLLAPSED text (mammal/bird/fish/...-> "animal", tree/flower/...-> "plant"),
which removes the taxon channel from the text entirely. If biome still
recovers above chance, the result is not the species/taxon channel.

Prints, for the anonymised and the collapsed text:
  own-biome retrieval mean percentile (vs 0.50, perm p)
  per-biome decodability (mean balanced acc vs label-shuffled null)
"""
from pathlib import Path
import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
MAP = ROOT / "dataset/mapping_v2"


def primary_biome(motif_ids):
    trad = pd.read_parquet(MAP / "traditions.parquet")
    tm = pd.read_parquet(MAP / "tradition_motif.parquet")
    tm["motif_id"] = tm["motif_id"].astype(str)
    b = trad.set_index("oid")["biome_wwf"].to_dict()
    out = {}
    for mid, sub in tm.groupby("motif_id"):
        c = {}
        for oid in sub["oid"]:
            x = b.get(oid)
            if isinstance(x, str) and x != "N/A":
                c[x] = c.get(x, 0) + 1
        if c:
            out[mid] = max(c, key=c.get)
    return out


def run(me_path, meta_path, ie, img_biome, img_taxon, order, label):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    meta = pd.read_parquet(meta_path)
    me = np.load(me_path)[meta["valid"].values]
    motif_ids = meta[meta["valid"].values]["motif_id"].astype(str).tolist()
    pb = primary_biome(motif_ids)
    prim = np.array([pb.get(m) for m in motif_ids], dtype=object)
    has = np.array([p is not None for p in prim])

    S = (me @ ie.T).astype(np.float32)
    Sc = S - S.mean(1, keepdims=True)                 # per-motif centred (for P)
    taxa = sorted([t for t in pd.unique(img_taxon) if t and t != "N/A"])
    K = len(order)

    # ---- Panel A: taxon-stratified own-biome retrieval ----
    P = np.full((Sc.shape[0], K), np.nan, dtype=np.float32)
    for j, bj in enumerate(order):
        cols = []
        for t in taxa:
            jm = (img_biome == bj) & (img_taxon == t)
            if jm.sum() >= 20:
                cols.append(Sc[:, jm].mean(axis=1))
        if cols:
            P[:, j] = np.mean(np.vstack(cols), axis=0)
    own = np.array([order.index(p) if p in order else -1 for p in prim])
    keep = (own >= 0) & ~np.isnan(P).any(axis=1)
    Pk, ok = P[keep], own[keep]
    rank = np.array([1 + int((Pk[m] > Pk[m, ok[m]]).sum()) for m in range(len(ok))])
    mean_pct = (1.0 - (rank - 1) / (K - 1)).mean()
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        perm = rng.integers(0, K, size=len(ok))
        rr = np.array([1 + int((Pk[m] > Pk[m, perm[m]]).sum()) for m in range(len(perm))])
        null.append((1.0 - (rr - 1) / (K - 1)).mean())
    p_pct = (np.array(null) >= mean_pct).mean()

    # ---- Panel B: per-biome decodability (PCA-50 of L2 residualised S) ----
    Sl = S - S.mean(1, keepdims=True)
    Sl /= (np.linalg.norm(Sl, axis=1, keepdims=True) + 1e-9)
    Xp = PCA(50, svd_solver="randomized", random_state=0).fit_transform(Sl)
    Xh, ph = Xp[has], prim[has]
    bals, bal0s = [], []
    for b in order:
        yb = (ph == b).astype(int)
        if yb.sum() < 15:
            continue
        pred = cross_val_predict(LogisticRegression(max_iter=300, class_weight="balanced"),
                                 Xh, yb, cv=5)
        bals.append(((pred[yb == 1] == 1).mean() + (pred[yb == 0] == 0).mean()) / 2)
        yp = rng.permutation(yb)
        pp = cross_val_predict(LogisticRegression(max_iter=300, class_weight="balanced"),
                               Xh, yp, cv=5)
        bal0s.append(((pp[yp == 1] == 1).mean() + (pp[yp == 0] == 0).mean()) / 2)
    print(f"[{label:11s}] retrieval mean percentile {mean_pct:.3f} vs 0.500 "
          f"(p={p_pct:.4f}) | decodability {np.mean(bals):.3f} vs null "
          f"{np.mean(bal0s):.3f} (lift {np.mean(bals)-np.mean(bal0s):+.3f})", flush=True)


def main():
    ie = np.load(EMB / "inat_basic/img_emb.npy")
    im = pd.read_parquet(EMB / "inat_basic/img_paths.parquet").reset_index(drop=True)
    img_biome = im["photo_biome_wwf"].fillna("").values
    img_taxon = im["iconic_taxon"].fillna("").values
    # 9-biome order (>=50 imgs, >=20 own-biome motifs); same for both text sets
    meta = pd.read_parquet(EMB / "motif_meta_llm_pass2_sentpooled.parquet")
    mids = meta[meta["valid"].values]["motif_id"].astype(str).tolist()
    pb = primary_biome(mids)
    prim = np.array([pb.get(m) for m in mids], dtype=object)
    order = [b for b in pd.read_csv(EMB / "inat_basic/v3_biome_test_sentpool_resid.csv")
             .sort_values("delta_strat", ascending=False)["biome"]
             if (img_biome == b).sum() >= 50 and (prim == b).sum() >= 20]
    print(f"biomes in retrieval set: {len(order)}", flush=True)
    run(EMB / "motif_emb_llm_pass2_abstract_sentpooled.npy",
        EMB / "motif_meta_llm_pass2_sentpooled.parquet",
        ie, img_biome, img_taxon, order, "anonymised")
    run(EMB / "motif_emb_llm_pass2_collapsed_sentpooled.npy",
        EMB / "motif_meta_llm_pass2_collapsed_sentpooled.parquet",
        ie, img_biome, img_taxon, order, "collapsed")


if __name__ == "__main__":
    main()
