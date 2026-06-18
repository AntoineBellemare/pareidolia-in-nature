"""Unsupervised biome-retrieval test.

Represent each myth by its residualised cosine-similarity vector to ALL
iNaturalist images (length ~46k), reduce with PCA -> UMAP, and ask
whether biome structure emerges from this geometry WITHOUT using biome
labels to build the embedding.

  S[m,i] = cos(motif_emb[m], img_emb[i])
  residualise per-motif (subtract each motif's mean over images) -- the
    same residualisation the headline Delta uses.
  L2-normalise rows -> PCA(50) -> UMAP(2D).

Colour the scatter by each myth's PRIMARY biome (the biome with the most
of that motif's traditions). Quantify biome retrievability with
KMeans ARI/NMI and a kNN leave-one-out classifier, versus a
label-shuffled baseline.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for shared utils
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
MAP = ROOT / "dataset/mapping_v2"
OUT = ROOT / "paper/figures"

from make_phase2_figures import short_biome, biome_color
from motif_specificity_controls import biome_motif_membership_count


def primary_biome(motif_ids):
    """Biome with the most traditions per motif; plus n_biomes (breadth)."""
    trad = pd.read_parquet(MAP / "traditions.parquet")
    tm = pd.read_parquet(MAP / "tradition_motif.parquet")
    tm["motif_id"] = tm["motif_id"].astype(str)
    oid2biome = trad.set_index("oid")["biome_wwf"].to_dict()
    prim, nbiome = {}, {}
    for mid, sub in tm.groupby("motif_id"):
        counts = {}
        for oid in sub["oid"]:
            b = oid2biome.get(oid)
            if isinstance(b, str) and b != "N/A":
                counts[b] = counts.get(b, 0) + 1
        if counts:
            prim[mid] = max(counts, key=counts.get)
            nbiome[mid] = len(counts)
    return ([prim.get(m) for m in motif_ids],
            [nbiome.get(m, 0) for m in motif_ids])


def retrievability(X, labels, name, k=15):
    """KMeans ARI/NMI + kNN leave-one-out accuracy vs shuffled baseline."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_predict
    lab = np.array(labels)
    keep = lab != None  # noqa
    X, lab = X[keep], lab[keep]
    uniq = sorted(set(lab))
    y = np.array([uniq.index(l) for l in lab])
    K = len(uniq)
    km = KMeans(n_clusters=K, n_init=6, random_state=0).fit(X)
    ari = adjusted_rand_score(y, km.labels_)
    nmi = normalized_mutual_info_score(y, km.labels_)
    # kNN leave-one-out (cross_val_predict with cv=loo is slow; use 5-fold)
    knn = KNeighborsClassifier(n_neighbors=k)
    pred = cross_val_predict(knn, X, y, cv=5)
    acc = float((pred == y).mean())
    # baselines
    rng = np.random.default_rng(0)
    chance = max(np.bincount(y)) / len(y)  # majority-class
    yperm = rng.permutation(y)
    pred_p = cross_val_predict(KNeighborsClassifier(n_neighbors=k), X, yperm, cv=5)
    acc_perm = float((pred_p == yperm).mean())
    print(f"  [{name}] n={len(y)} K={K}  "
          f"KMeans ARI={ari:.3f} NMI={nmi:.3f}  "
          f"kNN-5fold acc={acc:.3f} (majority={chance:.3f}, shuffled={acc_perm:.3f})",
          flush=True)
    return dict(name=name, n=len(y), K=K, ari=ari, nmi=nmi,
                knn_acc=acc, majority=chance, shuffled=acc_perm)


def main():
    motif_emb = np.load(EMB / "motif_emb_llm_pass2_abstract_sentpooled.npy")
    meta = pd.read_parquet(EMB / "motif_meta_llm_pass2_sentpooled.parquet")
    valid = meta["valid"].values
    motif_emb = motif_emb[valid]
    motif_ids = meta[valid]["motif_id"].astype(str).tolist()
    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    print(f"motifs={motif_emb.shape[0]} images={img_emb.shape[0]}", flush=True)

    # S = residualised cosine (per-motif centered), L2-normalised rows
    print("computing similarity matrix ...", flush=True)
    S = (motif_emb @ img_emb.T).astype(np.float32)          # (n_motif, n_img)
    S -= S.mean(axis=1, keepdims=True)                       # per-motif center
    S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)   # L2 rows
    print(f"S shape {S.shape}", flush=True)

    # PCA -> 50
    print("PCA -> 50 ...", flush=True)
    from sklearn.decomposition import PCA
    Xp = PCA(n_components=50, svd_solver="randomized",
             random_state=0).fit_transform(S)
    print(f"  PCA done, explained var (top5): "
          f"{PCA(n_components=5, random_state=0).fit(S).explained_variance_ratio_.round(3)}",
          flush=True)

    prim, nbio = primary_biome(motif_ids)
    prim = np.array(prim, dtype=object)
    nbio = np.array(nbio)

    # UMAP
    print("UMAP ...", flush=True)
    import umap
    XY = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="cosine",
                   random_state=42).fit_transform(Xp)

    # ---- quantify retrievability ----
    print("\n=== Biome retrievability from image-correlation geometry ===", flush=True)
    summ = []
    has = np.array([p is not None for p in prim])
    summ.append(retrievability(Xp[has], prim[has], "all motifs (PCA-50)"))
    specA = has & (nbio <= 3)
    summ.append(retrievability(Xp[specA], prim[specA], "Spec A (<=3 biomes)"))
    pd.DataFrame(summ).to_csv(EMB / "umap_biome_retrievability.csv", index=False)

    # ---- scatter plots ----
    biomes_present = [b for b in sorted(set(p for p in prim if p is not None))]
    def scatter(ax, mask, title):
        ax.set_facecolor("white")
        # plot grey background of all points
        ax.scatter(XY[:, 0], XY[:, 1], s=4, c="#e8e8e8", edgecolors="none", zorder=1)
        for b in biomes_present:
            m = mask & (prim == b)
            if m.sum() == 0:
                continue
            ax.scatter(XY[m, 0], XY[m, 1], s=9, c=biome_color(b),
                       edgecolors="none", alpha=0.8, zorder=2,
                       label=f"{short_biome(b)} ({m.sum()})")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        for s in ax.spines.values(): s.set_color("#ccc")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor("white")
    scatter(a1, has, "A. All motifs, coloured by primary biome")
    scatter(a2, specA, "B. Biome-specific (Spec A) motifs only")
    a2.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, markerscale=1.6)
    fig.suptitle("Myths embedded by their residualised image-correlation vector "
                 "(PCA-50 -> UMAP); colour = primary biome",
                 fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.92, right=0.86, wspace=0.05)
    out = OUT / "figS_myth_image_umap.png"
    fig.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}", flush=True)
    np.save(EMB / "umap_xy.npy", XY)


if __name__ == "__main__":
    main()
