"""Make the biome signal 'pop' visually, three honest ways.

Panel A: biome x biome alignment matrix. Rows = myth's primary biome,
   cols = image biome; cell = mean residualised cosine of those myths to
   those images, z-scored per row. A hot diagonal = myths align with
   their OWN biome's images. This is the effect, made visible.
Panel B: supervised biome-discriminant (LDA) 2D projection of the
   image-correlation geometry, coloured by biome. Shows biomes separate
   when you look along the biome-relevant axes that the unsupervised
   UMAP buries under content. Cross-validated point colours.
Panel C: per-biome decodability — 5-fold linear-probe accuracy of
   recovering each biome from the geometry, vs a label-shuffled null.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for shared utils
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
MAP = ROOT / "dataset/mapping_v2"
OUT = ROOT / "paper/figures"
from make_phase2_figures import short_biome, biome_color


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


def main():
    me = np.load(EMB / "motif_emb_llm_pass2_abstract_sentpooled.npy")
    meta = pd.read_parquet(EMB / "motif_meta_llm_pass2_sentpooled.parquet")
    v = meta["valid"].values; me = me[v]
    motif_ids = meta[v]["motif_id"].astype(str).tolist()
    ie = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet").reset_index(drop=True)
    img_biome = img_meta["photo_biome_wwf"].fillna("").values
    img_taxon = img_meta["iconic_taxon"].fillna("").values

    pb = primary_biome(motif_ids)
    prim = np.array([pb.get(m) for m in motif_ids], dtype=object)
    has = np.array([p is not None for p in prim])

    # residualised similarity (per-motif centred)
    S = (me @ ie.T).astype(np.float32)
    S -= S.mean(1, keepdims=True)

    biomes = sorted({b for b in prim if b is not None})
    # restrict to biomes with decent image + myth support
    biomes = [b for b in biomes if (img_biome == b).sum() >= 50
              and (prim == b).sum() >= 20]
    order = [b for b in pd.read_csv(EMB/"inat_basic/v3_biome_test_sentpool_resid.csv")
             .sort_values("delta_strat", ascending=False)["biome"] if b in biomes]

    # ---- Per-myth taxon-stratified similarity to each image-biome ----
    # P[m, j] = mean over iconic taxa t of the mean similarity of myth m to
    # (image-biome j AND taxon t) images. Uniform-over-taxa averaging removes
    # the taxon-abundance confound that makes whole image-biomes generically
    # attractive. Built once; both the biome x biome matrix (row-means) and
    # the own-biome retrieval curve are derived from it.
    taxa = sorted([t for t in pd.unique(img_taxon) if t and t != "N/A"])
    K = len(order)
    P = np.full((S.shape[0], K), np.nan, dtype=np.float32)
    for j, bj in enumerate(order):
        cols = []
        for t in taxa:
            jm = (img_biome == bj) & (img_taxon == t)
            if jm.sum() < 20:
                continue
            cols.append(S[:, jm].mean(axis=1))     # (n_myth,)
        if cols:
            P[:, j] = np.mean(np.vstack(cols), axis=0)

    # biome x biome matrix = row-means of P grouped by myth's primary biome
    M = np.full((K, K), np.nan)
    for i, bi in enumerate(order):
        mm = prim == bi
        if mm.sum():
            M[i] = np.nanmean(P[mm], axis=0)
    Z = (M - np.nanmean(M, axis=1, keepdims=True)) / (np.nanstd(M, axis=1, keepdims=True) + 1e-9)
    diag_pos = np.array([(not np.all(np.isnan(Z[i]))) and Z[i, i] > 0 for i in range(K)])
    recov = diag_pos.mean()

    # ---- own-biome retrieval: per myth, rank own biome among the K biomes ----
    own = np.array([order.index(p) if (p in order) else -1 for p in prim])
    keepr = (own >= 0) & ~np.isnan(P).any(axis=1)
    Pk, ownk = P[keepr], own[keepr]
    # rank of own biome (1 = own biome is the single most similar)
    own_rank = np.array([1 + int((Pk[m] > Pk[m, ownk[m]]).sum()) for m in range(len(ownk))])
    cmc = np.array([(own_rank <= k).mean() for k in range(1, K + 1)])       # observed
    chance = np.arange(1, K + 1) / K                                        # uniform null
    top1, top1_chance = cmc[0], chance[0]
    pct = 1.0 - (own_rank - 1) / (K - 1)                                    # 1=best, 0=worst
    mean_pct = pct.mean()
    # permutation null on mean percentile (shuffle which biome is "own")
    rng0 = np.random.default_rng(0)
    null_pct = []
    for _ in range(2000):
        perm = rng0.integers(0, K, size=len(ownk))
        rr = np.array([1 + int((Pk[m] > Pk[m, perm[m]]).sum()) for m in range(len(perm))])
        null_pct.append((1.0 - (rr - 1) / (K - 1)).mean())
    null_pct = np.array(null_pct)
    p_pct = (null_pct >= mean_pct).mean()

    # ---- per-biome decodability ----
    Xp = np.load(EMB / "umap_pca50.npy")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    Xh, ph = Xp[has], prim[has]
    # per-biome one-vs-rest probe accuracy vs shuffled
    rng = np.random.default_rng(0)
    perbio = []
    for b in order:
        yb = (ph == b).astype(int)
        if yb.sum() < 15:
            continue
        pred = cross_val_predict(LogisticRegression(max_iter=300, class_weight="balanced"),
                                  Xh, yb, cv=5)
        # balanced accuracy
        tpr = (pred[yb == 1] == 1).mean(); tnr = (pred[yb == 0] == 0).mean()
        bal = (tpr + tnr) / 2
        yp = rng.permutation(yb)
        pp = cross_val_predict(LogisticRegression(max_iter=300, class_weight="balanced"),
                                Xh, yp, cv=5)
        tpr0 = (pp[yp == 1] == 1).mean(); tnr0 = (pp[yp == 0] == 0).mean()
        bal0 = (tpr0 + tnr0) / 2
        perbio.append((b, bal, bal0))

    # ============ FIGURE ============
    fig = plt.figure(figsize=(19, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.36)
    fig.patch.set_facecolor("white")
    axR = fig.add_subplot(gs[0]); axA = fig.add_subplot(gs[1]); axC = fig.add_subplot(gs[2])

    # ---- Panel A: own-biome retrieval curve (CMC) ----
    axR.set_facecolor("white")
    kx = np.arange(1, K + 1)
    axR.fill_between(kx, chance, cmc, color="#cf6f3f", alpha=0.18, zorder=1)
    axR.plot(kx, cmc, "-o", color="#cf6f3f", lw=2.4, ms=5, zorder=3,
             label="observed (taxon-stratified)")
    axR.plot(kx, chance, "--", color="#888", lw=1.4, zorder=2, label="chance (random biome)")
    axR.set_xlim(1, K); axR.set_ylim(0, 1.0)
    axR.set_xticks(kx)
    axR.set_xlabel("own biome within top-k most-similar image biomes", fontsize=9)
    axR.set_ylabel("fraction of myths", fontsize=9)
    axR.set_title("A. Own-biome retrieval\n"
                  f"top-1 {top1:.0%} vs {top1_chance:.0%} chance · "
                  f"mean percentile {mean_pct:.2f} (p={p_pct:.3f})",
                  fontsize=11, fontweight="bold", loc="left")
    axR.legend(fontsize=8, loc="lower right", frameon=True)
    for s in axR.spines.values(): s.set_color("#ccc")
    axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)

    # ---- Panel B: taxon-stratified biome x biome alignment ----
    cmap = plt.get_cmap("RdBu_r")
    im = axA.imshow(Z, cmap=cmap, vmin=-2.2, vmax=2.2, aspect="auto")
    for i in range(K):
        axA.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                     edgecolor="#111", lw=1.6))
    axA.set_xticks(range(K))
    axA.set_xticklabels([short_biome(b) for b in order], rotation=55, ha="right", fontsize=7.5)
    axA.set_yticks(range(K))
    axA.set_yticklabels([short_biome(b) for b in order], fontsize=7.5)
    axA.set_xlabel("image biome", fontsize=9); axA.set_ylabel("myth's primary biome", fontsize=9)
    axA.set_title(f"B. Biome x biome alignment, taxon-stratified (row-z)\n"
                  f"own biome above its row average in {int(diag_pos.sum())}/{K} rows",
                  fontsize=11, fontweight="bold", loc="left")
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.03)
    cb.set_label("row z-score of mean similarity", fontsize=8)

    # ---- Panel C: per-biome decodability ----
    axC.set_facecolor("white")
    perbio_sorted = sorted(perbio, key=lambda r: r[1])
    yb = np.arange(len(perbio_sorted))
    bals = [r[1] for r in perbio_sorted]; bal0 = [r[2] for r in perbio_sorted]
    cols = [biome_color(r[0]) for r in perbio_sorted]
    axC.barh(yb, bals, color=cols, edgecolor="#222", lw=0.4, label="probe (balanced acc)")
    axC.scatter(bal0, yb, c="#444", marker="|", s=80, zorder=3, label="shuffled null")
    axC.axvline(0.5, color="#999", ls="--", lw=0.7)
    axC.set_yticks(yb); axC.set_yticklabels([short_biome(r[0]) for r in perbio_sorted], fontsize=7.5)
    axC.set_xlabel("balanced accuracy (one-vs-rest)", fontsize=9)
    axC.set_xlim(0.45, max(bals) * 1.08)
    axC.set_title("C. Per-biome decodability from geometry\n(vs label-shuffled null)",
                  fontsize=11, fontweight="bold", loc="left")
    axC.legend(fontsize=8, loc="lower right", frameon=True)
    for s in axC.spines.values(): s.set_color("#ccc")
    axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)

    out = OUT / "fig_biome_recovery.png"
    fig.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"own-biome retrieval: top-1 {top1:.1%} (chance {top1_chance:.1%}), "
          f"mean percentile {mean_pct:.3f} vs null {null_pct.mean():.3f} (p={p_pct:.4f})")
    print(f"diagonal above row-mean in {int(diag_pos.sum())}/{K} rows")
    print(f"per-biome probe vs null (mean): {np.mean([r[1] for r in perbio]):.3f} "
          f"vs {np.mean([r[2] for r in perbio]):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
