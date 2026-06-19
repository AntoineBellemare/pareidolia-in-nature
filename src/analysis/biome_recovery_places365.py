"""Strategy 3 (unsupervised geometry recovery) on Places365 scene imagery.

Mirror of src/analysis/biome_recovery_figure.py, but on the Places365
strict-landscape corpus (1,655 scenes over 11 WWF biomes, scene-only with no
iconic-taxon labels). Two differences forced by the corpus, both honest:

  * Panel A (own-biome retrieval) uses the MARGINAL per-biome similarity
    profile rather than the taxon-stratified one, because Places365 has no
    iconic-taxon labels -- this is the same marginal frame the paper already
    uses for Places365 in Fig 2 and S1.
  * Panel B (per-biome decodability) builds the Places365 analog of
    umap_pca50: PCA-50 of the per-motif-centred, per-motif-normalised
    myth x Places365-image similarity matrix (identical recipe to
    myth_image_umap2.py, just on the scene corpus).

Biome labels are NEVER used to build the geometry. No external API calls;
runs on existing local embeddings (Places365 image embeddings + the same
LLM-clean sentence-pooled motif embeddings as the iNaturalist version).
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


def bh(pvals):
    p = np.asarray(pvals, float); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for r in range(n - 1, -1, -1):
        i = o[r]; prev = min(prev, p[i] * n / (r + 1)); q[i] = prev
    return q


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
    me = me / (np.linalg.norm(me, axis=1, keepdims=True) + 1e-12)

    pe = np.load(EMB / "places365_strict/img_emb.npy").astype(np.float32)
    pe = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-12)
    pm = pd.read_parquet(EMB / "places365_strict/img_paths.parquet").reset_index(drop=True)
    img_biome = pm["photo_biome_wwf"].fillna("").values

    pbmap = primary_biome(motif_ids)
    prim = np.array([pbmap.get(m) for m in motif_ids], dtype=object)
    has = np.array([p is not None for p in prim])

    # myth x scene similarity, per-motif centred (residualised)
    S = (me @ pe.T).astype(np.float32)
    Sret = S - S.mean(1, keepdims=True)

    biomes = sorted({b for b in prim if b is not None})
    order = [b for b in biomes if (img_biome == b).sum() >= 50 and (prim == b).sum() >= 20]
    K = len(order)
    print(f"Places365 recovery: {K} biomes with >=50 scenes and >=20 primary motifs", flush=True)
    print("  biomes:", [short_biome(b) for b in order], flush=True)

    # ---- Panel A: marginal per-biome similarity, own-biome retrieval ----
    P = np.full((S.shape[0], K), np.nan, dtype=np.float32)
    for j, bj in enumerate(order):
        P[:, j] = Sret[:, img_biome == bj].mean(axis=1)

    own = np.array([order.index(p) if (p in order) else -1 for p in prim])
    keepr = (own >= 0) & ~np.isnan(P).any(axis=1)
    Pk, ownk = P[keepr], own[keepr]
    own_rank = np.array([1 + int((Pk[m] > Pk[m, ownk[m]]).sum()) for m in range(len(ownk))])
    cmc = np.array([(own_rank <= k).mean() for k in range(1, K + 1)])
    chance = np.arange(1, K + 1) / K
    top1, top1_chance = cmc[0], chance[0]
    pct = 1.0 - (own_rank - 1) / (K - 1)
    mean_pct = pct.mean()
    rng0 = np.random.default_rng(0)
    null_pct = []
    for _ in range(2000):
        perm = rng0.integers(0, K, size=len(ownk))
        rr = np.array([1 + int((Pk[m] > Pk[m, perm[m]]).sum()) for m in range(len(perm))])
        null_pct.append((1.0 - (rr - 1) / (K - 1)).mean())
    null_pct = np.array(null_pct)
    p_pct = (1 + int((null_pct >= mean_pct).sum())) / (1 + len(null_pct))

    # ---- Panel B: PCA-50 of the myth x scene geometry, per-biome decodability ----
    Sd = S - S.mean(1, keepdims=True)
    Sd /= (np.linalg.norm(Sd, axis=1, keepdims=True) + 1e-9)
    from sklearn.decomposition import PCA
    Xp = PCA(50, svd_solver="randomized", random_state=0).fit_transform(Sd)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    N_SHUF = 200
    Xh, ph = Xp[has], prim[has]

    def bal_acc(pred, y):
        return ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()) / 2

    perbio = []
    for bi, b in enumerate(order):
        yb = (ph == b).astype(int)
        if yb.sum() < 15:
            continue
        bal = bal_acc(cross_val_predict(
            LogisticRegression(max_iter=300, class_weight="balanced"), Xh, yb, cv=5), yb)
        gen = np.random.default_rng(900 + bi)
        nd = np.empty(N_SHUF)
        for s in range(N_SHUF):
            yp = gen.permutation(yb)
            nd[s] = bal_acc(cross_val_predict(
                LogisticRegression(max_iter=200, class_weight="balanced"), Xh, yp, cv=5), yp)
        p = (1 + int((nd >= bal).sum())) / (1 + N_SHUF)
        perbio.append([b, bal, float(nd.mean()), float(nd.std()), p])
    qv = bh(np.array([r[4] for r in perbio]))
    for r, q in zip(perbio, qv):
        r.append(float(q))

    # ============ FIGURE ============
    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)
    fig.patch.set_facecolor("white")
    axR = fig.add_subplot(gs[0]); axC = fig.add_subplot(gs[1])

    # Panel A: retrieval CMC
    axR.set_facecolor("white")
    kx = np.arange(1, K + 1)
    axR.fill_between(kx, chance, cmc, color="#cf6f3f", alpha=0.18, zorder=1)
    axR.plot(kx, cmc, "-o", color="#cf6f3f", lw=2.6, ms=6, zorder=3, label="observed (marginal)")
    axR.plot(kx, chance, "--", color="#888", lw=1.6, zorder=2, label="chance (random biome)")
    axR.set_xlim(1, K); axR.set_ylim(0, 1.0); axR.set_xticks(kx)
    axR.set_xlabel("own biome within top-$k$ most-similar scene biomes", fontsize=10)
    axR.set_ylabel("fraction of myths", fontsize=10)
    sig = "$p<0.001$" if p_pct < 0.001 else f"$p={p_pct:.3f}$"
    axR.set_title("A  Own-biome retrieval on Places365 scenes\n"
                  f"mean percentile {mean_pct:.2f} vs 0.50 by chance ({sig})",
                  fontsize=11.5, fontweight="bold", loc="left")
    axR.legend(fontsize=9, loc="lower right", frameon=True)
    for s in axR.spines.values(): s.set_color("#ccc")
    axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)

    # Panel B: decodability
    axC.set_facecolor("white")
    perbio_sorted = sorted(perbio, key=lambda r: r[1])
    yb = np.arange(len(perbio_sorted))
    bals = [r[1] for r in perbio_sorted]
    nmu = np.array([r[2] for r in perbio_sorted]); nsd = np.array([r[3] for r in perbio_sorted])
    qvs = [r[5] for r in perbio_sorted]
    cols = [biome_color(r[0]) for r in perbio_sorted]
    axC.barh(yb, bals, color=cols, edgecolor="#222", lw=0.4, zorder=2, label="probe (balanced acc.)")
    axC.errorbar(nmu, yb, xerr=2 * nsd, fmt="none", ecolor="#333", elinewidth=1.4,
                 capsize=3, zorder=4, label=r"shuffled null (mean $\pm2$ sd)")
    n_sig = int(sum(q < 0.05 for q in qvs))
    for i, q in enumerate(qvs):
        if q < 0.05:
            axC.text(bals[i] + 0.004, yb[i], "$\\bigstar$", va="center", ha="left",
                     fontsize=9, color="#b8860b")
    axC.axvline(0.5, color="#999", ls="--", lw=0.8)
    axC.set_yticks(yb); axC.set_yticklabels([short_biome(r[0]) for r in perbio_sorted], fontsize=9)
    axC.set_xlabel("balanced accuracy (one-vs-rest)", fontsize=10)
    axC.set_xlim(0.45, max(bals) * 1.07)
    axC.set_title(f"B  Per-biome decodability ({n_sig}/{len(yb)} at FDR $q<.05$)\n"
                  f"mean balanced acc. {np.mean(bals):.2f} vs {nmu.mean():.2f} (200-shuffle)",
                  fontsize=11, fontweight="bold", loc="left")
    axC.legend(fontsize=8.5, loc="lower right", frameon=True)
    for s in axC.spines.values(): s.set_color("#ccc")
    axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)
    fig.suptitle("Strategy 3 (unsupervised geometry recovery) on Places365 scene imagery",
                 fontsize=12, y=1.02)

    out = OUT / "fig_biome_recovery_p365.png"
    fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"own-biome retrieval: top-1 {top1:.1%} (chance {top1_chance:.1%}), "
          f"mean percentile {mean_pct:.3f} vs null {null_pct.mean():.3f} (p={p_pct:.4f})", flush=True)
    print(f"per-biome probe vs null (mean): {np.mean([r[1] for r in perbio]):.3f} "
          f"vs {np.mean([r[2] for r in perbio]):.3f}; {n_sig}/{len(perbio)} biomes FDR-sig", flush=True)
    for r in sorted(perbio, key=lambda r: -r[1]):
        print(f"  {short_biome(r[0]):30s} bal={r[1]:.3f} null={r[2]:.3f} p={r[4]:.3f} q={r[5]:.3f}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
