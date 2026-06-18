"""Enhanced unsupervised structure analysis of myth x image affinity.

Same geometry (residualised myth-image cosine -> PCA-50 -> UMAP) but
characterise WHICH structure dominates: biome, cultural macro-area, or
content (dominant taxon). Retrievability of each via kNN + linear probe
vs a label-shuffled null. 3-panel scatter coloured by each labelling.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for shared utils
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
MAP = ROOT / "dataset/mapping_v2"
ANA = ROOT / "dataset/analysis"
OUT = ROOT / "paper/figures"

from make_phase2_figures import short_biome, biome_color


def macroarea_from_coords(lat, lon):
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    if -45 <= lat <= -10 and 110 <= lon <= 155: return "Australia"
    if -15 <= lat <= 25 and 95 <= lon <= 180: return "Papunesia"
    if -45 <= lat <= 0 and 155 <= lon <= 180: return "Papunesia"
    if -25 <= lat <= -10 and 155 <= lon <= 180: return "Papunesia"
    if 15 <= lat <= 75 and -170 <= lon <= -50: return "North America"
    if 50 <= lat <= 80 and -180 <= lon <= -50: return "North America"
    if -56 <= lat <= 15 and -85 <= lon <= -34: return "South America"
    if -35 <= lat <= 35 and -20 <= lon <= 55: return "Africa"
    if lat > 25 and lon >= -15: return "Eurasia"
    if 5 <= lat <= 25 and 25 <= lon <= 100: return "Eurasia"
    return None

MACRO_COL = {"Africa": "#d98a3d", "Australia": "#b8553f", "Eurasia": "#4a7fb5",
             "North America": "#5fa86a", "Papunesia": "#9e6cb4",
             "South America": "#c9a23d", None: "#dddddd"}
TAXON_COL = {"mammal": "#d04f6f", "bird": "#76b6e5", "fish": "#2d8fb3",
             "reptile": "#6abc8f", "amphibian": "#4ea36f", "insect": "#a96cb0",
             "tree": "#7cbe5e", "plant": "#3a9d50", "other": "#bbbbbb"}


def labels_biome(motif_ids):
    trad = pd.read_parquet(MAP / "traditions.parquet")
    tm = pd.read_parquet(MAP / "tradition_motif.parquet")
    tm["motif_id"] = tm["motif_id"].astype(str)
    b = trad.set_index("oid")["biome_wwf"].to_dict()
    ma = {oid: macroarea_from_coords(r["lat"], r["lon"])
          for oid, r in trad.set_index("oid").iterrows()}
    prim_b, prim_m = {}, {}
    for mid, sub in tm.groupby("motif_id"):
        cb, cm = {}, {}
        for oid in sub["oid"]:
            x = b.get(oid)
            if isinstance(x, str) and x != "N/A":
                cb[x] = cb.get(x, 0)+1
            y = ma.get(oid)
            if isinstance(y, str):
                cm[y] = cm.get(y, 0)+1
        if cb: prim_b[mid] = max(cb, key=cb.get)
        if cm: prim_m[mid] = max(cm, key=cm.get)
    return prim_b, prim_m


def labels_taxon(motif_ids):
    gem = pd.read_csv(ANA / "llm_rewrite_specA_gemini_pass2.csv")
    gem = gem[gem.status == "OK"].copy(); gem["motif_id"] = gem.motif_id.astype(str)
    classes = ["mammal", "bird", "fish", "reptile", "amphibian", "insect",
               "tree", "plant"]
    out = {}
    g = gem.set_index("motif_id")["refined_translated_abstract_en"].fillna("")
    for mid in motif_ids:
        t = g.get(mid, "").lower()
        cnt = {c: len(re.findall(r"\b"+c+r"s?\b", t)) for c in classes}
        if max(cnt.values()) == 0:
            out[mid] = "other"
        else:
            out[mid] = max(cnt, key=cnt.get)
    return out


def retriev(X, labels, name):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import KMeans
    lab = np.array([l for l in labels], dtype=object)
    keep = np.array([l is not None for l in lab])
    Xk, lk = X[keep], lab[keep]
    uniq = sorted(set(lk)); y = np.array([uniq.index(l) for l in lk]); K = len(uniq)
    rng = np.random.default_rng(0)
    knn = cross_val_predict(KNeighborsClassifier(15), Xk, y, cv=5)
    acc_knn = (knn == y).mean()
    lr = cross_val_predict(LogisticRegression(max_iter=300, C=1.0), Xk, y, cv=5)
    acc_lr = (lr == y).mean()
    yp = rng.permutation(y)
    sh = cross_val_predict(KNeighborsClassifier(15), Xk, yp, cv=5)
    acc_sh = (sh == yp).mean()
    km = KMeans(K, n_init=5, random_state=0).fit(Xk)
    ari = adjusted_rand_score(y, km.labels_)
    print(f"  [{name:11s}] n={len(y)} K={K}  kNN={acc_knn:.3f} probe={acc_lr:.3f} "
          f"shuffled={acc_sh:.3f}  lift(probe-shuf)={acc_lr-acc_sh:+.3f}  ARI={ari:.3f}",
          flush=True)
    return dict(label=name, n=len(y), K=K, knn=acc_knn, probe=acc_lr,
                shuffled=acc_sh, lift=acc_lr-acc_sh, ari=ari)


def main():
    me = np.load(EMB / "motif_emb_llm_pass2_abstract_sentpooled.npy")
    meta = pd.read_parquet(EMB / "motif_meta_llm_pass2_sentpooled.parquet")
    v = meta["valid"].values; me = me[v]
    motif_ids = meta[v]["motif_id"].astype(str).tolist()
    ie = np.load(EMB / "inat_basic/img_emb.npy")

    S = (me @ ie.T).astype(np.float32)
    S -= S.mean(1, keepdims=True); S /= (np.linalg.norm(S, axis=1, keepdims=True)+1e-9)
    from sklearn.decomposition import PCA
    Xp = PCA(50, svd_solver="randomized", random_state=0).fit_transform(S)
    np.save(EMB / "umap_pca50.npy", Xp)

    pb, pm = labels_biome(motif_ids)
    pt = labels_taxon(motif_ids)
    biome = [pb.get(m) for m in motif_ids]
    macro = [pm.get(m) for m in motif_ids]
    taxon = [pt.get(m) for m in motif_ids]

    print("\n=== Retrievability from image-correlation geometry (PCA-50) ===", flush=True)
    rows = [retriev(Xp, biome, "biome"),
            retriev(Xp, macro, "macro-area"),
            retriev(Xp, taxon, "taxon/content")]
    pd.DataFrame(rows).to_csv(EMB / "umap_retrievability3.csv", index=False)

    XY = np.load(EMB / "umap_xy.npy") if (EMB/"umap_xy.npy").exists() else None
    if XY is None:
        import umap
        XY = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="cosine",
                       random_state=42).fit_transform(Xp)

    def scat(ax, labs, palette, title, legend):
        ax.set_facecolor("white")
        labs = np.array([l for l in labs], dtype=object)
        order = sorted(set(l for l in labs if l is not None),
                       key=lambda x: -(labs == x).sum())
        for l in order:
            m = labs == l
            c = palette(l) if callable(palette) else palette.get(l, "#ccc")
            lbl = (short_biome(l) if legend == "biome" else str(l))
            ax.scatter(XY[m, 0], XY[m, 1], s=7, c=c, edgecolors="none",
                       alpha=0.75, label=f"{lbl} ({m.sum()})")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        for s in ax.spines.values(): s.set_color("#ccc")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor("white")
    scat(axes[0], biome, biome_color, "A. coloured by primary BIOME", "biome")
    scat(axes[1], macro, MACRO_COL, "B. coloured by cultural MACRO-AREA", "macro")
    scat(axes[2], taxon, TAXON_COL, "C. coloured by dominant TAXON (content)", "taxon")
    for ax in axes[1:]:
        ax.legend(fontsize=7, loc="upper right", frameon=True, markerscale=1.4)
    fig.suptitle("Same myth x image-affinity geometry (PCA-50 -> UMAP), three labellings: "
                 "content dominates, geography and biome are weaker overlays",
                 fontsize=12, y=0.99)
    fig.subplots_adjust(top=0.93, wspace=0.04)
    out = OUT / "figS_myth_umap_3way.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
