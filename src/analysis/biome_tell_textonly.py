"""A1 (reviewer): a genuinely independent, text-only biome-predictability split.

The high/low biome-tell split in biome_tell.py derives predictability from
IMAGE prototypes (SigLIP similarity to per-biome mean image embeddings), so the
split variable and the alignment share the vision model. This script replaces
that split variable with a purely TEXT-side predictor that never sees an image:
a TF-IDF + logistic-regression classifier trained to predict a motif's biome
from its anonymised English text alone, evaluated out-of-fold.

Per motif we take the out-of-fold predicted-probability margin for its assigned
biome(s) as a text-only "tell" score, split motifs at the median, and recompute
the within-iconic-taxon stratified Delta on each half (same test as the
headline). If the LOW-tell half -- motifs whose biome a text classifier cannot
guess -- still aligns positively with biome imagery, the alignment is not
reducible to text-only biome predictability, and the split variable is
constructed with no reference to the image model.

Runs locally (scikit-learn, CPU); no external API. Writes
  dataset/imagery/embeddings/siglip2-large/v3_biome_tell_textonly_split.csv
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
MAP = ROOT / "dataset/mapping_v2"
ANL = ROOT / "dataset/analysis"

from motif_specificity_controls import biome_motif_membership_count
from biome_tell import stratified_test


def primary_biome(motif_ids):
    """Dominant biome per motif = the biome contributing most of its traditions."""
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    # ---- motif embeddings + meta + iNat images (for the Delta evaluation) ----
    motif_emb = np.load(EMB / "motif_emb_llm_pass2_abstract_sentpooled.npy")
    meta = pd.read_parquet(EMB / "motif_meta_llm_pass2_sentpooled.parquet")
    valid = meta["valid"].values
    motif_emb_v = motif_emb[valid]
    motif_ids = meta[valid]["motif_id"].astype(str).tolist()
    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet").reset_index(drop=True)

    # ---- anonymised English text (the same LLM-clean text used to embed) ----
    txt = pd.read_csv(ANL / "llm_rewrite_specA_gemini_pass2.csv")
    txt["motif_id"] = txt["motif_id"].astype(str)
    txt["text"] = (txt["refined_oneliner_en"].fillna("") + ". "
                   + txt["refined_translated_abstract_en"].fillna(""))
    tmap = txt.set_index("motif_id")["text"].to_dict()
    texts = [tmap.get(m, "") for m in motif_ids]

    # ---- labels: dominant biome per motif; membership for the tell margin ----
    prim = primary_biome(motif_ids)
    mb_set, _ = biome_motif_membership_count()
    y = np.array([prim.get(m, None) for m in motif_ids], dtype=object)

    have_txt = np.array([len(t.strip()) > 0 for t in texts])
    have_lab = np.array([isinstance(v, str) for v in y])
    keep = have_txt & have_lab
    # keep only classes with enough support for a 5-fold OOF fit
    ys = pd.Series(y[keep])
    common = ys.value_counts()
    ok_classes = set(common[common >= 10].index)
    keep = keep & np.array([isinstance(v, str) and v in ok_classes for v in y])
    print(f"text-only classifier: {int(keep.sum())} motifs, "
          f"{len(ok_classes)} biome classes (>=10 motifs each)", flush=True)

    texts_k = [texts[i] for i in range(len(texts)) if keep[i]]
    y_k = y[keep].astype(str)
    idx_k = np.where(keep)[0]

    # ---- TF-IDF + logistic regression, out-of-fold probabilities ----
    vec = TfidfVectorizer(sublinear_tf=True, min_df=3, ngram_range=(1, 2),
                          stop_words="english", max_features=40000)
    X = vec.fit_transform(texts_k)
    clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    proba = cross_val_predict(clf, X, y_k, cv=cv, method="predict_proba")
    classes = list(clf.fit(X, y_k).classes_)
    pred = np.array([classes[i] for i in proba.argmax(1)])
    bal = balanced_accuracy_score(y_k, pred)
    print(f"  out-of-fold biome balanced accuracy from text alone: {bal:.3f} "
          f"(chance ~ {1/len(classes):.3f})", flush=True)

    # ---- per-motif text-only tell margin (uses membership, image-free) ----
    cls_idx = {c: j for j, c in enumerate(classes)}
    tell = np.full(len(motif_ids), np.nan)
    for r, i in enumerate(idx_k):
        assigned = [cls_idx[b] for b in mb_set.get(motif_ids[i], set()) if b in cls_idx]
        nonass = [j for c, j in cls_idx.items() if c not in mb_set.get(motif_ids[i], set())]
        if not assigned or len(nonass) < 2:
            continue
        p = proba[r]
        tell[i] = (p[assigned].max() - p[nonass].mean()) / (p[nonass].std() + 1e-12)

    finite = ~np.isnan(tell)
    tv = tell[finite]
    emb_use = motif_emb_v[finite]
    ids_use = [motif_ids[i] for i in range(len(motif_ids)) if finite[i]]
    med = float(np.median(tv))
    print(f"  n motifs with text-tell: {len(tv)}; median tell = {med:+.3f}", flush=True)

    # ---- recompute the headline stratified Delta on each half ----
    rows = []
    for name, mask in [("low_tell_textonly", tv < med), ("high_tell_textonly", tv >= med)]:
        sub_emb = emb_use[mask]
        sub_ids = [ids_use[i] for i in range(len(ids_use)) if mask[i]]
        df = stratified_test(sub_emb, sub_ids, img_emb, img_meta)
        df["half"] = name
        rows.append(df)
        nsig = int((df["p_strat"] < 0.05).sum())
        print(f"  {name:20s} n={int(mask.sum()):5d}  "
              f"muDelta_strat = {df['delta_strat'].mean()*1000:+.3f}e-3  "
              f"marg = {df['delta_marg'].mean()*1000:+.3f}e-3  "
              f"sig biomes (p<.05): {nsig}/{len(df)}", flush=True)
    out = EMB / "v3_biome_tell_textonly_split.csv"
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
