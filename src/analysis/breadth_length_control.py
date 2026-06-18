"""Breadth-gradient sensitivity to a text-property confound.

Referee concern: universal (wide-breadth) motifs are longer / more abstract /
less biome-lexical, so the breadth gradient could be a text-property artefact
rather than ecological decoupling. We test whether breadth still predicts
per-motif own-biome alignment AFTER controlling for motif length.

Per-motif own-biome alignment delta_m = (taxon-stratified mean similarity of
motif m to its OWN biome's images) - (mean over the other biomes). Regress
delta_m on breadth (number of biomes a motif spans) and length (number of
sentences), standardized; report partial effects. Also report the breadth
gradient within length terciles.
"""
from pathlib import Path
import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
MAP = ROOT / "dataset/mapping_v2"
from make_phase2_figures import short_biome


def primary_and_breadth(motif_ids):
    trad = pd.read_parquet(MAP / "traditions.parquet")
    tm = pd.read_parquet(MAP / "tradition_motif.parquet"); tm["motif_id"] = tm["motif_id"].astype(str)
    b = trad.set_index("oid")["biome_wwf"].to_dict()
    prim, nb = {}, {}
    for mid, sub in tm.groupby("motif_id"):
        c = {}
        for oid in sub["oid"]:
            x = b.get(oid)
            if isinstance(x, str) and x != "N/A":
                c[x] = c.get(x, 0) + 1
        if c:
            prim[mid] = max(c, key=c.get); nb[mid] = len(c)
    return prim, nb


def main():
    me = np.load(EMB / "motif_emb_llm_pass2_abstract_sentpooled.npy")
    meta = pd.read_parquet(EMB / "motif_meta_llm_pass2_sentpooled.parquet")
    v = meta["valid"].values
    me = me[v]
    mids = meta[v]["motif_id"].astype(str).tolist()
    length = meta[v]["n_sentences"].values.astype(float)
    ie = np.load(EMB / "inat_basic/img_emb.npy")
    im = pd.read_parquet(EMB / "inat_basic/img_paths.parquet").reset_index(drop=True)
    ib = im["photo_biome_wwf"].fillna("").values
    tx = im["iconic_taxon"].fillna("").values

    prim, nb = primary_and_breadth(mids)
    primA = np.array([prim.get(m) for m in mids], dtype=object)
    breadth = np.array([nb.get(m, 0) for m in mids], dtype=float)

    biomes = sorted({p for p in primA if p})
    order = [b for b in biomes if (ib == b).sum() >= 50 and (primA == b).sum() >= 20]
    K = len(order)
    taxa = [t for t in np.unique(tx) if t and t != "N/A"]

    S = (me @ ie.T).astype(np.float32); S -= S.mean(1, keepdims=True)
    P = np.full((S.shape[0], K), np.nan, dtype=np.float32)
    for j, bj in enumerate(order):
        cols = [S[:, (ib == bj) & (tx == t)].mean(1) for t in taxa
                if ((ib == bj) & (tx == t)).sum() >= 20]
        if cols:
            P[:, j] = np.mean(np.vstack(cols), axis=0)

    # membership: motif m is "in" biome b if b is among its traditions' biomes
    from motif_specificity_controls import biome_motif_membership_count
    mb, _ = biome_motif_membership_count()
    inmem = np.zeros((len(mids), K), bool)
    for i, mid in enumerate(mids):
        s = mb.get(mid, set())
        for j, b in enumerate(order):
            if b in s:
                inmem[i, j] = True

    def agg_mu(mask):
        """Aggregate stratified muDelta over the K biomes, with the in-biome set
        restricted to motifs in `mask` (matches the headline Delta construction)."""
        ds = []
        for j in range(K):
            inb = inmem[:, j] & mask
            outb = ~inmem[:, j]
            if inb.sum() >= 5 and outb.sum() >= 5:
                ds.append(P[inb, j].mean() - P[outb, j].mean())
        return np.mean(ds) * 1000 if ds else np.nan

    valid_motif = ~np.isnan(P).any(axis=1)
    lnv = np.log(length + 1)
    print(f"n = {int(valid_motif.sum())} motifs; corr(breadth, log-length) = "
          f"{np.corrcoef(breadth[valid_motif], lnv[valid_motif])[0,1]:+.3f}", flush=True)
    print("Aggregate breadth gradient (stratified muDelta x1e-3) within length terciles:", flush=True)
    lt = np.quantile(length[valid_motif], [1/3, 2/3])
    print(f"  {'length':8s} {'specific(<=3)':>14s} {'mid(4-9)':>10s} {'universal(>=10)':>16s}", flush=True)
    for lo, hi, lab in [(-1, lt[0], "short"), (lt[0], lt[1], "mid"), (lt[1], 1e9, "long")]:
        lm = valid_motif & (length > lo) & (length <= hi)
        spec = agg_mu(lm & (breadth <= 3))
        midb = agg_mu(lm & (breadth >= 4) & (breadth <= 9))
        univ = agg_mu(lm & (breadth >= 10))
        print(f"  {lab:8s} {spec:14.3f} {midb:10.3f} {univ:16.3f}", flush=True)
    print("Overall (all lengths):", flush=True)
    print(f"  specific={agg_mu(valid_motif & (breadth<=3)):.3f}  "
          f"mid={agg_mu(valid_motif & (breadth>=4) & (breadth<=9)):.3f}  "
          f"universal={agg_mu(valid_motif & (breadth>=10)):.3f}", flush=True)


if __name__ == "__main__":
    main()
