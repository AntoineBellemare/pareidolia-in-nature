"""No-LLM-stripping reanalysis engine.

Significance for the biome--mythology alignment on the RAW (un-anonymised)
myths via two controls that hold species+place+ethnonym constant, instead of
the LLM anonymisation:

  DISCRETE  matched-permutation null: shuffle biome only within K-means blocks
            of the concatenated species+place+ethnonym identity embeddings.
            Hardened with a convergence check across K = 30/60/120/240.
  CONTINUOUS joint-identity-subspace projection: remove the top-k principal
            directions of the {species, place, ethnonym} bag embeddings from
            the full-myth embedding, recompute Delta, simple biome-shuffle null.

Statistic: within-iconic-taxon stratified Delta on iNaturalist; marginal Delta
on Places365 (scene imagery, no taxa).

CLI:
  python matched_null_figures.py perbiome      # both corpora, both methods
  python matched_null_figures.py convergence   # matched null per-biome, K sweep
  python matched_null_figures.py breadth       # muDelta per breadth bin (iNat)
  python matched_null_figures.py pertaxon      # per-(biome,taxon) Delta (iNat)
  python matched_null_figures.py all
Outputs: dataset/.../ladder/nolll_*.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
MAP = ROOT / "dataset/mapping_v2"
LAD = EMB / "ladder"
from motif_specificity_controls import biome_motif_membership_count

MIN_BIOME_IMGS = 50
MIN_TAXON_IMGS = 20
rng = np.random.default_rng(42)


# ---------- shared setup ----------
def load():
    man = pd.read_parquet(LAD / "manifest.parquet")
    motif_ids = man["motif_id"].astype(str).tolist()
    E = {n: np.load(LAD / f"emb_{n}.npy") for n in
         ["full", "species", "place", "ethnonym"]}
    valid = (man["valid_full"].values & man["valid_species"].values
             & man["valid_place"].values & man["valid_ethnonym"].values)
    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    in_B = np.zeros((len(motif_ids), len(biomes)), bool)
    nbreadth = np.zeros(len(motif_ids), int)
    for i, mid in enumerate(motif_ids):
        s = [b for b in mb_set.get(mid, set()) if isinstance(b, str) and b != "N/A"]
        nbreadth[i] = len(s)
        for j, b in enumerate(biomes):
            if b in s:
                in_B[i, j] = True
    return motif_ids, E, valid, biomes, in_B, nbreadth


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def joint_block(E, valid, K):
    from sklearn.cluster import KMeans
    joint = np.concatenate([l2(E["species"]), l2(E["place"]), l2(E["ethnonym"])], axis=1)
    idx = np.where(valid)[0]
    km = KMeans(n_clusters=min(K, len(idx) // 5), n_init=4, random_state=0).fit(joint[idx])
    block = np.full(len(valid), -1)
    block[idx] = km.labels_
    return block


def residualize_identity(E, valid, alpha=100.0, n_pca=200, folds=5):
    """PROPER continuous identity control. Residualise the full-myth embedding
    against each motif's own concatenated identity vector (species (+) place (+)
    ethnonym) by out-of-fold ridge regression, removing the identity-predictable
    component of every myth. Returns the residual embedding and the out-of-fold
    R^2 of full~identity (how much of the myth was identity-predictable)."""
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    idx = np.where(valid)[0]
    I = np.concatenate([l2(E["species"]), l2(E["place"]), l2(E["ethnonym"])], axis=1)[idx]
    I = I - I.mean(0, keepdims=True)
    I = PCA(min(n_pca, I.shape[0] - 1), random_state=0).fit_transform(I)
    F = E["full"][idx].astype(np.float64)
    pred = np.zeros_like(F)
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(idx):
        pred[te] = Ridge(alpha=alpha).fit(I[tr], F[tr]).predict(I[te])
    R = E["full"].astype(np.float64).copy()
    R[idx] = F - pred
    r2 = 1.0 - ((F - pred) ** 2).sum() / ((F - F.mean(0)) ** 2).sum()
    return R, float(r2)


def proj_full(E, valid, k):  # deprecated: register-axis removal, kept for reference
    idx = np.where(valid)[0]
    ID = np.vstack([E["species"][idx], E["place"][idx], E["ethnonym"][idx]])
    ID = ID - ID.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(ID, full_matrices=False)
    V = Vt[:k].T
    return E["full"] - (E["full"] @ V) @ V.T


def imgset(which):
    if which == "inat":
        ie = np.load(EMB / "inat_basic/img_emb.npy")
        m = pd.read_parquet(EMB / "inat_basic/img_paths.parquet").reset_index(drop=True)
        return ie, m["photo_biome_wwf"].fillna("").values, m["iconic_taxon"].fillna("").values
    pe = np.load(EMB / "places365_strict/img_emb.npy")
    m = pd.read_parquet(EMB / "places365_strict/img_paths.parquet").reset_index(drop=True)
    return pe, m["photo_biome_wwf"].fillna("").values, None


def score(sims, bm, taxon, stratify):
    if not stratify or taxon is None:
        return sims[bm].mean(0)
    cols = []
    for t in [x for x in np.unique(taxon[bm]) if x and x != "N/A"]:
        mm = bm & (taxon == t)
        if mm.sum() >= MIN_TAXON_IMGS:
            cols.append(sims[mm].mean(0))
    return None if not cols else np.mean(np.vstack(cols), axis=0)


# ---------- per-biome ----------
def perbiome(full_emb, ie, ib, taxon, in_B, biomes, valid, stratify,
             null, block=None, perm=1000, seed0=1000):
    femb = full_emb / (np.linalg.norm(full_emb, axis=1, keepdims=True) + 1e-12)
    sims = (ie @ femb.T).astype(np.float32)
    sims = sims - sims.mean(0, keepdims=True)
    idx = np.where(valid)[0]
    blk = block[idx] if block is not None else None
    rows = []
    for j, b in enumerate(biomes):
        bm = ib == b
        if bm.sum() < MIN_BIOME_IMGS:
            continue
        per = score(sims, bm, taxon, stratify)
        if per is None:
            continue
        lab = in_B[idx, j]
        if lab.sum() < 5 or (~lab).sum() < 5:
            continue
        d = float(per[idx[lab]].mean() - per[idx[~lab]].mean())
        gen = np.random.default_rng(seed0 + j)          # independent per test
        nd = np.empty(perm)
        for kk in range(perm):
            if null == "matched":
                sh = lab.copy()
                for bl in np.unique(blk):
                    mm = blk == bl
                    sh[mm] = gen.permutation(lab[mm])
            else:
                sh = gen.permutation(lab)
            nd[kk] = per[idx[sh]].mean() - per[idx[~sh]].mean()
        p = (1 + int((nd >= d).sum())) / (1 + perm)      # +1 smoothed, never 0
        rows.append({"biome": b, "delta_obs": d, "null_mean": float(nd.mean()),
                     "p": p, "n_in": int(lab.sum())})
    return pd.DataFrame(rows)


# ---------- main tasks ----------
def task_perbiome():
    motif_ids, E, valid, biomes, in_B, nbreadth = load()
    blk = joint_block(E, valid, 60)
    fp = proj_full(E, valid, 10)
    for corpus, strat in [("inat", True), ("p365", False)]:
        ie, ib, tx = imgset(corpus)
        dm = perbiome(E["full"], ie, ib, tx, in_B, biomes, valid, strat, "matched", blk)
        dpj = perbiome(fp, ie, ib, tx, in_B, biomes, valid, strat, "simple")
        dm = dm.rename(columns={"delta_obs": "delta_raw", "p": "p_matched",
                                "null_mean": "null_matched"})
        dpj = dpj.rename(columns={"delta_obs": "delta_proj", "p": "p_proj"})[
            ["biome", "delta_proj", "p_proj"]]
        out = dm.merge(dpj, on="biome", how="outer")
        out.to_csv(LAD / f"nolll_perbiome_{corpus}.csv", index=False)
        print(f"[perbiome {corpus}] matched survive "
              f"{int((out['p_matched']<0.05).sum())}/{len(out)}  "
              f"proj survive {int((out['p_proj']<0.05).sum())}/{len(out)}", flush=True)


def task_convergence():
    motif_ids, E, valid, biomes, in_B, nbreadth = load()
    ie, ib, tx = imgset("inat")
    pe, pb, _ = imgset("p365")
    rows = []
    for K in [30, 60, 120, 240]:
        blk = joint_block(E, valid, K)
        di = perbiome(E["full"], ie, ib, tx, in_B, biomes, valid, True, "matched", blk, perm=500)
        dp = perbiome(E["full"], pe, pb, None, in_B, biomes, valid, False, "matched", blk, perm=500)
        rows.append({"K": K, "blocks": int(len(np.unique(blk[blk >= 0]))),
                     "inat_survive": int((di["p"] < 0.05).sum()), "inat_n": len(di),
                     "p365_survive": int((dp["p"] < 0.05).sum()), "p365_n": len(dp)})
        print(f"[convergence K={K}] iNat {rows[-1]['inat_survive']}/{rows[-1]['inat_n']}  "
              f"P365 {rows[-1]['p365_survive']}/{rows[-1]['p365_n']}", flush=True)
    pd.DataFrame(rows).to_csv(LAD / "nolll_convergence.csv", index=False)


def task_breadth():
    motif_ids, E, valid, biomes, in_B, nbreadth = load()
    fp = proj_full(E, valid, 10)
    ie, ib, tx = imgset("inat")
    # breadth bins by number of biomes a motif spans
    bins = [(1, 3, "SpecA"), (4, 9, "Semi"), (10, 99, "Universal")]
    sims_raw = (ie @ E["full"].T).astype(np.float32); sims_raw -= sims_raw.mean(0, keepdims=True)
    sims_pj = (ie @ fp.T).astype(np.float32); sims_pj -= sims_pj.mean(0, keepdims=True)
    rows = []
    for lo, hi, lab in bins:
        binmask = (nbreadth >= lo) & (nbreadth <= hi) & valid
        for tag, sims in [("raw", sims_raw), ("proj", sims_pj)]:
            ds = []
            for j, b in enumerate(biomes):
                bm = ib == b
                if bm.sum() < MIN_BIOME_IMGS:
                    continue
                per = score(sims, bm, tx, True)
                if per is None:
                    continue
                inb = in_B[:, j] & binmask
                outb = (~in_B[:, j]) & valid
                if inb.sum() < 5:
                    continue
                ds.append(per[inb].mean() - per[outb].mean())
            ds = np.array(ds)
            boot = [np.mean(rng.choice(ds, len(ds), replace=True)) for _ in range(500)]
            rows.append({"bin": lab, "lo": lo, "method": tag, "n_motifs": int(binmask.sum()),
                         "mu": ds.mean() * 1000, "ci_lo": np.percentile(boot, 2.5) * 1000,
                         "ci_hi": np.percentile(boot, 97.5) * 1000})
        print(f"[breadth {lab}] raw mu={rows[-2]['mu']:+.3f}  proj mu={rows[-1]['mu']:+.3f}", flush=True)
    pd.DataFrame(rows).to_csv(LAD / "nolll_breadth.csv", index=False)


def task_pertaxon():
    motif_ids, E, valid, biomes, in_B, nbreadth = load()
    blk = joint_block(E, valid, 60)
    fp = proj_full(E, valid, 10)
    ie, ib, tx = imgset("inat")
    taxa = sorted([t for t in np.unique(tx) if t and t != "N/A"])
    idx = np.where(valid)[0]
    for tag, femb, null in [("raw", E["full"], "matched"), ("proj", fp, "simple")]:
        sims = (ie @ femb.T).astype(np.float32); sims -= sims.mean(0, keepdims=True)
        rows = []
        for j, b in enumerate(biomes):
            lab = in_B[idx, j]
            if lab.sum() < 5 or (~lab).sum() < 5:
                continue
            for t in taxa:
                cm = (ib == b) & (tx == t)
                if cm.sum() < MIN_TAXON_IMGS:
                    continue
                per = sims[cm].mean(0)
                d = float(per[idx[lab]].mean() - per[idx[~lab]].mean())
                nd = np.empty(400)
                for kk in range(400):
                    if null == "matched":
                        sh = lab.copy()
                        for bl in np.unique(blk[idx]):
                            mm = blk[idx] == bl
                            sh[mm] = rng.permutation(lab[mm])
                    else:
                        sh = rng.permutation(lab)
                    nd[kk] = per[idx[sh]].mean() - per[idx[~sh]].mean()
                rows.append({"biome": b, "taxon": t, "delta": d * 1000,
                             "p": float((nd >= d).mean()), "n_img": int(cm.sum())})
        pd.DataFrame(rows).to_csv(LAD / f"nolll_pertaxon_{tag}.csv", index=False)
        df = pd.DataFrame(rows)
        print(f"[pertaxon {tag}] cells {len(df)}  sig {int((df['p']<0.05).sum())}", flush=True)


def task_crosstab():
    """The must-fix 2x2: {raw, identity-residualised} embedding x {simple,
    matched} null, both corpora, with +1-smoothed p and BH-FDR q. Separates the
    null axis (permissiveness) from the embedding axis (identity removal)."""
    motif_ids, E, valid, biomes, in_B, nbreadth = load()
    blk = joint_block(E, valid, 60)
    R, r2 = residualize_identity(E, valid)
    print(f"identity residualisation: out-of-fold R^2(full~identity) = {r2:.3f} "
          f"(fraction of myth embedding predictable from identity)", flush=True)
    rows = []
    for corpus, strat in [("inat", True), ("p365", False)]:
        ie, ib, tx = imgset(corpus)
        for etag, femb, s0 in [("raw", E["full"], 1000), ("resid", R, 5000)]:
            for null in ["simple", "matched"]:
                d = perbiome(femb, ie, ib, tx, in_B, biomes, valid, strat, null,
                             blk, perm=1000, seed0=s0 + (0 if null == "simple" else 200))
                q = bh(d["p"].values)
                rows.append({"corpus": corpus, "emb": etag, "null": null,
                             "muD": d["delta_obs"].mean() * 1000,
                             "survive_p": int((d["p"] < 0.05).sum()),
                             "survive_q": int((q < 0.05).sum()), "n": len(d)})
                print(f"  [{corpus:4s} {etag:5s}+{null:7s}] "
                      f"muD={rows[-1]['muD']:+.3f}  p<.05 {rows[-1]['survive_p']}/{len(d)}  "
                      f"q<.05 {rows[-1]['survive_q']}/{len(d)}", flush=True)
    pd.DataFrame(rows).to_csv(LAD / "nolll_crosstab.csv", index=False)


def task_alphasweep():
    """Continuous-control robustness: vary the ridge strength and report how much
    identity is removed and how many biomes survive (resid + matched, iNat)."""
    motif_ids, E, valid, biomes, in_B, nbreadth = load()
    blk = joint_block(E, valid, 60)
    ie, ib, tx = imgset("inat")
    rows = []
    for a in [10.0, 100.0, 1000.0, 10000.0]:
        R, r2 = residualize_identity(E, valid, alpha=a)
        d = perbiome(R, ie, ib, tx, in_B, biomes, valid, True, "matched", blk, perm=500, seed0=7000)
        ds = perbiome(R, ie, ib, tx, in_B, biomes, valid, True, "simple", blk, perm=500, seed0=8000)
        rows.append({"alpha": a, "r2_identity": r2,
                     "resid_matched": int((d["p"] < 0.05).sum()),
                     "resid_simple": int((ds["p"] < 0.05).sum()), "n": len(d)})
        print(f"[alpha={a:7.0f}] R^2(full~id)={r2:.3f}  resid+matched "
              f"{rows[-1]['resid_matched']}/{len(d)}  resid+simple "
              f"{rows[-1]['resid_simple']}/{len(d)}", flush=True)
    pd.DataFrame(rows).to_csv(LAD / "nolll_alphasweep.csv", index=False)


def bh(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def task_projsweep():
    """Projection control across k = 5/10/20/40 (sensitivity of the continuous
    method to how much of the identity subspace is removed)."""
    motif_ids, E, valid, biomes, in_B, nbreadth = load()
    ie, ib, tx = imgset("inat")
    pe, pb, _ = imgset("p365")
    rows = []
    for k in [5, 10, 20, 40]:
        fp = proj_full(E, valid, k)
        di = perbiome(fp, ie, ib, tx, in_B, biomes, valid, True, "simple")
        dp = perbiome(fp, pe, pb, None, in_B, biomes, valid, False, "simple")
        rows.append({"k": k, "inat_mu": di["delta_obs"].mean() * 1000,
                     "inat_survive": int((di["p"] < 0.05).sum()), "inat_n": len(di),
                     "p365_survive": int((dp["p"] < 0.05).sum()), "p365_n": len(dp)})
        print(f"[projsweep k={k}] iNat muD={rows[-1]['inat_mu']:+.3f} "
              f"{rows[-1]['inat_survive']}/{rows[-1]['inat_n']}  "
              f"P365 {rows[-1]['p365_survive']}/{rows[-1]['p365_n']}", flush=True)
    pd.DataFrame(rows).to_csv(LAD / "nolll_projsweep.csv", index=False)


def task_fdr():
    """Add Benjamini-Hochberg q-values to the per-biome and per-taxon tables."""
    for corp in ["inat", "p365"]:
        d = pd.read_csv(LAD / f"nolll_perbiome_{corp}.csv")
        d["q_matched"] = bh(d["p_matched"].values)
        d["q_proj"] = bh(d["p_proj"].values)
        d.to_csv(LAD / f"nolll_perbiome_{corp}.csv", index=False)
        print(f"[fdr {corp}] matched q<.05 {int((d.q_matched<0.05).sum())}/{len(d)}  "
              f"proj q<.05 {int((d.q_proj<0.05).sum())}/{len(d)}", flush=True)
    for tag in ["raw", "proj"]:
        d = pd.read_csv(LAD / f"nolll_pertaxon_{tag}.csv")
        d["q"] = bh(d["p"].values)
        d.to_csv(LAD / f"nolll_pertaxon_{tag}.csv", index=False)
        print(f"[fdr pertaxon {tag}] q<.05 {int((d.q<0.05).sum())}/{len(d)}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tasks = {"perbiome": task_perbiome, "convergence": task_convergence,
             "breadth": task_breadth, "pertaxon": task_pertaxon,
             "projsweep": task_projsweep, "fdr": task_fdr,
             "crosstab": task_crosstab, "alphasweep": task_alphasweep}
    if which == "all":
        for f in tasks.values():
            f()
    else:
        tasks[which]()
