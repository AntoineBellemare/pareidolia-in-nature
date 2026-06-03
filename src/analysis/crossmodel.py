"""Compute within-iconic-taxon stratified Δ for M-CLIP, OpenCLIP-LAION-2B,
OpenCLIP-OpenAI on LLM-clean abstracts. Produces apples-to-apples
4-model comparison reporting BOTH marginal and stratified Δ.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
EMB = ROOT / "dataset/imagery/embeddings"

from motif_specificity_controls import biome_motif_membership_count

N_PERMS = 1000
MIN_TAXON_IMGS = 20
MIN_BIOME_IMGS = 50


def compute_strat(motif_emb_path, motif_meta_path, img_emb_path,
                    img_meta_path):
    motif_emb = np.load(motif_emb_path)
    motif_meta = pd.read_parquet(motif_meta_path)
    img_emb = np.load(img_emb_path)
    img_meta = pd.read_parquet(img_meta_path).reset_index(drop=True)

    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)

    mb_set, _ = biome_motif_membership_count()
    motif_ids = motif_meta["motif_id"].astype(str).tolist()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    in_B = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
    for i, mid in enumerate(motif_ids):
        for j, b in enumerate(biomes):
            if b in mb_set.get(mid, set()):
                in_B[i, j] = True

    img_biome = img_meta["photo_biome_wwf"].fillna("").values
    img_taxon = img_meta["iconic_taxon"].fillna("").values
    strata = sorted([t for t in pd.Series(img_taxon).unique()
                     if t and t != "N/A"])

    rng = np.random.default_rng(42)
    rows = []
    for j, b in enumerate(biomes):
        b_imgs = img_biome == b
        if b_imgs.sum() < MIN_BIOME_IMGS:
            continue
        in_b = in_B[:, j]
        if in_b.sum() < 5 or (~in_b).sum() < 5:
            continue

        # Marginal
        per = sims[b_imgs].mean(axis=0)
        d_marg = float(per[in_b].mean() - per[~in_b].mean())
        null_marg = np.empty(N_PERMS)
        for k in range(N_PERMS):
            shuf = rng.permutation(in_b)
            null_marg[k] = per[shuf].mean() - per[~shuf].mean()
        p_marg = float((null_marg >= d_marg).mean())

        # Stratified (uniform across iconic taxa)
        d_strata = []
        for t in strata:
            t_mask = b_imgs & (img_taxon == t)
            if t_mask.sum() < MIN_TAXON_IMGS:
                continue
            per_t = sims[t_mask].mean(axis=0)
            d_strata.append(
                per_t[in_b].mean() - per_t[~in_b].mean())
        if len(d_strata) < 2:
            d_strat = np.nan
            p_strat = np.nan
        else:
            d_strat = float(np.mean(d_strata))
            null_strat = np.empty(N_PERMS)
            for k in range(N_PERMS):
                shuf = rng.permutation(in_b)
                ds = []
                for t in strata:
                    t_mask = b_imgs & (img_taxon == t)
                    if t_mask.sum() < MIN_TAXON_IMGS:
                        continue
                    per_t = sims[t_mask].mean(axis=0)
                    ds.append(per_t[shuf].mean() - per_t[~shuf].mean())
                null_strat[k] = float(np.mean(ds))
            p_strat = float((null_strat >= d_strat).mean())

        rows.append({
            "biome": b,
            "n_imgs": int(b_imgs.sum()),
            "n_motifs_in_biome": int(in_b.sum()),
            "delta_marg": d_marg,
            "p_marg": p_marg,
            "delta_strat": d_strat,
            "p_strat": p_strat,
        })
    return pd.DataFrame(rows).sort_values("delta_marg", ascending=False)


def main():
    models = [
        ("mclip", EMB / "mclip/motif_emb_llm_clean_pass2.npy",
                  EMB / "mclip/motif_meta_llm_clean_pass2.parquet",
                  EMB / "mclip/img_emb.npy",
                  EMB / "mclip/img_paths.parquet"),
        ("openclip_laion2b",
                  EMB / "openclip_laion2b/motif_emb_llm_clean_pass2.npy",
                  EMB / "openclip_laion2b/motif_meta_llm_clean_pass2.parquet",
                  EMB / "openclip_laion2b/img_emb.npy",
                  EMB / "openclip_laion2b/img_paths.parquet"),
        ("openclip_openai",
                  EMB / "openclip_openai/motif_emb_llm_clean_pass2.npy",
                  EMB / "openclip_openai/motif_meta_llm_clean_pass2.parquet",
                  EMB / "openclip_openai/img_emb.npy",
                  EMB / "openclip_openai/img_paths.parquet"),
    ]
    summary = []
    for name, me, mm, ie, im in models:
        print(f"\n=== {name} ===", flush=True)
        df = compute_strat(me, mm, ie, im)
        out = EMB / f"{name}/biome_test_llm_clean_stratified.csv"
        df.to_csv(out, index=False)
        mμ_m = df["delta_marg"].mean() * 1000
        mμ_s = df["delta_strat"].mean() * 1000
        n_sig_m = int((df["p_marg"] < 0.05).sum())
        n_sig_s = int((df["p_strat"] < 0.05).sum())
        summary.append({
            "model": name,
            "mu_delta_marginal": mμ_m,
            "mu_delta_stratified": mμ_s,
            "n_sig_marginal": n_sig_m,
            "n_sig_stratified": n_sig_s,
        })
        print(f"  μΔ marginal   = {mμ_m:+.3f} ×10⁻³, sig {n_sig_m}/{len(df)}")
        print(f"  μΔ stratified = {mμ_s:+.3f} ×10⁻³, sig {n_sig_s}/{len(df)}")
        print(f"  wrote {out}", flush=True)
    print("\n=== SUMMARY ===")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
