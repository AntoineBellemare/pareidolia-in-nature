"""Compute stratified-Δ versions of the v3 robustness baselines:
  (a) word-shuffled sentence-pooled motifs
  (b) encyclopedic null (synthetic Wikipedia-style paragraphs, truncated SigLIP)

Both apply the within-iconic-taxon stratified residualised test on the
iNat image corpus, using the same residualised_test function as the v3
headline pipeline. Outputs:
  v2_R5a_sentpool_shuffled_biome_test_stratified.csv
  v2_R5b_wiki_biome_test_stratified.csv
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"

from motif_specificity_controls import biome_motif_membership_count
from recompute_all import residualised_test


def main():
    # Image side (shared)
    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet"
                                ).reset_index(drop=True)

    # ---- (a) Word-shuffled sentence-pooled motifs ----
    print("=== shuffled sentpool, stratified ===", flush=True)
    shuf_emb = np.load(EMB / "motif_emb_llm_pass2_abstract_sentpool_shuffled.npy")
    # The shuffled embeddings were computed from the same pass2-OK motif rows
    # as the headline sentpool, in the same order. Use the headline meta.
    meta = pd.read_parquet(EMB / "motif_meta_llm_pass2_sentpooled.parquet")
    valid = meta["valid"].values
    shuf_emb_v = shuf_emb[valid]
    meta_v = meta[valid].reset_index(drop=True)
    motif_ids = meta_v["motif_id"].astype(str).tolist()
    df_s = residualised_test(shuf_emb_v, motif_ids, img_emb, img_meta,
                              stratify_taxon=True)
    out_s = EMB / "v2_R5a_sentpool_shuffled_biome_test_stratified.csv"
    df_s.to_csv(out_s, index=False)
    print(f"  μΔ marg = {df_s['delta'].mean()*1000:+.3f}, "
          f"μΔ strat = {df_s['delta_strat'].mean()*1000:+.3f}", flush=True)
    print(f"  wrote {out_s}", flush=True)

    # ---- (b) Encyclopedic null ----
    print("\n=== encyclopedic null, stratified ===", flush=True)
    wiki_emb = np.load(EMB / "v2_R5b_wiki_motif_emb.npy")
    wiki_meta = pd.read_parquet(EMB / "v2_R5b_wiki_motif_meta.parquet"
                                  ).reset_index(drop=True)
    wiki_ids = wiki_meta["motif_id"].astype(str).tolist()
    df_w = residualised_test(wiki_emb, wiki_ids, img_emb, img_meta,
                              stratify_taxon=True)
    out_w = EMB / "v2_R5b_wiki_biome_test_stratified.csv"
    df_w.to_csv(out_w, index=False)
    print(f"  μΔ marg = {df_w['delta'].mean()*1000:+.3f}, "
          f"μΔ strat = {df_w['delta_strat'].mean()*1000:+.3f}", flush=True)
    print(f"  wrote {out_w}", flush=True)


if __name__ == "__main__":
    main()
