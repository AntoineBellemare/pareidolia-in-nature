"""
inat_basic_filter.py — apply a basic SigLIP-based filter to iNat photos.

UNLIKE YFCC/Places365 strict (which selects for *wide landscapes*),
iNat is intentionally species-focused. This filter only drops the obvious
non-nature junk: humans holding things, indoor scenes, urban shots,
vehicles, low-quality / black-frame / metadata-only images.

Positives we want to KEEP (species in habitat):
  - close-up animal/plant/fungus photos in natural context
  - habitat shots
  - any natural outdoor scene

Negatives we want to DROP:
  - person in frame (angler, hiker, indoor portrait)
  - indoor scene (kitchen, lab, room)
  - city / car / vehicle / road sign
  - black or test-card images

Strategy: per-image neg_score = max(similarity to negative prompts).
Drop if neg_score > NEG_THRESHOLD (lenient — keeps most iNat photos).

Outputs:
  embeddings/siglip2-large/inat_basic/img_emb.npy
  embeddings/siglip2-large/inat_basic/img_paths.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]  # project root
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
OUT = EMB / "inat_basic"
OUT.mkdir(parents=True, exist_ok=True)

NEGATIVE_PROMPTS = [
    "a person in the photo",
    "a portrait of a human",
    "people holding something",
    "an angler holding fish",
    "an indoor scene",
    "a kitchen or laboratory",
    "a city skyline or street",
    "a car or vehicle",
    "a road sign",
    "a black or all-white image",
    "a museum specimen on a white background",
    "a screenshot or document",
]

NEG_THRESHOLD = 0.08  # lenient — drops only obvious cases


def embed_prompts(prompts):
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("google/siglip2-large-patch16-384")
    model = AutoModel.from_pretrained(
        "google/siglip2-large-patch16-384").to(device).eval()
    inputs = tok(prompts, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=64).to(device)
    with torch.no_grad():
        f = model.get_text_features(**inputs)
        if hasattr(f, "pooler_output"): f = f.pooler_output
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy().astype(np.float32)


def main():
    print("Loading iNat embeddings + manifest …")
    img_emb = np.load(EMB / "img_emb.npy")
    img_meta = pd.read_parquet(EMB / "img_paths.parquet").reset_index(drop=True)
    print(f"  {len(img_meta):,} images")

    print("\nEmbedding negative probes …")
    neg = embed_prompts(NEGATIVE_PROMPTS)
    neg_score = (img_emb @ neg.T).max(axis=1)
    print(f"  neg_score distribution: "
          f"min={neg_score.min():.3f}, mean={neg_score.mean():.3f}, "
          f"max={neg_score.max():.3f}, median={np.median(neg_score):.3f}")

    keep_mask = neg_score <= NEG_THRESHOLD
    n_drop = int((~keep_mask).sum())
    print(f"\n  threshold {NEG_THRESHOLD}: drop {n_drop:,} of {len(img_meta):,} "
          f"({100*n_drop/len(img_meta):.1f}%)")

    # Per-biome impact
    img_meta = img_meta.copy()
    img_meta["neg_score"] = neg_score
    img_meta["kept_basic"] = keep_mask
    use_b = img_meta.get("photo_biome_wwf")
    if use_b is None: use_b = img_meta["tradition_biome_wwf"]
    by_biome = (img_meta.assign(b=use_b)
                .groupby("b")
                .agg(total=("photo_id", "size") if "photo_id" in img_meta.columns
                     else ("local_path", "size"),
                     kept=("kept_basic", "sum"),
                     median_neg=("neg_score", "median"))
                .reset_index())
    by_biome["dropped"] = by_biome["total"] - by_biome["kept"]
    by_biome["pct_dropped"] = (100 * by_biome["dropped"] / by_biome["total"]).round(1)
    print("\nPer-biome filter impact:")
    print(by_biome.to_string(index=False))

    # Save filtered subset
    img_meta_ok = img_meta[keep_mask].reset_index(drop=True)
    img_emb_ok = img_emb[keep_mask]
    np.save(OUT / "img_emb.npy", img_emb_ok)
    img_meta_ok.to_parquet(OUT / "img_paths.parquet", index=False)
    print(f"\nwrote {len(img_meta_ok):,} kept rows to {OUT}/")

    # Show 10 worst (highest neg_score) for sanity
    print("\nTop 10 worst (highest neg_score, will be dropped):")
    worst = img_meta.nlargest(10, "neg_score")[
        ["local_path", "neg_score"]].copy()
    if "iconic_taxon" in img_meta.columns:
        worst["taxon"] = img_meta.loc[worst.index, "iconic_taxon"].values
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
