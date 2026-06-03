"""SigLIP-2 sentence-pooled motif embeddings.

SigLIP-2 has a 64-token context cap, which truncates long Berezkin
abstracts to ~50 words. To capture the full motif content while keeping
SigLIP-2 as the embedder:

  1. Split each LLM-clean abstract into sentences (NLTK PunktSentenceTokenizer
     with English defaults; falls back to regex on "[.!?]" if NLTK missing).
  2. Embed each sentence with SigLIP-2 (each fits in 64 tokens).
  3. Mean-pool the sentence embeddings into one motif-level vector,
     L2-normalised.

Outputs:
  motif_emb_llm_pass2_abstract_sentpooled.npy
  motif_meta_llm_pass2_sentpooled.parquet
  biome_test_llm_pass2_abstract_sentpooled_resid.csv
  v2_sentpooled_biome_test_summary.txt

Reports also the per-motif sentence count distribution and the total
number of unique sentences embedded (the actual compute budget).
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]  # project root
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
ANA = ROOT / "dataset/analysis"

from motif_specificity_controls import biome_motif_membership_count


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    # Try NLTK first
    try:
        import nltk
        try:
            from nltk.tokenize import sent_tokenize
            return [s.strip() for s in sent_tokenize(text)
                    if s.strip()]
        except LookupError:
            nltk.download("punkt", quiet=True)
            from nltk.tokenize import sent_tokenize
            return [s.strip() for s in sent_tokenize(text)
                    if s.strip()]
    except Exception:
        # Fallback: regex on terminal punctuation
        pieces = re.split(r"(?<=[\.\!\?])\s+", text.strip())
        return [p.strip() for p in pieces if p.strip()]


def main():
    print("loading SigLIP-2 ...", flush=True)
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("google/siglip2-large-patch16-384")
    model = AutoModel.from_pretrained(
        "google/siglip2-large-patch16-384").to(device).eval()

    # Load LLM-clean abstracts (pass2 OK rows)
    p2 = pd.read_csv(ANA / "llm_rewrite_specA_gemini_pass2.csv")
    p2 = p2[p2["status"] == "OK"].copy()
    p2["motif_id"] = p2["motif_id"].astype(str)
    p2["text"] = (p2["refined_oneliner_en"].fillna("").astype(str)
                   + " " + p2["refined_translated_abstract_en"]
                   .fillna("").astype(str))
    print(f"motifs: {len(p2)}", flush=True)

    # Sentence-split each motif
    motif_sentences: list[list[str]] = []
    for t in p2["text"]:
        sents = split_sentences(t)
        # Drop sentences that are extremely short (< 3 words) — they
        # carry little biome content and just add noise to the mean
        sents = [s for s in sents if len(s.split()) >= 3]
        # Cap at 50 sentences per motif to bound compute on outliers
        motif_sentences.append(sents[:50])
    n_sent_per_motif = np.array([len(s) for s in motif_sentences])
    print(f"sentence count per motif: "
          f"mean={n_sent_per_motif.mean():.1f}, "
          f"median={int(np.median(n_sent_per_motif))}, "
          f"max={int(n_sent_per_motif.max())}, "
          f"total={int(n_sent_per_motif.sum())}", flush=True)

    # Flatten all sentences into one batch, embed, then re-group
    all_sentences: list[str] = []
    motif_slices: list[slice] = []
    cur = 0
    for sents in motif_sentences:
        motif_slices.append(slice(cur, cur + len(sents)))
        cur += len(sents)
        all_sentences.extend(sents)
    print(f"total unique sentences to embed: {len(all_sentences)}",
          flush=True)

    bs = 64
    embs = []
    for i in range(0, len(all_sentences), bs):
        batch = all_sentences[i:i+bs]
        with torch.no_grad():
            inputs = tok(batch, padding="max_length", truncation=True,
                          max_length=64, return_tensors="pt").to(device)
            f = model.get_text_features(**inputs)
            if hasattr(f, "pooler_output"):
                f = f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        embs.append(f.cpu().numpy().astype(np.float32))
        if (i // bs) % 50 == 0:
            print(f"  {i+len(batch)}/{len(all_sentences)}", flush=True)
    sent_emb = np.vstack(embs)
    print(f"sent_emb shape: {sent_emb.shape}", flush=True)

    # Mean-pool per motif, then re-normalise. Motifs with zero
    # sentences (very short or empty after the filter) get a zero vector
    # marker that we will exclude from the biome test.
    n_motifs = len(motif_sentences)
    dim = sent_emb.shape[1]
    motif_emb = np.zeros((n_motifs, dim), dtype=np.float32)
    valid = np.zeros(n_motifs, dtype=bool)
    for i, sl in enumerate(motif_slices):
        if sl.stop > sl.start:
            v = sent_emb[sl].mean(axis=0)
            n = np.linalg.norm(v) + 1e-12
            motif_emb[i] = v / n
            valid[i] = True
    print(f"motifs with at least one valid sentence: {valid.sum()}",
          flush=True)

    np.save(EMB / "motif_emb_llm_pass2_abstract_sentpooled.npy", motif_emb)
    meta = p2[["motif_id"]].copy()
    meta["n_sentences"] = n_sent_per_motif
    meta["valid"] = valid
    meta.to_parquet(
        EMB / "motif_meta_llm_pass2_sentpooled.parquet", index=False)

    # ============ Biome test ============
    print("\nrunning residualised biome test ...", flush=True)
    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet")
    img_meta = img_meta.reset_index(drop=True)
    img_biome = img_meta["photo_biome_wwf"].fillna("").values

    # Restrict to valid motifs
    me = motif_emb[valid]
    mm = meta[valid].reset_index(drop=True)
    motif_ids = mm["motif_id"].astype(str).tolist()

    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    in_B = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
    for i, mid in enumerate(motif_ids):
        a = mb_set.get(mid, set())
        for j, b in enumerate(biomes):
            if b in a:
                in_B[i, j] = True

    sims = img_emb @ me.T
    sims = sims - sims.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(42)
    rows = []
    for j, b in enumerate(biomes):
        b_imgs = img_biome == b
        if b_imgs.sum() < 5:
            continue
        in_b = in_B[:, j]
        if in_b.sum() < 5 or (~in_b).sum() < 5:
            continue
        per = sims[b_imgs].mean(axis=0)
        delta = float(per[in_b].mean() - per[~in_b].mean())
        null = np.empty(1000)
        for k in range(1000):
            shuf = rng.permutation(in_b)
            null[k] = per[shuf].mean() - per[~shuf].mean()
        p = float((null >= delta).mean())
        rows.append({"biome": b, "n_imgs": int(b_imgs.sum()),
                     "n_motifs_in_biome": int(in_b.sum()),
                     "delta": delta, "p_one_sided": p})
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    df.to_csv(EMB / "biome_test_llm_pass2_abstract_sentpooled_resid.csv",
              index=False)
    print(df.to_string(index=False))
    sig = int((df["p_one_sided"] < 0.05).sum())
    summary = (
        f"SigLIP-2 sentence-pooled embedding biome test\n"
        f"  motifs embedded: {valid.sum()}\n"
        f"  total sentences embedded: {len(all_sentences)}\n"
        f"  mean sentences per motif: {n_sent_per_motif.mean():.1f}\n"
        f"  μΔ marginal: {df['delta'].mean()*1000:+.3f} ×10⁻³\n"
        f"  significant biomes (p<.05): {sig}/{len(df)}\n"
    )
    (EMB / "v2_sentpooled_biome_test_summary.txt").write_text(
        summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
