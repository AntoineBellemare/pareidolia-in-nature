"""Proper bag-of-words test on sentence-pooled SigLIP: shuffle all words
in each motif, re-chunk into same number of pieces with same lengths,
embed each chunk with SigLIP, mean-pool, run biome test.

Compare to original sentence-pool (+0.244 marginal). If shuffled ~ original,
bag-of-words is confirmed at full content; if shuffled drops, structural
content matters."""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import random, re
import numpy as np
import pandas as pd
from pathlib import Path
import torch

EMB = Path("dataset/imagery/embeddings/siglip2-large")
ANA = Path("dataset/analysis")

from motif_specificity_controls import biome_motif_membership_count
from transformers import AutoTokenizer, AutoModel


def split_sents(text):
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[\.!\?])\s+", text.strip())
            if len(s.strip().split()) >= 3]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading SigLIP-2 on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained("google/siglip2-large-patch16-384")
    model = AutoModel.from_pretrained(
        "google/siglip2-large-patch16-384").to(device).eval()

    p2 = pd.read_csv(ANA / "llm_rewrite_specA_gemini_pass2.csv")
    p2 = p2[p2["status"] == "OK"].copy()
    p2["motif_id"] = p2["motif_id"].astype(str)
    p2["text"] = (p2["refined_oneliner_en"].fillna("").astype(str) + " "
                   + p2["refined_translated_abstract_en"]
                   .fillna("").astype(str))
    print(f"motifs: {len(p2)}", flush=True)

    all_chunks = []
    slices = []
    cur = 0
    for _, row in p2.iterrows():
        sents = split_sents(row["text"])[:50]
        if not sents:
            slices.append(slice(cur, cur))
            continue
        sent_lengths = [len(s.split()) for s in sents]
        all_words = " ".join(sents).split()
        rng = random.Random(hash(row["motif_id"]) & 0xffffffff)
        rng.shuffle(all_words)
        chunks = []
        p = 0
        for L in sent_lengths:
            ch = " ".join(all_words[p:p+L])
            if ch:
                chunks.append(ch)
            p += L
        slices.append(slice(cur, cur+len(chunks)))
        cur += len(chunks)
        all_chunks.extend(chunks)
    print(f"total shuffled chunks: {len(all_chunks)}", flush=True)

    embs = []
    bs = 64
    for i in range(0, len(all_chunks), bs):
        batch = all_chunks[i:i+bs]
        with torch.no_grad():
            inp = tok(batch, padding="max_length", truncation=True,
                       max_length=64, return_tensors="pt").to(device)
            f = model.get_text_features(**inp)
            if hasattr(f, "pooler_output"):
                f = f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        embs.append(f.cpu().numpy().astype(np.float32))
        if (i // bs) % 50 == 0:
            print(f"  {i+len(batch)}/{len(all_chunks)}", flush=True)
    chunk_emb = np.vstack(embs)

    n_motifs = len(p2)
    dim = chunk_emb.shape[1]
    motif_emb = np.zeros((n_motifs, dim), dtype=np.float32)
    valid = np.zeros(n_motifs, dtype=bool)
    for i, sl in enumerate(slices):
        if sl.stop > sl.start:
            v = chunk_emb[sl].mean(axis=0)
            n = np.linalg.norm(v) + 1e-12
            motif_emb[i] = v / n
            valid[i] = True
    print(f"valid motifs: {valid.sum()}", flush=True)
    np.save(EMB / "motif_emb_llm_pass2_abstract_sentpool_shuffled.npy",
             motif_emb)

    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet")
    img_meta = img_meta.reset_index(drop=True)
    img_biome = img_meta["photo_biome_wwf"].fillna("").values

    motif_emb_v = motif_emb[valid]
    motif_ids = p2.loc[valid, "motif_id"].tolist()
    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    in_B = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
    for i, mid in enumerate(motif_ids):
        for j, b in enumerate(biomes):
            if b in mb_set.get(mid, set()):
                in_B[i, j] = True
    sims = img_emb @ motif_emb_v.T
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
        d = float(per[in_b].mean() - per[~in_b].mean())
        null = np.empty(1000)
        for k in range(1000):
            shuf = rng.permutation(in_b)
            null[k] = per[shuf].mean() - per[~shuf].mean()
        p = float((null >= d).mean())
        rows.append({"biome": b, "delta": d, "p": p})
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    df.to_csv(EMB / "v2_R5a_sentpool_shuffled_biome_test.csv", index=False)
    mu = df["delta"].mean() * 1000
    sig = int((df["p"] < 0.05).sum())
    print(f"\nPROPER bag-of-words test (shuffle words, re-chunk, sent-pool)")
    print(f"  μΔ = {mu:+.3f} ×10⁻³, sig: {sig}/{len(df)}")
    print(f"\nComparison:")
    print(f"  Original sentence-pool: +0.244 (8/14 sig)")
    print(f"  Shuffled  sentence-pool: {mu:+.3f} ({sig}/{len(df)} sig)")
    print(f"\nDifference: {mu - 0.244:+.3f}")
    if abs(mu - 0.244) < 0.05:
        print("  -> bag-of-words confirmed (drop < 0.05)")
    elif mu < 0.244:
        print(f"  -> structure contributes ({0.244-mu:.3f} = {(0.244-mu)/0.244*100:.1f}% of signal)")


if __name__ == "__main__":
    main()
