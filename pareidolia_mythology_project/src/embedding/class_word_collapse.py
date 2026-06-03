"""Maximal class-word collapse on LLM-clean English motif text.

Replaces every animal-kingdom class word with "animal" and every plant-
kingdom class word with "plant", then re-embeds sentence-pooled SigLIP-2
and recomputes the per-(biome × iconic-taxon) Δ matrix.

The point: after this collapse the per-taxon matrix can no longer be
driven by class-word frequency × image-class co-occurrence. Class-word
frequency is now constant across all biomes (every animal is just
"animal", every plant just "plant"). If the per-(biome, taxon) cells
remain ecologically structured (mammals carry boreal, plants carry
Mediterranean, reptiles carry tropical), the residue is what biome-b's
actual fauna and flora LOOK LIKE in photographs aligning with biome-b's
motif text, not what class word the motif text contains.

Outputs:
  motif_emb_llm_pass2_collapsed_sentpooled.npy
  motif_meta_llm_pass2_collapsed_sentpooled.parquet
  v3_byTaxon_sentpool_collapsed_iNat.csv
"""

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

# All animal-kingdom class words → "animal"
ANIMAL_WORDS = ["mammal", "bird", "fish", "reptile", "amphibian",
                "insect", "mollusc", "crustacean", "arachnid"]
# All plant-kingdom class words + fungi → "plant"
PLANT_WORDS = ["tree", "flower", "plant", "fungus"]
# Whole-word, case-insensitive, optional plural -s
ANIMAL_RE = re.compile(
    r"\b(" + "|".join(ANIMAL_WORDS) + r")s?\b", re.IGNORECASE)
PLANT_RE = re.compile(
    r"\b(" + "|".join(PLANT_WORDS) + r")s?\b", re.IGNORECASE)


def collapse(text):
    if not isinstance(text, str): return ""
    out = ANIMAL_RE.sub("animal", text)
    out = PLANT_RE.sub("plant", out)
    return out


def split_sents(text):
    if not text: return []
    pieces = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [p.strip() for p in pieces if len(p.strip().split()) >= 3]


def embed_sentpool():
    print("loading SigLIP-2 ...", flush=True)
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device = {device}", flush=True)
    tok = AutoTokenizer.from_pretrained("google/siglip2-large-patch16-384")
    model = AutoModel.from_pretrained(
        "google/siglip2-large-patch16-384").to(device).eval()

    p2 = pd.read_csv(ANA / "llm_rewrite_specA_gemini_pass2.csv")
    p2 = p2[p2["status"] == "OK"].copy()
    p2["motif_id"] = p2["motif_id"].astype(str)
    p2["text"] = (p2["refined_oneliner_en"].fillna("").astype(str)
                   + " " + p2["refined_translated_abstract_en"]
                   .fillna("").astype(str))
    # Apply the maximal class-word collapse
    p2["text_collapsed"] = p2["text"].apply(collapse)
    n_repl_anim = p2["text"].apply(
        lambda t: len(ANIMAL_RE.findall(t) or [])).sum()
    n_repl_plant = p2["text"].apply(
        lambda t: len(PLANT_RE.findall(t) or [])).sum()
    print(f"  collapsed {n_repl_anim:,} animal-class tokens and "
          f"{n_repl_plant:,} plant-class tokens", flush=True)
    print(f"  motifs: {len(p2)}", flush=True)

    # Sentence-split + embed
    motif_sentences = [split_sents(t)[:50] for t in p2["text_collapsed"]]
    n_sent_per = np.array([len(s) for s in motif_sentences])
    print(f"  total sentences to embed: {int(n_sent_per.sum())}",
          flush=True)

    all_sents, slices = [], []
    cur = 0
    for sents in motif_sentences:
        slices.append(slice(cur, cur + len(sents)))
        cur += len(sents)
        all_sents.extend(sents)

    embs = []
    bs = 64
    for i in range(0, len(all_sents), bs):
        batch = all_sents[i:i+bs]
        with torch.no_grad():
            inp = tok(batch, padding="max_length", truncation=True,
                       max_length=64, return_tensors="pt").to(device)
            f = model.get_text_features(**inp)
            if hasattr(f, "pooler_output"):
                f = f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        embs.append(f.cpu().numpy().astype(np.float32))
        if (i // bs) % 50 == 0:
            print(f"    {i+len(batch)}/{len(all_sents)}", flush=True)
    sent_emb = np.vstack(embs)

    n_motifs = len(motif_sentences); dim = sent_emb.shape[1]
    motif_emb = np.zeros((n_motifs, dim), dtype=np.float32)
    valid = np.zeros(n_motifs, dtype=bool)
    for i, sl in enumerate(slices):
        if sl.stop > sl.start:
            v = sent_emb[sl].mean(axis=0)
            n = np.linalg.norm(v) + 1e-12
            motif_emb[i] = v / n
            valid[i] = True
    print(f"  motifs with valid embedding: {int(valid.sum())}", flush=True)

    np.save(EMB / "motif_emb_llm_pass2_collapsed_sentpooled.npy",
             motif_emb)
    meta = p2[["motif_id"]].copy()
    meta["n_sentences"] = n_sent_per
    meta["valid"] = valid
    meta.to_parquet(
        EMB / "motif_meta_llm_pass2_collapsed_sentpooled.parquet",
        index=False)
    return motif_emb[valid], meta[valid].reset_index(drop=True)


def per_taxon_test(motif_emb, motif_ids):
    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet"
                                ).reset_index(drop=True)

    mb_set, _ = biome_motif_membership_count()
    biomes = sorted({b for s in mb_set.values() for b in s
                     if isinstance(b, str) and b != "N/A"})
    in_B = np.zeros((len(motif_ids), len(biomes)), dtype=bool)
    for i, mid in enumerate(motif_ids):
        for j, b in enumerate(biomes):
            if b in mb_set.get(mid, set()):
                in_B[i, j] = True

    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)
    img_biome = img_meta["photo_biome_wwf"].fillna("").values
    img_taxon = img_meta["iconic_taxon"].fillna("").values

    N_PERMS = 1000
    MIN_IMGS = 10
    TAXON_ORDER = ["all", "Plantae", "Fungi", "Animalia", "Mammalia",
                    "Aves", "Reptilia", "Amphibia", "Actinopterygii",
                    "Insecta", "Arachnida", "Mollusca"]

    rng = np.random.default_rng(42)
    rows = []
    print(f"running per-taxon test on {len(biomes)} biomes × "
          f"{len(TAXON_ORDER)} taxa ...", flush=True)
    for j, b in enumerate(biomes):
        in_b = in_B[:, j]
        if in_b.sum() < 5 or (~in_b).sum() < 5: continue
        for t in TAXON_ORDER:
            if t == "all":
                mask = img_biome == b
            else:
                mask = (img_biome == b) & (img_taxon == t)
            if mask.sum() < MIN_IMGS: continue
            per = sims[mask].mean(axis=0)
            d = float(per[in_b].mean() - per[~in_b].mean())
            null = np.empty(N_PERMS)
            for k in range(N_PERMS):
                shuf = rng.permutation(in_b)
                null[k] = per[shuf].mean() - per[~shuf].mean()
            p = float((null >= d).mean())
            rows.append({"biome": b, "taxon_group": t,
                          "delta": d, "p_one_sided": p,
                          "n_imgs": int(mask.sum()),
                          "n_mot": int(in_b.sum())})
        print(f"  done biome {b[:40]}", flush=True)
    df = pd.DataFrame(rows)
    out = EMB / "v3_byTaxon_sentpool_collapsed_iNat.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} cells)", flush=True)


def main():
    motif_emb, meta = embed_sentpool()
    motif_ids = meta["motif_id"].astype(str).tolist()
    per_taxon_test(motif_emb, motif_ids)


if __name__ == "__main__":
    main()
