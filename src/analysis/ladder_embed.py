"""Ladder embed step: embed full Russian myth + 3 separated baselines
(species / place / ethnonym), sentence-pooled SigLIP-2. Saves embeddings
+ manifest for the stats battery.

Frame: raw Russian, multilingual SigLIP-2, vs the same iNaturalist
images as the headline. Self-contained Russian-original decomposition.
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for shared utils
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
RES = ROOT / "_entity_extraction/full"
OUT = EMB / "ladder"
OUT.mkdir(exist_ok=True)


def split_sents(text):
    if not text:
        return []
    pieces = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [p.strip() for p in pieces if len(p.strip().split()) >= 1]


def sentpool_embed(tok, model, device, texts, cap=50):
    motif_sents = [split_sents(t)[:cap] for t in texts]
    all_s, slices, cur = [], [], 0
    for s in motif_sents:
        slices.append(slice(cur, cur + len(s)))
        cur += len(s)
        all_s.extend(s)
    embs = []
    for i in range(0, len(all_s), 64):
        b = all_s[i:i+64]
        with torch.no_grad():
            inp = tok(b, padding="max_length", truncation=True,
                       max_length=64, return_tensors="pt").to(device)
            f = model.get_text_features(**inp)
            if hasattr(f, "pooler_output"):
                f = f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        embs.append(f.cpu().numpy().astype(np.float32))
        if (i // 64) % 100 == 0:
            print(f"    {i+len(b)}/{len(all_s)}", flush=True)
    sent_emb = np.vstack(embs) if embs else np.zeros((0, 1024), np.float32)
    dim = sent_emb.shape[1] if sent_emb.shape[0] else 1024
    out = np.zeros((len(texts), dim), np.float32)
    valid = np.zeros(len(texts), bool)
    for i, sl in enumerate(slices):
        if sl.stop > sl.start:
            v = sent_emb[sl].mean(0)
            out[i] = v / (np.linalg.norm(v) + 1e-12)
            valid[i] = True
    return out, valid


def main():
    # Merge species extraction
    sp = {}
    for i in range(1, 55):
        for r in json.loads((RES / f"result_{i}.json").read_text(encoding="utf-8")):
            sp[str(r["motif_id"])] = r
    print(f"species results: {len(sp)} motifs", flush=True)

    # Metadata place / ethnonym + full Russian
    ab = pd.read_parquet(ROOT / "dataset/mapping_v2/motif_abstracts.parquet")
    ab["motif_id"] = ab.motif_id.astype(str)
    def uniq(s):
        return sorted(set(x.strip().rstrip(".").strip()
                          for x in s.fillna("").astype(str) if x.strip()))
    def aggru(s):
        seen=set(); o=[]
        for x in s.fillna("").astype(str):
            x=x.strip()
            if x and x not in seen: seen.add(x); o.append(x)
        return " ".join(o)
    g = ab.groupby("motif_id").agg(regions=("region", uniq),
                                    peoples=("tradition_name_ru", uniq),
                                    full_ru=("abstract_ru", aggru))

    vm = pd.read_parquet(EMB / "motif_meta_llm_pass2_sentpooled.parquet")
    motif_ids = [m for m in vm[vm["valid"]]["motif_id"].astype(str)
                 if m in sp and m in g.index]
    print(f"ladder motifs: {len(motif_ids)}", flush=True)

    full_txt, sp_txt, pl_txt, et_txt = [], [], [], []
    for m in motif_ids:
        full_txt.append(g.loc[m, "full_ru"][:8000])
        sp_txt.append(". ".join(e.get("ru", "") for e in sp[m].get("species", []) if e.get("ru")))
        pl_txt.append(". ".join(g.loc[m, "regions"]))
        et_txt.append(". ".join(g.loc[m, "peoples"]))

    print("loading SigLIP-2 ...", flush=True)
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("google/siglip2-large-patch16-384")
    model = AutoModel.from_pretrained("google/siglip2-large-patch16-384").to(device).eval()

    manifest = {"motif_id": motif_ids}
    for name, txts in [("full", full_txt), ("species", sp_txt),
                        ("place", pl_txt), ("ethnonym", et_txt)]:
        print(f"embedding {name} ...", flush=True)
        e, v = sentpool_embed(tok, model, device, txts)
        np.save(OUT / f"emb_{name}.npy", e)
        manifest[f"valid_{name}"] = v
        print(f"  {name}: valid {int(v.sum())}/{len(v)}", flush=True)
    pd.DataFrame(manifest).to_parquet(OUT / "manifest.parquet", index=False)
    print(f"wrote {OUT}/manifest.parquet + emb_*.npy", flush=True)


if __name__ == "__main__":
    main()
