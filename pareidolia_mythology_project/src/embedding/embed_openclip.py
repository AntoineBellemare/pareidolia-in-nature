"""
embed_openclip.py — re-embed iNat images + motif text using an OpenCLIP
model, so we can test whether the biome×mythology effect replicates across
vision-language models with different training corpora.

Models tested:
  - OpenCLIP ViT-L/14, pretrained=laion2b_s32b_b82k  (LAION-2B, web image-text)
  - OpenCLIP ViT-L/14, pretrained=openai             (original CLIP, OpenAI web crawl)

If the effect appears in all three (SigLIP-2 + OpenCLIP-LAION + CLIP-OpenAI),
the "you've recovered SigLIP's specific cultural prior" attack loses bite.

Outputs per model SLUG (e.g. openclip_laion2b, openclip_openai):
  dataset/imagery/embeddings/{SLUG}/img_emb.npy
  dataset/imagery/embeddings/{SLUG}/img_paths.parquet  (symlinks meta)
  dataset/imagery/embeddings/{SLUG}/motif_emb_all.npy        (oneliners)
  dataset/imagery/embeddings/{SLUG}/motif_emb_abstracts.npy  (abstracts)
  dataset/imagery/embeddings/{SLUG}/motif_meta_*.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

ROOT = Path(__file__).resolve().parents[2]  # project root
MAP = ROOT / "dataset/mapping_v2"
EMB_ROOT = ROOT / "dataset/imagery/embeddings"
SIGLIP_DIR = EMB_ROOT / "siglip2-large"  # we'll mirror the manifest from here


MODELS = {
    "openclip_laion2b": ("ViT-L-14", "laion2b_s32b_b82k"),
    "openclip_openai":  ("ViT-L-14", "openai"),
}


def load_model(arch: str, pretrained: str, device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(arch)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


def embed_images(model, preprocess, manifest: pd.DataFrame, device,
                 batch_size=64):
    """Encode each image; rows with a missing local_path emit zero vector."""
    embs = np.zeros((len(manifest), model.visual.output_dim), dtype=np.float32)
    batch_imgs, batch_idx = [], []

    def flush():
        if not batch_imgs: return
        tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs[batch_idx] = feats.cpu().numpy().astype(np.float32)
        batch_imgs.clear(); batch_idx.clear()

    for i, row in tqdm(manifest.iterrows(), total=len(manifest), desc="img"):
        p = row.get("local_path")
        if not isinstance(p, str) or not Path(p).exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
            t = preprocess(img)
        except Exception:
            continue
        batch_imgs.append(t)
        batch_idx.append(i)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return embs


def embed_texts(model, tokenizer, texts: list[str], device, batch_size=32,
                max_chars=2000):
    """Encode each text. Long abstracts get truncated to max_chars first to
    keep tokenizer happy."""
    embs = np.zeros((len(texts), model.visual.output_dim), dtype=np.float32)
    for i in tqdm(range(0, len(texts), batch_size), desc="text"):
        batch = [(t or "")[:max_chars] for t in texts[i:i + batch_size]]
        toks = tokenizer(batch).to(device)
        with torch.no_grad():
            feats = model.encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs[i:i + len(batch)] = feats.cpu().numpy().astype(np.float32)
    return embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, choices=list(MODELS))
    ap.add_argument("--limit-imgs", type=int, default=None)
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-motifs", action="store_true")
    args = ap.parse_args()

    arch, pretrained = MODELS[args.slug]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.slug}] loading {arch} pretrained={pretrained} on {device}")
    model, preprocess, tokenizer = load_model(arch, pretrained, device)

    out_dir = EMB_ROOT / args.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========== IMAGES ============
    if not args.skip_images:
        manifest = pd.read_parquet(SIGLIP_DIR / "img_paths.parquet").reset_index(drop=True)
        if args.limit_imgs:
            manifest = manifest.head(args.limit_imgs)
        print(f"[{args.slug}] embedding {len(manifest):,} images …")
        img_emb = embed_images(model, preprocess, manifest, device)
        np.save(out_dir / "img_emb.npy", img_emb)
        manifest.to_parquet(out_dir / "img_paths.parquet", index=False)
        print(f"[{args.slug}]   img_emb shape: {img_emb.shape}")

    # =========== MOTIFS ============
    if not args.skip_motifs:
        motifs = pd.read_parquet(MAP / "motifs.parquet")[
            ["motif_id", "name_en", "description_en"]
        ].fillna("")

        # Oneliners
        motifs_all = motifs.copy()
        motifs_all["text_all"] = motifs_all["name_en"] + " " + motifs_all["description_en"]
        texts = motifs_all["text_all"].tolist()
        print(f"[{args.slug}] embedding {len(texts)} oneliner motifs …")
        all_emb = embed_texts(model, tokenizer, texts, device)
        np.save(out_dir / "motif_emb_all.npy", all_emb)
        motifs_all[["motif_id"]].to_parquet(out_dir / "motif_meta_all.parquet", index=False)

        # Abstracts (load Berezkin russian abstracts if present, else skip)
        ru_path = MAP / "motif_abstracts.parquet"
        if ru_path.exists():
            ru = pd.read_parquet(ru_path)
            # join on motif_id, pool multiple abstracts per motif via concatenation
            grouped = (ru.groupby("motif_id")["abstract_ru"]
                       .apply(lambda s: "\n\n".join(x for x in s if isinstance(x, str)))
                       .reset_index())
            # Reindex to motifs order (motifs with no abstract get empty string)
            joined = motifs[["motif_id"]].merge(grouped, on="motif_id", how="left")
            joined["abstract_ru"] = joined["abstract_ru"].fillna("")
            texts_ab = joined["abstract_ru"].tolist()
            n_with_text = sum(1 for t in texts_ab if t.strip())
            print(f"[{args.slug}] embedding {len(texts_ab)} abstracts "
                  f"({n_with_text} non-empty) …")
            ab_emb = embed_texts(model, tokenizer, texts_ab, device)
            np.save(out_dir / "motif_emb_abstracts.npy", ab_emb)
            joined[["motif_id"]].to_parquet(
                out_dir / "motif_meta_abstracts.parquet", index=False)
        else:
            print(f"[{args.slug}] no russian abstracts found at {ru_path}")

    print(f"[{args.slug}] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
