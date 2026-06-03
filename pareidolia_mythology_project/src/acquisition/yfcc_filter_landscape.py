"""
yfcc_filter_landscape.py — zero-shot quality filter for the YFCC manifest.

The original keyword-tag filter (landscape OR scenery OR mountain OR ... OR
outdoor OR nature OR scenic) pulled in many irrelevant photos: party shots,
portraits, indoor scenes that happen to include "outdoor" or "nature" in a
tag. This script fixes that by using the SigLIP-2-large text encoder to score
each YFCC image as "landscape vista" vs "people / portrait / indoor / party".

Procedure:
  1. Embed ~12 LANDSCAPE prompts and ~12 REJECT prompts.
  2. For each YFCC image, compute mean_sim_landscape − mean_sim_reject.
  3. Threshold (default >0): keep ~30-50% as "good landscape".
  4. Save a filtered img_emb.npy + img_paths.parquet at
     embeddings/siglip2-large/yfcc_filtered/.
  5. Write a small HTML contact sheet (top-100 kept + top-100 rejected) for
     visual verification.

Run:
    python yfcc_filter_landscape.py
    python yfcc_filter_landscape.py --threshold 0.005   # stricter
    python yfcc_filter_landscape.py --keep-frac 0.4     # keep top 40% by score
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
YFCC_DIR = EMB / "yfcc"
OUT_DIR = EMB / "yfcc_filtered"
HTML_DIR = ROOT / "dataset/imagery/yfcc100m"

MODEL = "google/siglip2-large-patch16-384"

# Landscape / nature scene prompts
LANDSCAPE_PROMPTS = [
    "a photograph of a natural landscape",
    "a wide vista of mountains and forest",
    "a desert scene with sand dunes",
    "a forest path with trees and undergrowth",
    "a river or lake in a wilderness area",
    "a coastline with the sea and rocks",
    "an open grassland or meadow",
    "a snowy mountain peak",
    "a panoramic outdoor nature scene",
    "scenery with no people, an empty natural environment",
    "a tropical jungle scene",
    "an arctic tundra with ice and snow",
]

# Stuff we definitely want to throw out
REJECT_PROMPTS = [
    "people at a party indoors",
    "a group of friends posing for the camera",
    "a portrait of a person",
    "a selfie taken indoors",
    "a wedding or celebration photo",
    "a concert or band performance",
    "the interior of a building or room",
    "a busy city street scene",
    "a parked car or other vehicle close-up",
    "a plate of food on a table",
    "an animal close-up portrait shot",
    "a sports event with players",
    "a person holding a microphone or musical instrument",
]


def embed_prompts(model_id: str, prompts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoProcessor, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[filter] loading {model_id} ...")
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    inputs = proc(text=prompts, return_tensors="pt",
                  padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        f = model.get_text_features(**inputs)
        if hasattr(f, "pooler_output"):
            f = f.pooler_output
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()


def make_inspect_html(kept: pd.DataFrame, rejected: pd.DataFrame, out: Path,
                       per_section: int = 100):
    import html
    parts = ["""<!doctype html><html><head><meta charset='utf-8'>
<title>YFCC landscape filter inspect</title>
<style>
body { font-family: system-ui, sans-serif; margin: 0; padding: 24px;
       background:#111; color:#ddd; }
h1 { font-weight:600; font-size:22px; margin:0 0 8px; }
h2 { color:#9cf; font-size:16px; margin:24px 0 8px; }
.row { display:flex; flex-wrap:wrap; gap:6px; }
.tile { width:120px; }
.tile img { width:120px; height:120px; object-fit:cover; border-radius:4px;
            background:#222; display:block; }
.cap { font-size:9px; color:#aaa; margin-top:2px; line-height:1.2;
       overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sub { color:#888; font-size:13px; }
.green { color:#9fcdab; } .red { color:#cf8a7f; }
</style></head><body>"""]
    parts.append("<h1>YFCC landscape filter — visual inspection</h1>")
    parts.append(f"<div class='sub'>Threshold separates 'landscape-like' "
                 f"(top section) from 'people/indoor/etc.' (bottom section).</div>")

    def block(df: pd.DataFrame, title: str, color_class: str):
        sub = df.head(per_section)
        parts.append(f"<h2 class='{color_class}'>{title} "
                     f"(showing {len(sub)} of {len(df):,})</h2>")
        parts.append("<div class='row'>")
        for _, r in sub.iterrows():
            url = r.get("photo_url") or r.get("photo_url_medium") or ""
            score = r.get("landscape_score", 0)
            tags = str(r.get("user_tags", ""))[:30]
            parts.append(
                f"<div class='tile'><img loading='lazy' "
                f"src='{html.escape(str(url))}'/>"
                f"<div class='cap'>{score:+.3f} · {html.escape(tags)}</div></div>"
            )
        parts.append("</div>")

    block(kept, "KEPT — highest landscape-score", "green")
    block(rejected, "REJECTED — lowest landscape-score (= most party/portrait/indoor)",
          "red")
    parts.append("</body></html>")
    out.write_text("".join(parts), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None,
                    help="Keep images with landscape_score > THRESHOLD. "
                         "If unset, use --keep-frac.")
    ap.add_argument("--keep-frac", type=float, default=0.5,
                    help="Keep this fraction of top-scoring images.")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    print("[filter] embedding landscape & reject prompts ...")
    land_vecs = embed_prompts(args.model, LANDSCAPE_PROMPTS)
    rej_vecs  = embed_prompts(args.model, REJECT_PROMPTS)
    print(f"  landscape prompts: {land_vecs.shape}")
    print(f"  reject prompts: {rej_vecs.shape}")

    print("[filter] loading YFCC embeddings ...")
    img_emb = np.load(YFCC_DIR / "img_emb.npy")
    manifest = pd.read_parquet(YFCC_DIR / "img_paths.parquet")
    print(f"  {len(img_emb):,} YFCC images")

    # Score: mean similarity to landscape prompts minus mean to reject prompts.
    sim_land = (img_emb @ land_vecs.T).mean(axis=1)
    sim_rej  = (img_emb @ rej_vecs.T).mean(axis=1)
    score = sim_land - sim_rej
    manifest = manifest.copy()
    manifest["landscape_score"] = score
    manifest["sim_landscape"] = sim_land
    manifest["sim_reject"] = sim_rej

    print(f"\n[filter] score distribution: "
          f"min={score.min():.3f}  median={np.median(score):.3f}  "
          f"mean={score.mean():.3f}  max={score.max():.3f}")

    # Decide threshold
    if args.threshold is not None:
        thr = args.threshold
    else:
        thr = float(np.quantile(score, 1.0 - args.keep_frac))
    keep_mask = score > thr
    print(f"\n[filter] threshold = {thr:.4f}  →  keeping "
          f"{int(keep_mask.sum()):,} / {len(score):,} "
          f"({100*keep_mask.mean():.0f}%)")

    # Save filtered embeddings + manifest
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "img_emb.npy", img_emb[keep_mask])
    manifest[keep_mask].reset_index(drop=True).to_parquet(
        OUT_DIR / "img_paths.parquet", index=False
    )
    print(f"  saved filtered img_emb.npy and img_paths.parquet to {OUT_DIR}")

    # Per-biome breakdown
    biome_col = ("photo_biome_wwf" if "photo_biome_wwf" in manifest.columns
                 else "tradition_biome_wwf")
    print("\nbefore vs after, per biome:")
    before = manifest.groupby(biome_col).size().rename("before")
    after  = manifest[keep_mask].groupby(biome_col).size().rename("after")
    print(pd.concat([before, after], axis=1).fillna(0).astype(int).sort_values(
        "before", ascending=False).to_string())

    # HTML inspect
    sorted_df = manifest.sort_values("landscape_score", ascending=False)
    kept_top = sorted_df.head(100)
    rej_bottom = sorted_df.tail(100).sort_values("landscape_score", ascending=True)
    html_out = HTML_DIR / "inspect_filtered.html"
    make_inspect_html(kept_top, rej_bottom, html_out)
    print(f"\n[filter] inspect HTML → {html_out}")


if __name__ == "__main__":
    main()
