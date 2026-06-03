"""
embed_and_analyze.py — Step 2 of the pareidolia → mythology pipeline.

What this script does, end-to-end:

  1. Fetches a CONTROL region landscape sample (Sahara, by default) so we have
     something to compare Scandinavian landscapes against.
  2. Fetches Wikimedia Commons images for (a) richer Norse mythological art
     than the Met collection alone, and (b) actual wide-landscape scenery for
     Scandinavia (since iNaturalist is mostly organism close-ups).
  3. Downloads the image bytes from all manifests assembled in step 1
     (pilot_collect.py) and the new ones added here.
  4. Loads SigLIP and computes embeddings for:
        - Norse entity text descriptions       (from Wikidata)
        - Norse mythological art images        (Met + Commons)
        - Scandinavian landscape images        (iNaturalist + Commons)
        - Sahara control landscape images
  5. Runs the headline signal checks:
        Test A: entity text ↔ landscape images
        Test B: Norse art ↔ landscape images
     Each with:
        - Mean cosine similarity, Scandi vs Control
        - Permutation test (10k shuffles) for p-value
        - Per-entity / per-artwork ranking metrics
        - Multi-scale comparison (whole image vs sub-crops)
  6. Writes a markdown report and plots to ./dataset/analysis/.

Dependencies:
    pip install requests pandas pyarrow tqdm pillow numpy matplotlib \\
                torch transformers

Hardware:
    Works on CPU but slow (SigLIP ~3-5s/image on CPU). GPU strongly recommended:
    ~50ms/image on a modern consumer GPU.

Usage:
    python embed_and_analyze.py --all
    python embed_and_analyze.py --section fetch         # add control + Commons
    python embed_and_analyze.py --section download      # fetch image bytes
    python embed_and_analyze.py --section embed         # compute embeddings
    python embed_and_analyze.py --section analyze       # signal check + report
    python embed_and_analyze.py --section analyze --multi-scale
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils


import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

USER_AGENT = "pareidolia-myth-research/0.2 (contact: your_email@example.com)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

PILOT_ROOT = Path("dataset/regions/nordic_boreal")
CONTROL_ROOT = Path("dataset/regions/sahara_control")
ANALYSIS_ROOT = Path("dataset/analysis")
EMB_ROOT = ANALYSIS_ROOT / "embeddings"

MODEL_NAME = "google/siglip-base-patch16-224"

# Wikimedia Commons categories worth pulling
COMMONS_ART_CATEGORIES = [
    "Norse mythology in art",
    "Paintings of Norse mythology",
    "Edda",
    "Vikings in art",
]
COMMONS_SCANDI_LANDSCAPE_CATEGORIES = [
    "Landscapes of Norway",
    "Landscapes of Iceland",
    "Fjords of Norway",
    "Mountains of Norway",
    "Boreal forests in Norway",
    "Landscapes of Sweden",
]
COMMONS_SAHARA_LANDSCAPE_CATEGORIES = [
    "Sahara",
    "Landscapes of Algeria",
    "Landscapes of Libya",
    "Deserts of Africa",
    "Sand dunes in Africa",
]

# Sahara rough bounding box for iNaturalist control
SAHARA_BBOX = {"swlat": 18.0, "swlng": -15.0, "nelat": 30.0, "nelng": 30.0}

# Caps so the pilot runs in reasonable time
LANDSCAPE_TARGET_PER_REGION = 300
ART_TARGET = 200

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def ensure_dirs() -> None:
    for d in (
        CONTROL_ROOT / "landscape/inaturalist",
        CONTROL_ROOT / "landscape/wikimedia",
        PILOT_ROOT / "landscape/wikimedia",
        PILOT_ROOT / "mythology_art/wikimedia",
        ANALYSIS_ROOT,
        EMB_ROOT,
    ):
        d.mkdir(parents=True, exist_ok=True)


def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------------------------------------------------------- #
# Section 1: Wikimedia Commons fetcher
# --------------------------------------------------------------------------- #

COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def commons_category_images(category: str, limit: int = 200) -> list[dict]:
    """Return file metadata for images in a Commons category."""
    out = []
    cmcontinue = None
    while len(out) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "file",
            "cmlimit": min(500, limit - len(out)),
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        members = data.get("query", {}).get("categorymembers", [])
        if not members:
            break

        # Resolve actual URLs for this batch
        titles = "|".join(m["title"] for m in members
                          if m["title"].lower().endswith((".jpg", ".jpeg", ".png")))
        if titles:
            info_params = {
                "action": "query",
                "titles": titles,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime|size",
                "iiurlwidth": 800,
                "format": "json",
            }
            ir = requests.get(COMMONS_API, params=info_params,
                              headers=HEADERS, timeout=60)
            ir.raise_for_status()
            pages = ir.json().get("query", {}).get("pages", {})
            for page in pages.values():
                ii = (page.get("imageinfo") or [{}])[0]
                meta = ii.get("extmetadata", {})
                out.append({
                    "title": page.get("title"),
                    "url": ii.get("thumburl") or ii.get("url"),
                    "original_url": ii.get("url"),
                    "license": (meta.get("LicenseShortName") or {}).get("value"),
                    "artist": (meta.get("Artist") or {}).get("value"),
                    "date": (meta.get("DateTimeOriginal") or {}).get("value"),
                    "description": (meta.get("ImageDescription") or {}).get("value"),
                    "category": category,
                })
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(0.5)
    return out[:limit]


def fetch_commons_set(categories: list[str], out_path: Path, target: int):
    print(f"[commons] fetching {len(categories)} categories → {out_path}")
    rows = []
    per_cat = max(target // len(categories), 30)
    for cat in categories:
        try:
            items = commons_category_images(cat, limit=per_cat)
            print(f"  {cat}: {len(items)} images")
            rows.extend(items)
        except Exception as e:
            print(f"  {cat}: failed ({e})")
    df = pd.DataFrame(rows).drop_duplicates(subset=["url"])
    df.to_parquet(out_path, index=False)
    print(f"[commons] saved {len(df)} images → {out_path}")
    return df


# --------------------------------------------------------------------------- #
# Section 2: iNaturalist Sahara control
# --------------------------------------------------------------------------- #

INAT_OBS = "https://api.inaturalist.org/v1/observations"


def fetch_sahara_inat(target: int = 300) -> pd.DataFrame:
    print(f"[inat-control] sampling Sahara observations ...")
    rows = []
    page = 1
    while len(rows) < target:
        r = requests.get(
            INAT_OBS,
            params={
                **{f"sw{k[2:]}" if k.startswith("sw") else f"ne{k[2:]}": v
                   for k, v in SAHARA_BBOX.items()},
                "quality_grade": "research",
                "photo_license": "cc0,cc-by,cc-by-nc",
                "geoprivacy": "open",
                "per_page": 100,
                "page": page,
                "order_by": "random",
            },
            headers=HEADERS, timeout=60,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            break
        for obs in results:
            photos = obs.get("photos") or []
            if not photos:
                continue
            p = photos[0]
            rows.append({
                "inat_id": obs.get("id"),
                "taxon": (obs.get("taxon") or {}).get("name"),
                "lat": (obs.get("geojson") or {}).get("coordinates", [None, None])[1],
                "lon": (obs.get("geojson") or {}).get("coordinates", [None, None])[0],
                "place_guess": obs.get("place_guess"),
                "photo_url_medium": p.get("url", "").replace("square", "medium"),
                "license": p.get("license_code"),
                "attribution": p.get("attribution"),
            })
        page += 1
        time.sleep(1.0)
    df = pd.DataFrame(rows[:target])
    out = CONTROL_ROOT / "landscape/inaturalist/manifest.parquet"
    df.to_parquet(out, index=False)
    print(f"[inat-control] saved {len(df)} obs → {out}")
    return df


# --------------------------------------------------------------------------- #
# Section 3: Image downloader (fetch bytes for all manifests)
# --------------------------------------------------------------------------- #


def download_image(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
        if r.status_code != 200 or len(r.content) < 1000:
            return False
        # validate it's an image
        try:
            Image.open(__import__("io").BytesIO(r.content)).verify()
        except Exception:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def download_from_manifest(manifest_path: Path, url_col: str, id_col: str,
                           out_dir: Path, max_n: int | None = None) -> pd.DataFrame:
    if not manifest_path.exists():
        print(f"[download] missing {manifest_path}, skipping")
        return pd.DataFrame()
    df = pd.read_parquet(manifest_path)
    if max_n:
        df = df.head(max_n)
    out_dir.mkdir(parents=True, exist_ok=True)
    df["local_path"] = ""
    ok = 0
    for i, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {manifest_path.parent.name}/{manifest_path.parent.parent.name}"):
        url = row.get(url_col)
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
        if ext not in {"jpg", "jpeg", "png", "webp"}:
            ext = "jpg"
        dest = out_dir / f"{row[id_col]}.{ext}"
        if download_image(url, dest):
            df.at[i, "local_path"] = str(dest)
            ok += 1
        time.sleep(0.05)  # be polite
    print(f"  {ok}/{len(df)} downloaded")
    df.to_parquet(manifest_path, index=False)  # save local_path back
    return df


def download_all_images():
    print("[download] fetching image bytes for all manifests ...")

    # Scandi landscape (iNat)
    download_from_manifest(
        PILOT_ROOT / "landscape/inaturalist/manifest.parquet",
        url_col="photo_url_medium", id_col="inat_id",
        out_dir=PILOT_ROOT / "landscape/inaturalist/images",
        max_n=LANDSCAPE_TARGET_PER_REGION,
    )
    # Scandi landscape (Commons)
    download_from_manifest(
        PILOT_ROOT / "landscape/wikimedia/manifest.parquet",
        url_col="url", id_col="title",
        out_dir=PILOT_ROOT / "landscape/wikimedia/images",
        max_n=LANDSCAPE_TARGET_PER_REGION,
    )
    # Sahara control (iNat)
    download_from_manifest(
        CONTROL_ROOT / "landscape/inaturalist/manifest.parquet",
        url_col="photo_url_medium", id_col="inat_id",
        out_dir=CONTROL_ROOT / "landscape/inaturalist/images",
        max_n=LANDSCAPE_TARGET_PER_REGION,
    )
    # Sahara control (Commons)
    download_from_manifest(
        CONTROL_ROOT / "landscape/wikimedia/manifest.parquet",
        url_col="url", id_col="title",
        out_dir=CONTROL_ROOT / "landscape/wikimedia/images",
        max_n=LANDSCAPE_TARGET_PER_REGION,
    )
    # Norse art (Met)
    download_from_manifest(
        PILOT_ROOT / "mythology_art/met/manifest.parquet",
        url_col="primary_image_small", id_col="met_id",
        out_dir=PILOT_ROOT / "mythology_art/met/images",
        max_n=ART_TARGET,
    )
    # Norse art (Commons)
    download_from_manifest(
        PILOT_ROOT / "mythology_art/wikimedia/manifest.parquet",
        url_col="url", id_col="title",
        out_dir=PILOT_ROOT / "mythology_art/wikimedia/images",
        max_n=ART_TARGET,
    )


# --------------------------------------------------------------------------- #
# Section 4: SigLIP embedding
# --------------------------------------------------------------------------- #


def _load_model(device: str | None = None):
    """Load SigLIP. Lazy to keep --help fast."""
    import torch
    from transformers import AutoProcessor, AutoModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model] loading {MODEL_NAME} on {device}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    return model, processor, device


def embed_texts(texts: list[str], batch_size: int = 16) -> np.ndarray:
    import torch
    model, processor, device = _load_model()
    out = []
    for batch in tqdm(list(batched(texts, batch_size)), desc="text"):
        inputs = processor(text=batch, return_tensors="pt",
                           padding="max_length", truncation=True).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.vstack(out)


def _multi_scale_crops(img: Image.Image) -> list[Image.Image]:
    """Return [whole, center-half, 4 corner-halves] = 6 views."""
    w, h = img.size
    s = min(w, h)
    half = s // 2
    crops = [img]
    cx, cy = (w - half) // 2, (h - half) // 2
    crops.append(img.crop((cx, cy, cx + half, cy + half)))
    for x, y in [(0, 0), (w - half, 0), (0, h - half), (w - half, h - half)]:
        crops.append(img.crop((x, y, x + half, y + half)))
    return crops


def embed_images(paths: list[Path], batch_size: int = 16,
                 multi_scale: bool = False) -> tuple[np.ndarray, list[Path], np.ndarray | None]:
    """
    Returns: (embeddings, valid_paths, scale_embeddings_or_None)
        - embeddings: (n_valid, dim) — whole-image features
        - scale_embeddings: if multi_scale, (n_valid, 6, dim)
    """
    import torch
    model, processor, device = _load_model()
    whole_emb = []
    scale_emb = [] if multi_scale else None
    valid = []

    for batch_paths in tqdm(list(batched(paths, batch_size)), desc="image"):
        images = []
        keep = []
        for p in batch_paths:
            try:
                im = Image.open(p).convert("RGB")
                if min(im.size) < 64:
                    continue
                images.append(im)
                keep.append(p)
            except Exception:
                continue
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            f = model.get_image_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)
        whole_emb.append(f.cpu().numpy())
        valid.extend(keep)

        if multi_scale:
            # 6 views per image; flatten into one big batch then reshape
            views = []
            for im in images:
                views.extend(_multi_scale_crops(im))
            v_inputs = processor(images=views, return_tensors="pt").to(device)
            with torch.no_grad():
                vf = model.get_image_features(**v_inputs)
            vf = vf / vf.norm(dim=-1, keepdim=True)
            vf = vf.cpu().numpy().reshape(len(images), 6, -1)
            scale_emb.append(vf)

    emb = np.vstack(whole_emb) if whole_emb else np.zeros((0, 768), dtype=np.float32)
    scale = np.concatenate(scale_emb, axis=0) if multi_scale and scale_emb else None
    return emb, valid, scale


# --------------------------------------------------------------------------- #
# Section 5: Compute all embeddings & cache them
# --------------------------------------------------------------------------- #


def _gather_local_paths(*manifests: Path) -> list[Path]:
    paths = []
    for m in manifests:
        if not m.exists():
            continue
        df = pd.read_parquet(m)
        if "local_path" not in df.columns:
            continue
        for p in df["local_path"]:
            if isinstance(p, str) and p and Path(p).exists():
                paths.append(Path(p))
    return paths


def compute_all_embeddings(multi_scale: bool = False):
    EMB_ROOT.mkdir(parents=True, exist_ok=True)

    # --- Norse entity text descriptions ---
    ent_path = PILOT_ROOT / "entities/norse_entities.parquet"
    if ent_path.exists():
        ent = pd.read_parquet(ent_path)
        ent = ent[ent["description"].fillna("").str.len() > 15].copy()
        # Construct text as "label: description" for context
        ent["embed_text"] = ent["label"].fillna("") + ": " + ent["description"].fillna("")
        ent = ent[ent["label"].notna()].reset_index(drop=True)
        print(f"[embed] {len(ent)} Norse entities with descriptions")
        text_emb = embed_texts(ent["embed_text"].tolist())
        np.save(EMB_ROOT / "norse_text.npy", text_emb)
        ent.to_parquet(EMB_ROOT / "norse_text_meta.parquet", index=False)

    # --- Norse mythological art ---
    art_paths = _gather_local_paths(
        PILOT_ROOT / "mythology_art/met/manifest.parquet",
        PILOT_ROOT / "mythology_art/wikimedia/manifest.parquet",
    )
    print(f"[embed] {len(art_paths)} Norse art images")
    if art_paths:
        art_emb, art_valid, art_scale = embed_images(art_paths, multi_scale=multi_scale)
        np.save(EMB_ROOT / "norse_art.npy", art_emb)
        pd.DataFrame({"path": [str(p) for p in art_valid]}).to_parquet(
            EMB_ROOT / "norse_art_paths.parquet", index=False)
        if art_scale is not None:
            np.save(EMB_ROOT / "norse_art_scales.npy", art_scale)

    # --- Scandinavian landscapes ---
    scandi_paths = _gather_local_paths(
        PILOT_ROOT / "landscape/inaturalist/manifest.parquet",
        PILOT_ROOT / "landscape/wikimedia/manifest.parquet",
    )
    print(f"[embed] {len(scandi_paths)} Scandinavian landscape images")
    if scandi_paths:
        s_emb, s_valid, s_scale = embed_images(scandi_paths, multi_scale=multi_scale)
        np.save(EMB_ROOT / "scandi_landscape.npy", s_emb)
        pd.DataFrame({"path": [str(p) for p in s_valid]}).to_parquet(
            EMB_ROOT / "scandi_landscape_paths.parquet", index=False)
        if s_scale is not None:
            np.save(EMB_ROOT / "scandi_landscape_scales.npy", s_scale)

    # --- Sahara control landscapes ---
    ctrl_paths = _gather_local_paths(
        CONTROL_ROOT / "landscape/inaturalist/manifest.parquet",
        CONTROL_ROOT / "landscape/wikimedia/manifest.parquet",
    )
    print(f"[embed] {len(ctrl_paths)} Sahara control landscape images")
    if ctrl_paths:
        c_emb, c_valid, c_scale = embed_images(ctrl_paths, multi_scale=multi_scale)
        np.save(EMB_ROOT / "control_landscape.npy", c_emb)
        pd.DataFrame({"path": [str(p) for p in c_valid]}).to_parquet(
            EMB_ROOT / "control_landscape_paths.parquet", index=False)
        if c_scale is not None:
            np.save(EMB_ROOT / "control_landscape_scales.npy", c_scale)

    print(f"[embed] all embeddings cached → {EMB_ROOT}")


# --------------------------------------------------------------------------- #
# Section 6: The actual signal check
# --------------------------------------------------------------------------- #


def permutation_test(query_emb: np.ndarray, scandi_emb: np.ndarray,
                     control_emb: np.ndarray, n_perms: int = 10_000,
                     rng_seed: int = 42) -> dict:
    """
    H0: query embeddings have no preferential similarity to Scandi landscapes
        over the pooled (Scandi + Control) landscape population.
    Test stat: mean(sim(query, scandi)) - mean(sim(query, control)).
    """
    rng = np.random.default_rng(rng_seed)
    all_land = np.vstack([scandi_emb, control_emb])
    n_scandi = scandi_emb.shape[0]

    observed = (query_emb @ scandi_emb.T).mean() - (query_emb @ control_emb.T).mean()

    # Vectorized null: instead of looping 10k matmuls, precompute query@all once.
    q_dot_all = query_emb @ all_land.T  # (n_query, n_total)
    n_total = q_dot_all.shape[1]
    null = np.empty(n_perms)
    for k in range(n_perms):
        perm = rng.permutation(n_total)
        s_idx = perm[:n_scandi]
        c_idx = perm[n_scandi:]
        null[k] = q_dot_all[:, s_idx].mean() - q_dot_all[:, c_idx].mean()

    p_one_sided = float((null >= observed).mean())
    return {
        "observed_delta": float(observed),
        "mean_sim_scandi": float((query_emb @ scandi_emb.T).mean()),
        "mean_sim_control": float((query_emb @ control_emb.T).mean()),
        "p_value_one_sided": p_one_sided,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_distribution": null,
    }


def per_item_ranking(query_emb: np.ndarray, query_labels: list,
                     scandi_emb: np.ndarray, control_emb: np.ndarray) -> pd.DataFrame:
    """For each query, where do Scandi images land in the similarity ranking?"""
    all_land = np.vstack([scandi_emb, control_emb])
    is_scandi = np.array([True] * len(scandi_emb) + [False] * len(control_emb))
    n = len(all_land)
    expected_top10 = is_scandi.mean()  # baseline if labels were random

    rows = []
    for i, lab in enumerate(query_labels):
        sims = query_emb[i] @ all_land.T
        order = np.argsort(-sims)
        ranks = is_scandi[order]
        rows.append({
            "query": lab,
            "top10_scandi_fraction": float(ranks[:10].mean()),
            "top50_scandi_fraction": float(ranks[:50].mean()),
            "mean_scandi_rank": float(np.where(ranks)[0].mean() if ranks.any() else np.nan),
            "baseline_scandi_fraction": float(expected_top10),
        })
    return pd.DataFrame(rows).sort_values("top10_scandi_fraction", ascending=False)


def run_analysis(multi_scale: bool = False) -> dict:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)

    # Load
    text_emb = np.load(EMB_ROOT / "norse_text.npy")
    text_meta = pd.read_parquet(EMB_ROOT / "norse_text_meta.parquet")
    art_emb = np.load(EMB_ROOT / "norse_art.npy")
    scandi_emb = np.load(EMB_ROOT / "scandi_landscape.npy")
    control_emb = np.load(EMB_ROOT / "control_landscape.npy")

    print(f"[analyze] sizes: text={text_emb.shape}, art={art_emb.shape}, "
          f"scandi={scandi_emb.shape}, control={control_emb.shape}")

    results = {
        "config": {
            "model": MODEL_NAME,
            "n_text": int(text_emb.shape[0]),
            "n_art": int(art_emb.shape[0]),
            "n_scandi": int(scandi_emb.shape[0]),
            "n_control": int(control_emb.shape[0]),
            "multi_scale": multi_scale,
        }
    }

    # Test A: entity text → landscape
    print("[analyze] Test A: entity text → landscape (permutation test)")
    results["test_A_text_to_landscape"] = permutation_test(text_emb, scandi_emb, control_emb)
    # Test B: Norse art → landscape
    print("[analyze] Test B: Norse art → landscape (permutation test)")
    results["test_B_art_to_landscape"] = permutation_test(art_emb, scandi_emb, control_emb)

    # Per-entity ranking
    results["per_entity"] = per_item_ranking(text_emb, text_meta["label"].tolist(),
                                              scandi_emb, control_emb)

    # Multi-scale comparison
    if multi_scale:
        s_scales = EMB_ROOT / "scandi_landscape_scales.npy"
        c_scales = EMB_ROOT / "control_landscape_scales.npy"
        if s_scales.exists() and c_scales.exists():
            s_scale = np.load(s_scales)
            c_scale = np.load(c_scales)
            ms = []
            scale_names = ["whole", "center-half", "TL", "TR", "BL", "BR"]
            for k, name in enumerate(scale_names):
                r = permutation_test(art_emb, s_scale[:, k], c_scale[:, k], n_perms=2000)
                ms.append({
                    "scale": name,
                    "delta": r["observed_delta"],
                    "p_value": r["p_value_one_sided"],
                })
            results["multi_scale"] = pd.DataFrame(ms)
            print(results["multi_scale"].to_string(index=False))

    return results


# --------------------------------------------------------------------------- #
# Section 7: Plotting & report
# --------------------------------------------------------------------------- #


def make_plots(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: Similarity distributions for Test A and Test B (null + observed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (key, title) in zip(axes, [
        ("test_A_text_to_landscape", "Test A: entity text → landscape"),
        ("test_B_art_to_landscape",  "Test B: Norse art → landscape"),
    ]):
        r = results[key]
        ax.hist(r["null_distribution"], bins=60, color="lightgray",
                edgecolor="gray", label="null (shuffled)")
        ax.axvline(r["observed_delta"], color="crimson", linewidth=2,
                   label=f"observed Δ = {r['observed_delta']:+.4f}\n"
                         f"p = {r['p_value_one_sided']:.4f}")
        ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xlabel("Δ = mean sim(Scandi) − mean sim(Control)")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(ANALYSIS_ROOT / "permutation_results.png", dpi=130)
    plt.close(fig)

    # Plot 2: Top-N Norse entities ranked by Scandi-leaning
    pe = results["per_entity"].head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    ypos = np.arange(len(pe))[::-1]
    baseline = pe["baseline_scandi_fraction"].iloc[0]
    ax.barh(ypos, pe["top10_scandi_fraction"], color="steelblue")
    ax.axvline(baseline, color="black", linestyle="--", linewidth=1,
               label=f"random baseline = {baseline:.2f}")
    ax.set_yticks(ypos)
    ax.set_yticklabels(pe["query"], fontsize=9)
    ax.set_xlabel("fraction of top-10 nearest landscapes that are Scandinavian")
    ax.set_title("Top 20 Norse entities by Scandi-leaning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_ROOT / "per_entity_ranking.png", dpi=130)
    plt.close(fig)

    # Plot 3: multi-scale, if available
    if "multi_scale" in results:
        ms = results["multi_scale"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(ms["scale"], ms["delta"], color="darkorange")
        for i, p in enumerate(ms["p_value"]):
            ax.text(i, ms["delta"].iloc[i], f"p={p:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel("Δ (Scandi − Control)")
        ax.set_title("Multi-scale: where does the signal live?")
        fig.tight_layout()
        fig.savefig(ANALYSIS_ROOT / "multi_scale.png", dpi=130)
        plt.close(fig)


def write_report(results: dict):
    cfg = results["config"]
    A = results["test_A_text_to_landscape"]
    B = results["test_B_art_to_landscape"]
    pe = results["per_entity"]
    lines = [
        "# Pilot signal-check report",
        "",
        f"Model: `{cfg['model']}`  ",
        f"Norse entities (text): **{cfg['n_text']}**  ",
        f"Norse art images: **{cfg['n_art']}**  ",
        f"Scandinavian landscapes: **{cfg['n_scandi']}**  ",
        f"Sahara control landscapes: **{cfg['n_control']}**  ",
        f"Multi-scale: **{cfg['multi_scale']}**",
        "",
        "## Test A — Norse entity text descriptions vs landscape images",
        "",
        f"- Mean cosine sim (entity, Scandi):  **{A['mean_sim_scandi']:+.4f}**",
        f"- Mean cosine sim (entity, Control): **{A['mean_sim_control']:+.4f}**",
        f"- Δ (Scandi − Control):              **{A['observed_delta']:+.4f}**",
        f"- Permutation p (one-sided, 10k):    **{A['p_value_one_sided']:.4f}**",
        "",
        "## Test B — Norse mythological art vs landscape images  ←  headline test",
        "",
        f"- Mean cosine sim (art, Scandi):  **{B['mean_sim_scandi']:+.4f}**",
        f"- Mean cosine sim (art, Control): **{B['mean_sim_control']:+.4f}**",
        f"- Δ (Scandi − Control):           **{B['observed_delta']:+.4f}**",
        f"- Permutation p (one-sided, 10k): **{B['p_value_one_sided']:.4f}**",
        "",
        "## Top 15 Scandi-leaning Norse entities",
        "",
        "| Entity | top-10 Scandi fraction | top-50 Scandi fraction |",
        "|---|---:|---:|",
    ]
    for _, r in pe.head(15).iterrows():
        lines.append(f"| {r['query']} | {r['top10_scandi_fraction']:.2f} "
                     f"| {r['top50_scandi_fraction']:.2f} |")
    lines += [
        "",
        f"Baseline (random) Scandi fraction = **{pe['baseline_scandi_fraction'].iloc[0]:.2f}**.",
        "Entities well above baseline are visually consistent with Scandinavian landscapes",
        "(in SigLIP space) more than chance would predict.",
        "",
        "## Caveats",
        "",
        "- Sample sizes are small; this is a sanity-check pilot, not a study.",
        "- iNaturalist images are mostly organism close-ups; Commons gives wider scenery.",
        "- SigLIP was trained on web captions, which encode cultural biases — a Norse-themed",
        "  caption may already drag toward Scandinavian imagery. The Test B (image↔image)",
        "  comparison is less subject to this than Test A.",
        "- Sahara was chosen as a maximally contrasting control. Replicate with other",
        "  control biomes (tropical rainforest, temperate steppe) before drawing conclusions.",
        "- Pareidolia is hypothesized to operate at sub-image scales; the multi-scale plot",
        "  (if run) is the first probe of that.",
        "",
        "## Plots",
        "",
        "- `permutation_results.png` — null distributions for Test A and Test B with observed Δ",
        "- `per_entity_ranking.png` — which Norse entities discriminate most",
        "- `multi_scale.png` — signal by image scale (if --multi-scale was passed)",
    ]
    (ANALYSIS_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    # Also save the per-entity table
    pe.to_csv(ANALYSIS_ROOT / "per_entity_ranking.csv", index=False)
    print(f"[report] wrote {ANALYSIS_ROOT/'report.md'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--section",
                   choices=["all", "fetch", "download", "embed", "analyze"],
                   default="all")
    p.add_argument("--multi-scale", action="store_true",
                   help="Embed and analyze multi-scale crops too")
    args = p.parse_args()

    ensure_dirs()

    if args.section in ("all", "fetch"):
        # Wikimedia art + landscapes for Norse pilot
        fetch_commons_set(COMMONS_ART_CATEGORIES,
                          PILOT_ROOT / "mythology_art/wikimedia/manifest.parquet",
                          target=ART_TARGET)
        fetch_commons_set(COMMONS_SCANDI_LANDSCAPE_CATEGORIES,
                          PILOT_ROOT / "landscape/wikimedia/manifest.parquet",
                          target=LANDSCAPE_TARGET_PER_REGION)
        # Sahara control
        fetch_sahara_inat(target=LANDSCAPE_TARGET_PER_REGION)
        fetch_commons_set(COMMONS_SAHARA_LANDSCAPE_CATEGORIES,
                          CONTROL_ROOT / "landscape/wikimedia/manifest.parquet",
                          target=LANDSCAPE_TARGET_PER_REGION)

    if args.section in ("all", "download"):
        download_all_images()

    if args.section in ("all", "embed"):
        compute_all_embeddings(multi_scale=args.multi_scale)

    if args.section in ("all", "analyze"):
        results = run_analysis(multi_scale=args.multi_scale)
        make_plots(results)
        write_report(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
