"""
inat_download_images.py — download the actual image bytes for the iNat manifest.

Reads dataset/imagery/inaturalist/manifest.parquet and downloads each photo at
the given size (medium by default) into dataset/imagery/inaturalist/images/.

Filenames are <photo_id>.<ext> so the run is resumable.

Usage:
    python inat_download_images.py                 # default: medium, 16 workers
    python inat_download_images.py --size large
    python inat_download_images.py --resume        # skip files already present
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import argparse, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm
from PIL import Image
import io

ROOT = Path(__file__).resolve().parents[2]  # project root
MANIFEST = ROOT / "dataset/imagery/inaturalist/manifest.parquet"
IMG_DIR = ROOT / "dataset/imagery/inaturalist/images"

UA = {"User-Agent": "pareidolia-myth-research/0.3"}


def fetch_one(url: str, dest: Path, timeout=30, validate=True):
    if dest.exists() and dest.stat().st_size > 1000:
        return "skip"
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code != 200 or len(r.content) < 1000:
            return f"fail:status={r.status_code}"
        if validate:
            try:
                Image.open(io.BytesIO(r.content)).verify()
            except Exception:
                return "fail:not-image"
        dest.write_bytes(r.content)
        return "ok"
    except Exception as e:
        return f"fail:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["square","small","medium","large","original"],
                    default="medium")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    m = pd.read_parquet(MANIFEST)
    if args.limit:
        m = m.head(args.limit)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Choose URL column based on size
    url_col = f"photo_url_{args.size}" if args.size != "square" else "photo_url_square"
    if url_col not in m.columns:
        # fall back: rewrite square URL with target size
        m[url_col] = m["photo_url_square"].fillna("").str.replace("/square.", f"/{args.size}.", regex=False)

    def make_task(row):
        url = row.get(url_col)
        pid = row.get("photo_id")
        if not isinstance(url, str) or not url.startswith("http") or pid is None:
            return None
        ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
        if ext not in {"jpg","jpeg","png","webp"}:
            ext = "jpg"
        dest = IMG_DIR / f"{int(pid)}.{ext}"
        return (url, dest)

    tasks = []
    for _, r in m.iterrows():
        t = make_task(r)
        if t:
            tasks.append(t)
    print(f"[download] {len(tasks):,} photos at size={args.size} -> {IMG_DIR}")

    stats = {"ok":0, "skip":0, "fail":0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one, url, dest): (url, dest) for url, dest in tasks}
        with tqdm(total=len(futures), desc="img") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                if res == "ok": stats["ok"] += 1
                elif res == "skip": stats["skip"] += 1
                else: stats["fail"] += 1
                pbar.update(1)
                pbar.set_postfix(stats)

    print(f"\n[done] in {(time.time()-t0)/60:.1f} min")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")

    # Add local_path back to manifest
    print("[download] indexing local paths...")
    existing = {p.stem: str(p) for p in IMG_DIR.glob("*.*") if p.stat().st_size > 1000}
    m["local_path"] = m["photo_id"].astype("Int64").astype(str).map(existing).fillna("")
    m.to_parquet(MANIFEST, index=False)
    n_local = (m["local_path"] != "").sum()
    print(f"  manifest now has local_path for {n_local:,} / {len(m):,} photos")


if __name__ == "__main__":
    main()
