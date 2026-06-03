"""
yfcc_download_images.py — download the YFCC manifest's photos at medium size
(`url_m` resolution) into dataset/imagery/yfcc100m/images/.

Resumable: skips already-downloaded files. Writes local_path back into the
manifest in place.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import argparse, time, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # project root
YF = ROOT / "dataset/imagery/yfcc100m"
IMG_DIR = YF / "images"
UA = {"User-Agent": "pareidolia-myth-research/0.4"}


def fetch_one(url: str, dest: Path, timeout=30):
    if dest.exists() and dest.stat().st_size > 1000:
        return "skip"
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code != 200 or len(r.content) < 1000:
            return f"fail:status={r.status_code}"
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
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    m = pd.read_parquet(YF / "manifest.parquet")
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[yfcc-dl] {len(m):,} photos to fetch")

    def make_task(row):
        url = row.get("photo_url") or row.get("photo_url_medium")
        pid = row.get("photo_id")
        if not isinstance(url, str) or not url.startswith("http") or pd.isna(pid):
            return None
        ext = url.rsplit(".",1)[-1].split("?")[0].lower()
        if ext not in {"jpg","jpeg","png","webp"}:
            ext = "jpg"
        dest = IMG_DIR / f"{int(pid)}.{ext}"
        return (url, dest)

    tasks = [t for t in (make_task(r) for _, r in m.iterrows()) if t]
    stats = {"ok":0, "skip":0, "fail":0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one, url, dest): (url, dest)
                   for url, dest in tasks}
        with tqdm(total=len(futures), desc="yfcc-dl") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                if res == "ok": stats["ok"] += 1
                elif res == "skip": stats["skip"] += 1
                else: stats["fail"] += 1
                pbar.update(1)
                pbar.set_postfix(stats)
    print(f"\n[done] in {(time.time()-t0)/60:.1f} min", stats)

    # Index local paths
    existing = {p.stem: str(p) for p in IMG_DIR.glob("*.*") if p.stat().st_size > 1000}
    m["local_path"] = m["photo_id"].astype("Int64").astype(str).map(existing).fillna("")
    m.to_parquet(YF / "manifest.parquet", index=False)
    n_local = (m["local_path"] != "").sum()
    print(f"  local_path set for {n_local:,} / {len(m):,}")


if __name__ == "__main__":
    main()
