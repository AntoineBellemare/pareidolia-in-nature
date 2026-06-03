"""
inat_bulk_sample.py — pull a per-tradition iNaturalist image manifest for the
v2 spine.

For each of the 958 traditions in `dataset/mapping_v2/traditions.parquet`:
  1. Query the iNaturalist API for research-grade observations within an
     adaptive bbox around the tradition's lat/lon (expanding 0.5°→1°→2°→5°
     until we get at least min_per_tradition hits).
  2. Filter to photos under CC0 / CC-BY / CC-BY-NC.
  3. Keep up to `per_tradition` random observations.
  4. Each observation is also tagged with the WWF ecoregion and biome of its
     own coordinates (so an image can be analysed under either the
     "nearest-tradition" link or the "image's own biome polygon" link).

Outputs (under dataset/imagery/inaturalist/):
  - manifest.parquet        : one row per (observation, photo)
  - per_tradition_stats.csv : how many hits per tradition + bbox used
  - sample_log.txt          : run log

Usage:
    python inat_bulk_sample.py                       # pilot: 50/tradition
    python inat_bulk_sample.py --per-tradition 200   # scale up
    python inat_bulk_sample.py --resume              # skip already-sampled

Rate limit: iNat asks for <= 60 req/min for unauthenticated use; we use 1.5s
spacing (≈40 req/min) by default, so the 958-tradition pilot takes ~24 minutes.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import argparse, json, time, sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]  # project root
SPINE = ROOT / "dataset/mapping_v2/traditions.parquet"
OUT = ROOT / "dataset/imagery/inaturalist"
OUT.mkdir(parents=True, exist_ok=True)

INAT = "https://api.inaturalist.org/v1/observations"
UA = {"User-Agent": "pareidolia-myth-research/0.3"}

# Try ever-larger bboxes until we have enough hits.
BBOX_LADDER_DEG = [0.5, 1.0, 2.0, 5.0]


def query_bbox(lat, lon, half_deg, per_page=200, max_retries=5):
    p = {
        "swlat": lat - half_deg, "swlng": lon - half_deg,
        "nelat": lat + half_deg, "nelng": lon + half_deg,
        "quality_grade": "research",
        "photo_license": "cc0,cc-by,cc-by-nc",
        "geoprivacy": "open",
        "per_page": per_page,
        "order_by": "random",
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(INAT, params=p, headers=UA, timeout=60)
            if r.status_code == 429:  # rate limited
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            time.sleep(2 ** attempt)  # 1, 2, 4, 8, 16 s
    raise last_err


def sample_tradition(oid, lat, lon, per_tradition, sleep_s):
    """Returns (rows, bbox_deg_used, total_available)."""
    last = None
    for half in BBOX_LADDER_DEG:
        time.sleep(sleep_s)
        try:
            data = query_bbox(lat, lon, half)
        except requests.HTTPError as e:
            return [], half, f"http_error:{e.response.status_code}"
        except Exception as e:
            # Last-resort: log and skip this tradition rather than crashing the whole run
            return [], half, f"error:{type(e).__name__}"
        last = data
        # If we already have enough candidates, stop expanding
        if data.get("total_results", 0) >= per_tradition or half == BBOX_LADDER_DEG[-1]:
            break

    if last is None:
        return [], BBOX_LADDER_DEG[0], "empty"

    rows = []
    for obs in last.get("results", [])[:per_tradition]:
        photos = obs.get("photos") or []
        if not photos:
            continue
        p = photos[0]
        url_sq = p.get("url") or ""
        # iNat exposes square/small/medium/large/original variants by URL substitution
        coords = (obs.get("geojson") or {}).get("coordinates") or [None, None]
        rows.append({
            "tradition_oid": oid,
            "inat_obs_id": obs.get("id"),
            "inat_obs_uuid": obs.get("uuid"),
            "taxon": (obs.get("taxon") or {}).get("name"),
            "taxon_rank": (obs.get("taxon") or {}).get("rank"),
            "iconic_taxon": (obs.get("taxon") or {}).get("iconic_taxon_name"),
            "lat": coords[1],
            "lon": coords[0],
            "place_guess": obs.get("place_guess"),
            "observed_on": obs.get("observed_on"),
            "photo_id": p.get("id"),
            "photo_url_square": url_sq,
            "photo_url_medium": url_sq.replace("/square.", "/medium."),
            "photo_url_large": url_sq.replace("/square.", "/large."),
            "license": p.get("license_code"),
            "attribution": p.get("attribution"),
        })
    return rows, half, last.get("total_results", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tradition", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit-traditions", type=int, default=None)
    args = ap.parse_args()

    trad = pd.read_parquet(SPINE)
    if args.limit_traditions:
        trad = trad.head(args.limit_traditions)

    manifest_path = OUT / "manifest.parquet"
    stats_path = OUT / "per_tradition_stats.csv"

    done = set()
    rows_all = []
    stats_all = []
    if args.resume and manifest_path.exists():
        prev = pd.read_parquet(manifest_path)
        done = set(prev["tradition_oid"].unique())
        rows_all = prev.to_dict("records")
        if stats_path.exists():
            stats_all = pd.read_csv(stats_path).to_dict("records")
        print(f"[resume] {len(done)} traditions already sampled, {len(rows_all):,} photo rows")

    todo = trad[~trad["oid"].isin(done)]
    print(f"[sample] {len(todo)} / {len(trad)} traditions to sample "
          f"(per_tradition={args.per_tradition}, sleep={args.sleep}s)")

    t_start = time.time()
    n_since_save = 0
    try:
        for _, t in tqdm(todo.iterrows(), total=len(todo), desc="traditions"):
            rows, half_used, total = sample_tradition(
                t["oid"], t["lat"], t["lon"], args.per_tradition, args.sleep
            )
            rows_all.extend(rows)
            stats_all.append({
                "tradition_oid": t["oid"],
                "group_Berezkin": t["group_Berezkin"],
                "lat": t["lat"], "lon": t["lon"],
                "biome_wwf": t.get("biome_wwf"),
                "ecoregion": t.get("ecoregion"),
                "bbox_half_deg_used": half_used,
                "total_available_in_final_bbox": total,
                "n_sampled": len(rows),
            })
            n_since_save += 1
            if n_since_save >= 25:
                pd.DataFrame(rows_all).to_parquet(manifest_path, index=False)
                pd.DataFrame(stats_all).to_csv(stats_path, index=False)
                n_since_save = 0
    except KeyboardInterrupt:
        print("\n[interrupt] saving partial results...")

    # Final save
    mdf = pd.DataFrame(rows_all)
    sdf = pd.DataFrame(stats_all)
    mdf.to_parquet(manifest_path, index=False)
    sdf.to_csv(stats_path, index=False)

    elapsed = time.time() - t_start
    print(f"\n[done] {len(mdf):,} photo rows from {sdf['tradition_oid'].nunique()} traditions "
          f"in {elapsed/60:.1f} min")
    print(f"  manifest -> {manifest_path}")
    print(f"  stats    -> {stats_path}")
    print()
    print("=== Per-bbox-used summary ===")
    print(sdf["bbox_half_deg_used"].value_counts().sort_index().to_string())
    print()
    print("=== Coverage ===")
    print(f"  traditions with >= per_tradition: {(sdf['n_sampled'] >= args.per_tradition).sum()}")
    print(f"  traditions with 0 sampled:        {(sdf['n_sampled'] == 0).sum()}")
    print()
    if not mdf.empty:
        # join biome via the tradition
        merged = mdf.merge(
            trad[["oid","biome_wwf"]].rename(columns={"oid":"tradition_oid"}),
            on="tradition_oid", how="left",
        )
        print("Photos per WWF biome (via tradition):")
        print(merged["biome_wwf"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
