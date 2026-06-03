"""
build_unified_manifest.py — concatenate per-source image manifests into a
single source-agnostic manifest with the schema in UNIFIED_SCHEMA.md.

Inputs (any subset that exists):
  dataset/imagery/inaturalist/manifest.parquet
  dataset/imagery/yfcc100m/manifest.parquet
  dataset/imagery/commons/manifest.parquet

Output:
  dataset/imagery/manifest.parquet

Each row is one (source, source_id) image. The output preserves the original
row order per source so existing embeddings stay aligned via `embed_idx`.

Usage:
    python build_unified_manifest.py                   # build from all sources
    python build_unified_manifest.py --sources inaturalist yfcc100m
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # project root
IMG = ROOT / "dataset/imagery"
OUT = IMG / "manifest.parquet"

SCHEMA = [
    "image_id", "source", "source_id",
    "lat", "lon", "local_path", "photo_url", "license",
    "text_tags", "iconic_taxon", "caption",
    "photo_biome_wwf", "photo_ecoregion", "photo_realm",
    "tradition_oid", "tradition_distance_km", "tradition_biome_wwf",
    "embed_idx",
]


def _ensure_columns(df, fill_value=None):
    out = pd.DataFrame()
    for c in SCHEMA:
        if c in df.columns:
            out[c] = df[c]
        else:
            out[c] = fill_value
    return out


def from_inaturalist(path: Path) -> pd.DataFrame:
    m = pd.read_parquet(path)
    df = pd.DataFrame()
    df["image_id"] = "inat_" + m["photo_id"].astype("Int64").astype(str)
    df["source"] = "inaturalist"
    df["source_id"] = m["photo_id"].astype("Int64").astype(str)
    df["lat"] = m["lat"]
    df["lon"] = m["lon"]
    df["local_path"] = m["local_path"]
    df["photo_url"] = m["photo_url_medium"]
    df["license"] = m["license"]
    df["text_tags"] = m["taxon"]
    df["iconic_taxon"] = m["iconic_taxon"]
    df["caption"] = m["place_guess"]
    df["photo_biome_wwf"] = m.get("photo_biome_wwf")
    df["photo_ecoregion"] = m.get("photo_ecoregion")
    df["photo_realm"] = m.get("photo_realm")
    df["tradition_oid"] = m["tradition_oid"].astype("Int64")
    df["tradition_distance_km"] = np.nan  # bbox-sampled, distance not meaningful
    df["tradition_biome_wwf"] = m.get("tradition_biome_wwf")
    # embed_idx is the row index in the inat-specific img_emb.npy
    df["embed_idx"] = np.arange(len(df), dtype=np.int64)
    return _ensure_columns(df)


def from_yfcc100m(path: Path) -> pd.DataFrame:
    m = pd.read_parquet(path)
    df = pd.DataFrame()
    df["image_id"] = "yfcc_" + m["photo_id"].astype(str)
    df["source"] = "yfcc100m"
    df["source_id"] = m["photo_id"].astype(str)
    df["lat"] = m["lat"]
    df["lon"] = m["lon"]
    df["local_path"] = m["local_path"]
    df["photo_url"] = m["photo_url"]
    df["license"] = m["license"]
    df["text_tags"] = m.get("user_tags")
    df["iconic_taxon"] = None
    df["caption"] = m.get("title")
    df["photo_biome_wwf"] = m.get("photo_biome_wwf")
    df["photo_ecoregion"] = m.get("photo_ecoregion")
    df["photo_realm"] = m.get("photo_realm")
    df["tradition_oid"] = m.get("tradition_oid")
    df["tradition_distance_km"] = m.get("tradition_distance_km")
    df["tradition_biome_wwf"] = m.get("tradition_biome_wwf")
    df["embed_idx"] = np.arange(len(df), dtype=np.int64)
    return _ensure_columns(df)


READERS = {
    "inaturalist": (IMG / "inaturalist/manifest.parquet", from_inaturalist),
    "yfcc100m":    (IMG / "yfcc100m/manifest.parquet",    from_yfcc100m),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(READERS),
                    help="which sources to include")
    args = ap.parse_args()

    pieces = []
    cursor = 0  # global embed_idx offset per source — only used if multiple
                # sources share a single embedding npy. For now each source has
                # its own embeddings, so we keep per-source 0..N indexing.
    for src in args.sources:
        if src not in READERS:
            print(f"[unified] unknown source: {src}, skipping")
            continue
        path, reader = READERS[src]
        if not path.exists():
            print(f"[unified] {src}: manifest not found at {path}, skipping")
            continue
        df = reader(path)
        print(f"[unified] {src}: {len(df):,} rows")
        pieces.append(df)
    if not pieces:
        print("[unified] no sources, nothing to write")
        return
    unified = pd.concat(pieces, ignore_index=True)
    unified.to_parquet(OUT, index=False)
    print(f"\n[unified] wrote {OUT}  ({len(unified):,} rows)")
    print("\nsource breakdown:")
    print(unified["source"].value_counts().to_string())
    print("\nphoto_biome_wwf coverage:")
    print(unified["photo_biome_wwf"].notna().sum(), "/", len(unified))


if __name__ == "__main__":
    main()
