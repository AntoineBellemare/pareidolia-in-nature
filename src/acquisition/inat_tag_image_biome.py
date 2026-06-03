"""
inat_tag_image_biome.py — add image-biome columns to the iNat manifest.

After inat_bulk_sample.py runs, the manifest has each photo's lat/lon (the
observation's actual coords, not the tradition centroid). This script does a
point-in-polygon join against the WWF Ecoregions2017 shapefile and writes
back two new columns:
  - photo_biome_wwf  : the biome of the photo's own coordinates
  - photo_ecoregion  : the ecoregion of the photo's own coordinates
  - photo_realm      : the realm of the photo's own coordinates

It also adds:
  - tradition_biome_wwf : copied from the tradition, for easy comparison
  - biome_match         : bool, photo_biome == tradition_biome

Run:
    python inat_tag_image_biome.py
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]  # project root
SPINE = ROOT / "dataset/mapping_v2/traditions.parquet"
SHP = ROOT / "raw_downloads/Ecoregions2017/Ecoregions2017.shp"
MANIFEST = ROOT / "dataset/imagery/inaturalist/manifest.parquet"


def main():
    m = pd.read_parquet(MANIFEST)
    trad = pd.read_parquet(SPINE)
    print(f"[tag] loaded {len(m):,} photos and {len(trad):,} traditions")

    # Inherit tradition_biome onto the photo (cheap dict lookup)
    t_biome = trad.set_index("oid")[["biome_wwf", "ecoregion"]].add_prefix("tradition_")
    m = m.merge(t_biome, left_on="tradition_oid", right_index=True, how="left")

    # WWF spatial join on each photo's actual coords
    print(f"[tag] loading WWF shapefile ...")
    eco = gpd.read_file(SHP).to_crs("EPSG:4326")
    cols = ["BIOME_NAME", "ECO_NAME", "REALM"]
    keep = [c for c in cols if c in eco.columns] + ["geometry"]
    eco = eco[keep]

    # only join photos with valid coords
    good = m["lat"].notna() & m["lon"].notna()
    print(f"[tag] photos with coords: {int(good.sum()):,} / {len(m):,}")

    pts = gpd.GeoDataFrame(
        m.loc[good, ["lat", "lon"]].copy(),
        geometry=[Point(xy) for xy in zip(m.loc[good, "lon"], m.loc[good, "lat"])],
        crs="EPSG:4326",
    )
    print(f"[tag] running spatial join ...")
    joined = gpd.sjoin(pts, eco, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    # Write back to m
    m["photo_biome_wwf"] = pd.NA
    m["photo_ecoregion"] = pd.NA
    m["photo_realm"] = pd.NA
    m.loc[good, "photo_biome_wwf"] = joined["BIOME_NAME"].values
    if "ECO_NAME" in joined.columns:
        m.loc[good, "photo_ecoregion"] = joined["ECO_NAME"].values
    if "REALM" in joined.columns:
        m.loc[good, "photo_realm"] = joined["REALM"].values

    m["biome_match"] = (m["photo_biome_wwf"] == m["tradition_biome_wwf"])

    m.to_parquet(MANIFEST, index=False)
    n_match = int(m["biome_match"].sum())
    print(f"\n[tag] photos in same biome as tradition: {n_match:,} / {len(m):,} "
          f"({100*n_match/max(len(m),1):.1f}%)")
    print("\nPhoto-biome distribution:")
    print(m["photo_biome_wwf"].value_counts(dropna=False).to_string())
    print("\nSaved → manifest.parquet (with photo_biome_wwf, photo_ecoregion, "
          "photo_realm, tradition_biome_wwf, tradition_ecoregion, biome_match)")


if __name__ == "__main__":
    main()
