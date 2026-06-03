"""
yfcc_tag_and_link.py — for the YFCC manifest:
  1. WWF point-in-polygon for each photo's coords -> photo_biome_wwf, photo_ecoregion, photo_realm.
  2. Nearest-tradition assignment by geodesic distance using a vectorized
     haversine over the 958 tradition centroids -> tradition_oid, tradition_distance_km,
     tradition_biome_wwf.

Mutates dataset/imagery/yfcc100m/manifest.parquet in place.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]  # project root
MAP = ROOT / "dataset/mapping_v2"
YF = ROOT / "dataset/imagery/yfcc100m"
SHP = ROOT / "raw_downloads/Ecoregions2017/Ecoregions2017.shp"


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized: lat1/lon1 scalar; lat2/lon2 arrays. Degrees in. km out."""
    R = 6371.0
    rad = np.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = (np.sin(dlat/2)**2
         + np.cos(lat1*rad) * np.cos(lat2*rad) * np.sin(dlon/2)**2)
    return 2*R*np.arcsin(np.sqrt(a))


def main():
    m = pd.read_parquet(YF / "manifest.parquet")
    print(f"[yfcc-tag] manifest: {len(m):,} photos")
    trad = pd.read_parquet(MAP / "traditions.parquet")
    print(f"[yfcc-tag] traditions: {len(trad)}")

    # ----- 1) WWF spatial join -----
    print("[yfcc-tag] loading WWF shapefile ...")
    eco = gpd.read_file(SHP).to_crs("EPSG:4326")
    keep = [c for c in ["BIOME_NAME","ECO_NAME","REALM"] if c in eco.columns] + ["geometry"]
    eco = eco[keep]

    good = m["lat"].notna() & m["lon"].notna()
    pts = gpd.GeoDataFrame(
        m.loc[good, ["lat","lon"]].copy(),
        geometry=[Point(xy) for xy in zip(m.loc[good, "lon"], m.loc[good, "lat"])],
        crs="EPSG:4326",
    )
    print(f"[yfcc-tag] sjoin {len(pts):,} points against {len(eco)} polygons ...")
    joined = gpd.sjoin(pts, eco, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    m["photo_biome_wwf"] = pd.NA
    m["photo_ecoregion"] = pd.NA
    m["photo_realm"]    = pd.NA
    m.loc[good, "photo_biome_wwf"] = joined["BIOME_NAME"].values
    if "ECO_NAME" in joined.columns:
        m.loc[good, "photo_ecoregion"] = joined["ECO_NAME"].values
    if "REALM" in joined.columns:
        m.loc[good, "photo_realm"] = joined["REALM"].values

    # ----- 2) Nearest tradition by haversine -----
    print(f"[yfcc-tag] assigning nearest of {len(trad)} traditions ...")
    trad_lat = trad["lat"].to_numpy()
    trad_lon = trad["lon"].to_numpy()
    trad_oid = trad["oid"].to_numpy()
    trad_biome = trad["biome_wwf"].to_numpy()

    nearest_oid = []
    nearest_dist = []
    nearest_biome = []
    for lat, lon in zip(m["lat"].to_numpy(), m["lon"].to_numpy()):
        if np.isnan(lat) or np.isnan(lon):
            nearest_oid.append(pd.NA); nearest_dist.append(np.nan); nearest_biome.append(pd.NA)
            continue
        d = haversine_km(lat, lon, trad_lat, trad_lon)
        i = int(np.argmin(d))
        nearest_oid.append(int(trad_oid[i]))
        nearest_dist.append(float(d[i]))
        nearest_biome.append(trad_biome[i])
    m["tradition_oid"] = nearest_oid
    m["tradition_distance_km"] = nearest_dist
    m["tradition_biome_wwf"] = nearest_biome

    out = YF / "manifest.parquet"
    m.to_parquet(out, index=False)
    print(f"\n[yfcc-tag] saved {out}")
    print(f"  photos w/ photo_biome_wwf: {m['photo_biome_wwf'].notna().sum():,}")
    print(f"  photos w/ tradition_oid:   {m['tradition_oid'].notna().sum():,}")
    print(f"  mean tradition distance:   {pd.Series(nearest_dist).mean():.0f} km")
    print("\nphoto_biome_wwf distribution:")
    print(m["photo_biome_wwf"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
