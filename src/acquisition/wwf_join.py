"""
wwf_join.py — The REAL biome assignment, for when you have the shapefile.

This supersedes the Köppen proxy in add_koppen_biomes.py. It does a true
point-in-polygon spatial join of each Berezkin tradition against the WWF
Terrestrial Ecoregions of the World (Olson et al. 2001) or the updated
Ecoregions 2017 layer.

Get the shapefile (we could not download it from the sandbox; you can):
  - Ecoregions 2017 (846 ecoregions, CC-BY 4.0):
        https://ecoregions.appspot.com/   -> "Shapefile (150mb zip)"
        or https://storage.googleapis.com/teow2016/Ecoregions2017.zip
  - Original WWF TEOW (825/867 ecoregions, Olson 2001):
        http://assets.worldwildlife.org/publications/15/files/original/official_teow.zip

Unzip somewhere and point --shp at the .shp file.

Field names differ between the two layers:
  - Ecoregions2017.shp : BIOME_NAME, ECO_NAME, REALM
  - wwf_terr_ecos.shp   : BIOME (integer 1-14), ECO_NAME, REALM

This script auto-detects which it is.

Usage:
    pip install geopandas shapely pandas pyarrow
    python wwf_join.py --shp /path/to/Ecoregions2017.shp

Dependencies: geopandas, shapely, pandas, pyarrow
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import argparse
from pathlib import Path
import pandas as pd

OUT = Path("dataset/mapping")

# WWF biome integer codes (original TEOW) -> names
WWF_BIOME_CODES = {
    1: "Tropical & Subtropical Moist Broadleaf Forests",
    2: "Tropical & Subtropical Dry Broadleaf Forests",
    3: "Tropical & Subtropical Coniferous Forests",
    4: "Temperate Broadleaf & Mixed Forests",
    5: "Temperate Conifer Forests",
    6: "Boreal Forests/Taiga",
    7: "Tropical & Subtropical Grasslands, Savannas & Shrublands",
    8: "Temperate Grasslands, Savannas & Shrublands",
    9: "Flooded Grasslands & Savannas",
    10: "Montane Grasslands & Shrublands",
    11: "Tundra",
    12: "Mediterranean Forests, Woodlands & Scrub",
    13: "Deserts & Xeric Shrublands",
    14: "Mangroves",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", required=True, help="path to the WWF ecoregions .shp")
    ap.add_argument("--snap-km", type=float, default=50.0,
                    help="if a point misses all polygons (e.g. coastal), snap to "
                         "nearest polygon within this many km")
    args = ap.parse_args()

    import geopandas as gpd
    from shapely.geometry import Point

    trad = pd.read_parquet(OUT / "traditions.parquet")
    eco = gpd.read_file(args.shp).to_crs("EPSG:4326")

    cols = {c.upper(): c for c in eco.columns}
    # Detect layer variant
    if "BIOME_NAME" in cols:
        biome_col = cols["BIOME_NAME"]
        eco["_biome"] = eco[biome_col]
    elif "BIOME" in cols:
        biome_col = cols["BIOME"]
        eco["_biome"] = eco[biome_col].map(WWF_BIOME_CODES).fillna(
            eco[biome_col].astype(str))
    else:
        raise SystemExit(f"Could not find a biome column in {list(eco.columns)}")

    eco_name = cols.get("ECO_NAME")
    realm = cols.get("REALM")
    keep = ["_biome", "geometry"]
    if eco_name:
        keep.insert(1, eco_name)
    if realm:
        keep.insert(1, realm)
    eco = eco[keep]

    pts = gpd.GeoDataFrame(
        trad.copy(),
        geometry=[Point(xy) for xy in zip(trad["lon"], trad["lat"])],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(pts, eco, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    # Snap the misses (coastal points etc.) to nearest polygon
    missing = joined["_biome"].isna()
    n_missing = int(missing.sum())
    if n_missing and args.snap_km > 0:
        print(f"snapping {n_missing} unmatched points to nearest ecoregion ...")
        eco_proj = eco.to_crs("EPSG:6933")  # equal-area meters
        for idx in joined[missing].index:
            p = Point(trad.loc[idx, "lon"], trad.loc[idx, "lat"])
            p_proj = gpd.GeoSeries([p], crs="EPSG:4326").to_crs("EPSG:6933").iloc[0]
            dists = eco_proj.distance(p_proj)
            j = dists.idxmin()
            if dists[j] <= args.snap_km * 1000:
                joined.loc[idx, "_biome"] = eco.loc[j, "_biome"]
                if eco_name:
                    joined.loc[idx, eco_name] = eco.loc[j, eco_name]
                if realm:
                    joined.loc[idx, realm] = eco.loc[j, realm]

    out = trad.copy()
    out["biome_wwf"] = joined["_biome"].values
    if eco_name:
        out["ecoregion"] = joined[eco_name].values
    if realm:
        out["realm"] = joined[realm].values
    out.to_parquet(OUT / "traditions.parquet", index=False)

    print("WWF biome distribution:")
    print(out["biome_wwf"].value_counts(dropna=False).to_string())
    print(f"\nUpdated {OUT/'traditions.parquet'} with biome_wwf"
          + (", ecoregion" if eco_name else "")
          + (", realm" if realm else ""))
    print("Re-run make_preliminary_figures.py with BIOME_COL='biome_wwf' "
          "to regenerate figures on the true ecoregion biomes.")


if __name__ == "__main__":
    main()
