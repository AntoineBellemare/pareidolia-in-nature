"""
build_spine.py — Build the geographic/mythological SPINE of the dataset
from the Berezkin Analytical Catalogue (obtained from the macleginn/
mythology-queries GitHub repo).

This is the join layer that was missing: it links every mythological
tradition to (a) geographic coordinates, (b) the motifs/mythemes present
in it, and (c) a coarse biome class. Everything else in the project hangs
off this.

Inputs (from mythology-queries/data/):
    coords.json              926 traditions with lat/lon
    traditions.json          926 traditions -> 2138-dim presence/absence vector
    motif_distributions.json 2138 motifs -> 926-dim presence/absence vector
    motif_list.json          ordered list of 2138 motif IDs
    new_descriptions.json    motif ID -> {name, description}

Outputs (to dataset/mapping/):
    traditions.parquet       tidy: tradition, lat, lon, biome, n_motifs
    motifs.parquet           tidy: motif_id, name, description, n_traditions
    tradition_motif.parquet  long: (tradition, motif_id) present pairs
    spine_summary.md         human-readable summary

Dependencies: pandas, pyarrow, numpy
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils


import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("mythology-queries/data")
OUT = Path("dataset/mapping")
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Coarse biome classifier from lat/lon.
# This is a FIRST-APPROXIMATION stand-in for a proper WWF-ecoregion point-in-
# polygon join. It uses latitude bands + a few longitude/aridity heuristics.
# Replace with a real shapefile intersection (geopandas + WWF TEOW) for the
# full study — see notes in spine_summary.md.
# --------------------------------------------------------------------------- #

def coarse_biome(lat: float, lon: float) -> str:
    a = abs(lat)
    if a >= 66.5:
        return "polar / tundra"
    if a >= 55:
        return "boreal / subarctic"
    if a >= 40:
        return "temperate"
    if a >= 23.5:
        # subtropical — flag the big desert belts crudely
        # Sahara/Arabia/Thar and SW US/N Mexico and Australian interior
        if (15 <= lon <= 60 and 15 <= lat <= 33) or \
           (-115 <= lon <= -100 and 25 <= lat <= 35) or \
           (120 <= lon <= 140 and -30 <= lat <= -20):
            return "subtropical desert"
        return "subtropical"
    # tropics
    return "tropical"


def main():
    coords = json.load(open(SRC / "coords.json"))
    traditions_vec = json.load(open(SRC / "traditions.json"))
    motif_dist = json.load(open(SRC / "motif_distributions.json"))
    motif_list = json.load(open(SRC / "motif_list.json"))
    descriptions = json.load(open(SRC / "new_descriptions.json"))

    # coords is index-aligned with the columns of motif_distributions
    # and with the rows order used throughout. Build the canonical order.
    coord_names = [c["Name"] for c in coords]
    n_trad = len(coord_names)
    n_motif = len(motif_list)

    assert len(motif_dist[motif_list[0]]) == n_trad, "motif vec != n traditions"
    assert len(traditions_vec[list(traditions_vec)[0]]) == n_motif, \
        "tradition vec != n motifs"

    # --- traditions table ---
    trad_rows = []
    for i, c in enumerate(coords):
        name = c["Name"]
        vec = traditions_vec.get(name)
        n_present = int(sum(vec)) if vec else None
        trad_rows.append({
            "tradition": name,
            "tradition_idx": i,
            "lat": c["Latitude"],
            "lon": c["Longitude"],
            "biome": coarse_biome(c["Latitude"], c["Longitude"]),
            "n_motifs": n_present,
        })
    trad_df = pd.DataFrame(trad_rows)
    trad_df.to_parquet(OUT / "traditions.parquet", index=False)

    # --- motifs table ---
    motif_rows = []
    for mid in motif_list:
        # motif ids in motif_list have a trailing _N variant suffix; the
        # descriptions are keyed by the base id (strip the _N)
        base = mid.rsplit("_", 1)[0]
        desc = descriptions.get(base) or descriptions.get(mid) or {}
        dist = motif_dist.get(mid, [])
        motif_rows.append({
            "motif_id": mid,
            "motif_base": base,
            "name": desc.get("name"),
            "description": desc.get("description"),
            "n_traditions": int(sum(dist)) if dist else 0,
        })
    motif_df = pd.DataFrame(motif_rows)
    motif_df.to_parquet(OUT / "motifs.parquet", index=False)

    # --- long-format tradition x motif (present pairs only) ---
    pairs = []
    motif_idx = {m: j for j, m in enumerate(motif_list)}
    for name, vec in traditions_vec.items():
        if name not in set(coord_names):
            continue
        for j, present in enumerate(vec):
            if present:
                pairs.append((name, motif_list[j]))
    pair_df = pd.DataFrame(pairs, columns=["tradition", "motif_id"])
    pair_df.to_parquet(OUT / "tradition_motif.parquet", index=False)

    # --- summary ---
    biome_counts = trad_df["biome"].value_counts()
    top_motifs = motif_df.sort_values("n_traditions", ascending=False).head(10)
    rare_named = motif_df[motif_df["name"].notna()].sort_values("n_traditions").head(10)

    lines = [
        "# Berezkin spine — build summary",
        "",
        f"- Traditions (geo-located): **{n_trad}**",
        f"- Motifs / mythemes: **{n_motif}**",
        f"- Present (tradition, motif) pairs: **{len(pair_df):,}**",
        f"- Matrix density: **{len(pair_df) / (n_trad * n_motif):.1%}**",
        "",
        "## Traditions by coarse biome",
        "",
        "| biome | n traditions |",
        "|---|---:|",
    ]
    for b, n in biome_counts.items():
        lines.append(f"| {b} | {n} |")
    lines += [
        "",
        "## 10 most widespread motifs (candidate universals)",
        "",
        "| motif | name | in N traditions |",
        "|---|---|---:|",
    ]
    for _, r in top_motifs.iterrows():
        lines.append(f"| {r['motif_id']} | {r['name'] or '—'} | {r['n_traditions']} |")
    lines += [
        "",
        "## Note on biome assignment",
        "",
        "The `biome` column here is a coarse latitude/longitude heuristic, NOT a",
        "real ecoregion join. For the actual study, replace `coarse_biome()` with a",
        "point-in-polygon intersection against the WWF Terrestrial Ecoregions",
        "shapefile (Olson et al. 2001) using geopandas:",
        "",
        "```python",
        "import geopandas as gpd",
        "eco = gpd.read_file('wwf_terr_ecos.shp')",
        "pts = gpd.GeoDataFrame(trad_df,",
        "        geometry=gpd.points_from_xy(trad_df.lon, trad_df.lat), crs='EPSG:4326')",
        "joined = gpd.sjoin(pts, eco[['BIOME','ECO_NAME','REALM','geometry']],",
        "                   how='left', predicate='within')",
        "```",
        "",
        "That gives each tradition its true biome (14 classes), ecoregion (825),",
        "and biogeographic realm (8) — the proper spatial spine for pairing",
        "mythologies with landscape imagery.",
        "",
        "## How this connects to the rest of the project",
        "",
        "- `traditions.parquet` is what you sample landscape imagery against:",
        "  for any tradition you now have lat/lon and biome, so you can pull",
        "  iNaturalist/YFCC images from that region.",
        "- `tradition_motif.parquet` tells you WHICH mythemes to look for in each",
        "  region — these are the targets the vision model should reconstruct.",
        "- `motifs.parquet` gives you the natural-language description of each",
        "  motif, which is exactly what you embed (Test A in embed_and_analyze.py).",
        "  e.g. a motif like 'Rainbow is a snake' is a directly pareidolic mytheme",
        "  you can hunt for in sky/landscape imagery.",
    ]
    (OUT / "spine_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Built spine: {n_trad} traditions, {n_motif} motifs, "
          f"{len(pair_df):,} pairs")
    print(f"Outputs in {OUT}/")
    return trad_df, motif_df, pair_df


if __name__ == "__main__":
    main()
