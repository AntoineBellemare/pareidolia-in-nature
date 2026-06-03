"""
pull_places365_biomes.py — fetch Places365 validation images, keep only
biome-relevant categories, build a clean landscape manifest.

Places365 validation set: 36,500 images (100 per category) at 256×256.
URL: http://data.csail.mit.edu/places/places365/val_256.tar (~620 MB)
Labels: places365_val.txt with `image_path category_idx` per line.

Mapping ~13 biome-relevant scene categories to WWF biomes gives us
~1,300 high-quality curated scene photos. Each biome ends up with
~100-300 images depending on how many Places365 categories back it.

Outputs:
  dataset/imagery/places365/images/  (only biome-relevant images kept)
  dataset/imagery/places365/img_paths.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import os, time, tarfile, shutil
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]  # project root
DEST = ROOT / "dataset/imagery/places365"
DEST.mkdir(parents=True, exist_ok=True)
IMG_DIR = DEST / "images"; IMG_DIR.mkdir(exist_ok=True)
RAW = DEST / "raw"; RAW.mkdir(exist_ok=True)

CATEGORIES_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
VAL_TAR_URL    = "http://data.csail.mit.edu/places/places365/val_256.tar"
# Val labels live inside the official filelist tar (small download)
FILELIST_TAR_URL = "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"


# WWF biome ← Places365 scene categories (curated mapping)
# IMPORTANT: only Places365 category names that actually exist in
# the 365-category vocabulary. Removed: "forest/needleleaf" (DOESN'T
# EXIST — silently dropped 3 biomes); "savanna" (DOESN'T EXIST —
# silently dropped Trop Grasslands).
BIOME_CATEGORY_MAP = {
    "Tropical & Subtropical Moist Broadleaf Forests": [
        "rainforest", "bamboo_forest",
    ],
    "Tropical & Subtropical Dry Broadleaf Forests": [
        "forest/broadleaf", "tree_farm",
    ],
    "Temperate Broadleaf & Mixed Forests": [
        "forest/broadleaf", "forest_path",
    ],
    "Temperate Grasslands, Savannas & Shrublands": [
        "field/wild", "pasture", "hayfield",
    ],
    "Mediterranean Forests, Woodlands & Scrub": [
        "vineyard", "orchard", "field/cultivated",
    ],
    "Montane Grasslands & Shrublands": [
        "mountain", "mountain_path",
    ],
    # Boreal: no native "needleleaf" category; closest visual analog
    # is snowy conifer landscape and high-latitude wooded scenes.
    "Boreal Forests/Taiga": [
        "snowfield", "mountain_snowy", "forest_path",
    ],
    "Tundra": [
        "tundra", "glacier", "iceberg",
    ],
    "Deserts & Xeric Shrublands": [
        "desert/sand", "desert/vegetation", "desert_road", "badlands",
    ],
    "Flooded Grasslands & Savannas": [
        "swamp", "marsh",
    ],
    # Mangroves: only the two least-off-topic proxies. Drop "creek"
    # (catches rocky temperate streams) and "coast" (catches sea-cliffs).
    "Mangroves": [
        "lagoon", "swamp",
    ],
    # Biomes with no satisfactory Places365 category are intentionally
    # absent: Tropical/Subtropical Coniferous Forests, Temperate Conifer
    # Forests, and Tropical/Subtropical Grasslands & Savannas. iNat
    # covers all 14 biomes; the Places365 panel is a scene-level
    # robustness check on biomes for which the corpus has native
    # vocabulary.
}


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name}")
        return dest
    print(f"  downloading {url}")
    urlretrieve(url, dest)
    return dest


def parse_categories(p: Path) -> dict[str, int]:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        path, idx = line.rsplit(" ", 1)
        name = path.lstrip("/")
        # strip leading letter directory: /a/abbey -> abbey,
        # /d/desert/sand -> desert/sand
        name_after = "/".join(name.split("/")[1:])
        out[name_after] = int(idx)
    return out


def parse_val_labels(p: Path) -> list[tuple[str, int]]:
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        path, idx = line.rsplit(" ", 1)
        rows.append((path.lstrip("/"), int(idx)))
    return rows


def main():
    print("Loading Places365 metadata …")
    cat_file = fetch(CATEGORIES_URL, RAW / "categories_places365.txt")
    # Download filelist tar (small) and extract places365_val.txt
    filelist_tar = RAW / "filelist_places365-standard.tar"
    fetch(FILELIST_TAR_URL, filelist_tar)
    val_file = RAW / "places365_val.txt"
    if not val_file.exists():
        print(f"  extracting places365_val.txt from filelist tar ...")
        with tarfile.open(filelist_tar) as tf:
            for m in tf.getmembers():
                if m.name.endswith("places365_val.txt"):
                    with tf.extractfile(m) as fsrc:
                        val_file.write_bytes(fsrc.read())
                    break
    cat_idx = parse_categories(cat_file)
    val_labels = parse_val_labels(val_file)
    print(f"  {len(cat_idx)} scene categories")
    print(f"  {len(val_labels):,} val images")

    # Build target idx set: indices of categories we want
    print("\nBiome → category mapping (with availability):")
    target_idx_to_biomes = {}  # idx → list of biomes claiming it
    for biome, names in BIOME_CATEGORY_MAP.items():
        for nm in names:
            if nm not in cat_idx:
                print(f"  {biome[:48]:48s}  ✗ MISSING category: {nm}")
                continue
            i = cat_idx[nm]
            target_idx_to_biomes.setdefault(i, []).append((biome, nm))
            print(f"  {biome[:48]:48s}  ✓ {nm:25s}  idx={i}")

    # Plan: which val images do we want?
    wanted = []
    for path, idx in val_labels:
        if idx in target_idx_to_biomes:
            for biome, nm in target_idx_to_biomes[idx]:
                wanted.append({"sub_path": path, "places_idx": idx,
                               "places_name": nm,
                               "photo_biome_wwf": biome})
    plan = pd.DataFrame(wanted)
    print(f"\nPlanned: {len(plan)} (image × biome) rows  "
          f"({plan['sub_path'].nunique()} unique images)")
    print(plan.groupby("photo_biome_wwf").size().to_string())

    # Download val tar (~620 MB)
    val_tar = RAW / "val_256.tar"
    print(f"\nDownloading val_256.tar (~620 MB) ...")
    fetch(VAL_TAR_URL, val_tar)

    # Extract ONLY the wanted images from the tar
    print("\nExtracting wanted images from val_256.tar ...")
    wanted_set = set(plan["sub_path"].unique())
    extracted = {}
    with tarfile.open(val_tar) as tf:
        for member in tqdm(tf.getmembers(), desc="scan"):
            # tar entries are like "val_256/Places365_val_NNNNNNNN.jpg"
            mname = member.name
            base = mname.split("/", 1)[-1] if "/" in mname else mname
            if base in wanted_set:
                with tf.extractfile(member) as fsrc:
                    if fsrc is None: continue
                    local = IMG_DIR / base
                    local.write_bytes(fsrc.read())
                    extracted[base] = str(local)
    plan["local_path"] = plan["sub_path"].map(extracted)
    plan_ok = plan[plan["local_path"].notna()].reset_index(drop=True)
    print(f"\nExtracted {plan_ok['sub_path'].nunique()} images, "
          f"yielding {len(plan_ok)} (image × biome) rows")
    print("Per biome:")
    print(plan_ok.groupby("photo_biome_wwf").size().to_string())

    # Compatible manifest
    plan_ok["tradition_biome_wwf"] = plan_ok["photo_biome_wwf"]
    plan_ok["iconic_taxon"] = "landscape"
    out_p = DEST / "img_paths.parquet"
    plan_ok.to_parquet(out_p, index=False)
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
