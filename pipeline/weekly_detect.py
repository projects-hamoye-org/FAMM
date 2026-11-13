#!/usr/bin/env python3
"""
FAMM weekly detection, as described in the paper (Sec. 2.2, 2.3, 3.9).

For each FAO GAUL first-level region of Ghana, builds the deployment
Sentinel-2 composite (rolling 7/14/30-day medians, two-stage focal fill,
clipped to the region) in Earth Engine, pulls it tile by tile straight into
memory with computePixels, and scores 256 px windows at a 128 px stride,
resized to 224, with MobileNetV3-Large. Windows more than half invalid are
skipped; residual invalid pixels are zero-filled. Detections are attributed
to a region and district with geoBoundaries ADM1/ADM2 and written as GeoJSON
for validate_geojson.py.

Authenticates as whatever Google credentials are active (the service account
via WIF in GitHub Actions, or your own gcloud login locally). Nothing is
exported to Drive or Cloud Storage and no tile is kept on disk.

Usage:
    python pipeline/weekly_detect.py                       # all GAUL regions
    python pipeline/weekly_detect.py --regions "Western,Central"
"""

import argparse
import concurrent.futures as cf
import datetime
import json
import math
import sys
import time
from pathlib import Path

import ee
import geopandas as gpd
import google.auth
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyproj import Transformer
from shapely.geometry import Point, box, shape
from shapely.ops import transform as shp_transform
from shapely.prepared import prep
from torchvision import models

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT         = "famm-472015"
CHECKPOINT_PATH = "models/mobilenetv3_best.pth"
ADM1_PATH       = "deployment/geoBoundaries-GHA-ADM1.geojson"
ADM2_PATH       = "deployment/geoBoundaries-GHA-ADM2.geojson"
OUTPUT_PATH     = "asm_monitoring_results.geojson"   # validator picks this up

# ── Imagery settings (match deployment/ee_export.py) ─────────────────────────
COUNTRY = "Ghana"
GAUL    = "FAO/GAUL/2015/level1"
BANDS   = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
CRS     = "EPSG:32630"
SCALE   = 10

# ── Model settings (match deployment/deployment_inference.py) ────────────────
CROP_SIZE          = 256   # native window, 2.56 km
NETWORK_INPUT_SIZE = 224   # resized before the stem, as in training
STRIDE             = 128   # 50% overlap
THRESHOLD          = 0.3
MAX_INVALID_FRAC   = 0.5   # skip windows with more than half their pixels missing

# ── Tiling ────────────────────────────────────────────────────────────────────
# Tiles step by TILE px but each request reaches CROP_SIZE - STRIDE px further
# right and down, so overlapping windows run seamlessly across tile edges.
# 1024 px x 11 uint16 bands ≈ 23 MB, under the computePixels response limit.
TILE      = 7 * STRIDE
FETCH     = TILE + CROP_SIZE - STRIDE
TILE_M    = TILE * SCALE
WORKERS   = 8
RETRIES   = 5
MAX_FAILED_FRACTION = 0.05


# ── Earth Engine ──────────────────────────────────────────────────────────────
def init_ee():
    creds, _ = google.auth.default(scopes=[
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/cloud-platform",
    ])
    # High-volume endpoint is the one Google recommends for many parallel
    # automated computePixels requests.
    ee.Initialize(credentials=creds, project=PROJECT,
                  opt_url="https://earthengine-highvolume.googleapis.com")


def gaul_regions():
    return (ee.FeatureCollection(GAUL)
            .filter(ee.Filter.eq("ADM0_NAME", COUNTRY)))


def mask_s2(image, roi):
    """Cloud + NDVI mask, as in deployment/ee_export.py."""
    image_date = image.date()
    cloud_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(roi)
        .filterDate(image_date.advance(-3, "day"), image_date.advance(3, "day"))
        .sort("system:time_start")
    )
    cloud_mask = ee.Image(ee.Algorithms.If(
        cloud_col.size().gt(0),
        ee.Image(cloud_col.first()).select("probability").lt(60),
        ee.Image(1),
    ))
    scaled = image.divide(10000)
    ndvi_mask = scaled.normalizedDifference(["B8", "B4"]).gte(0.25)
    return scaled.updateMask(cloud_mask.Or(ndvi_mask)).copyProperties(
        image, ["system:time_start"]
    )


def build_composite(tile_roi, region_geom, today):
    """ee_export.py's composite for one tile, clipped to its GAUL region.

    Returns the 10 bands as uint16 reflectance x 10000 plus a 'valid' band,
    with masked pixels set to 0 so the array is dense; 'valid' says which
    pixels are real. Pixels outside the region are invalid, as in the
    region-clipped exports the paper describes.
    """
    end = ee.Date(str(today))

    def collection(days):
        start = ee.Date(str(today - datetime.timedelta(days=days)))
        return (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(tile_roi)
            .filterDate(start, end)
            .map(lambda img: mask_s2(img, tile_roi))
            .select(BANDS)
        )

    composite = (collection(7).median()
                 .unmask(collection(14).median())
                 .unmask(collection(30).median()))

    # 3x3 mean then 5x5 median, as the paper states. Radii are in pixels: the
    # earlier 'meters' units gave a 1-pixel kernel that filled nothing
    # (checked in .github/workflows/wif_smoke_test.yml).
    composite = composite.unmask(composite.focal_mean(1, "square", "pixels"))
    composite = composite.unmask(composite.focal_median(2, "square", "pixels"))
    composite = composite.clip(region_geom)

    valid = composite.mask().reduce(ee.Reducer.min()).gt(0).rename("valid")
    return (composite.multiply(10000).round()
            .addBands(valid)
            .toUint16()
            .unmask(0))


def fetch_tile(region, x0, y_top, today):
    fetch_m = FETCH * SCALE
    tile_roi = ee.Geometry.Rectangle([x0, y_top - fetch_m, x0 + fetch_m, y_top],
                                     CRS, False)
    region_geom = gaul_regions().filter(ee.Filter.eq("ADM1_NAME", region)).geometry()
    request = {
        "expression": build_composite(tile_roi, region_geom, today),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": FETCH, "height": FETCH},
            "affineTransform": {"scaleX": SCALE, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -SCALE, "translateY": y_top},
            "crsCode": CRS,
        },
    }
    for attempt in range(RETRIES):
        try:
            return ee.data.computePixels(request)
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(5 * 2 ** attempt)


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    if not Path(CHECKPOINT_PATH).exists():
        sys.exit(f"❌ Model checkpoint not found: {CHECKPOINT_PATH}")
    model = models.mobilenet_v3_large(weights=None)
    model.features[0][0] = nn.Conv2d(len(BANDS), 16, kernel_size=3, stride=2,
                                     padding=1, bias=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


# ── Geography ─────────────────────────────────────────────────────────────────
def region_outlines(region_filter):
    """GAUL region outlines in CRS, for choosing which tiles to request.

    Simplified and buffered by 1 km so no sliver of a region is missed; the
    exact clip happens in Earth Engine.
    """
    fc = gaul_regions().map(lambda f: f.setGeometry(f.geometry().simplify(100)))
    info = fc.getInfo()["features"]
    names = sorted(f["properties"]["ADM1_NAME"] for f in info)
    unknown = set(region_filter) - set(names)
    if unknown:
        sys.exit(f"❌ Unknown region(s): {sorted(unknown)}\n   Valid names: {names}")
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform
    return {
        f["properties"]["ADM1_NAME"]: shp_transform(to_utm, shape(f["geometry"])).buffer(1000)
        for f in info
        if not region_filter or f["properties"]["ADM1_NAME"] in region_filter
    }


def tile_origins(outline):
    """Top-left corners of grid tiles that touch the outline."""
    minx, miny, maxx, maxy = outline.bounds
    x_start = math.floor(minx / TILE_M) * TILE_M
    y_start = math.ceil(maxy / TILE_M) * TILE_M
    prepared = prep(outline)
    origins = []
    y = y_start
    while y > miny:
        x = x_start
        while x < maxx:
            if prepared.intersects(box(x, y - TILE_M, x + TILE_M, y)):
                origins.append((x, y))
            x += TILE_M
        y -= TILE_M
    return origins


def locate(lon, lat, regions, districts):
    point = Point(lon, lat)
    region = regions[regions.contains(point)]
    district = districts[districts.contains(point)]
    return (district.iloc[0]["shapeName"] if not district.empty else "Unknown",
            region.iloc[0]["shapeName"] if not region.empty else "Unknown")


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_tile(model, arr, gaul_region, x0, y_top, to_lonlat, regions,
               districts, today, stats):
    # Missing pixels arrive as 0 with valid = 0, so the zero-fill described in
    # the paper is already applied; the skip decision uses the real mask.
    image = np.stack([arr[b] for b in BANDS]).astype(np.float32) / 10000.0
    invalid = arr["valid"] == 0

    windows, offsets = [], []
    for r in range(0, TILE, STRIDE):
        for c in range(0, TILE, STRIDE):
            if invalid[r:r + CROP_SIZE, c:c + CROP_SIZE].mean() > MAX_INVALID_FRAC:
                stats["skipped_invalid"] += 1
                continue
            windows.append(image[:, r:r + CROP_SIZE, c:c + CROP_SIZE])
            offsets.append((r, c))

    if not windows:
        return []
    stats["windows_scored"] += len(windows)

    batch = F.interpolate(torch.from_numpy(np.stack(windows)),
                          size=(NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE),
                          mode="bilinear", align_corners=False)
    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)[:, 1]

    features = []
    area_ha = (CROP_SIZE * SCALE) ** 2 / 10000.0
    for (r, c), conf in zip(offsets, probs.tolist()):
        if conf < THRESHOLD:
            continue
        cx = x0 + (c + CROP_SIZE / 2) * SCALE
        cy = y_top - (r + CROP_SIZE / 2) * SCALE
        lon, lat = to_lonlat.transform(cx, cy)
        district, region = locate(lon, lat, regions, districts)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [round(lon, 6), round(lat, 6)]},
            "properties": {
                "confidence":  round(conf, 4),
                "district":    district,
                "region":      region,
                "date":        str(today),
                "area_ha":     round(area_ha, 2),
                # One id per window and GAUL region, so the validator's
                # (tile_id, date) de-duplication only drops genuine repeats.
                "tile_id":     (f"{gaul_region.replace(' ', '')}"
                                f"_E{int(x0 + c * SCALE)}_N{int(y_top - r * SCALE)}"),
                "alert_level": "HIGH" if conf > 0.8 else "MEDIUM" if conf > 0.5 else "LOW",
            },
        })
    return features


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", default="",
                        help="Comma-separated GAUL ADM1 names; blank = all of Ghana")
    args = parser.parse_args()
    region_filter = [r.strip() for r in args.regions.split(",") if r.strip()]

    today = datetime.date.today()
    init_ee()
    outlines = region_outlines(region_filter)
    jobs = [(name, x, y) for name, outline in outlines.items()
            for x, y in tile_origins(outline)]

    regions   = gpd.read_file(ADM1_PATH).to_crs("EPSG:4326")
    districts = gpd.read_file(ADM2_PATH).to_crs("EPSG:4326")
    to_lonlat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    model = load_model()

    print("=" * 60)
    print(f"FAMM weekly detection: {today}")
    for name, outline in outlines.items():
        print(f"  {name}: {sum(1 for j in jobs if j[0] == name)} tiles")
    print(f"Total: {len(jobs)} tiles of {FETCH} x {FETCH} px, {WORKERS} in parallel")
    print("=" * 60)

    stats = {"windows_scored": 0, "skipped_invalid": 0}
    features, failed = [], []
    start = time.time()

    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_tile, name, x, y, today): (name, x, y)
                   for name, x, y in jobs}
        for done, future in enumerate(cf.as_completed(futures), 1):
            name, x0, y_top = futures[future]
            try:
                arr = future.result()
            except Exception as e:
                failed.append(futures[future])
                print(f"⚠️  {name} tile E{x0} N{y_top} failed: {str(e)[:200]}")
                continue
            features.extend(score_tile(model, arr, name, x0, y_top, to_lonlat,
                                       regions, districts, today, stats))
            if done % 50 == 0 or done == len(jobs):
                elapsed = time.time() - start
                eta = elapsed / done * (len(jobs) - done)
                print(f"  {done}/{len(jobs)} tiles, {len(features)} detections, "
                      f"{elapsed / 60:.1f} min elapsed, ~{eta / 60:.0f} min left",
                      flush=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    print("=" * 60)
    print(f"Tiles fetched     : {len(jobs) - len(failed)} / {len(jobs)}")
    print(f"Windows scored    : {stats['windows_scored']}")
    print(f"Windows skipped (>{MAX_INVALID_FRAC:.0%} invalid): {stats['skipped_invalid']}")
    print(f"Detections ≥ {THRESHOLD}  : {len(features)}")
    print(f"Run time          : {(time.time() - start) / 60:.1f} min")
    print(f"✅ Saved to {OUTPUT_PATH}")

    if len(failed) > MAX_FAILED_FRACTION * len(jobs):
        sys.exit(f"❌ {len(failed)} tiles failed (over {MAX_FAILED_FRACTION:.0%}); "
                 "not publishing a partial map.")


if __name__ == "__main__":
    main()
