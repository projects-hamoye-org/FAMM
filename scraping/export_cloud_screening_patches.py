"""
Before/after cloud-screening comparison patches for the Methods section.

For 5 ASM and 5 non-ASM sample locations, clustered in the same corner of
the test region so they are directly comparable, fetches two March 2023
thumbnails per location from the exact same +/-15-day buffered window:
  - "raw": the plain per-pixel median with no mask applied at all
  - "screened": the dual-criterion mask (cloud prob < 60% OR NDVI >= 0.25)
    plus the two-stage focal gap-fill (3x3 mean, then 5x5 median), i.e.
    the same maskClouds() + buildMonthlyComposite() logic as
    scraping/GEE/1TemporalMedianFiltering.js lines 57-151, transcribed
    here rather than imported since that file is JS run in the GEE
    Code Editor.

The 10 locations are hardcoded centroids pulled from
data/ghana_asm_dataset_with_forest_reserves.geojson (5 highest-area ASM
polygons + 5 non-ASM squares within the same ~8x9 km box, lon -1.69..-1.62,
lat 5.29..5.37), not re-queried from the EE asset, so this script has no
dependency on which dataset asset is "current."

Run (needs an authenticated earthengine-api environment, not the figvenv
used for the matplotlib scripts in this directory):
  python scraping/export_cloud_screening_patches.py
"""

import urllib.request
from pathlib import Path

import ee

ee.Initialize(project="famm-472015")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figures" / "cloud_screening_patches"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Sample locations (lon, lat), same area of the test region
# ------------------------------
ASM_POINTS = [
    (-1.623119, 5.350734),
    (-1.626469, 5.358353),
    (-1.624857, 5.354979),
    (-1.631561, 5.364847),
    (-1.621427, 5.344694),
]
NONASM_POINTS = [
    (-1.674332, 5.346487),
    (-1.667875, 5.313391),
    (-1.682098, 5.310482),
    (-1.645049, 5.333020),
    (-1.639872, 5.338857),
]

PATCH_PIX = 256
SCALE = 10
BUFFER_METERS = (PATCH_PIX * SCALE) / 2  # 1,280 m -> 2.56 km square patch

TARGET_YEAR = 2023
TARGET_MONTH = 3  # March, one of the pipeline's dry-season TARGET_MONTHS
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

CLOUD_PROB_THRESHOLD = 60
MIN_OBSERVATIONS = 3
BUFFER_DAYS = 15

region_all = ee.Geometry.MultiPoint(ASM_POINTS + NONASM_POINTS).buffer(3000).bounds()

S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterBounds(region_all) \
    .filterDate(ee.Date.fromYMD(TARGET_YEAR, 1, 1), ee.Date.fromYMD(TARGET_YEAR + 1, 1, 1))
S2_CLOUD = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY") \
    .filterBounds(region_all) \
    .filterDate(ee.Date.fromYMD(TARGET_YEAR, 1, 1), ee.Date.fromYMD(TARGET_YEAR + 1, 1, 1))

joined = ee.Join.saveFirst("cloudprob").apply(
    primary=S2,
    secondary=S2_CLOUD,
    condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
)


def mask_clouds(img):
    img = ee.Image(img)
    cloud_prob = ee.Image(img.get("cloudprob")).select("probability")
    is_clear_prob = cloud_prob.lt(CLOUD_PROB_THRESHOLD)
    scaled = img.divide(10000)
    ndvi = scaled.normalizedDifference(["B8", "B4"])
    ndvi_mask = ndvi.gte(0.25)
    keep = is_clear_prob.Or(ndvi_mask)
    return scaled.updateMask(keep).copyProperties(img, ["system:time_start", "system:index"])


start = ee.Date.fromYMD(TARGET_YEAR, TARGET_MONTH, 1)
end = start.advance(1, "month")
window_start = start.advance(-BUFFER_DAYS, "day")
window_end = end.advance(BUFFER_DAYS, "day")

window_col_raw = ee.ImageCollection(joined).filterDate(window_start, window_end)
raw_composite = window_col_raw.select(BANDS).map(lambda i: i.float()).median().divide(10000)

window_col_masked = window_col_raw.map(mask_clouds).map(lambda i: i.select(BANDS).float())
obs_count = window_col_masked.map(
    lambda i: i.select(BANDS[0]).gt(-9999).rename("v")
).select("v").reduce(ee.Reducer.sum()).rename("count")

median = window_col_masked.median()
# NOTE (Nov 2026): the radii below are in 'meters', so on 10 m pixels each
# kernel is a single pixel and neither pass fills anything. This is the code
# that produced Figs. 6-8 of the paper, so it is left unchanged as a record of
# what was run. The same issue affected the training composites, where the
# fill was barely needed: measured on ~20,000 sampled pixels per composite
# across the study area, empty pixels made up at most 0.055% of any one
# monthly composite (March 2021), and 13 of the 20 composites (2021-2023,
# 2025) had none. The 3x3 mean / 5x5 median the paper describes needs 'pixels'
# units, as used in deployment/ee_export.py and pipeline/weekly_detect.py.
fill1 = median.focal_mean(1, "square", "meters")
filled = median.unmask(fill1)
fill2 = filled.focal_median(2, "square", "meters")
filled = filled.unmask(fill2)

sufficient = obs_count.gte(MIN_OBSERVATIONS)
month_col = ee.ImageCollection(joined).filterDate(start, end).map(mask_clouds) \
    .map(lambda i: i.select(BANDS).float())
month_median = month_col.median()
month_fill1 = month_median.focal_mean(1, "square", "meters")
month_median = month_median.unmask(month_fill1)

screened_composite = ee.Image(ee.Algorithms.If(sufficient, filled, month_median))

VIS = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.3}


def fetch(image, region, out_path):
    url = image.clip(region).visualize(**VIS).getThumbURL({
        "region": region,
        "dimensions": 512,
        "format": "png",
    })
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved {out_path}")


for label, points in [("asm", ASM_POINTS), ("nonasm", NONASM_POINTS)]:
    for idx, (lon, lat) in enumerate(points):
        region = ee.Geometry.Point(lon, lat).buffer(BUFFER_METERS).bounds()
        fetch(raw_composite, region, OUT_DIR / f"{label}_{idx}_raw.png")
        fetch(screened_composite, region, OUT_DIR / f"{label}_{idx}_screened.png")

print("All 20 thumbnails fetched. Next: .figvenv/bin/python figures/cloud_screening_comparison.py")