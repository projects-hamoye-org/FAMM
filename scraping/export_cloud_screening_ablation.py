"""
NDVI-ablation follow-up to scraping/export_cloud_screening_patches.py.

That script's before/after figure (Sec. cloud_screening_qualitative,
Results_Section_ASM_Ghana.tex) shows what the full dual-criterion mask
(cloud probability < 60% OR NDVI >= 0.25) does relative to no masking at
all, but conflates the cloud-probability threshold and the NDVI rescue
clause into one step, so it cannot show what the NDVI clause specifically
contributes. This script adds a third variant per ASM location, built
from the identical +/-15-day buffered window: cloud-probability-only
masking (drop the "OR NDVI >= 0.25" rescue term, keep the same two-stage
gap-fill), so it can be compared directly against the full dual-criterion
result already fetched.

Only the 5 ASM locations are re-fetched (not the 5 non-ASM ones): the
NDVI clause exists specifically to stop bright, sparsely vegetated mine
surfaces being masked as cloud (Sec. cloud_removal), so the ablation is
most informative on ASM patches.

Run (needs an authenticated earthengine-api environment):
  python scraping/export_cloud_screening_ablation.py
"""

import urllib.request
from pathlib import Path

import ee

ee.Initialize(project="famm-472015")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figures" / "cloud_screening_patches"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASM_POINTS = [
    (-1.623119, 5.350734),
    (-1.626469, 5.358353),
    (-1.624857, 5.354979),
    (-1.631561, 5.364847),
    (-1.621427, 5.344694),
]

PATCH_PIX = 256
SCALE = 10
BUFFER_METERS = (PATCH_PIX * SCALE) / 2

TARGET_YEAR = 2023
TARGET_MONTH = 3
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

CLOUD_PROB_THRESHOLD = 60
MIN_OBSERVATIONS = 3
BUFFER_DAYS = 15

region_all = ee.Geometry.MultiPoint(ASM_POINTS).buffer(3000).bounds()

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


def mask_cloud_prob_only(img):
    """Same as export_cloud_screening_patches.py's mask_clouds(), minus the
    NDVI OR-clause: keep a pixel only if cloud probability < 60%."""
    img = ee.Image(img)
    cloud_prob = ee.Image(img.get("cloudprob")).select("probability")
    is_clear_prob = cloud_prob.lt(CLOUD_PROB_THRESHOLD)
    scaled = img.divide(10000)
    return scaled.updateMask(is_clear_prob).copyProperties(img, ["system:time_start", "system:index"])


start = ee.Date.fromYMD(TARGET_YEAR, TARGET_MONTH, 1)
end = start.advance(1, "month")
window_start = start.advance(-BUFFER_DAYS, "day")
window_end = end.advance(BUFFER_DAYS, "day")

window_col = ee.ImageCollection(joined).filterDate(window_start, window_end)
window_col_masked = window_col.map(mask_cloud_prob_only).map(lambda i: i.select(BANDS).float())

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
month_col = ee.ImageCollection(joined).filterDate(start, end).map(mask_cloud_prob_only) \
    .map(lambda i: i.select(BANDS).float())
month_median = month_col.median()
month_fill1 = month_median.focal_mean(1, "square", "meters")
month_median = month_median.unmask(month_fill1)

cloudprob_only_composite = ee.Image(ee.Algorithms.If(sufficient, filled, month_median))

VIS = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.3}


def fetch(image, region, out_path):
    url = image.clip(region).visualize(**VIS).getThumbURL({
        "region": region,
        "dimensions": 512,
        "format": "png",
    })
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved {out_path}")


for idx, (lon, lat) in enumerate(ASM_POINTS):
    region = ee.Geometry.Point(lon, lat).buffer(BUFFER_METERS).bounds()
    fetch(cloudprob_only_composite, region, OUT_DIR / f"asm_{idx}_cloudprob_only.png")

print("Done. Next: .figvenv/bin/python figures/cloud_screening_ablation.py")