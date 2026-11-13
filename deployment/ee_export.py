# pip install earthengine-api
import ee
import datetime

ee.Initialize(project="famm-472015")

# COUNTRY SETTINGS
COUNTRY = "Ghana"

# LOAD GHANA REGIONS (ADM1)
# Restricted to the two AOI-intersecting regions not yet downloaded
# (Ashanti, Central and Western already exported); remove this filter
# to resume exporting all five AOI regions at once.
REGIONS_TO_EXPORT = ['Brong Ahafo', 'Eastern']

regions = ee.FeatureCollection("FAO/GAUL/2015/level1") \
    .filter(ee.Filter.eq('ADM0_NAME', COUNTRY)) \
    .filter(ee.Filter.inList('ADM1_NAME', REGIONS_TO_EXPORT))

region_list = regions.toList(regions.size())

# SETTINGS
SCALE = 10
BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']

today = datetime.date.today()

start_7  = ee.Date(str(today - datetime.timedelta(days=7)))
start_14 = ee.Date(str(today - datetime.timedelta(days=14)))
start_30 = ee.Date(str(today - datetime.timedelta(days=30)))
end_date = ee.Date(str(today))

# CLOUD + NDVI MASK
def mask_s2(image, roi):

    image_date = image.date()

    cloud_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(roi)
        .filterDate(image_date.advance(-3,'day'),
                    image_date.advance(3,'day'))
        .sort("system:time_start")
    )

    cloud_mask = ee.Image(
        ee.Algorithms.If(
            cloud_col.size().gt(0),
            ee.Image(cloud_col.first()).select("probability").lt(60),
            ee.Image(1)
        )
    )

    scaled = image.divide(10000)

    ndvi = scaled.normalizedDifference(['B8','B4'])
    ndvi_mask = ndvi.gte(0.25)

    keep = cloud_mask.Or(ndvi_mask)

    return scaled.updateMask(keep).copyProperties(
        image, ['system:time_start']
    )


# LOOP THROUGH REGIONS
num_regions = regions.size().getInfo()

for i in range(num_regions):

    region_feature = ee.Feature(region_list.get(i))
    region_name = region_feature.get('ADM1_NAME').getInfo()
    ROI = region_feature.geometry()

    print(f"Processing region: {region_name}")

    # GET COLLECTION
    def get_collection(start_date):
        return (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(ROI)
            .filterDate(start_date, end_date)
            .map(lambda img: mask_s2(img, ROI))
            .select(BANDS)
        )

    col_7  = get_collection(start_7)
    col_14 = get_collection(start_14)
    col_30 = get_collection(start_30)

    weekly   = col_7.median()
    biweekly = col_14.median()
    monthly  = col_30.median()

    composite = weekly.unmask(biweekly).unmask(monthly)

    # Spatial gap-fill for whatever is still masked after the 7/14/30-day
    # temporal fallback, matching scraping/GEE/1TemporalMedianFiltering.js's
    # buildMonthlyComposite() two-stage focal fill (3x3 mean, then 5x5
    # median) instead of the previous composite.unmask(0). Zero-filling
    # fabricated a value indistinguishable from real dark reflectance and
    # let deployment_inference.py score it as if it were data; a focal
    # fill instead estimates the missing pixel from its real neighbours,
    # consistent with how every training/test composite is built. This is
    # heavier (two region-wide focal passes per export, on top of the
    # temporal chain above) but was requested specifically because zero
    # is not an acceptable stand-in for missing data here.
    # Radii are in pixels: 'meters' gave a 1-pixel kernel that filled nothing
    # (checked in .github/workflows/wif_smoke_test.yml).
    fill1 = composite.focal_mean(1, 'square', 'pixels')
    composite = composite.unmask(fill1)
    fill2 = composite.focal_median(2, 'square', 'pixels')
    composite = composite.unmask(fill2)

    # No blanket unmask(0): any pixel still masked after both the temporal
    # and spatial fallbacks (expected to be rare, e.g. a location with no
    # cloud-free observation anywhere in 30 days AND no valid neighbour
    # within a 5x5 window) is exported as nodata rather than a fabricated
    # zero. deployment_inference.py's sliding window must check for this
    # (see its own validity-fraction guard) instead of assuming every
    # exported tile is fully dense.
    composite = composite.clip(ROI)

    EXPORT_NAME = f"{COUNTRY}_{region_name}_Composite_{today}"

    # EXPORT IMAGE
    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=EXPORT_NAME,
        folder="EarthEngineExports",
        fileNamePrefix=EXPORT_NAME,
        region=ROI,
        scale=SCALE,
        maxPixels=1e13,
        fileFormat="GeoTIFF"
    )

    task.start()

    print(f"Started export for {region_name}")

    # EXPORT METADATA
    metadata_feature = ee.Feature(None, {
        'composite_name': EXPORT_NAME,
        'country': COUNTRY,
        'region': region_name,
        'bands': ','.join(BANDS),
        'scale': SCALE,
        'creation_date': str(today),
        'roi_area_m2': ROI.area(),
        'roi_bounds': ROI.bounds()
    })

    metadata_fc = ee.FeatureCollection([metadata_feature])

    meta_task = ee.batch.Export.table.toDrive(
        collection=metadata_fc,
        description=f"{EXPORT_NAME}_metadata",
        folder="EarthEngineExports",
        fileNamePrefix=f"{EXPORT_NAME}_metadata",
        fileFormat='GeoJSON'
    )

    meta_task.start()

    print(f"Metadata export started for {region_name}")

print("All region exports started.")