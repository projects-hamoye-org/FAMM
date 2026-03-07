# pip install earthengine-api
import ee
import datetime

ee.Initialize(project="famm-472015")

# COUNTRY SETTINGS
COUNTRY = "Ghana"

# LOAD GHANA REGIONS (ADM1)
regions = ee.FeatureCollection("FAO/GAUL/2015/level1") \
    .filter(ee.Filter.eq('ADM0_NAME', COUNTRY))

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
    composite = composite.unmask(0).clip(ROI)

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