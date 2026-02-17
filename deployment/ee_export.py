# pip install earthengine-api
import ee
import datetime

ee.Initialize(project="famm-472015")


#  ROI & DYNAMIC COUNTRY DETECTION

ROI = ee.Geometry.Polygon([
    [-2.1734693858250353, 4.984296847965302],
    [-1.5720532810091514, 5.121001314357858],
    [-0.9295708164086846, 5.421854403720199],
    [-1.436961333748421, 6.219931255643865],
    [-1.3346584712677179, 6.530992048541554],
    [-1.415359587336964, 6.779274372062422],
    [-1.6499660314807807, 6.831115397308761]
])

def get_country_name(geometry):
    # Load international boundaries dataset
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    # Filter for the country that contains the ROI center
    country_feature = countries.filterBounds(geometry.centroid()).first()
    # Get the country name (e.g., 'Ghana')
    name = ee.String(country_feature.get('country_na')).getInfo()
    return name.replace(" ", "_") # Ensure no spaces in filename

COUNTRY = get_country_name(ROI)


# SETTINGS

SCALE = 10
BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']

today = datetime.date.today()
start_7  = ee.Date(str(today - datetime.timedelta(days=7)))
start_14 = ee.Date(str(today - datetime.timedelta(days=14)))
start_30 = ee.Date(str(today - datetime.timedelta(days=30)))
end_date = ee.Date(str(today))


# ROBUST CLOUD + NDVI MASK

def mask_s2(image):
    image_date = image.date()
    cloud_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(ROI)
        .filterDate(image_date.advance(-3, 'day'),
                    image_date.advance(3, 'day'))
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
    return scaled.updateMask(keep).copyProperties(image, ['system:time_start'])


# GET COLLECTION

def get_collection(start_date):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ROI)
        .filterDate(start_date, end_date)
        .map(mask_s2)
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

# DYNAMIC EXPORT NAME
EXPORT_NAME = f"{COUNTRY}_Composite_{today}"


# EXPORT TO DRIVE

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

print(f" Composite export for {COUNTRY} started to Drive.")


# EXPORT COMPOSITE METADATA

metadata_feature = ee.Feature(None, {
    'composite_name': EXPORT_NAME,
    'country': COUNTRY,
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
print("📄 Composite metadata export started.")