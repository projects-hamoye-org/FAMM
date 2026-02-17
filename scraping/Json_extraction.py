import ee


# Initialize Earth Engine

ee.Initialize(project='famm-472015')


# USER PARAMETERS


START_YEAR = 2021
TRAIN_YEARS = 2

PATCH_PIX = 256
SCALE = 10
BUFFER_METERS = (PATCH_PIX * SCALE) / 2

CREATION_SEED = 42
SOURCE_TYPE = 'S2'
COUNTRY = 'Ghana'


# AOI


geometry = ee.Geometry.MultiPolygon([
    [[
        [-2.577107401987939, 6.5306601480568975],
        [-2.903948285164646, 6.2932353613892245],
        [-2.9766815269095392, 6.08434089968068],
        [-2.90664378427828, 5.908201200818874],
        [-2.8077586698808297, 5.711522903619994],
        [-2.4782606507938705, 5.339810442902648],
        [-2.1734693858250353, 4.984296847965302],
        [-1.5720532810091514, 5.121001314357858],
        [-0.9295708164086846, 5.421854403720199],
        [-1.436961333748421, 6.219931255643865],
        [-1.3346584712677179, 6.530992048541554],
        [-1.415359587336964, 6.779274372062422],
        [-1.6499660314807807, 6.831115397308761]
    ]]
])


# TRAIN / TEST REGIONS


train_region = ee.Geometry.Polygon([
    [-2.4663687360528996, 6.558669631005525],
    [-2.4663687360528996, 5.64105309928928],
    [-1.4501333844903996, 5.64105309928928],
    [-1.4501333844903996, 6.558669631005525]
])

test_region = ee.Geometry.Polygon([
    [-2.3400259626153996, 5.619186440055064],
    [-2.3400259626153996, 5.072261306943754],
    [-1.5654898298028996, 5.072261306943754],
    [-1.5654898298028996, 5.619186440055064]
])


# LOAD ASM DATASET (ALREADY FLAGGED WITH in_protected_area)


asmDataset = ee.FeatureCollection(
    "projects/famm-472015/assets/Ghana_ASM_Dataset_WithForestReserves"
)

asm = asmDataset.filter(ee.Filter.eq('label', 1))
nonAsm = asmDataset.filter(ee.Filter.eq('label', 0))

trainAsm = asm.filterBounds(train_region)
trainNonAsm = nonAsm.filterBounds(train_region)

testAsm = asm.filterBounds(test_region)
testNonAsm = nonAsm.filterBounds(test_region)


# LOAD COMPOSITES


train_years = [str(START_YEAR + i) for i in range(TRAIN_YEARS)]
test_year = str(START_YEAR + TRAIN_YEARS)

full_collection = ee.ImageCollection(
    "projects/famm-472015/assets/Ghana_ASM_Bimonthly"
)

train_composites = ee.ImageCollection([])
for year in train_years:
    train_composites = train_composites.merge(
        full_collection.filter(
            ee.Filter.stringContains('system:index', year)
        )
    )

test_composites = full_collection.filter(
    ee.Filter.stringContains('system:index', test_year)
)

train_composites_list = train_composites.toList(train_composites.size())
test_composites_list = test_composites.toList(test_composites.size())


# METADATA FEATURE BUILDER


def make_metadata_feature(feature, label, subset, index, comp_index):
    geom = feature.geometry()
    centroid = geom.centroid()
    bounds = centroid.buffer(BUFFER_METERS).bounds()

    patch_id = (
        ee.String(subset)
        .cat('_label').cat(ee.Number(label).format('%d'))
        .cat('_c').cat(ee.Number(comp_index).format('%d'))
        .cat('_i').cat(ee.Number(index).format('%d'))
    )

    return ee.Feature(None, {
        'image_id': patch_id,              # EXACT MATCH to GeoTIFF name
        'subset': subset,                  # train / test
        'label': label,                    # 1 = ASM, 0 = non-ASM
        'composite_index': comp_index,
        'feature_index': index,

        'longitude': centroid.coordinates().get(0),
        'latitude': centroid.coordinates().get(1),
        'area_m2': geom.area(),
        'bbox': bounds,

        'in_forest_reserve': feature.get('in_forest_reserve'),

        'country': 'Ghana',
        'buffer_m': BUFFER_METERS,
        'creation_seed': 42,
        'wdpa_used': True
    })



# BUILD METADATA FEATURE COLLECTION


metadata_features = []

# ---- Training ----
for c in range(train_composites_list.size().getInfo()):
    for i in range(trainAsm.size().getInfo()):
        metadata_features.append(
            make_metadata_feature(
                ee.Feature(trainAsm.toList(trainAsm.size()).get(i)),
                1, 'train', i, c + 1
            )
        )

    for j in range(trainNonAsm.size().getInfo()):
        metadata_features.append(
            make_metadata_feature(
                ee.Feature(trainNonAsm.toList(trainNonAsm.size()).get(j)),
                0, 'train', j, c + 1
            )
        )

# ---- Testing ----
for c in range(test_composites_list.size().getInfo()):
    for k in range(testAsm.size().getInfo()):
        metadata_features.append(
            make_metadata_feature(
                ee.Feature(testAsm.toList(testAsm.size()).get(k)),
                1, 'test', k, c + 1
            )
        )

    for l in range(testNonAsm.size().getInfo()):
        metadata_features.append(
            make_metadata_feature(
                ee.Feature(testNonAsm.toList(testNonAsm.size()).get(l)),
                0, 'test', l, c + 1
            )
        )

metadata_fc = ee.FeatureCollection(metadata_features)


# EXPORT METADATA AS GEOJSON


task = ee.batch.Export.table.toDrive(
    collection=metadata_fc,
    description='Ghana_ASM_Patch_Metadata_JSON',
    folder='ASM_Metadata',
    fileNamePrefix='ghana_asm_patch_metadata',
    fileFormat='GeoJSON'
)

task.start()

print('Metadata export task started.')

