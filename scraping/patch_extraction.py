import ee
ee.Initialize(project='famm-472015')

# ------------------------------
# User parameters
# ------------------------------
START_YEAR = 2021       # start year for training composites
TRAIN_YEARS = 2         # number of years for training (e.g., 2021-2022)
PATCH_PIX = 256
SCALE = 10
BUFFER_METERS = (PATCH_PIX * SCALE) / 2

# AOI (MultiPolygon)
geometry = ee.Geometry.MultiPolygon([
    [[[-2.577107401987939, 6.5306601480568975],
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
      [-1.6499660314807807, 6.831115397308761]]],
    [[[-2.1059451766557435, 5.950609199917876],
      [-2.0510135360307435, 5.906898947359114],
      [-2.5399051375932435, 6.306703638207781]]]
])

# Training region (Polygon)
train_region = ee.Geometry.Polygon([
    [[-2.4663687360528996, 6.558669631005525],
     [-2.4663687360528996, 5.64105309928928],
     [-1.4501333844903996, 5.64105309928928],
     [-1.4501333844903996, 6.558669631005525]]
])

# Testing region (Polygon)
test_region = ee.Geometry.Polygon([
    [[-2.3400259626153996, 5.619186440055064],
     [-2.3400259626153996, 5.072261306943754],
     [-1.5654898298028996, 5.072261306943754],
     [-1.5654898298028996, 5.619186440055064]]
])


# Load ASM dataset

asmDataset = ee.FeatureCollection("projects/famm-472015/assets/Ghana_ASM_Dataset_Robust")
asm = asmDataset.filter(ee.Filter.eq('label', 1))
nonAsm = asmDataset.filter(ee.Filter.eq('label', 0))

trainAsm = asm.filterBounds(train_region)
trainNonAsm = nonAsm.filterBounds(train_region)
testAsm = asm.filterBounds(test_region)
testNonAsm = nonAsm.filterBounds(test_region)

print('Train ASM count:', trainAsm.size().getInfo())
print('Train non-ASM count:', trainNonAsm.size().getInfo())
print('Test ASM count:', testAsm.size().getInfo())
print('Test non-ASM count:', testNonAsm.size().getInfo())

# ------------------------------
# Load composite image collection and filter by year
# Training years
# Training years
train_years = [str(START_YEAR + i) for i in range(TRAIN_YEARS)]  # ['2021','2022']

# Full collection
full_collection = ee.ImageCollection("projects/famm-472015/assets/Ghana_ASM_Bimonthly") \
    .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])

# Correct way: merge all years
train_composites = ee.ImageCollection([])  # start empty
for year in train_years:
    year_comps = full_collection.filter(ee.Filter.stringContains('system:index', year))
    train_composites = train_composites.merge(year_comps)

# Test composites (next year)
test_year = str(START_YEAR + TRAIN_YEARS)  # '2023'
test_composites = full_collection.filter(ee.Filter.stringContains('system:index', test_year))

print("All train composites count:", train_composites.size().getInfo())  # should now be 10
print("All test composites count:", test_composites.size().getInfo())    # depends on 2023 images




# Helper: export function with Drive folder structure

def export_patch(feature, label, subset, index, image, comp_index):
    centroid = feature.geometry().centroid()
    region = centroid.buffer(BUFFER_METERS).bounds()
    
    folder = f'{subset}/{"ASM" if label==1 else "Non-ASM"}'
    desc = f'{subset}_label{label}_c{comp_index}_i{index}'
    
    task = ee.batch.Export.image.toDrive(
        image=image.clip(region),
        description=desc,
        folder=folder,
        fileNamePrefix=desc,
        region=region,
        scale=SCALE,
        crs='EPSG:4326',
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task.start()


# Full export: all composites and all features

# Convert training/testing features to lists
trainAsm_list = trainAsm.toList(trainAsm.size())
trainNonAsm_list = trainNonAsm.toList(trainNonAsm.size())
testAsm_list = testAsm.toList(testAsm.size())
testNonAsm_list = testNonAsm.toList(testNonAsm.size())

# Convert composites to list
train_composites_list = train_composites.toList(train_composites.size())
test_composites_list = test_composites.toList(test_composites.size())

# Export training patches
for c_idx in range(train_composites_list.size().getInfo()):
    image = ee.Image(train_composites_list.get(c_idx))
    print(f'Exporting training composite {c_idx+1}/{train_composites_list.size().getInfo()}')

    for i in range(trainAsm_list.size().getInfo()):
        export_patch(ee.Feature(trainAsm_list.get(i)), 1, 'train', i, image, c_idx+1)
    for j in range(trainNonAsm_list.size().getInfo()):
        export_patch(ee.Feature(trainNonAsm_list.get(j)), 0, 'train', j, image, c_idx+1)

# Export testing patches
for c_idx in range(test_composites_list.size().getInfo()):
    image = ee.Image(test_composites_list.get(c_idx))
    print(f'Exporting test composite {c_idx+1}/{test_composites_list.size().getInfo()}')

    for k in range(testAsm_list.size().getInfo()):
        export_patch(ee.Feature(testAsm_list.get(k)), 1, 'test', k, image, c_idx+1)
    for l in range(testNonAsm_list.size().getInfo()):
        export_patch(ee.Feature(testNonAsm_list.get(l)), 0, 'test', l, image, c_idx+1)

print('All export tasks have been initiated. Check Drive folders for train/test patches.')

