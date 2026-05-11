var imageCollection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED"),
    imageCollection2 = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY"),
    geometry = /* color: #b41803 */ee.Geometry.Polygon(
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
          [-1.6499660314807807, 6.831115397308761]]]);
var geometry = /* color: #b41803 */geometry;
//----------------------------------------------
// 0. USER CONFIG
//----------------------------------------------
var aoi = geometry;
Map.centerObject(aoi, 11);

var START_YEAR = 2025;
var END_YEAR = 2025;
var TARGET_MONTHS = [1, 2, 3, 11, 12];   // months to produce composites for

// Sentinel-2 spectral bands we want in the final composite
var BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];

var CLOUD_PROB_THRESHOLD = 60;   // cloudprob threshold (0-100)
var MIN_OBSERVATIONS = 3;        // minimal good observations to trust median
var BUFFER_DAYS = 15;            // expand temporal window +/- this many days to get samples

// Parent asset folder where child composites are written (must exist)
var PARENT_ASSET = "projects/famm-472015/assets/Ghana_ASM_Bimonthly";

// Collections already available in your script (names kept)
var S2 = imageCollection
  .filterBounds(aoi)
  .filterDate(ee.Date.fromYMD(START_YEAR,1,1), ee.Date.fromYMD(END_YEAR+1,1,1));

var S2_CLOUD = imageCollection2
  .filterBounds(aoi)
  .filterDate(ee.Date.fromYMD(START_YEAR,1,1), ee.Date.fromYMD(END_YEAR+1,1,1));

// Join S2 to cloud probability collection (saves cloud probability under 'cloudprob')
var joined = ee.Join.saveFirst('cloudprob').apply({
  primary: S2,
  secondary: S2_CLOUD,
  condition: ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
});

//----------------------------------------------
// 1. IMPROVED CLOUD MASK (cloudprob + NDVI)
//----------------------------------------------
function maskClouds(img) {
  img = ee.Image(img);
  // Cloud probability band saved in join named 'cloudprob' (adjust if different)
  var cloudProb = ee.Image(img.get('cloudprob')).select('probability');
  // scale to 0-100 if needed (s2cloudless is 0..100)
  var isClearProb = cloudProb.lt(CLOUD_PROB_THRESHOLD);
  
  // also compute NDVI to avoid masking bright vegetation/soil
  var scaled = img.divide(10000); // Sentinel-2 L2A typically 0..10000
  var ndvi = scaled.normalizedDifference(['B8','B4']);
  // clouds tend to have low NDVI; keep pixels with NDVI >= ndviThresh OR clear probability
  var ndviThresh = 0.25;
  var ndviMask = ndvi.gte(ndviThresh);
  
  // final keep mask: either low cloudProb OR reasonably vegetated (avoids overmasking)
  var keep = isClearProb.or(ndviMask);
  
  // apply mask and return scaled reflectance
  return scaled.updateMask(keep).copyProperties(img, ['system:time_start','system:index']);
}

//----------------------------------------------
// 2. HELPER: safeSelectBands - returns image with BANDS or a nodata placeholder
//----------------------------------------------
function makeNodataImage() {
  // create a nodata image with the same band names, filled with masked pixels
  var blank = ee.Image.constant(0).rename(BANDS).updateMask(ee.Image.constant(0)); // fully masked
  return blank.set('nodata', 1);
}

function safeSelectBands(img) {
  img = ee.Image(img);
  var hasBands = img.bandNames().contains(BANDS[0]); // check presence of first band
  // if present select desired bands, else return nodata placeholder
  return ee.Algorithms.If(
    hasBands,
    img.select(BANDS),
    makeNodataImage()
  );
}

//----------------------------------------------
// 3. ROBUST MONTHLY MEDIAN FUNCTION (with gap fill)
//----------------------------------------------
function buildMonthlyComposite(yearNum, monthNum) {
  // monthNum and yearNum are client-side numbers
  var start = ee.Date.fromYMD(yearNum, monthNum, 1);
  var end = start.advance(1, 'month');
  
  // Expand window by BUFFER_DAYS to gather more observations (robust)
  var windowStart = start.advance(-BUFFER_DAYS, 'day');
  var windowEnd   = end.advance(BUFFER_DAYS, 'day');
  
  // Filter joined collection in that window and mask clouds
  var windowCol = ee.ImageCollection(joined)
    .filterDate(windowStart, windowEnd)
    .map(maskClouds)
    .map(function(i){ return i.select(BANDS).float(); }); // ensure float and band selection if present
    
  // count valid observations per pixel (non-masked)
  var obsCount = windowCol.map(function(i){ return i.select(BANDS[0]).gt(-9999).rename('v'); })
                          .select('v')
                          .reduce(ee.Reducer.sum())
                          .rename('count'); // sum of non-masked first band occurrences
  
  // compute median (server-side)
  var median = windowCol.median();
  median = ee.Image(safeSelectBands(median)); // ensures desired bandnames or a nodata placeholder
  
  // If median contains fully masked pixels (holes), apply small spatial fills:
  // 1) perform a focal mean (3x3) to fill single-pixel holes, then use it to unmask.
  // 2) perform a second pass focal median (5x5) for slightly larger gaps.
  var fill1 = median.focal_mean(1, 'square', 'meters');   // 3x3
  var filled = median.unmask(fill1);                      // fill missing where median masked
  var fill2 = filled.focal_median(2, 'square', 'meters'); // 5x5
  filled = filled.unmask(fill2);
  
  // As a final safety: if number of observations is below MIN_OBSERVATIONS, fallback to a larger temporal aggregation:
  var sufficient = obsCount.gte(MIN_OBSERVATIONS);
  // compute a fallback: monthly median without cloudmask? better: use entire month (non-buffered) median
  var monthCol = ee.ImageCollection(joined)
    .filterDate(start, end)
    .map(maskClouds)
    .map(function(i){ return i.select(BANDS).float(); });
  var monthMedian = ee.Image(safeSelectBands(monthCol.median()));
  var monthFill1 = monthMedian.focal_mean(1, 'square', 'meters');
  monthMedian = monthMedian.unmask(monthFill1);
  
  // Choose per-pixel between filled (from window) and fallback (monthMedian) based on sufficient observations
  // If sufficient true -> use filled, else use monthMedian
  var final = ee.Image(ee.Algorithms.If(
    sufficient,
    filled,
    monthMedian
  ));
  
  // Attach properties for export naming
  final = final.set({
    'year': yearNum,
    'month': monthNum,
    'start': start.format('YYYY-MM-dd'),
    'end': end.advance(-1,'day').format('YYYY-MM-dd'),
    'obs_count': obsCount.reduceRegion({
      reducer: ee.Reducer.min(),
      geometry: aoi,
      scale: 1000,
      bestEffort: true
    }).get('count')  // overall minimum count in the AOI as quick QA
  });
  
  return final;
}

//----------------------------------------------
// 4. BUILD ALL MONTHLY COMPOSITES (server-side list)
//----------------------------------------------
var compositesList = [];
for (var y = START_YEAR; y <= END_YEAR; y++) {
  TARGET_MONTHS.forEach(function(m) {
    compositesList.push(buildMonthlyComposite(y, m));
  });
}
// create imageCollection
var finalCollection = ee.ImageCollection(compositesList);

// filter out any pure nodata images (safety)
finalCollection = finalCollection.filter(ee.Filter.neq('nodata', 1));

// Print summary
print('Final monthly composite collection size:', finalCollection.size());
print('Sample composite properties:', finalCollection.first().toDictionary());

// show a couple on the map (safe)
Map.addLayer(finalCollection.first(), {bands: ['B4','B3','B2'], min:0, max:0.3}, 'Composite sample');

//----------------------------------------------
// 5. EXPORT: one asset per month (child assets), each containing 10 bands
//----------------------------------------------
var list = finalCollection.toList(finalCollection.size());
var n = list.size().getInfo();
print('Composites to export:', n);

for (var i = 0; i < n; i++) {
  var img = ee.Image(list.get(i));
  var startStr = img.get('start').getInfo();
  var endStr   = img.get('end').getInfo();
  var yr = img.get('year').getInfo();
  var mo = img.get('month').getInfo();
  var assetId = PARENT_ASSET + '/Composite_' + startStr + '_' + endStr;
  var desc = 'Composite_' + yr + '_' + (('0' + mo).slice(-2)) + '_' + startStr;
  print('Creating export task:', desc);
  Export.image.toAsset({
    image: img,
    description: desc,
    assetId: assetId,
    region: aoi,
    scale: 10,
    crs: 'EPSG:32630',
    maxPixels: 1e13
  });
}

print('Export tasks created. Go to Tasks to run them.');
