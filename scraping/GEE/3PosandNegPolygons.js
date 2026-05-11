var geometry = /* color: #b41803 */ee.Geometry.Polygon(
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
var imageCollection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED"),
    imageCollection2 = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY"),
    geometry = /* color: #b41803 */geometry;
          
// -----------------------------------------------------------
// 0. Load mining polygons
// -----------------------------------------------------------
var gm = ee.FeatureCollection("projects/famm-472015/assets/Mining_Polygons");
var aoi = geometry;

// Ghana filter
var gh_mines = gm.filter(ee.Filter.eq('GID_0', 'GHA'));
var gh_mines_aoi = gh_mines.filterBounds(aoi);

// ASM-scale (<1 km²)
var asm_sites = gh_mines_aoi
  .filter(ee.Filter.lt('Area_Num', 1e6))
  .map(function(f){ return f.set('label', 1); });

print('ASM polygons:', asm_sites.size());

// -----------------------------------------------------------
// 1. Generate random points inside AOI
// -----------------------------------------------------------
var numNeg = 500;

var randomPts = ee.FeatureCollection.randomPoints({
  region: aoi,
  points: numNeg * 3,   // generate extra to account for filtering overlaps
  seed: 42
});

print("Initial random points:", randomPts.size());

// -----------------------------------------------------------
// 2. Convert points to squares and filter overlaps
// -----------------------------------------------------------
var halfSize = 1280;

var neg_polygons = randomPts.map(function(f) {
  var square = f.geometry().buffer(halfSize).bounds();
  var intersects = gh_mines_aoi.filterBounds(square).size().gt(0);

  return ee.Feature(square)
           .set('label', 0)
          .set('valid', intersects.not());
})
.filter(ee.Filter.eq('valid', 1)) // keep only non-overlapping
.limit(numNeg); // ensure exactly numNeg negatives

print("Negative polygons:", neg_polygons.size());

// -----------------------------------------------------------
// 3. Merge with positive polygons
// -----------------------------------------------------------
var pos_polygons = asm_sites.map(function(f){ return f.set('label', 1); });

var labeledSamples = pos_polygons.merge(neg_polygons);

print("Positive polygons:", pos_polygons.size());
print("Negative polygons:", neg_polygons.size());
print("Total samples:", labeledSamples.size());

// -----------------------------------------------------------
// 4. Visualization
// -----------------------------------------------------------
Map.centerObject(aoi, 8);
// Map.addLayer(pos_polygons, {color:'red'}, 'ASM');
Map.addLayer(gh_mines_aoi, {color:'black'}, 'poly');
Map.addLayer(neg_polygons, {color:'blue'}, 'Background');

// -----------------------------------------------------------
// 5. Export dataset
// -----------------------------------------------------------
Export.table.toAsset({
  collection: labeledSamples,
  description: 'Export_Ghana_ASM_Dataset',
  assetId: 'projects/famm-472015/assets/Ghana_ASM_Dataset_Robust2'
});