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
          [-1.6499660314807807, 6.831115397308761]]]),
    imageCollection3 = ee.ImageCollection("projects/famm-472015/assets/2025_Composites_");
// ==============================
// 0️⃣ USER INPUTS
// ==============================
var aoi = geometry;

var wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons");

var forestReserves = wdpa
  .filterBounds(aoi)
  .filter(ee.Filter.eq('ISO3', 'GHA'))
  .filter(ee.Filter.eq('REALM', 'Terrestrial'))
  .filter(ee.Filter.inList('STATUS', ['Designated', 'Inscribed']))
  .filter(ee.Filter.stringContains('DESIG_ENG', 'Forest'));

print('Forest reserves in AOI:', forestReserves.size());
print('Unique designations:', forestReserves.aggregate_histogram('DESIG_ENG'));

// 🔑 UNION GEOMETRY (CRITICAL)
var forestGeom = forestReserves.geometry();

// ==============================
// 2️⃣ LOAD MINING POLYGONS (ASM = POSITIVE)
// ==============================
var gm = ee.FeatureCollection("projects/famm-472015/assets/Mining_Polygons");

var gh_mines_aoi = gm
  .filter(ee.Filter.eq('GID_0', 'GHA'))
  .filterBounds(aoi);

var pos_polygons = gh_mines_aoi
  .filter(ee.Filter.lt('Area_Num', 1e6))
  .map(function(f) {
    return f.set('label', 1);
  });

// ==============================
// 3️⃣ GENERATE NEGATIVE (NON-ASM) POLYGONS
// ==============================
var numNeg = 700;
var halfSize = 1280;

var randomPts = ee.FeatureCollection.randomPoints({
  region: aoi,
  points: numNeg * 3,
  seed: 42
});

var neg_polygons = randomPts.map(function(f) {
  var square = f.geometry().buffer(halfSize).bounds();
  var intersectsASM = gh_mines_aoi.filterBounds(square).size().gt(0);

  return ee.Feature(square)
    .set('label', 0)
    .set('valid', intersectsASM.not());
})
.filter(ee.Filter.eq('valid', 1))
.limit(numNeg);

// ==============================
// 4️⃣ FLAG FOREST RESERVE INTERSECTION (SAFE)
// ==============================
function flagForestReserve(f) {
  var intersects = f.geometry().intersects(forestGeom, 1);

  return f.set(
    'in_forest_reserve',
    ee.Algorithms.If(intersects, 1, 0)
  );
}


var pos_flagged = pos_polygons.map(flagForestReserve);
var neg_flagged = neg_polygons.map(flagForestReserve);

// ==============================
// 5️⃣ FINAL DATASET
// ==============================
var labeledSamples = pos_flagged.merge(neg_flagged);

print(
  'in_forest_reserve distribution:',
  labeledSamples.aggregate_histogram('in_forest_reserve')
);

// ==============================
// 6️⃣ DIAGNOSTICS
// ==============================
var pos_in_fr  = pos_flagged.filter(ee.Filter.eq('in_forest_reserve', 1));
var pos_out_fr = pos_flagged.filter(ee.Filter.eq('in_forest_reserve', 0));

var neg_in_fr  = neg_flagged.filter(ee.Filter.eq('in_forest_reserve', 1));
var neg_out_fr = neg_flagged.filter(ee.Filter.eq('in_forest_reserve', 0));

print('ASM inside FR:', pos_in_fr.size());
print('ASM outside FR:', pos_out_fr.size());
print('Negatives inside FR:', neg_in_fr.size());
print('Negatives outside FR:', neg_out_fr.size());

// ==============================
// 7️⃣ MAP
// ==============================
Map.centerObject(aoi, 8);
Map.addLayer(forestReserves, {color: 'green'}, 'Forest Reserves');
Map.addLayer(pos_in_fr, {color: 'yellow'}, 'ASM inside FR');
Map.addLayer(neg_in_fr, {color: 'cyan'}, 'Negatives inside FR');

// ==============================
// 8️⃣ EXPORT
// ==============================
Export.table.toAsset({
  collection: labeledSamples,
  description: 'Export_Ghana_ASM_Dataset_With_Forest_Reserves',
  assetId: 'projects/famm-472015/assets/Ghana_ASM_Dataset_WithForestReserves'
});
