var imageCollection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED"),
    imageCollection2 = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY"),
    geometry = /* color: #b41803 */ee.Geometry.MultiPolygon(
        [[[[-2.577107401987939, 6.5306601480568975],
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
           [-2.5399051375932435, 6.306703638207781]]]]);
// Load mining polygons
var gm = ee.FeatureCollection("projects/famm-472015/assets/Mining_Polygons");

// Filter making sure we only take polygons inside Ghana AOI (your 'geometry')
var gh_polygons = gm.filterBounds(geometry);

// Merge polygons into single AOI used throughout the script
var miningAOI = gh_polygons.geometry();

// Override AOI in the composite workflow
var aoi = miningAOI;

// Visualize for sanity check
Map.centerObject(aoi, 10);
Map.addLayer(gh_polygons, {color: 'red'}, 'Mining Areas');
