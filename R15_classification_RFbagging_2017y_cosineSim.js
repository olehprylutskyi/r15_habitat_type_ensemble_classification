/*
Supervised + unsupervised ensemble classification of the habitat type R15 -
Continental dry rocky steppic grassland and dwarf scrub on chalk outcrops.
https://biodiversity.europa.eu/habitats_eunis_revised/EUNISrev_R15

Earth observation data: Google Satellite Embedding V1 (annual)
https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL

Ground truth: vegetation releves by Anna Kuzemko, 2017
(projects/ee-olegpril12/assets/habitats/releves_r15)

Area of interest: grid cells from Kuzemko et al. (2021) Atlas of the
Grassland Habitats of Ukraine (ISBN 978-617-7849-93-2)
(projects/ee-olegpril12/assets/habitats/grid_ukraine_wgs84_grasslandAtlas_2021)

Strategy (n = 8 presence points, no true absences):
  A) Bagged Random Forest: repeatedly train on the same 8 presence points
     + a freshly-drawn random "other" sample each iteration; average the
     resulting probability surfaces.
  B) Unsupervised centroid-distance (cosine similarity) layer: embedding
     vectors are compared to the mean embedding vector of the 8 presence
     points directly, with no background/negative class needed.
  C) Ensemble: combine A and B. Agreement = high confidence; disagreement
     = flag for field verification.

Classification method: Random Forest (bagged) + cosine similarity ensemble
Author: Oleh Prylutskyi https://orcid.org/0000-0001-5730-517X
*/

// ======================================================================
// 0. USER PARAMETERS
// ======================================================================
// DRAFT (TESTING) MODE: set true while tuning/exploring in order to use less resources. 
// Set false for the final run and before exporting. Only N_ITERATIONS and NUM_TREES are 
// reduced, which is by far the most resourse-demanding part of the script.
var DRAFT = false;

var YEAR = 2017;                    // embedding year closest to the releves
var N_ITERATIONS = DRAFT ? 8 : 30;  // bagging replicates (8 – test run, 30 – production run)
var BACKGROUND_RATIO = 1.5;         // background:presence point ratio per iteration
var EXCLUSION_BUFFER_M = 200;       // buffer around presence pts excluded from bg sampling
var NUM_TREES = DRAFT ? 80 : 300;   // trees per RF, test vs. production mode
var RF_THRESHOLD = 0.5;             // binary threshold on mean RF probability
var HIGH_CONF = 0.6;                // agreement thresholds for combined confidence map
var LOW_CONF = 0.4;
var SAMPLE_SCALE = 10;              // AlpahEarth Embedding native resolution

// Inform user about settings currently applied
print(DRAFT
  ? 'DRAFT mode ON - reduced N_ITERATIONS/NUM_TREES for fast preview. Set DRAFT = false before exporting.'
  : 'DRAFT mode OFF - full settings in use.');

// ======================================================================
// 1. GROUND TRUTH AND AOI PREPARATION
// ======================================================================
// Spatial grid from the Atlas of grassland habitats of Ukraine (Kuzemko et al. 2021)
var aoiFC = ee.FeatureCollection(
  'projects/ee-olegpril12/assets/habitats/grid_ukraine_wgs84_grasslandAtlas_2021');

// Pionts with known habitat type, from curated vegetation releves
var presenceRaw = ee.FeatureCollection(
  'projects/ee-olegpril12/assets/habitats/releves_r15');

// Label presence points as class = 1
var presenceTrain = presenceRaw.map(function (f) {
  return f.set('class', 1);
});

print('N presence points:', presenceTrain.size());

// Calculate the number of background (different from the target class) pts
var N_BACKGROUND = presenceTrain.size().multiply(BACKGROUND_RATIO).round();
print('N background points per iteration (auto = round(' + BACKGROUND_RATIO + ' x n presence)):', N_BACKGROUND);

Map.addLayer(presenceTrain, { color: 'red' }, 'R15 presence points');

// ======================================================================
// 1B. RESTRICT AOI TO THE RELEVANT GRID CELLS
// ======================================================================
var selectedCells = aoiFC.filterBounds(presenceTrain.geometry());
// keeps only the features of aoiFC whose geometry spatially intersects the given geometry.
print('Grid cells intersecting presence points:', selectedCells.size());

var selectedCellsGeometry = selectedCells.geometry().dissolve(ee.ErrorMargin(1));
print('Selected cells area, km2:', selectedCellsGeometry.area(1).divide(1e6));

Map.centerObject(selectedCellsGeometry, 13);
Map.addLayer(selectedCellsGeometry, { color: 'yellow' }, 'Selected grid cells (AOI)', false);

// Background sampling geometry: selected cells minus a buffer around
// presence points, to minimize false coincidence with true presence pts.
var presenceBuffer = presenceTrain.geometry().buffer(EXCLUSION_BUFFER_M);
var aoiMinusBuffer = selectedCellsGeometry.difference(presenceBuffer, ee.ErrorMargin(1));

// ======================================================================
// 2. EARTH OBSERVATION DATA (SATELLITE EMBEDDING V1)
// ======================================================================
var embeddingCol = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

var embeddingImage = embeddingCol
  .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
  .filterBounds(selectedCellsGeometry)
  .mosaic()
  .clip(selectedCellsGeometry);

var bandNames = embeddingImage.bandNames();
print('Embedding bands:', bandNames);

Map.addLayer(
  embeddingImage,
  { min: -0.3, max: 0.3, bands: ['A01', 'A16', 'A09'] },
  'Embeddings ' + YEAR
);

// ======================================================================
// 3A. STRATEGY A - BAGGED RANDOM FOREST (same presence pts,
//     fresh random "other" sample each iteration)
// ======================================================================
// seed: i, passed to smileRandomForest, are seeded from the iteration index.
var iterList = ee.List.sequence(0, N_ITERATIONS - 1);

var probImages = ee.ImageCollection(iterList.map(function (i) {
  i = ee.Number(i);

  var bgPoints = ee.FeatureCollection.randomPoints({
    region: aoiMinusBuffer,
    points: N_BACKGROUND,
    seed: i
  }).map(function (f) {
    return f.set('class', 0);
  });

  var trainingFC = presenceTrain.select(['class']).merge(bgPoints);

  var trainingSamples = embeddingImage.sampleRegions({
    collection: trainingFC,
    properties: ['class'],
    scale: SAMPLE_SCALE,
    tileScale: 4
  });
  
  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: NUM_TREES,
    seed: i
  })
    .setOutputMode('PROBABILITY')
    .train({
      features: trainingSamples,
      classProperty: 'class',
      inputProperties: bandNames
    });

  return embeddingImage.classify(classifier).rename('prob').set('iter', i);
}));

// Measuring prediction uncertainty
// stdDev summarizes, per pixel, how much the 30 replicates disagreed with each
// other; high values flag pixels whose classification is sensitive to
// which random background points happened to be drawn, i.e. where the
// bagging step made the most difference.
var rfProbMean = probImages.mean().rename('rf_prob');
var rfUncertainty = probImages.reduce(ee.Reducer.stdDev()).rename('rf_uncertainty');

Map.addLayer(
  rfProbMean,
  { min: 0, max: 1, palette: ['ffffff', 'ffeda0', 'feb24c', 'f03b20'] },
  'A) Bagged RF probability'
);
Map.addLayer(
  rfUncertainty,
  { min: 0, max: 0.3, palette: ['ffffff', '2c7fb8'] },
  'A) RF ensemble uncertainty (stdDev)',
  false
);

// ======================================================================
// 3B. STRATEGY B - UNSUPERVISED CENTROID-DISTANCE (COSINE SIMILARITY)
// ======================================================================
// reduceRegion(mean) over the presence points' geometry computes, per
// embedding band, the average value across all 8 presence pixels - i.e. its 
// centroid in 64-dimensional embedding space.
var presenceBandMeans = embeddingImage.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: presenceTrain.geometry(),
  scale: SAMPLE_SCALE,
  tileScale: 4
});

// Force band order to match embeddingImage bandNames
var centroidValues = presenceBandMeans.values(bandNames);
var centroidImage = ee.Image.constant(centroidValues).rename(bandNames);

// Cosine similarity = dot product of the two vectors, divided by the
// product of their magnitudes. 
// reduce(ee.Reducer.sum()) collapses the 64 per-band values into one 
// scalar per pixel (i.e. sums across bands, not across pixels).
var dotProd = embeddingImage.multiply(centroidImage).reduce(ee.Reducer.sum());
var embMag = embeddingImage.pow(2).reduce(ee.Reducer.sum()).sqrt();
var centroidMag = centroidImage.pow(2).reduce(ee.Reducer.sum()).sqrt();

var cosineSim = dotProd
  .divide(embMag.multiply(centroidMag))
  .rename('cosine_sim');

// Rescale similarity to 0-1 using 2nd/98th percentile stretch over the AOI
var simPercentiles = cosineSim.reduceRegion({
  reducer: ee.Reducer.percentile([2, 98]),
  geometry: selectedCellsGeometry,
  scale: SAMPLE_SCALE,
  bestEffort: true,
  maxPixels: 1e9,
  tileScale: 8
});
var simP2 = ee.Number(simPercentiles.get('cosine_sim_p2'));
var simP98 = ee.Number(simPercentiles.get('cosine_sim_p98'));

var centroidSimNorm = cosineSim
  .unitScale(simP2, simP98)
  .clamp(0, 1)
  .rename('centroid_similarity');

Map.addLayer(
  centroidSimNorm,
  { min: 0, max: 1, palette: ['ffffff', 'c7e9c0', '41ab5d', '00441b'] },
  'B) Centroid-distance similarity'
);

// ======================================================================
// 4. ENSEMBLE - COMBINE A AND B
// ======================================================================
// Continuous combined confidence: geometric mean (sqrt(A x B)) rather
// than an arithmetic mean (A + B) / 2. This wasy, "combined confidence" 
// to mean "both methods independently agree this is likely R15", not 
// "at least one method thinks so".
var combinedConfidence = rfProbMean
  .multiply(centroidSimNorm)
  .sqrt()
  .rename('combined_confidence');

// Categorical agreement map, built with a "default value + overwrite"
//   1 = high confidence R15   (both methods agree, high)
//   0 = high confidence other (both methods agree, low)
//   2 = disagreement          (methods diverge -> field check)
var agreementClass = ee.Image(2)
  .where(rfProbMean.gte(HIGH_CONF).and(centroidSimNorm.gte(HIGH_CONF)), 1)
  .where(rfProbMean.lte(LOW_CONF).and(centroidSimNorm.lte(LOW_CONF)), 0)
  .rename('agreement_class')
  .clip(selectedCellsGeometry);

var agreementPalette = {
  min: 0,
  max: 2,
  palette: ['3182bd', 'd9d9d9', 'de2d26'] // blue = other, gray = uncertain, red = R15
};

Map.addLayer(combinedConfidence, { min: 0, max: 1, palette: ['ffffff', 'fee391', '993404'] },
  'C) Combined confidence (geometric mean)');
Map.addLayer(agreementClass, agreementPalette, 'C) Agreement / disagreement map');

// FINAL R15 EXTENT MASK - the primary "R15 pixels only" output.
// Requiring both independent methods to agree.
var r15ExtentMask = agreementClass.eq(1).selfMask().rename('R15_extent');
Map.addLayer(r15ExtentMask, { palette: ['de2d26'] },
  'R15 extent (masked, agreement class = 1, thr = ' + HIGH_CONF + ')');

// ======================================================================
// 5. LEAVE-ONE-OUT CROSS-VALIDATION
// ======================================================================
var presenceList = presenceTrain.toList(presenceTrain.size());
var nPresence = presenceTrain.size();

var loocvResults = ee.FeatureCollection(ee.List.sequence(0, nPresence.subtract(1)).map(function (idx) {
  idx = ee.Number(idx);
  var holdOut = ee.Feature(presenceList.get(idx));
  // Standard LOOCV step: train on all presence points EXCEPT the one being tested.
  var trainPts = presenceTrain.filter(
    ee.Filter.neq('system:index', holdOut.get('system:index'))
  );

  var bg = ee.FeatureCollection.randomPoints({
    region: aoiMinusBuffer,
    points: N_BACKGROUND,
    seed: idx.add(1000) // offset from the seeds used in Section 3A, so
                         // the background draws here are independent of
                         // (not a repeat of) the main bagging loop
  }).map(function (f) { return f.set('class', 0); });

  var trainFC = trainPts.select(['class']).merge(bg);
  var samples = embeddingImage.sampleRegions({
    collection: trainFC,
    properties: ['class'],
    scale: SAMPLE_SCALE,
    tileScale: 4
  });

  var clf = ee.Classifier.smileRandomForest({ numberOfTrees: NUM_TREES, seed: idx })
    .setOutputMode('PROBABILITY')
    .train({ features: samples, classProperty: 'class', inputProperties: bandNames });

  var predictedProb = embeddingImage.classify(clf)
    .reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: holdOut.geometry(),
      scale: SAMPLE_SCALE
    })
    .get('classification');

  return ee.Feature(null, { held_out_index: idx, predicted_prob: predictedProb });
}));

print('LOOCV predicted probability at each held-out presence point '
  + '(all should ideally be > 0.5):', loocvResults);

// ======================================================================
// 6. TOTAL R15 AREA
// ======================================================================
var r15AreaImage = ee.Image.pixelArea().updateMask(r15ExtentMask);
var r15AreaStats = r15AreaImage.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: selectedCellsGeometry,
  scale: SAMPLE_SCALE,
  maxPixels: 1e10,
  bestEffort: true,
  tileScale: 16
});
var r15AreaM2 = ee.Number(r15AreaStats.get('area'));
var r15AreaHa = r15AreaM2.divide(1e4);
var r15AreaKm2 = r15AreaM2.divide(1e6);

print('Total predicted R15 area, ha:', r15AreaHa);
print('Total predicted R15 area, km2:', r15AreaKm2);

// ======================================================================
// 7. METRICS TABLE
// ======================================================================
// Parameters and summary statistics
var loocvMeanProb = loocvResults.aggregate_mean('predicted_prob');
var loocvMinProb = loocvResults.aggregate_min('predicted_prob');

var metricsFeature = ee.Feature(null, {
  habitat_code: 'R15',
  embedding_dataset: 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL',
  embedding_year: YEAR,
  sample_scale_m: SAMPLE_SCALE,
  n_presence_points: presenceTrain.size(),
  n_background_points_per_iteration: N_BACKGROUND,
  exclusion_buffer_m: EXCLUSION_BUFFER_M,
  n_bagging_iterations: N_ITERATIONS,
  n_trees_per_forest: NUM_TREES,
  rf_probability_threshold: RF_THRESHOLD,
  agreement_high_confidence_threshold: HIGH_CONF,
  agreement_low_confidence_threshold: LOW_CONF,
  n_selected_grid_cells: selectedCells.size(),
  aoi_area_km2: selectedCellsGeometry.area(1).divide(1e6),
  r15_predicted_area_ha: r15AreaHa,
  r15_predicted_area_km2: r15AreaKm2,
  loocv_mean_predicted_prob: loocvMeanProb,
  loocv_min_predicted_prob: loocvMinProb,
  draft_mode_used: DRAFT
});

var metricsTable = ee.FeatureCollection([metricsFeature]);
print('Publication metrics:', metricsFeature);

// ======================================================================
// 8. EXPORTS
// ======================================================================
//   band 1: rf_prob             (Strategy A, bagged RF probability)
//   band 2: centroid_similarity (Strategy B, cosine similarity)
//   band 3: combined_confidence (Strategy C, geometric mean of A and B)
//   band 4: agreement_class     (0/1/2 categorical map, see Section 4)
Export.image.toDrive({
  image: rfProbMean.addBands(centroidSimNorm).addBands(combinedConfidence)
           .addBands(agreementClass).toFloat(),
  description: 'R15_ensemble_outputs',
  folder: 'GEE_data',
  region: selectedCellsGeometry,
  scale: SAMPLE_SCALE,
  maxPixels: 1e10
});

// R15-only extent image: single band, value 1 = R15, everything else
// masked (NoData).
Export.image.toDrive({
  image: r15ExtentMask.toFloat(),
  description: 'R15_extent_masked',
  folder: 'GEE_data',
  region: selectedCellsGeometry,
  scale: SAMPLE_SCALE,
  maxPixels: 1e10
});

// LOOCV results exported separately as a table (CSV)
Export.table.toDrive({
  collection: loocvResults,
  description: 'R15_LOOCV_results',
  folder: 'GEE_data',
  fileFormat: 'CSV'
});

// Publication metrics table (Section 7)
Export.table.toDrive({
  collection: metricsTable,
  description: 'R15_publication_metrics',
  folder: 'GEE_data',
  fileFormat: 'CSV'
});

// End of the script
