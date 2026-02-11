# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a grassland mowing event detection system using satellite time-series data (Sentinel-1 SAR and Sentinel-2 optical) with Random Forest classification. The project detects when grassland plots are mowed based on spectral/temporal changes in vegetation indices.

## Architecture

### Core Modules (in `src/`)

- **utils.py**: Main utility library containing:
  - Data loading/preprocessing (`prepare_sequences`, `load_and_concat_csv`, `interpolate_s2`)
  - Spatial cross-validation (`cv_leave_one_out_split`)
  - Temporal feature engineering (`add_temporal_features`) - uses only past observations to prevent data leakage
  - Custom evaluation metrics with temporal tolerance windows (`custom_f1_score`, `custom_error_matrix`)
  - Optuna hyperparameter optimization (`create_optuna_objective`)

- **functions.py**: Google Earth Engine helper functions for:
  - Grid creation for tiled exports (`create_grid`)
  - Time-series array manipulation (`get_shift`, `CreateDiffColl`)
  - Spatial aggregation (`reduce_quantile`)
  - Model prediction and post-processing (`predict_model`, `filter_close_mowing_events`, `mowing_results`)

- **management.py**: Management event detection utilities:
  - Close mowing removal (`remove_close_mowing`)
  - Grazing detection functions
  - Time shift calculations

### Notebooks (in `notebooks/`)

- **01_cross_validation.ipynb**: Leave-one-region-out cross-validation training with Optuna optimization
- **02_spatial_transfer.ipynb**: Spatial transfer validation to Slovakia and Switzerland test sites
- **03_gee_prediction.ipynb**: Google Earth Engine batch prediction pipeline for large-scale inference
- **04_model_development.ipynb**: Legacy model development notebook

### Data Structure

```
data/
├── train/
│   ├── GMI_Long.gpkg      # Reference mowing events (GeoPackage)
│   ├── S1/                # Sentinel-1 backscatter CSV time-series
│   ├── S2/                # Sentinel-2 optical CSV time-series
│   └── COH/               # Coherence CSV time-series
└── test/
    ├── SK_long.gpkg       # Slovakia validation data
    ├── SK_poly/           # Slovakia S2 time-series
    ├── CH_long.gpkg       # Switzerland validation data
    └── CH_poly/           # Switzerland S2 time-series
```

## Key Concepts

### Feature Engineering
- Temporal differences (t - t-1) for all bands including BSI and NDVI
- `diff_max`: Difference between pixel NDVI and 90th percentile spatial NDVI (pNDVI)
- NDVI texture features (standard deviation at 3x3 and 5x5 windows)
- Day-of-year (DOY) features

### Model Variables
The model uses 27 features combining spectral bands, vegetation indices, texture metrics, and their temporal differences:
```python
model_vars = [
    'blue_diff_1', 'green_diff_1', 'red_diff_1', 'nir_diff_1', 're1_diff_1',
    're2_diff_1', 're3_diff_1', 'swir1_diff_1', 'swir2_diff_1', 'ndvi_diff_1',
    'doy_diff_1', 'bsi_diff_1', 'blue', 'green', 'red', 'nir', 're1', 're2',
    're3', 'swir1', 'swir2', 'ndvi', 'bsi', 'diff_max', 'ndvi_texture_sd_5',
    'ndvi_texture_sd_5_diff_1', 'ndvi_texture_sd_3', 'ndvi_texture_sd_3_diff_1'
]
```

### Evaluation Metrics
- Custom F1 score with temporal tolerance window (default: 3 days before, 20 days after)
- DOY comparison for timing accuracy (RMSE, MAE, R²)
- Mowing frequency metrics (MAPE, offset)

### Training Regions
- LR, NB, NS: Three spatial regions used for leave-one-out cross-validation

## Dependencies

Core scientific stack:
- numpy, pandas, geopandas
- scikit-learn (RandomForestClassifier)
- optuna (hyperparameter optimization)
- scipy (Savitzky-Golay filtering)
- matplotlib, seaborn (visualization)
- earthengine-api, geemap (GEE inference)
- joblib (model serialization)

For GEE prediction notebook:
- SatSelect package: https://github.com/simonopravil/SatSelect

## Common Workflows

### Train model with cross-validation
Run `notebooks/01_cross_validation.ipynb` - performs leave-one-region-out CV with Optuna optimization

### Evaluate spatial transfer
Run `notebooks/02_spatial_transfer.ipynb` - loads trained model and evaluates on SK/CH test sites

### Run GEE prediction
Run `notebooks/03_gee_prediction.ipynb` - requires `ee.Authenticate()` and exports tiled results to Google Drive
