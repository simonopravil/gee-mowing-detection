# Data Directory

This directory contains the training and test data for the grassland mowing detection model.

## Directory Structure

```
data/
├── train/                    # Training data
│   ├── GMI_Long.gpkg        # Reference mowing events (GeoPackage)
│   ├── S1/                  # Sentinel-1 backscatter CSV time-series
│   ├── S2/                  # Sentinel-2 optical CSV time-series
│   └── COH/                 # Coherence CSV time-series
└── test/                    # Test/validation data
    ├── SK_long.gpkg         # Slovakia reference mowing events
    ├── SK_poly/             # Slovakia Sentinel-2 time-series
    ├── CH_long.gpkg         # Switzerland reference mowing events
    └── CH_poly/             # Switzerland Sentinel-2 time-series
```



## Notes

- Data files are excluded from version control via `.gitignore` due to size
- Contact the authors for access to the training data
- Users should provide their own data following the structure above
