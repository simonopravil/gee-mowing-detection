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

## Data Format

### Reference Data (GeoPackage)

The reference mowing events are stored in GeoPackage format with the following attributes:

| Column | Description |
|--------|-------------|
| `PLOTID` / `plotID` | Unique plot identifier |
| `date` | Date of the mowing event |
| `year` | Year of the event |
| `type` | Event type (`k` = mowing, `n` = no mowing observed) |

### Time-Series Data (CSV)

The satellite time-series data is stored as CSV files with one file per plot. Each CSV contains:

| Column | Description |
|--------|-------------|
| `PLOTID` | Plot identifier |
| `Date` | Observation date |
| `blue`, `green`, `red` | Visible band reflectances |
| `nir`, `re1`, `re2`, `re3` | Near-infrared and red-edge bands |
| `swir1`, `swir2` | Shortwave infrared bands |
| `ndvi` | Normalized Difference Vegetation Index |
| `spatial_max` | 90th percentile NDVI within spatial neighborhood |
| `ndvi_texture_sd_3` | NDVI texture (std dev) at 3x3 window |
| `ndvi_texture_sd_5` | NDVI texture (std dev) at 5x5 window |

## Data Acquisition

The satellite data was acquired from:

- **Sentinel-2**: Level-2A surface reflectance products from Google Earth Engine
- **Processing**: Cloud masking, atmospheric correction, and temporal compositing

Reference mowing events were collected through:
- Field observations
- Camera trap monitoring
- Farmer interviews

## Training Regions

The training data includes three spatial regions used for leave-one-out cross-validation:
- **LR**: [Region description]
- **NB**: [Region description]
- **NS**: [Region description]

## Notes

- Data files are excluded from version control via `.gitignore` due to size
- Contact the authors for access to the training data
- Users should provide their own data following the structure above
