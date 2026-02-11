<h1 align="center">
  <br>
  <img src="assets/icon.png" width="300" alt="GMD Logo">
  <br>
  Grassland Mowing Detection (GMD)
  <br>
</h1>

<p align="center">

  <a href="https://colab.research.google.com/drive/17legiv0ExrIMypzcVU4pjnhFNy2WHDOi#scrollTo=QDn6hX7pGqOy">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>

  <a href="https://www.grass4b.com/_sub/">
  <img src="https://img.shields.io/badge/Grasslands4Biodiversity-Website-green" alt="Grasslands4Biodiversity">
</a>

</p>

# Grassland Mowing Detection (GEE Prediction Pipeline)

This repository runs a Google Earth Engine (GEE) batch prediction pipeline to detect grassland mowing events from Sentinel-2 time series. The notebook `notebooks/prediction/gee_prediction.ipynb` is the entry point: it defines a geometry, creates tiles, runs SatSelect to build features, predicts with a pre-trained Random Forest model stored in Earth Engine assets, and exports results to Google Drive.

## What you need

- Python 3.9+ with Jupyter
- A Google Earth Engine account


## Run the prediction notebook

Open the notebook and run all cells:

```bash
jupyter notebook notebooks/prediction/gee_Mowing_prediction.ipynb
```

Inside the notebook, you can change:

- `region` and the asset paths for `aoi` and `mask`
- years in the loop
- the export folder name (`geeGMI`) if you want a different Drive folder

The notebook will:

1) Define geometry and mask
2) Create tiles and filter them by geometry
3) For each tile and year:
   - create features
   - predict with the pre-trained model stored in EE assets
   - export results to Google Drive

Exports are started as Earth Engine tasks and will appear in your Drive after completion.

## Project layout

```
.
├── notebooks/
│   ├── prediction/
│   │   └── gee_Mowing_prediction.ipynb    # Main GEE prediction notebook
│   └── development/
│       ├── 01_cross_validation.ipynb  # Model development
│       ├── 02_spatial_transfer.ipynb  # Model development
│       └── 04_model_development.ipynb # Model development
├── src/
│   └── functions.py               # GEE helper functions + model loader
├── requirements.txt
└── setup.py
```


## License

MIT License. See `LICENSE`.
