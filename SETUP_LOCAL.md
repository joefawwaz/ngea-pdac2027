# NGEA PDAC 2027, Local Setup (Mac)

## 1. Folder layout

Put these files into `~/Documents/ngea/` so it looks like this:

```
~/Documents/ngea/
├── notebooks/
│   ├── lct_prospectivity_v3.ipynb      # LCT (Li) pegmatite model
│   ├── lct_prospectivity_v3.py
│   ├── ree_anomaly_v4.ipynb            # REE label-free assessment
│   ├── ree_anomaly_v4.py
│   ├── lct_ree_comparison.py           # two-granite end-member figure
│   └── metallogenic_timeline.py        # geologic-time context figure
├── scripts/
│   └── preprocess.py                   # shared data loaders (both notebooks import this)
├── outputs/                            # auto-created on first run
├── Ireland/                            # your existing Tellus data
│   ├── Geochem/
│   ├── Geophysics/
│   └── Geology/
└── Northern Ireland/
    ├── 2. Geochem/
    └── 1. Geophysic/
```

The notebooks auto-detect the base path: when run from `~/Documents/ngea/notebooks/`,
they walk up and find the `Ireland/` folder. No path editing needed as long as the
layout matches above.

## 2. Conda environment

```bash
conda create -n ngea python=3.11 -y
conda activate ngea
conda install -c conda-forge -y \
  numpy pandas scipy scikit-learn \
  geopandas rasterio shapely pyproj fiona \
  matplotlib seaborn shap \
  jupyterlab ipykernel openpyxl xlrd folium tqdm
python -m ipykernel install --user --name ngea --display-name "Python (ngea)"
```

## 3. Run

```bash
cd ~/Documents/ngea
conda activate ngea
jupyter lab
```

Open a notebook, select the "Python (ngea)" kernel, set `DEV_MODE = True` for a fast
(~2 min) test run, then `DEV_MODE = False` for the full run.

## 4. KNOWN PENDING FIX, geophysics

`scripts/preprocess.py` still points to the old `.tif` geophysics rasters, which are
RGB visualization exports (NOT scientific values). The real data is in the Geosoft
`.XYZ` files (MAG_MERGE_DATA_2022, RAD_MERGE_DATA_2022, EM_*). These need to be:
  1. copied from OneDrive into `Ireland/Geophysics/`
  2. gridded once into single-band GeoTIFFs

Until that's done, the geophysics features are unreliable. The geochemistry, bedrock
geology, and labelling all work correctly. This is the next task.

## 5. Other pending fixes (not yet wired in)
- Replace ellipse granite approximations with real bedrock polygons
  (`IE_GSI_GSNI_Bedrock_Geology_1M_IE32_ITM_MS.shp`)
- Replace literature Li coordinates with real GSI occurrences
  (`IE_GSI_Mineral_Locations_IE26_ITM.shp`, filter MINERAL contains 'lith')
- Grid NI geophysics CSVs to extend coverage into Northern Ireland
