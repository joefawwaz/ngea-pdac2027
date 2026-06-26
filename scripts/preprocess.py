"""
preprocess.py, Reusable preprocessing functions for Li-REE prospectivity mapping
NGEA PDAC 2027, Tellus Programme dataset, Ireland

Functions handle:
  - Data loading and harmonisation across ROI/NI survey blocks
  - Censored value treatment
  - CRS reprojection to EPSG:2157 (Irish Transverse Mercator)
  - Centred log-ratio (CLR) transformation
  - Pathfinder ratio engineering
  - Anomaly detection (MAD-based)
  - Geophysical raster sampling at geochemical point locations
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
TARGET_CRS = "EPSG:2157"  # Irish Transverse Mercator

# Canonical element names we want to extract from heterogeneous column naming
# Maps canonical name to list of possible column name patterns (checked in order)
ELEMENT_MAP = {
    # Direct Li-REE indicators
    "Li": ["Li_mgkg_ICPar", "Li_mgkg", "Li"],
    "Rb": ["Rb_mgkg_ICPar", "Rb_mgkg_XRFS", "Rb_mgkg", "Rb"],
    "Cs": ["Cs_mgkg_ICPar", "Cs_mgkg_XRFS", "Cs_mgkg", "Cs"],
    "La": ["La_mgkg_ICPar", "La_mgkg_XRFS", "La_mgkg", "La"],
    "Ce": ["Ce_mgkg_ICPar", "Ce_mgkg_XRFS", "Ce_mgkg", "Ce"],
    "Nd": ["Nd_mgkg_XRFS", "Nd_mgkg", "Nd"],
    "Sm": ["Sm_mgkg_XRFS", "Sm_mgkg", "Sm"],
    "Y":  ["Y_mgkg_ICPar", "Y_mgkg_XRFS", "Y_mgkg", "Y"],
    "Yb": ["Yb_mgkg_ICPar", "Yb_mgkg_XRFS", "Yb_mgkg", "Yb"],
    "Lu": ["Lu_mgkg_ICPar", "Lu_mgkg", "Lu"],
    "Tb": ["Tb_mgkg_ICPar", "Tb_mgkg", "Tb"],
    # Pegmatite pathfinders
    "Nb": ["Nb_mgkg_ICPar", "Nb_mgkg_XRFS", "Nb_mgkg", "Nb"],
    "Ta": ["Ta_mgkg_ICPar", "Ta_mgkg_XRFS", "Ta_mgkg", "Ta"],
    "Be": ["Be_mgkg_ICPar", "Be_mgkg", "Be"],
    "Sn": ["Sn_mgkg_ICPar", "Sn_mgkg_XRFS", "Sn_mgkg", "Sn"],
    # REE pathfinders
    "Th": ["Th_mgkg_ICPar", "Th_mgkg_XRFS", "Th_mgkg", "Th"],
    "U":  ["U_mgkg_ICPar", "U_mgkg_XRFS", "U_mgkg", "U"],
    "Ba": ["Ba_mgkg_ICPar", "Ba_mgkg_XRFS", "Ba_mgkg", "Ba"],
    "Zr": ["Zr_mgkg_ICPar", "Zr_mgkg_XRFS", "Zr_mgkg", "Zr"],
    "P":  ["P_mgkg_ICPar", "P_mgkg", "P"],
    # Additional useful elements
    "K":  ["K_%_ICPar", "K_pct", "K2O_%_XRFS", "K2O"],
    "Sr": ["Sr_mgkg_ICPar", "Sr_mgkg_XRFS", "Sr_mgkg", "Sr"],
    "Ga": ["Ga_mgkg_ICPar", "Ga_mgkg_XRFS", "Ga_mgkg", "Ga"],
    "As": ["As_mgkg_ICPar", "As_mgkg_XRFS", "As_mgkg", "As"],
    "Co": ["Co_mgkg_ICPar", "Co_mgkg_XRFS", "Co_mgkg", "Co"],
    "Ni": ["Ni_mgkg_ICPar", "Ni_mgkg_XRFS", "Ni_mgkg", "Ni"],
    "Cr": ["Cr_mgkg_ICPar", "Cr_mgkg_XRFS", "Cr_mgkg", "Cr"],
    "Cu": ["Cu_mgkg_ICPar", "Cu_mgkg_XRFS", "Cu_mgkg", "Cu"],
    "Zn": ["Zn_mgkg_ICPar", "Zn_mgkg_XRFS", "Zn_mgkg", "Zn"],
    "V":  ["V_mgkg_ICPar", "V_mgkg_XRFS", "V_mgkg", "V"],
    "Pb": ["Pb_mgkg_ICPar", "Pb_mgkg_XRFS", "Pb_mgkg", "Pb"],
}

# Coordinate column mapping for different datasets
COORD_MAPS = {
    "itm": {"x": ["Easting_ITM", "EASTING_ITM"], "y": ["Northing_ITM", "NORTHING_ITM"]},
    "ing": {"x": ["Easting_ING", "X_ING", "EASTING", "Easting", "X"], 
            "y": ["Northing_ING", "Y_ING", "NORTHING", "Northing", "Y"]},
}


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def _find_col(df_cols, candidates):
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df_cols:
            return c
    return None


def _extract_coords(df, source_label=""):
    """Extract x, y coordinates and determine source CRS.
    Returns (x_series, y_series, crs_epsg_int)
    """
    cols = list(df.columns)
    
    # Try ITM first (EPSG:2157)
    xc = _find_col(cols, COORD_MAPS["itm"]["x"])
    yc = _find_col(cols, COORD_MAPS["itm"]["y"])
    if xc and yc:
        return df[xc], df[yc], 2157
    
    # Fall back to Irish National Grid (EPSG:29903)
    xc = _find_col(cols, COORD_MAPS["ing"]["x"])
    yc = _find_col(cols, COORD_MAPS["ing"]["y"])
    if xc and yc:
        # Distinguish ING vs ITM by coordinate range
        # ITM eastings ~400,000-800,000; ING eastings ~0-400,000
        x_mean = pd.to_numeric(df[xc], errors="coerce").mean()
        if x_mean > 400_000:
            return df[xc], df[yc], 2157  # Actually ITM
        else:
            return df[xc], df[yc], 29903  # Irish National Grid
    
    raise ValueError(f"Cannot find coordinate columns in {source_label}. Columns: {cols[:20]}")


def _harmonise_elements(df):
    """Map heterogeneous column names to canonical element names.
    Returns a new DataFrame with canonical column names + 'Sample_ID'.
    """
    result = {}
    
    # Try to get sample ID
    id_col = _find_col(df.columns, ["Sample_ID", "Sample", "Sample_No", "SAMPLE_NO", "SampleID"])
    if id_col:
        result["Sample_ID"] = df[id_col].astype(str)
    else:
        result["Sample_ID"] = [f"S{i}" for i in range(len(df))]
    
    cols = list(df.columns)
    found = []
    missing = []
    
    for elem, candidates in ELEMENT_MAP.items():
        col = _find_col(cols, candidates)
        if col:
            result[elem] = pd.to_numeric(df[col], errors="coerce")
            found.append(elem)
        else:
            missing.append(elem)
    
    return pd.DataFrame(result), found, missing


def _coerce_censored(series, half_dl=True):
    """Handle censored values: '<X' to 0.5*X, 'I.S.' to NaN, etc."""
    if series.dtype == object:
        def _parse(v):
            if isinstance(v, str):
                v = v.strip()
                if v.startswith("<"):
                    try:
                        dl = float(v[1:])
                        return dl * 0.5 if half_dl else dl
                    except ValueError:
                        return np.nan
                if v.upper() in ("I.S.", "N.D.", "ND", "BDL", "-", ""):
                    return np.nan
            try:
                return float(v)
            except (ValueError, TypeError):
                return np.nan
        return series.apply(_parse)
    return pd.to_numeric(series, errors="coerce")


def load_ireland_geochem(base_path):
    """Load and harmonise all Republic of Ireland Tellus geochemical data.
    
    Loads shallow topsoil (A-horizon) from all survey blocks as the primary
    dataset, this has the best Li and trace element coverage via ICP-AR analysis.
    """
    dfs = []
    
    # Shallow Topsoil A blocks
    a_files = [
        ("G1_G3_G6", "Ireland/Geochem/Shallow_Topsoil_A/G1_G3_G6_Blocks/58xxxxA-65xxxxA_Shallow_Topsoil_Download_v1.2.xlsx"),
        ("G5", "Ireland/Geochem/Shallow_Topsoil_A/G5/6117xxA-6174xxA_Shallow_Topsoil_Download_v1.1.xlsx"),
        ("G7_G9", "Ireland/Geochem/Shallow_Topsoil_A/G7_G9/IE_GSI_Geochemistry_Shallow_Topsoil_A_G7_G9_South_East_ITM_6175xxA-6307xxA_Download_v1.0.xlsx"),
        ("G8", "Ireland/Geochem/Shallow_Topsoil_A/G8/IE_GSI_Geochemistry_Shallow_Topsoil_A_ICPar_G8_Dublin_ITM_670xxxA-671xxxA_Download_v1.0.xlsx"),
    ]
    
    for block_name, rel_path in a_files:
        fpath = os.path.join(base_path, rel_path)
        if not os.path.exists(fpath):
            print(f"  ⚠ File not found: {rel_path}")
            continue
        
        print(f"  Loading Ireland A-horizon {block_name}...")
        raw = pd.read_excel(fpath)
        
        # Extract coords
        x, y, crs = _extract_coords(raw, block_name)
        
        # Harmonise elements
        harm, found, miss = _harmonise_elements(raw)
        harm["x"] = pd.to_numeric(x, errors="coerce")
        harm["y"] = pd.to_numeric(y, errors="coerce")
        harm["source_crs"] = crs
        harm["block"] = block_name
        harm["region"] = "ROI"
        harm["sample_type"] = "topsoil_A"
        
        print(f" to {len(harm)} samples, {len(found)} elements found")
        if miss:
            print(f" to Missing: {', '.join(miss[:10])}")
        
        dfs.append(harm)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_ni_geochem(base_path):
    """Load and harmonise Northern Ireland Tellus geochemical data.
    
    Uses Soils A AquaRegia (HDL) as primary, best trace element coverage including Li.
    """
    dfs = []
    
    ni_files = [
        ("NI_A_AquaRegia", "Northern Ireland/2. Geochem/Regional_Soils_A_AquaRegia_HDL.xls", "topsoil_A"),
        ("NI_S_NearTotal", "Northern Ireland/2. Geochem/Regional_Soils_S_NearTotal_HDL.xls", "topsoil_S"),
    ]
    
    for block_name, rel_path, stype in ni_files:
        fpath = os.path.join(base_path, rel_path)
        if not os.path.exists(fpath):
            print(f"  ⚠ File not found: {rel_path}")
            continue
        
        print(f"  Loading {block_name}...")
        raw = pd.read_excel(fpath)
        
        # Coerce censored values in string columns
        for col in raw.columns:
            if raw[col].dtype == object and col not in ["Sample", "SampleType", "Samp_S", "Analytical_method", "Analytical_Method"]:
                raw[col] = _coerce_censored(raw[col])
        
        # NI uses Irish National Grid
        x, y, crs = _extract_coords(raw, block_name)
        
        harm, found, miss = _harmonise_elements(raw)
        harm["x"] = pd.to_numeric(x, errors="coerce")
        harm["y"] = pd.to_numeric(y, errors="coerce")
        harm["source_crs"] = crs
        harm["block"] = block_name
        harm["region"] = "NI"
        harm["sample_type"] = stype
        
        print(f" to {len(harm)} samples, {len(found)} elements found")
        if miss:
            print(f" to Missing: {', '.join(miss[:10])}")
        
        dfs.append(harm)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_li_stream_sediments(base_path):
    """Load the SE Ireland legacy Li stream sediment dataset (AAS reanalysis)."""
    fpath = os.path.join(base_path, 
        "Ireland/Geochem/Lithium_Stream Sediments/SEIreland_Stream_Sediments_Li_AAS_data_20210215.xlsx")
    if not os.path.exists(fpath):
        print("  ⚠ Li stream sediment file not found")
        return pd.DataFrame()
    
    print("  Loading Li stream sediments (SE Ireland)...")
    raw = pd.read_excel(fpath)
    
    # This file has X_ING, Y_ING, X_ITM, Y_ITM, Li_AAS, Sample_ID, Li_Quality
    result = pd.DataFrame()
    result["Sample_ID"] = raw.get("Sample_ID", raw.index).astype(str)
    result["Li"] = pd.to_numeric(raw.get("Li_AAS", raw.get("Li")), errors="coerce")
    
    # Use ITM if available
    if "X_ITM" in raw.columns:
        result["x"] = pd.to_numeric(raw["X_ITM"], errors="coerce")
        result["y"] = pd.to_numeric(raw["Y_ITM"], errors="coerce")
        result["source_crs"] = 2157
    else:
        result["x"] = pd.to_numeric(raw.get("X_ING", raw.get("Easting_ING")), errors="coerce")
        result["y"] = pd.to_numeric(raw.get("Y_ING", raw.get("Northing_ING")), errors="coerce")
        result["source_crs"] = 29903
    
    result["block"] = "Li_StreamSed"
    result["region"] = "ROI"
    result["sample_type"] = "stream_sediment"
    
    print(f" to {len(result)} samples with Li data")
    return result


def reproject_to_itm(df):
    """Reproject all samples to EPSG:2157 (ITM).
    
    Handles mixed CRS within the same DataFrame by splitting, reprojecting, and merging.
    """
    if df.empty:
        return df
    
    # Drop rows with no coordinates
    df = df.dropna(subset=["x", "y"]).copy()
    
    parts = []
    for crs_code, group in df.groupby("source_crs"):
        crs_code = int(crs_code)
        if crs_code == 2157:
            parts.append(group)
        else:
            gdf = gpd.GeoDataFrame(
                group, geometry=gpd.points_from_xy(group["x"], group["y"]),
                crs=f"EPSG:{crs_code}"
            )
            gdf = gdf.to_crs(TARGET_CRS)
            group = group.copy()
            group["x"] = gdf.geometry.x.values
            group["y"] = gdf.geometry.y.values
            group["source_crs"] = 2157
            parts.append(group)
    
    result = pd.concat(parts, ignore_index=True)
    return result


# ──────────────────────────────────────────────
# GEOCHEMICAL TRANSFORMS
# ──────────────────────────────────────────────

def clr_transform(df, element_cols):
    """Apply centred log-ratio (CLR) transformation.
    
    clr(x_i) = ln(x_i / g(x))
    where g(x) = geometric mean of all components.
    
    Zeros/negatives are replaced with half the minimum positive value
    per element before transformation.
    """
    data = df[element_cols].copy()
    
    # Replace zeros/negatives with half-minimum positive
    for col in data.columns:
        pos = data[col][data[col] > 0]
        if len(pos) > 0:
            half_min = pos.min() / 2
            data.loc[data[col] <= 0, col] = half_min
        else:
            data[col] = np.nan
    
    # Drop rows that are all NaN
    valid = data.dropna(how="all")
    if valid.empty:
        return pd.DataFrame(index=df.index, columns=[f"clr_{c}" for c in element_cols])
    
    # CLR transform: ln(x_i / geometric_mean)
    log_data = np.log(data)
    geo_mean = log_data.mean(axis=1)  # This is ln(geometric_mean)
    clr = log_data.sub(geo_mean, axis=0)
    
    clr.columns = [f"clr_{c}" for c in element_cols]
    return clr


def engineer_ratios(df):
    """Engineer geochemical pathfinder ratios.
    
    Only computes ratios where both elements are present.
    Returns a DataFrame of ratio columns.
    """
    ratios = pd.DataFrame(index=df.index)
    
    def _safe_ratio(num, den, name):
        if num in df.columns and den in df.columns:
            n = pd.to_numeric(df[num], errors="coerce")
            d = pd.to_numeric(df[den], errors="coerce")
            d_safe = d.replace(0, np.nan)
            ratios[name] = n / d_safe
            return True
        return False
    
    # K/Rb, pegmatite fractionation proxy (low = evolved)
    # K may be in % while Rb in mg/kg, need to convert K% to ppm
    if "K" in df.columns and "Rb" in df.columns:
        k_ppm = pd.to_numeric(df["K"], errors="coerce")
        # If K appears to be in %, convert to ppm
        if k_ppm.median() < 10:  # Likely percentage
            k_ppm = k_ppm * 10000
        rb = pd.to_numeric(df["Rb"], errors="coerce").replace(0, np.nan)
        ratios["K_Rb"] = k_ppm / rb
    
    _safe_ratio("Rb", "Sr", "Rb_Sr")
    _safe_ratio("Nb", "Ta", "Nb_Ta")  # LCT < 5
    _safe_ratio("Th", "U", "Th_U")
    
    # ΣLREE = La + Ce + Nd (Pr, Sm often missing)
    lree_cols = [c for c in ["La", "Ce", "Nd", "Sm"] if c in df.columns]
    if len(lree_cols) >= 2:
        ratios["sum_LREE"] = df[lree_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    
    # ΣHREE = Y + Yb + Lu + Tb (most HREE missing from Tellus)
    hree_cols = [c for c in ["Y", "Yb", "Lu", "Tb"] if c in df.columns]
    if len(hree_cols) >= 1:
        ratios["sum_HREE"] = df[hree_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    
    # LREE/HREE
    if "sum_LREE" in ratios.columns and "sum_HREE" in ratios.columns:
        hree_safe = ratios["sum_HREE"].replace(0, np.nan)
        ratios["LREE_HREE"] = ratios["sum_LREE"] / hree_safe
    
    return ratios


# ──────────────────────────────────────────────
# ANOMALY DETECTION
# ──────────────────────────────────────────────

def mad_anomaly(series, threshold_multiplier=3.0):
    """Detect anomalies using Median Absolute Deviation (MAD).
    
    More robust than mean ± 2σ for skewed geochemical distributions.
    Anomaly threshold = median + threshold_multiplier * MAD * 1.4826
    (1.4826 is the consistency constant for normal distributions)
    
    Returns: (threshold_value, boolean_mask_of_anomalies)
    """
    s = series.dropna()
    if len(s) == 0:
        return np.nan, pd.Series(False, index=series.index)
    
    med = s.median()
    mad = np.median(np.abs(s - med))
    
    if mad == 0:
        # Use IQR-based fallback
        q75, q25 = s.quantile(0.75), s.quantile(0.25)
        iqr = q75 - q25
        threshold = q75 + threshold_multiplier * iqr
    else:
        threshold = med + threshold_multiplier * 1.4826 * mad
    
    return threshold, series >= threshold


# ──────────────────────────────────────────────
# GEOPHYSICS RASTER SAMPLING
# ──────────────────────────────────────────────

def sample_raster_at_points(raster_path, x_coords, y_coords, band=1):
    """Sample a GeoTIFF raster at given (x, y) coordinate locations.
    
    Uses vectorized coordinate transform + direct numpy array indexing
    for maximum speed (~5-10x faster than rasterio.sample for large arrays).
    
    Returns array of raster values at each point. Out-of-bounds points get NaN.
    """
    try:
        import rasterio
    except ImportError:
        print("  ⚠ rasterio not installed, skipping raster sampling")
        return np.full(len(x_coords), np.nan)
    
    if not os.path.exists(raster_path):
        print(f"  ⚠ Raster not found: {raster_path}")
        return np.full(len(x_coords), np.nan)
    
    n = len(x_coords)
    result = np.full(n, np.nan, dtype=np.float64)
    
    x = np.asarray(x_coords, dtype=np.float64)
    y = np.asarray(y_coords, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    
    if valid.sum() == 0:
        return result
    
    with rasterio.open(raster_path) as src:
        # Guard: refuse RGB visualisation exports (3-band uint8, 0-255). These are
        # colour images, not scientific values, sampling them yields meaningless
        # 0-255 features. Real scientific grids are single-band float.
        if src.count >= 3 and src.dtypes[0] == "uint8":
            print(f"  ⚠ {os.path.basename(raster_path)} looks like a {src.count}-band "
                  f"uint8 RGB image, skipping (not scientific data)")
            return result
        data = src.read(band)
        nodata = src.nodata
        t = src.transform
        
        # Vectorized inverse transform: (x,y) to (col,row)
        xv, yv = x[valid], y[valid]
        cols = ((xv - t.c) / t.a).astype(np.int64)
        rows = ((yv - t.f) / t.e).astype(np.int64)
        
        # Bounds check
        in_bounds = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
        
        # Extract values for in-bounds points
        vals = np.full(valid.sum(), np.nan, dtype=np.float64)
        vals[in_bounds] = data[rows[in_bounds], cols[in_bounds]].astype(np.float64)
        
        # Mask nodata
        if nodata is not None:
            vals[vals == nodata] = np.nan
        
        result[valid] = vals
    
    return result


def sample_all_geophysics(df, base_path):
    """Sample all Ireland Tellus geophysical rasters at geochemical sample locations.
    
    Adds columns: TMI, TDR, K_rad, Th_rad, U_rad, EM_res3 to the dataframe.
    Requires coordinates in EPSG:2157 (ITM).
    """
    import time
    
    # NOTE: point at the single-band scientific grids produced from the Tellus .XYZ
    # line data (block-mean, 500 m, EPSG:2157). The original *_COAST/TIFF rasters in
    # the Tellus download are 3-band uint8 RGB visualisation exports, NOT values.
    raster_map = {
        "TMI":    "Ireland/Geophysics/gridded/TMI.tif",
        "TDR":    "Ireland/Geophysics/gridded/TDR.tif",
        "MAG_1VD": "Ireland/Geophysics/gridded/MAG_1VD.tif",
        "K_rad":  "Ireland/Geophysics/gridded/K_rad.tif",
        "Th_rad": "Ireland/Geophysics/gridded/Th_rad.tif",
        "U_rad":  "Ireland/Geophysics/gridded/U_rad.tif",
        "EM_res3_2F": "Ireland/Geophysics/gridded/EM_res3_2F.tif",
        "EM_res12_2F": "Ireland/Geophysics/gridded/EM_res12_2F.tif",
    }
    
    x = df["x"].values
    y = df["y"].values
    
    t0 = time.time()
    for col_name, rel_path in raster_map.items():
        fpath = os.path.join(base_path, rel_path)
        t1 = time.time()
        df[col_name] = sample_raster_at_points(fpath, x, y)
        dt = time.time() - t1
        n_valid = df[col_name].notna().sum()
        print(f"  Sampling {col_name}... {n_valid} values ({dt:.1f}s)")
    
    total = time.time() - t0
    print(f"  Total geophysics sampling: {total:.1f}s")
    
    return df


# ──────────────────────────────────────────────
# BEDROCK GEOLOGY
# ──────────────────────────────────────────────

def load_bedrock_geology(base_path):
    """Load the GSI bedrock geology classification table."""
    fpath = os.path.join(base_path, "Ireland/Geology/Excel/Bedrock.csv")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    return pd.read_csv(fpath, sep=";")
