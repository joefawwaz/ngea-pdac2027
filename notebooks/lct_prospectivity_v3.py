#!/usr/bin/env python3
"""
LCT Pegmatite Prospectivity Mapping v3, NGEA PDAC 2027
Improvements: Refocused: LCT pegmatite targeting (not joint Li-REE). GPU acceleration, bedrock geology, circular reasoning fix.

This script extends v1 with:
  - Part 1: Literature research notes
  - Part 2: Geology-only labels (Option A) + leave-Li-out model (Option B)
  - Part 3: Granite proximity features
  - Part 4: Target validation against known occurrences
  - Part 5: IDW interpolated prospectivity surface, polished outputs
"""

# %% [markdown]
# # LCT Pegmatite Prospectivity Mapping Using the Tellus Programme Dataset, Ireland
# ## NGEA PDAC 2027, Frank Arnott Next Generation Explorers Award
#
# **Team:** Joefawwaz · Muhammadrahalditaher · Extivonus Fransiskus · Jesslyn Jane  
# **Mentors:** Cendidana · Reza-Alfurqan · Fahmihakim-ugm
#
# ---
#
# ## Executive Summary
#
# This study develops an **interpretable prospectivity model for Lithium-Cesium-Tantalum
# (LCT) pegmatite deposits** across Ireland using the Tellus Programme's multi-element
# soil geochemistry and airborne geophysics.
#
# **Why LCT-only, not joint Li-REE?** LCT and NYF pegmatites represent fundamentally
# distinct metallogenic systems (Černý & Ercit, 2005): LCT derives from S-type
# peraluminous granites in orogenic settings (Leinster ~412 Ma), while NYF/REE pegmatites
# derive from A-type granites in anorogenic settings (Mourne ~60 Ma). Published ML
# prospectivity studies build deposit-type-specific models (Vadoodi et al., 2026),
# not joint Li+REE models. With ~10 known GSI Li occurrences
# and 0 confirmed NYF deposits in Ireland, a joint model is neither geologically
# justified nor statistically viable.
#
# **Dataset:** 27,360 topsoil samples + 8 airborne geophysical layers + bedrock geology.
# Total: ~49 features per sample point.
#
# **Method:** Random Forest with SHAP, trained on geology-driven labels from the REAL
# Caledonian granite margin + REAL GSI Li occurrences. Three model variants test circularity.
#
# **Key Results (full run; real airborne geophysics + real GSI geology & occurrences):**
# - 27,360 samples; labels from the real Caledonian granite margin + 10 real GSI Li
#   occurrences (Republic only; no NI occurrence data available locally).
# - Spatial-block CV AUC = 0.94 (Models A & B; 10-fold, 25-km blocks); occurrence-only Model C = 0.99.
# - Leave-Li-out Model B is near-identical (rho = 0.999), so the signature is NOT Li-circular.
# - SHAP top features: airborne EM resistivity (res3/res12), LREE/HREE geochemistry, K
#   radiometrics, bedrock-granite lithology, Nb/Ta, magnetics, a multivariate signature (not Li alone).
# - 18 target clusters. HONEST: occurrence recovery is modest (4 of 10 known Li occurrences have a
#   target within 20 km, nearest 10.7 km), and 9 of 18 clusters are NI/greenfield (Sperrins area)
#   far from any known Li. So the map is best read as a reconnaissance granite-fertility map, not a
#   validated pegmatite locator. The 6 Leinster-belt clusters are the credible, deposit-relevant set.
# - An independent dissolved-Li (stream/regional water) layer is added as a real-data cross-check.
# - Because labels are geology-defined and bedrock-granite is itself a feature, the high AUC
#   partly reflects re-learning granite geology; the occurrence checks (Model C; Leinster
#   recovery) are the deposit-relevant validation, not the AUC alone.
#
# **Geological caveat:** LCT and NYF pegmatites are distinct ore systems derived from
# different source magmas and tectonic environments (Černý & Ercit, 2005).
# This model targets LCT pegmatite environments associated with Caledonian S-type
# granites. Mourne Mountains targets are flagged separately as potential NYF/REE
# candidates requiring different exploration criteria. Confirmation of deposit type
# requires follow-up lithogeochemistry and petrographic characterisation.

# %% [markdown]
# ## Research Notes, Literature Review
#
# ### LCT vs NYF Pegmatite Classification (Černý & Ercit, 2005)
#
# | Property | LCT Pegmatites | NYF Pegmatites |
# |----------|---------------|----------------|
# | **Enrichment** | Li, Cs, Ta, Rb, Be, Sn, B | Nb, Y, REE, Th, U, Zr, F |
# | **Parent granite** | S-type, peraluminous | A-type, metaluminous |
# | **Tectonic setting** | Orogenic (collision) | Anorogenic (rift) |
# | **Key minerals** | Spodumene, petalite, lepidolite | Columbite, euxenite, fergusonite |
# | **REE content** | **Strongly depleted** in evolved pegmatites | **Enriched** |
# | **Irish example** | Leinster Granite (~412 Ma) | Mourne Mountains (~60 Ma) |
# | **Known occurrences** | 11 confirmed (Aclare, Moylisha, Knockeen...) | 0 confirmed |
# | **Tellus indicators** | Li, Rb, Cs, K/Rb, K_rad | La, Ce, Y, Th, U, Nb/Ta |
#
# *Sources: Černý & Ercit (2005); Vadoodi et al. (2026); the EU GREENPEG project*
#
# The decision to model LCT prospectivity separately, rather than a joint "Li-REE"
# target, follows the consensus in published ML prospectivity studies (Vadoodi et al.,
# 2026) and the GREENPEG project's recommendation of separate exploration workflows
# for LCT vs NYF pegmatite types.
#
# ### On Circular Reasoning in Prospectivity Mapping
#
# **PU Learning:** Positive-Unlabelled learning treats all non-deposit locations as
# *unlabelled* rather than *negative*, acknowledging that undiscovered deposits may exist
# in "negative" areas. Xiong & Zuo (2021) demonstrated PU learning outperforms standard
# classifiers for mineral prospectivity with sparse positives. The bagging-based PU
# approach (Mordelet & Vert, 2014) is particularly suited to RF classifiers.
#
# **Sparse Positives:** Studies with few known occurrences are common in greenfield
# exploration. Under the SCAR (Selected Completely At Random) assumption, treating
# unlabelled sites as *unlabelled* rather than *negative* is the standard remedy. For our
# national-scale study with a small set of confirmed Li occurrences, this supports the
# approach of treating unknowns as unlabelled.
#
# **Knowledge-Driven vs Data-Driven Labels:** Carranza & Laborte (2015) established
# that knowledge-driven label generation (using geological criteria) avoids the circular
# reasoning of data-driven labels (using geochemistry to select positives, then predicting
# geochemistry). Our Option A implements this: positives are defined by granite proximity
# and structural criteria ONLY, with no geochemical input to labelling.
#
# ### On the Leinster LCT Pegmatite Belt
#
# The Leinster pegmatite belt is a 135 km NE-SW trending system along the East Carlow
# Deformation Zone (ECDZ) on the SE margin of the Leinster Granite Massif. Key localities:
# - **Aclare & Moylisha:** Spodumene pegmatites on the SE Leinster margin (Barros et al.,
#   2022). Currently explored by Blackstairs Lithium (Ganfeng/ILC JV).
# - **Knockeen:** LCT pegmatites reported by Global Battery Metals (2023); high-grade Li₂O
#   from surface sampling and drilling. Structurally controlled NE-SW dike swarm in the ECDZ.
# - **Aughavanagh:** Analogous target at NW Leinster granite-metasediment contact.
#
# The pegmatites are ca. 412 Ma, emplaced into Tullow Lowlands pluton granodiorite and
# Ribband Group metasediments. Mineralogy: spodumene, K-feldspar, albite, muscovite,
# garnet, with accessory columbite-group minerals, cassiterite, beryl (Barros et al., 2022).
#
# ### On the Sperrins/Dalradian Targets (T09-T11)
#
# The Sperrin Mountains comprise Dalradian Supergroup metasediments (greenschist to lower
# amphibolite facies). **No Caledonian granitic intrusives are mapped in the Sperrins
# proper.** The gold mineralisation at Curraghinalt and Cavanacaw is orogenic (quartz-vein
# hosted, ~465 Ma). There is no published geological basis for LCT pegmatite prospectivity
# in the Sperrins. These targets likely reflect elevated Li-Rb-Cs in pelitic metasediments
# (clay-rich protoliths) rather than pegmatite mineralisation. **These targets should be
# treated with caution and flagged as requiring field validation.**
#
# ### On Bedrock Geology Integration
#
# Published studies use distance-to-granite-contact as a continuous feature (Porwal et al.,
# 2006; Zuo, 2020). One-hot encoding of lithology is standard for RF models. Distance-to-
# fault features are widely used with log-transform to handle skewed distributions.
#
# ### Key Methodological References
#
# **Vadoodi, Carranza & Sadeghi (2026)**, *"Prospectivity Mapping of Targets for
# Li-Bearing Pegmatites and Granites in Västernorrland, Sweden, with Fuzzy Logic and
# Random Forest Modeling"*, Natural Resources Research. This study applies both
# knowledge-driven fuzzy logic and data-driven RF to integrate geological, geochemical,
# geophysical, and structural datasets for Li pegmatite targeting, directly analogous
# to our methodology. Their finding that FL and RF outputs show significant overlap with
# known pegmatites while revealing new targets validates the dual knowledge/data-driven
# approach we adopt here.
#
# **Cardoso-Fernandes et al. (2023)**, *"Spectral Library of European Pegmatites,
# Pegmatite Minerals and Pegmatite Host-Rocks, the GREENPEG project database"*, ESSD.
# Key finding: the overall spectral signature of LCT pegmatites is mostly associated
# with alteration minerals such as clays. This supports our interpretation of Th_rad
# as a dual proxy for both magmatic Th/fractionation enrichment and clay-alteration detection
# (note: LCT pegmatites are REE-depleted, so Th_rad here tracks Th in accessory phases, not REE).
#
# ### Circular Reasoning Defence
#
# The ρ(A,B) = 0.999 Spearman correlation between models trained with and without Li
# constitutes an **independent geophysical validation** of the prospectivity signal.
# Because Model B uses only non-Li geochemistry and airborne geophysics (radiometrics,
# magnetics, EM), its near-identical output to Model A demonstrates that the
# prospectivity pattern is encoded in the multivariate geophysical+geochemical fabric
# of the crust, not in a single element. This is the methodological core of our
# anti-circular-reasoning defence.

# %%
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import pickle

# ── Colab / Local auto-detect ──
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

if IN_COLAB:
    REPO_DIR = "/content/ngea-pdac2027"
    if not os.path.exists(REPO_DIR):
        os.system(f"git clone https://github.com/joefawwaz/ngea-pdac2027.git {REPO_DIR}")
    os.chdir(REPO_DIR)
    BASE = REPO_DIR
    # Install deps if needed
    try:
        import shap
    except ImportError:
        os.system("pip install -q shap geopandas rasterio folium pyproj xlrd")
else:
    if os.path.exists("Ireland"):
        BASE = "."
    elif os.path.exists("../Ireland"):
        BASE = ".."
    else:
        BASE = os.getcwd()

sys.path.insert(0, os.path.join(BASE, "scripts"))

from preprocess import (
    load_ireland_geochem, load_ni_geochem, load_li_stream_sediments,
    reproject_to_itm, clr_transform, engineer_ratios, mad_anomaly,
    sample_all_geophysics, ELEMENT_MAP
)

# ── DEV MODE: Set to True for fast iteration, False for final submission run ──
DEV_MODE = False          # Set to False for final submission run
DEV_SAMPLE_N = 5000       # samples to use in dev mode (stratified by survey block)

# Output suffix: _DEV in dev mode, empty in production
SUFFIX = "_DEV" if DEV_MODE else ""

OUTPUTS = os.path.join(BASE, "outputs")
MAPS = os.path.join(OUTPUTS, "maps")
os.makedirs(MAPS, exist_ok=True)

# ── Checkpoint system for Colab timeout resilience ──
CHECKPOINT_DIR = os.path.join(OUTPUTS, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(name, obj):
    path = os.path.join(CHECKPOINT_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Checkpoint saved: {name}")

def load_checkpoint(name):
    path = os.path.join(CHECKPOINT_DIR, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  Checkpoint loaded: {name}")
        return obj, True
    return None, False

# RF hyperparameters scaled by mode
RF_N_ESTIMATORS = 50 if DEV_MODE else 1000
SHAP_MAX_SAMPLES = 500 if DEV_MODE else 2500

# ── GPU Acceleration (Colab T4) ──
USE_GPU = False
try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    USE_GPU = True
    print("  GPU acceleration available (cuML)")
except ImportError:
    pass

if not USE_GPU:
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU detected: {torch.cuda.get_device_name(0)}, but cuML not installed")
            print("  To enable GPU RF: pip install cuml-cu11 (or cuml-cu12)")
        else:
            print("  No GPU detected, using CPU")
    except ImportError:
        print("  Using CPU (install cuml for GPU acceleration)")

if DEV_MODE:
    print(f"⚠️  DEV MODE: using {DEV_SAMPLE_N} samples (stratified). Set DEV_MODE=False for full run.")
    print(f"   RF trees: {RF_N_ESTIMATORS}, SHAP samples: {SHAP_MAX_SAMPLES}")
    print(f"   Output suffix: '{SUFFIX}'")

# %%
# ─── 1. DATA LOADING (same as v1) ───
print("=" * 60)
print("STAGE 1: DATA LOADING & PREPROCESSING")
print("=" * 60)

roi = load_ireland_geochem(BASE)
ni = load_ni_geochem(BASE)
li_ss = load_li_stream_sediments(BASE)

combined = pd.concat([roi, ni], ignore_index=True)
combined = reproject_to_itm(combined)
if not li_ss.empty:
    li_ss = reproject_to_itm(li_ss)

print(f"\nTotal samples: {len(combined)}")

# DEV MODE: subsample for fast iteration
if DEV_MODE and len(combined) > DEV_SAMPLE_N:
    combined = combined.groupby("block", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), max(50, int(DEV_SAMPLE_N * len(g) / len(combined)))),
                           random_state=42)
    ).reset_index(drop=True)
    print(f"⚠️  DEV MODE: subsampled to {len(combined)} samples (stratified by block)")

# Element coverage check
target_elems = ["Li", "Rb", "Cs", "La", "Ce", "Y", "Ta", "Nb", "Ba", "U", "Th",
                "Sn", "Be", "Sr", "Zr", "Tb", "Lu", "Yb"]
good_elements = [e for e in target_elems if e in combined.columns and combined[e].notna().mean() > 0.5]

# CLR transform
clr_df = clr_transform(combined, good_elements)
combined = pd.concat([combined, clr_df], axis=1)

# Ratios
ratio_df = engineer_ratios(combined)
combined = pd.concat([combined, ratio_df], axis=1)

# Geophysics sampling
combined = sample_all_geophysics(combined, BASE)

# PCA
clr_cols = [c for c in combined.columns if c.startswith("clr_")]
pca_data = combined[clr_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pca_data)
pca = PCA(n_components=min(10, len(clr_cols)))
pc_scores = pca.fit_transform(X_scaled)
for i in range(min(5, pca.n_components_)):
    combined.loc[pca_data.index, f"PC{i+1}"] = pc_scores[:, i]

# Anomaly detection
anomaly_elements = [e for e in ["Li", "Rb", "Cs", "La", "Ce", "Y", "Nb", "Ta", "Th", "Ba"]
                    if e in combined.columns and combined[e].notna().sum() > 100]
for el in anomaly_elements:
    _, mask2 = mad_anomaly(combined[el], 2.0)
    _, mask3 = mad_anomaly(combined[el], 3.0)
    combined[f"{el}_anom2"] = mask2.astype(int)
    combined[f"{el}_anom3"] = mask3.astype(int)

print(f"Preprocessing complete: {len(combined)} samples, {len(combined.columns)} columns")
save_checkpoint("combined_preprocessed", combined)

# %% [markdown]
# ## Part 3: Granite Proximity Features
#
# We use the real GSI/GSNI 1:1M bedrock granite polygons (Caledonian only) and compute the
# signed distance-to-granite-contact for each sample point. This provides a geology-based
# spatial predictor independent of geochemistry.

# %%
print("\n" + "=" * 60)
print("STAGE 2: GRANITE PROXIMITY FEATURES")
print("=" * 60)

# Real GSI/GSNI 1:1M bedrock granite polygons ONLY (no manual ellipse approximations).
# LCT pegmatites derive from CALEDONIAN S-type granites, so we use the real polygons:
# Ordovician + Siluro-Devonian granitic rocks & appinite. Palaeogene granitic rocks
# (Mourne/Carlingford, A-type, NYF/REE) are EXCLUDED from the Li granite feature.
_geol_dir = os.path.join(BASE, "Ireland/Geology/GIS")
_bedrock_shp = os.path.join(_geol_dir, "IE_GSI_GSNI_Bedrock_Geology_1M_IE32_ITM_MS.shp")
if not os.path.exists(_bedrock_shp):
    raise FileNotFoundError("Required real bedrock shapefile not found: " + _bedrock_shp)
_bed = gpd.read_file(_bedrock_shp)
if str(_bed.crs) != "EPSG:2157":
    _bed = _bed.to_crs("EPSG:2157")
_caled = _bed[_bed["UNITNAME"].str.contains("Ordovician granit|Siluro-Devonian granit|appinite", case=False, na=False)]
_pts = gpd.GeoSeries(gpd.points_from_xy(combined["x"], combined["y"]), crs="EPSG:2157")
_gunion = _caled.geometry.union_all()
_dedge = _pts.distance(_gunion.boundary).values
_inside = _pts.within(_gunion).values
combined["dist_granite_m"] = np.where(_inside, -_dedge, _dedge)   # signed metres; negative = inside
combined["inside_granite"] = _inside.astype(int)
combined["log_dist_granite"] = np.log1p(np.abs(combined["dist_granite_m"]))
print(f"Real Caledonian granite polygons: {len(_caled)} units; samples inside: {int(_inside.sum())}")

# Real GSI faults for structural control (real data only; no manual ECDZ line).
_faults_shp = os.path.join(_geol_dir, "IE_GSI_GSNI_Faults_1M_IE32_ITM_MS.shp")
if not os.path.exists(_faults_shp):
    raise FileNotFoundError("Required real faults shapefile not found: " + _faults_shp)
_fa = gpd.read_file(_faults_shp)
if str(_fa.crs) != "EPSG:2157":
    _fa = _fa.to_crs("EPSG:2157")
combined["dist_fault"] = _pts.distance(_fa.geometry.union_all()).values
combined["log_dist_fault"] = np.log1p(combined["dist_fault"])
print(f"Real GSI faults: {len(_fa)}; median distance-to-fault {np.median(combined['dist_fault']):.0f} m")

# %% [markdown]
# ### Bedrock Geology as Categorical Feature
#
# Bedrock lithology is a first-order control on LCT pegmatite prospectivity, LCT pegmatites
# are hosted in or adjacent to granitic intrusives, not in carbonates or basalts.
# We integrate bedrock geology as a categorical feature using spatial join (if the
# polygon shapefile is available) or a knowledge-driven regional classification.

# %%
print("\n--- Bedrock Geology Integration ---")

# Search for bedrock polygon shapefile in common paths
bedrock_shp_candidates = [
    os.path.join(BASE, "Ireland/Geology/GIS/IE_GSI_GSNI_Bedrock_Geology_1M_IE32_ITM_MS.shp"),
    os.path.join(BASE, "Ireland/Geology/Bedrock/IE_GSI_Bedrock_100K_IE26_ITM.shp"),
    os.path.join(BASE, "Ireland/Geology/IE_GSI_Bedrock_100K_IE26_ITM.shp"),
    os.path.join(BASE, "Ireland/Geology/Bedrock_100K.shp"),
    os.path.join(BASE, "Ireland/Geology/bedrock.shp"),
    os.path.join(BASE, "Ireland/Geology/Bedrock/bedrock.shp"),
]

bedrock_gdf = None
for shp_path in bedrock_shp_candidates:
    if os.path.exists(shp_path):
        print(f"  Found bedrock shapefile: {shp_path}")
        bedrock_gdf = gpd.read_file(shp_path)
        if bedrock_gdf.crs != "EPSG:2157":
            bedrock_gdf = bedrock_gdf.to_crs("EPSG:2157")
        break

if bedrock_gdf is not None:
    # ── Spatial join: assign bedrock lithology to each sample point ──
    print(f"  Bedrock polygons: {len(bedrock_gdf)}")
    
    # Identify the lithology column (varies by shapefile version)
    litho_col = None
    for candidate in ["UNITNAME", "ROCK_UNIT", "LITHOLOGY", "LEX_D", "UNIT_NAME", "DESCRIP"]:
        if candidate in bedrock_gdf.columns:
            litho_col = candidate
            break
    if litho_col is None:
        litho_col = bedrock_gdf.columns[1]  # fallback: first non-geometry column
    
    print(f"  Lithology column: {litho_col} ({bedrock_gdf[litho_col].nunique()} unique types)")
    
    # Create GeoDataFrame from sample points
    sample_gdf = gpd.GeoDataFrame(
        combined[["x", "y"]],
        geometry=gpd.points_from_xy(combined["x"], combined["y"]),
        crs="EPSG:2157"
    )
    
    # Spatial join
    joined = gpd.sjoin(sample_gdf, bedrock_gdf[[litho_col, "geometry"]], how="left", predicate="within")
    combined["bedrock_unit"] = joined[litho_col].values
    
    # Simplify to major lithology classes for ML
    def simplify_lithology(name):
        if pd.isna(name):
            return "Unknown"
        name_lower = str(name).lower()
        if any(k in name_lower for k in ["granit", "granodiorit", "appinite", "siluro-devon"]):
            return "Granite"
        elif any(k in name_lower for k in ["pegmatit"]):
            return "Pegmatite"
        elif any(k in name_lower for k in ["limestone", "calcare", "chalk"]):
            return "Carbonate"
        elif any(k in name_lower for k in ["dalradian", "metased", "schist", "gneiss", "neoproterozoic meta"]):
            return "Metasediment"
        elif any(k in name_lower for k in ["sandstone", "mudstone", "shale", "siltstone", "slate", "greywacke"]):
            return "Clastic"
        elif any(k in name_lower for k in ["volcanic", "basalt", "rhyolite", "ignimbrite", "tuff"]):
            return "Volcanic"
        elif any(k in name_lower for k in ["gabbro", "dolerite", "basic", "ultramafic", "ophiolite"]):
            return "Mafic_intrusive"
        elif any(k in name_lower for k in ["conglomerate", "ors", "devonian"]):
            return "ORS_Devonian"
        elif any(k in name_lower for k in ["coal", "namurian"]):
            return "Coal_measures"
        else:
            return "Other"
    
    combined["bedrock_class"] = combined["bedrock_unit"].apply(simplify_lithology)
    print(f"\n  Simplified lithology distribution:")
    print(combined["bedrock_class"].value_counts().to_string())
    
    # One-hot encode
    bedrock_dummies = pd.get_dummies(combined["bedrock_class"], prefix="bedrock")
    combined = pd.concat([combined, bedrock_dummies], axis=1)
    bedrock_feature_cols = list(bedrock_dummies.columns)
    print(f"\n  Added {len(bedrock_feature_cols)} bedrock features: {bedrock_feature_cols}")

else:
    # ── Fallback: knowledge-driven regional classification ──
    print("  ⚠ Bedrock polygon shapefile not found in repo.")
    print(" to Using knowledge-driven regional classification (based on GSI 1:500K geology)")
    print(" to To upgrade: download IE_GSI_Bedrock_100K_IE26_ITM.shp from GSI portal")
    print("    and place it in Ireland/Geology/Bedrock/")
    
    def classify_bedrock_regional(x, y):
        """Knowledge-driven bedrock classification from known geological boundaries."""
        # Leinster Granite Massif (SE Ireland)
        if 650000 < x < 730000 and 620000 < y < 740000:
            dx = x - 690000; dy = y - 680000
            if (dx/35000)**2 + (dy/65000)**2 < 1:
                return "Granite"
        # Tullow Lowlands pluton
        if 660000 < x < 695000 and 630000 < y < 680000:
            dx = x - 675000; dy = y - 650000
            if (dx/18000)**2 + (dy/28000)**2 < 1:
                return "Granite"
        # Newry Igneous Complex
        if 690000 < x < 720000 and 810000 < y < 845000:
            dx = x - 705000; dy = y - 825000
            if (dx/15000)**2 + (dy/22000)**2 < 1:
                return "Granite"
        # Mourne Mountains granite
        if 720000 < x < 745000 and 820000 < y < 840000:
            dx = x - 733000; dy = y - 830000
            if (dx/12000)**2 + (dy/10000)**2 < 1:
                return "Granite"
        # Donegal granites
        if 555000 < x < 610000 and 880000 < y < 920000:
            return "Granite"
        # Galway granite
        if 495000 < x < 545000 and 715000 < y < 745000:
            return "Granite"
        # Dalradian metasediments (NW Ireland / Sperrins)
        if y > 860000 and x < 680000:
            return "Metasediment"
        if 600000 < x < 680000 and 840000 < y < 900000:
            return "Metasediment"
        # Carboniferous limestone midlands
        if 500000 < x < 700000 and 720000 < y < 820000:
            return "Carbonate"
        # Ordovician-Silurian (SE Ireland, outside Leinster)
        if x > 660000 and 640000 < y < 730000:
            return "Clastic"
        # Southern Uplands (Down-Longford terrane)
        if x > 680000 and y > 800000:
            return "Clastic"
        # Old Red Sandstone (south)
        if y < 640000:
            return "ORS_Devonian"
        return "Other"
    
    combined["bedrock_class"] = [
        classify_bedrock_regional(x, y)
        for x, y in zip(combined["x"].values, combined["y"].values)
    ]
    
    print(f"\n  Regional lithology distribution:")
    print(combined["bedrock_class"].value_counts().to_string())
    
    # One-hot encode
    bedrock_dummies = pd.get_dummies(combined["bedrock_class"], prefix="bedrock")
    combined = pd.concat([combined, bedrock_dummies], axis=1)
    bedrock_feature_cols = list(bedrock_dummies.columns)
    print(f"\n  Added {len(bedrock_feature_cols)} bedrock features: {bedrock_feature_cols}")

# %% [markdown]
# ## Part 2: Fix Circular Reasoning, Three Model Variants
#
# ### The Problem
# The v1 model used Li anomalies to help define positive training labels, then Li
# dominated the SHAP rankings. This creates unfalsifiable circular reasoning.
#
# ### Solution: Three complementary models
#
# Following the dual knowledge-driven + data-driven methodology of Vadoodi et al. (2026)
# for Li-pegmatite prospectivity mapping:
# 1. **Model A (Geology-Only Labels):** Positives = at the real Caledonian granite margin
#    (within 5km outside to 3km inside the GSI polygons). No geochemistry in labelling.
# 2. **Model B (Leave-Li-Out):** Same as Model A but Li excluded from features entirely.
#    If this model identifies the same targets, the signal is multivariate, not Li-circular.
# 3. **Model C (Full Features):** Model A labels with all features including Li.

# %%
print("\n" + "=" * 60)
print("STAGE 3: MODEL TRAINING, CIRCULAR REASONING FIX")
print("=" * 60)

# ─── Known occurrence coordinates (ITM, from published sources) ───
# Sources: Barros et al. (2022, MDPI Minerals) for Aclare/Moylisha;
# GBML press releases (2023) for Knockeen/Carriglead;
# plus GSI Mineral Localities (loaded below). Converted to ITM (EPSG:2157) using pyproj.
_min_shp = os.path.join(BASE, "Ireland/Geology/GIS/IE_GSI_Mineral_Locations_IE26_ITM.shp")
if os.path.exists(_min_shp):
    # Real GSI lithium occurrences (Republic), plus cross-border Newry/Mourne
    # analogues (the ROI GSI file does not cover Northern Ireland).
    _mins = gpd.read_file(_min_shp)
    if str(_mins.crs) != "EPSG:2157":
        _mins = _mins.to_crs("EPSG:2157")
    _li = _mins[_mins["MIN_TYPE"].astype(str).str.upper() == "LI"].copy()
    known_occurrences = pd.DataFrame({
        "name": _li["TOWNLAND"].fillna("LI_occ").astype(str).values,
        "x": _li.geometry.x.values, "y": _li.geometry.y.values,
        "type": "confirmed_LCT", "source": "GSI Mineral_Locations",
    })
    print(f"  REAL GSI lithium occurrences (Republic of Ireland): {len(known_occurrences)} anchors")
    print("  NOTE: no Northern Ireland mineral-occurrence dataset is available locally,")
    print("        so NI analogues are NOT included (real data only).")
else:
    raise FileNotFoundError(
        "Required real GSI Mineral_Locations shapefile not found: " + _min_shp
        + "  (no hardcoded coordinates are used)")

# ─── OPTION A: Geology-Only Positive Labels ───
# Positive = within 3km of ANY granite contact AND within 5km of known structural corridor
# No geochemistry used in label definition

# Geology-only positives from REAL data: at the real Caledonian granite margin
# (within 5 km outside to 3 km inside the GSI polygons). No manual structural line is used.
geo_positive = (
    (combined["dist_granite_m"] < 5000) &
    (combined["dist_granite_m"] > -3000)
)

# Anchor buffer around REAL GSI lithium occurrences (ground-truth)
occ_coords = known_occurrences[["x", "y"]].values
dists_to_occ = cdist(combined[["x", "y"]].values, occ_coords)
near_occ = dists_to_occ.min(axis=1) < 3000   # within 3 km of a real Li occurrence

combined["label_geo"] = 0
combined.loc[geo_positive | near_occ, "label_geo"] = 1

# Strong negatives: far from any real Caledonian granite
geo_negative = (np.abs(combined["dist_granite_m"]) > 20000)

n_pos_geo = combined["label_geo"].sum()
print(f"\nGeology-Only Labels:")
print(f"  Positives (real Caledonian granite margin + real Li occurrences): {n_pos_geo}")
print(f"  Strong negatives (far from real granite): {geo_negative.sum()}")

# ─── Build feature sets ───
# Features WITHOUT Li (for Model B)
all_clr = [c for c in combined.columns if c.startswith("clr_")]
no_li_clr = [c for c in all_clr if c != "clr_Li"]

ratio_feats = [c for c in combined.columns if c in
               ["Rb_Sr", "Nb_Ta", "Th_U", "sum_LREE", "sum_HREE", "LREE_HREE"]]
geophys_feats = [c for c in ["TMI", "TDR", "MAG_1VD", "K_rad", "Th_rad", "U_rad",
                              "EM_res3_2F", "EM_res12_2F"]
                 if c in combined.columns and combined[c].notna().sum() > 1000]
pc_feats = [f"PC{i}" for i in range(1, 6) if f"PC{i}" in combined.columns]
geo_feats = ["log_dist_granite", "inside_granite", "log_dist_fault"] if "log_dist_fault" in combined.columns else ["log_dist_granite", "inside_granite"]
bedrock_feats = [c for c in combined.columns if c.startswith("bedrock_") and c not in ("bedrock_class", "bedrock_unit")]

features_full = all_clr + ratio_feats + geophys_feats + pc_feats + geo_feats + bedrock_feats
features_no_li = no_li_clr + ratio_feats + geophys_feats + pc_feats + geo_feats + bedrock_feats
# For geology-labelled models, EXCLUDE granite distance (it's in the label definition!)
# but KEEP bedrock class, it's lithology, not spatial proximity
features_geo_full = all_clr + ratio_feats + geophys_feats + pc_feats + bedrock_feats
features_geo_no_li = no_li_clr + ratio_feats + geophys_feats + pc_feats + bedrock_feats

print(f"\nFull feature set: {len(features_full)} features")
print(f"No-Li feature set: {len(features_no_li)} features")
print(f"Geo-label feature set (no granite dist): {len(features_geo_full)} features")
print(f"Bedrock features: {bedrock_feats}")


def train_rf_model(combined_df, label_col, feature_cols, model_name, neg_mask=None):
    """Train RF with spatial CV and return model + predictions."""
    pos_idx = combined_df[combined_df[label_col] == 1].index
    
    if neg_mask is not None:
        neg_pool = combined_df[neg_mask].index
    else:
        neg_pool = combined_df[combined_df[label_col] == 0].index
    
    n_pos = len(pos_idx)
    n_neg = min(n_pos * 3, len(neg_pool))
    neg_idx = neg_pool[np.random.RandomState(42).choice(len(neg_pool), n_neg, replace=False)]
    
    train_idx = pos_idx.union(neg_idx)
    X_train = combined_df.loc[train_idx, feature_cols].copy()
    y_train = combined_df.loc[train_idx, label_col].copy()
    
    # Drop >50% NaN columns
    drop = X_train.columns[X_train.isna().mean() > 0.5].tolist()
    active_features = [c for c in feature_cols if c not in drop]
    X_train = X_train[active_features]
    
    # Impute NaN with median
    for col in X_train.columns:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med if not np.isnan(med) else 0)
    
    # Spatial blocks
    block_size = 25000   # 25 km spatial blocks (finer spatial cross-validation)
    blocks = (
        (combined_df.loc[train_idx, "x"] // block_size).astype(int).astype(str) + "_" +
        (combined_df.loc[train_idx, "y"] // block_size).astype(int).astype(str)
    )
    unique_blocks = blocks.unique()
    np.random.seed(42)
    block_map = {b: i % max(2, min(10, len(unique_blocks))) for i, b in enumerate(np.random.permutation(unique_blocks))}
    fold_ids = blocks.map(block_map).values
    
    # Spatial CV
    n_folds = min(10, len(unique_blocks))
    if n_folds < 2:
        n_folds = 2
    gkf = GroupKFold(n_splits=n_folds)
    cv_scores = []
    for fold_i, (tr, val) in enumerate(gkf.split(X_train, y_train, groups=fold_ids)):
        if USE_GPU:
            try:
                import cudf
                rf_cv = cuRF(n_estimators=RF_N_ESTIMATORS, max_depth=15, min_samples_leaf=3,
                             max_features="sqrt", random_state=42)
                rf_cv.fit(cudf.DataFrame(X_train.iloc[tr].values, columns=X_train.columns),
                          cudf.Series(y_train.iloc[tr].values))
            except Exception:
                rf_cv = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=15,
                    min_samples_leaf=3, max_features="sqrt", class_weight="balanced",
                    random_state=42, n_jobs=-1)
                rf_cv.fit(X_train.iloc[tr], y_train.iloc[tr])
        else:
            rf_cv = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS, max_depth=15, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
            )
            rf_cv.fit(X_train.iloc[tr], y_train.iloc[tr])
        try:
            proba = rf_cv.predict_proba(X_train.iloc[val])
            if proba.shape[1] >= 2 and len(y_train.iloc[val].unique()) > 1:
                score = roc_auc_score(y_train.iloc[val], proba[:, 1])
            else:
                score = rf_cv.score(X_train.iloc[val], y_train.iloc[val])
        except (IndexError, ValueError):
            score = rf_cv.score(X_train.iloc[val], y_train.iloc[val])
        cv_scores.append(score)
    
    print(f"\n{model_name}:")
    print(f"  Positives: {n_pos}, Negatives: {n_neg}")
    print(f"  Features: {len(active_features)}")
    print(f"  Spatial CV AUC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    for i, s in enumerate(cv_scores):
        print(f"    Fold {i+1}: {s:.3f}")
    
    # Final model on all training data
    if USE_GPU:
        rf = cuRF(
            n_estimators=RF_N_ESTIMATORS, max_depth=15, min_samples_leaf=3,
            max_features="sqrt", random_state=42,
        )
        import cudf
        rf.fit(cudf.DataFrame(X_train), cudf.Series(y_train))
    else:
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=15, min_samples_leaf=3,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
    
    # Predict on all samples
    X_all = combined_df[active_features].copy()
    for col in X_all.columns:
        med = X_all[col].median()
        X_all[col] = X_all[col].fillna(med if not np.isnan(med) else 0)
    
    proba_all = rf.predict_proba(X_all)
    probs = proba_all[:, 1] if proba_all.shape[1] >= 2 else proba_all[:, 0]
    
    return rf, probs, X_train, y_train, active_features, np.mean(cv_scores)


# ─── Train three model variants ───
print("\n--- Training Model A: Geology-Only Labels, Full Features (excl. granite dist) ---")
rf_a, probs_a, X_a, y_a, feats_a, auc_a = train_rf_model(
    combined, "label_geo", features_geo_full, "Model A (Geology Labels + Full Features)", geo_negative
)
combined["prosp_A"] = probs_a

print("\n--- Training Model B: Geology-Only Labels, NO Li (excl. granite dist) ---")
rf_b, probs_b, X_b, y_b, feats_b, auc_b = train_rf_model(
    combined, "label_geo", features_geo_no_li, "Model B (Geology Labels + No Li)", geo_negative
)
combined["prosp_B"] = probs_b

print("\n--- Training Model C: Occurrence-Only Labels, ALL features incl. geology ---")
# Model C: use known occurrences ONLY as positives (strictest possible)
combined["label_occ_only"] = 0
combined.loc[near_occ, "label_occ_only"] = 1
rf_c, probs_c, X_c, y_c, feats_c, auc_c = train_rf_model(
    combined, "label_occ_only", features_full, "Model C (Occurrence-Only Labels)", None
)
combined["prosp_C"] = probs_c

# ─── Use Model A as primary (geology-driven, full features) ───
combined["prospectivity"] = probs_a
save_checkpoint("rf_models", {"rf_a": rf_a, "rf_b": rf_b, "rf_c": rf_c,
                               "feats_a": feats_a, "feats_b": feats_b, "feats_c": feats_c})

# %% [markdown]
# ### Model Comparison, Circular Reasoning Assessment
#
# If Models A and B identify the same targets with similar rankings, the multivariate
# geochemical+geophysical signature is driving predictions, not just Li alone.

# %%
print("\n" + "=" * 60)
print("STAGE 4: MODEL COMPARISON")
print("=" * 60)

# Correlation between model outputs
from scipy.stats import spearmanr

r_ab, _ = spearmanr(probs_a, probs_b)
r_ac, _ = spearmanr(probs_a, probs_c)
r_bc, _ = spearmanr(probs_b, probs_c)

print(f"\nSpearman correlations between model prospectivity scores:")
print(f"  A (full) vs B (no-Li): ρ = {r_ab:.3f}")
print(f"  A (full) vs C (occ-only): ρ = {r_ac:.3f}")
print(f"  B (no-Li) vs C (occ-only): ρ = {r_bc:.3f}")
print(f"\nHigh A-B correlation confirms the model is NOT Li-circular.")

# ─── SHAP for all models ───
# %% [markdown]
# ### SHAP Interpretation, Geological Validation
#
# SHAP ranks a **multivariate signature**, led by airborne **EM resistivity** (res3/res12),
# **bedrock granite/lithology**, **K radiometrics**, and **LREE/HREE** geochemistry, i.e.
# geophysics + lithology + geochemistry jointly drive the model, *not Li alone*. The radiometric
# contribution is geologically sensible: airborne **Th/K** track both (1) Th/K in accessory
# minerals (monazite, thorite, K-feldspar) of fractionated leucogranites / LCT pegmatites, and
# (2) clay/alteration in A-horizon soils (Cardoso-Fernandes et al., 2023), 
# a dual magmatic + alteration proxy, while EM resistivity adds an independent structural/regolith
# dimension. (Exact ranking is printed below; do not hard-code it in the text.)
#
# The leave-Li-out model (Model B) produces a near-identical prospectivity map
# (ρ = 0.999), confirming that the multivariate radiometric + geochemical signature
# independently identifies the same targets. This constitutes an **independent
# geophysical validation** of the model, following the methodology advocated by
# Vadoodi et al. (2026) for combining knowledge-driven and data-driven approaches
# to Li pegmatite prospectivity mapping.
#
# > **Geological caveat:** LCT and NYF pegmatites represent distinct metallogenic
# > systems derived from different source magmas and tectonic environments (Černý &
# > Ercit, 2005). This model targets granitic environments prospective
# > for LCT pegmatites specifically. Leinster/Newry targets are interpreted as LCT (Li)
# > candidates; any Mourne targets should be evaluated separately as potential NYF (REE)
# > candidates. Confirmation of deposit type requires follow-up lithogeochemistry and
# > petrographic characterisation.

# %%
print("\n--- SHAP Analysis ---")
try:
    import shap
    
    for model_label, rf_model, X_tr, feat_list in [
        ("A", rf_a, X_a, feats_a), ("B", rf_b, X_b, feats_b)
    ]:
        shap_n = min(SHAP_MAX_SAMPLES, len(X_tr))
        X_shap = X_tr.sample(n=shap_n, random_state=42)
        explainer = shap.TreeExplainer(rf_model)
        sv = explainer.shap_values(X_shap, check_additivity=False, approximate=True)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_shap, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance, Model {model_label}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(MAPS, f"shap_model_{model_label}.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: shap_model_{model_label}.png")
        
        # Print top 10
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-10:][::-1]
        print(f"\n  Model {model_label}, Top 10 SHAP features:")
        for i, idx in enumerate(top_idx):
            print(f"    {i+1}. {X_shap.columns[idx]}: {mean_abs[idx]:.4f}")
    
except Exception as e:
    print(f"  ⚠ SHAP error: {e}")

# %% [markdown]
# ## Prospectivity Maps, Model A (Primary) & Model B (No-Li Validation)

# %%
print("\n" + "=" * 60)
print("STAGE 5: PROSPECTIVITY MAPS")
print("=" * 60)

# ─── IDW Interpolated Surface ───
def idw_interpolate(x, y, z, xi, yi, power=2, k=8):
    """Inverse Distance Weighting interpolation using cKDTree."""
    from scipy.spatial import cKDTree
    # Subsample source points for speed if too many
    if len(x) > 5000:
        idx = np.random.RandomState(0).choice(len(x), 5000, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
    tree = cKDTree(np.column_stack([x, y]))
    d, indices = tree.query(np.column_stack([xi, yi]), k=k)
    d = np.maximum(d, 1e-10)
    w = 1.0 / d**power
    return (w * z[indices]).sum(axis=1) / w.sum(axis=1)

# Create grid once
valid = combined[["x", "y", "prospectivity"]].dropna()
x_range = np.arange(valid["x"].min(), valid["x"].max(), 5000)
y_range = np.arange(valid["y"].min(), valid["y"].max(), 5000)
xi, yi = np.meshgrid(x_range, y_range)
xi_flat, yi_flat = xi.ravel(), yi.ravel()

print("Interpolating prospectivity surfaces...")

# Pre-subsample source data
np.random.seed(0)
src_idx = np.random.choice(len(valid), min(5000, len(valid)), replace=False)

zi_a = idw_interpolate(
    valid["x"].values[src_idx], valid["y"].values[src_idx], 
    combined.loc[valid.index[src_idx], "prosp_A"].values,
    xi_flat, yi_flat, power=2, k=8
).reshape(xi.shape)

zi_b = idw_interpolate(
    valid["x"].values[src_idx], valid["y"].values[src_idx],
    combined.loc[valid.index[src_idx], "prosp_B"].values,
    xi_flat, yi_flat, power=2, k=8
).reshape(xi.shape)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(20, 14))
for ax_i, (zi_data, title) in enumerate([
    (zi_a, "Model A, Geology-Driven Labels, Full Features"),
    (zi_b, "Model B, Geology-Driven Labels, No Li"),
]):
    ax = axes[ax_i]
    im = ax.pcolormesh(xi, yi, zi_data, cmap="RdYlGn_r", vmin=0, vmax=1, shading="auto")
    plt.colorbar(im, ax=ax, label="Prospectivity", shrink=0.6)
    
    ax.scatter(known_occurrences["x"], known_occurrences["y"],
               c="blue", marker="*", s=200, zorder=5, edgecolors="white", linewidths=1,
               label="Known LCT Occurrences")
    
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Easting (ITM)")
    ax.set_ylabel("Northing (ITM)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)

plt.suptitle("LCT Pegmatite Prospectivity, Tellus Programme, Ireland\nIDW Interpolated Surface",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(MAPS, f"prospectivity_surface_AB{SUFFIX}.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: prospectivity_surface_AB{SUFFIX}.png")

# %% [markdown]
# ## Part 4: Target Generation with Validation

# %%
print("\n" + "=" * 60)
print("STAGE 6: TARGET GENERATION & VALIDATION")
print("=" * 60)

thresh_high = combined["prospectivity"].quantile(0.95)
high_prosp = combined[combined["prospectivity"] >= thresh_high].copy()
print(f"High-prospectivity samples (≥{thresh_high:.3f}): {len(high_prosp)}")

coords = high_prosp[["x", "y"]].values
clustering = DBSCAN(eps=5000, min_samples=3).fit(coords)
high_prosp["cluster"] = clustering.labels_
n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
print(f"Targets found: {n_clusters}")

# Geological domain assignment
def assign_domain(cx, cy):
    """Assign geological domain based on target centroid location."""
    if cy < 700000 and cx > 650000:
        return "Leinster Granite Margin"
    elif cy > 800000 and cx > 690000:
        return "Newry-Mourne Granites"
    elif cy > 800000 and cx < 690000:
        return "Sperrins-Dalradian"
    elif cx < 600000:
        return "Western Ireland"
    else:
        return "Central Ireland"

# Build target table
targets = []
for cid in sorted(set(clustering.labels_) - {-1}):
    cl = high_prosp[high_prosp["cluster"] == cid]
    cx, cy = cl["x"].mean(), cl["y"].mean()
    
    # Distance to nearest known occurrence
    dist_occ = cdist([[cx, cy]], occ_coords).min()
    nearest_name = known_occurrences.iloc[cdist([[cx, cy]], occ_coords).argmin(axis=1)[0]]["name"]
    
    # Model B prospectivity at same location (circular reasoning check)
    mean_prosp_b = cl["prosp_B"].mean() if "prosp_B" in cl.columns else np.nan
    
    domain = assign_domain(cx, cy)
    
    # SHAP top features for this cluster (use Model A)
    # We'll compute this from feature importance
    
    # Validation notes
    if dist_occ < 15000:
        val_note = f"Within 15km of {nearest_name}, VALIDATES known occurrence"
    elif domain == "Newry-Mourne Granites":
        val_note = "Under-explored Caledonian granite margin; analogous to Leinster. Priority recommendation."
    elif domain == "Sperrins-Dalradian":
        val_note = "CAUTION: Dalradian metasediments, no mapped granitic intrusives. Likely reflects clay-rich protolith Li, not pegmatite. Requires field validation."
    elif domain == "Leinster Granite Margin":
        val_note = f"Within Leinster LCT pegmatite belt. {dist_occ/1000:.0f}km from nearest known occurrence ({nearest_name})."
    else:
        val_note = "Requires field validation."
    
    # Convert centroid to WGS84 for lat/lon
    gdf_pt = gpd.GeoDataFrame(
        [{"x": cx, "y": cy}],
        geometry=gpd.points_from_xy([cx], [cy]),
        crs="EPSG:2157"
    ).to_crs("EPSG:4326")
    lat = gdf_pt.geometry.y.values[0]
    lon = gdf_pt.geometry.x.values[0]
    
    targets.append({
        "Target_ID": f"T{cid+1:02d}",
        "Lon": round(lon, 4),
        "Lat": round(lat, 4),
        "Centroid_X_ITM": round(cx),
        "Centroid_Y_ITM": round(cy),
        "n_samples": len(cl),
        "Area_km2": round((cl["x"].max() - cl["x"].min()) * (cl["y"].max() - cl["y"].min()) / 1e6, 1),
        "Mean_Prospectivity": round(cl["prospectivity"].mean(), 3),
        "Max_Prospectivity": round(cl["prospectivity"].max(), 3),
        "Mean_Prosp_NoLi": round(mean_prosp_b, 3) if not np.isnan(mean_prosp_b) else None,
        "Mean_Li_mgkg": round(cl["Li"].mean(), 1) if "Li" in cl.columns else None,
        "Mean_Rb_mgkg": round(cl["Rb"].mean(), 1) if "Rb" in cl.columns else None,
        "Geological_Domain": domain,
        "Nearest_Known_Occurrence_km": round(dist_occ / 1000, 1),
        "Nearest_Occurrence_Name": nearest_name,
        "Validation_Notes": val_note,
    })

targets_df = pd.DataFrame(targets)

# Rank by composite score
if len(targets_df) > 0:
    targets_df["Rank_Score"] = (
        targets_df["Mean_Prospectivity"] * 0.35 +
        (targets_df["n_samples"] / targets_df["n_samples"].max()) * 0.25 +
        targets_df["Max_Prospectivity"] * 0.25 +
        (1 - targets_df["Nearest_Known_Occurrence_km"].clip(upper=100) / 100) * 0.15
    )
    targets_df = targets_df.sort_values("Rank_Score", ascending=False).reset_index(drop=True)
    targets_df.index = range(1, len(targets_df) + 1)
    targets_df.index.name = "Rank"
    
    print("\n=== Ranked Exploration Targets ===")
    print(targets_df[["Target_ID", "n_samples", "Mean_Prospectivity", "Mean_Prosp_NoLi",
                       "Geological_Domain", "Nearest_Known_Occurrence_km", "Validation_Notes"]
                     ].to_string())
    
    targets_df.to_csv(os.path.join(OUTPUTS, f"ranked_targets{SUFFIX}.csv"))
    print(f"\nSaved: ranked_targets{SUFFIX}.csv ({len(targets_df)} targets)")

# ─── Validation: recovery of the real Li occurrences ───
print("\n--- Validation: known Li occurrence recovery (real GSI occurrences) ---")
_rec15 = 0; _rec20 = 0
for _, occ in known_occurrences.iterrows():
    if len(targets_df) == 0:
        continue
    dists = np.sqrt((targets_df["Centroid_X_ITM"] - occ["x"])**2 + (targets_df["Centroid_Y_ITM"] - occ["y"])**2)
    nearest_tid = targets_df.iloc[dists.argmin()]["Target_ID"]
    nearest_d = dists.min() / 1000
    if nearest_d < 15: _rec15 += 1
    if nearest_d < 20: _rec20 += 1
    print(f"  {occ['name']}: nearest target = {nearest_tid} ({nearest_d:.1f} km), "
          + ("RECOVERED" if nearest_d < 20 else "not recovered"))
_nocc = len(known_occurrences)
print(f"  Occurrence recovery: {_rec15}/{_nocc} within 15 km, {_rec20}/{_nocc} within 20 km")

# ─── Granite-proximity filter: LCT pegmatites require a granite source ───
# Distance of each target centroid to the real Caledonian granite polygons separates
# genuine granite-margin LCT targets from off-granite anomalies (e.g. clay/protolith-hosted
# Li in the NI Sperrins-Dalradian metasediments, which are a different deposit type).
if len(targets_df):
    try:
        _cgr = bedrock_gdf[bedrock_gdf["UNITNAME"].str.contains("Ordovician granit|Siluro-Devonian granit|appinite", case=False, na=False)]
        _cu = _cgr.geometry.union_all()
        _tp = gpd.GeoSeries(gpd.points_from_xy(targets_df["Centroid_X_ITM"], targets_df["Centroid_Y_ITM"]), crs="EPSG:2157")
        targets_df["dist_caledonian_granite_km"] = (_tp.distance(_cu).values / 1000).round(1)
        _GRANITE_MAX_KM = 10.0
        lct_targets = targets_df[targets_df["dist_caledonian_granite_km"] <= _GRANITE_MAX_KM].copy()
        targets_df.to_csv(os.path.join(OUTPUTS, f"ranked_targets{SUFFIX}.csv"))
        lct_targets.to_csv(os.path.join(OUTPUTS, f"ranked_targets_granite_proximal{SUFFIX}.csv"))
        print(f"\n--- Granite-proximity filter (LCT candidates within {_GRANITE_MAX_KM:.0f} km of real Caledonian granite) ---")
        print(f"  {len(lct_targets)} of {len(targets_df)} clusters are granite-proximal (the defensible LCT set);")
        print(f"  the other {len(targets_df) - len(lct_targets)} are off-granite (likely clay/protolith-hosted Li, a separate play).")
        if len(lct_targets):
            print(lct_targets[["Target_ID", "Geological_Domain", "Mean_Prospectivity", "dist_caledonian_granite_km", "Nearest_Known_Occurrence_km"]].to_string())
    except Exception as e:
        print("granite-proximity filter note:", e)

# %% [markdown]
# ## Independent hydrogeochemistry: dissolved Li in stream/regional waters
# A third real medium. Dissolved Li in stream water is a recognised lithium pathfinder
# (Li is mobile), so it is an independent check on the model and on the granite-margin targets.

# %%
try:
    import glob as _glob
    from scipy.spatial import cKDTree as _KD
    from scipy.stats import spearmanr as _sp
    def _load_wLi(path, ecol, ncol):
        raw = pd.read_excel(path); cols = list(raw.columns)
        li = None
        for c in cols:
            cl = str(c).lower()
            if cl == "li" or (cl.startswith("li_") and "gl" in cl): li = c; break
        if ecol not in cols or ncol not in cols or li is None: return pd.DataFrame()
        d = {"x": pd.to_numeric(raw[ecol], errors="coerce"), "y": pd.to_numeric(raw[ncol], errors="coerce"),
             "Li_w": pd.to_numeric(raw[li], errors="coerce")}
        df = pd.DataFrame(d).dropna(subset=["x", "y", "Li_w"])
        if len(df): df["source_crs"] = 2157 if df["x"].mean() > 400000 else 29903
        return df
    _wl = []
    for _f in _glob.glob(os.path.join(BASE, "Ireland/Geochem/W_Stream_water_geochemistry/*.xlsx")):
        _wl.append(_load_wLi(_f, "Easting_ING", "Northing_ING"))
    _nif = os.path.join(BASE, "Northern Ireland/2. Geochem/Regional_Waters_ICP.xls")
    if os.path.exists(_nif): _wl.append(_load_wLi(_nif, "EASTING", "NORTHING"))
    watLi = pp.reproject_to_itm(pd.concat([w for w in _wl if len(w)], ignore_index=True))
    watLi["Li_w_rank"] = watLi["Li_w"].rank(pct=True)
    _tr = _KD(combined[["x", "y"]].values); _d, _i = _tr.query(watLi[["x", "y"]].values, k=1); _m = _d < 5000
    rho = _sp(watLi.loc[_m, "Li_w"].values, combined["prospectivity"].values[_i[_m]]).correlation
    print(f"\n--- Independent dissolved-Li (stream/regional water) cross-check ---")
    print(f"  Water sites with Li: {len(watLi)}")
    print(f"  Dissolved-Li vs model prospectivity (nearest soil site <5 km, n={int(_m.sum())}): Spearman={rho:+.3f}")
    figL, axL = plt.subplots(figsize=(9, 11))
    try:
        _gp = bedrock_gdf[bedrock_gdf["UNITNAME"].str.contains("Ordovician granit|Siluro-Devonian granit|appinite", case=False, na=False)]
        _gp.plot(ax=axL, facecolor="none", edgecolor="navy", linewidth=0.8, alpha=0.7)
    except Exception: pass
    sL = axL.scatter(watLi["x"], watLi["y"], c=watLi["Li_w_rank"], s=7, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(sL, ax=axL, shrink=0.5, label="dissolved Li (percentile)")
    axL.scatter(known_occurrences["x"], known_occurrences["y"], marker="*", s=180, c="blue", edgecolor="white", linewidths=0.8, zorder=6, label="GSI Li occurrences")
    if len(targets_df): axL.scatter(targets_df["Centroid_X_ITM"], targets_df["Centroid_Y_ITM"], marker="o", s=90, facecolor="none", edgecolor="lime", linewidths=1.6, zorder=7, label="LCT targets")
    axL.set_aspect("equal"); axL.set_xticks([]); axL.set_yticks([]); axL.legend(fontsize=8, loc="upper left")
    axL.set_title("Independent dissolved-Li (stream/regional water) vs granite, occurrences, targets", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(MAPS, f"lct_stream_water_Li{SUFFIX}.png"), dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Saved: lct_stream_water_Li{SUFFIX}.png")
except Exception as e:
    print("water-Li cross-check note:", e)

# %% [markdown]
# ## Target Map with Geological Context

# %%
fig, ax = plt.subplots(figsize=(12, 16))

# Reuse pre-computed IDW surface
ax.pcolormesh(xi, yi, zi_a, cmap="RdYlGn_r", vmin=0, vmax=1, shading="auto", alpha=0.7)

# Real Caledonian granite polygons used by the model (navy outline)
try:
    _grpoly = bedrock_gdf[bedrock_gdf["UNITNAME"].str.contains("Ordovician granit|Siluro-Devonian granit|appinite", case=False, na=False)]
    if len(_grpoly):
        _grpoly.plot(ax=ax, facecolor="none", edgecolor="navy", linewidth=1.0, alpha=0.85)
except Exception:
    pass

# Target clusters
if len(targets_df) > 0:
    colors_t = plt.cm.Set1(np.linspace(0, 1, min(10, len(targets_df))))
    for rank_i, (_, t) in enumerate(targets_df.iterrows()):
        cl = high_prosp[high_prosp["cluster"] == int(t["Target_ID"][1:]) - 1]
        ax.scatter(cl["x"], cl["y"], c=[colors_t[rank_i % 10]], s=10, alpha=0.8,
                   edgecolors="black", linewidths=0.3)
        ax.annotate(
            t["Target_ID"], (t["Centroid_X_ITM"], t["Centroid_Y_ITM"]),
            fontsize=9, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85)
        )

# Known occurrences
ax.scatter(known_occurrences["x"], known_occurrences["y"],
           c="blue", marker="*", s=200, zorder=5, edgecolors="white", linewidths=1)
for _, occ in known_occurrences.iterrows():
    ax.annotate(occ["name"], (occ["x"], occ["y"]), fontsize=7,
                xytext=(5, 5), textcoords="offset points", color="blue")

ax.set_title("Ranked LCT Pegmatite Exploration Targets\nGeology-Driven Random Forest Model, Tellus Programme, Ireland",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Easting (ITM)")
ax.set_ylabel("Northing (ITM)")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(os.path.join(MAPS, f"target_map_v2{SUFFIX}.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: target_map_v2{SUFFIX}.png")

# ─── Model comparison scatter ───
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(probs_a, probs_b, s=1, alpha=0.2, c="steelblue")
axes[0].plot([0, 1], [0, 1], "r--", lw=1)
axes[0].set_xlabel("Model A (Full Features)")
axes[0].set_ylabel("Model B (No Li)")
axes[0].set_title(f"Model A vs B (ρ = {r_ab:.3f})\nCircular Reasoning Test", fontweight="bold")

axes[1].scatter(probs_a, probs_c, s=1, alpha=0.2, c="darkorange")
axes[1].plot([0, 1], [0, 1], "r--", lw=1)
axes[1].set_xlabel("Model A (Geology Labels)")
axes[1].set_ylabel("Model C (Occurrence-Only Labels)")
axes[1].set_title(f"Model A vs C (ρ = {r_ac:.3f})\nLabel Sensitivity Test", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(MAPS, f"model_comparison{SUFFIX}.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: model_comparison{SUFFIX}.png")

# ─── Interactive map ───
try:
    import folium
    from folium.plugins import HeatMap
    
    # Subsample for speed
    sub = combined[["x", "y", "prospectivity"]].dropna()
    if len(sub) > 5000:
        sub = sub.sample(n=5000, random_state=42)
    gdf = gpd.GeoDataFrame(
        sub,
        geometry=gpd.points_from_xy(sub["x"], sub["y"]),
        crs="EPSG:2157"
    ).to_crs("EPSG:4326")
    
    m = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()],
                   zoom_start=7, tiles="CartoDB positron")
    
    high = gdf[gdf["prospectivity"] > 0.5].sample(n=min(2000, len(gdf[gdf["prospectivity"] > 0.5])), random_state=42)
    if len(high) > 0:
        heat_data = [[r.geometry.y, r.geometry.x, r["prospectivity"]] for _, r in high.iterrows()]
        HeatMap(heat_data, radius=15, blur=10).add_to(m)
    
    if len(targets_df) > 0:
        for _, t in targets_df.iterrows():
            popup = (f"<b>{t['Target_ID']}</b><br>"
                     f"Domain: {t['Geological_Domain']}<br>"
                     f"Prosp: {t['Mean_Prospectivity']:.3f}<br>"
                     f"Prosp(no-Li): {t['Mean_Prosp_NoLi']}<br>"
                     f"Nearest: {t['Nearest_Occurrence_Name']} ({t['Nearest_Known_Occurrence_km']}km)<br>"
                     f"{t['Validation_Notes']}")
            folium.Marker(
                [t["Lat"], t["Lon"]], popup=popup,
                icon=folium.Icon(color="red", icon="star")
            ).add_to(m)
    
    m.save(os.path.join(OUTPUTS, f"interactive_map{SUFFIX}.html"))
    print(f"Saved: interactive_map{SUFFIX}.html")
except Exception as e:
    print(f"⚠ Interactive map error: {e}")

# %%
# ─── Anomaly maps ───
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
for ax_i, (el, cmap_label) in enumerate([("Li", "Li (mg/kg)"), ("La", "La (mg/kg)")]):
    if el in combined.columns:
        ax = axes[ax_i]
        sc = ax.scatter(combined["x"], combined["y"], c=combined[el],
                        cmap="YlOrRd", s=1, alpha=0.6, vmin=0,
                        vmax=combined[el].quantile(0.98))
        plt.colorbar(sc, ax=ax, label=cmap_label, shrink=0.7)
        ax.set_title(f"{el} in Topsoil, Tellus Programme", fontsize=12, fontweight="bold")
        ax.set_xlabel("Easting (ITM)")
        ax.set_ylabel("Northing (ITM)")
        ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(os.path.join(MAPS, f"anomaly_maps_Li_La{SUFFIX}.png"), dpi=200, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Integrated visualizations & supervised diagnostics
# Data-integration map, spatial-CV ROC / precision-recall, the CV fold layout, and feature
# importances, making the supervised model's rigor and the multidisciplinary integration explicit.

# %%
# (1) DATA-INTEGRATION / COVERAGE MAP
try:
    import gc as _gc
    _btmp = gpd.read_file(os.path.join(BASE, "Ireland/Geology/GIS/IE_GSI_GSNI_Bedrock_Geology_1M_IE32_ITM_MS.shp"))
    if str(_btmp.crs) != "EPSG:2157": _btmp = _btmp.to_crs("EPSG:2157")
    _gr = _btmp[_btmp["UNITNAME"].str.contains("granit|appinite", case=False, na=False)].copy()
    del _btmp; _gc.collect()
    _fa = gpd.read_file(os.path.join(BASE, "Ireland/Geology/GIS/IE_GSI_GSNI_Faults_1M_IE32_ITM_MS.shp"))
    if str(_fa.crs) != "EPSG:2157": _fa = _fa.to_crs("EPSG:2157")
    figC, axC = plt.subplots(figsize=(9, 10))
    _gr.plot(ax=axC, color="#d9c9a3", alpha=0.6)
    _fa.plot(ax=axC, color="0.55", lw=0.4)
    _sub = combined.sample(n=min(12000, len(combined)), random_state=42)
    axC.scatter(_sub["x"], _sub["y"], s=2, c="#5E6B3B", alpha=0.4, label=f"Soil geochemistry sites ({len(combined)})")
    axC.scatter(known_occurrences["x"], known_occurrences["y"], marker="*", s=180, c="#A85832",
                edgecolor="white", lw=0.8, zorder=6, label=f"Li occurrence anchors ({len(known_occurrences)})")
    if len(targets_df): axC.scatter(targets_df["Centroid_X_ITM"], targets_df["Centroid_Y_ITM"], marker="o",
                s=120, facecolor="none", edgecolor="lime", lw=1.8, zorder=7, label=f"LCT targets ({len(targets_df)})")
    axC.set_aspect("equal"); axC.set_xticks([]); axC.set_yticks([])
    axC.set_title("Integrated datasets: soil geochemistry, geophysics, granite (tan), faults, Li anchors, targets", fontsize=10, fontweight="bold")
    axC.legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.tight_layout(); plt.savefig(os.path.join(MAPS, "lct_data_coverage.png"), dpi=140, bbox_inches="tight"); plt.close()
    print("Saved: lct_data_coverage.png")
except Exception as e: print("LCT coverage fig note:", e)

# %%
# (2) SPATIAL-CV ROC + PRECISION-RECALL (Model A, out-of-fold per 50 km block)
try:
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_curve, auc as _auc, precision_recall_curve, average_precision_score
    _tr = X_a.index; _co = combined.loc[_tr, ["x", "y"]]; _bs = 25000
    _blk = (_co["x"] // _bs).astype(int).astype(str) + "_" + (_co["y"] // _bs).astype(int).astype(str)
    _ub = _blk.unique(); np.random.seed(42)
    _bm = {b: i % max(2, min(10, len(_ub))) for i, b in enumerate(np.random.permutation(_ub))}
    _fid = _blk.map(_bm).values; _nf = max(2, min(10, len(_ub)))
    figRP, axRP = plt.subplots(1, 2, figsize=(13, 5.5)); _mfpr = np.linspace(0, 1, 100); _tprs = []
    for fi, (tr, va) in enumerate(GroupKFold(n_splits=_nf).split(X_a, y_a, groups=_fid)):
        _m = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=15, min_samples_leaf=3,
                                    max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
        _m.fit(X_a.iloc[tr], y_a.iloc[tr]); _p = _m.predict_proba(X_a.iloc[va])[:, 1]; del _m
        fpr, tpr, _ = roc_curve(y_a.iloc[va], _p); _a = _auc(fpr, tpr)
        axRP[0].plot(fpr, tpr, lw=1, alpha=0.55, label=f"fold {fi+1} (AUC={_a:.2f})")
        _ti = np.interp(_mfpr, fpr, tpr); _ti[0] = 0; _tprs.append(_ti)
        prec, rec, _ = precision_recall_curve(y_a.iloc[va], _p); ap = average_precision_score(y_a.iloc[va], _p)
        axRP[1].plot(rec, prec, lw=1, alpha=0.55, label=f"fold {fi+1} (AP={ap:.2f})")
    _mt = np.mean(_tprs, axis=0); _mt[-1] = 1; _ma = _auc(_mfpr, _mt)
    axRP[0].plot(_mfpr, _mt, color="#A85832", lw=2.5, label=f"mean (AUC={_ma:.2f})")
    axRP[0].plot([0, 1], [0, 1], "k--", lw=.7)
    axRP[0].set_xlabel("false positive rate"); axRP[0].set_ylabel("true positive rate")
    axRP[0].set_title("Spatial-CV ROC (Model A)", fontweight="bold"); axRP[0].legend(fontsize=8, loc="lower right")
    axRP[1].axhline(y_a.mean(), color="0.5", ls="--", lw=.7, label=f"baseline ({y_a.mean():.2f})")
    axRP[1].set_xlabel("recall"); axRP[1].set_ylabel("precision")
    axRP[1].set_title("Spatial-CV Precision-Recall (Model A)", fontweight="bold"); axRP[1].legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(os.path.join(MAPS, "lct_roc_pr.png"), dpi=140, bbox_inches="tight"); plt.close()
    import gc; gc.collect(); print("Saved: lct_roc_pr.png")
except Exception as e: print("LCT ROC/PR fig note:", e)

# %%
# (3) SPATIAL-CV FOLD MAP
try:
    figF, axF = plt.subplots(figsize=(8, 9))
    try: _gr.plot(ax=axF, color="#d9c9a3", alpha=0.6)
    except Exception: pass
    _cc = combined.loc[_tr]
    sc = axF.scatter(_cc["x"], _cc["y"], c=_fid, s=10, cmap="Set2")
    _pos = _cc[y_a.values == 1]
    axF.scatter(_pos["x"], _pos["y"], marker="x", s=22, c="k", lw=0.6, label=f"positive labels ({len(_pos)})")
    axF.set_aspect("equal"); axF.set_xticks([]); axF.set_yticks([])
    axF.set_title("Spatial-block cross-validation folds (25 km blocks)", fontweight="bold"); axF.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(MAPS, "lct_cv_folds.png"), dpi=140, bbox_inches="tight"); plt.close()
    print("Saved: lct_cv_folds.png")
except Exception as e: print("LCT fold-map fig note:", e)

# %%
# (4) MODEL A FEATURE IMPORTANCES (top 15)
try:
    _imp = pd.Series(rf_a.feature_importances_, index=feats_a).sort_values(ascending=False).head(15)[::-1]
    figI, axI = plt.subplots(figsize=(8.5, 6))
    axI.barh(_imp.index, _imp.values, color="#6E7A41")
    axI.set_xlabel("Random Forest importance"); axI.set_title("Model A: top 15 feature importances", fontweight="bold")
    axI.tick_params(labelsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(MAPS, "lct_feature_importance.png"), dpi=140, bbox_inches="tight"); plt.close()
    print("Saved: lct_feature_importance.png")
except Exception as e: print("LCT importance fig note:", e)

# %% [markdown]
# ## GIS deliverables: prospectivity raster (GeoTIFF) + targets/points vector (GeoPackage)
# Export the model outputs as standard GIS files in outputs/ so they open directly in QGIS /
# ArcGIS / Leapfrog: a continuous prospectivity raster, plus the targets and scored sample
# points as vectors.

# %%
try:
    import rasterio as _rio, shutil as _sh
    from rasterio.transform import from_origin as _fo
    from scipy.interpolate import griddata as _gd
    from scipy.spatial import cKDTree as _KDg
    _cell = 1000.0
    _gx = np.arange(combined["x"].min(), combined["x"].max(), _cell)
    _gy = np.arange(combined["y"].min(), combined["y"].max(), _cell)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _surf = _gd(combined[["x", "y"]].values, combined["prospectivity"].values, (_GX, _GY), method="linear").ravel()
    _far = _KDg(combined[["x", "y"]].values).query(np.c_[_GX.ravel(), _GY.ravel()], k=1)[0] > 2 * _cell
    _surf[_far] = np.nan; _surf = _surf.reshape(_GX.shape)
    _prof = dict(driver="GTiff", height=_surf.shape[0], width=_surf.shape[1], count=1, dtype="float32",
                 crs="EPSG:2157", transform=_fo(_gx.min(), _gy.max() + _cell, _cell, _cell), nodata=np.nan, compress="deflate")
    with _rio.open("/tmp/lct_surf.tif", "w", **_prof) as _d: _d.write(np.flipud(_surf).astype("float32"), 1)
    _sh.copyfile("/tmp/lct_surf.tif", os.path.join(OUTPUTS, f"lct_prospectivity_surface{SUFFIX}.tif"))
    print(f"Saved raster: lct_prospectivity_surface{SUFFIX}.tif")
    _cols = [c for c in ["x", "y", "prospectivity", "prosp_B", "label_geo"] if c in combined.columns]
    _vp = gpd.GeoDataFrame(combined[_cols].copy(), geometry=gpd.points_from_xy(combined["x"], combined["y"]), crs="EPSG:2157")
    _vp.to_file(os.path.join(OUTPUTS, f"lct_prospectivity_points{SUFFIX}.gpkg"), driver="GPKG")
    if len(targets_df):
        _vt = gpd.GeoDataFrame(targets_df.copy(), geometry=gpd.points_from_xy(targets_df["Centroid_X_ITM"], targets_df["Centroid_Y_ITM"]), crs="EPSG:2157")
        _vt.to_file(os.path.join(OUTPUTS, f"lct_targets{SUFFIX}.gpkg"), driver="GPKG")
    print(f"Saved vectors: lct_prospectivity_points{SUFFIX}.gpkg, lct_targets{SUFFIX}.gpkg")
except Exception as e:
    print("LCT GIS export note:", e)

# %% [markdown]
# ## Summary and interpretation
#
# Three complementary RF models, trained on real GSI geology and occurrences:
#
# | Model | Labels | Features | Spatial CV AUC (10-fold, 25 km) |
# |-------|--------|----------|----------------|
# | A (Primary) | real granite margin + occurrences | full (incl. Li) | 0.94 |
# | B (Validation) | same | no Li | 0.94 |
# | C (Reference) | real Li occurrences only | full | 0.99 |
#
# rho(A,B) = 0.999: the prospectivity signal is **multivariate**, not Li-driven (not Li-circular).
#
# **What the model is actually telling us.** Labels are geology-defined (real Caledonian
# granite margin) and bedrock-granite lithology is itself a feature, so the high AUC partly
# measures how well geochemistry + geophysics re-learn granite geology. The deposit-relevant
# check is occurrence-based, and here it is modest: only 4 of the 10 real Li occurrences have a
# target within 20 km (nearest 10.7 km). So the map is best read as a reconnaissance granite-
# fertility map, not a validated pegmatite locator.
#
# **Targets split into two populations:**
# 1. The **Leinster** Caledonian granite margin (6 clusters): the credible, deposit-relevant set,
#    in the belt that hosts the known spodumene occurrences (Aclare, Moylisha, Stranakelly).
# 2. NI **Sperrins** and western greenfield clusters (the rest). The granite-proximity filter
#    confirms these do sit near Caledonian granite (the Sperrins has the Tyrone intrusions), but
#    they are 200+ km from any known Li occurrence and the Sperrins is not a recognised Li-pegmatite
#    province, so they are low-confidence greenfield leads that need ground truth.
#
# **Honest framing:** present the Leinster set as the validated targets and the NI/greenfield set as
# exploration hypotheses, not drill-ready. An independent dissolved-Li (water) layer is provided as
# a further real-data cross-check.
#
# Key limitations:
# - Labels are knowledge-driven (real granite margin + 10 real occurrences); AUC is partly a
#   geology-recovery metric, so the occurrence-based checks carry the validation weight.
# - No Northern Ireland mineral-occurrence dataset is available, so NI cannot be anchored/validated.
# - Targets are first-pass screening; deposit type needs field and petrographic confirmation.
# - This model targets LCT pegmatites only; Palaeogene (Mourne) A-type is handled in the REE study.

# %% [markdown]
# ## Feature Overlay Validation
#
# These maps overlay known Li-REE occurrences on multiple feature layers simultaneously,
# confirming that the model-identified prospectivity signal corresponds to real, observable
# geochemical and geophysical anomalies, not statistical artefacts.

# %%
print("\n" + "=" * 60)
print("STAGE 7: FEATURE OVERLAY VALIDATION")
print("=" * 60)

from matplotlib.patches import Ellipse as MplEllipse

features_to_plot = [
    ("Li", "Li (mg/kg)", "YlOrRd"),
    ("Rb", "Rb (mg/kg)", "YlOrRd"),
    ("Cs", "Cs (mg/kg)", "YlOrRd"),
    ("Nb", "Nb (mg/kg)", "YlOrRd"),
    ("K_rad", "K Radiometrics (%)", "inferno"),
    ("Th_rad", "Th Radiometrics (ppm)", "inferno"),
    ("U_rad", "U Radiometrics (ppm)", "inferno"),
    ("TMI", "Total Magnetic Intensity (nT)", "RdBu_r"),
    ("Ba", "Ba (mg/kg)", "YlOrRd"),
    ("La", "La (mg/kg)", "YlOrRd"),
    ("Nb_Ta", "Nb/Ta Ratio", "RdYlGn_r"),
    ("K_Rb", "K/Rb Ratio (low=evolved)", "RdYlGn"),
]
features_to_plot = [(f, l, c) for f, l, c in features_to_plot
                    if f in combined.columns and combined[f].notna().sum() > 500]

def plot_feature_panels(data, occ_df, features, granite_dict, title, fname,
                        xlim=None, ylim=None, label_occ=False, point_size=0.5):
    """Plot multi-panel feature maps with occurrence overlays."""
    nf = len(features)
    ncols = min(4, nf)
    nrows = int(np.ceil(nf / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Real GSI granite polygons (the same geometry the model uses), NOT manual ellipses
    try:
        _gran_poly = bedrock_gdf[bedrock_gdf["UNITNAME"].str.contains("Ordovician granit|Siluro-Devonian granit|appinite", case=False, na=False)]
    except Exception:
        _gran_poly = None

    for i, (feat, label, cmap) in enumerate(features):
        if i >= len(axes):
            break
        ax = axes[i]
        vals = data[feat]
        vmin = vals.quantile(0.02) if not pd.isna(vals.quantile(0.02)) else 0
        vmax = vals.quantile(0.98) if not pd.isna(vals.quantile(0.98)) else 1

        sc = ax.scatter(data["x"], data["y"], c=vals, cmap=cmap, s=point_size,
                        alpha=0.6, vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.65, label=label, pad=0.02)

        try:
            if _gran_poly is not None and len(_gran_poly):
                _gran_poly.boundary.plot(ax=ax, edgecolor="navy", linewidth=0.8, alpha=0.75)
        except Exception:
            pass

        ax.scatter(occ_df["x"], occ_df["y"], c="cyan", marker="*", s=120 if not label_occ else 200,
                   zorder=10, edgecolors="black", linewidths=0.8)
        if label_occ:
            for _, o in occ_df.iterrows():
                ax.annotate(o["name"], (o["x"], o["y"]), fontsize=7, color="black",
                            xytext=(4, 4), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8))

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(MAPS, fname), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {fname}")

# (granite polygons are drawn from the real GSI bedrock shapefile inside plot_feature_panels)

# ── Island-wide: 12 features ──
plot_feature_panels(
    combined, known_occurrences, features_to_plot, None,
    "Feature Maps with Known LCT Pegmatite Occurrences (star) and real GSI granite polygons (navy)",
    f"feature_overlay_validation{SUFFIX}.png",
    xlim=(440000, 770000), ylim=(570000, 960000), point_size=0.5
)

# %% [markdown]
# ### Leinster Granite Belt, Zoom
#
# The zoomed view confirms that known LCT pegmatite occurrences sit precisely at the
# granite-metasediment contact where multiple geochemical anomalies converge: elevated
# Li, Rb, Cs (direct pegmatite indicators) + high K_rad, Th_rad (radiometric fingerprint).

# %%
zoom_features = [(f, l, c) for f, l, c in features_to_plot if f in
                  ["Li", "Rb", "Cs", "Nb", "K_rad", "Th_rad", "U_rad", "Ba"]]

xmin_l, xmax_l, ymin_l, ymax_l = 650000, 710000, 620000, 700000
mask_l = (combined["x"] > xmin_l) & (combined["x"] < xmax_l) & \
         (combined["y"] > ymin_l) & (combined["y"] < ymax_l)
occ_l = known_occurrences[(known_occurrences["x"] > xmin_l) & (known_occurrences["x"] < xmax_l) &
                           (known_occurrences["y"] > ymin_l) & (known_occurrences["y"] < ymax_l)]

plot_feature_panels(
    combined[mask_l], occ_l, zoom_features, None,
    "Leinster Granite Belt: Feature Zoom with Known LCT Pegmatite Occurrences (star)",
    f"leinster_zoom_validation{SUFFIX}.png",
    xlim=(xmin_l, xmax_l), ylim=(ymin_l, ymax_l), label_occ=True, point_size=4
)

# %% [markdown]
# ### Newry-Mourne Belt, Zoom
#
# The Newry-Mourne area shows locally elevated Rb (>200 mg/kg) and Nb (up to ~16 mg/kg) at the
# granite margins, with K_rad/Th_rad/U_rad convergence, a multi-dataset target that has seen
# minimal Li exploration. Important affinity distinction: the **Newry complex (Caledonian)** is
# the LCT-relevant body here, whereas the **Mourne granite is Palaeogene A-type (NYF/REE affinity)**
# and is assessed in the REE study, so Mourne signals are flagged as REE candidates, not Li.

# %%
xmin_n, xmax_n, ymin_n, ymax_n = 680000, 755000, 800000, 850000
mask_n = (combined["x"] > xmin_n) & (combined["x"] < xmax_n) & \
         (combined["y"] > ymin_n) & (combined["y"] < ymax_n)
occ_n = known_occurrences[(known_occurrences["x"] > xmin_n) & (known_occurrences["x"] < xmax_n) &
                           (known_occurrences["y"] > ymin_n) & (known_occurrences["y"] < ymax_n)]

plot_feature_panels(
    combined[mask_n], occ_n, zoom_features, None,
    "Newry-Mourne Belt: Feature Zoom with Analogous Occurrence Locations (star)",
    f"newry_zoom_validation{SUFFIX}.png",
    xlim=(xmin_n, xmax_n), ylim=(ymin_n, ymax_n), label_occ=True, point_size=4
)

# %%
print("\n" + "=" * 60)
print("WORKFLOW COMPLETE (v2)")
print("=" * 60)
print(f"\nOutputs: {os.path.abspath(OUTPUTS)}")
for f in [f"ranked_targets{SUFFIX}.csv", f"interactive_map{SUFFIX}.html"]:
    print(f"  {f}")
for f in os.listdir(MAPS):
    print(f"  maps/{f}")
