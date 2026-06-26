# %% [markdown]
# # REE Prospectivity in Ireland, open, multi-deposit-type, data-driven
# ### NGEA PDAC 2027 · Tellus dataset · label-free workflow
#
# **Scope & philosophy.** Ireland has **no confirmed REE deposit**, so there is no label to
# train/score a classifier (a supervised AUC here only re-learns a hand-made rule, invalid).
# We therefore do **unsupervised geochemical anomaly detection + data-driven favourability**,
# and we keep the deposit style **open**: REE form in carbonatites & alkaline intrusions,
# ion-adsorption clays (weathering of *any* evolved granite), placers, and black shales /
# phosphorites (USGS REE deposit models; Nat. Commun. 2020 on ion-adsorption clays). No single host
# (e.g. the Mourne A-type granite) is assumed, the data decides.
#
# **Design choices that matter:**
# 1. **Correct medium, near-total (XRF).** Aqua-regia partial leach under-reports resistant
#    REE hosts (zircon/monazite) by up to ~90×, so we build the index on **XRF/near-total**
#    chemistry, pooling all near-total soil horizons of both jurisdictions.
# 2. **Multi-element factor maps**, CLR-based LREE-Th / HREE-Y / HFSE Zr-Nb enrichment indices
#    (PCA loadings reported for context) keep different deposit styles distinct.
# 3. **Interpretable mineral-systems favourability**, prospectivity is resolved into SOURCE
#    (radiometrics + evolved-granite + P₂O₅), PATHWAY (faults) and TRAP (clay, Fe-Mn, circumneutral
#    pH, relief); a weights-of-evidence pass over the same layers is reported as a supporting
#    evidence-discrimination diagnostic. No assumed single host/buffer.
# 4. **Validation against real GSI REE-affinity occurrences** (U, fluorite, baryte, Mo, W,
#    pegmatite, phosphate) via Prediction-Area + spatial permutation, multi-host, independent.

# %%
import os, sys, glob, warnings
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from IPython.display import Image, display
except Exception:
    def display(*a, **k): pass
    def Image(*a, **k): return None
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
warnings.filterwarnings("ignore")
BASE = "." if os.path.exists("Ireland") else (".." if os.path.exists("../Ireland") else os.getcwd())
sys.path.insert(0, os.path.join(BASE, "scripts"))
import preprocess as pp
OUT = os.path.join(BASE, "outputs"); MAPS = os.path.join(OUT, "maps"); os.makedirs(MAPS, exist_ok=True)
REE = ["La", "Ce", "Nd", "Sm", "Y", "Yb", "Lu", "Tb", "Nb", "Ta", "Zr", "Th", "U"]
OX = ["Al2O3", "P2O5", "Fe2O3", "MnO", "TiO2", "CaO"]
print("BASE:", os.path.abspath(BASE))

# %% [markdown]
# ## 1. Load ALL near-total soil geochemistry (A + S horizons, ROI + NI) + weathering proxies
# Near-total REE (XRF / near-total digestions) from every available soil survey, both
# jurisdictions. Aqua-regia-only surveys are skipped for the REE index (wrong medium).

# %%
def _find(cols, names):
    for n in names:
        if n in cols: return n
    return None

def load_soil_nt(files, region, horizon):
    out = []
    for f in files:
        try:
            raw = pd.read_excel(f); cols = list(raw.columns)
        except Exception:
            continue
        xc = _find(cols, ["Easting_ITM", "EASTING_ITM", "Easting_ING", "X_ING", "EASTING", "Easting", "X"])
        yc = _find(cols, ["Northing_ITM", "NORTHING_ITM", "Northing_ING", "Y_ING", "NORTHING", "Northing", "Y"])
        if xc is None or yc is None:
            continue
        d = {"x": pd.to_numeric(raw[xc], errors="coerce"), "y": pd.to_numeric(raw[yc], errors="coerce")}
        for e in REE:  # near-total only: XRFS / near-total / plain (NOT ICP aqua-regia)
            c = _find(cols, [f"{e}_mgkg_XRFS", f"{e}_XRFS", f"{e}_mgkg_NT", f"{e}_mgkg", e, f"{e}_ppm"])
            if c: d[e] = pd.to_numeric(raw[c], errors="coerce")
        for ox in OX:
            c = _find(cols, [f"{ox}_%_XRFS", f"{ox}_pct", ox, f"{ox}_%"])
            if c: d[ox] = pd.to_numeric(raw[c], errors="coerce")
        c = _find(cols, ["pH_CaCl2", "pH_H2O", "pH"]);  d["pH"] = pd.to_numeric(raw[c], errors="coerce") if c else np.nan
        c = _find(cols, ["LOI_450C_%", "LOI_%", "LOI"]); d["LOI"] = pd.to_numeric(raw[c], errors="coerce") if c else np.nan
        df = pd.DataFrame(d)
        df["source_crs"] = 2157 if "ITM" in (xc or "") else (29903 if "ING" in (xc or "") else (2157 if df["x"].mean() > 400000 else 29903))
        df["region"] = region; df["horizon"] = horizon
        df["block"] = f"{region}_{horizon}_{os.path.basename(f)[:8]}"
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

srcs = [
    load_soil_nt(glob.glob(os.path.join(BASE, "Ireland/Geochem/Shallow_Topsoil_A/*/*.xlsx")), "ROI", "A"),
    load_soil_nt(glob.glob(os.path.join(BASE, "Ireland/Geochem/Deeper_Topsoil_S/*/*.xlsx")), "ROI", "S"),
    load_soil_nt(glob.glob(os.path.join(BASE, "Northern Ireland/2. Geochem/Regional_Soils_A_XRF.xls")), "NI", "A"),
    load_soil_nt(glob.glob(os.path.join(BASE, "Northern Ireland/2. Geochem/Regional_Soils_S_NearTotal_HDL.xls")), "NI", "S"),
]
A = pp.reproject_to_itm(pd.concat([s for s in srcs if len(s)], ignore_index=True)).reset_index(drop=True)
elems = [e for e in REE if e in A.columns and A[e].notna().mean() > 0.4]
A = A[A[elems].notna().any(axis=1)].reset_index(drop=True)
print(f"Near-total soil samples pooled: {len(A)}")
print(A.groupby(["region", "horizon"]).size().to_string())
print(f"REE/HFSE elements used (>40% coverage): {elems}")
print("weathering proxies present:", [c for c in ["pH", "LOI"] + OX if c in A.columns and A[c].notna().mean() > 0.4])

# %% [markdown]
# ## 1b. Data-quality control, why near-total? Aqua Regia vs XRF at the same NI sites
# Regional soils exist in two digestions. Aqua regia is a *partial* leach that does not dissolve
# resistant REE hosts (zircon, monazite), so it under-reports REE. We quantify this at co-located
# NI sites and confirm near-total (XRF) is the correct medium for an REE study.

# %%
_ar = pp.reproject_to_itm(pp.load_ni_geochem(BASE)); _ar = _ar[_ar["block"] == "NI_A_AquaRegia"].reset_index(drop=True)
_xr = A[(A["region"] == "NI") & (A["horizon"] == "A")].reset_index(drop=True)
if len(_ar) and len(_xr):
    _d, _j = cKDTree(_xr[["x", "y"]].values).query(_ar[["x", "y"]].values, k=1); _m = _d < 100
    print(f"Co-located NI A-horizon sites (Aqua Regia vs XRF, <100 m): {int(_m.sum())}")
    print(f"  {'elem':5s}{'AquaRegia':>11s}{'XRF':>9s}{'XRF/AR':>8s}")
    for e in ["La", "Ce", "Nd", "Y", "Yb", "Nb", "Zr", "Th"]:
        if e in _ar.columns and e in _xr.columns:
            ar = _ar.loc[_m, e].median(); xr = _xr.iloc[_j[_m]][e].median()
            print(f"  {e:5s}{ar:11.2f}{xr:9.2f}{xr / (ar + 1e-9):8.1f}")
    print("  XRF/AR >> 1 confirms aqua regia misses resistant-mineral-hosted REE, so near-total is used throughout.")

# %% [markdown]
# ## 2. Directed enrichment index (per-survey ranks) + CLR
# Per-survey percentile ranks level method/horizon differences; the mean over the suite is a
# directed "how REE-enriched is this site vs its peers" index. CLR handles compositional closure.

# %%
A = pd.concat([A, pp.clr_transform(A, elems)], axis=1)
clr_cols = [f"clr_{e}" for e in elems]
A["REE_anom"] = pd.DataFrame({e: A.groupby("block")[e].rank(pct=True) for e in elems}).mean(axis=1)

# %% [markdown]
# ## 3. Multi-element factor maps (LREE-Th / HREE-Y / HFSE Zr-Nb), keep styles distinct

# %%
Xs = np.nan_to_num(StandardScaler().fit_transform(A[clr_cols].fillna(A[clr_cols].median())))
pca = PCA(n_components=min(5, len(clr_cols)), random_state=42).fit(Xs)
print("PCA var explained (top3):", (pca.explained_variance_ratio_[:3] * 100).round(1), "%")
print(pd.DataFrame(pca.components_[:3].T, index=elems, columns=["PC1", "PC2", "PC3"]).round(2).to_string())
def fac(pos):
    w = np.array([1.0 if e in pos else 0.0 for e in elems])
    return pd.Series(Xs @ w, index=A.index).rank(pct=True)
A["F_LREE"] = fac([e for e in ["La", "Ce", "Nd", "Sm", "Th"] if e in elems])
A["F_HREE"] = fac([e for e in ["Y", "Yb", "Lu", "Tb"] if e in elems])
A["F_HFSE"] = fac([e for e in ["Zr", "Nb", "Ta"] if e in elems])

# %% [markdown]
# ## 4. Evidence layers: geology (all hosts), structure, geophysics, weathering, no host prior

# %%
A = pp.sample_all_geophysics(A, BASE)
bed = gpd.read_file(os.path.join(BASE, "Ireland/Geology/GIS/IE_GSI_GSNI_Bedrock_Geology_1M_IE32_ITM_MS.shp"))
bed = bed.to_crs("EPSG:2157") if str(bed.crs) != "EPSG:2157" else bed
pts = gpd.GeoSeries(gpd.points_from_xy(A["x"], A["y"]), crs="EPSG:2157")
A["UNITNAME"] = gpd.sjoin(gpd.GeoDataFrame(A[["x"]].copy(), geometry=pts, crs="EPSG:2157"),
                          bed[["UNITNAME", "geometry"]], how="left", predicate="within")["UNITNAME"].reindex(A.index).values
def host(u):
    s = str(u).lower()
    if "palaeogene" in s and "granit" in s: return "A-type granite"
    if "granit" in s or "appinite" in s: return "Caledonian granite"
    if any(k in s for k in ["dalradian", "schist", "gneiss", "metased", "quartzite"]): return "Metasediment"
    if any(k in s for k in ["limestone", "calcar", "dolomit", "chalk"]): return "Carbonate"
    if any(k in s for k in ["sandstone", "mudstone", "shale", "greywacke", "siltstone", "slate"]): return "Clastic/shale"
    if any(k in s for k in ["basalt", "rhyolite", "volcan", "tuff"]): return "Volcanic"
    return "Other"
A["host"] = A["UNITNAME"].map(host)
A["dist_granite_km"] = pts.distance(bed[bed["UNITNAME"].str.contains("granit|appinite", case=False, na=False)].geometry.union_all()).values / 1000.0
fa = gpd.read_file(os.path.join(BASE, "Ireland/Geology/GIS/IE_GSI_GSNI_Faults_1M_IE32_ITM_MS.shp"))
fa = fa.to_crs("EPSG:2157") if str(fa.crs) != "EPSG:2157" else fa
A["dist_fault_km"] = pts.distance(fa.geometry.union_all()).values / 1000.0
A["elev"] = pp.sample_raster_at_points(os.path.join(BASE, "Ireland/Geology/GIS/dem_irl_itm-1.tif"), A["x"].values, A["y"].values)  # relief context (ion-adsorption/placer)

# %% [markdown]
# ## 5. Source vs secondary control (independent diagnostics)
# Correlate the index with airborne radiometrics (independent) and weathering proxies to see
# whether anomalies are primary fertility or clay/Fe-Mn/phosphate scavenging (ion-adsorption style).

# %%
print("Spearman(REE_anom, x):")
for c in ["Th_rad", "U_rad", "K_rad", "Al2O3", "P2O5", "Fe2O3", "MnO", "pH", "LOI", "dist_granite_km"]:
    if c in A.columns and A[c].notna().sum() > 1000:
        print(f"   {c:14s} {spearmanr(A['REE_anom'], A[c], nan_policy='omit').correlation:+.3f}")
if "elev" in A.columns:
    print(f"   {'elevation':14s} {spearmanr(A['REE_anom'], A['elev'], nan_policy='omit').correlation:+.3f}")

# %% [markdown]
# ## 5b. Weathering correction, is there a PRIMARY signal under the regolith overprint?
# The raw anomaly is regolith-controlled (clay/Fe-Mn). To test for primary bedrock fertility we
# regress the index on the weathering proxies (clay, Fe-Mn oxides, LOI, pH) and keep the
# **residual** = REE enrichment *beyond* what weathering explains (Th/Zr are immobile and REE is
# largely residual: Su et al. 2017). We also add **La/Yb** (LREE/HREE
# fractionation) to discriminate styles, high = carbonatite/alkaline LREE; low = HREE / ion-adsorption.

# %%
from sklearn.linear_model import LinearRegression
wcols = [c for c in ["Al2O3", "Fe2O3", "MnO", "LOI", "pH"] if c in A.columns and A[c].notna().mean() > 0.4]
Wm = StandardScaler().fit_transform(A[wcols].fillna(A[wcols].median()))
lr = LinearRegression().fit(Wm, A["REE_anom"].values)
r2 = lr.score(Wm, A["REE_anom"].values)
A["REE_resid"] = pd.Series(A["REE_anom"].values - lr.predict(Wm), index=A.index).rank(pct=True)
print(f"Weathering proxies {wcols} explain R^2={r2:.2f} of the REE anomaly (high => regolith-dominated)")
print("Residual (primary-fertility candidate) vs raw anomaly, correlation with primary controls:")
for c in ["dist_granite_km", "Th_rad", "U_rad"]:
    if c in A.columns:
        print(f"   {c:16s} residual {spearmanr(A['REE_resid'], A[c], nan_policy='omit').correlation:+.3f} | raw {spearmanr(A['REE_anom'], A[c], nan_policy='omit').correlation:+.3f}")
if "La" in A.columns and "Yb" in A.columns:
    A["LaYb"] = A["La"] / (A["Yb"] + 1e-6); A["LaYb_rank"] = A["LaYb"].rank(pct=True)
    print(f"LREE/HREE (La/Yb): median={A['LaYb'].median():.1f}; P90={A['LaYb'].quantile(.9):.1f} (high=LREE carbonatite/alkaline)")

# %% [markdown]
# ## 6. C-A fractal anomaly threshold + weights-of-evidence (evidence-discrimination diagnostic)

# %%
def ca_break(v):
    v = np.asarray(v); v = v[np.isfinite(v)]
    ts = np.unique(np.quantile(v, np.linspace(0.5, 0.999, 60))); ar = np.array([(v >= t).sum() for t in ts], float)
    k = ar > 0; ts, ar = ts[k], ar[k]; lx, ly = np.log10(ts), np.log10(ar); best = None
    for i in range(6, len(ts) - 6):
        try:
            e = sum(np.sum((ly[s] - np.polyval(np.polyfit(lx[s], ly[s], 1), lx[s])) ** 2) for s in (slice(0, i), slice(i, len(ts))) if np.ptp(lx[s]) > 0)
        except Exception:
            continue
        if best is None or e < best[0]: best = (e, ts[i])
    return best[1] if best else float(np.quantile(v, 0.95))
t_anom = ca_break(A["REE_anom"].values)
anom = (A["REE_anom"] >= t_anom).values
base = anom.mean()
print(f"C-A anomaly threshold={t_anom:.3f}; anomalous sites={int(anom.sum())} ({base*100:.1f}%)")

def woe(series, qbins=True, eps=1e-6):
    s = series.fillna(series.median()) if series.dtype.kind in "fc" else series.fillna("NA")
    cats = pd.qcut(s.rank(method="first"), 4, labels=False, duplicates="drop") if qbins else s.astype("category").cat.codes
    w = np.zeros(len(s))
    for g in pd.unique(cats):
        m = (cats == g).values; p = anom[m].mean() if m.sum() else base
        w[m] = np.log((p + eps) / (base + eps))
    return w

ev = {"host": (A["host"], False), "near_granite": (-A["dist_granite_km"], True), "near_fault": (-A["dist_fault_km"], True),
      "Th_rad": (A.get("Th_rad"), True), "U_rad": (A.get("U_rad"), True), "clay_Al2O3": (A.get("Al2O3"), True),
      "phosphate_P2O5": (A.get("P2O5"), True), "FeMn_oxide": ((A.get("Fe2O3", pd.Series(0, index=A.index)).fillna(0) + A.get("MnO", pd.Series(0, index=A.index)).fillna(0)), True),
      "elevation": (A.get("elev"), True)}
W = np.zeros(len(A)); infl = {}
for nm, (ser, q) in ev.items():
    if ser is not None and pd.Series(ser).notna().sum() > 1000:
        wv = woe(pd.Series(ser, index=A.index), qbins=q); W += wv; infl[nm] = float(np.max(wv) - np.min(wv))
A["woe_score"] = pd.Series(W, index=A.index).rank(pct=True)  # diagnostic; final favourability built in 6c (mineral systems)
print("Evidence influence (max-min WoE weight; larger = more discriminating):")
for k, v in sorted(infl.items(), key=lambda t: -t[1]): print(f"   {k:16s} {v:.2f}")

# %% [markdown]
# ## 6b. UCC-normalised REE patterns of the anomalies (signature & fractionation)
# Normalising to Upper Continental Crust (Rudnick & Gao 2003) shows the enrichment factor and the
# LREE-vs-HREE slope of the targets vs background, the diagnostic REE "spider" signature that
# fingerprints the style (steep LREE = monazite/alkaline-carbonatite; flat/HREE = ion-adsorption/xenotime).

# %%
UCC = {"La": 31, "Ce": 63, "Nd": 27, "Sm": 4.7, "Y": 21, "Yb": 1.9, "Nb": 12, "Ta": 0.9, "Zr": 193, "Th": 10.5, "U": 2.7}
order_e = [e for e in ["La", "Ce", "Nd", "Sm", "Y", "Yb", "Nb", "Ta", "Zr", "Th", "U"] if e in A.columns]
q975 = np.nanquantile(A["REE_anom"], 0.975)
grp = {"strong (top 2.5%)": A["REE_anom"] >= q975,
       "anomalous": (A["REE_anom"] >= t_anom) & (A["REE_anom"] < q975),
       "background": A["REE_anom"] < t_anom}
fig, ax = plt.subplots(figsize=(10, 6))
for lbl, msk in grp.items():
    patt = [A.loc[msk, e].median() / UCC[e] for e in order_e]
    ax.plot(order_e, patt, marker="o", label=f"{lbl} (n={int(msk.sum())})")
ax.set_yscale("log"); ax.axhline(1, color="k", lw=.6, ls="--")
ax.set_ylabel("median sample / Upper Continental Crust"); ax.set_title("UCC-normalised REE/HFSE patterns", fontweight="bold")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_ucc_patterns.png"), dpi=140, bbox_inches="tight"); plt.close(); display(Image(filename=os.path.join(MAPS, "ree_v4_ucc_patterns.png")))
print("Strong-anomaly median enrichment (xUCC):", {e: round(float(A.loc[grp['strong (top 2.5%)'], e].median() / UCC[e]), 1) for e in order_e})
laN, ybN = A.loc[grp['strong (top 2.5%)'], 'La'].median() / UCC['La'], A.loc[grp['strong (top 2.5%)'], 'Yb'].median() / UCC['Yb']
print(f"(La/Yb)_UCC of strong anomalies = {laN / ybN:.2f}  (>1 LREE-enriched; <1 HREE-enriched)")

# %% [markdown]
# ## 6c. Mineral-systems decomposition, an *interpretable, explainable* favourability
# We resolve prospectivity into the components of a mineral system (McCuaig & Hronsky 2014; for
# ion-adsorption REE: Nat. Commun. 2020), so every target is *explained*:
#   • **SOURCE**, REE fertility: airborne Th/U radiometrics, proximity to evolved granite, P₂O₅ (monazite/apatite).
#   • **PATHWAY**, structural architecture: proximity to mapped faults.
#   • **TRAP**, regolith accumulation: clay (Al₂O₃ ≈ kaolinite/halloysite), Fe-Mn oxides,
#     circumneutral pH (REE³⁺ adsorption optimum) and subdued relief (preservation).
# Favourability = geometric mean of the observed anomaly with a coherent SOURCE×PATHWAY×TRAP.

# %%
def rk(s, inv=False):
    if s is None: return pd.Series(np.nan, index=A.index)
    r = pd.Series(np.asarray(s, dtype=float), index=A.index).rank(pct=True)
    return (1 - r) if inv else r
_ph = A["pH"] if "pH" in A.columns else pd.Series(np.nan, index=A.index)
ph_opt = pd.Series(np.exp(-((_ph - 6.0) ** 2) / 2.0), index=A.index).rank(pct=True)   # circumneutral optimum
_femn = A.get("Fe2O3", pd.Series(0, index=A.index)).fillna(0) + A.get("MnO", pd.Series(0, index=A.index)).fillna(0)
A["SOURCE"] = pd.concat([rk(A.get("Th_rad")), rk(A.get("U_rad")), rk(A.get("dist_granite_km"), inv=True), rk(A.get("P2O5"))], axis=1).mean(axis=1)
A["PATHWAY"] = rk(A.get("dist_fault_km"), inv=True)
A["TRAP"] = pd.concat([rk(A.get("Al2O3")), rk(_femn), ph_opt, rk(A.get("elev"), inv=True)], axis=1).mean(axis=1)
_sys = (A["SOURCE"].clip(1e-3) * A["PATHWAY"].clip(1e-3) * A["TRAP"].clip(1e-3)) ** (1 / 3)
A["favourability"] = ((A["REE_anom"].clip(1e-3) * _sys) ** 0.5).rank(pct=True)
print("Mineral-system sub-indices, Spearman with observed anomaly:")
for c in ["SOURCE", "PATHWAY", "TRAP"]:
    print(f"   {c:8s} {spearmanr(A['REE_anom'], A[c], nan_policy='omit').correlation:+.3f}")
figm, axm = plt.subplots(1, 4, figsize=(20, 6))
for a, col in zip(axm, ["SOURCE", "PATHWAY", "TRAP", "favourability"]):
    sc = a.scatter(A["x"], A["y"], c=A[col], s=2, cmap="viridis", vmin=0, vmax=1); plt.colorbar(sc, ax=a, shrink=0.5)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title(col, fontweight="bold")
plt.suptitle("Mineral-system components and favourability", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_mineral_systems.png"), dpi=140, bbox_inches="tight"); plt.close(); display(Image(filename=os.path.join(MAPS, "ree_v4_mineral_systems.png")))

# %% [markdown]
# ## 7. Validation & hypothesis discrimination
# Label-free checks: (a) spatial coincidence with **real GSI REE-affinity occurrences**
# (U, fluorite, baryte, Mo, W, pegmatite, phosphate), this tests the *primary granite /
# hydrothermal* hypothesis; if the soil REE anomaly is a secondary (weathering/clay) signal it
# should NOT coincide with those primary occurrences. (b) **A to S horizon reproducibility**, does
# the anomaly repeat in an independently-sampled soil horizon?

# %%
mins = gpd.read_file(os.path.join(BASE, "Ireland/Geology/GIS/IE_GSI_Mineral_Locations_IE26_ITM.shp"))
mins = mins.to_crs("EPSG:2157") if str(mins.crs) != "EPSG:2157" else mins
gset = {"U", "URAN", "TORB", "AUTU", "PITC", "FLUO", "BARY", "Bary", "PHOS", "BE", "BERY", "Be", "PEGM", "MO", "W", "SN", "TA", "NMR", "nmr"}
occ = mins[mins["MIN_TYPE"].isin(gset)].copy()
treeA = cKDTree(A[["x", "y"]].values)
occ_xy = np.c_[occ.geometry.x.values, occ.geometry.y.values]
dnear, nn = treeA.query(occ_xy, k=1)
keep = dnear < 5000; occ_xy, nn = occ_xy[keep], nn[keep]; occ = occ.iloc[keep]
site_rank = A["favourability"].rank(ascending=False, pct=True).values  # 0=best
occ_rank = site_rank[nn]
print(f"Independent REE-affinity occurrences (soil site <5km): {len(occ)}")
for fr in [0.05, 0.10, 0.20]:
    cap = (occ_rank <= fr).mean(); print(f"   top {int(fr*100):2d}% favourability captures {cap*100:4.0f}% of occurrences (lift x{cap/fr:.1f})")
rng = np.random.default_rng(0); obs = (occ_rank <= 0.10).mean(); nsite = len(A); null = np.empty(10000)
for i in range(10000):
    null[i] = ((rng.permutation(nsite) / nsite)[nn] <= 0.10).mean()
pval = (null >= obs).mean()
print(f"   permutation null top-10%: observed={obs*100:.0f}%, null≈{null.mean()*100:.0f}%, p={pval:.4f}")
print("   low capture (lift<1) means the soil REE anomaly does NOT track primary granite/")
print("      hydrothermal occurrences, consistent with a SECONDARY weathering/clay origin.")
Ah = A[A["horizon"] == "A"]; Sh = A[A["horizon"] == "S"]; repro = float("nan")
if len(Ah) > 100 and len(Sh) > 100:
    dd2, jj2 = cKDTree(Sh[["x", "y"]].values).query(Ah[["x", "y"]].values, k=1); mk = dd2 < 200
    if mk.sum() > 50:
        repro = spearmanr(Ah.loc[mk, "REE_anom"].values, Sh.iloc[jj2[mk]]["REE_anom"].values).correlation
        print(f"   A vs S horizon reproducibility (n={int(mk.sum())} <200 m): Spearman={repro:+.3f}  (independent-medium validation)")
res_rank = A["REE_resid"].rank(ascending=False, pct=True).values
print(f"   occurrence top-10% capture: weathering-corrected residual = {(res_rank[nn] <= 0.10).mean()*100:.0f}%  vs raw anomaly {(occ_rank <= 0.10).mean()*100:.0f}%"
      f"  (does removing the regolith overprint reveal a primary signal?)")

# %% [markdown]
# ## 7b. Independent medium, stream sediments (catchment-scale; sees through regolith)
# Stream sediments integrate upstream catchment bedrock and are the classic reconnaissance REE
# medium, the GSI/GSNI Tellus critical-metals work on the Mournes used exactly this
# (GSI/GSNI Tellus). Because they reflect *provenance* rather than in-situ weathering, they are a strong
# **independent** check and a better primary-fertility proxy than residual soils.

# %%
strm = pp.reproject_to_itm(load_soil_nt(glob.glob(os.path.join(BASE, "Ireland/Geochem/C_Stream_sediment_geochemistry/*.xlsx")), "ROI", "stream"))
selem = [e for e in elems if e in strm.columns and strm[e].notna().mean() > 0.4]
strm = strm[strm[selem].notna().any(axis=1)].reset_index(drop=True)
strm["REE_anom"] = pd.DataFrame({e: strm[e].rank(pct=True) for e in selem}).mean(axis=1)
print(f"Stream-sediment sites: {len(strm)}; near-total elements: {selem}")
cell5 = 5000.0
cid = lambda df: (np.floor(df["x"] / cell5).astype(int).astype(str) + "_" + np.floor(df["y"] / cell5).astype(int).astype(str))
gs = A.assign(cid=cid(A)).groupby("cid")["REE_anom"].median(); gt = strm.assign(cid=cid(strm)).groupby("cid")["REE_anom"].median()
both = pd.concat([gs.rename("soil"), gt.rename("stream")], axis=1).dropna()
rho_ss = spearmanr(both["soil"], both["stream"]).correlation
print(f"Soil vs stream-sediment agreement on 5 km cells (n={len(both)}): Spearman={rho_ss:+.3f}  (independent media)")
ptsS = gpd.GeoSeries(gpd.points_from_xy(strm["x"], strm["y"]), crs="EPSG:2157")
strm["dist_granite_km"] = ptsS.distance(bed[bed["UNITNAME"].str.contains("granit|appinite", case=False, na=False)].geometry.union_all()).values / 1000
print(f"   rho(stream REE_anom, dist_granite_km) = {spearmanr(strm['REE_anom'], strm['dist_granite_km']).correlation:+.3f}  (soil was ~+0.02)")
treeS = cKDTree(strm[["x", "y"]].values); dS, nnS = treeS.query(occ_xy, k=1); keepS = dS < 5000
srank = strm["REE_anom"].rank(ascending=False, pct=True).values
print(f"   stream top-10% captures {(srank[nnS[keepS]] <= 0.10).mean()*100:.0f}% of occurrences  "
      f"(soil raw {(occ_rank <= 0.10).mean()*100:.0f}%, weathering-residual {(res_rank[nn] <= 0.10).mean()*100:.0f}%)")

fig, ax = plt.subplots(1, 2, figsize=(13, 7))
s = ax[0].scatter(strm["x"], strm["y"], c=strm["REE_anom"], s=4, cmap="inferno", vmin=0, vmax=1); plt.colorbar(s, ax=ax[0], shrink=0.5)
ax[0].set_aspect("equal"); ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].set_title("Stream-sediment REE anomaly (catchment)", fontsize=10, fontweight="bold")
ax[1].scatter(both["soil"], both["stream"], s=8, alpha=0.4); ax[1].plot([0, 1], [0, 1], "k--", lw=.7)
ax[1].set_xlabel("soil REE anomaly (5 km cell median)"); ax[1].set_ylabel("stream REE anomaly"); ax[1].set_title(f"Soil vs stream agreement (rho={rho_ss:+.2f})", fontsize=10, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_streamsed.png"), dpi=140, bbox_inches="tight"); plt.close(); display(Image(filename=os.path.join(MAPS, "ree_v4_streamsed.png")))

# %% [markdown]
# ## 7c. Independent medium: stream/regional WATERS (dissolved REE)
# A third real geochemical medium, hydrogeochemistry: dissolved REE in ROI Tellus stream
# water and NI regional waters. Dissolved REE is low and strongly pH-controlled, so this is
# a conservative independent cross-check, not a primary medium. Weak agreement with the soil
# anomaly would reinforce that the soil signal is a solid-phase regolith feature.

# %%
try:
    def _load_water(path, ecol, ncol, elems):
        raw = pd.read_excel(path); cols = list(raw.columns)
        def fw(std):
            if std.upper() in cols: return std.upper()
            for c in cols:
                cl = str(c).lower()
                if cl == std.lower(): return c
                if cl.startswith(std.lower() + "_") and "gl" in cl: return c
            return None
        if ecol not in cols or ncol not in cols: return pd.DataFrame()
        d = {"x": pd.to_numeric(raw[ecol], errors="coerce"), "y": pd.to_numeric(raw[ncol], errors="coerce")}
        for e in elems:
            c = fw(e)
            if c is not None: d[e] = pd.to_numeric(raw[c], errors="coerce")
        df = pd.DataFrame(d).dropna(subset=["x", "y"])
        if len(df): df["source_crs"] = 2157 if df["x"].mean() > 400000 else 29903
        return df
    _wl = []
    for _f in glob.glob(os.path.join(BASE, "Ireland/Geochem/W_Stream_water_geochemistry/*.xlsx")):
        _wl.append(_load_water(_f, "Easting_ING", "Northing_ING", ["La", "Ce", "Nd", "Y", "Yb"]))
    _nif = os.path.join(BASE, "Northern Ireland/2. Geochem/Regional_Waters_ICP.xls")
    if os.path.exists(_nif): _wl.append(_load_water(_nif, "EASTING", "NORTHING", ["La", "Ce", "Nd", "Y"]))
    wat = pp.reproject_to_itm(pd.concat([w for w in _wl if len(w)], ignore_index=True))
    wree = [e for e in ["La", "Ce", "Nd", "Y", "Yb"] if e in wat.columns and wat[e].notna().mean() > 0.3]
    wat = wat[wat[wree].notna().any(axis=1)].reset_index(drop=True)
    wat["REE_anom"] = pd.DataFrame({e: wat[e].rank(pct=True) for e in wree}).mean(axis=1)
    print(f"Stream/regional water sites: {len(wat)}; dissolved REE elements used: {wree}")
    cw = lambda df: (np.floor(df["x"] / 5000).astype(int).astype(str) + "_" + np.floor(df["y"] / 5000).astype(int).astype(str))
    gso = A.assign(c=cw(A)).groupby("c")["REE_anom"].median(); gwa = wat.assign(c=cw(wat)).groupby("c")["REE_anom"].median()
    bw = pd.concat([gso.rename("soil"), gwa.rename("water")], axis=1).dropna()
    rho_w = spearmanr(bw["soil"], bw["water"]).correlation if len(bw) > 10 else float("nan")
    print(f"Soil vs water agreement on 5 km cells (n={len(bw)}): Spearman={rho_w:+.3f}")
    print("   dissolved REE is pH-controlled and low; weak agreement is expected and reinforces a solid-phase regolith origin for the soil anomaly.")
    figw, axw = plt.subplots(1, 2, figsize=(13, 6))
    sw = axw[0].scatter(wat["x"], wat["y"], c=wat["REE_anom"], s=5, cmap="viridis", vmin=0, vmax=1); plt.colorbar(sw, ax=axw[0], shrink=0.5)
    axw[0].set_aspect("equal"); axw[0].set_xticks([]); axw[0].set_yticks([]); axw[0].set_title("Dissolved REE in stream/regional water", fontsize=10, fontweight="bold")
    axw[1].scatter(bw["soil"], bw["water"], s=8, alpha=0.4); axw[1].plot([0, 1], [0, 1], "k--", lw=.6)
    axw[1].set_xlabel("soil REE anomaly (5 km cell)"); axw[1].set_ylabel("water REE anomaly"); axw[1].set_title(f"Soil vs water (rho={rho_w:+.2f})", fontsize=10, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_water.png"), dpi=140, bbox_inches="tight"); plt.close(); display(Image(filename=os.path.join(MAPS, "ree_v4_water.png")))
except Exception as e:
    print("water cross-check note:", e)

# %% [markdown]
# ## 8. Targets (multi-host) + continuous favourability raster (GeoTIFF)

# %%
hi = A[A["favourability"] >= A["favourability"].quantile(0.985)].copy(); rows = []
if len(hi) > 5:
    hi["cl"] = DBSCAN(eps=10, min_samples=4).fit(hi[["x", "y"]].values / 1000).labels_
    for c in sorted(set(hi["cl"]) - {-1}):
        m = hi[hi["cl"] == c]
        facs = {"LREE": m["F_LREE"].mean(), "HREE": m["F_HREE"].mean(), "HFSE": m["F_HFSE"].mean()}
        rows.append({"x": round(m["x"].mean()), "y": round(m["y"].mean()), "n": len(m),
                     "favourability": round(m["favourability"].mean(), 3), "REE_anom": round(m["REE_anom"].mean(), 2),
                     "SOURCE": round(m["SOURCE"].mean(), 2), "PATHWAY": round(m["PATHWAY"].mean(), 2), "TRAP": round(m["TRAP"].mean(), 2),
                     "dom_factor": max(facs, key=facs.get), "LaYb": round(m["LaYb"].median(), 1) if "LaYb" in m.columns else np.nan,
                     "dom_host": m["host"].mode().iloc[0], "region": m["region"].mode().iloc[0], "near_granite_km": round(m["dist_granite_km"].mean(), 1)})
tdf = pd.DataFrame(rows).sort_values("favourability", ascending=False).reset_index(drop=True)
if len(tdf): tdf.insert(0, "target", [f"R{i+1:02d}" for i in range(len(tdf))])
tdf.to_csv(os.path.join(OUT, "ree_v4_targets.csv"), index=False)
print(f"Targets: {len(tdf)}")
if len(tdf):
    print(tdf[["target", "x", "y", "n", "favourability", "REE_anom", "SOURCE", "PATHWAY", "TRAP", "dom_factor", "dom_host", "region"]].to_string(index=False))
    print("\n=== MINERAL-SYSTEM SCORECARDS (why each target is there) ===")
    for _, t in tdf.iterrows():
        lead = max([("SOURCE", t["SOURCE"]), ("PATHWAY", t["PATHWAY"]), ("TRAP", t["TRAP"])], key=lambda z: z[1])[0]
        style = {"HREE": "HREE / ion-adsorption-clay", "LREE": "LREE / monazite-alkaline", "HFSE": "HFSE / alkaline-A-type"}[t["dom_factor"]]
        bits = []
        if t["SOURCE"] >= 0.6: bits.append("fertile source (radiometric Th/U +/- evolved granite)")
        if t["PATHWAY"] >= 0.6: bits.append("fault-focused")
        if t["TRAP"] >= 0.6: bits.append("clay-rich regolith trap (circumneutral pH, low relief)")
        print(f"{t['target']}: {t['region']}, {t['dom_host']}, ~{t['near_granite_km']}km to granite | "
              f"S={t['SOURCE']:.2f} P={t['PATHWAY']:.2f} T={t['TRAP']:.2f} | {style}, La/Yb={t['LaYb']}")
        print(f"    why: system led by {lead}; " + (", ".join(bits) if bits else "mixed/weak components") + ".")

# %% [markdown]
# ### Target zoom-ins (local mineral-system context)
# %%
gr = bed[bed["UNITNAME"].str.contains("granit|appinite", case=False, na=False)]
tz = tdf.head(6)  # zoom the 6 leading targets for readability
if len(tz):
    ncol = min(len(tz), 2); nrow = int(np.ceil(len(tz) / ncol))
    figz, axz = plt.subplots(nrow, ncol, figsize=(7 * ncol, 6 * nrow)); axz = np.atleast_1d(axz).ravel(); R = 30000
    for i, (_, t) in enumerate(tz.iterrows()):
        a = axz[i]
        sub = A[A["x"].between(t["x"] - R, t["x"] + R) & A["y"].between(t["y"] - R, t["y"] + R)]
        sc = a.scatter(sub["x"], sub["y"], c=sub["REE_anom"], s=12, cmap="inferno", vmin=0, vmax=1)
        try:
            gg = gr.cx[t["x"] - R:t["x"] + R, t["y"] - R:t["y"] + R]
            if len(gg): gg.boundary.plot(ax=a, color="cyan", lw=1.0)
        except Exception: pass
        try:
            ff = fa.cx[t["x"] - R:t["x"] + R, t["y"] - R:t["y"] + R]
            if len(ff): ff.plot(ax=a, color="0.45", lw=0.5)
        except Exception: pass
        a.scatter(t["x"], t["y"], marker="*", s=320, facecolor="none", edgecolor="lime", lw=2)
        a.set_title(f"{t['target']}: {t['dom_factor']} | S{t['SOURCE']:.1f} P{t['PATHWAY']:.1f} T{t['TRAP']:.1f}", fontsize=9, fontweight="bold")
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for j in range(len(tz), len(axz)): axz[j].axis("off")
    figz.colorbar(sc, ax=axz.tolist(), shrink=0.5, label="REE anomaly index")
    plt.suptitle("Target zoom-ins, local anomaly + faults (grey) + granite (cyan)", fontweight="bold")
    plt.savefig(os.path.join(MAPS, "ree_v4_target_zooms.png"), dpi=140, bbox_inches="tight"); plt.close(); display(Image(filename=os.path.join(MAPS, "ree_v4_target_zooms.png")))

# %%
import rasterio, shutil
from rasterio.transform import from_origin
cell = 1000.0   # finer favourability raster (1 km)
gx = np.arange(A["x"].min(), A["x"].max(), cell); gy = np.arange(A["y"].min(), A["y"].max(), cell); GX, GY = np.meshgrid(gx, gy)
surf = griddata(A[["x", "y"]].values, A["favourability"].values, (GX, GY), method="linear").ravel()
surf[treeA.query(np.c_[GX.ravel(), GY.ravel()], k=1)[0] > 2 * cell] = np.nan; surf = surf.reshape(GX.shape)
prof = dict(driver="GTiff", height=surf.shape[0], width=surf.shape[1], count=1, dtype="float32",
            crs="EPSG:2157", transform=from_origin(gx.min(), gy.max() + cell, cell, cell), nodata=np.nan, compress="deflate")
with rasterio.open("/tmp/ree_v4_surface.tif", "w", **prof) as dst: dst.write(np.flipud(surf).astype("float32"), 1)
try: shutil.copyfile("/tmp/ree_v4_surface.tif", os.path.join(OUT, "ree_v4_favourability_surface.tif"))
except Exception as e: print("tif copy note:", e)

# %% [markdown]
# ## 9. Figures

# %%
fig, ax = plt.subplots(2, 3, figsize=(17, 11)); ax = ax.ravel()
xl = (A["x"].min() - 1e4, A["x"].max() + 1e4); yl = (A["y"].min() - 1e4, A["y"].max() + 1e4)
def E(a, t):
    a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=10, fontweight="bold")
for a, col, t in [(ax[0], "REE_anom", "REE anomaly index (near-total)"), (ax[1], "F_LREE", "LREE-Th factor"),
                  (ax[2], "F_HREE", "HREE-Y factor"), (ax[3], "F_HFSE", "HFSE Zr-Nb factor"), (ax[4], "favourability", "Data-driven favourability")]:
    s = a.scatter(A["x"], A["y"], c=A[col], s=3, cmap="inferno", vmin=0, vmax=1); plt.colorbar(s, ax=a, shrink=0.55); E(a, t)
ax[5].scatter(A["x"], A["y"], c="0.85", s=2); ax[5].scatter(occ.geometry.x, occ.geometry.y, c="red", s=9, label=f"REE-affinity occ ({len(occ)})")
if len(tdf): ax[5].scatter(tdf["x"], tdf["y"], marker="*", s=120, facecolor="none", edgecolor="lime", linewidths=1.5, label="targets")
E(ax[5], "Occurrences + targets"); ax[5].legend(fontsize=7, loc="upper left")
plt.suptitle("REE prospectivity: anomaly, element factors, favourability, validation", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_maps.png"), dpi=140, bbox_inches="tight"); plt.close(); display(Image(filename=os.path.join(MAPS, "ree_v4_maps.png")))

fig, ax = plt.subplots(1, 2, figsize=(12, 8))
for a, col, t in [(ax[0], "REE_resid", "Weathering-corrected residual\n(primary-fertility candidate)"),
                  (ax[1], "LaYb_rank", "LREE/HREE = La/Yb (style discriminator)")]:
    if col in A.columns:
        s = a.scatter(A["x"], A["y"], c=A[col], s=3, cmap="inferno", vmin=0, vmax=1); plt.colorbar(s, ax=a, shrink=0.5)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=10, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_residual_fractionation.png"), dpi=140, bbox_inches="tight"); plt.close(); display(Image(filename=os.path.join(MAPS, "ree_v4_residual_fractionation.png")))

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
hh = pd.DataFrame({"area%": A["host"].value_counts(normalize=True) * 100, "anom%": A.loc[anom, "host"].value_counts(normalize=True) * 100})
hh["lift"] = (hh["anom%"] / hh["area%"]); hh = hh.fillna(0).sort_values("lift")
ax[0].barh(hh.index, hh["lift"], color=["tab:red" if v > 1 else "0.6" for v in hh["lift"]]); ax[0].axvline(1, color="k", ls="--")
ax[0].set_title("Host-lithology enrichment (data-driven)"); ax[0].set_xlabel("lift x"); ax[0].tick_params(labelsize=8)
fr = np.linspace(0.002, 1, 200); ax[1].plot(fr * 100, [(occ_rank <= f).mean() * 100 for f in fr], color="tab:purple"); ax[1].plot([0, 100], [0, 100], "k--", lw=.7)
ax[1].set_xlabel("% area (favourability-ranked)"); ax[1].set_ylabel("% occurrences captured"); ax[1].set_title("Prediction-Area (real occurrences)")
ax[2].hist(null * 100, bins=40, color="0.7"); ax[2].axvline(obs * 100, color="r", lw=2, label=f"obs {obs*100:.0f}% (p={pval:.3f})")
ax[2].set_xlabel("top-10% capture (random)"); ax[2].set_title("Permutation null"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_validation.png"), dpi=140, bbox_inches="tight"); plt.close(); display(Image(filename=os.path.join(MAPS, "ree_v4_validation.png")))

A[["x", "y", "region", "horizon", "host", "REE_anom", "F_LREE", "F_HREE", "F_HFSE", "favourability"]].to_csv(os.path.join(OUT, "ree_v4_scores.csv"), index=False)
print("Saved: ree_v4_scores.csv, ree_v4_targets.csv, ree_v4_favourability_surface.tif, maps/ree_v4_*.png")

# %% [markdown]
# ## GIS deliverables: scored points and targets as vector (GeoPackage)
# The continuous favourability is already exported as a GeoTIFF raster
# (ree_v4_favourability_surface.tif). Here we also export the scored sample points and the
# ranked targets as GeoPackage vectors, so the whole result opens directly in QGIS / ArcGIS.

# %%
try:
    _vp = gpd.GeoDataFrame(A[["x", "y", "REE_anom", "favourability", "F_LREE", "F_HREE", "F_HFSE", "host", "region"]].copy(),
                           geometry=gpd.points_from_xy(A["x"], A["y"]), crs="EPSG:2157")
    _vp.to_file(os.path.join(OUT, "ree_v4_points.gpkg"), driver="GPKG")
    if len(tdf):
        _vt = gpd.GeoDataFrame(tdf.copy(), geometry=gpd.points_from_xy(tdf["x"], tdf["y"]), crs="EPSG:2157")
        _vt.to_file(os.path.join(OUT, "ree_v4_targets.gpkg"), driver="GPKG")
    print("Saved vectors: ree_v4_points.gpkg, ree_v4_targets.gpkg")
except Exception as e:
    print("REE GIS vector export note:", e)

# %% [markdown]
# ## 9b. Integrated visualizations (data integration, multivariate structure, target scorecards)
# Extra figures that make the multidisciplinary *integration* and the per-target *reasoning*
# explicit. The Frank Arnott Award scores innovation in **data integration and visualisation**,
# so these turn the analysis into legible exploration products.

# %%
# (1) DATA-INTEGRATION / COVERAGE MAP, every dataset on one canvas
try:
    figC, axC = plt.subplots(figsize=(9, 10))
    try: bed.boundary.plot(ax=axC, color="0.85", lw=0.3)
    except Exception: pass
    try: gr.plot(ax=axC, color="#d9c9a3", alpha=0.55)
    except Exception: pass
    try: fa.plot(ax=axC, color="0.55", lw=0.4)
    except Exception: pass
    _Aa = A[A["horizon"] == "A"]; _As = A[A["horizon"] == "S"]
    axC.scatter(_Aa["x"], _Aa["y"], s=3, c="#5E6B3B", alpha=0.5, label=f"Soil A-horizon ({len(_Aa)})")
    axC.scatter(_As["x"], _As["y"], s=3, c="#3b6fb6", alpha=0.4, label=f"Soil S-horizon ({len(_As)})")
    try: axC.scatter(strm["x"], strm["y"], s=5, c="#1f9e89", alpha=0.7, label=f"Stream sediment ({len(strm)})")
    except Exception: pass
    axC.scatter(occ.geometry.x, occ.geometry.y, s=24, c="red", edgecolor="k", lw=0.3, label=f"REE-affinity occ ({len(occ)})", zorder=6)
    if len(tdf): axC.scatter(tdf["x"], tdf["y"], marker="*", s=190, facecolor="none", edgecolor="lime", lw=1.6, label=f"REE targets ({len(tdf)})", zorder=7)
    axC.set_aspect("equal"); axC.set_xticks([]); axC.set_yticks([])
    axC.set_title("Integrated datasets: soil + stream geochemistry, geophysics, geology, occurrences, targets", fontsize=10.5, fontweight="bold")
    axC.legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_data_coverage.png"), dpi=140, bbox_inches="tight"); plt.close()
    display(Image(filename=os.path.join(MAPS, "ree_v4_data_coverage.png")))
except Exception as e: print("coverage fig note:", e)

# %%
# (2) MULTIVARIATE CORRELATION HEATMAP, how anomaly, mineral-system indices and evidence relate
try:
    cc = [c for c in ["REE_anom", "REE_resid", "SOURCE", "PATHWAY", "TRAP", "favourability", "Al2O3",
                      "Fe2O3", "MnO", "P2O5", "pH", "Th_rad", "U_rad", "K_rad", "dist_fault_km",
                      "dist_granite_km", "elev", "LaYb"] if c in A.columns and A[c].notna().sum() > 1000]
    M = A[cc].corr(method="spearman")
    figH, axH = plt.subplots(figsize=(10.5, 9))
    im = axH.imshow(M.values, cmap="BrBG", vmin=-1, vmax=1)
    axH.set_xticks(range(len(cc))); axH.set_xticklabels(cc, rotation=90, fontsize=8)
    axH.set_yticks(range(len(cc))); axH.set_yticklabels(cc, fontsize=8)
    for i in range(len(cc)):
        for j in range(len(cc)):
            axH.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if abs(M.values[i, j]) > 0.55 else "0.2")
    plt.colorbar(im, ax=axH, shrink=0.7, label="Spearman correlation")
    axH.set_title("Multivariate structure: REE anomaly, mineral-system indices and evidence layers", fontsize=10, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_correlation.png"), dpi=140, bbox_inches="tight"); plt.close()
    display(Image(filename=os.path.join(MAPS, "ree_v4_correlation.png")))
except Exception as e: print("correlation fig note:", e)

# %%
# (3) INDIVIDUAL ELEMENT MAPS, the raw near-total geochemistry behind the index
try:
    emap = [e for e in ["La", "Ce", "Nd", "Sm", "Y", "Yb", "Zr", "Nb", "Th", "U"]
            if e in A.columns and A[e].notna().mean() > 0.4][:8]
    nc = 4; nr = int(np.ceil(len(emap) / nc))
    figE, axE = plt.subplots(nr, nc, figsize=(4 * nc, 4 * nr)); axE = np.atleast_1d(axE).ravel()
    for i, e in enumerate(emap):
        s = axE[i].scatter(A["x"], A["y"], c=A[e].rank(pct=True), s=2, cmap="inferno", vmin=0, vmax=1)
        axE[i].set_aspect("equal"); axE[i].set_xticks([]); axE[i].set_yticks([])
        axE[i].set_title(f"{e} (percentile)", fontsize=9, fontweight="bold")
    for j in range(len(emap), len(axE)): axE[j].axis("off")
    plt.suptitle("Near-total soil geochemistry: individual REE and HFSE elements", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_element_maps.png"), dpi=140, bbox_inches="tight"); plt.close()
    display(Image(filename=os.path.join(MAPS, "ree_v4_element_maps.png")))
except Exception as e: print("element maps fig note:", e)

# %%
# (4) CONCENTRATION-AREA FRACTAL PLOT, how the anomaly threshold is chosen
try:
    _v = A["REE_anom"].values; _v = _v[np.isfinite(_v)]
    _ts = np.unique(np.quantile(_v, np.linspace(0.5, 0.999, 60))); _ar = np.array([(_v >= t).mean() * 100 for t in _ts], float)
    figA2, axA2 = plt.subplots(figsize=(7, 6))
    axA2.semilogy(_ts, _ar, "o-", ms=3, color="#6E7A41")
    axA2.axvline(t_anom, color="#A85832", lw=1.6, ls="--", label=f"C-A threshold = {t_anom:.3f}")
    axA2.set_xlabel("REE anomaly index (threshold)"); axA2.set_ylabel("area above threshold (%)")
    axA2.set_title("Concentration-Area fractal threshold (Cheng et al. 1994)", fontsize=10, fontweight="bold")
    axA2.legend(); axA2.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_ca_plot.png"), dpi=140, bbox_inches="tight"); plt.close()
    display(Image(filename=os.path.join(MAPS, "ree_v4_ca_plot.png")))
except Exception as e: print("C-A fig note:", e)

# %%
# (5) PER-TARGET MINERAL-SYSTEM RADAR SCORECARDS, why each target ranks
try:
    if len(tdf):
        tr = tdf.head(6); lbl = ["SOURCE", "PATHWAY", "TRAP", "REE_anom", "favourability"]
        ang = np.linspace(0, 2 * np.pi, len(lbl), endpoint=False); ang = np.r_[ang, ang[:1]]
        nc = 3; nr = int(np.ceil(len(tr) / nc))
        figR, axR = plt.subplots(nr, nc, figsize=(4 * nc, 4 * nr), subplot_kw=dict(polar=True)); axR = np.atleast_1d(axR).ravel()
        for i, (_, t) in enumerate(tr.iterrows()):
            vals = [float(t.get(k, 0) or 0) for k in lbl]; vals = np.r_[vals, vals[:1]]
            axR[i].plot(ang, vals, color="#A85832", lw=2); axR[i].fill(ang, vals, color="#A85832", alpha=0.25)
            axR[i].set_xticks(ang[:-1]); axR[i].set_xticklabels(lbl, fontsize=7); axR[i].set_ylim(0, 1)
            axR[i].set_title(f"{t['target']} ({t['dom_host']}, {t['region']})", fontsize=8.5, fontweight="bold", pad=12)
        for j in range(len(tr), len(axR)): axR[j].axis("off")
        plt.suptitle("Per-target mineral-system scorecards (SOURCE / PATHWAY / TRAP / anomaly / favourability)", fontsize=11, fontweight="bold")
        plt.tight_layout(); plt.savefig(os.path.join(MAPS, "ree_v4_target_radar.png"), dpi=140, bbox_inches="tight"); plt.close()
        display(Image(filename=os.path.join(MAPS, "ree_v4_target_radar.png")))
except Exception as e: print("radar fig note:", e)

# %% [markdown]
# ## 10. Verdict, what the data says (no Mourne assumption)
# * **The REE soil anomaly is real and reproducible**, it repeats across independently sampled
#   A- and S-horizons and correlates strongly with *independent* airborne Th/U radiometrics (ρ≈+0.7).
# * **Its dominant control is weathering/regolith, not bedrock fertility.** The index is most
#   strongly associated with clay (Al₂O₃ ρ≈+0.68), Fe-Mn oxides and pH; weights-of-evidence ranks
#   clay as the #1 discriminator, while **distance-to-granite is irrelevant (ρ≈0)**. That is the
#   signature of an **ion-adsorption / secondary-scavenging** style (REE held on clays in the
#   weathering profile), *not* a primary A-type-granite halo.
# * **It is decoupled from primary mineralisation**, the raw favourability does not coincide
#   with the 336 granite/hydrothermal REE-affinity occurrences (top-10% captures ~4%, lift<1, p≈1).
# * **But a primary signal IS recoverable.** Weathering proxies explain only R²≈0.44 of the
#   anomaly; the **weathering-corrected residual** re-acquires a (weak) granite association and
#   captures **~15%** of the real occurrences in its top-10% area vs **~4%** for the raw anomaly
#   (~4× better), removing the regolith overprint unmasks a modest primary-fertility component.
#   For primary targeting, use the residual map; for ion-adsorption/regolith targeting, the raw anomaly.
# * **Independent stream-sediment medium agrees only weakly** (soil to stream 5 km-cell ρ≈+0.06):
#   the catchment-integrated stream signal and the in-situ soil signal are largely decoupled,
#   reinforcing that the soil anomaly is a *regolith* feature. Stream sediments capture ~6% of
#   occurrences in their top-10% (vs soil ~4%, residual ~15%) and are marginally granite-associated, 
#   a useful complementary reconnaissance layer (mirrors the GSI Tellus Mourne critical-metals workflow).
# * **A third independent medium, dissolved REE in stream/regional waters, is even more decoupled**
#   (soil-to-water 5 km-cell rho about -0.06 across ~12,700 water sites): dissolved REE is low and
#   pH-controlled, so its independence from the soil anomaly further confirms the soil signal is a
#   solid-phase (clay/regolith) feature, not a dissolved/hydromorphic one.
# * **Style discrimination:** La/Yb (median ≈10, P90≈19) is mostly LREE-dominated, isolating
#   localised high-La/Yb (carbonatite/alkaline-affinity) provinces from HREE (ion-adsorption) areas.
# * **Targets are multi-host and island-wide** (clay-rich shale/metasediment regolith), *not*
#   confined to Mourne. The LREE-Th / HREE-Y / HFSE factor maps separate styles for follow-up.
#
# * **Every target is explainable (mineral-systems decomposition).** Favourability is resolved into
#   SOURCE (radiometric Th/U + evolved-granite + P₂O₅), PATHWAY (faults) and TRAP (clay, Fe-Mn,
#   circumneutral pH, low relief); per-target scorecards + zoom-ins state *why* each anomaly exists.
#   The Irish REE targets are **TRAP-led** (clay-rich regolith), an ion-adsorption signature.
#
# **Operational reading:** this is a **regolith / ion-adsorption-clay REE favourability** map, 
# the play the data actually support, validated by cross-horizon reproducibility and independent
# radiometrics. A *primary* (carbonatite/alkaline/A-type) search needs lithogeochemistry on the
# intrusions, not regional soils. Caveats: soil REE is weathering-modulated (not a direct
# bedrock-grade proxy); no REE deposit exists for positive economic validation; WoE weights are
# learned from the anomaly (transparent, hence validated against *independent* occurrences/radiometrics).
# %% [markdown]
# ## 11. Significance, use, and honest limitations
# **Is this useful, given Ireland has no REE deposit?** Yes, but be precise about *why*.
# Per EuRare/BGS there has been **"no systematic evaluation of REE resources in the British
# Isles,"** yet REE *mineralisation* is documented (Mourne alkaline granite + alluvial
# fergusonite/gadolinite; the Beara-Allahies carbonatite in SW Ireland; regional vein/nodular
# monazite). Ireland is an **under-evaluated frontier**, not a barren terrane, the gap this study fills.
#
# **Why it matters**
# 1. **First island-wide, data-driven REE baseline** from the public Tellus survey, the
#    systematic screen that did not previously exist; a starting layer for explorers and policy.
# 2. **Critical-raw-materials relevance**, REE are an EU/Ireland strategic priority; a
#    reconnaissance favourability + *style* map says where, and *how*, to look.
# 3. **Method, not just a map**, a transferable, **label-free** workflow (anomaly to element
#    factors to data-driven evidence weighting to multi-medium/occurrence validation) for the
#    common case of *no training deposits*; reusable for any commodity or terrane.
# 4. **It changes the exploration recipe**, the data show the Irish soil REE signal is dominantly
#    **secondary/regolith (ion-adsorption-style)** with only a modest primary residual. Actionable:
#    for *primary* targets use the near-total residual + stream sediments + heavy-mineral concentrate
#    on the alkaline centres; for *ion-adsorption* potential, target the clay-rich regolith the raw
#    anomaly maps. It also de-risks by showing the signal is **not** a simple Mourne bedrock halo.
# 5. **Companion to the economically live LCT (lithium) model**, the Leinster Li belt has real,
#    actively explored occurrences, so the wider project has a grounded economic arm too.
#
# **Honest limitations, this is not "perfect" (nothing is):**
# * No REE deposit exists in Ireland to **no positive economic ground truth**; all validation is
#   indirect (cross-horizon reproducibility, independent radiometrics, affinity-proxy occurrences).
# * Favourability evidence-weights are learned from the anomaly (transparent, but not independent of it).
# * Soil REE is weathering-modulated; the primary residual is real but **modest** (~15% occ. capture).
#   Heavy-mineral-concentrate and catchment/drainage modelling would strengthen primary targeting.
# * Targets are first-pass screening, **not drill-ready**; field + petrographic follow-up required.
# * The continuous surface is simple IDW; anisotropic kriging would be a refinement.
print("\nDONE, open, multi-deposit-type, data-driven REE prospectivity (v4).")
