# Critical-metal prospectivity of Ireland's evolved granites from open Tellus data
### A validated lithium (LCT) model and a label-free rare-earth (REE) assessment
**NGEA · PDAC 2027, Frank Arnott Next Generation Explorers Award**
Team: Joefawwaz · Muhammadrahalditaher · Extivonus Fransiskus · Jesslyn Jane · Mentors: Cendidana · Reza-Alfurqan · Fahmihakim-ugm

---

## Abstract
We integrate the public **Tellus** geoscience programme (≈27,000 soil + stream geochemistry sites, airborne magnetics/radiometrics/EM, 1:1M bedrock & structure) into a single reproducible pipeline that targets two critical-metal systems hosted by Ireland's **evolved granites**. The two commodities are end-members of one petrogenetic story (Černý & Ercit, 2005): **lithium** in **LCT pegmatites** from **S-type, Caledonian, orogenic** granites (Leinster), and **rare-earth elements** in **NYF systems** from **A-type, Palaeogene, anorogenic** granites (Mourne-Carlingford). Crucially, the two demand **different ML paradigms**, which is the methodological core of the work: Li has real occurrences and is modelled **supervised with spatial cross-validation and occurrence validation**; REE has **no Irish deposit**, so it is modelled **label-free** (unsupervised anomaly detection + data-driven weights-of-evidence favourability), validated by reproducibility, independent geophysics and real REE-affinity occurrences. The Li model recovers 11/13 known occurrences (spatial-CV AUC ≈0.86) and flags Leinster and Newry-Mourne targets; the REE assessment shows the Irish soil REE signal is dominantly **secondary, clay/regolith-hosted (ion-adsorption style)** with a modest, recoverable primary-fertility residual, the **first systematic, island-wide REE screen** of a terrane EuRare/BGS describe as never having had one.

---

## 1. Rationale
The EU and Ireland classify REE and Li as strategic raw materials, yet Ireland's potential is unevenly assessed: the Leinster Li belt is under active exploration (Ganfeng/Blackstairs; Global Battery Metals), while for REE, EuRare/BGS state there has been **"no systematic evaluation of REE resources in the British Isles."** The Tellus open dataset makes a first, data-driven, national screen possible. Our contribution is not a claimed discovery but a **transferable workflow** demonstrated on a real frontier, plus the geological insight it yields.

## 2. Data & workflow
- **Geochemistry:** near-total (XRF) REE from four soil surveys (Shallow-A + Deeper-S horizons, ROI + NI; ~25,000 sites) and 7,719 stream sediments; plus pH, LOI and XRF major oxides.
- **Geophysics:** Tellus airborne magnetics, radiometrics (K/Th/U), EM, re-gridded from the raw `.XYZ` to single-band scientific rasters (the supplied colour `.tif`s were RGB visualisations, unusable for ML).
- **Geology:** GSI/GSNI 1:1M bedrock & faults, GSI Mineral Localities, DEM.
- **One pipeline, two paradigms**, shared loaders/CRS/CLR preprocessing, then a *supervised* path (Li) and a *label-free* path (REE).
- *Data-quality control:* at co-located NI sites, near-total XRF reports **Zr ×90, Nb ×23, Yb ×4.6** more than aqua regia, so all REE work uses near-total (the partial leach misses zircon/monazite-hosted REE).

## 3. Lithium (LCT), validated supervised model
- Geology-driven labels from the **real Caledonian granite margin** (GSI 1:1M polygons; Palaeogene A-type excluded) + **real GSI Li occurrence anchors** (Republic of Ireland; no NI occurrence data available), Random Forest, **25-km spatial-block cross-validation (5-fold)**. *(Result figures below are refreshed after the next run.)*
- **Spatial-CV AUC ≈ 0.86** (Model A full / B no-Li); ρ(A,B) = **0.995** (Li is *not* the sole driver, multivariate signature is robust).
- **11/13 known occurrences recovered**; SHAP top features are real geophysics + bedrock + geochemistry (EM resistivity, bedrock-granite, K/Th radiometrics, LREE/HREE).
- **11 ranked targets**, led by the Newry-Mourne and Leinster granite margins (validating Aclare/Moylisha/Knockeen). *Fig: `outputs/maps/target_map_v2.png`, `outputs/maps/shap_model_A.png`.*

## 4. Rare earths (REE), label-free frontier assessment (`ree_anomaly_v4.ipynb`)
- **Unsupervised** multi-element enrichment index (per-survey ranks; near-total), CLR + PCA **factor maps** (LREE-Th / HREE-Y / HFSE), Concentration-Area fractal thresholds, and a **data-driven weights-of-evidence favourability** (no assumed host). *Fig: `outputs/maps/ree_v4_maps.png`.*
- **The anomaly is real**: A to S-horizon reproducibility **ρ = +0.88**; independent airborne **Th_rad +0.73**.
- **Its control is weathering/regolith, not bedrock**: clay Al₂O₃ **+0.68** (top weight), Fe-Mn oxides, pH; **granite proximity ≈ 0**. UCC-normalised patterns are modest (~1.1-1.4× crust) and **HREE-leaning ((La/Yb)_UCC ≈ 0.87)** to **ion-adsorption-clay affinity**. *Figs: `ree_v4_residual_fractionation.png`, `ree_v4_ucc_patterns.png`.*
- **A primary signal is recoverable**: weathering explains only R²≈0.44; the **weathering-corrected residual** captures **~15%** of real REE-affinity occurrences in its top-10% area vs **~4%** for the raw anomaly (~4×). Stream sediments are decoupled from soils (ρ≈+0.06), confirming an in-situ regolith signal. *Figs: `ree_v4_streamsed.png`, `ree_v4_validation.png`.*
- **Interpretable mineral-systems favourability**: prospectivity is resolved into **SOURCE** (radiometric Th/U + evolved-granite + P₂O₅), **PATHWAY** (faults) and **TRAP** (clay, Fe-Mn, circumneutral pH, low relief), so every target is *explained*. **17 ranked targets** (multi-host, ROI + NI; clay-rich metasediment/shale regolith) come with per-target SOURCE/PATHWAY/TRAP scorecards and zoom-ins; a continuous GeoTIFF surface is provided. *Figs: `ree_v4_mineral_systems.png`, `ree_v4_target_zooms.png`.*

## 5. The unifying story, two granite end-members
*Fig: `outputs/maps/lct_vs_ree_endmembers.png`.* Li targets sit on the **S-type Caledonian** Leinster granites (orogenic); the REE/NYF play belongs to the **A-type Palaeogene** Mourne-Carlingford centres (anorogenic). One dataset, one workflow, two complementary critical-metal systems, and two ML paradigms (supervised vs label-free) chosen by what ground truth exists.

## 6. Innovation & significance
1. **First systematic, island-wide REE screen** of an under-evaluated frontier, from open data.
2. A **transferable label-free workflow** for the common no-deposit case, reusable for any commodity/terrane.
3. **Actionable geology**: REE in Irish soils is largely ion-adsorption/regolith, not a Mourne bedrock halo, so primary search should use near-total residual + stream sediment + heavy-mineral concentrate on the alkaline centres; ion-adsorption search should target clay-rich regolith.
4. **Honest, multi-line validation** instead of a single (often circular) score, including catching and fixing our own early mistakes (an invalid AUC, RGB geophysics, an aqua-regia digestion bias).

## 7. Honest limitations & next steps
No Irish REE deposit exists, so there is **no positive economic ground truth**; REE validation is indirect (reproducibility, independent radiometrics, affinity-proxy occurrences). Soil REE is weathering-modulated; the primary residual is real but modest. Targets are **first-pass screening, not drill-ready**. Next: heavy-mineral-concentrate and drainage/catchment modelling, anisotropic kriging, and field/petrographic follow-up on the leading targets.

---

### Figure list
1. `lct_vs_ree_endmembers.png`, two granite end-members (headline).
2. `target_map_v2.png`, `shap_model_A.png`, LCT targets and feature importance.
3. `ree_v4_maps.png`, REE anomaly, LREE/HREE/HFSE factors, favourability, occurrences+targets.
3b. `ree_v4_mineral_systems.png`, SOURCE / PATHWAY / TRAP components combined into favourability.
3c. `ree_v4_target_zooms.png`, per-target zoom-ins with mineral-system scores.
4. `ree_v4_ucc_patterns.png`, UCC-normalised REE signatures (style discrimination).
5. `ree_v4_residual_fractionation.png`, weathering-corrected residual + La/Yb.
6. `ree_v4_streamsed.png`, independent stream-sediment cross-check.
7. `ree_v4_validation.png`, host enrichment, Prediction-Area, permutation null.

### Key references
Černý & Ercit (2005) *Can. Mineral.* 43; EuRare/BGS, REE in the UK & Ireland; Cheng et al. (1994) C-A fractal; Filzmoser, Garrett & Reimann (2005) robust compositional outliers; Carranza (2008) Prediction-Area validation; Su et al. (2017) REE weathering fractionation; GSI Tellus critical-metals (Mournes); Liu et al. (2025) isolation-forest MPM.
