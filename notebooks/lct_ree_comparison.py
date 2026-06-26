#!/usr/bin/env python3
"""Side-by-side LCT (Li) vs REE end-member figure for the NGEA PDAC 2027 story.
Shows Ireland's two evolved-granite critical-metal systems:
  - Li / LCT pegmatites to S-type (Caledonian) granites  [Leinster]
  - REE / NYF to A-type (Palaeogene) granites   [Mourne-Carlingford]
"""
import warnings, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
B = "."; OUT = "outputs"; MAPS = OUT + "/maps"

bed = gpd.read_file(f"{B}/Ireland/Geology/GIS/IE_GSI_GSNI_Bedrock_Geology_1M_IE32_ITM_MS.shp").to_crs("EPSG:2157")
caled = bed[bed["UNITNAME"].str.contains("Siluro-Devonian granit|Ordovician granit", case=False, na=False)]
atype = bed[bed["UNITNAME"].str.contains("Palaeogene granit", case=False, na=False)]
mins = gpd.read_file(f"{B}/Ireland/Geology/GIS/IE_GSI_Mineral_Locations_IE26_ITM.shp").to_crs("EPSG:2157")
li = mins[mins["MIN_TYPE"] == "LI"]
lct = pd.read_csv(f"{OUT}/ranked_targets.csv")
ree_s = pd.read_csv(f"{OUT}/ree_v4_scores.csv")
ree_t = pd.read_csv(f"{OUT}/ree_v4_targets.csv")

fig, ax = plt.subplots(1, 2, figsize=(16, 9))
xl, yl = (415000, 775000), (510000, 965000)
def frame(a, t):
    bed.boundary.plot(ax=a, color="0.88", lw=0.2)
    a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(t, fontweight="bold", fontsize=11)

frame(ax[0], "Li: LCT pegmatites\nS-type / Caledonian / orogenic granites")
caled.plot(ax=ax[0], color="#3b6fb6", alpha=0.6)
ax[0].scatter([], [], color="#3b6fb6", marker="s", s=60, label="Caledonian (S-type) granite")
ax[0].scatter(li.geometry.x, li.geometry.y, c="orange", s=22, edgecolor="k", lw=0.3, label=f"GSI Li occurrences ({len(li)})", zorder=5)
ax[0].scatter(lct["Centroid_X_ITM"], lct["Centroid_Y_ITM"], marker="*", s=240, facecolor="none", edgecolor="lime", lw=2, label=f"LCT targets ({len(lct)})", zorder=6)
ax[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
ax[0].text(0.02, 0.02, "Leinster belt: spodumene pegmatites\n(Aclare, Moylisha, Knockeen)", transform=ax[0].transAxes, fontsize=7, style="italic")

frame(ax[1], "REE: NYF system\nA-type / Palaeogene / anorogenic granites")
sc = ax[1].scatter(ree_s["x"], ree_s["y"], c=ree_s["favourability"], s=2, cmap="inferno", vmin=0, vmax=1)
atype.plot(ax=ax[1], facecolor="none", edgecolor="cyan", lw=1.6)
ax[1].scatter([], [], facecolor="none", edgecolor="cyan", marker="s", s=60, label="A-type (Palaeogene) granite")
ax[1].scatter(ree_t["x"], ree_t["y"], marker="*", s=240, facecolor="none", edgecolor="lime", lw=2, label=f"REE targets ({len(ree_t)})", zorder=6)
plt.colorbar(sc, ax=ax[1], shrink=0.5, label="REE favourability")
ax[1].legend(loc="upper left", fontsize=8, framealpha=0.9)
ax[1].text(0.02, 0.02, "Mourne-Carlingford alkaline centres;\nsoil REE = clay/regolith-hosted (ion-adsorption)", transform=ax[1].transAxes, fontsize=7, style="italic")

plt.suptitle("Ireland's evolved granites host complementary critical-metal systems (Černý & Ercit, 2005)", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{MAPS}/lct_vs_ree_endmembers.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"saved {MAPS}/lct_vs_ree_endmembers.png | Caledonian polys={len(caled)} A-type polys={len(atype)} Li occ={len(li)} LCT targets={len(lct)} REE targets={len(ree_t)}")
