#!/usr/bin/env python3
"""4D context figure: Ireland's two critical-metal events through geologic time.
A simple, legible timeline that ties the two-granite story to TIME (the 4th dimension),
echoing the data-integration ethos of the Frank Arnott Award. Earth-tone palette, no dashes/arrows.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CLAY = "#A85832"; OLIVE = "#6E7A41"; OCHRE = "#C89A3C"; INK = "#33291C"; PAPER = "#F4EEE3"; SAGE = "#9DA886"
plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(13, 5.6)); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.set_xlim(500, -10); ax.set_ylim(0, 10)            # left = old, right = today
ax.axhline(3.0, color=INK, lw=2.2, zorder=2)         # the time arrow-free baseline

# era backdrop bands (subtle)
eras = [(485, 444, "Ordovician"), (444, 419, "Silurian"), (419, 359, "Devonian"),
        (359, 299, "Carboniferous"), (299, 252, "Permian"), (252, 66, "Mesozoic"), (66, 0, "Cenozoic")]
for i, (a, b, nm) in enumerate(eras):
    ax.axvspan(a, b, ymin=0.0, ymax=0.27, color=(SAGE if i % 2 == 0 else "#CDBfA0"), alpha=0.30, zorder=1)
    ax.text((a + b) / 2, 1.5, nm, ha="center", va="center", fontsize=7.5, color=INK, rotation=0)

# tick marks
for t in [500, 450, 400, 350, 300, 250, 200, 150, 100, 50, 0]:
    ax.plot([t, t], [2.85, 3.15], color=INK, lw=1)
    ax.text(t, 2.4, f"{t}", ha="center", va="top", fontsize=8, color=INK)
ax.text(250, 0.5, "Age (millions of years before present)", ha="center", fontsize=9, color=INK, style="italic")

def event(age, color, title, lines, side):
    yb = 3.0; yt = 6.4 if side == "up" else -99
    ax.plot([age, age], [yb, 6.2], color=color, lw=2, zorder=3)
    ax.scatter([age], [yb], s=130, color=color, edgecolor="white", lw=1.5, zorder=4)
    box = FancyBboxPatch((age - 62, 6.2), 124, 3.0, boxstyle="round,pad=0.02,rounding_size=8",
                         mutation_aspect=0.02, fc="white", ec=color, lw=2, zorder=4)
    ax.add_patch(box)
    ax.text(age, 8.85, title, ha="center", va="top", fontsize=11, fontweight="bold", color=color, zorder=5)
    for k, ln in enumerate(lines):
        ax.text(age, 8.25 - k * 0.62, ln, ha="center", va="top", fontsize=8.3, color=INK, zorder=5)

event(405, CLAY, "Lithium  (LCT pegmatites)",
      ["Caledonian orogeny, Iapetus closure", "S-type Leinster granite, about 405 Ma",
       "orogenic, fertile for Li, Sn, Ta", "modelled SUPERVISED (real occurrences)"], "up")
event(56, OLIVE, "Rare earths  (NYF system)",
      ["Palaeogene British-Irish igneous province", "Mourne about 56 Ma, Carlingford about 61 Ma",
       "anorogenic, within-plate (A-type affinity)", "modelled LABEL-FREE (no deposit yet)"], "up")

# Caledonian context bracket
ax.annotate("", xy=(400, 3.0), xytext=(470, 3.0))
ax.text(437, 3.45, "Caledonian terrane assembly", ha="center", fontsize=8, color="#6b5d4b", style="italic")

ax.set_title("Ireland's two critical-metal events in geologic time: one dataset, one workflow, two granite end-members",
             fontsize=12.5, fontweight="bold", color=INK, pad=12)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
plt.tight_layout()
plt.savefig("outputs/maps/metallogenic_timeline.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
print("saved outputs/maps/metallogenic_timeline.png")
