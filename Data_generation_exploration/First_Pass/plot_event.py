#!/usr/bin/env python
"""Visualise one simulated LHCb event from the CSVs written by dump_event.py.

Usage (plain python with matplotlib + numpy, e.g. the conda base env):

    python plot_event.py <results_dir> <event_index> <figures_dir>

Produces:
    event_display.png     x-z and y-z views: MC hits per detector, primary vertices
    vertex_zoom.png       interaction-region zoom: pile-up vertices + VP hits
    generator_summary.png per-collision multiplicity, momentum spectrum, eta dist
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

results = sys.argv[1] if len(sys.argv) > 1 else "results"
EVT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
figdir = sys.argv[3] if len(sys.argv) > 3 else "figures"
os.makedirs(figdir, exist_ok=True)

# ---- palette (validated: dataviz reference palette, light mode) -------------
SURFACE = "#fcfcfb"
TEXT1, TEXT2 = "#0b0b0b", "#52514e"
DET_COLOR = {  # categorical slots 1-4, fixed order
    "VP": "#2a78d6",
    "UT": "#008300",
    "FT": "#e87ba4",
    "Muon": "#eda100",
}
NEUTRAL = "#c9c8c2"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": TEXT1,
        "axes.edgecolor": TEXT2,
        "axes.labelcolor": TEXT1,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "axes.grid": True,
        "grid.color": "#e7e6e1",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "font.size": 11,
    }
)

PDG_NAME = {
    211: "pi+", -211: "pi-", 111: "pi0", 321: "K+", -321: "K-", 130: "K0L",
    310: "K0S", 2212: "p", -2212: "pbar", 2112: "n", -2112: "nbar",
    11: "e-", -11: "e+", 13: "mu-", -13: "mu+", 22: "gamma",
}


def read(name):
    path = os.path.join(results, name)
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)]
    return [r for r in rows if int(r["evt"]) == EVT]


parts = read("particles.csv")
verts = read("vertices.csv")
hits = read("hits.csv")
colls = read("collisions.csv")

pvs = [v for v in verts if v["is_primary"] == "1"]
print(
    "event %d: %d MCParticles, %d MCVertices (%d primary), %d hits, %d pp collisions"
    % (EVT, len(parts), len(verts), len(pvs), len(hits), len(colls))
)

# hits grouped by detector, as (entry, exit) segment arrays [mm]
seg = defaultdict(list)
for h in hits:
    seg[h["det"]].append(
        [
            (float(h["entry_z"]), float(h["entry_x"]), float(h["entry_y"])),
            (float(h["exit_z"]), float(h["exit_x"]), float(h["exit_y"])),
        ]
    )

# detector z-regions for context bands [mm] (approximate, for annotation only)
REGIONS = [
    ("VELO", -300, 800), ("RICH1", 1000, 2200), ("UT", 2200, 2800),
    ("Magnet", 3000, 7800), ("SciFi (FT)", 7800, 9500), ("RICH2", 9500, 11900),
    ("ECAL+HCAL", 12300, 15000), ("Muon", 15200, 18900),
]

# ------------------------------------------------------------- event display
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for ax, comp, label in [(axes[0], 1, "x [mm]"), (axes[1], 2, "y [mm]")]:
    for name, z0, z1 in REGIONS:
        ax.axvspan(z0, z1, color=NEUTRAL, alpha=0.25, lw=0)
        if comp == 1:  # label once, on the top panel
            ax.text(
                (z0 + z1) / 2, 0.97, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color=TEXT2,
            )
    ax.axhline(0, color=NEUTRAL, lw=0.8)  # beam line
    # trajectories: connect the hits of each particle in z-order (thin, under the marks)
    byparticle = defaultdict(list)
    for h in hits:
        if h["mc_key"]:
            byparticle[h["mc_key"]].append(
                (
                    (float(h["entry_z"]) + float(h["exit_z"])) / 2,
                    (float(h["entry_x"]) + float(h["exit_x"])) / 2,
                    (float(h["entry_y"]) + float(h["exit_y"])) / 2,
                )
            )
    for pts in byparticle.values():
        if len(pts) < 3:
            continue
        pts.sort()
        zs_t = [q[0] for q in pts]
        vs_t = [q[comp] for q in pts]
        ax.plot(zs_t, vs_t, color=TEXT2, lw=0.5, alpha=0.35, zorder=1.5)
    for det, segments in seg.items():
        arr = np.array(segments)  # (n, 2, 3): [point, (z, x, y)]
        zs = arr[:, :, 0].T
        vs = arr[:, :, comp].T
        ax.plot(zs, vs, color=DET_COLOR[det], lw=1.6, solid_capstyle="round")
    for v in pvs:
        val = float(v["x"]) if comp == 1 else float(v["y"])
        ax.plot(float(v["z"]), val, marker="*", ms=13, color=TEXT1, mec=SURFACE, mew=0.5, ls="none")
    ax.set_ylabel(label)
axes[1].set_xlabel("z [mm]  (beam axis; collisions happen near z = 0)")
handles = [plt.Line2D([], [], color=c, lw=2.5, label=d) for d, c in DET_COLOR.items() if seg.get(d)]
handles.append(plt.Line2D([], [], marker="*", ms=12, color=TEXT1, ls="none", label="primary vertex"))
axes[0].legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=9)
nch = sum(1 for p in parts if p["charge3"] not in ("", "0"))
fig.suptitle(
    "One simulated LHCb bunch crossing — event %d: %d pp collisions, %d MC particles "
    "(%d charged), %d tracker hits" % (EVT, len(colls), len(parts), nch, len(hits)),
    fontsize=12,
)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "event_display.png"), dpi=160)
plt.close(fig)

# ------------------------------------------------------------- vertex zoom
fig, ax = plt.subplots(figsize=(10, 5.5))
sec = [v for v in verts if v["is_primary"] != "1"]
if sec:
    ax.plot(
        [float(v["z"]) for v in sec], [float(v["x"]) for v in sec],
        ls="none", marker="o", ms=3.5, color=NEUTRAL, mec="none", label="secondary vertex",
    )
vp = np.array(seg["VP"]) if seg.get("VP") else None
if vp is not None:
    ax.plot(vp[:, :, 0].T, vp[:, :, 1].T, color=DET_COLOR["VP"], lw=1.2)
    ax.plot([], [], color=DET_COLOR["VP"], lw=2, label="VP (VELO pixel) hits")
if pvs:
    ax.plot(
        [float(v["z"]) for v in pvs], [float(v["x"]) for v in pvs],
        ls="none", marker="*", ms=14, color=TEXT1, mec=SURFACE, mew=0.5,
        label="primary vertices (%d)" % len(pvs),
    )
ax.set_xlim(-400, 900)
ax.set_ylim(-60, 60)  # beam-pipe scale; far-off secondary vertices are clipped
ax.set_xlabel("z [mm]")
ax.set_ylabel("x [mm]")
ax.legend(loc="best", framealpha=0.9, fontsize=9)
ax.set_title("Interaction region: pile-up vertices along the beam line (event %d)" % EVT)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "vertex_zoom.png"), dpi=160)
plt.close(fig)

# --------------------------------------------------------- generator summary
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
BLUE = "#2a78d6"

# (a) charged multiplicity per pp collision (via pv_key grouping)
bypv = defaultdict(int)
for p in parts:
    if p["charge3"] not in ("", "0") and p["pv_key"] != "":
        bypv[p["pv_key"]] += 1
counts = sorted(bypv.values(), reverse=True)
axes[0].bar(range(1, len(counts) + 1), counts, color=BLUE, width=0.7)
axes[0].set_xlabel("pp collision (sorted)")
axes[0].set_ylabel("charged MC particles")
axes[0].set_title("Multiplicity per collision", fontsize=11)

# (b) momentum spectrum of charged particles
pmag = [
    np.sqrt(float(p["px"]) ** 2 + float(p["py"]) ** 2 + float(p["pz"]) ** 2) / 1000.0
    for p in parts
    if p["charge3"] not in ("", "0")
]
if pmag:
    bins = np.logspace(np.log10(0.01), np.log10(max(max(pmag), 10)), 40)
    axes[1].hist(pmag, bins=bins, color=BLUE)
    axes[1].set_xscale("log")
axes[1].set_xlabel("|p| [GeV]")
axes[1].set_ylabel("charged MC particles")
axes[1].set_title("Momentum spectrum", fontsize=11)

# (c) pseudorapidity, LHCb acceptance shaded
eta = []
for p in parts:
    if p["charge3"] in ("", "0"):
        continue
    px, py, pz = float(p["px"]), float(p["py"]), float(p["pz"])
    pt = np.hypot(px, py)
    if pt > 0:
        eta.append(np.arcsinh(pz / pt))
if eta:
    axes[2].hist(eta, bins=np.linspace(-8, 8, 49), color=BLUE)
axes[2].axvspan(2, 5, color=NEUTRAL, alpha=0.35, lw=0)
axes[2].text(3.5, 0.95, "LHCb\nacceptance", transform=axes[2].get_xaxis_transform(),
             ha="center", va="top", fontsize=8.5, color=TEXT2)
axes[2].set_xlabel(r"pseudorapidity $\eta$")
axes[2].set_ylabel("charged MC particles")
axes[2].set_title("Where the particles go", fontsize=11)

fig.suptitle("Generator view of event %d (%d pp collisions in the crossing)" % (EVT, len(colls)), fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "generator_summary.png"), dpi=160)
plt.close(fig)

# leading particles printout
def pname(pid):
    return PDG_NAME.get(int(pid), "pdg %s" % pid)

charged = [p for p in parts if p["charge3"] not in ("", "0")]
charged.sort(key=lambda p: -(float(p["px"]) ** 2 + float(p["py"]) ** 2 + float(p["pz"]) ** 2))
print("\nleading charged particles:")
for p in charged[:10]:
    mom = np.sqrt(float(p["px"]) ** 2 + float(p["py"]) ** 2 + float(p["pz"]) ** 2) / 1000.0
    nh = sum(1 for h in hits if h["mc_key"] == p["key"])
    print("  %-8s |p| = %8.1f GeV   hits: %d" % (pname(p["pid"]), mom, nh))
print("\nfigures written to", figdir)
