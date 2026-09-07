#!/usr/bin/env python
"""C0.5 - the detector schematic, and where the four leg classes live on it.

Every z in this figure is measured, not drawn from memory:

  * the sensor planes are the z clusters of the harvested MCHit states
    (`Official_xdigi/results/states.npz`, the file the v2 training set is built
    from). Cutting that list of z values wherever a gap exceeds 15 mm gives 26
    clusters: the VELO modules, four UT planes and twelve SciFi layers in three
    stations. The cluster midpoints are the plane positions used here, and they
    are written to `results/detector_planes.csv` so the numbers can be read
    without the figure.
  * each detector's transverse extent is the largest |x| of the hits assigned
    to it, so the boxes are the acceptance the sample actually populates.
  * the magnet's shading is where |B| along the beam line exceeds 5 per cent of
    its peak, read from the v8r1 map itself, and the peak is marked at the z
    where |B| is largest. The faint profile behind the tracks is that same
    |B|(0, 0, z), on the right-hand axis.

Over that, about twenty real particles: their own hit states joined, and the
RK6 reference path across the magnet drawn through them, coloured by truth
momentum. Four annotated arrows mark the leg classes the training set is built
from - A vertex fetch, B cross-magnet (both directions), C plane to plane,
D downstream track to vertex.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python plot_schematic.py

Outputs:
    figures/lhcb_legs_schematic.png
    results/detector_planes.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import csv
import json

import numpy as np

import use_shared                                            # noqa: F401
from _shared.prepare import P_BANDS, _particle_key, magnet_leg_rows, p_band_index
from _shared.reference import make_field, rk6_dense_rows

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
STATES_NPZ = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "Data_generation_exploration",
    "Official_xdigi", "results", "states.npz")

FIELD = "up"
Z_LO, Z_HI = -300.0, 9600.0
MAGNET_SHADE_FRACTION = 0.20     # shade where |B| on the beam line exceeds this
N_TRACKS_PER_BAND = 4
PLANE_GAP_MM = 15.0          # a gap wider than this separates two planes


def plane_clusters(z, det, det_names, gap=PLANE_GAP_MM):
    """The sensor planes, as the z clusters of the harvested states."""
    order = np.argsort(z)
    zs, ds = z[order], det[order]
    cuts = np.flatnonzero(np.diff(zs) > gap)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(zs) - 1]
    out = []
    for a, b in zip(starts, ends):
        n = b - a + 1
        if n < 200:                       # a handful of strays, not a plane
            continue
        sub = ds[a:b + 1]
        which = det_names[np.bincount(sub).argmax()]
        out.append({"detector": which, "z_lo": float(zs[a]),
                    "z_hi": float(zs[b]), "z_mid": float(0.5 * (zs[a] + zs[b])),
                    "n_states": int(n)})
    return out


def main():
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable

    st = np.load(STATES_NPZ, allow_pickle=True)
    det_names = np.array([str(s) for s in st["det_names"]])
    planes = plane_clusters(st["z"], st["det"].astype(np.int64), det_names)
    with open(os.path.join(RESULTS, "detector_planes.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["detector", "z_lo", "z_hi", "z_mid",
                                          "n_states"])
        w.writeheader()
        w.writerows(planes)
    print("%d planes: %s" % (len(planes),
                             {d: sum(1 for p in planes if p["detector"] == d)
                              for d in det_names}))

    # the detector boxes: z extent from the planes, |x| extent from the hits
    boxes = {}
    for d in det_names:
        m = det_names[st["det"].astype(np.int64)] == d
        boxes[str(d)] = {"z0": float(st["z"][m].min()),
                         "z1": float(st["z"][m].max()),
                         "half_x": float(np.quantile(np.abs(st["x"][m]), 0.999))}

    # the field along the beam line, from the map
    field = make_field(FIELD)
    zs = np.linspace(Z_LO, Z_HI, 2000)
    Bx, By, Bz = field(np.zeros_like(zs), np.zeros_like(zs), zs)
    Bmag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    peak_z = float(zs[np.argmax(Bmag)])
    peak_B = float(Bmag.max())
    inside = Bmag > MAGNET_SHADE_FRACTION * peak_B
    mag_z0, mag_z1 = float(zs[inside].min()), float(zs[inside].max())
    print("|B| peaks at z = %.0f mm, %.3f T; >%.0f%% of peak over %.0f - %.0f mm"
          % (peak_z, peak_B, 100 * MAGNET_SHADE_FRACTION, mag_z0, mag_z1))

    # ---- the tracks ---------------------------------------------------------
    sel = magnet_leg_rows(verbose=False)
    band = p_band_index(sel["P"])
    rng = np.random.default_rng(7)
    pick = []
    for b in range(len(P_BANDS)):
        pool = np.flatnonzero((band == b) & (sel["DIRECTION"] > 0))
        rng.shuffle(pool)
        pick.extend(pool[:N_TRACKS_PER_BAND].tolist())
    pick = np.array(pick)
    skey = st["evt"].astype(np.int64) * 10_000_000 + st["mc_key"]
    pkey = _particle_key(sel["EVT"], sel["MCKEY"])
    Zg, Sg, V = rk6_dense_rows(sel["S0"][pick], sel["z0"][pick],
                               sel["z1"][pick], sample_mm=20.0, step=0.5,
                               field=field)

    # ---- the figure ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(Z_LO, Z_HI)
    ax.set_ylim(-3700, 3700)

    ax.axvspan(mag_z0, mag_z1, color="#f2c14e", alpha=0.16, lw=0, zorder=0)
    ax.text(0.5 * (mag_z0 + mag_z1), 3400,
            "MAGNET  (|B| > %.0f %% of its peak)" % (100 * MAGNET_SHADE_FRACTION),
            ha="center", fontsize=11, weight="bold", color="#8a6d1f")
    ax.axvline(peak_z, color="#8a6d1f", ls="--", lw=1.2, zorder=1)
    ax.text(peak_z + 60, -3450, "|B| peak  z = %.2f m, %.2f T"
            % (peak_z / 1000.0, peak_B), fontsize=8, color="#8a6d1f")

    axB = ax.twinx()
    axB.fill_between(zs, 0, Bmag, color="#8a6d1f", alpha=0.12, lw=0, zorder=0)
    axB.plot(zs, Bmag, color="#8a6d1f", lw=1.0, alpha=0.45, zorder=1)
    axB.set_ylim(0, peak_B * 4.2)
    axB.set_ylabel("|B| on the beam line  [T]   (faint)", color="#8a6d1f")
    axB.tick_params(axis="y", colors="#8a6d1f", labelsize=8)

    labels = {"VP": "VELO", "UT": "UT", "FT": "SciFi T1-T3"}
    colours = {"VP": "#4c72b0", "UT": "#55a868", "FT": "#c44e52"}
    for d, b in boxes.items():
        ax.add_patch(plt.Rectangle((b["z0"], -b["half_x"]),
                                   b["z1"] - b["z0"], 2 * b["half_x"],
                                   fill=False, ec=colours[d], lw=1.6,
                                   zorder=3))
        ax.text(0.5 * (b["z0"] + b["z1"]), 3400, labels[d],
                ha="center", fontsize=10, color=colours[d], weight="bold")
    for p in planes:
        ax.plot([p["z_mid"], p["z_mid"]],
                [-boxes[p["detector"]]["half_x"],
                 boxes[p["detector"]]["half_x"]],
                color=colours[p["detector"]], lw=0.6, alpha=0.55, zorder=2)
    # name the three SciFi stations from the plane list
    ft = [p["z_mid"] for p in planes if p["detector"] == "FT"]
    for i, grp in enumerate(np.array_split(np.sort(ft), 3)):
        ax.text(float(grp.mean()), -3450, "T%d" % (i + 1),
                ha="center", fontsize=9, color=colours["FT"])

    norm = LogNorm(vmin=1.0, vmax=200.0)
    cmap = plt.get_cmap("viridis")
    for j, i in enumerate(pick):
        c = cmap(norm(sel["P"][i]))
        m = skey == pkey[i]
        zz, xx = st["z"][m], st["x"][m]
        o = np.argsort(zz)
        ax.plot(zz[o], xx[o], ".", ms=4.5, color=c, zorder=5)
        ax.plot(zz[o], xx[o], "-", lw=0.7, alpha=0.55, color=c, zorder=4)
        ax.plot(Zg[j][V[j]], Sg[j][V[j], 0], "-", lw=1.9, color=c, zorder=6)

    # ---- the four leg classes ----------------------------------------------
    ut_last = max(p["z_mid"] for p in planes if p["detector"] == "UT")
    ft_first = min(p["z_mid"] for p in planes if p["detector"] == "FT")
    vp_first = min(p["z_mid"] for p in planes if p["detector"] == "VP")
    arrows = [
        ("A  vertex fetch", vp_first, 0.0, 2600, "#7f7f7f"),
        ("B  cross-magnet  (both directions)", ut_last, ft_first, 2050,
         "#000000"),
        ("C  plane to plane", sorted(ft)[1], sorted(ft)[2], -2350, "#7f7f7f"),
        ("D  downstream track to vertex", ft_first, 0.0, -2900, "#7f7f7f"),
    ]
    for name, za, zb, y, col in arrows:
        two = name.startswith("B")
        ax.annotate("", xy=(zb, y), xytext=(za, y),
                    arrowprops=dict(arrowstyle="<->" if two else "->",
                                    color=col, lw=2.0 if two else 1.4,
                                    shrinkA=0, shrinkB=0), zorder=7)
        lz = 0.5 * (za + zb) if abs(zb - za) > 400 else max(za, zb) + 250
        ha = "center" if abs(zb - za) > 400 else "left"
        ax.text(lz, y + 130, name, ha=ha, fontsize=9.5,
                color=col, weight="bold" if two else "normal")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axB, pad=0.06, fraction=0.03)
    cb.set_label("truth momentum  p  [GeV]")

    ax.set_xlabel("z  [mm]")
    ax.set_ylabel("x  [mm]")
    ax.set_title("LHCb tracking side view, the four leg classes, and %d real "
                 "particles.\nPoints and thin lines: the particle's own MCHit "
                 "states.  Thick lines: the RK6 reference across the magnet "
                 "(v8r1.%s).  Plane positions from the harvested hit z clusters."
                 % (len(pick), FIELD), fontsize=11)
    ax.grid(alpha=0.22, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "lhcb_legs_schematic.png"), dpi=150)
    plt.close(fig)
    print("wrote figures/lhcb_legs_schematic.png")

    with open(os.path.join(RESULTS, "schematic_meta.json"), "w") as f:
        json.dump({"planes": planes, "boxes": boxes,
                   "magnet": {"peak_z_mm": peak_z, "peak_B_T": peak_B,
                              "shaded_from_mm": mag_z0, "shaded_to_mm": mag_z1,
                              "criterion": "|B|(0,0,z) > 5% of its peak"},
                   "field": FIELD,
                   "n_tracks": int(len(pick)),
                   "plane_gap_mm": PLANE_GAP_MM,
                   "source": os.path.relpath(STATES_NPZ, HERE)}, f, indent=1)


if __name__ == "__main__":
    main()
