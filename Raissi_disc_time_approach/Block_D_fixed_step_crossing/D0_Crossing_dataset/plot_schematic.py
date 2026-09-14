#!/usr/bin/env python
"""D0 - the detector schematic of Block D: where the data sit, what the
crossing is, and how each step count cuts it.

The side view of `C0_Magnet_tracks_dataset/plot_schematic.py` (every z measured
from the harvested hit states and the field map, nothing drawn from memory),
with the Block D content on top of it:

  * the two frozen planes, z0 = 2648.2 mm (last UT plane) and z1 = 7826.0 mm
    (first SciFi plane), drawn through the detector as the crossing;
  * twenty real particles of the D0 set, four per momentum band: their hit
    states (points), their real last-UT and first-SciFi states (rings) - the
    start state and the data ground truth - and the RK6 truth between them
    (thick line), coloured by momentum;
  * below the tracks, the five step grids N = 1, 4, 16, 64, 128 as rows of
    ticks between z0 and z1, with |B| on the beam line behind them.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python plot_schematic.py

Outputs:
    figures/block_d_schematic.png
    results/schematic_meta.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import json
import sys

import numpy as np

import use_shared                                            # noqa: F401
from _shared.prepare import P_BANDS
from _shared.reference import FROZEN_LEG, make_field

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
C0 = os.path.join(use_shared.SHARED_ROOT, "Block_C_step_size_and_stages", "C0_Magnet_tracks_dataset")
sys.path.insert(0, C0)
from plot_schematic import (MAGNET_SHADE_FRACTION, STATES_NPZ, Z_HI, Z_LO,   # noqa: E402
                            plane_clusters)

FIELD = "up"
N_VALUES = (1, 4, 16, 64, 128)
N_TRACKS_PER_BAND = 4
Z0, Z1 = float(FROZEN_LEG["z0"]), float(FROZEN_LEG["z1"])
L = Z1 - Z0


def main():
    os.makedirs(FIGURES, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm

    D = np.load(os.path.join(RESULTS, "crossing_particles.npz"), allow_pickle=False)
    st = np.load(STATES_NPZ, allow_pickle=True)
    det_names = np.array([str(s) for s in st["det_names"]])
    planes = plane_clusters(st["z"], st["det"].astype(np.int64), det_names)
    boxes = {}
    for d in det_names:
        m = det_names[st["det"].astype(np.int64)] == d
        boxes[str(d)] = {"z0": float(st["z"][m].min()), "z1": float(st["z"][m].max()),
                         "half_x": float(np.quantile(np.abs(st["x"][m]), 0.999))}
    field = make_field(FIELD)
    zs = np.linspace(Z_LO, Z_HI, 2000)
    Bx, By, Bz = field(np.zeros_like(zs), np.zeros_like(zs), zs)
    Bmag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    peak_z, peak_B = float(zs[np.argmax(Bmag)]), float(Bmag.max())
    inside = Bmag > MAGNET_SHADE_FRACTION * peak_B
    mag_z0, mag_z1 = float(zs[inside].min()), float(zs[inside].max())

    # twenty test particles, four per band
    P, band = D["test_P"], D["test_PBAND"]
    rng = np.random.default_rng(7)
    pick = []
    for b in range(len(P_BANDS)):
        pool = np.flatnonzero(band == b)
        rng.shuffle(pool)
        pick.extend(pool[:N_TRACKS_PER_BAND].tolist())
    pick = np.array(pick)
    skey = st["evt"].astype(np.int64) * 10_000_000 + st["mc_key"]
    pkey = D["test_EVT"].astype(np.int64) * 10_000_000 + D["test_MCKEY"].astype(np.int64)

    fig, (ax, axg) = plt.subplots(2, 1, figsize=(15, 9.5), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1.35]})
    ax.set_xlim(Z_LO, Z_HI)
    ax.set_ylim(-3700, 3700)
    ax.axvspan(mag_z0, mag_z1, color="#f2c14e", alpha=0.16, lw=0, zorder=0)
    ax.text(0.5 * (mag_z0 + mag_z1), 3400, "MAGNET  (|B| > %.0f %% of its peak)"
            % (100 * MAGNET_SHADE_FRACTION), ha="center", fontsize=11, weight="bold",
            color="#8a6d1f")
    axB = ax.twinx()
    axB.fill_between(zs, 0, Bmag, color="#8a6d1f", alpha=0.12, lw=0, zorder=0)
    axB.plot(zs, Bmag, color="#8a6d1f", lw=1.0, alpha=0.45, zorder=1)
    axB.set_ylim(0, peak_B * 4.2)
    axB.set_ylabel("|B| on the beam line  [T]   (faint)", color="#8a6d1f")
    axB.tick_params(axis="y", colors="#8a6d1f", labelsize=8)
    labels = {"VP": "VELO", "UT": "UT", "FT": "SciFi T1-T3"}
    colours = {"VP": "#4c72b0", "UT": "#55a868", "FT": "#c44e52"}
    for d, b in boxes.items():
        ax.add_patch(plt.Rectangle((b["z0"], -b["half_x"]), b["z1"] - b["z0"],
                                   2 * b["half_x"], fill=False, ec=colours[d], lw=1.6, zorder=3))
        ax.text(0.5 * (b["z0"] + b["z1"]), 3400, labels[d], ha="center", fontsize=10,
                color=colours[d], weight="bold")
    for p in planes:
        ax.plot([p["z_mid"]] * 2, [-boxes[p["detector"]]["half_x"], boxes[p["detector"]]["half_x"]],
                color=colours[p["detector"]], lw=0.6, alpha=0.55, zorder=2)

    # the crossing
    for z, name in ((Z0, "z0 = %.1f mm\nlast UT plane\n(the start state)" % Z0),
                    (Z1, "z1 = %.1f mm\nfirst SciFi plane\n(the data truth)" % Z1)):
        ax.axvline(z, color="#111111", lw=1.8, zorder=4)
    ax.text(Z0 - 80, -2450, "z0 = %.1f mm\nlast UT plane\nthe start state" % Z0,
            ha="right", fontsize=9, color="#111111")
    ax.text(Z1 + 80, -2450, "z1 = %.1f mm\nfirst SciFi plane\nthe data truth" % Z1,
            ha="left", fontsize=9, color="#111111")
    ax.annotate("", xy=(Z1, 2350), xytext=(Z0, 2350),
                arrowprops=dict(arrowstyle="<->", color="#111111", lw=2.0, shrinkA=0, shrinkB=0),
                zorder=7)
    ax.text(0.5 * (Z0 + Z1), 2480, "the crossing, L = %.1f mm: one network per step, "
            "N steps chained" % L, ha="center", fontsize=10, weight="bold")

    norm = LogNorm(vmin=1.0, vmax=200.0)
    cmap = plt.get_cmap("viridis")
    truth = D["test_truth"]
    planes_z = D["planes"]
    for i in pick:
        c = cmap(norm(P[i]))
        m = skey == pkey[i]
        zz, xx = st["z"][m], st["x"][m]
        o = np.argsort(zz)
        ax.plot(zz[o], xx[o], ".", ms=4.5, color=c, zorder=5)
        ax.plot(zz[o], xx[o], "-", lw=0.7, alpha=0.5, color=c, zorder=4)
        ax.plot(planes_z, truth[i, :, 0], "-", lw=2.0, color=c, zorder=6)
        ax.plot([D["test_z_pre"][i]], [D["test_S_pre"][i, 0]], "o", ms=7, mfc="none",
                mec=c, mew=1.6, zorder=8)
        ax.plot([D["test_z_post"][i]], [D["test_S_post"][i, 0]], "s", ms=7, mfc="none",
                mec=c, mew=1.6, zorder=8)
    ax.plot([], [], "o", ms=7, mfc="none", mec="k", label="real last-UT state (start)")
    ax.plot([], [], "s", ms=7, mfc="none", mec="k", label="real first-SciFi state (data truth)")
    ax.plot([], [], "-", lw=2.0, color="k", label="RK6 truth across the crossing (field-only)")
    ax.plot([], [], ".", ms=5, color="k", label="the particle's other hit states")
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=[ax, axg], pad=0.05, fraction=0.025)
    cb.set_label("truth momentum  p  [GeV]")
    ax.set_ylabel("x  [mm]")
    ax.set_title("Block D: the data on the detector. %d real particles of the test split, "
                 "four per momentum band, the crossing they are extrapolated over, "
                 "and the RK6 truth (v8r1.%s)." % (len(pick), FIELD), fontsize=11)
    ax.grid(alpha=0.22, zorder=0)

    # the step grids
    cat = {1: "#2a78d6", 4: "#eb6834", 16: "#1baf7a", 64: "#eda100", 128: "#e87ba4"}
    axg.axvspan(mag_z0, mag_z1, color="#f2c14e", alpha=0.16, lw=0, zorder=0)
    for i, N in enumerate(N_VALUES):
        zk = Z0 + np.arange(N + 1) * L / N
        axg.plot(zk, np.full_like(zk, i), "|", ms=14 if N <= 16 else 8, mew=1.6 if N <= 16 else 1.0,
                 color=cat[N])
        axg.text(Z1 + 120, i, "N = %d, dz = %.0f mm, %d network%s" % (N, L / N, N, "s" if N > 1 else ""),
                 va="center", fontsize=9)
    axg.set_yticks([])
    axg.set_ylim(-0.7, len(N_VALUES) - 0.3)
    axg.set_xlabel("z  [mm]")
    axg.set_ylabel("the step grids")
    axg.grid(axis="x", alpha=0.22)
    fig.savefig(os.path.join(FIGURES, "block_d_schematic.png"), dpi=150, bbox_inches="tight")
    print("wrote figures/block_d_schematic.png")
    with open(os.path.join(RESULTS, "schematic_meta.json"), "w") as f:
        json.dump({"tracks": [{"EVT": int(D["test_EVT"][i]), "MCKEY": int(D["test_MCKEY"][i]),
                               "p_GeV": float(P[i])} for i in pick],
                   "planes_source": os.path.relpath(STATES_NPZ, HERE),
                   "magnet": {"peak_z_mm": peak_z, "peak_B_T": peak_B,
                              "shaded_from_mm": mag_z0, "shaded_to_mm": mag_z1},
                   "crossing": {"z0": Z0, "z1": Z1, "L": L, "N_values": list(N_VALUES)}},
                  f, indent=1)


if __name__ == "__main__":
    main()
