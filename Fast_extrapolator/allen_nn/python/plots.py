"""The figures of P3 from allen_nn/results/*.csv (writes allen_nn/figures/*.png)."""
from __future__ import annotations

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS, FIGURES = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
TOOL, ARKN, NYS, PARKF, REC = "#2563eb", "#dc2626", "#9ca3af", "#d97706", "#7c3aed"


def rows(name):
    with open(os.path.join(RESULTS, name)) as handle:
        return list(csv.DictReader(handle))


def colour(method):
    if method.startswith("tool"):
        return TOOL
    if method.startswith("ARKN"):
        return ARKN
    if method.startswith("ParKF"):
        return PARKF
    if method.startswith("Rec"):
        return REC
    return NYS


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED)


def accuracy_vs_time():
    end = [r for r in rows("endpoint_errors.csv") if r["band"] == "all"]
    timing_file = sorted(glob.glob(os.path.join(RESULTS, "timing_*.csv")))[0]
    timing = {r["method"]: float(r["lowest_ns"]) for r in rows(os.path.basename(timing_file))}
    spec_of = {"ARKN (Allen)": "arkn", "ParKF step (2665->7826)": "parkf"}
    for k in (6, 8, 12, 32):
        spec_of["Nystrom %d steps" % k] = "nystrom:%d" % k
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, q, title in ((axes[0], "radial_um_median", "median"), (axes[1], "radial_um_p95", "95 %")):
        for r in end:
            m = r["method"]
            spec = spec_of.get(m)
            if m.startswith("tool"):
                k, w = m.split()[1][2:], m.split()[2]
                spec = next((s for s in timing if s.startswith("tool:") and s.endswith("linear_nystrom_%s_%s.json" % (k, w))), None)
            if spec is None or spec not in timing:
                continue
            ax.scatter(timing[spec] / 1e3, float(r[q]), s=40, color=colour(m), zorder=3)
            if q == "radial_um_p95" and m.startswith("tool") and m != "tool K=12 2x64":
                continue  # the tool points overlap on the right; the left panel names them
            ax.annotate(m.replace(" (Allen)", "").replace(" (2665->7826)", ""), (timing[spec] / 1e3, float(r[q])),
                        textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=INK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("time per call with transport matrix (us, one core, float)", color=INK)
        ax.set_ylabel("radial end-point error, %s (um)" % title, color=INK)
        style(ax)
    fig.suptitle("Magnet crossing 2648.2 -> 7826 mm, 13,447 test tracks, trilinear map", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "accuracy_vs_time.png"), dpi=150)


def bands():
    end = rows("endpoint_errors.csv")
    keep = ["tool K=8 2x64", "tool K=12 2x64", "ARKN (Allen)", "Nystrom 8 steps", "ParKF step (2665->7826)"]
    names = ["p<3", "3-8", "8-20", ">20"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.16
    for i, m in enumerate(keep):
        v = [float(next(r for r in end if r["method"] == m and r["band"] == b)["radial_um_median"]) for b in names]
        ax.bar(np.arange(4) + (i - 2) * width, v, width, color=colour(m) if not m.endswith("12 2x64") else "#93c5fd",
               label=m)
    ax.set_xticks(np.arange(4), [b + " GeV" for b in names])
    ax.set_yscale("log")
    ax.set_ylabel("median radial end-point error (um)", color=INK)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "error_by_momentum.png"), dpi=150)


def transport():
    tr = [r for r in rows("transport_errors.csv") if r["band"] == "all"]
    keep = ["tool K=8 2x64", "tool K=12 2x64", "ARKN (Allen)", "Nystrom 8 steps", "ParKF step (2665->7826)"]
    changes = ["x0 50um", "y0 50um", "tx0 0.3mrad", "ty0 0.3mrad", "q/p 1%"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.16
    for ax, q in ((axes[0], "position_error_um_median"), (axes[1], "position_error_um_p95")):
        for i, m in enumerate(keep):
            v = [float(next(r for r in tr if r["method"] == m and r["change"] == c)[q]) for c in changes]
            ax.bar(np.arange(5) + (i - 2) * width, v, width, label=m,
                   color=colour(m) if not m.endswith("12 2x64") else "#93c5fd")
        ax.set_xticks(np.arange(5), changes, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel(("median" if "median" in q else "95 %") + " error of the predicted shift (um)", color=INK)
        style(ax)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Transport matrix: end-position shift for a one-sigma start change, vs finite differences",
                 color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "transport_matrix_error.png"), dpi=150)


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    accuracy_vs_time()
    bands()
    transport()
    print("wrote", FIGURES)
