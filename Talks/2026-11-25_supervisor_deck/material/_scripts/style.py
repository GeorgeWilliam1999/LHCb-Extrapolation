"""Shared figure style and paths for the deck material."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(OUT, "_work")
STORE = "/data/bfys/gscriven/rkpinn_store"
COLOURS = ["#0F4C5C", "#E36414", "#5B8E7D", "#6C5B7B", "#C9A227", "#5F6C7B"]
GREY = "#5F6C7B"
LIGHT = "#B8C0C9"
BANDS = ("below 3 GeV", "3 to 8 GeV", "8 to 20 GeV", "20 to 50 GeV", "above 50 GeV")
BAND_LABELS = ("< 3", "3–8", "8–20", "20–50", "> 50")
EDGES = (0.0, 3.0, 8.0, 20.0, 50.0, float("inf"))
FIRST_PLANE, LAST_PLANE = 2648.2, 7826.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.prop_cycle": matplotlib.cycler(color=COLOURS), "figure.figsize": (9, 5),
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def band_masks(p):
    import numpy as np
    return [(b, (p >= lo) & (p < hi)) for b, lo, hi in zip(BANDS, EDGES[:-1], EDGES[1:])]
