"""The endpoint error of the network-free references against the number of steps,
cubic map, 13,447 test tracks. Reads results/*.csv; writes figures/references_against_steps.png."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [dict(r, file=f) for f in ("rk4.csv", "exact_scheme.csv")
        for r in csv.DictReader(open(os.path.join(HERE, "results", f)))]
FLOOR_UM = 2.6e-6      # where the exact scheme at 64 to 256 steps stops falling: the reference's own error

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
series = [("classic RK4", "rk4.csv", "4", "black", "s")] + [
    ("exact scheme, %s stages" % q, "exact_scheme.csv", q, c, "o")
    for q, c in (("2", "#1f77b4"), ("4", "#2ca02c"), ("8", "#ff7f0e"), ("16", "#d62728"))]
for ax, band in zip(axes, ("all momenta", "3 to 8 GeV")):
    for label, f, q, colour, marker in series:
        pts = sorted((int(r["number_of_steps"]), float(r["endpoint_radial_p50_micrometres"]))
                     for r in rows if r["file"] == f and r["number_of_stages"] == q
                     and r["momentum_band"] == band)
        ax.plot(*zip(*pts), marker=marker, color=colour, label=label)
    ax.axhline(FLOOR_UM, color="grey", ls=":", lw=1)
    ax.text(2.2, FLOOR_UM * 1.6, "reference floor", color="grey", fontsize=8)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("number of steps across 5,178 mm")
    ax.set_title("%s (%s tracks)" % (band, next(r["number_of_tracks"] for r in rows if r["momentum_band"] == band)))
    ax.grid(True, which="major", alpha=0.3)
axes[0].set_ylabel("endpoint radial error, median (µm)")
axes[0].legend(fontsize=8)
fig.suptitle("Network-free references on the cubic field map, test split of 5558be31abeb")
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "references_against_steps.png"), dpi=150)
