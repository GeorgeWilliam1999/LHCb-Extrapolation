"""The figures of the experiment, from results/networks.csv and results/references.csv.

    draw_figure.py

Writes, for each momentum band,
    figures/step_length_against_error_<band>.png   four panels: single step and end of region,
                                                   the error (top) and the error divided by the
                                                   step length (bottom)
    figures/stages_at_64_steps_<band>.png          the same two errors against the stages

Every point is the median over the tracks; the bar is the 16th to 84th percentile.
The relative error is the radial error in micrometres divided by the step length in
millimetres, so 1 micrometre per millimetre is a relative error of 1e-3. The step
length is the same for every track of a series, so its quantiles are those of the
error divided by the step length.
A package run that has not plateaued on the validation error is drawn hollow. The
frozen networks of the paper ended at a restart cap and are drawn filled. The
direct-states and frozen markers are moved about 12 per cent left and right of their step
length so that they do not hide the 64-step point.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS, FIGURES = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
LENGTH_MM = 7826.0 - 2648.2
# the reference categorical order of the dataviz palette, fixed
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, INK_2 = "#8a8985", "#0b0b0b", "#52514e"
PANELS = (("single_step", "Single step: one step from the reference state"),
          ("endpoint", "End of region: the whole chain, z = 2648.2 to 7826.0 mm"))
STRAIGHT = "straight line, per-track scale"


def read(name):
    with open(os.path.join(RESULTS, name)) as handle:
        return list(csv.DictReader(handle))


def band_of(file):
    return {"all momenta": "all_momenta", "3 to 8 GeV": "3_to_8_GeV"}[file]


def values(rows, what, relative=False):
    dz = np.array([float(r["step_length_mm"]) for r in rows])
    order = np.argsort(dz)
    rows, dz = [rows[i] for i in order], dz[order]
    lo, mid, hi = (np.array([float(r["%s_radial_p%s_micrometres" % (what, p)]) for r in rows])
                   / (dz if relative else 1.0) for p in ("16", "50", "84"))
    return rows, dz, mid, np.vstack([mid - lo, hi - mid])


def style(ax, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
    ax.grid(True, which="minor", color="#f0efec", linewidth=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)


def network_points(ax, rows, what, colour, marker, label, x_of=None, relative=False):
    if not rows:
        return
    rows, dz, mid, bars = values(rows, what, relative)
    x = dz if x_of is None else np.array([x_of(r) for r in rows])
    done = np.array([r["state"] in ("plateaued", "at the restart cap") for r in rows])
    ax.errorbar(x, mid, yerr=bars, fmt="none", ecolor=colour, elinewidth=1.2, capsize=3, zorder=3)
    if len(rows) > 1:
        ax.plot(x, mid, color=colour, linewidth=2, zorder=3)
    ax.plot(x[done], mid[done], marker, color=colour, markersize=8, linestyle="none",
            markeredgecolor="white", markeredgewidth=1.5, zorder=4, label=label)
    if (~done).any():
        ax.plot(x[~done], mid[~done], marker, markerfacecolor="white", markeredgecolor=colour,
                markeredgewidth=1.8, markersize=8, linestyle="none", zorder=4,
                label=label + " (not plateaued)")
    return rows, x, mid


def step_length_figure(band):
    nets = [r for r in read("networks.csv") if r["momentum_band"] == band]
    refs = [r for r in read("references.csv") if r["momentum_band"] == band]
    fig, grid = plt.subplots(2, 2, figsize=(12, 9.6), sharex=True)
    for ax, (what, title), relative in zip(grid.flat, PANELS + PANELS, (False, False, True, True)):
        style(ax, title.split(":")[0] + ", error divided by the step length" if relative else title)
        for q, dash in ((4, "--"), (8, ":"), (16, "-.")):
            rows, dz, mid, _ = values([r for r in refs if r["scheme"].startswith("exact")
                                       and int(r["number_of_stages"]) == q], what, relative)
            ax.plot(dz, mid, color=GRAY, linewidth=1.2, linestyle=dash, zorder=1,
                    label="exact scheme, %d stages" % q)
        rows, dz, mid, bars = values([r for r in refs if r["scheme"].startswith("exact")
                                      and int(r["number_of_stages"]) == 2], what, relative)
        ax.errorbar(dz, mid, yerr=bars, color=ORANGE, linewidth=2, capsize=2, marker="s",
                    markersize=5, zorder=2, label="exact Gauss–Legendre scheme, 2 stages")
        rows, dz, mid, bars = values([r for r in refs if r["scheme"].startswith("classic")], what, relative)
        ax.errorbar(dz, mid, yerr=bars, color=AQUA, linewidth=2, capsize=2, marker="^",
                    markersize=5, zorder=2, label="classic fourth-order Runge–Kutta")
        network_points(ax, [r for r in nets if r["series"] == STRAIGHT
                            and int(r["number_of_stages"]) == 2], what, BLUE, "o",
                       "network, straight line, per-track scale, 2 stages", relative=relative)
        for series, colour, marker, shift, label in (
                ("direct states", YELLOW, "D", 0.89, "network, direct states"),
                ("frozen weights, stages summed", MAGENTA, "P", 1.12,
                 "frozen network of the paper, stages summed")):
            got = network_points(ax, [r for r in nets if r["series"] == series], what,
                                 colour, marker, label + " (stages rise with steps)",
                                 x_of=lambda r, s=shift: s * float(r["step_length_mm"]),
                                 relative=relative)
            if got:
                for r, x, y in zip(*got):
                    ax.annotate("%s/%s" % (r["number_of_steps"], r["number_of_stages"]), (x, y),
                                fontsize=7, color=INK_2, xytext=(6, -3), textcoords="offset points")
        ax.set_xlim(14, 3800)
        if relative:
            ax.set_xlabel("step length dz (mm)", color=INK_2, fontsize=9)
            ax.set_ylabel("radial error / step length, micrometres per mm", color=INK_2, fontsize=9)
            continue
        ax.set_ylabel("radial error, micrometres", color=INK_2, fontsize=9)
        top = ax.secondary_xaxis("top", functions=(lambda d: LENGTH_MM / d, lambda n: LENGTH_MM / n))
        top.set_xlabel("number of steps across the magnet", color=INK_2, fontsize=9)
        top.tick_params(colors=INK_2, labelsize=8)
        top.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
        top.set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256"])
    handles, labels = grid[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Step length against error, %s, 1,452 test tracks, median and 16th–84th percentile"
                 % band, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, "step_length_against_error_%s.png" % band_of(band))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def stages_figure(band):
    nets = [r for r in read("networks.csv") if r["momentum_band"] == band
            and r["series"] == STRAIGHT and int(r["number_of_steps"]) == 64]
    refs = [r for r in read("references.csv") if r["momentum_band"] == band
            and r["scheme"].startswith("exact") and int(r["number_of_steps"]) == 64]
    if len(nets) < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True)
    stage = lambda r: int(r["number_of_stages"])  # noqa: E731
    for ax, (what, title) in zip(axes, PANELS):
        style(ax, title)
        ax.set_xscale("log", base=2)
        rows, _, mid, bars = values(refs, what)
        x = [stage(r) for r in rows]
        order = np.argsort(x)
        ax.errorbar(np.array(x)[order], mid[order], yerr=bars[:, order], color=ORANGE, linewidth=2,
                    capsize=2, marker="s", markersize=5, label="exact Gauss–Legendre scheme")
        network_points(ax, sorted(nets, key=stage), what, BLUE, "o",
                       "network, straight line, per-track scale", x_of=stage)
        ax.set_xticks([2, 4, 8, 16])
        ax.set_xticklabels(["2", "4", "8", "16"])
        ax.set_xlabel("number of stages, 64 steps of 80.9 mm", color=INK_2, fontsize=9)
        ax.set_ylabel("radial error, micrometres", color=INK_2, fontsize=9)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Stages at one step length, %s" % band, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    path = os.path.join(FIGURES, "stages_at_64_steps_%s.png" % band_of(band))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for band in ("all momenta", "3 to 8 GeV"):
        print(step_length_figure(band))
        print(stages_figure(band))
