"""The tables, figures and results note of stage 1, from the tables in results/
and from the tables of stage 0.

    figures/frontier.png                  compiled time against median error: the classical
                                          methods by family, the LHCb defaults and the present
                                          networks; one panel a field map
    figures/error_against_the_number_of_steps.png
                                          the error of the fixed-step integrator for 8 to 96
                                          equal steps, on both maps
    figures/shifted_planes.png            the same integrators on 15 crossings with other planes
    figures/floor_of_the_layouts.png      the error of the exact scheme of each layout
    results/cheapest_methods.csv          for each map and target, the fastest method of each
                                          family that meets it
    results/what_a_chained_network_costs.csv
                                          the lowest time of a chained network on each layout
                                          whose exact scheme meets target A
    RESULTS.md                            the results note

Target A: median 1 um, 95 % at 10 um, every momentum band at 3 um or less.
Target B: 5 um, 40 um, 10 um.
LHCb level: not worse than the adaptive Nystrom integrator of LHCb at its default
tolerance on the same map, in the median, at 95 % and in the worst band.

    analysis.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS, FIGURES = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
STAGE_0 = os.path.join(os.path.dirname(HERE), "stage_0_baselines", "results")
MAPS = ("linear", "cubic")
MEASURES = ("radial_median_um", "radial_95_um", "worst_band_median_um")
TARGET_A = dict(zip(MEASURES, (1.0, 10.0, 3.0)))
TARGET_B = dict(zip(MEASURES, (5.0, 40.0, 10.0)))
STYLE = {"RK4, equal steps": ("#7f7f7f", "o", "-"),
         "Nystrom, equal steps": ("#1a1a1a", "s", "-"),
         "RK4, step plan": ("#bcbd22", "o", ":"),
         "Nystrom, step plan": ("#8c564b", "s", ":"),
         "HLT2 adaptive Nystrom": ("#d62728", "^", "-"),
         "HLT2 adaptive Cash-Karp": ("#e377c2", "v", "-"),
         "Allen adaptive Nystrom": ("#ff9896", "<", "--"),
         "Gauss-Legendre sweeps": ("#2ca02c", "x", "")}


def g(value, digits=3):
    """A number with `digits` significant digits, and no exponent from 100 to 10 million."""
    if value != value:
        return ""
    if abs(value) >= 1e3 and abs(value - round(value)) < 1e-9:
        return "{:,}".format(int(round(value)))
    if 1e2 <= abs(value) < 1e7:
        return "{:,.0f}".format(float(("%%.%dg" % digits) % value))
    return ("%%.%dg" % digits) % value


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def meets(row, target):
    return all(row[key] <= limit for key, limit in target.items())


def those_that_meet(frame, target):
    if frame.empty:
        return frame
    return frame[[meets(r, target) for _, r in frame.iterrows()]]


def lhcb_default(frontier, of_map):
    return frontier[(frontier["map_of_the_field"] == of_map)
                    & (frontier["family"] == "HLT2 adaptive Nystrom")
                    & (frontier["tolerance_um"] == 5.0)].iloc[0]


def cash_karp_default(frontier, of_map):
    return frontier[(frontier["map_of_the_field"] == of_map)
                    & (frontier["family"] == "HLT2 adaptive Cash-Karp")
                    & (frontier["tolerance_um"] == 1.0)].iloc[0]


# ------------------------------------------------------------------ figures
def frontier_figure(frontier, stage_0):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), sharey=True)
    handles = {}
    for ax, of_map in zip(axes, MAPS):
        rows = frontier[frontier["map_of_the_field"] == of_map]
        for family, (colour, marker, line) in STYLE.items():
            part = rows[rows["family"] == family].sort_values("ns_lowest")
            if part.empty:
                continue
            handles[family], = ax.plot(part["ns_lowest"], part["radial_median_um"], color=colour,
                                       marker=marker, ls=line if line else "none", lw=0.9, ms=5)
        default = lhcb_default(frontier, of_map)
        ax.scatter([default["ns_lowest"]], [default["radial_median_um"]], s=190, facecolor="none",
                   edgecolor="#d62728", lw=1.6, zorder=4)
        ax.annotate("LHCb default\n(tolerance 5 um)", (default["ns_lowest"], default["radial_median_um"]),
                    textcoords="offset points", xytext=(10, 6), fontsize=8, color="#d62728")
        networks = stage_0[(stage_0["family"] == "network") & (stage_0["tanh"] == "fast")
                           & (stage_0["tracks"] == of_map)]
        handles["present trained networks"] = ax.scatter(
            networks["ns_lowest"], networks["radial_median_um"], s=34, color="#1f77b4",
            edgecolor="k", lw=0.3, zorder=3)
        if of_map == "linear":
            utt = stage_0[stage_0["label"].str.startswith("HLT1 parametrised step")]
            handles["HLT1 parametrised step"] = ax.scatter(
                utt["ns_lowest"], utt["radial_median_um"], s=90, color="#ff7f0e", marker="*", zorder=4)
        ax.axhline(1.0, color="#2ca02c", lw=1.0, ls=":")
        ax.text(0.99, 1.1, "target A: median 1 um", color="#2ca02c", fontsize=8, ha="right",
                transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(150, 3e5)
        ax.set_ylim(0.01, 3000)
        ax.set_xlabel("compiled time for one track (ns), one thread")
        ax.set_title("%s field map%s" % (of_map, " (the field of LHCb)" if of_map == "linear" else
                                         " (the standard of the project)"))
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("median radial error on the last plane (um), 13,447 test tracks")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=5, fontsize=8.5,
               frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(FIGURES, "frontier.png"), dpi=150)
    plt.close(fig)


def steps_figure(scan, frontier):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True)
    for ax, of_map in zip(axes, MAPS):
        rows = scan[(scan["map_of_the_field"] == of_map) & (scan["family"] == "equal steps")
                    & (scan["sample"] == "test")].sort_values("steps")
        ax.plot(rows["steps"], rows["radial_95_um"], color="#9ecae1", marker=".", ms=4, lw=0.8,
                label="95 % of the tracks")
        ax.plot(rows["steps"], rows["radial_median_um"], color="#08519c", marker="o", ms=3.5, lw=1.0,
                label="median")
        ax.axhline(1.0, color="#2ca02c", lw=1.0, ls=":")
        ax.text(0.995, 1.12, "target A: median 1 um", color="#2ca02c", fontsize=8, ha="right",
                transform=ax.get_yaxis_transform())
        default = lhcb_default(frontier, of_map)
        ax.axhline(default["radial_median_um"], color="#d62728", lw=1.0, ls="--")
        ax.text(0.995, default["radial_median_um"] * 1.12,
                "the adaptive Nystrom integrator of LHCb on this map: %s um (%s field reads)"
                % (g(default["radial_median_um"]), g(default["field_reads_mean"])),
                color="#d62728", fontsize=8, ha="right", transform=ax.get_yaxis_transform())
        ax.set_yscale("log")
        ax.set_ylim(0.05, 400)
        ax.set_ylabel("radial error on the last plane (um)")
        ax.set_title("%s map%s" % (of_map, ": the map has a kink at every plane of its grid (100 mm apart)"
                                   if of_map == "linear" else ": no kinks"), fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper center", ncol=2)
    axes[1].set_xlabel("number of equal steps from 2648.2 mm to 7826 mm (the Nystrom integrator; "
                       "field reads = 2 x steps + 1)")
    axes[1].set_xlim(7, 97)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "error_against_the_number_of_steps.png"), dpi=150)
    plt.close(fig)


def shifted_figure(shifted):
    labels = list(dict.fromkeys(shifted["label"]))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for i, label in enumerate(labels):
        rows = shifted[shifted["label"] == label]
        ours = rows[(rows["shift_of_the_first_plane_mm"] == 0) & (rows["shift_of_the_last_plane_mm"] == 0)]
        rest = rows.drop(ours.index)
        ax.scatter(np.full(len(rest), i), rest["radial_median_um"], s=22, color="#6b6b6b", alpha=0.7,
                   label="14 crossings with moved planes" if i == 0 else None)
        ax.scatter([i], ours["radial_median_um"], s=80, facecolor="none", edgecolor="#d62728", lw=1.8,
                   label="the planes of this experiment" if i == 0 else None)
    ax.axhline(1.0, color="#2ca02c", lw=1.0, ls=":")
    ax.text(0.995, 1.08, "target A: median 1 um", color="#2ca02c", fontsize=8, ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("Nystrom, ", "").replace(" equal steps", "\nequal steps")
                        .replace("HLT2 adaptive ", "LHCb adaptive\n").replace(", tolerance", "\ntolerance")
                        for l in labels], fontsize=8)
    ax.set_ylabel("median radial error on the last plane (um)")
    ax.set_title("Linear map: the same integrators when the first plane, the last plane or both "
                 "move by 5 mm to 50 mm", fontsize=10)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "shifted_planes.png"), dpi=150)
    plt.close(fig)


def floor_figure(floor):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    steps, stages = sorted(floor["steps"].unique()), sorted(floor["stages"].unique())
    for ax, of_map in zip(axes, MAPS):
        grid = np.full((len(stages), len(steps)), np.nan)
        rows = floor[floor["map_of_the_field"] == of_map]
        for _, r in rows.iterrows():
            grid[stages.index(r["stages"]), steps.index(r["steps"])] = r["radial_median_um"]
        ax.imshow(np.log10(grid), origin="lower", cmap="viridis_r", vmin=-3.5, vmax=4, aspect="auto")
        for _, r in rows.iterrows():
            i, j = stages.index(r["stages"]), steps.index(r["steps"])
            good = meets(r, TARGET_A)
            ax.text(j, i, g(r["radial_median_um"], 2) + ("\nmeets A" if good else ""), ha="center",
                    va="center", fontsize=8, color="white" if grid[i, j] < 30 else "black",
                    fontweight="bold" if good else "normal")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps)
        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels(stages)
        ax.set_xlabel("steps across the magnet")
        ax.set_ylabel("Gauss-Legendre stages a step")
        ax.set_title("%s map: median error of the exact scheme (um)" % of_map, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "floor_of_the_layouts.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ the note
def main():
    frontier = pd.read_csv(os.path.join(RESULTS, "classical_frontier.csv"))
    floor = pd.read_csv(os.path.join(RESULTS, "floor_of_the_layouts.csv"))
    chosen = pd.read_csv(os.path.join(RESULTS, "chosen_plans.csv"))
    scan = pd.read_csv(os.path.join(RESULTS, "step_length_scan.csv"))
    shifted = pd.read_csv(os.path.join(RESULTS, "shifted_planes.csv"))
    readings = pd.read_csv(os.path.join(RESULTS, "two_readings_of_the_map.csv"))
    stage_0 = pd.read_csv(os.path.join(STAGE_0, "methods.csv"))
    stage_0["tanh"] = stage_0["tanh"].fillna("")
    calls = pd.read_csv(os.path.join(STAGE_0, "micro_mlp.csv"))
    reads = pd.read_csv(os.path.join(STAGE_0, "micro_field.csv"))
    os.makedirs(FIGURES, exist_ok=True)
    frontier_figure(frontier, stage_0)
    steps_figure(scan, frontier)
    shifted_figure(shifted)
    floor_figure(floor)
    host = frontier["host"].iloc[0]

    def classical(of_map, family, steps):
        return frontier[(frontier["map_of_the_field"] == of_map) & (frontier["family"] == family)
                        & (frontier["steps"] == steps)].iloc[0]

    # the fastest method of each family for each target
    cheapest = []
    for of_map in MAPS:
        rows = frontier[frontier["map_of_the_field"] == of_map]
        default = lhcb_default(frontier, of_map)
        level = {key: float(default[key]) for key in MEASURES}
        for name, target in (("target A", TARGET_A), ("target B", TARGET_B), ("the LHCb level", level)):
            for family in STYLE:
                good = those_that_meet(rows[rows["family"] == family], target)
                if good.empty:
                    if not rows[rows["family"] == family].empty:
                        cheapest.append(dict(map=of_map, target=name, family=family, method="none"))
                    continue
                r = good.sort_values("ns_lowest").iloc[0]
                cheapest.append(dict(map=of_map, target=name, family=family, method=r["label"],
                                     field_reads=r["field_reads_mean"], ns=r["ns_lowest"],
                                     radial_median_um=r["radial_median_um"],
                                     radial_95_um=r["radial_95_um"],
                                     worst_band_median_um=r["worst_band_median_um"]))
    cheapest = pd.DataFrame(cheapest)
    cheapest.to_csv(os.path.join(RESULTS, "cheapest_methods.csv"), index=False)

    def fastest(of_map, target_name):
        part = cheapest[(cheapest["map"] == of_map) & (cheapest["target"] == target_name)
                        & (cheapest["method"] != "none")]
        return part.sort_values("ns").iloc[0] if len(part) else None

    # the unit costs of stage 0
    def call_ns(width):
        return float(calls[(calls["precision"] == "float") & (calls["depth"] == 2)
                           & (calls["width"] == width)
                           & (calls["activation"] == "tanh from the vector exponential")]["ns_lowest"].iloc[0])

    def read_ns(reader):
        return float(reads[(reads["reader"] == reader) & (reads["precision"] == "float")
                           & (reads["pattern"] == "along the tracks")]["ns_lowest"].iloc[0])

    read_of = {"linear": read_ns("rec"), "cubic": read_ns("cubic")}
    chained = []
    for of_map in MAPS:
        best = fastest(of_map, "target A")
        for _, r in those_that_meet(floor[floor["map_of_the_field"] == of_map], TARGET_A).iterrows():
            points = int(r["steps"] * r["stages"])
            row = dict(map=of_map, steps=int(r["steps"]), stages=int(r["stages"]), stage_points=points,
                       floor_median_um=r["radial_median_um"], floor_95_um=r["radial_95_um"],
                       reads_ns=points * read_of[of_map])
            for width in (16, 32, 64):
                row["lowest_time_ns_width_%d" % width] = r["steps"] * call_ns(width) + points * read_of[of_map]
            row["fastest_classical_method"] = best["method"] if best is not None else ""
            row["fastest_classical_ns"] = best["ns"] if best is not None else float("nan")
            chained.append(row)
    chained = pd.DataFrame(chained)
    chained.to_csv(os.path.join(RESULTS, "what_a_chained_network_costs.csv"), index=False)

    out = []
    w = out.append
    w("# Stage 1: results\n")
    w("Written by `scripts/analysis.py` from the tables in `results/`. No network is trained in "
      "this stage. 13,447 test tracks, field map v8r1 (magnet up), one crossing from 2648.2 mm to "
      "7826 mm. Timed on %s, one thread, one pinned core, one track a call.\n" % host)
    w("Target A: median error 1 um or less, 95 % of the tracks at 10 um or less, the median of every "
      "momentum band at 3 um or less. Target B: 5 um, 40 um, 10 um. The LHCb level: not worse than "
      "the adaptive Nystrom integrator of LHCb at its default tolerance, in the same three numbers.\n")

    # ---------------------------------------------------------------- summary
    lin = lhcb_default(frontier, "linear")
    ck = cash_karp_default(frontier, "linear")
    a_lin, b_lin, l_lin = fastest("linear", "target A"), fastest("linear", "target B"), fastest("linear", "the LHCb level")
    a_cub = fastest("cubic", "target A")
    ours = shifted[(shifted["shift_of_the_first_plane_mm"] == 0) & (shifted["shift_of_the_last_plane_mm"] == 0)]
    moved = shifted.drop(ours.index)

    def spread(label, frame=moved):
        part = frame[frame["label"] == label]["radial_median_um"]
        return part.min(), part.max()

    w("## Summary\n")
    w("1. **The LHCb reference on our tracks.** The adaptive Nystrom integrator at its default "
      "tolerance: %s field reads, %.2f us, median error %s um, 95 %% %s um, worst band %s um. The "
      "adaptive Cash-Karp integrator: %s reads, %.2f us, %s um, %s um, %s um."
      % (g(lin["field_reads_mean"]), lin["ns_lowest"] / 1e3, g(lin["radial_median_um"]),
         g(lin["radial_95_um"]), g(lin["worst_band_median_um"]), g(ck["field_reads_mean"]),
         ck["ns_lowest"] / 1e3, g(ck["radial_median_um"]), g(ck["radial_95_um"]),
         g(ck["worst_band_median_um"])))
    w("2. **A fixed-step integrator with no network is faster and more accurate than both, on the "
      "planes of this experiment.** The fastest method at the LHCb level or better is \"%s\": %.2f us, "
      "%s um, %s um, %s um. The fastest that meets target B is \"%s\": %.2f us. The fastest that meets "
      "target A is \"%s\": %.2f us, %s um, %s um, %s um."
      % (l_lin["method"], l_lin["ns"] / 1e3, g(l_lin["radial_median_um"]), g(l_lin["radial_95_um"]),
         g(l_lin["worst_band_median_um"]), b_lin["method"], b_lin["ns"] / 1e3, a_lin["method"],
         a_lin["ns"] / 1e3, g(a_lin["radial_median_um"]), g(a_lin["radial_95_um"]),
         g(a_lin["worst_band_median_um"])))
    low19, high19 = spread("Nystrom, 19 equal steps")
    low32, high32 = spread("Nystrom, 32 equal steps")
    low69, high69 = spread("Nystrom, 69 equal steps")
    lows, highs = spread("HLT2 adaptive Nystrom, tolerance 5 um")
    equal = scan[(scan["map_of_the_field"] == "linear") & (scan["family"] == "equal steps")
                 & (scan["sample"] == "test")].set_index("steps")["radial_median_um"]
    w("3. **That accuracy is tied to the planes.** The linear map has a kink at every plane of its "
      "grid, and the error of a fixed-step integrator depends on where its steps fall on that grid: "
      "with 25 equal steps the median error is %s um, with 32 it is %s um. When the planes of the "
      "crossing move by 5 mm to 50 mm, the integrator with 19 steps gives %s um to %s um, with 32 "
      "steps %s um to %s um, and with 69 steps %s um to %s um. The adaptive integrator of LHCb gives "
      "%s um to %s um on the same crossings. So on the LHCb map a fixed-step integrator with 32 "
      "steps stays better than LHCb, but it does not hold 1 um when the planes change."
      % (g(equal[25]), g(equal[32]), g(low19, 2), g(high19, 2), g(low32, 2), g(high32, 2),
         g(low69, 2), g(high69, 2), g(lows), g(highs)))
    r4, ny = classical("linear", "RK4, equal steps", 32), classical("linear", "Nystrom, equal steps", 32)
    w("4. **The number of field reads does not set the time of one track.** With 32 steps the "
      "Nystrom integrator reads the field %s times and takes %.2f us; RK4 reads it %s times and takes "
      "%.2f us. The likely cause: each operation of a step waits for the result of the one before "
      "it, and that chain sets the time, not the count of reads. This cause is not tested here."
      % (g(ny["field_reads_mean"]), ny["ns_lowest"] / 1e3, g(r4["field_reads_mean"]),
         r4["ns_lowest"] / 1e3))
    two = readings.set_index("label")
    apart = two.loc["The linear reading, integrated with 2,048 steps"]
    as_lhcb = two.loc["HLT2 adaptive Nystrom, linear reading (as LHCb runs it)"]
    w("5. **The two readings of the map are %s um apart.** That is the median distance on the last "
      "plane between the linear reading (LHCb) and the cubic reading (the standard of this project) "
      "of the same map points; 95 %% of the tracks are within %s um. Against the cubic reference the "
      "LHCb integrator has a median error of %s um. Every method that reads the map linearly has "
      "this error against the cubic reference."
      % (g(apart["radial_median_um"]), g(apart["radial_95_um"]), g(as_lhcb["radial_median_um"])))
    if a_cub is not None:
        w("6. **On the cubic map** the error falls smoothly with the number of steps. The fastest "
          "classical method that meets target A is \"%s\": %.2f us, %s um. A cubic field read costs "
          "%.0f ns, a linear one %.0f ns."
          % (a_cub["method"], a_cub["ns"] / 1e3, g(a_cub["radial_median_um"]), read_of["cubic"],
             read_of["linear"]))
    for number, of_map in enumerate(MAPS, 7):
        part = those_that_meet(floor[floor["map_of_the_field"] == of_map], TARGET_A)
        fewest = int((part["steps"] * part["stages"]).min()) if len(part) else 0
        w("%d. **Layouts on the %s map.** %d of %d layouts have an exact scheme that meets target A"
          "%s. A chained network with a summed end state cannot be better than the exact scheme of "
          "its layout, so on this map it reads the field at least %d times."
          % (number, of_map, len(part), len(floor[floor["map_of_the_field"] == of_map]),
             ": " + ", ".join("%d steps x %d stages" % (r["steps"], r["stages"])
                              for _, r in part.iterrows()) if len(part) else "", fewest))
    w("")

    # ---------------------------------------------------------------- frontier
    w("## 1. The classical frontier\n")
    w("![frontier](figures/frontier.png)\n")
    w("**The fastest method of each family for each target** (`results/cheapest_methods.csv`):\n")
    body = []
    for _, r in cheapest.iterrows():
        if r["method"] == "none":
            body.append([r["map"], r["target"], r["family"], "none", "", "", "", "", ""])
        else:
            body.append([r["map"], r["target"], r["family"], r["method"], g(r["field_reads"]),
                         "%.0f" % r["ns"], g(r["radial_median_um"]), g(r["radial_95_um"]),
                         g(r["worst_band_median_um"])])
    w(table(["Map", "Target", "Family", "Fastest method that meets it", "Field reads",
             "Time a track (ns)", "Median error (um)", "95 % (um)", "Worst band (um)"], body) + "\n")

    # ---------------------------------------------------------------- steps and the grid
    w("## 2. The error of a fixed-step integrator and the grid of the map\n")
    w("![steps](figures/error_against_the_number_of_steps.png)\n")
    w("On the linear map the error jumps by a factor of up to 100 from one number of steps to the "
      "next. The linear reading has a kink at every plane of the grid of the map, 100 mm apart, and "
      "every track meets these kinks at the same z. A fixed-step integrator makes a small error at "
      "each kink. When the step length is close to a whole number of cells, the kinks fall at the "
      "same place in every step and the errors add; for other step lengths they cancel in part. On "
      "the cubic map there are no kinks and the error falls smoothly (the level of 0.1 um is single "
      "precision). The training tracks give the same curve as the test tracks "
      "(`results/step_length_scan.csv`).\n")
    body = []
    for steps in (16, 19, 20, 24, 25, 26, 29, 31, 32, 33, 34, 35, 36, 48, 50, 64, 69, 77, 96):
        row = [steps, "%.1f" % ((7826.0 - 2648.2) / steps)]
        for of_map in MAPS:
            r = scan[(scan["map_of_the_field"] == of_map) & (scan["family"] == "equal steps")
                     & (scan["sample"] == "test") & (scan["steps"] == steps)].iloc[0]
            row += [g(r["radial_median_um"]), g(r["radial_95_um"]),
                    "A" if meets(r, TARGET_A) else ("B" if meets(r, TARGET_B) else "")]
        body.append(row)
    w(table(["Equal steps", "Step length (mm)", "Linear map: median (um)", "95 % (um)", "meets",
             "Cubic map: median (um)", "95 % (um)", "meets"], body) + "\n")
    w("![shifted](figures/shifted_planes.png)\n")
    w("The same integrators on 15 crossings of the linear map: the planes of this experiment, and "
      "14 crossings with the first plane, the last plane or both moved by 5 mm to 50 mm "
      "(`results/shifted_planes.csv`). Median error in um:\n")
    labels = list(dict.fromkeys(shifted["label"]))
    body = []
    for label in labels:
        o = ours[ours["label"] == label].iloc[0]
        low, high = spread(label)
        part = moved[moved["label"] == label]
        body.append([label, g(o["field_reads_mean"]), g(o["radial_median_um"]), g(low, 2), g(high, 2),
                     g(part["radial_95_um"].max()),
                     "%d of %d" % (sum(meets(r, TARGET_A) for _, r in part.iterrows()), len(part))])
    w(table(["Method", "Field reads", "Our planes: median (um)", "Moved planes: lowest median (um)",
             "highest median (um)", "highest 95 % (um)", "Moved crossings that meet target A"], body) + "\n")

    # ---------------------------------------------------------------- reads are not time
    w("## 3. Field reads and time\n")
    w("The Nystrom integrator reads the field at three points a step and shares one of them with "
      "the next step; RK4 reads it at four. The errors are the same. On the linear map the Nystrom "
      "integrator is not faster, although it reads the field half as often. The likely cause: a "
      "read of the linear map is cheap, and the time of a step is set by the chain of operations in "
      "which each one waits for the result of the one before it (in the Nystrom step all four "
      "evaluations of the slopes are in one chain; in the RK4 step two reads can run side by side). "
      "This cause is not tested here; a test is to carry several tracks in one loop. On the cubic "
      "map a read is expensive and the Nystrom integrator is faster.\n")
    body = []
    for steps in (8, 16, 20, 32, 64):
        row = [steps]
        for of_map in MAPS:
            a, b = classical(of_map, "RK4, equal steps", steps), classical(of_map, "Nystrom, equal steps", steps)
            row += [g(a["field_reads_mean"]), "%.0f" % a["ns_lowest"], g(b["field_reads_mean"]),
                    "%.0f" % b["ns_lowest"]]
        body.append(row)
    w(table(["Equal steps", "Linear map: RK4 reads", "RK4 time (ns)", "Nystrom reads", "Nystrom time (ns)",
             "Cubic map: RK4 reads", "RK4 time (ns)", "Nystrom reads", "Nystrom time (ns)"], body) + "\n")

    # ---------------------------------------------------------------- LHCb tolerances
    w("## 4. The LHCb integrators at other tolerances\n")
    w("The tolerance of the adaptive Nystrom integrator is scanned from 0.1 um to 100 um (default "
      "5 um); the two tolerances of the Cash-Karp integrator are scaled together (default: factor 1). "
      "On the linear map a tighter tolerance buys little: the step control meets the kinks of the "
      "map.\n")
    for of_map in MAPS:
        rows = frontier[(frontier["map_of_the_field"] == of_map)
                        & frontier["family"].isin(["HLT2 adaptive Nystrom", "HLT2 adaptive Cash-Karp",
                                                   "Allen adaptive Nystrom"])]
        body = [[r["label"], g(r["field_reads_mean"]), "%.0f" % r["ns_lowest"], g(r["radial_median_um"]),
                 g(r["radial_95_um"]), g(r["worst_band_median_um"])] for _, r in rows.iterrows()]
        w("**%s map**\n" % of_map.capitalize())
        w(table(["Method", "Field reads", "Time a track (ns)", "Median error (um)", "95 % (um)",
                 "Worst band (um)"], body) + "\n")

    # ---------------------------------------------------------------- two readings
    w("## 5. The two readings of the map\n")
    w("The test tracks of the cubic track set, carried with methods that read the map as named, "
      "scored against the cubic reference (`results/two_readings_of_the_map.csv`):\n")
    body = [[r["label"], g(r["field_reads_mean"]), g(r["radial_median_um"]), g(r["radial_95_um"]),
             g(r["radial_median_um_0_3_gev"]), g(r["radial_median_um_3_8_gev"]),
             g(r["radial_median_um_8_20_gev"]), g(r["radial_median_um_20_50_gev"]),
             g(r["radial_median_um_above_50_gev"])] for _, r in readings.iterrows()]
    w(table(["Method", "Field reads", "Median error (um)", "95 % (um)", "below 3 GeV", "3 to 8 GeV",
             "8 to 20 GeV", "20 to 50 GeV", "above 50 GeV"], body) + "\n")

    # ---------------------------------------------------------------- step plans
    w("## 6. Steps of unequal length\n")
    kept = chosen[chosen["beta"] == 0]
    w("A step plan puts shorter steps where a step costs more error (`scripts/step_plans.py`). The "
      "exponent beta of a plan is chosen on 2,000 training tracks; beta = 0 is equal steps. Equal "
      "steps are chosen in %d of %d cases. The table gives the Nystrom integrator on the test "
      "tracks: a plan lowers the error at some numbers of steps and changes nothing at 16 and 32.\n"
      % (len(kept), len(chosen)))
    body = []
    for of_map in MAPS:
        for steps in (6, 8, 10, 12, 16, 20, 24, 32):
            beta = chosen[(chosen["map_of_the_field"] == of_map) & (chosen["integrator"] == "nystrom_plan")
                          & (chosen["steps"] == steps)]["beta"].iloc[0]
            equal_steps = classical(of_map, "Nystrom, equal steps", steps)
            plan = classical(of_map, "Nystrom, step plan", steps)
            body.append([of_map, steps, g(beta), g(equal_steps["radial_median_um"]),
                         g(equal_steps["radial_95_um"]), g(plan["radial_median_um"]),
                         g(plan["radial_95_um"])])
    w(table(["Map", "Steps", "Chosen beta", "Equal steps: median (um)", "95 % (um)",
             "Step plan: median (um)", "95 % (um)"], body) + "\n")

    # ---------------------------------------------------------------- floor
    w("## 7. The floor of each layout\n")
    w("![floor](figures/floor_of_the_layouts.png)\n")
    w("The exact Gauss-Legendre scheme of a layout is what a network with a summed end state gives "
      "when its loss is zero. It is computed with sweeps of the stage map until nothing changes.\n")
    body = []
    for _, r in floor.iterrows():
        body.append([r["map_of_the_field"], int(r["steps"]), int(r["stages"]),
                     int(r["steps"] * r["stages"]), g(r["radial_median_um"]),
                     g(r["radial_95_um"]), g(r["worst_band_median_um"]),
                     "yes" if meets(r, TARGET_A) else "no", "yes" if r["converged"] else "NO"])
    w(table(["Map", "Steps", "Stages", "Stage points (field reads of one sweep)", "Median error (um)",
             "95 % (um)", "Worst band (um)", "Meets target A", "Sweeps converged"], body) + "\n")

    # ---------------------------------------------------------------- what a network must beat
    w("## 8. What a chained network would cost at target A\n")
    w("For each layout whose exact scheme meets target A: the lowest time of a chained network "
      "with a summed end state. It is one network call a step and one field read a stage point, "
      "with the costs of stage 0: a call of %.0f ns (2 x 16), %.0f ns (2 x 32) or %.0f ns (2 x 64) and "
      "a read of %.0f ns (linear) or %.0f ns (cubic). It assumes that the small network reaches the "
      "exact scheme, which no present network does (`results/what_a_chained_network_costs.csv`).\n"
      % (call_ns(16), call_ns(32), call_ns(64), read_of["linear"], read_of["cubic"]))
    body = [[r["map"], r["steps"], r["stages"], r["stage_points"], g(r["floor_median_um"]),
             "%.0f" % r["lowest_time_ns_width_16"], "%.0f" % r["lowest_time_ns_width_32"],
             "%.0f" % r["lowest_time_ns_width_64"], "%.0f" % r["fastest_classical_ns"]]
            for _, r in chained.iterrows()]
    w(table(["Map", "Steps", "Stages", "Stage points", "Floor: median (um)", "Lowest time, 2 x 16 (ns)",
             "2 x 32 (ns)", "2 x 64 (ns)", "Fastest classical method at target A (ns)"], body) + "\n")

    # ---------------------------------------------------------------- every method
    w("## 9. Every method of the stage\n")
    for of_map in MAPS:
        rows = frontier[frontier["map_of_the_field"] == of_map].sort_values("ns_lowest")
        w("**%s map**, fastest first; A or B = the target that the method meets on the planes of "
          "this experiment.\n" % of_map.capitalize())
        body = []
        for _, r in rows.iterrows():
            body.append([r["label"], r["precision"], g(r["field_reads_mean"]), "%.0f" % r["ns_lowest"],
                         "%.0f" % r["cold_ns_median"], g(r["radial_median_um"]), g(r["radial_95_um"]),
                         g(r["worst_band_median_um"]),
                         "A" if meets(r, TARGET_A) else ("B" if meets(r, TARGET_B) else "")])
        w(table(["Method", "Precision", "Field reads", "Time a track (ns)", "Cold caches (ns)",
                 "Median error (um)", "95 % (um)", "Worst band (um)", ""], body) + "\n")

    with open(os.path.join(HERE, "RESULTS.md"), "w") as handle:
        handle.write("\n".join(out))
    print("wrote RESULTS.md, %d lines" % len(out))
    print("\n".join(out[6:16]))


if __name__ == "__main__":
    sys.exit(main())
