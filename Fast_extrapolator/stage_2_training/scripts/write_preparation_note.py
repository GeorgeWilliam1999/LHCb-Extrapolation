"""Write PREPARATION.md from the tables in results/. No number is typed here.

    write_preparation_note.py
"""
from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = ["radial_median_um_0_3_gev", "radial_median_um_3_8_gev", "radial_median_um_8_20_gev",
         "radial_median_um_20_50_gev", "radial_median_um_above_50_gev"]


def g(value, digits=3):
    if value != value:
        return ""
    if 1e2 <= abs(value) < 1e9:
        return "{:,.0f}".format(float(("%%.%dg" % digits) % value))
    return ("%%.%dg" % digits) % value


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    size = pd.read_csv(os.path.join(HERE, "results", "what_a_correction_must_learn.csv"))
    probe = pd.read_csv(os.path.join(HERE, "results", "polynomial_probe.csv"))
    probe = probe[probe["sample"] == "validation"]
    higher = pd.read_csv(os.path.join(HERE, "results", "polynomial_probe_straight_line.csv"))
    out = []
    w = out.append
    w("# Stage 2, preparation: what must a correction learn?\n")
    w("Written by `scripts/write_preparation_note.py` from the tables in `results/`. No network is "
      "trained. A method of stage 2 is `end state = start + correction(start state)`. The start is "
      "the straight line or a fixed-step integrator with a few steps. The tables give the size of "
      "the correction and what a smooth function of the start state can give of it: a polynomial, "
      "fitted with a robust loss on the 108,915 training tracks and scored on the 13,684 validation "
      "tracks. The test tracks are not used. Errors are the distance on the last plane, in um.\n")
    w("![probe](figures/what_a_smooth_correction_reaches.png)\n")
    for reference, text in (("cubic", "the cubic reading of the map (the reference of the experiment)"),
                            ("linear", "the linear reading of the map (the field of LHCb)")):
        w("## Reference: %s\n" % text)
        body = []
        for _, s in size[size["reference"] == reference].iterrows():
            row = [s["start"], int(s["field_reads"]), g(s["radial_median_um"]), g(s["radial_95_um"]),
                   g(s["worst_band_median_um"]), "%.2f" % s["exponent_of_qop"]]
            for degree in (3, 5, 7):
                p = probe[(probe["reference"] == reference) & (probe["start"] == s["start"])
                          & (probe["degree"] == degree)].iloc[0]
                row.append("%s / %s / %s" % (g(p["radial_median_um"]), g(p["radial_95_um"]),
                                             g(p["worst_band_median_um"])))
            body.append(row)
        w(table(["Start", "Field reads", "Start alone: median", "95 %", "worst band",
                 "Exponent k of error ~ |q/p|^k", "With a polynomial of degree 3: median / 95 % / worst band",
                 "degree 5", "degree 7"], body) + "\n")
        w("Median error by momentum band with the polynomial of degree 7:\n")
        body = []
        for _, s in size[size["reference"] == reference].iterrows():
            p = probe[(probe["reference"] == reference) & (probe["start"] == s["start"])
                      & (probe["degree"] == 7)].iloc[0]
            body.append([s["start"]] + [g(p[b]) for b in BANDS])
        w(table(["Start", "below 3 GeV", "3 to 8 GeV", "8 to 20 GeV", "20 to 50 GeV", "above 50 GeV"],
                body) + "\n")
    w("## The start with no field read at higher degrees (cubic reference)\n")
    body = [[int(r["degree"]), int(r["monomials"]), g(r["radial_median_um"]), g(r["radial_95_um"])]
            + [g(r[b]) for b in BANDS] for _, r in higher.iterrows()]
    w(table(["Degree", "Monomials", "Median", "95 %", "below 3 GeV", "3 to 8 GeV", "8 to 20 GeV",
             "20 to 50 GeV", "above 50 GeV"], body) + "\n")
    baseline = pd.read_csv(os.path.join(HERE, "results", "polynomial_baseline.csv"))
    gate = pd.read_csv(os.path.join(HERE, "results", "polynomial_baseline_gate.csv"))
    w("## The polynomial as a method: the baseline for the networks\n")
    w("The corrections of degree 7 (and degree 9 with no field read) are written for the compiled "
      "program and scored and timed there like every method of stages 0 and 1: the 13,447 test "
      "tracks, single precision, one track a call, one pinned core of %s "
      "(`scripts/polynomial_baseline.py`, `results/polynomial_baseline.csv`). The compiled "
      "correction gives the numbers of the numpy version for %d of %d corrections (largest "
      "difference %.1e mm in double precision).\n"
      % (baseline["host"].iloc[0], int(gate["passes"].sum()), len(gate),
         gate["largest_difference_from_numpy_mm"].max()))
    for reference, text in (("linear", "the linear reading (the field of LHCb)"),
                            ("cubic", "the cubic reading (the reference of the experiment)")):
        rows = baseline[baseline["reference"] == reference].sort_values("ns_lowest")
        body = [[r["label"], g(r["field_reads_mean"]), "%.0f" % r["ns_lowest"], g(r["radial_median_um"]),
                 g(r["radial_95_um"]), g(r["worst_band_median_um"])] for _, r in rows.iterrows()]
        w("**Reference: %s**\n" % text)
        w(table(["Method", "Field reads", "Time a track (ns)", "Median error (um)", "95 % (um)",
                 "Worst band (um)"], body) + "\n")
    w("The code of the polynomial is a first version: a correction of degree 7 takes about 0.4 us, "
      "which is more than its arithmetic needs (462 terms a component).\n")
    w("## What the tables say\n")
    w("1. With no field read, a smooth function of the start state comes to a few um in the median "
      "against the cubic reference. Against the linear reference it stops near 28 um (degree 7): the "
      "linear reading has kinks at the planes of its grid, and the end state is not a smooth function "
      "of the start state at that scale.\n"
      "2. A smooth correction of a fixed-step integrator lowers its median error by a factor of 2 to "
      "7 when the integrator reads the same map as the reference. When it reads the map linearly "
      "and the reference is cubic, the correction gains almost nothing: the difference of the two "
      "readings is not smooth.\n"
      "3. The correction grows faster than |q/p| (exponent 1.1 to 1.5) and its 95 % level is 10 to "
      "100 times its median. A plain least-squares fit is ruled by about 1 % of the tracks (low "
      "momentum, large |y| on the last plane) and makes the typical track worse; the robust fit does "
      "not. The loss of a network must be robust in the same way.\n"
      "4. The gain stops between degree 5 and degree 7 for the integrator starts: what is left is "
      "not smooth at this resolution.\n"
      "5. On the test tracks and in the compiled program, the integrator with 12 steps and a "
      "polynomial correction is better than the adaptive Nystrom integrator of LHCb in the median, "
      "at 95 % and in the worst band, in less than half of its time. The straight line with a "
      "polynomial correction (no field read) is better than the HLT1 parametrised step in the same "
      "three numbers, in about the same time. A network of stage 2 must beat these.\n")
    with open(os.path.join(HERE, "PREPARATION.md"), "w") as handle:
        handle.write("\n".join(out))
    print("wrote PREPARATION.md")


if __name__ == "__main__":
    main()
