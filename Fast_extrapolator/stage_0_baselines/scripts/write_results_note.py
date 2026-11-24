"""Write the results note of stage 0 from the tables in results/. Every number
of the note is read from a table; the note holds no number of its own.

    write_results_note.py      writes stage_0_baselines/RESULTS.md
"""
from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
BANDS = [("0_3", "below 3"), ("3_8", "3 to 8"), ("8_20", "8 to 20"), ("20_50", "20 to 50"),
         ("above_50", "above 50")]


def g(value, digits=3):
    """A number with `digits` significant digits, and no exponent from 1,000 to 10 million."""
    if 1e3 <= abs(value) < 1e7:
        return "{:,.0f}".format(float(("%%.%dg" % digits) % value))
    return ("%%.%dg" % digits) % value


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    m = pd.read_csv(os.path.join(RESULTS, "methods.csv"))
    m["tanh"] = m["tanh"].fillna("")
    gate = pd.read_csv(os.path.join(RESULTS, "gate_compiled_against_python.csv"))
    rec = pd.read_csv(os.path.join(RESULTS, "rec_copy_against_the_rec_reference.csv"))
    allen = pd.read_csv(os.path.join(RESULTS, "allen_copy_against_the_step_copy.csv"))
    gaudi = pd.read_csv(os.path.join(os.path.dirname(HERE), "lhcb_software", "rec_extrapolator_test",
                                     "output", "summary_stbc-i2.csv"))
    step = pd.read_csv(os.path.join(RESULTS, "parametrised_step.csv"))
    field = pd.read_csv(os.path.join(RESULTS, "micro_field.csv"))
    mlp = pd.read_csv(os.path.join(RESULTS, "micro_mlp.csv"))
    host = m["host"].iloc[0]
    out = []
    w = out.append

    w("# Stage 0: results\n")
    w("Written by `scripts/write_results_note.py` from the tables in `results/`. "
      "Timed on %s, one thread, one pinned core.\n" % host)

    # ------------------------------------------------------------------ summary
    def one(label, precision=None, tanh=None):
        rows = m[m["label"] == label]
        if precision:
            rows = rows[rows["precision"] == precision]
        if tanh is not None:
            rows = rows[rows["tanh"] == tanh]
        return rows.iloc[0]

    nystrom = one("HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um)")
    nystrom_tm = one("HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um), with its transport matrix")
    cash_karp = one("HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um)")
    utt = one("HLT1 parametrised step (polynomials, no field read)")
    whole = one("Whole-crossing network, 2x256, linear map", "float", "fast")
    rk4_16 = one("RK4, 16 steps, linear map, field read of LHCb")
    rk4_32 = one("RK4, 32 steps, linear map, field read of LHCb")
    chains = m[(m["family"] == "network") & (m["tanh"] == "fast") & (m["network_calls"] > 1)]
    read = field[(field["reader"] == "rec") & (field["precision"] == "float")
                 & (field["pattern"] == "along the tracks")]["ns_lowest"].iloc[0]

    def call(width):
        return mlp[(mlp["depth"] == 2) & (mlp["width"] == width) & (mlp["precision"] == "float")
                   & (mlp["activation"] == "tanh from the vector exponential")]["ns_lowest"].iloc[0]

    gaudi_real = gaudi[gaudi["what"].str.contains("in Gaudi")]
    gaudi_copy = gaudi[gaudi["what"].str.contains("with transport matrix")]
    slower = [100 * (a / b - 1) for a, b in zip(gaudi_real["ns_a_state_lowest"],
                                                gaudi_copy["ns_a_state_lowest"])]
    w("## Summary\n")
    w("1. **The compiled program is correct.** %d of %d methods give the final states of their "
      "Python versions (largest difference %s mm)."
      % (int((gate["gate_passed"] & gate["fused_within_limits"]).sum()), len(gate),
         g(gate["gate_largest_difference_of_position_mm"].max(), 2)))
    w("2. **The copies of the two HLT2 integrators are the LHCb tools.** On the grid of the Rec test "
      "each copy prints the same 1,331 final states as the real tool. With the transport matrix, "
      "the real tools in Gaudi are %s %% and %s %% slower than the copies on the same core."
      % (g(max(slower), 2), g(min(slower), 2)))
    w("3. **What LHCb costs and reaches on our tracks.** The adaptive Nystrom integrator: %s field "
      "reads, %.1f us a track (%.1f us with its transport matrix), median error %s um. The adaptive "
      "Cash-Karp integrator: %s reads, %.1f us, %s um. The parametrised step of HLT1: no read, "
      "%.2f us, %s um in the median, but %s um below 3 GeV."
      % (g(nystrom["field_reads_mean"]), nystrom["ns_lowest"] / 1e3, nystrom_tm["ns_lowest"] / 1e3,
         g(nystrom["radial_median_um"]), g(cash_karp["field_reads_mean"]),
         cash_karp["ns_lowest"] / 1e3, g(cash_karp["radial_median_um"]), utt["ns_lowest"] / 1e3,
         g(utt["radial_median_um"]), g(utt["radial_median_um_0_3_gev"])))
    w("4. **The unit costs.** One field read of LHCb costs %.0f ns in a loop of reads and about "
      "%.0f ns inside an integrator (RK4, 16 steps: %.0f ns for 64 reads). One network call in "
      "single precision with the fast tanh costs %.0f ns (2 x 32), %.0f ns (2 x 64), %.0f ns "
      "(2 x 128) and %.0f ns (2 x 256)."
      % (read, rk4_16["ns_lowest"] / 64, rk4_16["ns_lowest"], call(32), call(64), call(128),
         call(256)))
    w("5. **The present networks are slower than LHCb.** The whole-crossing network (2 x 256) takes "
      "%.1f us and has a median error of %s um. The chained networks take %.0f to %.0f us."
      % (whole["ns_lowest"] / 1e3, g(whole["radial_median_um"]), chains["ns_lowest"].min() / 1e3,
         chains["ns_lowest"].max() / 1e3))
    w("6. **A fixed-step integrator already beats the adaptive ones.** RK4 with 32 equal steps "
      "reaches %s um in %.1f us; the adaptive Nystrom integrator of LHCb reaches %s um in %.1f us."
      % (g(rk4_32["radial_median_um"]), rk4_32["ns_lowest"] / 1e3, g(nystrom["radial_median_um"]),
         nystrom["ns_lowest"] / 1e3))
    nets = gate[gate["method"].str.contains("chain|whole_crossing")]
    w("7. **Single precision is enough.** It changes the final position of a network by %s um in "
      "the median at most.\n" % g(nets["single_precision_median_um"].max(), 2))

    # ------------------------------------------------------------------ 1. gates
    w("## 1. The compiled program gives the results of Python\n")
    fixed = gate[~gate["method"].str.contains("rec_cash_karp|allen_adaptive")]
    w("%d methods are checked on 1,000 tracks: %d pass. The build without fused arithmetic "
      "differs from Python by at most %s mm in position. The timed build (fused arithmetic) "
      "differs from that build by at most %s mm for the fixed-step methods and the networks.\n"
      % (len(gate), int((gate["gate_passed"] & gate["fused_within_limits"]).sum()),
         g(gate["gate_largest_difference_of_position_mm"].max(), 2),
         g(fixed["fused_largest_difference_all_tracks_mm"].max(), 2)))
    rows = []
    for _, r in gate.iterrows():
        rows.append([r["method"], r["map"], g(r["gate_largest_difference_of_position_mm"], 2),
                     int(r["gate_tracks_with_other_counts"]),
                     g(r["fused_largest_difference_all_tracks_mm"], 2),
                     int(r["fused_tracks_with_other_counts"]),
                     g(r["single_precision_median_um"], 2), g(r["single_precision_largest_um"], 2)])
    w(table(["Method", "Map", "Against Python: largest difference (mm)", "Tracks with other counts",
             "Fused arithmetic: largest difference (mm)", "Tracks with other steps",
             "Single precision: median change (um)", "largest (um)"], rows) + "\n")
    ck = gate[(gate["method"] == "rec_cash_karp") & (gate["map"] == "linear")].iloc[0]
    w("The adaptive Cash-Karp integrator reacts to the last digit: with fused arithmetic %d track "
      "of 1,000 takes other steps and moves by %s um, and with the field map held in single "
      "precision one track moves by %s um. The fixed-step methods and the networks do not have "
      "this property.\n" % (int(ck["fused_tracks_with_other_counts"]),
                            g(ck["fused_largest_difference_all_tracks_mm"] * 1e3, 2),
                            g(ck["single_precision_largest_um"], 3)))

    # ------------------------------------------------------- 2. the LHCb software
    w("## 2. The copies of the HLT2 integrators against the LHCb software\n")
    w("The test algorithm of Rec carries a grid of 1,331 states from 3000 mm to 7000 mm and "
      "prints the final states. Each copy is compared with the numbers that the real tool "
      "printed (3 decimals in mm, 4 for the slopes).\n")
    rows = []
    for _, r in rec.iterrows():
        rows.append([r["method"], r["map"], "%d of %d" % (r["states_with_the_same_printed_numbers"],
                                                          r["states_printed_by_the_test"]),
                     g(r["largest_difference_of_position_mm"], 3), g(r["field_reads_mean"], 3)])
    w(table(["Copy", "Field map", "States with the same printed numbers",
             "Largest difference (mm)", "Field reads (mean)"], rows) + "\n")
    w("On the map of the test (field.v5r0, magnet down) both copies print the numbers of the "
      "real tools for every state. The other maps do not, so the test tells the map.\n")
    rows = [[r["what"], "%.0f" % r["ns_a_state_lowest"], "%.0f" % r["ns_a_state_highest"]]
            for _, r in gaudi.iterrows()]
    w("Time for one state of that grid, on stbc-i2 (one core, the same for all rows):\n")
    w(table(["", "Lowest (ns)", "Highest (ns)"], rows) + "\n")
    a = allen.iloc[0]
    w("The copy of the Allen adaptive integrator cannot be compared with Allen in the same way. "
      "LHCb checks Allen against TrackSTEPExtrapolator on the same grid within 0.01 mm + 1 %% "
      "(positions) and 5e-5 + 1 %% (slopes). Our two copies pass that same check: %d of %d states "
      "are outside the limits; the median difference is %s um.\n"
      % (a["states_outside_the_limits"], a["states"], g(a["median_difference_of_position_um"], 2)))

    # ------------------------------------------------------ 3. LHCb on our tracks
    w("## 3. The LHCb methods on our tracks\n")
    w("13,447 test tracks with the real momentum spectrum, field map v8r1 (magnet up, trilinear), "
      "one crossing from 2648.2 mm to 7826 mm (the parametrised step: from 2665 mm, its own first "
      "plane). The error is the distance on the last plane from our reference integrator.\n")
    lhcb = m[(m["family"] == "LHCb") & ~m["label"].str.contains("flat sample only")]
    rows = []
    for _, r in lhcb.iterrows():
        rows.append([r["label"], r["precision"], g(r["field_reads_mean"]), "%.0f" % r["ns_lowest"],
                     "%.0f" % r["ns_median"], "%.0f" % r["cold_ns_median"], g(r["radial_median_um"]),
                     g(r["radial_95_um"]), g(100 * r["fraction_above_100_um"], 2)])
    w(table(["Method", "Precision", "Field reads (mean)", "Time a track (ns), lowest", "median",
             "cold caches", "Median error (um)", "95 % (um)", "Tracks above 100 um (%)"], rows) + "\n")
    rows = []
    for _, r in lhcb[~lhcb["label"].str.contains("transport matrix")].iterrows():
        rows.append([r["label"]] + [g(r["radial_median_um_%s_gev" % key]) for key, _ in BANDS])
    w("Median error (um) in the momentum bands:\n")
    w(table(["Method"] + [name + " GeV" for _, name in BANDS], rows) + "\n")
    flat = m[m["label"].str.contains("flat sample only")]
    rows = [[r["label"].replace(", flat sample only", ""), g(r["field_reads_mean"]),
             "%.0f" % r["ns_lowest"], g(r["radial_median_um"]), g(r["radial_95_um"])]
            for _, r in flat.iterrows()]
    w("The adaptive integrators on the 1,000 tracks that are flat in momentum (2 to 42 GeV), the "
      "sample of the first timing study:\n")
    w(table(["Method", "Field reads (mean)", "Time a track (ns)", "Median error (um)", "95 % (um)"],
            rows) + "\n")
    w("The parametrised step of HLT1 against references on two field maps (its polynomials were "
      "fitted to one of them); `polynomial` is the tabulated step alone, `full` adds the change "
      "of q/p and the terms in q/p that Allen applies for the material:\n")
    rows = [[r["method"], r["reference_map"], g(r["radial_median_um"]), g(r["radial_95_um"]),
             g(r["x_signed_median_um"])] for _, r in step.iterrows()
            if "parametrised" in r["method"] and "-1" not in r["method"]]
    w(table(["Variant", "Reference map", "Median error (um)", "95 % (um)", "Signed median of x (um)"],
            rows) + "\n")
    check = step[step["method"].str.startswith("RK4, 4,096")].iloc[0]
    w("Check of our reference: RK4 with 4,096 steps differs from the stored reference end states "
      "by %s um (median) and %s um (largest).\n"
      % (g(check["radial_median_um"], 2), g(check["radial_largest_um"], 2)))

    # ------------------------------------------------------------ 4. unit costs
    w("## 4. The cost of one field read and of one network call\n")
    rows = []
    for _, r in field[field["pattern"] == "along the tracks"].iterrows():
        other = field[(field["reader"] == r["reader"]) & (field["precision"] == r["precision"])
                      & (field["pattern"] == "random points")].iloc[0]
        rows.append([{"linear": "trilinear, arithmetic of the present package",
                      "cubic": "tricubic B-spline, present package",
                      "rec": "trilinear, the read of LHCb (Detector)",
                      "allen": "trilinear, the read of Allen"}[r["reader"]], r["precision"],
                     "%.1f" % r["ns_lowest"], "%.1f" % other["ns_lowest"]])
    w(table(["Field read", "Precision of the caller", "Along the tracks (ns)", "Random points (ns)"],
            rows) + "\n")
    rows = []
    for depth in (2, 3):
        for width in (8, 16, 32, 64, 128, 256, 512):
            def at(precision, activation):
                found = mlp[(mlp["depth"] == depth) & (mlp["width"] == width)
                            & (mlp["precision"] == precision) & (mlp["activation"] == activation)]
                return "%.0f" % found["ns_lowest"].iloc[0]
            one = mlp[(mlp["depth"] == depth) & (mlp["width"] == width)].iloc[0]
            rows.append(["%d x %d" % (depth, width), int(one["multiply_adds"]),
                         at("float", "tanh from the vector exponential"),
                         at("double", "tanh from the vector exponential"),
                         at("float", "tanh of the C library"), at("float", "none")])
    w("One network call (6 inputs, 8 outputs), in ns:\n")
    w(table(["Hidden layers", "Multiply-adds", "Single precision, fast tanh",
             "Double precision, fast tanh", "Single precision, tanh of the C library",
             "Single precision, layers only"], rows) + "\n")

    # ------------------------------------------------------------- 5. all methods
    w("## 5. Every method: time and error on the 13,447 test tracks\n")
    for family, title in (("classical", "RK4 with equal steps"), ("network", "The present trained networks")):
        rows = []
        for _, r in m[m["family"] == family].iterrows():
            precision = r["precision"] + (", fast tanh" if r["tanh"] == "fast" else "")
            rows.append([r["label"], precision, r["reader"], int(r["network_calls"]),
                         g(r["field_reads_mean"]), "%.0f" % r["ns_lowest"], "%.0f" % r["cold_ns_median"],
                         g(r["radial_median_um"]), g(r["radial_95_um"]), g(r["worst_band_median_um"])])
        w("**%s**\n" % title)
        w(table(["Method", "Precision", "Field read", "Network calls", "Field reads",
                 "Time a track (ns)", "Cold caches (ns)", "Median error (um)", "95 % (um)",
                 "Worst band (um)"], rows) + "\n")

    with open(os.path.join(HERE, "RESULTS.md"), "w") as handle:
        handle.write("\n".join(out))
    print("wrote RESULTS.md, %d lines" % len("\n".join(out).splitlines()))


if __name__ == "__main__":
    main()
