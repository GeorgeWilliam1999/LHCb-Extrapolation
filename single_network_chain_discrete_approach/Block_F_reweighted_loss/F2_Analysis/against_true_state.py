#!/usr/bin/env python
"""F2 - the networks against the Geant4-true SciFi state, beside RK6 against it.

Three references exist for a crossing:
  RK6 endpoint      the field-only reference (order-6 Runge-Kutta, 0.1 mm) from the
                    particle's real last-UT state; what every network is trained
                    towards (label-free) and scored against elsewhere in F2/F3;
  true SciFi state  the particle's real state on its own first SciFi plane, from the
                    Geant4 MCHit (entry/exit midpoint; slopes = the hit's own
                    displacement / dz).  It contains everything Geant4 did to the
                    particle between the two planes - multiple scattering, energy loss -
                    and NO detector resolution (no digitisation, no pattern recognition);
  network endpoint  the network applied N times from the same real last-UT state.

Every run's record.json already scores the chain against both (metrics.chain_scores,
Block E's E1): the network's z1 state and the RK6 truth are each carried from z1 to the
particle's own SciFi plane with RK6 (|z_post - z1| < 60 mm) and compared with the true
state there.  This script only tabulates those numbers, per momentum band, for the
reweighted-loss networks, and draws them.  Position error = max(|dx|, |dy|) at the SciFi
plane; slope error = max(|dtx|, |dty|) x 1e3 (a slope difference in units of 1e-3,
labelled mrad in the records; tx = dx/dz is dimensionless, no arctan is applied).

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python against_true_state.py
Out:  results/against_true_state.csv, figures/against_true_state.png
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "F1_Training", "results", "full")
OUT_CSV = os.path.join(HERE, "results", "against_true_state.csv")
OUT_FIG = os.path.join(HERE, "figures", "against_true_state.png")
KEYS = [("nn_vs_rk6", "vs_rk6_endpoint"), ("nn_vs_true", "vs_real_scifi_state"),
        ("rk6_vs_true", "rk6_truth_vs_real_scifi_state")]


def name(rec):
    return "N = %d, q = %d, dz = %.0f mm" % (rec["N"], rec["q"], rec["dz_mm"])


def main():
    rows, recs = [], []
    for d in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, d, "record.json")
        if not os.path.exists(p):
            continue
        rec = json.load(open(p))
        t = rec["test"]
        recs.append((rec, t))
        bands = ["all"] + list(t["vs_rk6_endpoint"]["by_p_band"].keys())
        for band in bands:
            row = dict(network=name(rec), N=rec["N"], q=rec["q"], restarts=rec["restarts"], band=band)
            for short, key in KEYS:
                b = t[key] if band == "all" else t[key]["by_p_band"][band]
                row["n"] = b["n"]
                row[short + "_pos_med_um"] = b["pos_med_um"]
                row[short + "_pos_p95_um"] = b["pos_p95_um"]
                row[short + "_slope_med_1e-3"] = b["slope_med_mrad"]
            rows.append(row)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("test tracks, position error max(|dx|,|dy|) at the particle's own first SciFi plane, median [um]")
    print("%-28s %-11s %5s %14s %14s %14s" % ("network", "band", "n", "NN vs RK6", "NN vs true", "RK6 vs true"))
    for r in rows:
        print("%-28s %-11s %5d %14.1f %14.1f %14.1f" % (r["network"], r["band"], r["n"], r["nn_vs_rk6_pos_med_um"],
                                                     r["nn_vs_true_pos_med_um"], r["rk6_vs_true_pos_med_um"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bands = [b for b in dict.fromkeys(r["band"] for r in rows) if b != "all"]
    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    rk = [next(r for r in rows if r["band"] == b)["rk6_vs_true_pos_med_um"] for b in bands]
    ax.plot(x, rk, "k-o", lw=2.2, ms=6, label="RK6 (field only) vs Geant4-true state")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (rec, t) in enumerate(recs):
        rr = [r for r in rows if r["network"] == name(rec) and r["band"] != "all"]
        ax.plot(x, [r["nn_vs_true_pos_med_um"] for r in rr], "-s", color=colors[i], ms=5, alpha=0.9,
                label="%s vs Geant4-true state" % name(rec))
        ax.plot(x, [r["nn_vs_rk6_pos_med_um"] for r in rr], "--^", color=colors[i], ms=5, alpha=0.9,
                label="%s vs RK6" % name(rec))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_xlabel("momentum band")
    ax.set_ylabel("median max(|dx|, |dy|) at the first SciFi plane [um]")
    ax.set_title("One network per step length, chained across the magnet: three references (test tracks)", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print("wrote", os.path.relpath(OUT_CSV, HERE), os.path.relpath(OUT_FIG, HERE))


if __name__ == "__main__":
    sys.exit(main())
