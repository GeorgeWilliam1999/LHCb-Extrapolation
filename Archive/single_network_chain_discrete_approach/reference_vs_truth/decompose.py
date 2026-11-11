#!/usr/bin/env python
"""Decompose the gap between the field-only RK6 reference and the Geant4 truth.

Three states exist for every magnet crossing in the set:

  S0           the particle's real last-UT state carried to z0 = 2648.2 mm with RK6
               (the start of the crossing; its q/p is the particle's ORIGIN momentum,
               because the truth dump never wrote MCHit::p() -- see
               Official_xdigi/dump_xdigi.py and Data/harvest_states.py);
  truth_zpost  the field-only RK6 reference carried from z0 across the magnet to z1 and
               on to the particle's own first SciFi plane z_post (0.1 mm steps, v8r1 up
               map, q/p held constant: no energy loss, no scattering);
  S_post       the particle's REAL state on that same plane, read off the Geant4 MCHit
               (entry/exit midpoint, slopes from the hit's own displacement). It contains
               everything Geant4 did to the particle and NO detector resolution.

This script measures d = S_post - truth_zpost (true minus reference) on all 14,482
crossings of Block E's track set (training + validation + test concatenated) and splits it
into the two parts that behave differently under a charge flip:

  * a SYSTEMATIC part, visible as a signed median of dx that flips sign with the charge
    while dx / bend does not. A momentum error enters the equation of motion as q/p, so it
    scales the bend: dx / bend ~ dp / p, and dx/bend x P reads directly as a momentum
    deficit in the same units as P;
  * a RANDOM part, the charge-blind 68 % half-width of dx about its own median, which
    falls as 1/p: multiple scattering. It is compared with the Highland formula for the
    air between the two planes, and the width is inverted for the effective thickness in
    radiation lengths that the measured width implies.

Nothing is fitted and nothing is corrected here: the script only measures, and prints the
RK6 self-consistency numbers it quotes beside the measurement so the two can be compared.

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python decompose.py
Out:  results/decomposition_by_band.csv   per momentum band and charge (and pooled)
      results/decomposition_by_pid.csv    per particle type at 5-25 GeV
      results/highland_air.csv            the air-only Highland prediction per band
      results/three_references.csv        network, reference and truth, per band
      results/rk6_self_consistency.csv    the gates and exact-scheme rows quoted beside it
      results/decompose.log               everything printed
      figures/signed_centre_by_charge.png
      figures/width_vs_momentum.png
      figures/three_references.png
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.abspath(os.path.join(HERE, ".."))
PROJECT = os.path.abspath(os.path.join(FOLDER, ".."))

TRACKS = os.path.join(FOLDER, "Block_E_single_network_chain", "E0_Track_dataset",
                      "results", "tracks.npz")
TRACKS_META = os.path.join(FOLDER, "Block_E_single_network_chain", "E0_Track_dataset",
                           "results", "tracks_meta.json")
GATES = os.path.join(PROJECT, "Data_generation_exploration", "Data", "results", "gates.json")
E3_QDZ = os.path.join(FOLDER, "Block_E_single_network_chain", "E3_Analysis",
                      "results", "error_qdz_chain.csv")
F3_QDZ = os.path.join(FOLDER, "Block_F_reweighted_loss", "F3_Analysis",
                      "results", "error_qdz_chain.csv")
E3_COMP = os.path.join(FOLDER, "Block_E_single_network_chain", "E3_Analysis",
                       "results", "comparators.csv")
F3_COMP = os.path.join(FOLDER, "Block_F_reweighted_loss", "F3_Analysis",
                       "results", "comparators.csv")
AGAINST_TRUE = os.path.join(FOLDER, "Block_F_reweighted_loss", "F2_Analysis",
                            "results", "against_true_state.csv")

RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

BANDS = [(1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 25.0),
         (25.0, 50.0), (50.0, 100.0), (100.0, 200.0), (0.0, 1e9)]
BEND_CUT_MM = 20.0          # |bend| below this carries no usable dx/bend ratio
PID_NAMES = {13: "muon", 211: "pion", 321: "kaon", 2212: "proton"}

# Highland, air only, between the two planes
X_AIR_MM = 5177.8           # the crossing length z1 - z0 (read back from the npz below)
X0_AIR_MM = 304.0e3         # radiation length of air, 304 m


class Tee:
    """Print to the terminal and to results/decompose.log at the same time."""

    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def band_label(lo, hi):
    return "all momenta" if hi > 1e8 else "%g-%g GeV" % (lo, hi)


def half_width_68(v):
    """68 % half-width of v about its own median (a median-centred width, not a fit)."""
    if len(v) == 0:
        return float("nan")
    return float(np.quantile(np.abs(v - np.median(v)), 0.68))


def highland_theta0(p_GeV, x_over_X0):
    """Highland's RMS plane scattering angle [rad] for a particle of momentum p.

    theta0 = 13.6 MeV / (beta c p) * z * sqrt(x/X0) * (1 + 0.038 ln(x/X0))

    where p is in GeV (so 13.6 MeV becomes 13.6e-3 GeV), z = 1 for a unit charge and
    beta = 1 for every track in this set (the slowest is 1.5 GeV, far above any mass here).
    """
    return (13.6e-3 / p_GeV) * np.sqrt(x_over_X0) * (1.0 + 0.038 * np.log(x_over_X0))


def highland_displacement_mm(p_GeV, x_over_X0, L_mm):
    """RMS lateral displacement after a length L of scatterer, theta0 * L / sqrt(3).

    A particle that scatters continuously along L ends up displaced by L * theta / sqrt(3)
    on average rather than by L * theta: the deflections earned near the end of the path
    have almost no lever arm left to act over. sqrt(3) is the standard thin-plate result.
    """
    return highland_theta0(p_GeV, x_over_X0) * L_mm / np.sqrt(3.0)


def solve_x_over_X0(width_mm, p_GeV, L_mm):
    """Invert the displacement formula for x/X0: what thickness does this width imply?

    The log term makes it implicit, so it is solved by bisection on log10(x/X0) over a
    range that covers everything from a hundredth of the air estimate to a full
    radiation length.
    """
    def f(t):
        return highland_displacement_mm(p_GeV, t, L_mm) - width_mm

    lo, hi = 1e-5, 1.0
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    tee = Tee(os.path.join(RESULTS, "decompose.log"))
    sys.stdout = tee

    # ---- load, all three splits concatenated -------------------------------
    d = np.load(TRACKS)
    z0 = float(d["z0"])
    z1 = float(d["z1"])
    L = float(d["L"])
    splits = ["train", "val", "test"]
    S0 = np.concatenate([d[s + "_S0"] for s in splits])
    S_post = np.concatenate([d[s + "_S_post"] for s in splits])
    truth_zpost = np.concatenate([d[s + "_truth_zpost"] for s in splits])
    z_post = np.concatenate([d[s + "_z_post"] for s in splits])
    P = np.concatenate([d[s + "_P"] for s in splits])
    PID = np.concatenate([d[s + "_PID"] for s in splits])

    n_all = len(P)
    print("tracks.npz: %s" % TRACKS)
    print("rows: train %d + val %d + test %d = %d"
          % (len(d["train_P"]), len(d["val_P"]), len(d["test_P"]), n_all))
    print("crossing: z0 = %.1f mm, z1 = %.1f mm, L = %.1f mm, field %s, RK6 step %.1f mm"
          % (z0, z1, L, str(d["field"]), float(d["rk6_step_mm"])))
    print("z_post window: |z_post - z1| max %.2f mm (min %.2f mm), selection window %.0f mm"
          % (np.abs(z_post - z1).max(), np.abs(z_post - z1).min(), float(d["window_mm"])))

    # ---- the residual and the bend -----------------------------------------
    # d = true minus reference, on the particle's own SciFi plane
    dS = S_post - truth_zpost                      # (n, 5) in mm and dimensionless slopes
    dx = dS[:, 0] * 1e3                            # um
    dy = dS[:, 1] * 1e3                            # um
    dtx = dS[:, 2] * 1e3                           # slope difference x 1e3
    dty = dS[:, 3] * 1e3
    q = np.sign(S0[:, 4])                          # q/p carries the charge

    # the bend: how far the reference track ends up from where a straight line
    # through the start state would have put it, in the bending plane
    bend = truth_zpost[:, 0] - (S0[:, 0] + S0[:, 2] * (z_post - z0))   # mm
    ratio = (dx * 1e-3) / bend                     # dimensionless, dx and bend both in mm

    print("charges: q+ %d, q- %d" % (int((q > 0).sum()), int((q < 0).sum())))
    print("|bend| median %.1f mm; %d of %d rows have |bend| > %.0f mm"
          % (np.median(np.abs(bend)), int((np.abs(bend) > BEND_CUT_MM).sum()), n_all,
             BEND_CUT_MM))

    # ---- per band and charge ------------------------------------------------
    rows = []
    band_pooled = {}
    for lo, hi in BANDS:
        in_band = (P >= lo) & (P < hi)
        label = band_label(lo, hi)
        for cname, csel in (("q+", q > 0), ("q-", q < 0), ("both", np.ones_like(q, bool))):
            m = in_band & csel
            r = dict(band=label, charge=cname, n=int(m.sum()))
            if m.sum() == 0:
                rows.append(r)
                continue
            r["median_P_GeV"] = float(np.median(P[m]))
            r["median_dx_um"] = float(np.median(dx[m]))
            r["median_dy_um"] = float(np.median(dy[m]))
            r["median_abs_dx_um"] = float(np.median(np.abs(dx[m])))
            r["median_abs_dy_um"] = float(np.median(np.abs(dy[m])))
            r["median_abs_dtx_1e-3"] = float(np.median(np.abs(dtx[m])))
            r["median_abs_dty_1e-3"] = float(np.median(np.abs(dty[m])))
            r["rms_dx_um"] = float(np.sqrt(np.mean(dx[m] ** 2)))
            r["halfwidth68_dx_um"] = half_width_68(dx[m])
            mb = m & (np.abs(bend) > BEND_CUT_MM)
            r["n_bend_cut"] = int(mb.sum())
            r["median_dx_over_bend"] = float(np.median(ratio[mb])) if mb.sum() else float("nan")
            r["median_bend_mm"] = float(np.median(bend[mb])) if mb.sum() else float("nan")
            # the momentum deficit the over-bend implies, in MeV
            r["implied_dp_MeV"] = (r["median_dx_over_bend"] * float(np.median(P[mb])) * 1e3
                                   if mb.sum() else float("nan"))
            if cname == "both":
                mx = np.maximum(np.abs(dx[m]), np.abs(dy[m]))
                r["median_max_dx_dy_um"] = float(np.median(mx))
                r["p95_max_dx_dy_um"] = float(np.quantile(mx, 0.95))
                # the width that measures the RANDOM part alone: each charge is first
                # moved onto its own median, so the charge-antisymmetric systematic
                # cannot inflate the pooled width, and the two charges are then pooled
                pos, neg = dx[in_band & (q > 0)], dx[in_band & (q < 0)]
                if len(pos) and len(neg):
                    centred = np.concatenate([pos - np.median(pos), neg - np.median(neg)])
                    r["halfwidth68_dx_charge_corrected_um"] = float(
                        np.quantile(np.abs(centred), 0.68))
                band_pooled[label] = r
            rows.append(r)

    fields = ["band", "charge", "n", "median_P_GeV", "median_dx_um", "median_dy_um",
              "median_dx_over_bend", "implied_dp_MeV", "halfwidth68_dx_um", "rms_dx_um",
              "median_abs_dx_um", "median_abs_dtx_1e-3", "median_abs_dy_um",
              "median_abs_dty_1e-3", "halfwidth68_dx_charge_corrected_um",
              "median_max_dx_dy_um", "p95_max_dx_dy_um", "median_bend_mm", "n_bend_cut"]
    with open(os.path.join(RESULTS, "decomposition_by_band.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("\n=== true minus field-only reference, per momentum band and charge "
          "(all %d crossings, at the particle's own first SciFi plane) ===" % n_all)
    print("%-12s %-5s %6s %12s %12s %14s %12s %14s"
          % ("band", "q", "n", "med dx [um]", "med dy [um]", "med dx/bend",
             "dp [MeV]", "68% hw [um]"))
    for r in rows:
        if not r.get("n"):
            continue
        print("%-12s %-5s %6d %12.1f %12.1f %14.2e %12.1f %14.1f"
              % (r["band"], r["charge"], r["n"], r["median_dx_um"], r["median_dy_um"],
                 r["median_dx_over_bend"], r["implied_dp_MeV"], r["halfwidth68_dx_um"]))

    print("\n=== pooled over charge: the max metric max(|dx|, |dy|), and the "
          "charge-corrected width ===")
    print("%-14s %6s %16s %16s %24s"
          % ("band", "n", "median [um]", "p95 [um]", "68% half-width dx [um]"))
    for lo, hi in BANDS:
        r = band_pooled.get(band_label(lo, hi))
        if r and r["n"]:
            print("%-14s %6d %16.1f %16.1f %24.1f"
                  % (r["band"], r["n"], r["median_max_dx_dy_um"], r["p95_max_dx_dy_um"],
                     r.get("halfwidth68_dx_charge_corrected_um", float("nan"))))

    # ---- by particle type, 5-25 GeV ----------------------------------------
    sel = (P >= 5.0) & (P < 25.0) & (np.abs(bend) > BEND_CUT_MM)
    pid_rows = []
    for pid in (13, 211, 321, 2212):
        m = sel & (np.abs(PID) == pid)
        pid_rows.append(dict(
            pid=pid, name=PID_NAMES[pid], n=int(m.sum()),
            median_P_GeV=float(np.median(P[m])) if m.sum() else float("nan"),
            median_dx_over_bend=float(np.median(ratio[m])) if m.sum() else float("nan"),
            median_abs_dx_um=float(np.median(np.abs(dx[m]))) if m.sum() else float("nan"),
            implied_dp_MeV=(float(np.median(ratio[m])) * float(np.median(P[m])) * 1e3
                            if m.sum() else float("nan"))))
    with open(os.path.join(RESULTS, "decomposition_by_pid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pid_rows[0].keys()))
        w.writeheader()
        w.writerows(pid_rows)

    print("\n=== by particle type, 5-25 GeV, |bend| > %.0f mm ===" % BEND_CUT_MM)
    print("%-8s %6s %10s %14s %14s %12s"
          % ("type", "n", "med P", "med dx/bend", "med |dx| [um]", "dp [MeV]"))
    for r in pid_rows:
        print("%-8s %6d %10.2f %14.2e %14.1f %12.1f"
              % (r["name"], r["n"], r["median_P_GeV"], r["median_dx_over_bend"],
                 r["median_abs_dx_um"], r["implied_dp_MeV"]))

    # ---- Highland, air only -------------------------------------------------
    x_over_X0_air = L / X0_AIR_MM
    hl_rows = []
    for lo, hi in BANDS:
        r = band_pooled.get(band_label(lo, hi))
        if not r or not r["n"] or hi > 1e8:
            continue
        p = r["median_P_GeV"]
        pred = highland_displacement_mm(p, x_over_X0_air, L) * 1e3     # um
        meas = r["halfwidth68_dx_charge_corrected_um"]
        hl_rows.append(dict(
            band=r["band"], n=r["n"], median_P_GeV=p,
            x_over_X0_air=x_over_X0_air,
            theta0_air_urad=highland_theta0(p, x_over_X0_air) * 1e6,
            highland_air_width_um=pred,
            measured_halfwidth68_dx_um=meas,
            ratio_measured_over_air=meas / pred,
            effective_x_over_X0=solve_x_over_X0(meas * 1e-3, p, L)))
    with open(os.path.join(RESULTS, "highland_air.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hl_rows[0].keys()))
        w.writeheader()
        w.writerows(hl_rows)

    print("\n=== multiple scattering: measured width against Highland for air alone ===")
    print("air between the planes: x/X0 = %.1f mm / %.0f mm = %.5f (%.3f %% of a "
          "radiation length)" % (L, X0_AIR_MM, x_over_X0_air, 100 * x_over_X0_air))
    print("%-12s %10s %14s %16s %10s %18s"
          % ("band", "med P", "air width [um]", "measured 68% [um]", "ratio",
             "effective x/X0"))
    for r in hl_rows:
        print("%-12s %10.2f %14.1f %16.1f %10.2f %18.4f"
              % (r["band"], r["median_P_GeV"], r["highland_air_width_um"],
                 r["measured_halfwidth68_dx_um"], r["ratio_measured_over_air"],
                 r["effective_x_over_X0"]))
    r1025 = next(r for r in hl_rows if r["band"] == "10-25 GeV")
    print("at 10-25 GeV the measured width implies x/X0 = %.4f (%.1f %% of a radiation "
          "length), %.1f times the air alone"
          % (r1025["effective_x_over_X0"], 100 * r1025["effective_x_over_X0"],
             r1025["effective_x_over_X0"] / x_over_X0_air))

    # ---- what RK6 agrees with: the gates and the exact scheme ---------------
    gates = json.load(open(GATES))
    self_rows = [
        dict(what="RK6 re-propagation closure, median", value_um=gates["G1_reprop_closure_mm"]["median"] * 1e3,
             n=gates["G1_reprop_closure_mm"]["n"], source="Data/results/gates.json G1"),
        dict(what="RK6 re-propagation closure, worst", value_um=gates["G1_reprop_closure_mm"]["worst"] * 1e3,
             n=gates["G1_reprop_closure_mm"]["n"], source="Data/results/gates.json G1"),
        dict(what="RK6 step convergence 5 mm vs 1 mm, median", value_um=gates["G3_step_convergence_mm_5vs1"]["median"] * 1e3,
             n=gates["G3_step_convergence_mm_5vs1"]["n"], source="Data/results/gates.json G3"),
        dict(what="RK6 step convergence 5 mm vs 1 mm, worst", value_um=gates["G3_step_convergence_mm_5vs1"]["worst"] * 1e3,
             n=gates["G3_step_convergence_mm_5vs1"]["n"], source="Data/results/gates.json G3"),
        dict(what="label vs the next hit (short legs), median", value_um=gates["G2_label_vs_next_hit_mm"]["median"] * 1e3,
             n=gates["G2_label_vs_next_hit_mm"]["n"], source="Data/results/gates.json G2"),
        dict(what="label vs the next hit (short legs), median above 5 GeV",
             value_um=gates["G2_label_vs_next_hit_mm"]["median_p_gt_5GeV"] * 1e3,
             n=gates["G2_label_vs_next_hit_mm"]["n"], source="Data/results/gates.json G2"),
        dict(what="label vs the next hit (short legs), median below 2 GeV",
             value_um=gates["G2_label_vs_next_hit_mm"]["median_p_lt_2GeV"] * 1e3,
             n=gates["G2_label_vs_next_hit_mm"]["n"], source="Data/results/gates.json G2"),
        dict(what="label vs the next hit (short legs), p95", value_um=gates["G2_label_vs_next_hit_mm"]["p95"] * 1e3,
             n=gates["G2_label_vs_next_hit_mm"]["n"], source="Data/results/gates.json G2"),
    ]
    # the exact collocation scheme at the settings that were trained: these rows live in
    # error_qdz_chain.csv (column exact_med_um), not in comparators.csv
    for path, tag in ((E3_QDZ, "E3_Analysis/results/error_qdz_chain.csv"),
                      (F3_QDZ, "F3_Analysis/results/error_qdz_chain.csv")):
        for row in csv.DictReader(open(path)):
            if (int(row["N"]), int(row["q"])) in ((64, 2), (128, 8), (256, 16)):
                self_rows.append(dict(
                    what="exact scheme, N = %s steps, q = %s stages, dz = %s mm, endpoint "
                         "radial median" % (row["N"], row["q"], row["dz_mm"]),
                    value_um=float(row["exact_med_um"]), n=int(row["n"]), source=tag))
    for path, tag in ((E3_COMP, "E3_Analysis/results/comparators.csv"),
                      (F3_COMP, "F3_Analysis/results/comparators.csv")):
        for row in csv.DictReader(open(path)):
            self_rows.append(dict(what="comparator: %s, endpoint radial median" % row["what"],
                                  value_um=float(row["med_um"]), n=int(row["n"]), source=tag))
    with open(os.path.join(RESULTS, "rk6_self_consistency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["what", "value_um", "n", "source"])
        w.writeheader()
        w.writerows(self_rows)

    print("\n=== what the reference does agree with (its own accuracy) ===")
    for r in self_rows:
        print("%-78s %14.4g um  (n = %d)  [%s]" % (r["what"], r["value_um"], r["n"], r["source"]))

    # ---- the three references, and the two ratios the page quotes ----------
    at_rows = list(csv.DictReader(open(AGAINST_TRUE)))
    tr_rows = []
    for r in at_rows:
        rk = float(r["rk6_vs_true_pos_med_um"])
        nt = float(r["nn_vs_true_pos_med_um"])
        nr = float(r["nn_vs_rk6_pos_med_um"])
        tr_rows.append(dict(
            network=r["network"], band=r["band"], n=int(r["n"]),
            nn_vs_rk6_pos_med_um=nr, nn_vs_true_pos_med_um=nt, rk6_vs_true_pos_med_um=rk,
            nn_vs_true_over_rk6_vs_true_pct=100.0 * (nt / rk - 1.0),
            rk6_vs_true_over_nn_vs_rk6=rk / nr))
    with open(os.path.join(RESULTS, "three_references.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tr_rows[0].keys()))
        w.writeheader()
        w.writerows(tr_rows)

    print("\n=== the three references (test tracks): how close the networks are to the "
          "reference, and how far both are from the truth ===")
    print("%-28s %-11s %6s %12s %12s %12s %10s %10s"
          % ("network", "band", "n", "vs ref", "vs true", "ref vs true",
             "diff [%]", "ratio"))
    for r in tr_rows:
        print("%-28s %-11s %6d %12.1f %12.1f %12.1f %10.1f %10.1f"
              % (r["network"], r["band"], r["n"], r["nn_vs_rk6_pos_med_um"],
                 r["nn_vs_true_pos_med_um"], r["rk6_vs_true_pos_med_um"],
                 r["nn_vs_true_over_rk6_vs_true_pct"], r["rk6_vs_true_over_nn_vs_rk6"]))
    big = [r for r in tr_rows if r["band"] != "1-2 GeV"]
    print("excluding the 1-2 GeV band (5 test tracks): network-vs-truth differs from "
          "reference-vs-truth by %.1f to %+.1f %%"
          % (min(r["nn_vs_true_over_rk6_vs_true_pct"] for r in big),
             max(r["nn_vs_true_over_rk6_vs_true_pct"] for r in big)))
    print("                              reference-vs-truth is %.0f to %.0f times the "
          "network-vs-reference error"
          % (min(r["rk6_vs_true_over_nn_vs_rk6"] for r in big),
             max(r["rk6_vs_true_over_nn_vs_rk6"] for r in big)))
    print("including it: %.1f to %+.1f %% and %.0f to %.0f times"
          % (min(r["nn_vs_true_over_rk6_vs_true_pct"] for r in tr_rows),
             max(r["nn_vs_true_over_rk6_vs_true_pct"] for r in tr_rows),
             min(r["rk6_vs_true_over_nn_vs_rk6"] for r in tr_rows),
             max(r["rk6_vs_true_over_nn_vs_rk6"] for r in tr_rows)))

    # ---- figures ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [band_label(lo, hi) for lo, hi in BANDS
              if hi < 1e8 and band_pooled.get(band_label(lo, hi))
              and band_pooled[band_label(lo, hi)]["n"]]
    xi = np.arange(len(labels))
    by = {(r["band"], r["charge"]): r for r in rows}

    # 1. the signed centre, split by charge
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax[0].axhline(0.0, color="0.6", lw=1)
    for cname, style, col in (("q+", "-o", "tab:blue"), ("q-", "-s", "tab:red")):
        ax[0].plot(xi, [by[(b, cname)]["median_dx_um"] * 1e-3 for b in labels], style,
                   color=col, label="positive tracks" if cname == "q+" else "negative tracks")
    ax[0].set_xticks(xi)
    ax[0].set_xticklabels(labels, rotation=30, ha="right")
    ax[0].set_ylabel("median of true minus reference in x [mm]")
    ax[0].set_title("The signed centre flips sign with the charge", fontsize=10)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].axhline(0.0, color="0.6", lw=1)
    for cname, style, col in (("q+", "-o", "tab:blue"), ("q-", "-s", "tab:red")):
        ax[1].plot(xi, [by[(b, cname)]["median_dx_over_bend"] * 1e3 for b in labels], style,
                   color=col, label="positive tracks" if cname == "q+" else "negative tracks")
    ax[1].set_xticks(xi)
    ax[1].set_xticklabels(labels, rotation=30, ha="right")
    ax[1].set_ylabel("median of (true minus reference in x) / bend  [x 1e-3]")
    ax[1].set_title("Divided by the bend it does not: a relative over-bend", fontsize=10)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.suptitle("Geant4-true SciFi state minus the field-only reference, all %d crossings"
                 % n_all, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "signed_centre_by_charge.png"), dpi=150)
    plt.close(fig)

    # 2. the width against Highland
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pm = [band_pooled[b]["median_P_GeV"] for b in labels]
    ax.plot(pm, [band_pooled[b]["halfwidth68_dx_charge_corrected_um"] for b in labels],
            "k-o", lw=2,
            label="measured 68 % half-width of the residual in x")
    pg = np.logspace(np.log10(min(pm)), np.log10(max(pm)), 100)
    ax.plot(pg, highland_displacement_mm(pg, x_over_X0_air, L) * 1e3, "--",
            color="tab:blue",
            label="scattering in the air between the planes (%.2f %% of a radiation length)"
                  % (100 * x_over_X0_air))
    x_eff = r1025["effective_x_over_X0"]
    ax.plot(pg, highland_displacement_mm(pg, x_eff, L) * 1e3, "-.", color="tab:orange",
            label="scattering in %.1f %% of a radiation length (the thickness the\n10-25 GeV width implies)"
                  % (100 * x_eff))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("momentum [GeV]")
    ax.set_ylabel("68 % half-width of true minus reference in x [um]")
    ax.set_title("The random part of the gap falls as 1/p, as scattering does", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "width_vs_momentum.png"), dpi=150)
    plt.close(fig)

    # 3. the three references, redrawn from F2's table (same drawing, own file)
    at = at_rows
    nets = list(dict.fromkeys(r["network"] for r in at))
    bnds = [b for b in dict.fromkeys(r["band"] for r in at) if b != "all"]
    xj = np.arange(len(bnds))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(xj, [float(next(r for r in at if r["band"] == b)["rk6_vs_true_pos_med_um"])
                 for b in bnds], "k-o", lw=2.2, ms=6,
            label="field-only reference vs the Geant4-true state")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, net in enumerate(nets):
        rr = [r for r in at if r["network"] == net and r["band"] != "all"]
        ax.plot(xj, [float(r["nn_vs_true_pos_med_um"]) for r in rr], "-s", color=colors[i],
                ms=5, alpha=0.9, label="%s vs the Geant4-true state" % net)
        ax.plot(xj, [float(r["nn_vs_rk6_pos_med_um"]) for r in rr], "--^", color=colors[i],
                ms=5, alpha=0.9, label="%s vs the field-only reference" % net)
    ax.set_yscale("log")
    ax.set_xticks(xj)
    ax.set_xticklabels(bnds)
    ax.set_xlabel("momentum band")
    ax.set_ylabel("median max(|dx|, |dy|) at the first SciFi plane [um]")
    ax.set_title("Three references for one crossing, endpoint error on the 1,452 test tracks",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "three_references.png"), dpi=150)
    plt.close(fig)

    print("\nwrote results/decomposition_by_band.csv, results/decomposition_by_pid.csv,")
    print("      results/highland_air.csv, results/rk6_self_consistency.csv,")
    print("      figures/signed_centre_by_charge.png, figures/width_vs_momentum.png,")
    print("      figures/three_references.png")
    tee.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
