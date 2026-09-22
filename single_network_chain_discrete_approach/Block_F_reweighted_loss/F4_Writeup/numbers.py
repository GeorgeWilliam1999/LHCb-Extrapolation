#!/usr/bin/env python
"""F4 - every number quoted in the write-up that no results file already holds.

The write-up rule is that a number in the page comes from a named results file
or from a script saved beside it. Most of them come from the CSV and JSON files
in F0/F1/F2/F3 results/ and from the convergence check rerun into
`F4_Writeup/results/convergence_check.csv`. This script computes the rest, and
writes them to `results/writeup_numbers.json` so the page can be checked
against a file rather than against a transcript.

What it computes:

  growth_along_z     the endpoint radial median of the partial chain at the
                     quarter, the half and the end of the crossing, per run,
                     read from F3's `error_vs_z.csv` (which stores one row per
                     plane); plus chain / single-step, from `error_qdz_chain.csv`
                     and `error_qdz_single_step.csv`.
  wall_per_restart   the median wall-clock seconds per restart of each run, from
                     its own `history.csv`, de-duplicated by restart number the
                     way `Block_G_low_momentum_window/G2_Analysis/convergence.py`
                     does it (the N = 256, q = 16 run was written by two jobs at
                     once from restart 1,198 on).
  run_constants      D_ref, lev_ref and I_bar as each run stored them in
                     `scale.json`, with KAPPA and the lever-arm range, so the
                     worked example in the Method can be checked.
  worked_example     the weight the loss puts on two concrete states: a 3 GeV
                     track at the first output plane of the chain, and a 20 GeV
                     track at the last one, for N = 64, q = 2. Computed by
                     importing F0's own `weighted_loss.py`, not by retyping its
                     formulas.
  momentum_content   the fraction of the test and training tracks in 10-50 GeV
                     and below 5 GeV, from the track dataset.
  starting_x         the case study's endpoint error against the starting |x|,
                     pulled out of `case_study_error_vs_p_x0.csv` for the
                     5-10 GeV row (the band where the |x0| trend is cleanest and
                     every |x0| bin is populated).

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python numbers.py
Out:  results/writeup_numbers.json  (and the same table printed)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys                      # noqa: E402
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
# this file is called numbers.py, which is also a standard-library module that
# numpy imports; drop this folder from the import path before numpy is loaded so
# the standard one is found
for _p in ("", ".", HERE):
    while _p in sys.path:
        sys.path.remove(_p)

import csv                      # noqa: E402
import json                     # noqa: E402

import numpy as np              # noqa: E402

BLOCK = os.path.abspath(os.path.join(HERE, ".."))
F0 = os.path.join(BLOCK, "F0_Weighting")
F1 = os.path.join(BLOCK, "F1_Training", "results", "full")
F3 = os.path.join(BLOCK, "F3_Analysis", "results")
E0 = os.path.abspath(os.path.join(BLOCK, "..", "Block_E_single_network_chain",
                                  "E0_Track_dataset", "results", "tracks.npz"))
RUNS = [(64, 2), (128, 8), (256, 16)]
Z0, Z1 = 2648.2, 7826.0


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------- growth along the crossing --
def growth_along_z():
    """Radial median of the partial chain at a quarter, a half and the end."""
    vz = rows(os.path.join(F3, "error_vs_z.csv"))
    chain = {(int(r["N"]), int(r["q"])): r for r in rows(os.path.join(F3, "error_qdz_chain.csv"))}
    step = {(int(r["N"]), int(r["q"])): r for r in rows(os.path.join(F3, "error_qdz_single_step.csv"))}
    out = {}
    for N, q in RUNS:
        planes = [r for r in vz if int(r["N"]) == N and int(r["q"]) == q]
        planes.sort(key=lambda r: int(r["plane"]))
        assert len(planes) == N + 1, (N, q, len(planes))
        med = [float(r["med_um"]) for r in planes]
        end = med[-1]
        key = "N=%d,q=%d" % (N, q)
        out[key] = dict(
            quarter_um=med[N // 4], half_um=med[N // 2], end_um=end,
            end_over_half=end / med[N // 2], half_over_quarter=med[N // 2] / med[N // 4],
            chain_med_um=float(chain[(N, q)]["med_um"]),
            single_step_med_um=float(step[(N, q)]["step_med_um"]),
            chain_over_single_step=float(chain[(N, q)]["med_um"]) / float(step[(N, q)]["step_med_um"]),
            straight_line_step_med_um=float(step[(N, q)]["straight_line_step_med_um"]),
            step_over_straight_pct=100.0 * float(step[(N, q)]["step_over_straight"]),
        )
    return out


# ------------------------------------------------------ wall clock per restart --
def wall_per_restart():
    """Median seconds per restart, from each run's own de-duplicated history."""
    out = {}
    for N, q in RUNS:
        path = os.path.join(F1, "N%03d_q%02d" % (N, q), "history.csv")
        hist = rows(path)
        field = "wall_s" if "wall_s" in hist[0] else None
        if field is None:
            cand = [k for k in hist[0] if "wall" in k.lower()]
            field = cand[0] if cand else None
        seen = {}
        for r in hist:
            seen[int(r["restart"])] = r                 # last write kept, as G2 does
        w = np.array([float(seen[k][field]) for k in sorted(seen)])
        out["N=%d,q=%d" % (N, q)] = dict(
            history_rows=len(hist), unique_restarts=len(seen),
            duplicated_rows=len(hist) - len(seen),
            wall_field=field,
            median_s_per_restart=float(np.median(w)),
            total_h=float(w.sum() / 3600.0),
        )
    return out


# ------------------------------------------------------ the run's own constants --
def run_constants():
    import importlib.util
    sys.path.insert(0, F0)
    spec = importlib.util.spec_from_file_location("f0_weighted_loss",
                                                  os.path.join(F0, "weighted_loss.py"))
    wl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wl)
    out = dict(KAPPA=float(wl.KAPPA), QOP_TO_GEV=float(wl.QOP_TO_GEV),
               P_LO=wl.P_LO, P_HI=wl.P_HI, ROLLOFF=float(wl.ROLLOFF),
               W_FLOOR=wl.W_FLOOR, CLAMP=wl.CLAMP, runs={})
    for N, q in RUNS:
        sc = json.load(open(os.path.join(F1, "N%03d_q%02d" % (N, q), "scale.json")))
        w = sc.get("weighting", sc)
        dz = (Z1 - Z0) / N
        out["runs"]["N=%d,q=%d" % (N, q)] = dict(
            dz_mm=dz,
            D_ref_mm=w.get("D_ref"), lev_ref_mm=w.get("lev_ref"), i_bar_T_mm=w.get("i_bar"),
            lever_min_mm=dz, lever_max_mm=(Z1 - Z0) + dz,
        )
    return out, wl


# ---------------------------------------------------------- the worked example --
def worked_example(wl):
    """The weight on two concrete states, for N = 64, q = 2, from F0's own code."""
    sc = json.load(open(os.path.join(F1, "N064_q02", "scale.json")))
    w = sc.get("weighting", sc)
    const = dict(D_ref=w["D_ref"], lev_ref=w["lev_ref"], i_bar=w["i_bar"], z1=Z1,
                 L=Z1 - Z0, mode="full", clamp=wl.CLAMP, p_lo=wl.P_LO, p_hi=wl.P_HI,
                 rolloff=wl.ROLLOFF, w_floor=wl.W_FLOOR)
    dz = (Z1 - Z0) / 64.0
    out = {}
    cases = [("3 GeV track, first output plane", 3.0, Z0 + dz),
             ("3 GeV track, last output plane", 3.0, Z1),
             ("20 GeV track, first output plane", 20.0, Z0 + dz),
             ("20 GeV track, last output plane", 20.0, Z1)]
    for name, p_gev, z_out in cases:
        qop = wl.QOP_TO_GEV / p_gev
        D = float(wl.track_bend(np.array([qop]), const["i_bar"], const["L"])[0])
        W = float(wl.band_window(np.array([p_gev]))[0])
        a_unclamped = float(np.sqrt(W) * const["D_ref"] / D)
        lever = max(Z1 - z_out, 0.0) + dz
        out[name] = dict(p_GeV=p_gev, qop=qop, track_bend_D_mm=D, window_W=W,
                         a_unclamped=a_unclamped, lever_mm=lever,
                         weight_on_a_position_residual_per_mm=a_unclamped / const["D_ref"],
                         weight_on_a_slope_residual_per_unit_slope=a_unclamped * lever / const["D_ref"])
    sw = lambda k: out[k]["weight_on_a_slope_residual_per_unit_slope"]   # noqa: E731
    pw = lambda k: out[k]["weight_on_a_position_residual_per_mm"]        # noqa: E731
    out["ratios"] = dict(
        slope_20GeV_first_over_3GeV_first=sw("20 GeV track, first output plane") / sw("3 GeV track, first output plane"),
        slope_20GeV_first_over_20GeV_last=sw("20 GeV track, first output plane") / sw("20 GeV track, last output plane"),
        slope_20GeV_first_over_3GeV_last=sw("20 GeV track, first output plane") / sw("3 GeV track, last output plane"),
        position_20GeV_over_3GeV=pw("20 GeV track, last output plane") / pw("3 GeV track, last output plane"),
    )
    out["D_ref_mm"] = const["D_ref"]
    out["note"] = ("a_n is shown before the per-batch clamp to [1/5, 5] of the batch median; "
                   "the clamp is what stops the >50 GeV tail taking a fifth of the loss")
    return out


# --------------------------------------------------------- momentum content --
def momentum_content():
    D = np.load(E0)
    out = {}
    for split in ("train", "val", "test"):
        P = np.asarray(D["%s_P" % split])
        out[split] = dict(n=int(P.size),
                          frac_10_50_GeV=float(((P >= 10) & (P < 50)).mean()),
                          frac_below_5_GeV=float((P < 5).mean()),
                          frac_5_30_GeV=float(((P >= 5) & (P < 30)).mean()),
                          frac_4_6_GeV=float(((P >= 4) & (P < 6)).mean()))
    out["total_tracks"] = sum(out[s]["n"] for s in ("train", "val", "test"))
    return out


# ------------------------------------------------- the starting-x dependence --
def starting_x():
    """Endpoint |error| against the track's |x| on the last UT plane, 5-10 GeV."""
    r = rows(os.path.join(F3, "case_study_error_vs_p_x0.csv"))
    out = {}
    for p_lo, p_hi in (("5.0", "7.0"), ("10.0", "15.0")):
        for comp in ("x", "y"):
            band = [t for t in r if t["component"] == comp
                    and t["p_lo"] == p_lo and t["p_hi"] == p_hi
                    and t["n"] not in ("0", "") and t["med"] not in ("", "nan")]
            prof = [dict(x0_lo=float(t["x0_lo"]), x0_hi=float(t["x0_hi"]),
                         n=int(t["n"]), med_um=float(t["med"])) for t in band]
            prof = [b for b in prof if np.isfinite(b["med_um"])]
            if not prof:
                continue
            key = "%s_%s-%s_GeV" % (comp, p_lo, p_hi)
            near = min(prof, key=lambda b: abs(0.5 * (b["x0_lo"] + b["x0_hi"])))
            far = max(prof, key=lambda b: abs(0.5 * (b["x0_lo"] + b["x0_hi"])))
            out[key] = dict(profile=prof,
                            near_beamline_um=near["med_um"],
                            near_beamline_bin_mm=[near["x0_lo"], near["x0_hi"]],
                            farthest_um=far["med_um"],
                            farthest_bin_mm=[far["x0_lo"], far["x0_hi"]],
                            ratio_far_over_near=far["med_um"] / near["med_um"])
    out["note"] = ("bins of the track's x on the last UT plane; the case study's network is "
                   "N = 64, q = 2, chosen on validation among the three; endpoint errors")
    return out


# ------------------------------------------- the networks against the truth --
def true_state_ratios():
    """How far the networks are from the Geant4 truth, relative to RK6.

    `against_true_state.csv` gives, per momentum band, the median max(|dx|, |dy|)
    of the network against the true SciFi state, of RK6 against it, and of the
    network against RK6. The two numbers the write-up needs are the ratio of the
    first two (how much worse than the field-only reference the network is, as
    seen from the truth) and the ratio of the RK6-against-truth gap to the
    network-against-RK6 error (how much larger the material effect is than the
    quantity the loss controls).
    """
    F2 = os.path.join(BLOCK, "F2_Analysis", "results", "against_true_state.csv")
    out = {}
    for r in rows(F2):
        band, net = r["band"], r["network"]
        nn_true, rk_true = float(r["nn_vs_true_pos_med_um"]), float(r["rk6_vs_true_pos_med_um"])
        nn_rk6 = float(r["nn_vs_rk6_pos_med_um"])
        out.setdefault(band, {"n": int(r["n"]), "rk6_vs_true_um": rk_true, "networks": {}})
        out[band]["networks"][net] = dict(
            nn_vs_true_um=nn_true, nn_vs_rk6_um=nn_rk6,
            nn_true_over_rk6_true_pct=100.0 * (nn_true / rk_true - 1.0),
            truth_gap_over_nn_vs_rk6=rk_true / nn_rk6)
    bands_no_1_2 = [b for b in out if b != "1-2 GeV"]
    devs = [abs(v["nn_true_over_rk6_true_pct"])
            for b in bands_no_1_2 for v in out[b]["networks"].values()]
    ratios = [v["truth_gap_over_nn_vs_rk6"]
              for b in out for v in out[b]["networks"].values()]
    out["summary"] = dict(
        max_abs_pct_difference_excluding_1_2_GeV=max(devs),
        max_abs_pct_difference_2_to_25_GeV=max(
            abs(v["nn_true_over_rk6_true_pct"])
            for b in ("2-5 GeV", "5-10 GeV", "10-25 GeV") for v in out[b]["networks"].values()),
        truth_gap_over_nn_vs_rk6_min=min(ratios), truth_gap_over_nn_vs_rk6_max=max(ratios),
        note="the 1-2 GeV band holds 5 test tracks and is an indication only")
    return out


def main():
    const, wl = run_constants()
    out = dict(
        generated_for="the reweighted-loss write-up, Block_F_reweighted_loss/F4_Writeup",
        date="2026-09-22",
        growth_along_z=growth_along_z(),
        wall_per_restart=wall_per_restart(),
        constants=const,
        worked_example=worked_example(wl),
        momentum_content=momentum_content(),
        starting_x=starting_x(),
        true_state_ratios=true_state_ratios(),
    )
    res = os.path.join(HERE, "results")
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "writeup_numbers.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
