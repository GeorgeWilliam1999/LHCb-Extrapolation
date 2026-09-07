#!/usr/bin/env python
"""Which field polarity does the official sample actually have?

This script exists because building C0 forced the question, and the answer was
not the one the repository had been assuming.

Every label in the training set, and every experiment in
`../Raissi_disc_time_approach` before Block C, was built with the **MagDown**
v8r1 map. The cross-magnet legs make that assumption testable for the first
time, because a leg-B pair gives both halves of the test for free: the forward
row starts at the particle's last UT plane, and the backward row of the SAME
particle starts at that particle's ACTUAL first SciFi plane state. So the
forward row can be integrated with each polarity and the answer compared with
where the particle really was.

The test is not a subtle one. Under MagDown the bend across the magnet comes
out with the WRONG SIGN for every particle in the sample, and the endpoint
misses the truth by of order a metre; under MagUp the residual collapses onto
the 1/p multiple-scattering line that the training set's own G2 gate reports
for the short legs. The sample's conditions tag is `sim-20231017-vc-mu100`.

What is measured

  1. The two maps are exact negatives of one another (a cheap re-check of
     `../Magnet_up_field`'s A4.1 finding, on random in-map points).
  2. Forward leg-B row, integrated with each polarity, against the particle's
     real first-SciFi-plane state; and the mirror test, backward row against
     the real last-UT-plane state. Per momentum band: median, p95, and the
     fraction of particles whose BEND SIGN comes out right.
  3. What each polarity does to the fiducial requirement - how many legs leave
     the field map when propagated through each.
  4. A figure: the residual against momentum for both polarities, the sign
     agreement, and one real track drawn in x-z with its hits and both
     polarities' paths.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python check_polarity.py

Outputs:
    results/polarity_check.json
    figures/field_polarity.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import json

import numpy as np

import use_shared                                            # noqa: F401
from _shared.prepare import (P_BANDS, _particle_key, magnet_leg_rows,
                             p_band_index)
from _shared.reference import (field_bounds, field_md5, field_path, make_field,
                               rk4_rows, rk6_dense_rows, rk6_rows)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
STATES_NPZ = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "Data_generation_exploration",
    "Official_xdigi", "results", "states.npz")


def pair_up(sel):
    """Match every particle's forward leg-B row with its backward one."""
    key = _particle_key(sel["EVT"], sel["MCKEY"])
    fwd = sel["DIRECTION"] > 0
    order = np.lexsort((~fwd, key))
    k = key[order]
    assert np.array_equal(k[0::2], k[1::2]), "rows are not in matched pairs"
    fi, bi = order[0::2], order[1::2]
    assert fwd[fi].all() and (~fwd[bi]).all()
    return fi, bi


def residuals(sel, fi, bi, field, step=5.0):
    """Endpoint of each direction against the particle's real state there."""
    out = {}
    for name, a, b in (("forward: UT -> first SciFi plane", fi, bi),
                       ("backward: first SciFi plane -> UT", bi, fi)):
        S1 = rk4_rows(sel["S0"][a], sel["z0"][a], sel["z1"][a], step=step,
                      field=field)
        truth = sel["S0"][b]
        gap = np.abs(S1[:, :2] - truth[:, :2]).max(axis=1)
        dtx_ref = S1[:, 2] - sel["S0"][a][:, 2]
        dtx_true = truth[:, 2] - sel["S0"][a][:, 2]
        out[name] = {"gap_mm": gap, "sign_ok": np.sign(dtx_ref) == np.sign(dtx_true)}
    return out


def band_table(gap, sign_ok, band, P):
    rows = []
    for b, (lo, hi) in enumerate(P_BANDS):
        m = band == b
        if not m.any():
            continue
        rows.append({"p_band_GeV": [lo, hi], "n": int(m.sum()),
                     "median_mm": float(np.median(gap[m])),
                     "p95_mm": float(np.quantile(gap[m], 0.95)),
                     "bend_sign_correct_fraction": float(sign_ok[m].mean())})
    rows.append({"p_band_GeV": "all", "n": int(len(gap)),
                 "median_mm": float(np.median(gap)),
                 "p95_mm": float(np.quantile(gap, 0.95)),
                 "bend_sign_correct_fraction": float(sign_ok.mean())})
    return rows


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    out = {"question": "which v8r1 polarity did the official minbias sample "
                       "expected_2024_minbias_xdigi run with?",
           "conditions_tag": "sim-20231017-vc-mu100 (DDDB dddb-20231017)",
           "why_it_matters": "every label in "
                             "Data_generation_exploration/Official_xdigi/"
                             "training_v2 was built with the MagDown map"}

    # ---- 1. the two maps ---------------------------------------------------
    up, down = make_field("up"), make_field("down")
    rng = np.random.default_rng(0)
    lo, hi = field_bounds(down)
    pts = rng.uniform(lo, hi, size=(200_000, 3))
    bu = np.stack(up(pts[:, 0], pts[:, 1], pts[:, 2]), axis=1)
    bd = np.stack(down(pts[:, 0], pts[:, 1], pts[:, 2]), axis=1)
    out["maps"] = {
        "up": {"file": field_path("up"), "md5": field_md5("up")},
        "down": {"file": field_path("down"), "md5": field_md5("down")},
        "max_abs_B_up_plus_B_down_T": float(np.abs(bu + bd).max()),
        "n_points": 200_000,
        "meaning": "0.0 means the maps are exact negatives, so the only thing "
                   "the polarity changes is the sign of the bend",
    }
    print("maps are exact negatives to %.3g T" %
          out["maps"]["max_abs_B_up_plus_B_down_T"])

    # ---- 2. the propagation test -------------------------------------------
    sel = magnet_leg_rows(verbose=True)
    fi, bi = pair_up(sel)
    band = p_band_index(sel["P"][fi])
    out["population"] = {
        "legs": int(len(sel["P"])), "particles": int(len(fi)),
        "selection": "magnet_leg_rows(): leg B, UT plane, both directions, "
                     "2 < eta < 5, 1 < p < 200 GeV, non-electron"}
    out["propagation_test"] = {
        "what": "the leg integrated with each polarity (fp64 RK4, 5 mm), "
                "compared with the particle's OWN truth state at the far "
                "plane, which is the start state of its other-direction row",
        "by_polarity": {}}
    keep = {}
    for name, field in (("up", up), ("down", down)):
        res = residuals(sel, fi, bi, field)
        keep[name] = res
        out["propagation_test"]["by_polarity"][name] = {
            direction: band_table(v["gap_mm"], v["sign_ok"], band,
                                  sel["P"][fi])
            for direction, v in res.items()}
        for direction, v in res.items():
            print("  %-5s %-38s median %9.3f mm  sign right %.3f"
                  % (name, direction, np.median(v["gap_mm"]),
                     v["sign_ok"].mean()))

    # ---- 3. what it does to the fiducial requirement -----------------------
    sub = np.sort(np.random.default_rng(1).permutation(len(sel["P"]))[:4000])
    out["fiducial_effect"] = {
        "what": "legs whose RK6 path (20 mm screening step) leaves the field "
                "map, on a random 4000-leg subset",
        "n": int(len(sub))}
    for name, field in (("up", up), ("down", down)):
        _, Sg, valid = rk6_dense_rows(sel["S0"][sub], sel["z0"][sub],
                                      sel["z1"][sub], sample_mm=20.0,
                                      step=20.0, field=field)
        bad = (((Sg[:, :, 0] < lo[0]) | (Sg[:, :, 0] > hi[0])
                | (Sg[:, :, 1] < lo[1]) | (Sg[:, :, 1] > hi[1])) & valid
               ).any(axis=1)
        bsub = p_band_index(sel["P"][sub])
        out["fiducial_effect"][name] = {
            "legs_leaving_the_map": int(bad.sum()),
            "fraction": float(bad.mean()),
            "by_p_band": {"%g-%g" % P_BANDS[b]: float(bad[bsub == b].mean())
                          for b in range(len(P_BANDS))
                          if (bsub == b).any()},
            "by_direction": {
                "UT -> SciFi": float(bad[sel["DIRECTION"][sub] > 0].mean()),
                "SciFi -> UT": float(bad[sel["DIRECTION"][sub] < 0].mean())},
        }
        print("  %-5s legs leaving the map: %d / %d"
              % (name, bad.sum(), len(sub)))

    up_med = out["propagation_test"]["by_polarity"]["up"][
        "forward: UT -> first SciFi plane"][-1]["median_mm"]
    dn_med = out["propagation_test"]["by_polarity"]["down"][
        "forward: UT -> first SciFi plane"][-1]["median_mm"]
    out["verdict"] = (
        "MagUp. The forward cross-magnet leg lands %.2f mm from the particle's "
        "real first-SciFi state with the up map and %.0f mm with the down map, "
        "and the sign of the bend is right for %.1f%% of particles with up "
        "against %.1f%% with down. The up-map residual falls as 1/p, which is "
        "the multiple-scattering line the training set's own G2 gate reports "
        "for the short legs - i.e. exactly the material effect that is out of "
        "the label's scope by design. The down-map residual is the deflection "
        "itself, doubled."
        % (up_med, dn_med,
           100 * out["propagation_test"]["by_polarity"]["up"][
               "forward: UT -> first SciFi plane"][-1][
                   "bend_sign_correct_fraction"],
           100 * out["propagation_test"]["by_polarity"]["down"][
               "forward: UT -> first SciFi plane"][-1][
                   "bend_sign_correct_fraction"]))
    out["consequence"] = [
        "the v2 training set's labels (train_official_v2.npz) do not describe "
        "the events they were harvested from; the Y column is the field-only "
        "propagation of the right start state through the wrong polarity",
        "experiments that only ever compared a network with that same engine "
        "(One_step_network_v2, General_leg_network, Chained_legs, ...) are "
        "internally consistent and their conclusions about how well a network "
        "solves the scheme stand; what does not stand is any claim that those "
        "labels are where the simulated particle went",
        "../Magnet_up_field is framed as 'a polarity for which no labelled "
        "sample has ever been produced'. It is in fact the polarity of the "
        "sample, and its own A4.2 table already shows the tell: the fiducial "
        "cut removes 0 states on MagUp and 21/20/15 on MagDown",
        "Block C therefore builds on MagUp, and records this file as the "
        "reason. THIS NEEDS GEORGE'S DECISION for the rest of the line.",
    ]
    with open(os.path.join(RESULTS, "polarity_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n" + out["verdict"])

    # ---- 4. the figure ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    P = sel["P"][fi]
    for name, colour in (("up", "#1f77b4"), ("down", "#d62728")):
        g = keep[name]["forward: UT -> first SciFi plane"]["gap_mm"]
        ax.plot(P, g, ".", ms=1.5, alpha=0.25, color=colour, rasterized=True)
        meds = [np.median(g[band == b]) for b in range(len(P_BANDS))
                if (band == b).any()]
        cent = [np.sqrt(P_BANDS[b][0] * P_BANDS[b][1])
                for b in range(len(P_BANDS)) if (band == b).any()]
        ax.plot(cent, meds, "o-", color=colour, lw=2, ms=6,
                label="v8r1.%s (band medians)" % name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("truth momentum p [GeV]")
    ax.set_ylabel("|label - the particle's real SciFi state| [mm]")
    ax.set_title("Forward cross-magnet leg against reality")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    w = 0.35
    xs = np.arange(len(P_BANDS))
    for i, (name, colour) in enumerate((("up", "#1f77b4"), ("down", "#d62728"))):
        s = keep[name]["forward: UT -> first SciFi plane"]["sign_ok"]
        vals = [s[band == b].mean() if (band == b).any() else np.nan
                for b in range(len(P_BANDS))]
        ax.bar(xs + (i - 0.5) * w, vals, w, color=colour,
               label="v8r1.%s" % name)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%g-%g" % b for b in P_BANDS], fontsize=8)
    ax.set_xlabel("momentum band [GeV]")
    ax.set_ylabel("fraction with the bend sign correct")
    ax.set_ylim(0, 1.05)
    ax.set_title("Does the track bend the right way?")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # one real track, its hits, and both polarities' paths
    ax = axes[2]
    st = np.load(STATES_NPZ, allow_pickle=True)
    skey = st["evt"].astype(np.int64) * 10_000_000 + st["mc_key"]
    pk = _particle_key(sel["EVT"], sel["MCKEY"])
    cand = np.flatnonzero((sel["DIRECTION"] > 0) & (sel["P"] > 3.0)
                          & (sel["P"] < 6.0))
    i = cand[0]
    m = skey == pk[i]
    zz, xx = st["z"][m], st["x"][m]
    o = np.argsort(zz)
    ax.plot(zz[o], xx[o], "k.-", ms=7, lw=1.0, label="the particle's own hits")
    for name, field, colour in (("up", up, "#1f77b4"), ("down", down, "#d62728")):
        Zg, Sg, V = rk6_dense_rows(sel["S0"][i:i + 1], sel["z0"][i:i + 1],
                                   sel["z1"][i:i + 1], sample_mm=50.0,
                                   step=0.5, field=field)
        ax.plot(Zg[0][V[0]], Sg[0][V[0], 0], "-", lw=2, color=colour,
                label="RK6 through v8r1.%s" % name)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")
    ax.set_title("One %.1f GeV track (EVT %d, MCKEY %d)"
                 % (sel["P"][i], sel["EVT"][i], sel["MCKEY"][i]))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("The official minbias sample is MagUp, not MagDown "
                 "(%d particles)" % len(fi), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES, "field_polarity.png"), dpi=140)
    plt.close(fig)
    print("wrote figures/field_polarity.png")


if __name__ == "__main__":
    main()
