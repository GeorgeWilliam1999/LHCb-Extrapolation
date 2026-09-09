#!/usr/bin/env python
"""C3.1 - the residual scales for the magnet-to-magnet set, and their check.

`../C0_Magnet_tracks_dataset/results/magnet_tracks_v3.npz` is read and written
back out unchanged as `results/magnet_tracks_v3_residual.npz` with three
further arrays, computed with the **MagUp** map (the sample's own polarity):

    IB           (N,)  I_B = INT |B| dz along the straight line   [T mm]
    SCALE_POS    (N,)  kappa |qop| I_B |dz| / 2, floored at 1e-9  [mm]
    SCALE_SLOPE  (N,)  kappa |qop| I_B, floored at 1e-12          [-]

plus the two floors and the number of field samples as scalars, so the file
carries its own definition. The model recomputes the same numbers from
(S, extra) at forward time; storing them makes the dataset self-describing and
this check reproducible, and the torch twin is compared against the numpy
function below.

## The check the scales have to pass

The network emits `net_j = (reference_j - straight_j) / scale_j`, which is only
a well-conditioned target if it is O(1). The residual redesign
(`../../Block_A_technique_works/A3a_General_leg_network/README_residual.md`) established that on three fixed
leg types; this dataset spans |dz| from 0.05 mm to 5.2 m in six strata, four
orders of magnitude wider, so the question has to be asked again **per
stratum**. This script forms exactly that ratio at the endpoint, using the RK6
reference `Y` already in the dataset - the only place a label appears, and it
is a diagnostic, never an input to the scale - and histograms it per stratum.

The endpoint is the only node at which the ratio can be formed here, because
`magnet_tracks_v3.npz` stores the end state and not the Gauss nodes (those are
q-dependent and are built per q by `prepare_nodes.py`). At the endpoint the
"flat" and "poly" node profiles of the residual design coincide - the last
entry of `cout` is 1 by construction - so there is no profile to choose
between and the flat profile the residual wave selected is used.

## The floors

The residual design floored the position scale at 1e-3 mm and the slope scale
at 1e-6, which is sensible for legs of 70 mm and up. Stratum 0 here is
|dz| ~ 0.1 mm, where the true deviation from a straight line is of order 1e-7
mm - four orders of magnitude below the old floor. This script uses the
lowered floors of `grid_model.py` (1e-9 mm, 1e-12) and reports, per stratum,
how many samples are still floored; that count must be zero or near it, or the
floor and not the field would be setting the scale.

## The straight-line column

C4 needs to know where the straight line is *already* good enough, so for each
stratum this script also records the straight-line endpoint error against the
RK6 reference - median, p95 and the fraction below one micrometre. A stratum in
which a straight line is already sub-micrometre is one in which no network can
show an improvement worth reporting, and that has to be visible before the
grid is read.

Outputs
    results/magnet_tracks_v3_residual.npz
    results/scale_check.json
    figures/scale_check.png
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json

import numpy as np

import use_shared                                        # noqa: F401
from _shared.prepare import STRATUM_NAMES
from _shared.reference import KAPPA, field_md5, field_path, make_field
from grid_model import (FIELD_WHICH, N_FIELD_SAMPLES, POS_FLOOR_MM,
                        SLOPE_FLOOR, field_integral_numpy,
                        scales_from_field_integral, straight_line_endpoint)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
SOURCE = os.path.join(HERE, "..", "C0_Magnet_tracks_dataset", "results",
                      "magnet_tracks_v3.npz")
SPLIT_NAMES = ("train", "val", "test")
OK_LO, OK_HI = 0.1, 10.0


def quantiles(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {}
    return {"n": int(len(a)),
            "p05": float(np.quantile(a, 0.05)),
            "median": float(np.median(a)),
            "p95": float(np.quantile(a, 0.95)),
            "max": float(a.max())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--field", choices=("down", "up"), default=FIELD_WHICH)
    ap.add_argument("--out", default=os.path.join(RESULTS,
                                                  "magnet_tracks_v3_residual.npz"))
    a = ap.parse_args(argv)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    d = np.load(os.path.abspath(a.source))
    data = {k: d[k] for k in d.files}
    X = np.asarray(data["X"], dtype=np.float64)
    Y = np.asarray(data["Y"], dtype=np.float64)
    STRAT = np.asarray(data["STRATUM"])
    DIR = np.asarray(data["DIRECTION"])
    SPLIT = np.asarray(data["SPLIT"])
    S, z0, dz = X[:, :5], X[:, 5], X[:, 6]

    fld = make_field(a.field)
    IB = field_integral_numpy(S, z0, dz, fld)
    pos, slope = scales_from_field_integral(S, dz, IB)
    data["IB"] = IB
    data["SCALE_POS"] = pos
    data["SCALE_SLOPE"] = slope
    data["residual_pos_floor_mm"] = np.array(POS_FLOOR_MM)
    data["residual_slope_floor"] = np.array(SLOPE_FLOOR)
    data["residual_n_field_samples"] = np.array(N_FIELD_SAMPLES)
    data["residual_node_profile"] = np.array("flat")
    data["residual_field"] = np.array(a.field)

    # -- the ratio the network has to emit, and the straight line ------------
    straight = straight_line_endpoint(S, dz)                     # (N, 4)
    dev = np.abs(Y[:, :4] - straight)
    ratio_pos = dev[:, :2].max(axis=1) / pos
    ratio_slope = dev[:, 2:4].max(axis=1) / slope
    line_err_um = dev[:, :2].max(axis=1) * 1e3

    check = {
        "source_dataset": os.path.relpath(os.path.abspath(a.source), HERE),
        "output_dataset": os.path.relpath(os.path.abspath(a.out), HERE),
        "n_rows": int(len(X)),
        "field": {"which": a.field, "file": field_path(a.field),
                  "md5": field_md5(a.field)},
        "definition": {
            "kappa": KAPPA,
            "qop_convention": "qop = 0.299792458 q / p[GeV] (Allen); |q/p| in "
                              "the scale is |S[:, 4]| = |qop|",
            "I_B": "INT_{z0}^{z0+dz} |B(x_s, y_s, z)| dz along the straight "
                   "line, %d-point midpoint rule, T mm" % N_FIELD_SAMPLES,
            "scale_pos_mm": "kappa * |qop| * I_B * |dz| / 2, floored at %g mm"
                            % POS_FLOOR_MM,
            "scale_slope": "kappa * |qop| * I_B, floored at %g" % SLOPE_FLOOR,
            "floors_vs_residual_design": "the residual wave used 1e-3 mm and "
                                         "1e-6; lowered here because stratum 0 "
                                         "(|dz| ~ 0.1 mm) has a true deviation "
                                         "of order 1e-7 mm",
            "node_profile": "flat; at the endpoint the poly profile coincides "
                            "with it (cout[-1] = 1), so there is nothing to "
                            "choose between here",
            "label_free": "only the start state and the leg enter the scale; "
                          "the RK6 reference is used below for the diagnostic "
                          "only",
        },
        "criterion": "the endpoint deviation/scale median must lie in "
                     "%g-%g in every stratum" % (OK_LO, OK_HI),
        "per_stratum": {},
        "per_split": {},
    }

    for i, name in enumerate(STRATUM_NAMES):
        m = STRAT == i
        if not m.any():
            continue
        check["per_stratum"][name] = {
            "index": i,
            "n": int(m.sum()),
            "abs_dz_mm": {"median": float(np.median(np.abs(dz[m]))),
                          "min": float(np.abs(dz[m]).min()),
                          "max": float(np.abs(dz[m]).max())},
            "I_B_Tmm_median": float(np.median(IB[m])),
            "scale_pos_mm_median": float(np.median(pos[m])),
            "scale_slope_median": float(np.median(slope[m])),
            "floored_pos": int((pos[m] <= POS_FLOOR_MM).sum()),
            "floored_slope": int((slope[m] <= SLOPE_FLOOR).sum()),
            "ratio_position_endpoint": quantiles(ratio_pos[m]),
            "ratio_slope_endpoint": quantiles(ratio_slope[m]),
            "straight_line_endpoint_error_um": {
                "median": float(np.median(line_err_um[m])),
                "p95": float(np.quantile(line_err_um[m], 0.95)),
                "max": float(line_err_um[m].max()),
                "fraction_below_1um": float((line_err_um[m] < 1.0).mean()),
                "fraction_below_0.1um": float((line_err_um[m] < 0.1).mean()),
            },
        }
        for s, sname in enumerate(SPLIT_NAMES):
            ms = m & (SPLIT == s)
            check["per_stratum"][name]["straight_line_endpoint_error_um"][
                "median_%s" % sname] = float(np.median(line_err_um[ms]))
        for sign, dname in ((1, "forward"), (-1, "backward")):
            md = m & (DIR == sign)
            check["per_stratum"][name]["straight_line_endpoint_error_um"][
                "median_%s" % dname] = float(np.median(line_err_um[md]))

    for s, sname in enumerate(SPLIT_NAMES):
        m = SPLIT == s
        check["per_split"][sname] = {
            "n": int(m.sum()),
            "ratio_position_endpoint_median": float(np.median(ratio_pos[m])),
            "straight_line_endpoint_median_um": float(np.median(line_err_um[m])),
        }

    meds = {t: check["per_stratum"][t]["ratio_position_endpoint"]["median"]
            for t in check["per_stratum"]}
    ok = all(OK_LO <= v <= OK_HI for v in meds.values())
    check["all_strata_within_band"] = bool(ok)
    check["verdict"] = (
        "the endpoint deviation/scale median is %s in every stratum (%s), "
        "%s O(1) on the whole four-decade range of |dz|"
        % ("inside %g-%g" % (OK_LO, OK_HI) if ok else "NOT inside the band",
           ", ".join("%s %.2f" % (t, v) for t, v in meds.items()),
           "so the target is" if ok else "so the target is NOT"))

    # -- the torch twin must agree with the numpy integral -------------------
    import torch
    from grid_model import GridResidualNetwork
    from _shared.reference import gauss_legendre
    c, _, _ = gauss_legendre(4)
    n_probe = 5000
    extra_mean = np.array([z0.mean(), dz.mean()])
    extra_scale = np.array([z0.std(), dz.std()])
    model = GridResidualNetwork(4, S.std(axis=0), np.ones(5), width=8, depth=1,
                                c=c, extra_mean=extra_mean,
                                extra_scale=extra_scale, field=fld)
    St = torch.tensor(S[:n_probe])
    EXt = torch.tensor(np.stack([(z0[:n_probe] - extra_mean[0]) / extra_scale[0],
                                 (dz[:n_probe] - extra_mean[1]) / extra_scale[1]],
                                axis=1))
    z0t, dzt = model.z0_dz(EXt)
    IBt = model.field_integral(St, z0t, dzt).numpy()
    check["torch_numpy_field_integral_max_rel_diff"] = float(
        np.abs(IBt - IB[:n_probe]).max() / np.abs(IB[:n_probe]).max())
    check["torch_numpy_leg_max_abs_diff_mm"] = float(max(
        np.abs(z0t.numpy() - z0[:n_probe]).max(),
        np.abs(dzt.numpy() - dz[:n_probe]).max()))

    np.savez_compressed(os.path.abspath(a.out), **data)
    with open(os.path.join(RESULTS, "scale_check.json"), "w") as f:
        json.dump(check, f, indent=1)

    print(json.dumps({
        "verdict": check["verdict"],
        "all_strata_within_band": check["all_strata_within_band"],
        "ratio_medians": meds,
        "straight_line_median_um": {
            t: check["per_stratum"][t]["straight_line_endpoint_error_um"]["median"]
            for t in check["per_stratum"]},
        "floored_pos": {t: check["per_stratum"][t]["floored_pos"]
                        for t in check["per_stratum"]},
        "torch_numpy_field_integral_max_rel_diff":
            check["torch_numpy_field_integral_max_rel_diff"],
    }, indent=1))

    # -- the figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = plt.cm.viridis(np.linspace(0, 0.92, len(STRATUM_NAMES)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for i, name in enumerate(STRATUM_NAMES):
        m = STRAT == i
        if not m.any():
            continue
        for ax, arr, lab in ((axes[0], ratio_pos, "position"),
                             (axes[1], ratio_slope, "slope")):
            v = arr[m]
            v = v[np.isfinite(v) & (v > 0)]
            ax.hist(np.log10(v), bins=70, histtype="step", lw=1.6,
                    color=colours[i],
                    label="%s (median %.2f)" % (name, np.median(v)))
        v = line_err_um[m]
        v = v[np.isfinite(v) & (v > 0)]
        axes[2].hist(np.log10(v), bins=70, histtype="step", lw=1.6,
                     color=colours[i],
                     label="%s (median %.3g um)" % (name, np.median(v)))
    for ax, lab in ((axes[0], "position"), (axes[1], "slope")):
        for v in (OK_LO, OK_HI):
            ax.axvline(np.log10(v), ls="--", color="0.4", lw=1.0)
        ax.axvline(0.0, ls=":", color="k", lw=1.0)
        ax.set_xlabel("log10( |RK6 reference - straight line| / scale ), "
                      "endpoint, %s" % lab)
        ax.set_ylabel("rows")
        ax.set_title("is the target O(1)?  (%s)" % lab)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[2].axvline(0.0, ls="--", color="0.4", lw=1.0)
    axes[2].set_xlabel("log10( straight-line endpoint error / um )")
    axes[2].set_ylabel("rows")
    axes[2].set_title("where the straight line already wins\n"
                      "(dashed: 1 um)")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    fig.suptitle("C3.1  residual scale check on the magnet-to-magnet set "
                 "(MagUp, floors %g mm / %g)" % (POS_FLOOR_MM, SLOPE_FLOOR),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIGURES, "scale_check.png"), dpi=140)
    plt.close(fig)
    print("wrote figures/scale_check.png and results/scale_check.json")
    return check


if __name__ == "__main__":
    main()
