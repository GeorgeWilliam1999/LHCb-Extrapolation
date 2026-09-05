#!/usr/bin/env python
"""Add the residual scales to the wave-1 dataset, and check they are O(1).

The population is not changed. `results/general_legs.npz` is read and written
back out unchanged as `results/general_legs_residual.npz` with three further
arrays per split:

    {split}_IB           (N,)  I_B = INT |B| dz along the straight line [T mm]
    {split}_scale_pos    (N,)  kappa |qop| I_B |dz| / 2   [mm]
    {split}_scale_slope  (N,)  kappa |qop| I_B             [-]

plus the floors and the chosen node profile as scalars, so the dataset carries
its own definition. The model recomputes the same numbers from (S, extra) at
forward time - `residual_model.field_integral` is the torch twin of the numpy
function used here, and the two are compared below - but storing them makes the
dataset self-describing and the check reproducible.

## The check the scales have to pass

The network's job is to emit `net_j = (reference_j - straight_j) / scale_j`.
That is only a well-conditioned target if it is O(1) on every leg type at once,
which is the whole hypothesis of this wave. So before any training this script
forms exactly that ratio using the RK4 reference already in the dataset - the
only place a label is used, and it is used for a diagnostic, never for the
scale - and histograms it per leg type, for both candidate node profiles:

    flat  every one of the q+1 outputs of a sample shares the sample's scale
    poly  output j is scaled by c_j^2 (positions) and c_j (slopes), i.e. by how
          far along the step the node sits

The rule, fixed before looking: take `flat` unless its endpoint ratio medians
fall outside 0.1-10 on some leg while `poly`'s do not.

Outputs
    results/general_legs_residual.npz
    results/residual_scale_check.json
    figures/residual_scale_check.png
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import json

import numpy as np

import use_shared                                        # noqa: F401
from _shared.reference import KAPPA, make_field
from residual_model import (POS_FLOOR_MM, SLOPE_FLOOR, N_FIELD_SAMPLES,
                            field_integral_numpy, scales_from_field_integral,
                            straight_line_states)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
SPLITS = ("train", "val", "test")
LEGS = ("A", "B", "C")
OK_LO, OK_HI = 0.1, 10.0


def ratios(ref, S, dz, c, scale_pos, scale_slope, profile):
    """|reference - straight| / scale, per node, positions and slopes."""
    straight = straight_line_states(S, dz, c)                    # (N, q+1, 4)
    dev = np.abs(ref[:, :, :4] - straight)
    cout = np.append(np.asarray(c), 1.0)
    fp = cout ** 2 if profile == "poly" else np.ones_like(cout)
    fs = cout if profile == "poly" else np.ones_like(cout)
    pos = dev[:, :, :2].max(axis=2) / (scale_pos[:, None] * fp[None, :])
    slope = dev[:, :, 2:4].max(axis=2) / (scale_slope[:, None] * fs[None, :])
    return pos, slope


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


def main():
    os.makedirs(FIGURES, exist_ok=True)
    src = os.path.join(RESULTS, "general_legs.npz")
    d = np.load(src)
    data = {k: d[k] for k in d.files}
    c = np.asarray(data["c"])
    fld = make_field(str(data["field"]) if "field" in data else "down")

    check = {
        "source_dataset": os.path.relpath(src, HERE),
        "definition": {
            "kappa": KAPPA,
            "qop_convention": "qop = 0.299792458 q / p[GeV] (Allen); |q/p| in "
                              "the scale is |S[:, 4]| = |qop|",
            "I_B": "INT_{z0}^{z0+dz} |B(x_s, y_s, z)| dz along the straight "
                   "line, %d-point midpoint rule, T mm" % N_FIELD_SAMPLES,
            "scale_pos_mm": "kappa * |qop| * I_B * |dz| / 2, floored at %g mm"
                            % POS_FLOOR_MM,
            "scale_slope": "kappa * |qop| * I_B, floored at %g" % SLOPE_FLOOR,
            "label_free": "only the start state and the leg enter; the RK4 "
                          "reference is used below for the diagnostic only",
        },
        "rule": ("take the flat node profile unless its endpoint ratio medians "
                 "fall outside %g-%g on some leg while poly's do not"
                 % (OK_LO, OK_HI)),
        "per_split": {},
        "histogram": {},
    }

    hist_data = {}
    for split in SPLITS:
        S = np.asarray(data["%s_S" % split])
        dz = np.asarray(data["%s_dz" % split])
        z0 = np.asarray(data["%s_z0" % split])
        ref = np.asarray(data["%s_ref" % split])
        L = np.asarray(data["%s_LEG" % split])
        IB = field_integral_numpy(S, z0, dz, fld)
        pos, slope = scales_from_field_integral(S, dz, IB)
        data["%s_IB" % split] = IB
        data["%s_scale_pos" % split] = pos
        data["%s_scale_slope" % split] = slope
        check["per_split"][split] = {
            "n": int(len(S)),
            "I_B_Tmm_median": float(np.median(IB)),
            "scale_pos_mm": {t: float(np.median(pos[L == i]))
                             for i, t in enumerate(LEGS) if (L == i).any()},
            "scale_slope": {t: float(np.median(slope[L == i]))
                            for i, t in enumerate(LEGS) if (L == i).any()},
            "floored_pos": int((pos <= POS_FLOOR_MM).sum()),
            "floored_slope": int((slope <= SLOPE_FLOOR).sum()),
        }
        if split != "test":
            continue
        for profile in ("flat", "poly"):
            rp, rs = ratios(ref, S, dz, c, pos, slope, profile)
            block = {}
            for i, t in enumerate(LEGS):
                m = L == i
                if not m.any():
                    continue
                block[t] = {
                    "position_endpoint": quantiles(rp[m, -1]),
                    "position_all_nodes": quantiles(rp[m, :]),
                    "slope_endpoint": quantiles(rs[m, -1]),
                    "slope_all_nodes": quantiles(rs[m, :]),
                }
            check["histogram"][profile] = block
            if profile == "flat":
                hist_data = {t: (rp[L == i, -1], rs[L == i, -1])
                             for i, t in enumerate(LEGS) if (L == i).any()}
            hist_data.setdefault("_poly", {})
            hist_data["_poly"][profile] = {
                t: (rp[L == i, -1], rs[L == i, -1])
                for i, t in enumerate(LEGS) if (L == i).any()}

    # ---- the choice --------------------------------------------------------
    def ok(profile):
        return all(OK_LO <= check["histogram"][profile][t]["position_endpoint"]["median"] <= OK_HI
                   for t in check["histogram"][profile])
    flat_ok, poly_ok = ok("flat"), ok("poly")
    chosen = "flat" if (flat_ok or not poly_ok) else "poly"
    check["flat_within_band"] = bool(flat_ok)
    check["poly_within_band"] = bool(poly_ok)
    check["node_profile_chosen"] = chosen
    check["verdict"] = (
        "the %s profile puts the endpoint deviation/scale median at %s on legs "
        "%s - %s O(1) on all three"
        % (chosen,
           ", ".join("%s %.2f" % (t, check["histogram"][chosen][t]
                                  ["position_endpoint"]["median"])
                     for t in sorted(check["histogram"][chosen])),
           ", ".join(sorted(check["histogram"][chosen])),
           "which is" if (flat_ok if chosen == "flat" else poly_ok)
           else "which is NOT"))

    data["residual_node_profile"] = np.array(chosen)
    data["residual_pos_floor_mm"] = np.array(POS_FLOOR_MM)
    data["residual_slope_floor"] = np.array(SLOPE_FLOOR)
    data["residual_n_field_samples"] = np.array(N_FIELD_SAMPLES)

    out_npz = os.path.join(RESULTS, "general_legs_residual.npz")
    np.savez_compressed(out_npz, **data)
    check["output_dataset"] = os.path.relpath(out_npz, HERE)

    # ---- the torch twin must agree with the numpy integral -----------------
    import torch
    from residual_model import build_residual_model
    model = build_residual_model(data, 50, 4, node_profile=chosen)
    S = torch.tensor(np.asarray(data["test_S"])[:2000])
    EX = torch.tensor(np.asarray(data["test_extra"])[:2000])
    z0t, dzt = model.z0_dz(EX)
    IBt = model.field_integral(S, z0t, dzt).numpy()
    check["torch_numpy_field_integral_max_abs_diff_Tmm"] = float(
        np.abs(IBt - np.asarray(data["test_IB"])[:2000]).max())
    check["torch_numpy_leg_max_abs_diff_mm"] = float(max(
        np.abs(z0t.numpy() - np.asarray(data["test_z0"])[:2000]).max(),
        np.abs(dzt.numpy() - np.asarray(data["test_dz"])[:2000]).max()))

    with open(os.path.join(RESULTS, "residual_scale_check.json"), "w") as f:
        json.dump(check, f, indent=1)
    print(json.dumps({k: check[k] for k in
                      ("node_profile_chosen", "flat_within_band",
                       "poly_within_band", "verdict",
                       "torch_numpy_field_integral_max_abs_diff_Tmm")},
                     indent=1))

    # ---- the figure --------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colours = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}
    titles = {"A": "A  vertex fetch", "B": "B  cross-magnet",
              "C": "C  plane-to-plane"}
    for col, profile in enumerate(("flat", "poly")):
        for row, (which, name) in enumerate(
                ((0, "position"), (1, "slope"))):
            ax = axes[row, col]
            for t, (rp, rs) in hist_data["_poly"][profile].items():
                a = (rp if which == 0 else rs)
                a = a[np.isfinite(a) & (a > 0)]
                ax.hist(np.log10(a), bins=60, histtype="step", lw=1.6,
                        color=colours[t], label="%s (median %.2f)"
                        % (titles[t], np.median(a)))
            for v in (OK_LO, OK_HI):
                ax.axvline(np.log10(v), ls="--", color="0.4", lw=1.0)
            ax.axvline(0.0, ls=":", color="k", lw=1.0)
            ax.set_xlabel("log10( |reference - straight line| / scale )  "
                          "at the endpoint")
            ax.set_ylabel("test states")
            ax.set_title("%s node profile - %s" % (profile, name))
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle("Residual redesign: is the deviation from a straight line "
                 "O(1) on its own scale?  (dashed: the 0.1-10 band; chosen: %s)"
                 % chosen, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES, "residual_scale_check.png"), dpi=140)
    plt.close(fig)
    print("wrote figures/residual_scale_check.png")


if __name__ == "__main__":
    main()
