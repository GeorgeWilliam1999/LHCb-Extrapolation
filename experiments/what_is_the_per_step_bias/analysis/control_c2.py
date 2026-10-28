"""Control of C.2 and C.3: does the network matter when the stage equations
are swept or Newton-stepped afterwards?

The same fixed-point sweeps (0, 1, 2, 3, 5) and Newton steps (1, 2) of the
2-stage Gauss-Legendre stage equations, from six starting guesses of the stages:

  network <key>     the stages of each of the three runs (last round)
  straight line     Y_i = (x + tx c_i h, y + ty c_i h, tx, ty): zero curvature, no field call
  Euler             Y_i = S0 + c_i h f(S0): one field call a step
  RK4               classic RK4 from z0 to each stage plane z0 + c_i h: 4 field calls a stage

Field calls per step = calls of the start + 2 per sweep or Newton step + 2 for
the summed end state. A Newton step takes the field gradient with its 2 calls.

    control_c2.py      writes results/c2_control_endpoint.csv,
                       results/c2_control_stage_error.csv, figures/c2_control.png
"""
from __future__ import annotations

import numpy as np
import torch

from common import (BANDS, EQUATION, FIGURES, NUMBER_OF_STEPS, RUNS, SHORT, START_PLANES_MM,
                    STEP_MM, chain, endpoint_scores, network_of, on_planes, setup_threads,
                    stages_and_end, write_rows)
import stage_equations as se

C = se.C.numpy()
SWEEPS = (0, 1, 2, 3, 5)
NEWTON = (1, 2)


def straight_line_start(states, planes):
    d = (C * STEP_MM)[None, :]
    return np.stack([states[:, None, 0] + states[:, None, 2] * d,
                     states[:, None, 1] + states[:, None, 3] * d,
                     np.repeat(states[:, None, 2], 2, axis=1),
                     np.repeat(states[:, None, 3], 2, axis=1)], axis=-1)


def euler_start(states, planes):
    f = EQUATION.rates(states, planes)[:, :4]
    return states[:, None, :4] + (C * STEP_MM)[None, :, None] * f[:, None, :]


def rk4_start(states, planes):
    out = []
    for c in C:
        h = c * STEP_MM
        k1 = EQUATION.rates(states, planes)
        k2 = EQUATION.rates(states + 0.5 * h * k1, planes + 0.5 * h)
        k3 = EQUATION.rates(states + 0.5 * h * k2, planes + 0.5 * h)
        k4 = EQUATION.rates(states + h * k3, planes + h)
        out.append((states + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))[:, :4])
    return np.stack(out, axis=1)


def starts():
    out = []
    for _, key in RUNS:
        out.append(("network, " + SHORT[key], key, 0))
    out += [("straight line (no field)", None, 0), ("Euler predictor", None, 1),
            ("RK4 predictor", None, 8)]
    return out


def start_function(name, key):
    if key is not None:
        net = network_of(key)
        return lambda s, z: stages_and_end(net, s, z)[0]
    return {"straight line (no field)": straight_line_start, "Euler predictor": euler_start,
            "RK4 predictor": rk4_start}[name]


def iterate(states, planes, stages, sweeps, newton):
    end, Y = se.polish(states, planes, stages, sweeps=sweeps, newton_steps=newton)
    return end, Y


def main():
    setup_threads()
    ref = on_planes("test")
    n = len(ref)
    S = ref[:, :-1].reshape(-1, 5)
    Z = np.broadcast_to(START_PLANES_MM, (n, NUMBER_OF_STEPS)).reshape(-1).copy()
    _, exact_stages, worst = se.exact_step(S, Z)
    print("exact stages, largest residual %.1e" % worst, flush=True)
    end_rows, stage_rows = [], []
    for name, key, start_calls in starts():
        guess = start_function(name, key)
        Y0 = guess(S, Z)
        for method, counts in (("sweeps", SWEEPS), ("Newton steps", NEWTON)):
            for k in counts:
                if method == "Newton steps" and k == 0:
                    continue
                sw, nw = (k, 0) if method == "sweeps" else (0, k)
                field_calls = (start_calls + 2 * k + 2) * NUMBER_OF_STEPS
                # the stage error of one step from the reference state
                _, Y = iterate(S, Z, Y0, sw, nw)
                d = (Y - exact_stages)
                pos = np.abs(d[..., :2]).max(axis=(1, 2)) * 1e3
                slo = np.abs(d[..., 2:]).max(axis=(1, 2)) * 1e6
                stage_rows.append(dict(start=name, method=method, k=k,
                                       field_calls_per_track=field_calls,
                                       stage_position_error_median_um=float(np.median(pos)),
                                       stage_position_error_p84_um=float(np.quantile(pos, 0.84)),
                                       stage_slope_error_median_urad=float(np.median(slo)),
                                       stage_slope_error_p84_urad=float(np.quantile(slo, 0.84))))
                # the whole chain

                def step(s, z, sw=sw, nw=nw):
                    return iterate(s, z, guess(s, z), sw, nw)[0]
                final = chain(step, ref[:, 0])[:, -1]
                scores, _ = endpoint_scores(final)
                row = dict(start=name, method=method, k=k, field_calls_per_track=field_calls,
                           network_calls_per_track=64 if key else 0)
                row.update(scores)
                end_rows.append(row)
                print(name, method, k, field_calls, "%.4f" % scores["endpoint_radial_median_um"],
                      "stage pos %.3g um" % stage_rows[-1]["stage_position_error_median_um"],
                      flush=True)
    write_rows("c2_control_endpoint.csv", end_rows)
    write_rows("c2_control_stage_error.csv", stage_rows)
    figure(end_rows, stage_rows)


def figure(end_rows, stage_rows):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, _, _ in starts():
        r = [x for x in stage_rows if x["start"] == name and x["method"] == "sweeps"]
        axes[0].semilogy([x["k"] for x in r],
                         [max(x["stage_position_error_median_um"], 1e-9) for x in r], "o-",
                         label=name)
        e = [x for x in end_rows if x["start"] == name]
        axes[1].loglog([x["field_calls_per_track"] for x in e if x["method"] == "sweeps"],
                       [x["endpoint_radial_median_um"] for x in e if x["method"] == "sweeps"],
                       "o-", label=name)
    axes[0].set_xlabel("fixed-point sweeps")
    axes[0].set_ylabel("stage position error vs the exact stages, median (um)")
    axes[0].set_title("one step from the reference state")
    axes[1].axhline(0.147, color="grey", ls="--", lw=0.8, label="exact scheme 0.147 um")
    axes[1].axhline(1.08, color="black", ls=":", lw=0.8, label="RK4 1.08 um at 256")
    axes[1].set_xlabel("field calls per track")
    axes[1].set_ylabel("test endpoint radial median (um)")
    axes[1].set_title("64 steps chained (sweeps 0, 1, 2, 3, 5)")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, "c2_control.png"), dpi=120)
    print("written: figures/c2_control.png")


if __name__ == "__main__":
    main()
