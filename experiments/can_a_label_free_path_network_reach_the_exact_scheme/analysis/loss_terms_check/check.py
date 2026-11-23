"""Three candidate loss terms, measured on trained networks, before any training.

For A1 (round 37) and A2 (round 36), on the 2,000 test tracks of
../failure_of_A1 (seed 0) and on each run's 4,096 first-round training states,
along the network's own chain:

    r        the stage residual, (n, N, q, 4), as in the loss
    Y'       Y - r, one fixed-point sweep of the stages
    delta_n  h sum_i b_i (F(Y) - F(Y'))_i, the local defect (x, y, tx, ty);
             its x part is h sum_i b_i r_tx exactly
    c_n      delta_xy + delta_txty (z_last - z_{n+1}), carried to the last plane
    s        the spread of x and y of the first round's states (the loss's divisor)

    L0   the present loss: mean over n, i, d of (r / sigma_d)^2
    T1   carried term:   mean over tracks and x, y of (sum_n c_n / s)^2
    T2   endpoint-weighted residual: mean over tracks, n and x, y of (c_n / s)^2
    TB   block sums:     mean over tracks, x, y and the 8 blocks of 8 steps of
                         (sum_{n in block} c_n / s)^2
    T3   TB + T1

For each: the size against L0, the Spearman correlation of its value per track
with the track's true endpoint radial error (test tracks only), the share from
the position part (delta_x, delta_y) and the slope part (delta_tx, delta_ty
times the lever), and lambda so that lambda T / (L0 + lambda T) is 0.5 and 0.9.

    PYTHONNOUSERSITE=1 PYTHONPATH=../../../../RK_Pinn_module/src:../failure_of_A1 \
        THREADS=4 nice /data/bfys/gscriven/conda/envs/TE/bin/python check.py

Reads the store. Writes results.json here; nothing else.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

import compute as C                       # ../failure_of_A1/compute.py
from rkpinn.losses.stage_residual_of_a_path import residual_of_a_path

RUNS = {"A1": ("3ccd44d61cd8", "37"), "A2": ("0b41ecdd2505", "36")}
BLOCK = 8
SHARES = (0.5, 0.9)
HERE = os.path.dirname(os.path.abspath(__file__))


def pieces(run, states):
    net = run.network
    eq = net.equation_of_motion
    with torch.no_grad():
        track = net.predict(torch.as_tensor(states))
        r = residual_of_a_path(track, eq, True)                         # (n, N, q, 4)
        Y, F = track.stage_states, track.stage_rates
        n, N, q = Y.shape[:3]
        Fs = eq.rates((Y - r).reshape(n, N * q, 4), torch.as_tensor(states)[:, 4:5],
                      net.stage_planes_mm).reshape(n, N, q, 4)
        b = net.weights_of_the_tableau
        h = net.step_length_mm
        delta = h * torch.einsum("i,bnid->bnd", b, F - Fs)              # (n, N, 4)
        check = (delta[..., 0] - h * torch.einsum("i,bni->bn", b, r[..., 2])).abs().max().item()
    starts = net.layout.start_planes_mm
    lever = float(net.layout.last_plane_mm) - (starts + h)              # (N,)
    pos = delta[..., :2].numpy()
    slope = delta[..., 2:].numpy() * lever[None, :, None]
    sigma = np.asarray(run.constants["loss"]["divisor_per_component"])
    L0_track = ((r.numpy() / sigma) ** 2).mean(axis=(1, 2, 3))
    L0_pos = ((r.numpy()[..., :2] / sigma[:2]) ** 2).sum(-1).mean(axis=(1, 2)) / 4
    L0_slope = ((r.numpy()[..., 2:] / sigma[2:]) ** 2).sum(-1).mean(axis=(1, 2)) / 4
    return dict(pos=pos, slope=slope, s=sigma[:2], L0=L0_track, L0_pos=L0_pos,
                L0_slope=L0_slope, sum_of_carried=(pos + slope).sum(1), check=check)


def terms(c, s):
    """Per track: T1, T2, TB, T3 for carried defects c (n, N, 2)."""
    u = c / s
    T1 = (u.sum(1) ** 2).mean(-1)
    T2 = (u ** 2).mean(axis=(1, 2))
    n, N = u.shape[:2]
    TB = (u.reshape(n, N // BLOCK, BLOCK, 2).sum(2) ** 2).mean(axis=(1, 2))
    return dict(T1=T1, T2=T2, TB=TB, T3=TB + T1)


def measure(p, true_endpoint=None):
    full = terms(p["pos"] + p["slope"], p["s"])
    only_pos = terms(p["pos"], p["s"])
    only_slope = terms(p["slope"], p["s"])
    L0 = float(p["L0"].mean())
    out = {"L0": L0, "L0_share_position": float(p["L0_pos"].mean() / L0),
           "largest_difference_delta_x_against_h_b_r_tx_mm": p["check"]}
    if true_endpoint is not None:
        out["L0_spearman_with_endpoint"] = float(spearmanr(p["L0"], true_endpoint)[0])
    for name, value in full.items():
        T = float(value.mean())
        a, b = float(only_pos[name].mean()), float(only_slope[name].mean())
        row = {"value": T, "over_L0": T / L0, "share_position": a / (a + b),
               "share_slope": b / (a + b),
               "lambda": {str(s): s / (1 - s) * L0 / T for s in SHARES}}
        if true_endpoint is not None:
            row["spearman_with_endpoint"] = float(spearmanr(value, true_endpoint)[0])
        out[name] = row
    return out


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    tracks = C.load_tracks(C.store, C.TRACKS_KEY)
    test = tracks.split("test")
    rows = C.sample_rows(test.number_of_tracks)
    exact, _, _ = C.load_exact_states(C.store, C.TRACKS_KEY, "test", 64, 2)
    results = {}
    for name, (key, which) in RUNS.items():
        run, number = C.run_of(key, which)
        p = pieces(run, test.start_state[rows])
        whole = run.network.whole_track(test.start_state[rows], run.layout)
        end = whole.states_on_planes()[:, -1, :2]
        true = np.hypot(*((end - test.reference_end_state[rows, :2]) * 1e3).T)
        net_minus_exact = end - exact[rows, -1, :2]
        estimate = {"correlation_x": float(np.corrcoef(p["sum_of_carried"][:, 0],
                                                       net_minus_exact[:, 0])[0, 1]),
                    "correlation_y": float(np.corrcoef(p["sum_of_carried"][:, 1],
                                                       net_minus_exact[:, 1])[0, 1]),
                    "median_true_radial_um": float(np.median(true)),
                    "median_estimated_radial_um": float(np.median(
                        np.hypot(*p["sum_of_carried"].T) * 1e3))}
        results[name] = dict(round=number, test_2000=measure(p, true),
                             estimate_of_the_endpoint=estimate,
                             training_4096=measure(pieces(run, run.first_round.states)))
        print(name, "done", flush=True)
    with open(os.path.join(HERE, "results.json"), "w") as h:
        json.dump(results, h, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
