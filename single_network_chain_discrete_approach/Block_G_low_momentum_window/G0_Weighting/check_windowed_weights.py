#!/usr/bin/env python
"""Block G gates W1-W4: the windowed loss, before any training.

The module under test is `windowed_loss.py`, which is Block F's
`F0_Weighting/weighted_loss.py` with the momentum window read from the run's
own constants instead of the module defaults. Block F's folder is read-only:
everything here imports from it (the numpy twin `weights_numpy`, the state
drawer `draw_pairs` and the stub network `FixedOutputs` all come from
`F0_Weighting/check_weights.py`).

Gate W1 - the twin gate with a MOVED window. For N = 2 and 256, q = 2 and 16,
on 2,000 track states drawn over the start planes, and for every candidate
window: this module's torch weights equal the independent numpy implementation
(relative difference below 1e-13) under every mode. The same gate also
demonstrates the trap it was written for: Block F's own `weights`, handed the
same non-default constants, DISAGREES with the twin by order unity, because its
`per_track_factor` calls `band_window(p)` with the module defaults 10-50 GeV.
For the 10-50 GeV row the two agree, which is the other half of the statement -
nothing has moved for the window Block F actually ran.

Gate W2 - the `blockE` path is untouched. With mode `blockE` the weights are
1 / in_scale and `weighted_loss` through this module must reproduce
`_shared.model.physics_loss` to floating-point round-off (below 1e-13).

Gate W3 - the reweighting still does not move the minimum. The q-stage scheme
is solved exactly without a network on 40 states per case; those stage and end
states must give machine zero under EVERY mode with the anchor window - below
1e-10 times the straight line's loss on the same states.

Gate W4 - the pre-flight, and the only one that can still change the design.
On the TRAINED N = 64, q = 2 network of Block F (`F1_Training/results/full/
N064_q02`, phase `done`), 8,000 (track, plane) states drawn evenly over the
planes: for each candidate window, where the loss comes from by momentum and
along z, and how well a state's share of the loss ranks with the endpoint cost
its single-step error actually incurs. That cost reference is a diagnostic
only; it never enters the loss, which stays label-free.

Every W4 quantity is reported twice: over all 8,000 states, and with the most
extreme 1 % of states (80 of them) removed and the rest renormalised. That is
not cosmetic. On this trained network the residual distribution has a very
heavy tail - a single state can carry two thirds of the whole objective - so
the untrimmed shares describe where that tail happens to sit rather than what
the window does. The trimmed column is the one that separates the windows.

The caution W4 measures rather than assumes: the per-track factor is
a_n = sqrt(W(p)) * D_ref / D_n with D_n proportional to |q/p|, so a_n grows
like p while sqrt(W) falls outside the band. A low window and the track-bend
normalisation pull in opposite directions, and only the measurement says who
wins.

Note on the draw: the pre-flight uses its own generator, seeded 20260918 as in
Block F's gate 4, created immediately before the draw. The states are therefore
reproducible within this script but are not the same 8,000 states as Block F's
gate 4, whose generator had already been consumed by the three gates before it.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python check_windowed_weights.py
Output:  results/check_windowed_weights.json, results/preflight_windows.csv,
         figures/window_preflight.png; prints ALL WINDOW CHECKS PASS or the failure
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"
# the modules this imports live in read-only folders: do not drop .pyc files there
sys.dont_write_bytecode = True

import csv    # noqa: E402
import json   # noqa: E402
import time   # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

import use_shared    # noqa: E402,F401
import windowed_loss as GL                                          # noqa: E402  (puts F0 on the path)
import weighted_loss as WL                                          # noqa: E402  (Block F's, untouched)
import check_weights as CW                                          # noqa: E402  (Block F's gates)

from _shared.model import LHCbRates, physics_loss, reconstruction_residuals   # noqa: E402
from _shared.reference import gauss_legendre, make_field, rk6_rows            # noqa: E402
from chain_network import build, load_network, znodes_for           # noqa: E402
from exact_solver import solve_state                                # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = use_shared.SHARED_ROOT
E = os.path.join(ROOT, "Block_E_single_network_chain")
F = os.path.join(ROOT, "Block_F_reweighted_loss")
TRACKS = os.path.join(E, "E0_Track_dataset", "results", "tracks.npz")
TRAINED = os.path.join(F, "F1_Training", "results", "full", "N064_q02")

TWIN_TOL = 1e-13
TRAP_MIN = 1e-6            # the disagreement the trap must produce, at least
LOSS_TOL = 1e-13
ZERO_RATIO_TOL = 1e-10
PREFLIGHT_STATES = 8000
PREFLIGHT_SEED = 20260918
RK6_LOCAL_STEP = 1.0
P_EDGES = np.array([1, 2, 3, 5, 8, 10, 20, 50, 100, 200.0])
MODES = ("blockE", "full", "no_lever", "no_track", "no_window")
WINDOW_MODES = tuple(m for m in MODES if WL.MODES[m]["window"])
CLAMP_SCAN = (0.0, 2.0, 3.0, 5.0, 10.0)

# (p_lo, p_hi, role) - the windows put to George. Clamp 5 throughout.
CANDIDATES = [(3.0, 8.0, "anchor"),
              (3.0, 10.0, "candidate"),
              (3.0, 20.0, "candidate"),
              (2.0, 15.0, "candidate"),
              (4.0, 6.0, "narrow reference"),
              (10.0, 50.0, "baseline")]
ANCHOR = (3.0, 8.0)
BASELINE = (10.0, 50.0)
FOCUS = (3.0, 8.0)         # the region George asked the optimisation to serve


def wlabel(p_lo, p_hi):
    """How a window is named everywhere a human reads it."""
    return "%g-%g GeV" % (p_lo, p_hi)


def main():
    t_all = time.time()
    D = np.load(TRACKS)
    Z0, Z1, L, n_max = float(D["z0"]), float(D["z1"]), float(D["L"]), int(D["n_max"])
    truth = D["train_truth"]
    fld = make_field(str(D["field"]))
    rates = LHCbRates(make_field(str(D["field"])))
    ibar = GL.i_bar(fld, Z0, Z1)
    rng = np.random.default_rng(20260918)
    rec = {"i_bar_T_mm": ibar, "z0": Z0, "z1": Z1, "L": L,
           "candidates": [dict(p_lo=lo, p_hi=hi, role=role, label=wlabel(lo, hi))
                          for lo, hi, role in CANDIDATES],
           "trained_network": TRAINED,
           "W1": [], "W2": [], "W3": [], "W4": {}, "clamp_scan": {}}
    ok = True

    # ----------------------------------------------------------------- gate W1 --
    print("gate W1: the numpy twin, with the window moved", flush=True)
    for N in (2, 256):
        dz = L / N
        S, k, _ = CW.draw_pairs(truth, N, n_max, 2000, rng)
        z_start = Z0 + k * dz
        in_scale = S.std(axis=0)
        for q in (2, 16):
            model = build(q, N, L, Z0, in_scale, fld, seed=0)
            St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
            for p_lo, p_hi, role in CANDIDATES:
                const = GL.reference_constants(model, S, z_start, Z1, ibar, L,
                                               p_lo=p_lo, p_hi=p_hi)
                r = {"N": N, "q": q, "p_lo": p_lo, "p_hi": p_hi, "role": role,
                     "window": wlabel(p_lo, p_hi), "worst_twin_rel": 0.0,
                     "worst_blockF_rel": 0.0}
                for mode in MODES:
                    wn = CW.weights_numpy(model, S, z_start, const, mode)
                    wg = GL.weights(model, St, zt, const, mode).numpy()
                    rel = float(np.max(np.abs(wg - wn) / np.abs(wn)))
                    r["twin_rel_%s" % mode] = rel
                    r["worst_twin_rel"] = max(r["worst_twin_rel"], rel)
                    if mode in WINDOW_MODES:
                        wf = WL.weights(model, St, zt, const, mode).numpy()
                        relf = float(np.max(np.abs(wf - wn) / np.abs(wn)))
                        r["blockF_rel_%s" % mode] = relf
                        r["worst_blockF_rel"] = max(r["worst_blockF_rel"], relf)
                lev = GL.lever_arms(model.cout, dz, zt, Z1).numpy()
                r["lever_max_mm"] = float(lev.max())
                r["lever_min_mm"] = float(lev.min())
                r["lever_expected_max_mm"] = float(L + dz * (1.0 - float(model.c[0])))
                default_window = (p_lo == WL.P_LO and p_hi == WL.P_HI)
                # the trap: with a moved window the read-only module disagrees with
                # the twin; with its own window it agrees
                r["trap_demonstrated"] = bool(r["worst_blockF_rel"] > TRAP_MIN)
                r["ok"] = bool(
                    r["worst_twin_rel"] < TWIN_TOL
                    and abs(r["lever_min_mm"] - dz) < 1e-9 * max(dz, 1.0)
                    and abs(r["lever_max_mm"] - r["lever_expected_max_mm"]) < 1e-9 * L
                    and ((r["worst_blockF_rel"] < TWIN_TOL) if default_window
                         else r["trap_demonstrated"]))
                ok &= r["ok"]
                rec["W1"].append(r)
                print("  N=%3d q=%2d  window %-10s twin %.1e   read-only module vs twin %.2e"
                      "  ->  %s" % (N, q, wlabel(p_lo, p_hi), r["worst_twin_rel"],
                                    r["worst_blockF_rel"], r["ok"]), flush=True)
    pw = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0, 200.0])
    rec["window_at"] = {}
    for p_lo, p_hi, _role in CANDIDATES:
        vals = GL.band_window(pw, p_lo, p_hi)
        rec["window_at"][wlabel(p_lo, p_hi)] = {("%g GeV" % p): float(v) for p, v in zip(pw, vals)}
        print("  window %-10s " % wlabel(p_lo, p_hi)
              + " ".join("%g:%.3f" % (p, v) for p, v in zip(pw, vals)), flush=True)

    # ----------------------------------------------------------------- gate W2 --
    print("gate W2: the unweighted path is unchanged", flush=True)
    for N in (2, 256):
        dz = L / N
        S, k, _ = CW.draw_pairs(truth, N, n_max, 500, rng)
        z_start = Z0 + k * dz
        in_scale = S.std(axis=0)
        for q in (2, 16):
            c, A, b = gauss_legendre(q)
            model = build(q, N, L, Z0, in_scale, fld, seed=0)
            const = GL.reference_constants(model, S, z_start, Z1, ibar, L,
                                           p_lo=ANCHOR[0], p_hi=ANCHOR[1])
            St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
            zn = znodes_for(model, zt)
            At, bt = torch.tensor(A), torch.tensor(b)
            shared = float(physics_loss(model, rates, St, dz, zn, At, bt, zt).item())
            w = GL.weights(model, St, zt, const, "blockE")
            mine = float(GL.weighted_loss(model, rates, St, dz, zn, At, bt, zt, w).item())
            r = {"N": N, "q": q, "shared": shared, "weighted_path": mine,
                 "rel": abs(mine - shared) / shared}
            r["ok"] = bool(r["rel"] < LOSS_TOL)
            ok &= r["ok"]
            rec["W2"].append(r)
            print("  N=%3d q=%2d  shared %.6e  weighted path %.6e  rel %.1e  ->  %s"
                  % (N, q, shared, mine, r["rel"], r["ok"]), flush=True)

    # ----------------------------------------------------------------- gate W3 --
    print("gate W3: the minimum does not move, window %s" % wlabel(*ANCHOR), flush=True)
    for N in (2, 256):
        dz = L / N
        S, k, _ = CW.draw_pairs(truth, N, n_max, 40, rng)
        z_start = Z0 + k * dz
        in_scale = truth[:, ::n_max // N][:, :N].reshape(-1, 5).std(axis=0)
        for q in (2, 16):
            tab = gauss_legendre(q)
            c, A, b = tab
            exact = np.empty((len(S), q + 1, 4))
            n_fail = 0
            for i in range(len(S)):
                stages, S1, conv, _, _ = solve_state(S[i], z_start[i], dz, tab, fld)
                exact[i, :q], exact[i, q] = stages[:, :4], S1[:4]
                n_fail += (not conv)
            straight = np.stack([np.stack([S[:, 0] + S[:, 2] * cj * dz, S[:, 1] + S[:, 3] * cj * dz,
                                           S[:, 2], S[:, 3]], axis=1)
                                 for cj in np.append(c, 1.0)], axis=1)
            model = build(q, N, L, Z0, in_scale, fld, seed=0)
            const = GL.reference_constants(model, S, z_start, Z1, ibar, L,
                                           p_lo=ANCHOR[0], p_hi=ANCHOR[1])
            St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
            zn = znodes_for(model, zt)
            At, bt = torch.tensor(A), torch.tensor(b)
            r = {"N": N, "q": q, "p_lo": ANCHOR[0], "p_hi": ANCHOR[1],
                 "solves_not_converged": int(n_fail), "worst_ratio": 0.0}
            for mode in MODES:
                w = GL.weights(model, St, zt, const, mode)
                ls = {}
                for name, outs in (("exact", exact), ("straight", straight)):
                    stub = CW.FixedOutputs(outs, in_scale, q)
                    ls[name] = float(GL.weighted_loss(stub, rates, St, dz, zn, At, bt, zt, w).item())
                ratio = ls["exact"] / ls["straight"]
                r["ratio_%s" % mode] = ratio
                r["worst_ratio"] = max(r["worst_ratio"], ratio)
            r["ok"] = bool(n_fail == 0 and r["worst_ratio"] < ZERO_RATIO_TOL)
            ok &= r["ok"]
            rec["W3"].append(r)
            print("  N=%3d q=%2d  exact/straight, worst over modes %.1e  ->  %s"
                  % (N, q, r["worst_ratio"], r["ok"]), flush=True)

    # ----------------------------------------------------------------- gate W4 --
    print("gate W4: the pre-flight, on the trained network N = 64, q = 2", flush=True)
    with open(os.path.join(TRAINED, "progress.json")) as f:
        prog = json.load(f)
    rec["trained_network_phase"] = prog.get("phase")
    rec["trained_network_round"] = prog.get("round")
    print("  %s (phase %s, round %s)" % (TRAINED, prog.get("phase"), prog.get("round")), flush=True)
    with open(os.path.join(TRAINED, "scale.json")) as f:
        sc = json.load(f)
    model = load_network(TRAINED, fld)
    N, q = int(sc["N"]), int(sc["q"])
    dz = float(model.dz)
    print("  N = %d, q = %d, dz = %.2f mm" % (N, q, dz), flush=True)
    c, A, b = gauss_legendre(q)
    At, bt = torch.tensor(A), torch.tensor(b)
    prng = np.random.default_rng(PREFLIGHT_SEED)
    S, k, tr = CW.draw_pairs(truth, N, n_max, PREFLIGHT_STATES, prng, even=True)
    z_start = Z0 + k * dz
    P = GL.QOP_TO_GEV / np.abs(S[:, 4])
    St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
    zn = znodes_for(model, zt)
    rec["n_preflight_states"] = int(len(S))

    with torch.no_grad():
        r_phys = (reconstruction_residuals(model, rates, St, dz, zn, At, bt, zt)
                  * model.in_scale[:4]).numpy()
    # what each state's single-step error actually costs at the first SciFi
    # plane (diagnostic only, never in the loss)
    out = model(St, zt).detach().numpy()[:, -1, :]
    ref = rk6_rows(S, z_start, z_start + dz, step=RK6_LOCAL_STEP, field=fld)
    lev_end = (Z1 - (z_start + dz)) + dz
    cost = np.hypot((out[:, 0] - ref[:, 0]) + (out[:, 2] - ref[:, 2]) * lev_end,
                    (out[:, 1] - ref[:, 1]) + (out[:, 3] - ref[:, 3]) * lev_end) * 1e3   # um

    from scipy.stats import spearmanr
    rows, shares, trimmed = [], {}, {}
    focus = (P >= FOCUS[0]) & (P < FOCUS[1])
    n_trim = max(1, len(S) // 100)
    for p_lo, p_hi, role in CANDIDATES:
        lab = wlabel(p_lo, p_hi)
        const = GL.reference_constants(model, S, z_start, Z1, ibar, L, p_lo=p_lo, p_hi=p_hi)
        w = GL.weights(model, St, zt, const, "full").numpy()
        per_state = ((r_phys * w) ** 2).mean(axis=(1, 2))
        share = per_state / per_state.sum()
        shares[lab] = share
        # the same shares with the most extreme 1 % of states removed: the
        # residual distribution has a very heavy tail on this trained network
        # (one state can carry two thirds of the loss), and the untrimmed
        # shares then say more about that tail than about the window
        keep = np.ones(len(share), bool)
        keep[np.argsort(share)[::-1][:n_trim]] = False
        share_t = np.where(keep, share, 0.0)
        share_t = share_t / share_t.sum()
        trimmed[lab] = share_t
        own = (P >= p_lo) & (P < p_hi)
        d = dict(p_lo=p_lo, p_hi=p_hi, role=role,
                 largest_single_state_share_pct=100 * float(share.max()),
                 share_own_band_trimmed_pct=100 * float(share_t[own].sum()),
                 share_3_8_GeV_trimmed_pct=100 * float(share_t[focus].sum()),
                 share_above_50_GeV_trimmed_pct=100 * float(share_t[P >= 50].sum()),
                 share_above_100_GeV_trimmed_pct=100 * float(share_t[P >= 100].sum()),
                 share_own_band_pct=100 * float(share[own].sum()),
                 tracks_own_band_pct=100 * float(own.mean()),
                 share_3_8_GeV_pct=100 * float(share[focus].sum()),
                 tracks_3_8_GeV_pct=100 * float(focus.mean()),
                 share_above_50_GeV_pct=100 * float(share[P >= 50].sum()),
                 share_above_100_GeV_pct=100 * float(share[P >= 100].sum()),
                 top_1pct_of_states_share_pct=100 * float(
                     np.sort(share)[::-1][:max(1, len(share) // 100)].sum()),
                 spearman_share_vs_cost=float(spearmanr(share, cost).statistic),
                 spearman_share_vs_cost_in_own_band=float(
                     spearmanr(share[own], cost[own]).statistic),
                 spearman_share_vs_cost_in_3_8_GeV=float(
                     spearmanr(share[focus], cost[focus]).statistic))
        for j in range(4):
            m = (k >= j * N // 4) & (k < (j + 1) * N // 4)
            d["share_quarter_%d_pct" % (j + 1)] = 100 * float(share[m].sum())
            d["share_quarter_%d_trimmed_pct" % (j + 1)] = 100 * float(share_t[m].sum())
            rows.append(dict(window=lab, p_lo=p_lo, p_hi=p_hi, role=role,
                             kind="quarter of z", lo=j + 1, hi=j + 1, n_states=int(m.sum()),
                             share_of_loss_pct=100 * float(share[m].sum()),
                             share_of_loss_trimmed_pct=100 * float(share_t[m].sum()),
                             share_of_states_pct=100 * float(m.mean())))
        for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
            m = (P >= lo) & (P < hi)
            rows.append(dict(window=lab, p_lo=p_lo, p_hi=p_hi, role=role,
                             kind="momentum band [GeV]", lo=float(lo), hi=float(hi),
                             n_states=int(m.sum()),
                             share_of_loss_pct=100 * float(share[m].sum()),
                             share_of_loss_trimmed_pct=100 * float(share_t[m].sum()),
                             share_of_states_pct=100 * float(m.mean())))
            d["share_%g_%g_GeV_pct" % (lo, hi)] = 100 * float(share[m].sum())
            d["share_%g_%g_GeV_trimmed_pct" % (lo, hi)] = 100 * float(share_t[m].sum())
        rec["W4"][lab] = d
        print("  %-10s (%-16s) own band %5.1f%%  3-8 GeV %5.1f%%  >50 GeV %5.1f%%  >100 GeV %5.1f%%"
              "  top 1%% %5.1f%%  rho all %+.3f / own %+.3f / 3-8 %+.3f"
              % (lab, role, d["share_own_band_pct"], d["share_3_8_GeV_pct"],
                 d["share_above_50_GeV_pct"], d["share_above_100_GeV_pct"],
                 d["top_1pct_of_states_share_pct"], d["spearman_share_vs_cost"],
                 d["spearman_share_vs_cost_in_own_band"],
                 d["spearman_share_vs_cost_in_3_8_GeV"]), flush=True)
        print("             quarters of z: %5.1f%% %5.1f%% %5.1f%% %5.1f%%"
              % tuple(d["share_quarter_%d_pct" % (j + 1)] for j in range(4)), flush=True)
        print("             largest single state %5.1f%%  |  with the most extreme 1%% of states "
              "removed: own band %5.1f%%  3-8 GeV %5.1f%%  >50 GeV %5.1f%%  quarters of z "
              "%4.1f%% %4.1f%% %4.1f%% %4.1f%%"
              % (d["largest_single_state_share_pct"], d["share_own_band_trimmed_pct"],
                 d["share_3_8_GeV_trimmed_pct"], d["share_above_50_GeV_trimmed_pct"],
                 d["share_quarter_1_trimmed_pct"], d["share_quarter_2_trimmed_pct"],
                 d["share_quarter_3_trimmed_pct"], d["share_quarter_4_trimmed_pct"]), flush=True)

    # the clamp scan for the anchor window - reported only, the clamp stays at 5
    print("  clamp scan, window %s (report only; the clamp stays at 5)" % wlabel(*ANCHOR),
          flush=True)
    own_anchor = (P >= ANCHOR[0]) & (P < ANCHOR[1])
    for cl in CLAMP_SCAN:
        const = GL.reference_constants(model, S, z_start, Z1, ibar, L,
                                       p_lo=ANCHOR[0], p_hi=ANCHOR[1], clamp=cl)
        # how often the clamp actually bites, which is why the scan looks as it does
        a_raw = GL.per_track_factor(S[:, 4], const, WL.MODES["full"], clamp=0.0)
        med = float(np.median(a_raw))
        clipped_lo = float((a_raw < med / cl).mean()) if cl > 1.0 else 0.0
        clipped_hi = float((a_raw > med * cl).mean()) if cl > 1.0 else 0.0
        w = GL.weights(model, St, zt, const, "full").numpy()
        per_state = ((r_phys * w) ** 2).mean(axis=(1, 2))
        share = per_state / per_state.sum()
        keep = np.ones(len(share), bool)
        keep[np.argsort(share)[::-1][:n_trim]] = False
        share_t = np.where(keep, share, 0.0)
        share_t = share_t / share_t.sum()
        s = dict(clamp=("off" if cl == 0 else "%g" % cl),
                 tracks_clamped_low_pct=100 * clipped_lo,
                 tracks_clamped_high_pct=100 * clipped_hi,
                 largest_single_state_share_pct=100 * float(share.max()),
                 share_3_8_GeV_trimmed_pct=100 * float(share_t[focus].sum()),
                 share_10_50_GeV_trimmed_pct=100 * float(share_t[(P >= 10) & (P < 50)].sum()),
                 share_above_50_GeV_trimmed_pct=100 * float(share_t[P >= 50].sum()),
                 share_own_band_pct=100 * float(share[own_anchor].sum()),
                 share_3_8_GeV_pct=100 * float(share[focus].sum()),
                 share_2_3_GeV_pct=100 * float(share[(P >= 2) & (P < 3)].sum()),
                 share_10_50_GeV_pct=100 * float(share[(P >= 10) & (P < 50)].sum()),
                 share_above_50_GeV_pct=100 * float(share[P >= 50].sum()),
                 share_above_100_GeV_pct=100 * float(share[P >= 100].sum()),
                 top_1pct_of_states_share_pct=100 * float(
                     np.sort(share)[::-1][:max(1, len(share) // 100)].sum()),
                 spearman_share_vs_cost=float(spearmanr(share, cost).statistic),
                 spearman_share_vs_cost_in_3_8_GeV=float(
                     spearmanr(share[focus], cost[focus]).statistic))
        rec["clamp_scan"][s["clamp"]] = s
        print("    clamp %4s: 3-8 GeV %5.1f%%  10-50 GeV %5.1f%%  >50 GeV %5.1f%%  >100 GeV %5.1f%%"
              "  top 1%% %5.1f%%  rho %+.3f" % (s["clamp"], s["share_3_8_GeV_pct"],
                                                s["share_10_50_GeV_pct"],
                                                s["share_above_50_GeV_pct"],
                                                s["share_above_100_GeV_pct"],
                                                s["top_1pct_of_states_share_pct"],
                                                s["spearman_share_vs_cost"]), flush=True)
        print("               with the most extreme 1%% of states removed: 3-8 GeV %5.1f%%  "
              "10-50 GeV %5.1f%%  >50 GeV %5.1f%%  (largest single state %5.1f%%)  |  "
              "tracks clamped: %4.1f%% low, %4.1f%% high"
              % (s["share_3_8_GeV_trimmed_pct"], s["share_10_50_GeV_trimmed_pct"],
                 s["share_above_50_GeV_trimmed_pct"], s["largest_single_state_share_pct"],
                 s["tracks_clamped_low_pct"], s["tracks_clamped_high_pct"]), flush=True)

    majority_own = [wlabel(lo, hi) for lo, hi, _ in CANDIDATES
                    if rec["W4"][wlabel(lo, hi)]["share_own_band_pct"] > 50.0]
    majority_focus = [wlabel(lo, hi) for lo, hi, _ in CANDIDATES
                      if rec["W4"][wlabel(lo, hi)]["share_3_8_GeV_pct"] > 50.0]
    majority_own_t = [wlabel(lo, hi) for lo, hi, _ in CANDIDATES
                      if rec["W4"][wlabel(lo, hi)]["share_own_band_trimmed_pct"] > 50.0]
    majority_focus_t = [wlabel(lo, hi) for lo, hi, _ in CANDIDATES
                        if rec["W4"][wlabel(lo, hi)]["share_3_8_GeV_trimmed_pct"] > 50.0]
    rec["windows_with_majority_in_own_band"] = majority_own
    rec["windows_with_majority_in_3_8_GeV"] = majority_focus
    rec["windows_with_majority_in_own_band_trimmed"] = majority_own_t
    rec["windows_with_majority_in_3_8_GeV_trimmed"] = majority_focus_t
    rec["n_states_trimmed"] = int(n_trim)
    print("  windows taking more than half the loss inside their own band: %s"
          % (", ".join(majority_own) or "none"), flush=True)
    print("  windows taking more than half the loss inside 3-8 GeV: %s"
          % (", ".join(majority_focus) or "none"), flush=True)
    print("  the same with the most extreme 1%% of states removed - own band: %s"
          % (", ".join(majority_own_t) or "none"), flush=True)
    print("  the same with the most extreme 1%% of states removed - 3-8 GeV: %s"
          % (", ".join(majority_focus_t) or "none"), flush=True)

    # ------------------------------------------------------------------ output --
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "preflight_windows.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    rec["all_pass"] = bool(ok)
    rec["wall_s"] = time.time() - t_all
    with open(os.path.join(HERE, "results", "check_windowed_weights.json"), "w") as f:
        json.dump(rec, f, indent=1)

    # ------------------------------------------------------------------ figure --
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    ax = axes.ravel()
    centres = np.sqrt(P_EDGES[:-1] * P_EDGES[1:])
    for p_lo, p_hi, role in CANDIDATES:
        lab = wlabel(p_lo, p_hi)
        style = "-o" if (p_lo, p_hi) in (ANCHOR, BASELINE) else "--o"
        lw = 2.2 if (p_lo, p_hi) in (ANCHOR, BASELINE) else 1.2
        for a, sr in ((ax[0], shares[lab]), (ax[1], trimmed[lab])):
            sh = [sr[(P >= lo) & (P < hi)].sum() * 100
                  for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:])]
            a.plot(centres, sh, style, ms=4, lw=lw, label="loss window %s" % lab)
        qs = [trimmed[lab][(k >= j * N // 4) & (k < (j + 1) * N // 4)].sum() * 100
              for j in range(4)]
        ax[2].plot([1, 2, 3, 4], qs, style, ms=4, lw=lw, label="loss window %s" % lab)
    frac = np.array([(P >= lo) & (P < hi)
                     for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:])]).mean(axis=1) * 100
    for a, title in ((ax[0], "Where the loss comes from, by momentum\n"
                             "(shaded: 3-8 GeV, the region of interest)"),
                     (ax[1], "The same, with the most extreme 1% of states removed\n"
                             "(the heavy tail of the residual, not the window)")):
        a.plot(centres, frac, "k:", lw=1.4, label="share of the states")
        a.axvspan(FOCUS[0], FOCUS[1], color="0.88", zorder=0)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("momentum [GeV]")
        a.set_ylabel("share of the loss [%]")
        a.set_title(title, fontsize=10)
        a.legend(fontsize=7)
    ax[2].axhline(25, color="k", ls=":", lw=1.4, label="flat")
    ax[2].set_xticks([1, 2, 3, 4], ["first", "second", "third", "fourth"])
    ax[2].set_xlabel("quarter of the crossing the step starts in")
    ax[2].set_ylabel("share of the loss [%]")
    ax[2].set_title("Where the loss comes from along z, with the most extreme\n"
                    "1% of states removed", fontsize=10)
    ax[2].legend(fontsize=7)
    o = np.argsort(cost)
    xo = np.arange(len(o)) / len(o) * 100
    for (p_lo, p_hi), col in zip((ANCHOR, BASELINE), ("C0", "C5")):
        lab = wlabel(p_lo, p_hi)
        ax[3].plot(xo, np.cumsum(shares[lab][o]) * 100, lw=2.0, color=col,
                   label="loss window %s" % lab)
        ax[3].plot(xo, np.cumsum(trimmed[lab][o]) * 100, lw=1.4, ls="--", color=col,
                   label="loss window %s, most extreme 1%% of states removed" % lab)
    ax[3].plot([0, 100], [0, 100], "k:", lw=1.4, label="no ranking")
    ax[3].set_xlabel("states ordered by the endpoint cost of their single-step error [%]")
    ax[3].set_ylabel("cumulative share of the loss [%]")
    ax[3].set_title("Does the loss look where the error is?\n"
                    "(lower is better: weight on the costly states)", fontsize=10)
    ax[3].legend(fontsize=7, loc="upper left")
    for x in ax:
        x.grid(alpha=0.3, which="both")
    fig.suptitle("Pre-flight for the momentum window in the loss: trained network "
                 "N = %d, q = %d, dz = %.1f mm, %s states on the start planes"
                 % (N, q, dz, format(len(S), ",")), fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "window_preflight.png"), dpi=130)

    print("wall %.1f s" % rec["wall_s"], flush=True)
    print("ALL WINDOW CHECKS PASS" if ok
          else "A WINDOW CHECK FAILED - see results/check_windowed_weights.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
