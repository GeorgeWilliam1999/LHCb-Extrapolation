#!/usr/bin/env python
"""Block F gates 1-4: the reweighted loss, before any training.

Gate 1 - the weights are what they claim to be. For N = 2 and 256, q = 2 and
16, on 2,000 RK6 track states drawn over the start planes, the torch weights of
every mode equal an independent numpy implementation (relative difference below
1e-13), and the lever arm, the window and the per-track factor each take the
values the module's docstring says at known momenta and planes.

Gate 2 - the weighted path reproduces the shared loss. With mode `blockE` the
weights are 1 / in_scale, and `weighted_loss` must then agree with
`_shared.model.physics_loss` to floating-point round-off (below 1e-13
relative). This checks that multiplying the normalised residual back into
physical units and applying weights has not changed the arithmetic.

Gate 3 - the reweighting does not move the minimum. The q-stage scheme is
solved exactly without a network (Block C's `exact_solver.solve_state`) on 40
states per case; handing those stage and end states to the loss must give
machine zero under EVERY mode - below 1e-10 times the straight line's loss on
the same states. A weight cannot move a zero, and this is the check that none
of them does.

Gate 4 - the pre-flight (the one that can still change the design). With Block
E's trained N = 64, q = 2 network, on 8,000 (track, plane) states drawn evenly
over the planes:
  a. where the loss comes from, by momentum band and by position along z,
     under Block E's weighting and each Block F mode;
  b. how well a state's share of the loss ranks with the endpoint cost its
     error actually incurs - the local error against RK6 taken from the SAME
     state, position plus slope times the distance left. That reference is a
     diagnostic only; it never enters the loss, which stays label-free.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python check_weights.py
Output:  results/check_weights.json, results/preflight_shares.csv,
         figures/weighting_preflight.png; prints ALL WEIGHT CHECKS PASS or the failure
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import csv    # noqa: E402
import json   # noqa: E402
import sys    # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

import use_shared    # noqa: E402,F401
from _shared.model import LHCbRates, physics_loss                      # noqa: E402
from _shared.reference import gauss_legendre, make_field, rk6_rows     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(HERE, "..", "..", "Block_E_single_network_chain")
sys.path.insert(0, os.path.join(E, "E1_Network_grid"))
sys.path.insert(0, os.path.join(use_shared.SHARED_ROOT, "..", "multi_network_chain_discrete_approach",
                                "Block_C_step_size_and_stages", "C2_Exact_scheme_table"))
from chain_network import build, load_network, znodes_for     # noqa: E402
from exact_solver import solve_state                          # noqa: E402

import weighted_loss as WL                                    # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

TRACKS = os.path.join(E, "E0_Track_dataset", "results", "tracks.npz")
CASE = os.path.join(E, "E1_Network_grid", "results", "N064_q02")
TWIN_TOL = 1e-13
LOSS_TOL = 1e-13
ZERO_RATIO_TOL = 1e-10
PREFLIGHT_STATES = 8000
RK6_LOCAL_STEP = 1.0
P_EDGES = np.array([1, 2, 5, 10, 20, 50, 100, 200.0])
MODES = ("blockE", "full", "no_lever", "no_track", "no_window")


class FixedOutputs(torch.nn.Module):
    """A stand-in network that returns given stage and end states."""

    def __init__(self, out, in_scale, q):
        super().__init__()
        self.q, self.n_extra = q, 1
        self.register_buffer("out", torch.as_tensor(out))
        self.register_buffer("in_scale", torch.as_tensor(np.asarray(in_scale)))

    def forward(self, S, extra=None):
        return self.out


def draw_pairs(truth, N, n_max, n, rng, even=False):
    """n (track, start plane) pairs; `even` spreads them equally over the planes."""
    stride = n_max // N
    if even:
        per = max(1, n // N)
        tr = rng.integers(0, len(truth), per * N)
        k = np.repeat(np.arange(N), per)
    else:
        tr = rng.integers(0, len(truth), n)
        k = rng.integers(0, N, n)
    return truth[tr, k * stride], k, tr


def weights_numpy(model, S, z_start, const, mode):
    """An independent numpy implementation of `weighted_loss.weights`."""
    q1 = model.q + 1
    cout = np.append(np.asarray(model.c.numpy()), 1.0)
    dz = float(model.dz)
    if mode == "blockE":
        return np.broadcast_to(1.0 / model.in_scale[:4].numpy(), (len(S), q1, 4)).copy()
    sw = WL.MODES[mode]
    z_out = np.asarray(z_start)[:, None] + cout[None, :] * dz
    lev = np.maximum(const["z1"] - z_out, 0.0) + dz
    if not sw["lever"]:
        lev = np.full_like(lev, const["lev_ref"])
    p = 0.299792458 / np.abs(np.asarray(S)[:, 4])
    if sw["window"]:
        w = np.ones_like(p)
        lo = p < const["p_lo"]
        hi = p > const["p_hi"]
        w[lo] = np.exp(-(np.log(p[lo] / const["p_lo"]) / const["rolloff"]) ** 2)
        w[hi] = np.exp(-(np.log(p[hi] / const["p_hi"]) / const["rolloff"]) ** 2)
        w = np.maximum(w, const["w_floor"])
    else:
        w = np.ones_like(p)
    a = np.sqrt(w)
    if sw["track"]:
        D = 1.0e-3 * np.abs(np.asarray(S)[:, 4]) * const["i_bar"] * const["L"]
        a = a * const["D_ref"] / D
    c = const["clamp"]
    if c and c > 1.0:
        med = np.median(a)
        a = np.clip(a, med / c, med * c)
    ones = np.ones_like(lev)
    g = np.stack([ones, ones, lev, lev], axis=-1)
    return a[:, None, None] * g / const["D_ref"]


def main():
    D = np.load(TRACKS)
    Z0, Z1, L, n_max = float(D["z0"]), float(D["z1"]), float(D["L"]), int(D["n_max"])
    truth = D["train_truth"]
    fld = make_field(str(D["field"]))
    rates = LHCbRates(make_field(str(D["field"])))
    ibar = WL.i_bar(fld, Z0, Z1)
    rng = np.random.default_rng(20260918)
    rec = {"i_bar_T_mm": ibar, "z0": Z0, "z1": Z1, "L": L,
           "gate1": [], "gate2": [], "gate3": [], "gate4": {}}
    ok = True

    # ------------------------------------------------------------------ gate 1 --
    for N in (2, 256):
        dz = L / N
        S, k, _ = draw_pairs(truth, N, n_max, 2000, rng)
        z_start = Z0 + k * dz
        in_scale = S.std(axis=0)
        for q in (2, 16):
            model = build(q, N, L, Z0, in_scale, fld, seed=0)
            const = WL.reference_constants(model, S, z_start, Z1, ibar, L)
            St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
            r = {"N": N, "q": q, "worst_twin_rel": 0.0}
            for mode in MODES:
                wt = WL.weights(model, St, zt, const, mode).numpy()
                wn = weights_numpy(model, S, z_start, const, mode)
                rel = float(np.max(np.abs(wt - wn) / np.abs(wn)))
                r["twin_rel_%s" % mode] = rel
                r["worst_twin_rel"] = max(r["worst_twin_rel"], rel)
            lev = WL.lever_arms(model.cout, dz, zt, Z1).numpy()
            r["lever_max_mm"] = float(lev.max())
            r["lever_min_mm"] = float(lev.min())
            r["lever_expected_max_mm"] = float(L + dz * (1.0 - float(model.c[0])))
            r["ok"] = bool(r["worst_twin_rel"] < TWIN_TOL
                           and abs(r["lever_min_mm"] - dz) < 1e-9 * max(dz, 1.0)
                           and abs(r["lever_max_mm"] - r["lever_expected_max_mm"]) < 1e-9 * L)
            ok &= r["ok"]
            rec["gate1"].append(r)
            print("gate 1 N=%3d q=%2d: twins %.1e  lever %.1f .. %.1f mm -> %s"
                  % (N, q, r["worst_twin_rel"], r["lever_min_mm"], r["lever_max_mm"], r["ok"]),
                  flush=True)
    # the window at named momenta
    pw = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
    rec["window_at"] = {("%g GeV" % p): float(v) for p, v in zip(pw, WL.band_window(pw))}
    print("gate 1 window: " + "  ".join("%g GeV %.3f" % (p, v)
                                        for p, v in zip(pw, WL.band_window(pw))), flush=True)

    # ------------------------------------------------------------------ gate 2 --
    for N in (2, 256):
        dz = L / N
        S, k, _ = draw_pairs(truth, N, n_max, 500, rng)
        z_start = Z0 + k * dz
        in_scale = S.std(axis=0)
        for q in (2, 16):
            c, A, b = gauss_legendre(q)
            model = build(q, N, L, Z0, in_scale, fld, seed=0)
            const = WL.reference_constants(model, S, z_start, Z1, ibar, L)
            St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
            zn = znodes_for(model, zt)
            At, bt = torch.tensor(A), torch.tensor(b)
            shared = float(physics_loss(model, rates, St, dz, zn, At, bt, zt).item())
            w = WL.weights(model, St, zt, const, "blockE")
            mine = float(WL.weighted_loss(model, rates, St, dz, zn, At, bt, zt, w).item())
            r = {"N": N, "q": q, "shared": shared, "weighted_path": mine,
                 "rel": abs(mine - shared) / shared}
            r["ok"] = bool(r["rel"] < LOSS_TOL)
            ok &= r["ok"]
            rec["gate2"].append(r)
            print("gate 2 N=%3d q=%2d: shared %.6e  weighted path %.6e  rel %.1e -> %s"
                  % (N, q, shared, mine, r["rel"], r["ok"]), flush=True)

    # ------------------------------------------------------------------ gate 3 --
    for N in (2, 256):
        dz = L / N
        S, k, _ = draw_pairs(truth, N, n_max, 40, rng)
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
            const = WL.reference_constants(model, S, z_start, Z1, ibar, L)
            St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
            zn = znodes_for(model, zt)
            At, bt = torch.tensor(A), torch.tensor(b)
            r = {"N": N, "q": q, "solves_not_converged": int(n_fail), "worst_ratio": 0.0}
            for mode in MODES:
                w = WL.weights(model, St, zt, const, mode)
                ls = {}
                for name, outs in (("exact", exact), ("straight", straight)):
                    stub = FixedOutputs(outs, in_scale, q)
                    ls[name] = float(WL.weighted_loss(stub, rates, St, dz, zn, At, bt, zt, w).item())
                ratio = ls["exact"] / ls["straight"]
                r["ratio_%s" % mode] = ratio
                r["worst_ratio"] = max(r["worst_ratio"], ratio)
            r["ok"] = bool(n_fail == 0 and r["worst_ratio"] < ZERO_RATIO_TOL)
            ok &= r["ok"]
            rec["gate3"].append(r)
            print("gate 3 N=%3d q=%2d: exact/straight worst over modes %.1e -> %s"
                  % (N, q, r["worst_ratio"], r["ok"]), flush=True)

    # ------------------------------------------------------------------ gate 4 --
    print("gate 4: the pre-flight on Block E's trained N = 64, q = 2 network", flush=True)
    model = load_network(CASE, fld)
    N, q = 64, 2
    dz = L / N
    c, A, b = gauss_legendre(q)
    At, bt = torch.tensor(A), torch.tensor(b)
    S, k, tr = draw_pairs(truth, N, n_max, PREFLIGHT_STATES, rng, even=True)
    z_start = Z0 + k * dz
    P = 0.299792458 / np.abs(S[:, 4])
    St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
    zn = znodes_for(model, zt)
    const = WL.reference_constants(model, S, z_start, Z1, ibar, L)

    from _shared.model import reconstruction_residuals
    with torch.no_grad():
        r_phys = (reconstruction_residuals(model, rates, St, dz, zn, At, bt, zt)
                  * model.in_scale[:4]).numpy()

    # what each state's error actually costs at z1 (diagnostic only)
    out = model(St, zt).detach().numpy()[:, -1, :]
    ref = rk6_rows(S, z_start, z_start + dz, step=RK6_LOCAL_STEP, field=fld)
    lev_end = (Z1 - (z_start + dz)) + dz
    cost = np.hypot((out[:, 0] - ref[:, 0]) + (out[:, 2] - ref[:, 2]) * lev_end,
                    (out[:, 1] - ref[:, 1]) + (out[:, 3] - ref[:, 3]) * lev_end) * 1e3   # um

    from scipy.stats import spearmanr
    rows, shares = [], {}
    for mode in MODES:
        w = WL.weights(model, St, zt, const, mode)
        w = w.numpy() if torch.is_tensor(w) else np.asarray(w)
        per_state = ((r_phys * w) ** 2).mean(axis=(1, 2))
        share = per_state / per_state.sum()
        shares[mode] = share
        rho = float(spearmanr(share, cost).statistic)
        in_band = (P >= 10) & (P < 50)
        rho_band = float(spearmanr(share[in_band], cost[in_band]).statistic)
        for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
            m = (P >= lo) & (P < hi)
            rows.append(dict(mode=mode, kind="momentum", lo=lo, hi=hi, n=int(m.sum()),
                             share_pct=100 * float(share[m].sum()),
                             tracks_pct=100 * float(m.mean())))
        for j in range(4):
            m = (k >= j * N // 4) & (k < (j + 1) * N // 4)
            rows.append(dict(mode=mode, kind="quarter_of_z", lo=j + 1, hi=j + 1, n=int(m.sum()),
                             share_pct=100 * float(share[m].sum()),
                             tracks_pct=100 * float(m.mean())))
        band = (P >= 10) & (P < 50)
        rec["gate4"][mode] = dict(
            spearman_share_vs_cost=rho,
            spearman_share_vs_cost_in_band=rho_band,
            share_10_50_GeV_in_first_quarter_pct=100 * float(
                share[in_band & (k < N // 4)].sum() / max(share[in_band].sum(), 1e-300)),
            share_10_50_GeV_in_last_quarter_pct=100 * float(
                share[in_band & (k >= 3 * N // 4)].sum() / max(share[in_band].sum(), 1e-300)),
            share_2_5_GeV_pct=100 * float(share[(P >= 2) & (P < 5)].sum()),
            share_10_50_GeV_pct=100 * float(share[band].sum()),
            share_above_50_GeV_pct=100 * float(share[P >= 50].sum()),
            share_first_quarter_pct=100 * float(share[k < N // 4].sum()),
            share_last_quarter_pct=100 * float(share[k >= 3 * N // 4].sum()),
            top_1pct_of_states_share_pct=100 * float(
                np.sort(share)[::-1][:max(1, len(share) // 100)].sum()))
        print("  %-10s 2-5 GeV %5.1f%%  10-50 GeV %5.1f%%  >50 GeV %5.1f%%  |  first quarter of z "
              "%5.1f%%  last %5.1f%%  |  top 1%% of states %5.1f%%  |  rho(share, cost) %+.3f"
              % (mode, rec["gate4"][mode]["share_2_5_GeV_pct"],
                 rec["gate4"][mode]["share_10_50_GeV_pct"],
                 rec["gate4"][mode]["share_above_50_GeV_pct"],
                 rec["gate4"][mode]["share_first_quarter_pct"],
                 rec["gate4"][mode]["share_last_quarter_pct"],
                 rec["gate4"][mode]["top_1pct_of_states_share_pct"], rho), flush=True)
        print("             within 10-50 GeV: rho %+.3f, and %.0f%% of the band's loss is in the "
              "first quarter of z, %.0f%% in the last"
              % (rho_band, rec["gate4"][mode]["share_10_50_GeV_in_first_quarter_pct"],
                 rec["gate4"][mode]["share_10_50_GeV_in_last_quarter_pct"]), flush=True)

    # the effect of the clamp, which is the knob the pre-flight sets
    rec["clamp_scan"] = {}
    for cl in (0.0, 2.0, 3.0, 5.0, 10.0, 20.0):
        cc = dict(const)
        cc["clamp"] = cl
        w = WL.weights(model, St, zt, cc, "full").numpy()
        per_state = ((r_phys * w) ** 2).mean(axis=(1, 2))
        share = per_state / per_state.sum()
        band = (P >= 10) & (P < 50)
        rec["clamp_scan"]["%g" % cl] = dict(
            share_2_5_GeV_pct=100 * float(share[(P >= 2) & (P < 5)].sum()),
            share_10_50_GeV_pct=100 * float(share[band].sum()),
            share_above_50_GeV_pct=100 * float(share[P >= 50].sum()),
            top_1pct_of_states_share_pct=100 * float(
                np.sort(share)[::-1][:max(1, len(share) // 100)].sum()),
            spearman_share_vs_cost=float(spearmanr(share, cost).statistic))
        s = rec["clamp_scan"]["%g" % cl]
        print("  clamp %4s: 2-5 GeV %5.1f%%  10-50 GeV %5.1f%%  >50 GeV %5.1f%%  top 1%% %5.1f%%  "
              "rho %+.3f" % ("off" if cl == 0 else "%g" % cl, s["share_2_5_GeV_pct"],
                             s["share_10_50_GeV_pct"], s["share_above_50_GeV_pct"],
                             s["top_1pct_of_states_share_pct"], s["spearman_share_vs_cost"]),
              flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "preflight_shares.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    rec["all_pass"] = bool(ok)
    with open(os.path.join(HERE, "results", "check_weights.json"), "w") as f:
        json.dump(rec, f, indent=1)

    # ------------------------------------------------------------------ figure --
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    centres = np.sqrt(P_EDGES[:-1] * P_EDGES[1:])
    for mode in MODES:
        sh = [shares[mode][(P >= lo) & (P < hi)].sum() * 100
              for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:])]
        ax[0].plot(centres, sh, "-o", ms=4, label=mode)
        qs = [shares[mode][(k >= j * N // 4) & (k < (j + 1) * N // 4)].sum() * 100 for j in range(4)]
        ax[1].plot([1, 2, 3, 4], qs, "-o", ms=4, label=mode)
    frac = np.array([(P >= lo) & (P < hi) for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:])]).mean(axis=1) * 100
    ax[0].plot(centres, frac, "k--", lw=1, label="share of the states")
    ax[0].axvspan(10, 50, color="0.9", zorder=0)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("momentum [GeV]")
    ax[0].set_ylabel("share of the loss [%]")
    ax[0].set_title("Where the loss comes from, by momentum\n(shaded: the 10-50 GeV band)", fontsize=10)
    ax[0].legend(fontsize=7)
    ax[1].axhline(25, color="k", ls="--", lw=1, label="flat")
    ax[1].set_xticks([1, 2, 3, 4], ["first", "second", "third", "fourth"])
    ax[1].set_xlabel("quarter of the crossing the step starts in")
    ax[1].set_ylabel("share of the loss [%]")
    ax[1].set_title("Where the loss comes from, along z", fontsize=10)
    ax[1].legend(fontsize=7)
    for mode in ("blockE", "full"):
        o = np.argsort(cost)
        ax[2].plot(np.arange(len(o)) / len(o) * 100, np.cumsum(shares[mode][o]) * 100,
                   label=mode)
    ax[2].plot([0, 100], [0, 100], "k--", lw=1, label="no ranking")
    ax[2].set_xlabel("states ordered by their true cost at z1 [%]")
    ax[2].set_ylabel("cumulative share of the loss [%]")
    ax[2].set_title("Does the loss look where the error is?\n(lower is better: weight on the costly states)",
                    fontsize=10)
    ax[2].legend(fontsize=7)
    for x in ax:
        x.grid(alpha=0.3, which="both")
    fig.suptitle("Block F pre-flight: Block E's N = 64, q = 2 network, %d states on the start planes"
                 % len(S), fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "weighting_preflight.png"), dpi=130)

    print("ALL WEIGHT CHECKS PASS" if ok else "A WEIGHT CHECK FAILED - see results/check_weights.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
