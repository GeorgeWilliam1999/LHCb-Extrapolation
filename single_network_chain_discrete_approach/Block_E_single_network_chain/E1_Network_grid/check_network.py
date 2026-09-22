#!/usr/bin/env python
"""E1 gates 1-3: the Block E network before any training.

Gate 1 - the model (at q = 2 and 16, on the 2,589 mm step of N = 2 and the
20.2 mm step of N = 256, on 2,000 RK6 track states drawn over the start planes):
  a. with the last layer zeroed the output is the straight line, exactly;
  b. the torch field integrals and all four scales equal their numpy twins
     (relative difference below 1e-12);
  c. one batch mixing every start plane gives the same outputs as each plane's
     tracks on their own (below 1e-12): the start plane really is per track;
  d. the physics loss and its gradient are finite.

Gate 2 - the loss. For 40 track states per case, the q-stage scheme is solved
exactly without a network (Block C's `exact_solver.solve_state`). Handing
those stage and end states to the shared physics loss, with the start plane per
track, must give a loss at machine precision: below 1e-10 times the loss of the
straight line on the same states. This checks the residual, the per-track stage
planes and the start-plane input together.

Gate 3 - the y scale. On 20,000 (track, plane) pairs per step count, the true
one-step deviation from the straight line (the RK6 state dz further along minus
the straight line) is divided by the network's scale, per component. "Of order
one" is fixed in advance as: median |ratio| between 0.01 and 1.5, and 99th
percentile below 10, for x, y, tx and ty at every N. For comparison, the ratio
for y and ty under Block D's single scale is recorded too.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python check_network.py
Output:  results/check_network.json, figures/y_scale_check.png;
         prints ALL NETWORK CHECKS PASS or the failing check
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import json   # noqa: E402
import sys    # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

import use_shared    # noqa: E402,F401
from _shared.model import LHCbRates, physics_loss            # noqa: E402
from _shared.reference import gauss_legendre, make_field      # noqa: E402
from chain_network import (build, field_integrals_numpy, scales_numpy,  # noqa: E402
                           straight_numpy, znodes_for)

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKS = os.path.join(HERE, "..", "E0_Track_dataset", "results", "tracks.npz")
sys.path.insert(0, os.path.join(use_shared.SHARED_ROOT, "..", "multi_network_chain_discrete_approach",
                                "Block_C_step_size_and_stages", "C2_Exact_scheme_table"))
from exact_solver import solve_state                           # noqa: E402

REL_TOL = 1e-12
LOSS_RATIO_TOL = 1e-10
Y_MED_RANGE = (0.01, 1.5)
Y_P99_MAX = 10.0


def draw_pairs(truth, N, n_max, n, rng, with_next=False):
    """n random (track, start plane) pairs of the chain with N steps."""
    stride = n_max // N
    tr = rng.integers(0, len(truth), n)
    k = rng.integers(0, N, n)
    S = truth[tr, k * stride]
    out = (S, k)
    if with_next:
        out = out + (truth[tr, (k + 1) * stride],)
    return out


class FixedOutputs(torch.nn.Module):
    """A stand-in network that returns given stage and end states."""

    def __init__(self, out, in_scale, q):
        super().__init__()
        self.q, self.n_extra = q, 1
        self.register_buffer("out", torch.as_tensor(out))
        self.register_buffer("in_scale", torch.as_tensor(in_scale))

    def forward(self, S, extra=None):
        return self.out


def main():
    D = np.load(TRACKS)
    Z0, L, n_max = float(D["z0"]), float(D["L"]), int(D["n_max"])
    truth = D["train_truth"]
    fld = make_field(str(D["field"]))
    rates = LHCbRates(make_field(str(D["field"])))
    rng = np.random.default_rng(20260916)
    rec = {"gate1": [], "gate2": [], "gate3": {}}
    ok = True

    # ------------------------------------------------------------------ gate 1 --
    for N in (2, 256):
        dz = L / N
        S, k = draw_pairs(truth, N, n_max, 2000, rng)
        z_start = Z0 + k * dz
        in_scale = S.std(axis=0)
        for q in (2, 16):
            c, A, b = gauss_legendre(q)
            model = build(q, N, L, Z0, in_scale, fld, seed=0)
            St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
            r = {"N": N, "q": q, "n_parameters": int(sum(p.numel() for p in model.parameters()))}
            # (b) twins
            I_B, I_y = model.field_integrals(St, zt)
            nI_B, nI_y = field_integrals_numpy(S, z_start, dz, fld)
            sc = model.residual_scale(St, zt)[:, 0, :].numpy()
            npx, npy, nsx, nsy = scales_numpy(S, dz, nI_B, nI_y)
            twin = np.stack([npx, npy, nsx, nsy], axis=1)
            r["twin_rel_I_B"] = float(np.max(np.abs(I_B.numpy() - nI_B) / nI_B))
            r["twin_rel_I_y"] = float(np.max(np.abs(I_y.numpy() - nI_y) / np.maximum(nI_y, 1e-300)))
            r["twin_rel_scales"] = float(np.max(np.abs(sc - twin) / twin))
            # (c) mixed start planes in one batch
            full = model(St, zt).detach().numpy()
            parts = np.empty_like(full)
            for kk in np.unique(k):
                m = np.flatnonzero(k == kk)
                model._cache = None
                parts[m] = model(torch.as_tensor(S[m]), torch.as_tensor(z_start[m])).detach().numpy()
            model._cache = None
            r["mixed_batch_max_abs"] = float(np.abs(full - parts).max())
            # (d) loss and gradient
            model.zero_grad()
            loss = physics_loss(model, rates, St, dz, znodes_for(model, zt), torch.tensor(A),
                                torch.tensor(b), zt)
            loss.backward()
            r["loss_at_init"] = float(loss.item())
            r["loss_and_grad_finite"] = bool(np.isfinite(loss.item()) and all(
                torch.isfinite(p.grad).all().item() for p in model.parameters()))
            # (a) zeroed last layer
            with torch.no_grad():
                model.net[-1].weight.zero_()
                model.net[-1].bias.zero_()
            model._cache = None
            out = model(St, zt).detach()
            r["zeroed_minus_straight_max_abs"] = float((out - model.straight(St)).abs().max().item())
            r["ok"] = bool(r["zeroed_minus_straight_max_abs"] == 0.0 and r["twin_rel_I_B"] < REL_TOL
                           and r["twin_rel_I_y"] < REL_TOL and r["twin_rel_scales"] < REL_TOL
                           and r["mixed_batch_max_abs"] < REL_TOL and r["loss_and_grad_finite"])
            ok &= r["ok"]
            rec["gate1"].append(r)
            print("gate 1 N=%3d q=%2d: straight %.1e  twins %.1e %.1e %.1e  mixed %.1e  loss %.3e  -> %s"
                  % (N, q, r["zeroed_minus_straight_max_abs"], r["twin_rel_I_B"], r["twin_rel_I_y"],
                     r["twin_rel_scales"], r["mixed_batch_max_abs"], r["loss_at_init"], r["ok"]), flush=True)

    # ------------------------------------------------------------------ gate 2 --
    for N in (2, 256):
        dz = L / N
        S, k = draw_pairs(truth, N, n_max, 40, rng)
        z_start = Z0 + k * dz
        in_scale = truth[:, ::n_max // N][:, :N].reshape(-1, 5).std(axis=0)
        for q in (2, 16):
            tab = gauss_legendre(q)
            c, A, b = tab
            exact = np.empty((len(S), q + 1, 4))
            n_fail, worst = 0, 0.0
            for i in range(len(S)):
                stages, S1, conv, _, resid = solve_state(S[i], z_start[i], dz, tab, fld)
                exact[i, :q] = stages[:, :4]
                exact[i, q] = S1[:4]
                n_fail += (not conv)
                worst = max(worst, float(resid))
            straight = np.stack([np.stack([S[:, 0] + S[:, 2] * cj * dz, S[:, 1] + S[:, 3] * cj * dz,
                                           S[:, 2], S[:, 3]], axis=1) for cj in np.append(c, 1.0)], axis=1)
            St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
            zn = zt[:, None] + torch.as_tensor(c)[None, :] * dz
            losses = {}
            for name, outs in (("exact", exact), ("straight", straight)):
                stub = FixedOutputs(outs, in_scale, q)
                losses[name] = float(physics_loss(stub, rates, St, dz, zn, torch.tensor(A),
                                                  torch.tensor(b), zt).item())
            r = {"N": N, "q": q, "n": int(len(S)), "solves_not_converged": int(n_fail),
                 "solver_worst_residual": worst, "loss_exact": losses["exact"],
                 "loss_straight": losses["straight"],
                 "ratio": losses["exact"] / losses["straight"]}
            r["ok"] = bool(n_fail == 0 and r["ratio"] < LOSS_RATIO_TOL)
            ok &= r["ok"]
            rec["gate2"].append(r)
            print("gate 2 N=%3d q=%2d: loss exact %.3e  straight %.3e  ratio %.1e  unconverged %d -> %s"
                  % (N, q, r["loss_exact"], r["loss_straight"], r["ratio"], n_fail, r["ok"]), flush=True)

    # ------------------------------------------------------------------ gate 3 --
    ratios = {}
    for N in (2, 64, 128, 256):
        dz = L / N
        S, k, S_next = draw_pairs(truth, N, n_max, 20000, rng, with_next=True)
        z_start = Z0 + k * dz
        dev = S_next[:, :4] - straight_numpy(S, dz)
        I_B, I_y = field_integrals_numpy(S, z_start, dz, fld)
        px, py, sx, sy = scales_numpy(S, dz, I_B, I_y)
        rr = {"x": dev[:, 0] / px, "y": dev[:, 1] / py, "tx": dev[:, 2] / sx, "ty": dev[:, 3] / sy,
              "y_blockD_scale": dev[:, 1] / px, "ty_blockD_scale": dev[:, 3] / sx}
        ratios[N] = rr
        g = {}
        n_ok = True
        for name, v in rr.items():
            a = np.abs(v)
            g[name] = {"median": float(np.median(a)), "p99": float(np.quantile(a, 0.99)),
                       "max": float(a.max())}
            if not name.endswith("blockD_scale"):
                good = Y_MED_RANGE[0] <= g[name]["median"] <= Y_MED_RANGE[1] and g[name]["p99"] <= Y_P99_MAX
                g[name]["ok"] = bool(good)
                n_ok &= good
        g["y_floor_active_fraction"] = float(np.mean(
            np.isclose(sy, np.maximum(1e-3 * sx, 1e-12)) & (sy > 1e-12)))
        g["ok"] = bool(n_ok)
        ok &= n_ok
        rec["gate3"][str(N)] = g
        print("gate 3 N=%3d: |dev/scale| median (p99)  x %.3g (%.3g)  y %.3g (%.3g)  tx %.3g (%.3g)  "
              "ty %.3g (%.3g) | Block D scale: y %.3g, ty %.3g | y floor active %.1f%% -> %s"
              % (N, g["x"]["median"], g["x"]["p99"], g["y"]["median"], g["y"]["p99"],
                 g["tx"]["median"], g["tx"]["p99"], g["ty"]["median"], g["ty"]["p99"],
                 g["y_blockD_scale"]["median"], g["ty_blockD_scale"]["median"],
                 100 * g["y_floor_active_fraction"], n_ok), flush=True)

    rec["PASS"] = bool(ok)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "check_network.json"), "w") as f:
        json.dump(rec, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    bins = np.linspace(-6, 2, 81)
    colours = {2: "#1f77b4", 64: "#ff7f0e", 128: "#2ca02c", 256: "#d62728"}
    for j, name in enumerate(("x", "y", "tx", "ty")):
        for N, rr in ratios.items():
            ax[j].hist(np.log10(np.abs(rr[name]) + 1e-300), bins=bins, histtype="step", color=colours[N],
                       lw=1.4, label="N = %d" % N)
            if name in ("y", "ty"):
                ax[j].hist(np.log10(np.abs(rr[name + "_blockD_scale"]) + 1e-300), bins=bins, histtype="step",
                           color=colours[N], lw=1.0, ls="--")
        ax[j].axvspan(np.log10(Y_MED_RANGE[0]), np.log10(Y_MED_RANGE[1]), color="0.9", zorder=0)
        ax[j].set_xlabel("log10 |true one-step deviation / scale|  (%s)" % name)
        ax[j].set_title(name + ("  (dashed: Block D's single scale)" if name in ("y", "ty") else ""), fontsize=10)
    ax[0].set_ylabel("(track, plane) pairs")
    ax[0].legend(fontsize=8)
    fig.suptitle("Gate 3: the output scales against the true RK6 deviation from the straight line, "
                 "20,000 training pairs per N (shaded: the accepted median range)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "y_scale_check.png"), dpi=130)
    print("ALL NETWORK CHECKS PASS" if ok else "NETWORK CHECKS FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
