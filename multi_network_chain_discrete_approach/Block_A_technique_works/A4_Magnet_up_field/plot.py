#!/usr/bin/env python
"""A4 figure: the label-free network on the magnet-up field.

    figures/magnet_up_results.png

Panel 1  endpoint-error distributions on the test split - the up-field
         physics (label-free) runs pooled over all seeds, with each seed's own
         median marked; the up-field supervised twin; the magnet-down physics
         runs from `../../Block_0_first_pass/S2b_One_step_network_v2/results/predictions.npz` (which
         carries per-track errors); the two exact-scheme ceilings and the
         straight line.
Panel 2  the same errors against track momentum, where the two polarities can
         be compared track by track.
Panel 3  three test tracks drawn in the (z, x) plane: the fp64 RK4 reference on
         each polarity from the SAME start state, and each network's own path
         through the eight Gauss-Legendre stage planes. The bend changes sign.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import glob    # noqa: E402
import json    # noqa: E402

import numpy as np       # noqa: E402
import matplotlib        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import torch             # noqa: E402

import use_shared        # noqa: F401,E402
from _shared.evaluate import predict           # noqa: E402
from _shared.model import OneStepNetwork       # noqa: E402
from _shared.reference import make_field, rk4_rows   # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
V2 = os.path.join(HERE, "..", "..", "Block_0_first_pass", "S2b_One_step_network_v2", "results")
A1 = os.path.join(HERE, "..", "A2_Network_size_and_seed_study", "results")

SURFACE, TEXT1, TEXT2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, GREEN, MAGENTA, YELLOW, NEUTRAL = ("#2a78d6", "#008300", "#e87ba4",
                                         "#eda100", "#c9c8c2")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": TEXT1, "axes.edgecolor": TEXT2,
    "axes.labelcolor": TEXT1, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.grid": True, "grid.color": "#e7e6e1", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 10.5,
})


def endpoint_errors_um(out, ref):
    """(N,) endpoint error in um, in the shared metric: the larger of |dx|, |dy|
    at the leg's end plane (`_shared/evaluate.py::score_against_reference`)."""
    return 1e3 * np.abs(out[:, -1, :2] - ref[:, -1, :2]).max(axis=1)


def load_up_models(data):
    """Every converged up-field run, as (mode, seed, model)."""
    got = []
    for p in sorted(glob.glob(os.path.join(RES, "up_*.json"))):
        with open(p) as f:
            r = json.load(f)
        ck = os.path.join(RES, r["tag"] + ".pt")
        if not os.path.exists(ck):
            continue
        m = OneStepNetwork(int(data["q"]), data["in_scale"], data["out_scale"],
                           width=r["width"], depth=r["depth"], n_extra=0)
        m.load_state_dict(torch.load(ck, weights_only=True))
        m.eval()
        got.append((r["mode"], r["seed"], m, r))
    return got


def main():
    os.makedirs(FIG, exist_ok=True)
    up = np.load(os.path.join(RES, "frozen_leg_up.npz"))
    S = up["test_S"]
    ref = up["test_ref"]
    P = up["test_P"]
    z0, z1 = float(up["z0"]), float(up["z1"])
    zout = up["zout"]

    models = load_up_models(up)
    errs = {}
    for mode, seed, m, r in models:
        out = np.asarray(predict(m, torch.tensor(S)))
        errs[(mode, seed)] = endpoint_errors_um(out, ref)

    phys = [errs[k] for k in errs if k[0] == "physics"]
    twin = [errs[k] for k in errs if k[0] == "data"]

    # the magnet-down per-track errors. Two sets: the three-seed v2 baseline,
    # which stored them, and A1's ten seeds at this same 50x4 architecture,
    # recomputed here from its checkpoints on its own dataset - that is the
    # like-for-like comparator, being the same architecture and the same
    # ten-seed protocol on one thread.
    dn_pred = np.load(os.path.join(V2, "predictions.npz"))
    dn_phys = [dn_pred["physics_seed%d_end_err_um" % s] for s in (0, 1, 2)]
    dn_P = dn_pred["test_P"]

    a1_phys, a1_P = [], None
    a1_npz = os.path.join(A1, "frozen_leg_q08.npz")
    if os.path.exists(a1_npz):
        a1 = np.load(a1_npz, allow_pickle=False)
        a1_P = a1["test_P"]
        for seed in range(10):
            ck = os.path.join(A1, "w050_d4_s%d.pt" % seed)
            if not os.path.exists(ck):
                continue
            m = OneStepNetwork(int(a1["q"]), a1["in_scale"], a1["out_scale"],
                               width=50, depth=4, n_extra=0)
            m.load_state_dict(torch.load(ck, weights_only=True))
            m.eval()
            a1_phys.append(endpoint_errors_um(
                np.asarray(predict(m, torch.tensor(a1["test_S"]))),
                a1["test_ref"]))

    with open(os.path.join(RES, "ceiling_summary.json")) as f:
        # the like-for-like ceiling: the exact scheme solved on these very
        # test states, not on the 32 momentum-stratified legs
        ceil = {k: {"median_err_um": v["endpoint_med_um"]}
                for k, v in json.load(f)["on_the_test_states"].items()}
    straight = 1e3 * np.abs(
        S[:, :2] + S[:, 2:4] * (z1 - z0) - ref[:, -1, :2]).max(axis=1)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # ---------------------------------------------------- panel 1: the spread
    bins = np.logspace(0, 7, 70)

    def pooled(sets, **kw):
        """Several seeds' errors as one curve, divided by the number of seeds,
        so that runs with different seed counts are on the same vertical axis."""
        e = np.concatenate(sets)
        ax[0].hist(e, bins=bins, weights=np.full(len(e), 1.0 / len(sets)), **kw)

    if phys:
        pooled(phys, histtype="stepfilled", color=GREEN, alpha=0.30, lw=0,
               label="MagUp, physics loss (%d seeds)" % len(phys))
        pooled(phys, histtype="step", color=GREEN, lw=2)
        for e in phys:
            ax[0].axvline(np.median(e), color=GREEN, lw=0.9, alpha=0.55)
    if twin:
        pooled(twin, histtype="step", color=MAGENTA, lw=2, ls="--",
               label="MagUp, data loss (%d seeds)" % len(twin))
    if a1_phys:
        pooled(a1_phys, histtype="step", color=BLUE, lw=2,
               label="MagDown, physics loss (%d seeds, A1 50x4)" % len(a1_phys))
    pooled(dn_phys, histtype="step", color=BLUE, lw=1.3, ls=":",
           label="MagDown, physics loss (3 seeds, v2)")
    pooled([straight], histtype="step", color=NEUTRAL, lw=2,
           label="straight line")
    ax[0].axvline(ceil["up"]["median_err_um"], color=GREEN, lw=1.6, ls=":")
    ax[0].axvline(ceil["down"]["median_err_um"], color=BLUE, lw=1.6, ls=":")
    ax[0].annotate("exact-scheme ceiling\non these states:\n%.0f um up, %.0f um down"
                   % (ceil["up"]["median_err_um"], ceil["down"]["median_err_um"]),
                   xy=(ceil["up"]["median_err_um"], ax[0].get_ylim()[1] * 0.22),
                   xytext=(1.6, ax[0].get_ylim()[1] * 0.60),
                   fontsize=8, color=TEXT2, ha="left", va="center",
                   arrowprops=dict(arrowstyle="->", color=TEXT2, lw=0.9))
    ax[0].set_xscale("log")
    ax[0].set_xlabel("endpoint error [um]")
    ax[0].set_ylabel("test tracks per seed")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title("endpoint error on the test split\n"
                    "(thin vertical lines: each MagUp physics seed's median)",
                    fontsize=11)

    # ------------------------------------------------- panel 2: vs momentum
    if phys:
        ax[1].scatter(P, np.median(np.stack(phys), axis=0), s=6, alpha=0.35,
                      color=GREEN, lw=0, label="MagUp physics (seed median)")
    d2_phys, d2_P = (a1_phys, a1_P) if a1_phys else (dn_phys, dn_P)
    ax[1].scatter(d2_P, np.median(np.stack(d2_phys), axis=0), s=6, alpha=0.35,
                  color=BLUE, lw=0, label="MagDown physics (seed median)")
    for pol, e, pp, col in (("up", phys, P, GREEN), ("down", d2_phys, d2_P, BLUE)):
        if not e:
            continue
        med = np.median(np.stack(e), axis=0)
        edges = np.logspace(np.log10(max(pp.min(), 0.2)), np.log10(pp.max()), 14)
        idx = np.digitize(pp, edges)
        cx = [0.5 * (edges[i - 1] + edges[i]) for i in range(1, len(edges))
              if (idx == i).sum() > 3]
        cy = [np.median(med[idx == i]) for i in range(1, len(edges))
              if (idx == i).sum() > 3]
        ax[1].plot(cx, cy, color=col, lw=2.4,
                   label="Mag%s binned median" % pol.capitalize())
    ax[1].axhline(ceil["up"]["median_err_um"], color=NEUTRAL, lw=1.4, ls=":")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("track momentum p [GeV]")
    ax[1].set_ylabel("endpoint error [um]")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_title("where the error lives in momentum", fontsize=11)

    # -------------------------------------------- panel 3: three trajectories
    order = np.argsort(P)
    pick = [order[int(f * (len(order) - 1))] for f in (0.10, 0.50, 0.90)]
    zg = np.linspace(z0, z1, 60)
    up_f, dn_f = make_field("up"), make_field("down")

    up_model = next((m for md, sd, m, _ in models
                     if md == "physics" and sd == 0), None)
    if up_model is None and models:
        up_model = models[0][2]
    dn_ck = os.path.join(V2, "one_step_physics_seed0.pt")
    dn_data = np.load(os.path.join(V2, "frozen_leg_data.npz"))
    dn_model = OneStepNetwork(8, dn_data["in_scale"], dn_data["out_scale"],
                              width=50, depth=4, n_extra=0)
    dn_model.load_state_dict(torch.load(dn_ck, weights_only=True))
    dn_model.eval()

    Ssel = S[pick]
    up_out = (np.asarray(predict(up_model, torch.tensor(Ssel)))
              if up_model is not None else None)
    dn_out = np.asarray(predict(dn_model, torch.tensor(Ssel)))

    for j, i in enumerate(pick):
        s0 = S[i:i + 1]
        for fld, col, name in ((up_f, GREEN, "MagUp"), (dn_f, BLUE, "MagDown")):
            xs = [s0[0, 0]]
            cur, zp = s0.copy(), z0
            for zt in zg[1:]:
                cur = rk4_rows(cur, np.array([zp]), np.array([zt]), field=fld)
                xs.append(cur[0, 0]); zp = zt
            ax[2].plot(zg / 1000, xs, color=col, lw=1.8, alpha=0.85,
                       label=("%s RK4 reference" % name) if j == 0 else None)
        if up_out is not None:
            ax[2].plot(zout / 1000, up_out[j, :, 0], "o", ms=4.5, color=GREEN,
                       mfc="none", mew=1.4,
                       label="MagUp network stages" if j == 0 else None)
        ax[2].plot(zout / 1000, dn_out[j, :, 0], "s", ms=4.0, color=BLUE,
                   mfc="none", mew=1.4,
                   label="MagDown network stages" if j == 0 else None)
        ax[2].annotate("p = %.1f GeV" % P[i], (zg[-1] / 1000, xs[-1]),
                       fontsize=8, color=TEXT2, xytext=(3, 0),
                       textcoords="offset points", va="center")
    ax[2].set_xlabel("z [m]"); ax[2].set_ylabel("x [mm]")
    ax[2].legend(fontsize=8, loc="best")
    ax[2].set_title("three test tracks, both polarities\n"
                    "(same start state; the bend changes sign)", fontsize=11)

    fig.suptitle("A4 - the one-step network trained with no labels on the "
                 "magnet-up field", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "magnet_up_results.png"), dpi=150)
    print("wrote", os.path.join(FIG, "magnet_up_results.png"))

    np.savez_compressed(
        os.path.join(RES, "test_errors.npz"),
        test_P=P, straight_um=straight,
        **{"up_%s_seed%d_end_err_um" % (m, s): errs[(m, s)] for (m, s) in errs})
    print("wrote", os.path.join(RES, "test_errors.npz"))


if __name__ == "__main__":
    main()
