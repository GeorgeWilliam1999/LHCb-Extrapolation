#!/usr/bin/env python
"""A4.1 - is the magnet-up map the map we think it is, and how does it differ
from the magnet-down one?

Three things:

  1. identity     the file path and the md5 of the up map, beside the down one;
  2. parity       the differentiable torch twin of the field, which the physics
                  loss takes its gradients through, must agree with the numpy
                  loader on 200,000 random in-map points for the UP map (the
                  same gate that was run for down);
  3. shape        By and |B| sampled along the frozen leg's straight line and
                  along 300 real cross-magnet (leg B) straight-line paths, for
                  both polarities, so that "up is down with By flipped" can be
                  checked rather than assumed.

    results/field_up_parity.json    identity + parity + the flip measurement
    figures/field_up_vs_down.png    the profiles
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json    # noqa: E402

import numpy as np      # noqa: E402
import matplotlib       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import torch            # noqa: E402

import use_shared       # noqa: F401,E402
from _shared.field_torch import parity              # noqa: E402
from _shared.reference import (FROZEN_LEG, field_bounds, field_md5,   # noqa: E402
                               field_path, load_training, make_field)

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")

SURFACE, TEXT1, TEXT2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, GREEN, MAGENTA, NEUTRAL = "#2a78d6", "#008300", "#e87ba4", "#c9c8c2"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": TEXT1, "axes.edgecolor": TEXT2,
    "axes.labelcolor": TEXT1, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.grid": True, "grid.color": "#e7e6e1", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 10.5,
})


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    out = {}

    # -- 1. identity -------------------------------------------------------
    for which in ("up", "down"):
        out[which] = {"file": field_path(which), "md5": field_md5(which)}
        print(which, out[which])
    fu, fd = make_field("up"), make_field("down")
    lo, hi = field_bounds(fu)
    out["map_bounds_mm"] = {"lo": lo.tolist(), "hi": hi.tolist()}
    out["bounds_identical"] = bool(
        np.array_equal(lo, field_bounds(fd)[0]) and
        np.array_equal(hi, field_bounds(fd)[1]))

    # -- 2. the torch twin on the up map -----------------------------------
    out["parity_up"] = parity("up", n=200_000)
    out["parity_down"] = parity("down", n=200_000)

    # -- 3. is up the mirror of down? --------------------------------------
    # random in-map points, both maps, component by component
    rng = np.random.default_rng(0)
    pts = lo + rng.random((200_000, 3)) * (hi - lo)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    Bu = np.stack(fu(x, y, z), axis=1)
    Bd = np.stack(fd(x, y, z), axis=1)
    scale = np.abs(Bd).max()
    out["mirror_test"] = {
        "n_points": 200_000,
        "max_abs_B_down_T": float(scale),
        "max_abs_Bup_plus_Bdown_T": [float(np.abs(Bu[:, i] + Bd[:, i]).max())
                                     for i in range(3)],
        "exact_sign_flip": bool(np.array_equal(Bu, -Bd)),
        "max_abs_diff_magnitude_T": float(np.abs(
            np.linalg.norm(Bu, axis=1) - np.linalg.norm(Bd, axis=1)).max()),
        "median_abs_diff_magnitude_T": float(np.median(np.abs(
            np.linalg.norm(Bu, axis=1) - np.linalg.norm(Bd, axis=1)))),
    }

    # -- the profiles ------------------------------------------------------
    z0, z1 = FROZEN_LEG["z0"], FROZEN_LEG["z1"]
    zg = np.linspace(z0, z1, 400)
    zeros = np.zeros_like(zg)
    prof_axis = {w: np.stack(f(zeros, zeros, zg), axis=1)
                 for w, f in (("up", fu), ("down", fd))}

    d = load_training(split="train", leg="B")
    X = d["X"].astype(np.float64)
    fwd = X[:, 6] > X[:, 5]
    X = X[fwd]
    sel = np.random.default_rng(3).choice(len(X), size=min(300, len(X)),
                                          replace=False)
    zl = np.linspace(2600, 7900, 200)
    prof_legs = {}
    for w, f in (("up", fu), ("down", fd)):
        P = np.empty((len(sel), len(zl), 3))
        for i, r in enumerate(sel):
            xs = X[r, 0] + X[r, 2] * (zl - X[r, 5])
            ys = X[r, 1] + X[r, 3] * (zl - X[r, 5])
            P[i] = np.stack(f(xs, ys, zl), axis=1)
        prof_legs[w] = P

    out["frozen_leg_axis_profile"] = {
        "z_mm": [z0, z1],
        "By_up_max_T": float(np.abs(prof_axis["up"][:, 1]).max()),
        "By_down_max_T": float(np.abs(prof_axis["down"][:, 1]).max()),
        "By_up_signed_at_peak_T": float(
            prof_axis["up"][np.argmax(np.abs(prof_axis["up"][:, 1])), 1]),
        "By_down_signed_at_peak_T": float(
            prof_axis["down"][np.argmax(np.abs(prof_axis["down"][:, 1])), 1]),
    }
    out["real_leg_profile"] = {
        "n_legs": int(len(sel)),
        "median_signed_By_integral_up_Tmm": float(np.median(
            np.trapz(prof_legs["up"][:, :, 1], zl, axis=1))),
        "median_signed_By_integral_down_Tmm": float(np.median(
            np.trapz(prof_legs["down"][:, :, 1], zl, axis=1))),
        "max_abs_|B|_difference_T": float(np.abs(
            np.linalg.norm(prof_legs["up"], axis=2)
            - np.linalg.norm(prof_legs["down"], axis=2)).max()),
    }

    with open(os.path.join(RES, "field_up_parity.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ---------------------------------------------------------------- plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    ax[0].axhline(0, color=NEUTRAL, lw=1)
    ax[0].plot(zg / 1000, prof_axis["down"][:, 1], color=BLUE, lw=2, label="MagDown")
    ax[0].plot(zg / 1000, prof_axis["up"][:, 1], color=MAGENTA, lw=2, ls="--",
               label="MagUp")
    ax[0].set_xlabel("z [m]"); ax[0].set_ylabel("By [T]")
    ax[0].legend(fontsize=8.5)
    ax[0].set_title("By on the frozen leg's axis (x = y = 0)", fontsize=11)

    for w, col, ls in (("down", BLUE, "-"), ("up", MAGENTA, "--")):
        By = prof_legs[w][:, :, 1]
        med = np.median(By, axis=0)
        loq, hiq = np.percentile(By, [10, 90], axis=0)
        ax[1].fill_between(zl / 1000, loq, hiq, color=col, alpha=0.18, lw=0)
        ax[1].plot(zl / 1000, med, color=col, lw=2, ls=ls,
                   label="Mag%s median By" % w.capitalize())
    ax[1].axhline(0, color=NEUTRAL, lw=1)
    ax[1].set_xlabel("z [m]"); ax[1].set_ylabel("By [T]")
    ax[1].legend(fontsize=8.5)
    ax[1].set_title("By along %d real cross-magnet legs" % len(sel), fontsize=11)

    for w, col, ls in (("down", BLUE, "-"), ("up", MAGENTA, "--")):
        mag = np.linalg.norm(prof_legs[w], axis=2)
        ax[2].plot(zl / 1000, np.median(mag, axis=0), color=col, lw=2, ls=ls,
                   label="Mag%s median |B|" % w.capitalize())
    ax[2].set_xlabel("z [m]"); ax[2].set_ylabel("|B| [T]")
    ax[2].legend(fontsize=8.5)
    ax[2].set_title("field magnitude: identical for the two polarities",
                    fontsize=11)

    fig.suptitle("A4.1 the magnet-up map: identity, parity and shape "
                 "(By flips sign, |B| is unchanged)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "field_up_vs_down.png"), dpi=150)
    print("wrote", os.path.join(FIG, "field_up_vs_down.png"))
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("mirror_test", "frozen_leg_axis_profile",
                               "real_leg_profile")}, indent=1))


if __name__ == "__main__":
    main()
