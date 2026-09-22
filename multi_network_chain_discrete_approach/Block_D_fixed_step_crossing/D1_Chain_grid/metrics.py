#!/usr/bin/env python
"""The scoring of Block D, in one place, per component.

A track state is (x, y, tx, ty, q/p). The two positions and the two slopes
have different physics - a slope error is what the magnet does to the track,
a position error is that slope error integrated over the rest of the crossing
- so every score is reported per component (George 2026-09-14), alongside the
programme's summary measures:

    pos   = max(|dx|, |dy|)  in um        slope = max(|dtx|, |dty|)  in mrad
    x, y  in um   tx, ty  in mrad          each with median, p95, mean of the
                                          absolute error, and the signed mean
                                          (the bias)
    q/p   must not change at all: it is carried through every leg, and the
          maximum |dq/p| over the chain is recorded and must be exactly 0.

`chain_scores` is the chain's whole verdict from its predicted states on every
plane, used by train_chain.py when a chain finishes and by score_components.py
to rescore a chain from its states.npz.
"""
from __future__ import annotations

import numpy as np

import use_shared                                        # noqa: F401
from _shared.prepare import P_BANDS
from _shared.reference import RK6_STEP, rk6_rows

COMPONENTS = (("x", 0, 1e3, "um"), ("y", 1, 1e3, "um"),
              ("tx", 2, 1e3, "mrad"), ("ty", 3, 1e3, "mrad"))
SCORED = ("val", "test")


def pos_err_um(pred, truth):
    return np.abs(pred[:, :2] - truth[:, :2]).max(axis=1) * 1e3


def slope_err_mrad(pred, truth):
    return np.abs(pred[:, 2:4] - truth[:, 2:4]).max(axis=1) * 1e3


def _q(a, p):
    return float(np.quantile(a, p))


def stats(pos, slope, pband=None):
    out = {"pos_med_um": float(np.median(pos)), "pos_p95_um": _q(pos, 0.95),
           "pos_mean_um": float(pos.mean()),
           "slope_med_mrad": float(np.median(slope)), "slope_p95_mrad": _q(slope, 0.95),
           "n": int(len(pos))}
    if pband is not None:
        out["by_p_band"] = {}
        for i, (lo, hi) in enumerate(P_BANDS):
            m = pband == i
            if m.any():
                out["by_p_band"]["%g-%g GeV" % (lo, hi)] = {
                    "pos_med_um": float(np.median(pos[m])), "pos_p95_um": _q(pos[m], 0.95),
                    "slope_med_mrad": float(np.median(slope[m])), "n": int(m.sum())}
    return out


def component_stats(pred, truth, pband=None):
    """Per component: median, p95 and mean of |error|, the signed mean (bias);
    plus the q/p check."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    out = {}
    for name, i, scale, unit in COMPONENTS:
        d = (pred[:, i] - truth[:, i]) * scale
        a = np.abs(d)
        out["%s_med_%s" % (name, unit)] = float(np.median(a))
        out["%s_p95_%s" % (name, unit)] = _q(a, 0.95)
        out["%s_mean_%s" % (name, unit)] = float(a.mean())
        out["%s_bias_%s" % (name, unit)] = float(d.mean())
        if pband is not None:
            for j, (lo, hi) in enumerate(P_BANDS):
                m = pband == j
                if m.any():
                    out.setdefault("by_p_band", {}).setdefault("%g-%g GeV" % (lo, hi), {})[
                        "%s_med_%s" % (name, unit)] = float(np.median(a[m]))
    if pred.shape[1] > 4 and truth.shape[1] > 4:
        out["qop_max_abs_change"] = float(np.abs(pred[:, 4] - truth[:, 4]).max())
    return out


def per_plane_curves(states_s, truth_s, stride, N):
    keys = ["pos_med_um", "pos_p95_um", "slope_med_mrad"] + \
           ["%s_%s_%s" % (n, st, u) for n, _, _, u in COMPONENTS for st in ("med", "p95")]
    pp = {k: [] for k in keys}
    for kk in range(N + 1):
        t, p = truth_s[:, kk * stride], states_s[:, kk]
        pp["pos_med_um"].append(float(np.median(pos_err_um(p, t))))
        pp["pos_p95_um"].append(_q(pos_err_um(p, t), 0.95))
        pp["slope_med_mrad"].append(float(np.median(slope_err_mrad(p, t))))
        for n, i, sc, u in COMPONENTS:
            a = np.abs(p[:, i] - t[:, i]) * sc
            pp["%s_med_%s" % (n, u)].append(float(np.median(a)))
            pp["%s_p95_%s" % (n, u)].append(_q(a, 0.95))
    return pp


def chain_scores(states, D, N, fld):
    """{split: the chain's scores} from the predicted states on every plane.

    states  {split: (n, N+1, 5)}   D  the D0 arrays   fld  the field map
    """
    Z1, L = float(D["z1"]), float(D["L"])
    n_max = int(D["n_max"])
    stride = n_max // N
    out = {}
    for s in SCORED:
        n = len(states[s])
        truth = np.asarray(D["%s_truth" % s])[:n]
        S0 = np.asarray(D["%s_S0" % s])[:n]
        pband = np.asarray(D["%s_PBAND" % s])[:n]
        truth_end = truth[:, n_max]
        pred_end = states[s][:, N]
        straight = np.concatenate([S0[:, :2] + S0[:, 2:4] * L, S0[:, 2:4], S0[:, 4:5]], axis=1)
        carried = rk6_rows(pred_end, Z1, np.asarray(D["%s_z_post" % s])[:n],
                           step=RK6_STEP, field=fld)
        real = np.asarray(D["%s_S_post" % s])[:n]
        truth_carried = np.asarray(D["%s_truth_zpost" % s])[:n]

        def block(pred, tr, band=True):
            pb = pband if band else None
            d = stats(pos_err_um(pred, tr), slope_err_mrad(pred, tr), pb)
            d["components"] = component_stats(pred, tr, pb)
            return d

        qop_dev = float(np.abs(states[s][:, :, 4] - S0[:, None, 4]).max())
        out[s] = {
            "vs_rk6_endpoint": block(pred_end, truth_end),
            "vs_real_scifi_state": block(carried, real),
            "rk6_truth_vs_real_scifi_state": block(truth_carried, real),
            "straight_line_vs_rk6_endpoint": block(straight, truth_end, band=False),
            "per_plane": per_plane_curves(states[s], truth, stride, N),
            "qop_passthrough_max_abs_change": qop_dev,
        }
        assert qop_dev == 0.0, "q/p changed along the chain (%g): the passthrough is broken" % qop_dev
    return out
