"""Scoring helpers, and the leg-by-leg chaining used by the chaining experiment.

The metrics are exactly those of the baseline experiment
(`One_step_network_v2/training.py::score` plus the endpoint slope added by its
continuation pass), lifted out so every experiment reports the same numbers:

  endpoint_med_um / endpoint_p95_um  median and 95th percentile of the endpoint
        position error - the larger of |dx|, |dy| against the fp64 RK4
        reference, in micrometres
  stage_med_um    the same measure taken over all q stage states at once
  slope_med_mrad  median endpoint slope error, the larger of |dtx|, |dty|, in
        milliradians
  rho_mean / rho_median  the agreed scalar relative error at the endpoint
  straight_med_um the straight-line baseline: what you get by ignoring the
        magnet entirely
  n               how many states were scored

`chain` is the new piece: it applies the network leg after leg, feeding each
predicted endpoint back in as the next leg's start state, which is what a real
extrapolator does and what a single-leg score cannot tell you.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from .reference import make_field, rho, rk4_rows
except ImportError:                       # pragma: no cover  (run as a script)
    from reference import make_field, rho, rk4_rows


def split_arrays(data, split):
    """The (S, ref, z0, dz, extra, znodes) of one split, whatever the kind."""
    S = np.asarray(data["%s_S" % split])
    ref = np.asarray(data["%s_ref" % split])
    z0 = np.asarray(data["%s_z0" % split])
    dz = np.asarray(data["%s_dz" % split])
    extra = np.asarray(data["%s_extra" % split]) if "%s_extra" % split in data else None
    znodes = (np.asarray(data["%s_znodes" % split])
              if "%s_znodes" % split in data else np.asarray(data["znodes"]))
    return S, ref, z0, dz, extra, znodes


def predict(model, S, extra=None):
    """(N, q+1, 4) network outputs, in physical units, no grad."""
    St = torch.as_tensor(np.asarray(S, dtype=np.float64))
    with torch.no_grad():
        if model.n_extra == 0:
            out = model(St)
        else:
            out = model(St, torch.as_tensor(np.asarray(extra, dtype=np.float64)))
    return out.numpy()


def score_against_reference(out, S, ref, dz):
    """The metric dict for one set of predictions against the RK4 reference.

    out  (N, q+1, 4) network outputs      ref (N, q+1, 5) reference states
    S    (N, 5) start states              dz  (N,) or scalar step lengths
    """
    out = np.asarray(out)
    ref = np.asarray(ref)
    S = np.asarray(S)
    dz = np.broadcast_to(np.asarray(dz, dtype=np.float64).reshape(-1), (len(S),))
    end_err = np.abs(out[:, -1, :2] - ref[:, -1, :2]).max(axis=1) * 1e3      # um
    stage_err = np.abs(out[:, :-1, :2] - ref[:, :-1, :2]).max(axis=(1, 2)) * 1e3
    slope_err = np.abs(out[:, -1, 2:4] - ref[:, -1, 2:4]).max(axis=1) * 1e3  # mrad
    r = rho(ref[:, -1, :4], out[:, -1, :])
    straight = S[:, :2] + S[:, 2:4] * dz[:, None]
    line_err = np.abs(straight - ref[:, -1, :2]).max(axis=1) * 1e3
    return {
        "endpoint_med_um": float(np.median(end_err)),
        "endpoint_p95_um": float(np.quantile(end_err, 0.95)),
        "stage_med_um": float(np.median(stage_err)),
        "slope_med_mrad": float(np.median(slope_err)),
        "rho_mean": float(r.mean()),
        "rho_median": float(np.median(r)),
        "straight_med_um": float(np.median(line_err)),
        "n": int(len(S)),
    }


def score_split(model, data, split="test"):
    """Score a trained model on one split of a prepared dataset."""
    S, ref, z0, dz, extra, _ = split_arrays(data, split)
    out = predict(model, S, extra)
    return score_against_reference(out, S, ref, dz), out


def chain(model, S0, legs, extra_mean=None, extra_scale=None):
    """Apply the network leg after leg, feeding its own endpoint forward.

    model       a trained OneStepNetwork. If it was trained on general legs
                (n_extra = 2) the per-leg (z0, dz) are fed in as extra inputs,
                normalised with extra_mean / extra_scale (the values stored in
                the dataset npz); a frozen-leg model ignores them and must only
                be chained over copies of its own leg.
    S0          (N, 5) start states, physical units
    legs        sequence of (z0, z1) plane pairs in mm; each entry may be a
                scalar pair (the same leg for every sample) or a pair of (N,)
                arrays (a per-sample leg)
    Returns     (N, len(legs), 5) the state after each leg; qop is carried
                through unchanged, as the scheme demands.
    """
    S = np.asarray(S0, dtype=np.float64).copy()
    n = len(S)
    out_states = np.empty((n, len(legs), 5))
    for i, (z0, z1) in enumerate(legs):
        z0 = np.broadcast_to(np.asarray(z0, dtype=np.float64).reshape(-1), (n,))
        z1 = np.broadcast_to(np.asarray(z1, dtype=np.float64).reshape(-1), (n,))
        dz = z1 - z0
        extra = None
        if model.n_extra:
            if extra_mean is None or extra_scale is None:
                raise ValueError("chaining a general-leg model needs "
                                 "extra_mean and extra_scale from its dataset")
            extra = np.stack([(z0 - extra_mean[0]) / extra_scale[0],
                              (dz - extra_mean[1]) / extra_scale[1]], axis=1)
        pred = predict(model, S, extra)
        S = np.concatenate([pred[:, -1, :], S[:, 4:5]], axis=1)   # qop passthrough
        out_states[:, i] = S
    return out_states


def chain_reference(S0, legs, field="down", step=5.0):
    """The fp64 RK4 truth for the same chain, for scoring `chain` against."""
    fld = make_field(field) if isinstance(field, str) else field
    S = np.asarray(S0, dtype=np.float64).copy()
    n = len(S)
    out_states = np.empty((n, len(legs), 5))
    for i, (z0, z1) in enumerate(legs):
        z0 = np.broadcast_to(np.asarray(z0, dtype=np.float64).reshape(-1), (n,))
        z1 = np.broadcast_to(np.asarray(z1, dtype=np.float64).reshape(-1), (n,))
        S = rk4_rows(S, z0, z1, step=step, field=fld)
        out_states[:, i] = S
    return out_states


def chain_errors(pred_states, ref_states):
    """Per-leg endpoint error in um (the larger of |dx|, |dy|), median and p95."""
    err = np.abs(pred_states[:, :, :2] - ref_states[:, :, :2]).max(axis=2) * 1e3
    return {
        "per_leg_med_um": np.median(err, axis=0).tolist(),
        "per_leg_p95_um": np.quantile(err, 0.95, axis=0).tolist(),
        "final_med_um": float(np.median(err[:, -1])),
        "final_p95_um": float(np.quantile(err[:, -1], 0.95)),
        "n": int(len(err)),
    }
