#!/usr/bin/env python
"""Block F - the reweighted RK-PINN loss. The ONLY thing Block F changes.

Block E's loss is the mean squared reconstruction residual, each component
divided by the POOLED spread of that component over the training states:

    loss_E = mean over states n, outputs j, components d of
             ( (rec - S)_{n,j,d} / in_scale_d ) ** 2

That asks for equal ABSOLUTE accuracy in every component, on every track, at
every plane, and it gets it: the measured per-step slope error is flat along
the crossing (E3_Analysis/results/error_anatomy.csv). Two consequences, both
measured, are what Block F is for.

  1. 97 percent of the residual comes from 2-5 GeV tracks, because a soft track
     bends more and so has a bigger residual in absolute terms. The momenta
     that matter for physics, 10-50 GeV, are 33 percent of the sample and a few
     percent of the loss.
  2. A slope residual and a position residual count the same, although a slope
     error made at the first plane is multiplied by 5,097 mm before it reaches
     the SciFi plane and a position error is not. 72 percent of the endpoint
     error is that lever-arm term, and a step's cost at z1 correlates +0.996
     with the distance it still has to travel (and only +0.12 with |B|, which
     is why this module does NOT weight by the field; Spearman over the 64
     steps of the N = 64, q = 2 anatomy, `field_correlations.py`).

So Block F measures every residual as the displacement it would cause at the
SciFi plane, as a fraction of the track's own total bend, with the 10-50 GeV
band weighted up:

    loss_F = mean over n, j, d of ( a_n * e_{n,j,d} / D_ref ) ** 2

    e_x  = r_x                    e_y  = r_y                     [mm]
    e_tx = r_tx * lever_{n,j}     e_ty = r_ty * lever_{n,j}      [mm]

    lever_{n,j} = z1 - (z_start_n + c_j dz) + dz
        the distance left to travel after that output's plane, PLUS one step,
        so the last plane is not weighted to zero. It is additive, not a
        floor: every plane carries the extra dz (1.6 percent of L at N = 64;
        half of L at N = 2, which Block F does not run);

    a_n = sqrt(W(p_n)) * D_ref / D_n,  clamped to [1/CLAMP, CLAMP] of its median
        (the median of the round's own batch, so the clamp thresholds move
        slightly between rounds; D_ref below does not)
    D_n = KAPPA |q/p|_n * I_bar * L
        the track's total transverse bend across the whole crossing. It is a
        constant of the TRACK, not of the step: dividing by the per-step field
        integral instead would ask for constant RELATIVE accuracy along z and
        so make the high-field middle worse in absolute terms, which is the
        wrong direction (George 2026-09-18).
    W(p) = 1 on [P_LO, P_HI], Gaussian roll-off in log p outside, floored
        a smooth window, not a cut: one network still has to work everywhere.

D_ref is a fixed constant of the run (the median of D_n over the round-1
states), so the objective does not drift between rounds; it only sets the
overall size of the loss, which the trainer rescales anyway.

Both weights are functions of the input state and the field map alone, so the
loss stays label-free.

Ablations, for the diagnosis if the combined weighting is ambiguous. Each
switches ONE factor off by replacing the quantity that varies with its own
average, so the size of the loss is unchanged and only its distribution moves:

    full        everything on
    no_lever    lever_{n,j} -> its mean over the planes
    no_track    D_n -> D_ref (so a_n carries the window only)
    no_window   W -> 1
    blockE      Block E's weights, 1 / in_scale: the isolation gate

x and y are weighted equally, in mm, because the endpoint measure is radial.
Block E's separate y scale stays where it belongs, in the network's output
parameterisation, and no longer doubles as a loss weight.
"""
from __future__ import annotations

import numpy as np
import torch

import use_shared                                       # noqa: F401
from _shared.model import physics_loss, reconstruction_residuals
from _shared.reference import KAPPA

torch.set_default_dtype(torch.float64)

QOP_TO_GEV = 0.299792458      # qop = QOP_TO_GEV * q / p[GeV], the Allen convention
P_LO, P_HI = 10.0, 50.0       # GeV: the band that matters (George 2026-09-18)
ROLLOFF = float(np.log(2.0))  # the window falls by 1/e a factor of two outside the band
W_FLOOR = 0.05                # no track is ever ignored completely
CLAMP = 5.0                   # per-track weight limited to [1/CLAMP, CLAMP] of its median
N_IBAR_SAMPLES = 1024

MODES = {
    "full":      dict(lever=True,  track=True,  window=True),
    "no_lever":  dict(lever=False, track=True,  window=True),
    "no_track":  dict(lever=True,  track=False, window=True),
    "no_window": dict(lever=True,  track=True,  window=False),
    "blockE":    dict(lever=False, track=False, window=False),
}


# ------------------------------------------------------------- the pieces --
def momentum_gev(qop):
    """|p| in GeV from the fifth state component."""
    if torch.is_tensor(qop):
        return QOP_TO_GEV / qop.abs().clamp_min(1e-12)
    return QOP_TO_GEV / np.maximum(np.abs(qop), 1e-12)


def band_window(p, p_lo=P_LO, p_hi=P_HI, rolloff=ROLLOFF, floor=W_FLOOR):
    """A smooth top hat on [p_lo, p_hi]: 1 inside, Gaussian in log p outside."""
    if torch.is_tensor(p):
        w = torch.ones_like(p)
        w = torch.where(p < p_lo, torch.exp(-(torch.log(p / p_lo) / rolloff) ** 2), w)
        w = torch.where(p > p_hi, torch.exp(-(torch.log(p / p_hi) / rolloff) ** 2), w)
        return w.clamp_min(floor)
    w = np.ones_like(p)
    w = np.where(p < p_lo, np.exp(-(np.log(p / p_lo) / rolloff) ** 2), w)
    w = np.where(p > p_hi, np.exp(-(np.log(p / p_hi) / rolloff) ** 2), w)
    return np.maximum(w, floor)


def i_bar(field, z0, z1, n=N_IBAR_SAMPLES):
    """INT |B| dz over the whole crossing on the z axis [T mm] - a pure constant."""
    u = (np.arange(int(n), dtype=np.float64) + 0.5) / int(n)
    z = z0 + u * (z1 - z0)
    Bx, By, Bz = field(np.zeros_like(z), np.zeros_like(z), z)
    B = np.sqrt(np.asarray(Bx) ** 2 + np.asarray(By) ** 2 + np.asarray(Bz) ** 2)
    return float(B.mean() * (z1 - z0))


def track_bend(qop, ibar, L):
    """D_n: the track's total transverse bend across the crossing [mm]."""
    lib = torch if torch.is_tensor(qop) else np
    return KAPPA * lib.abs(qop) * ibar * L


def lever_arms(cout, dz, z_start, z1):
    """(n, q+1) the distance each output still has to travel, plus dz [mm] (additive, not a floor)."""
    if torch.is_tensor(z_start):
        z_out = z_start[:, None] + cout[None, :] * dz
        return (z1 - z_out).clamp_min(0.0) + dz
    z_out = np.asarray(z_start)[:, None] + np.asarray(cout)[None, :] * dz
    return np.maximum(z1 - z_out, 0.0) + dz


def reference_constants(model, S, z_start, z1, ibar, L, mode="full", clamp=CLAMP):
    """The two fixed constants of a run, from its round-1 states.

    D_ref   the median track bend, the unit the loss is measured in
    lev_ref the mean lever arm, used when the lever is switched off
    """
    qop = np.asarray(S)[:, 4]
    lev = lever_arms(np.asarray(model.cout.numpy()), float(model.dz), z_start, z1)
    return dict(D_ref=float(np.median(track_bend(qop, ibar, L))),
                lev_ref=float(lev.mean()), i_bar=float(ibar), z1=float(z1), L=float(L),
                mode=str(mode), clamp=float(clamp), p_lo=P_LO, p_hi=P_HI,
                rolloff=ROLLOFF, w_floor=W_FLOOR)


def per_track_factor(qop, const, switches, clamp=None):
    """a_n, the part of the weight that depends only on the track."""
    is_t = torch.is_tensor(qop)
    lib = torch if is_t else np
    p = momentum_gev(qop)
    w = band_window(p) if switches["window"] else (torch.ones_like(p) if is_t else np.ones_like(p))
    a = lib.sqrt(w)
    if switches["track"]:
        a = a * const["D_ref"] / track_bend(qop, const["i_bar"], const["L"])
    c = const["clamp"] if clamp is None else clamp
    if c and c > 1.0:
        # torch.median returns the LOWER of the two middle values and numpy
        # averages them; quantile(0.5) is the one both libraries agree on, and
        # the twin gate is what caught the difference
        med = (torch.quantile(a, 0.5) if is_t else np.median(a))
        a = a.clamp(med / c, med * c) if is_t else np.clip(a, med / c, med * c)
    return a


def weights(model, S, z_start, const, mode=None):
    """(n, q+1, 4) the weight that multiplies the physical residual."""
    mode = const["mode"] if mode is None else mode
    sw = MODES[mode]
    is_t = torch.is_tensor(S)
    if mode == "blockE":                       # Block E's weights, for the gate
        inv = 1.0 / (model.in_scale[:4] if is_t else model.in_scale[:4].numpy())
        shape = (len(S), model.q + 1, 4)
        return inv[None, None, :].expand(shape) if is_t else np.broadcast_to(inv, shape)
    cout = model.cout if is_t else model.cout.numpy()
    lev = lever_arms(cout, float(model.dz), z_start, const["z1"])
    if not sw["lever"]:
        lev = lev * 0 + const["lev_ref"]
    a = per_track_factor(S[:, 4], const, sw)
    ones = lev * 0 + 1.0
    g = (torch.stack if is_t else np.stack)([ones, ones, lev, lev], axis=-1)
    return (a[:, None, None] * g) / const["D_ref"]


# --------------------------------------------------------------- the loss --
def weighted_loss(model, rates, S, dz, znodes, A, b, extra, w):
    """The Block F loss: the physical residual, weighted, squared, averaged."""
    r = reconstruction_residuals(model, rates, S, dz, znodes, A, b, extra) * model.in_scale[:4]
    return ((r * w) ** 2).mean()


def loss_for(mode, model, rates, S, dz, znodes, A, b, extra, w):
    """Dispatch: `blockE` goes through the shared loss itself, bit for bit."""
    if mode == "blockE":
        return physics_loss(model, rates, S, dz, znodes, A, b, extra)
    return weighted_loss(model, rates, S, dz, znodes, A, b, extra, w)
