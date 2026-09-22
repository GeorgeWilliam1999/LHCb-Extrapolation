#!/usr/bin/env python
"""Block G - Block F's reweighted loss with the momentum window actually obeyed.

Block G changes ONE thing about Block F: where the momentum window W(p) sits.
Block F put it on 10-50 GeV; the region that matters for the supervisor's
question is around 5 GeV. Everything else - the tracks, the network, the seed,
the optimiser, the rounds, the clamp, the lever weighting, the track-bend
normalisation, the roll-off width and the floor - is imported unchanged from
`Block_F_reweighted_loss/F0_Weighting/weighted_loss.py`, which is read-only.

## The trap this module exists to fix

`weighted_loss.py` carries the window in two places at once:

  * module constants `P_LO = 10.0`, `P_HI = 50.0`, used as the DEFAULT arguments
    of `band_window(p, p_lo=P_LO, p_hi=P_HI, rolloff=ROLLOFF, floor=W_FLOOR)`;
  * the run's own constants dict, `const["p_lo"]`, `const["p_hi"]`,
    `const["rolloff"]`, `const["w_floor"]`, which is what gets written into
    `scale.json["weighting"]` and so is what a run claims it trained with.

`per_track_factor` calls `band_window(p)` with NO arguments. The torch training
path therefore always used the module constants and never looked at the dict.
For Block F the two agreed (the dict was filled from the same constants), so
none of Block F's runs is affected. The moment the window moves - which is the
whole of Block G - they stop agreeing: the run folder would say "3-8 GeV" while
the optimiser kept minimising the 10-50 GeV objective, and nothing in the
training output would show it. The numpy twin in `F0_Weighting/check_weights.py`
reads the dict, so the twin gate is what catches it; gate W1 in
`check_windowed_weights.py` demonstrates both halves (F0's torch path disagrees
with the twin under a moved window, this module's agrees).

The fix is not applied to Block F's file. Block F is read-only and its runs are
still training. Instead this module re-implements exactly the three functions
that touch the window:

    reference_constants   takes p_lo / p_hi / rolloff / w_floor as ARGUMENTS and
                          writes them into the dict (F0 hard-codes the module
                          constants there);
    per_track_factor      passes const["p_lo"], const["p_hi"], const["rolloff"],
                          const["w_floor"] into band_window;
    weights               F0's function, re-implemented here only so that it
                          calls THIS per_track_factor.

Everything else (MODES, QOP_TO_GEV, P_LO, P_HI, ROLLOFF, W_FLOOR, CLAMP,
momentum_gev, band_window, i_bar, track_bend, lever_arms, weighted_loss,
loss_for) is re-exported from F0 untouched, so there is one implementation of
the physics and Block G cannot drift from Block F by accident.

With p_lo = 10 and p_hi = 50 every function here returns exactly what F0's
returns, bit for bit. That is the premise of the training-side isolation gate:
the default window reproduces Block F's round-1 losses to the last digit.

One property of the weight worth keeping in mind when a window is chosen: the
per-track factor is a_n = sqrt(W(p_n)) * D_ref / D_n with D_n proportional to
|q/p|, so a_n grows like p while sqrt(W) falls outside the band. The two pull
against each other, and whether a given window actually concentrates the loss
inside itself is a measurement (the pre-flight, gate W4), not an assumption.
"""
from __future__ import annotations

import os
import sys

import numpy as np                                      # noqa: F401
import torch

import use_shared                                       # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
F0_WEIGHTING = os.path.normpath(os.path.join(
    _HERE, "..", "..", "Block_F_reweighted_loss", "F0_Weighting"))
if F0_WEIGHTING not in sys.path:
    sys.path.insert(0, F0_WEIGHTING)

# Block F's module, imported and never modified.
from weighted_loss import (                             # noqa: E402,F401
    MODES, QOP_TO_GEV, P_LO, P_HI, ROLLOFF, W_FLOOR, CLAMP, N_IBAR_SAMPLES,
    momentum_gev, band_window, i_bar, track_bend, lever_arms,
    weighted_loss, loss_for)
from weighted_loss import reference_constants as _f0_reference_constants   # noqa: E402

torch.set_default_dtype(torch.float64)


def reference_constants(model, S, z_start, z1, ibar, L, mode="full", clamp=CLAMP,
                        p_lo=P_LO, p_hi=P_HI, rolloff=ROLLOFF, w_floor=W_FLOOR):
    """F0's constants dict, with the window taken from the ARGUMENTS.

    Identical to F0's for p_lo = P_LO, p_hi = P_HI, rolloff = ROLLOFF,
    w_floor = W_FLOOR (F0 writes the module constants into those four keys).
    """
    const = _f0_reference_constants(model, S, z_start, z1, ibar, L, mode=mode, clamp=clamp)
    const["p_lo"] = float(p_lo)
    const["p_hi"] = float(p_hi)
    const["rolloff"] = float(rolloff)
    const["w_floor"] = float(w_floor)
    return const


def per_track_factor(qop, const, switches, clamp=None):
    """a_n, the part of the weight that depends only on the track.

    F0's function with one change: `band_window` is given the run's own
    window from `const` instead of falling back to the module defaults.
    """
    is_t = torch.is_tensor(qop)
    lib = torch if is_t else np
    p = momentum_gev(qop)
    if switches["window"]:
        w = band_window(p, const["p_lo"], const["p_hi"], const["rolloff"], const["w_floor"])
    else:
        w = torch.ones_like(p) if is_t else np.ones_like(p)
    a = lib.sqrt(w)
    if switches["track"]:
        a = a * const["D_ref"] / track_bend(qop, const["i_bar"], const["L"])
    c = const["clamp"] if clamp is None else clamp
    if c and c > 1.0:
        # torch.median returns the LOWER of the two middle values and numpy
        # averages them; quantile(0.5) is the one both libraries agree on
        med = (torch.quantile(a, 0.5) if is_t else np.median(a))
        a = a.clamp(med / c, med * c) if is_t else np.clip(a, med / c, med * c)
    return a


def weights(model, S, z_start, const, mode=None):
    """(n, q+1, 4) the weight that multiplies the physical residual.

    F0's function, re-implemented only so that it reaches the
    `per_track_factor` above; the lever arms, the modes and D_ref are F0's.
    """
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
