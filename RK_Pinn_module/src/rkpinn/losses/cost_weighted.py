"""The cost-weighted loss: equation 20 of the second mini-paper.

Every residual is measured as the displacement it would cause at the end of
the track, as a fraction of the track's own total bend, with a band of
momentum weighted up:

    loss = mean over n, j, d of  ( a[n] * g[n, j, d] * r[n, j, d] / D_ref ) ** 2

    g[n, j, d] = (1, 1, l[n, j], l[n, j])   for d = x, y, slope in x, slope in y

    l[n, j] = last plane - (z_n + c_j dz) + dz
        the lever arm: the distance the output still has to travel, plus one
        step, so that the last plane is not weighted to zero. A slope error is
        multiplied by the distance it still has to travel; a position error
        is not.

    a[n] = clamp( sqrt(W(p_n)) * D_ref / D_n,  [m / clamp, m * clamp] )
        m is the median of the unclamped factor over the batch.

    D_n = kappa * |q/p|_n * I * L
        the total bend of track n over the whole crossing, with I the mean of
        |B| on the axis over the crossing and L its length. It is a constant
        of the track, not of the step.

    W(p) = 1 inside the momentum window, a Gaussian in log p outside it with
        the width `roll_off`, and never below `floor`.

    D_ref  the reference bend: the median of D_n over the first round's
        states. It is fixed for the run, so the loss does not drift.

Both weights depend on the input state and the field map alone, so the loss
needs no label.

Every constant is an argument, and the loss reads every constant from the
constants of the run. None is a default in the code: in the frozen code the
momentum window was both a module default and a constant of the run, and the
training path read the default.

Each of the three factors can be switched off. A factor that is off is 1.
With all three off and a reference bend of 1 the loss is the unweighted loss,
to the last bit. (The ablations of the frozen study replaced a factor by its
average instead. They are not ported.)

Ported from `F0_Weighting/weighted_loss.py` and
`G0_Weighting/windowed_loss.py`. The weights are identical to the frozen ones
to the last bit, given the same states and constants.
Gates: tests/test_losses.py.
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.equation_of_motion.lhcb import CHARGE_OVER_MOMENTUM_PER_INVERSE_GEV, KAPPA
from rkpinn.losses.stage_residual import check_terms, stage_residual
from rkpinn.registry import register

SMALLEST_CHARGE_OVER_MOMENTUM = 1e-12


def momentum_gev(charge_over_momentum):
    """|p| in GeV from the fifth component of the state."""
    qop = charge_over_momentum
    if torch.is_tensor(qop):
        return CHARGE_OVER_MOMENTUM_PER_INVERSE_GEV / qop.abs().clamp_min(
            SMALLEST_CHARGE_OVER_MOMENTUM)
    return CHARGE_OVER_MOMENTUM_PER_INVERSE_GEV / np.maximum(
        np.abs(qop), SMALLEST_CHARGE_OVER_MOMENTUM)


def momentum_window(p, lowest_gev, highest_gev, roll_off, floor):
    """1 inside the window; a Gaussian in log p outside; never below the floor."""
    p_lo, p_hi = lowest_gev, highest_gev
    if torch.is_tensor(p):
        w = torch.ones_like(p)
        w = torch.where(p < p_lo, torch.exp(-(torch.log(p / p_lo) / roll_off) ** 2), w)
        w = torch.where(p > p_hi, torch.exp(-(torch.log(p / p_hi) / roll_off) ** 2), w)
        return w.clamp_min(floor)
    w = np.ones_like(p)
    w = np.where(p < p_lo, np.exp(-(np.log(p / p_lo) / roll_off) ** 2), w)
    w = np.where(p > p_hi, np.exp(-(np.log(p / p_hi) / roll_off) ** 2), w)
    return np.maximum(w, floor)


def mean_field_integral(field_map, first_plane_mm, last_plane_mm, number_of_samples):
    """The integral of |B| along the z axis over the crossing, in tesla mm."""
    n = int(number_of_samples)
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    z = first_plane_mm + u * (last_plane_mm - first_plane_mm)
    Bx, By, Bz = field_map(np.zeros_like(z), np.zeros_like(z), z)
    B = np.sqrt(np.asarray(Bx) ** 2 + np.asarray(By) ** 2 + np.asarray(Bz) ** 2)
    return float(B.mean() * (last_plane_mm - first_plane_mm))


def track_bend_mm(charge_over_momentum, field_integral, length_mm):
    """D_n: the total bend of a track over the crossing."""
    lib = torch if torch.is_tensor(charge_over_momentum) else np
    return KAPPA * lib.abs(charge_over_momentum) * field_integral * length_mm


def lever_arms_mm(nodes_of_terms, step_length_mm, start_planes_mm, last_plane_mm):
    """l: the distance each output still has to travel, plus one step; shape (n, J)."""
    dz = step_length_mm
    if torch.is_tensor(start_planes_mm):
        z_out = start_planes_mm[:, None] + nodes_of_terms[None, :] * dz
        return (last_plane_mm - z_out).clamp_min(0.0) + dz
    z_out = np.asarray(start_planes_mm)[:, None] + np.asarray(nodes_of_terms)[None, :] * dz
    return np.maximum(last_plane_mm - z_out, 0.0) + dz


def factor_of_the_track(charge_over_momentum, constants):
    """a[n], the part of the weight that depends on the track alone."""
    qop = charge_over_momentum
    is_t = torch.is_tensor(qop)
    lib = torch if is_t else np
    p = momentum_gev(qop)
    if constants["momentum_window_is_on"]:
        low, high = constants["momentum_window_gev"]
        w = momentum_window(p, low, high, constants["roll_off"], constants["floor"])
    else:
        w = torch.ones_like(p) if is_t else np.ones_like(p)
    a = lib.sqrt(w)
    if constants["track_bend_is_on"]:
        a = a * constants["reference_bend_mm"] / track_bend_mm(
            qop, constants["field_integral_tesla_mm"], constants["length_mm"])
    c = constants["clamp"]
    if c and c > 1.0:
        # torch.median returns the lower of the two middle values and numpy
        # averages them; the quantile at one half is what both agree on
        med = torch.quantile(a, 0.5) if is_t else np.median(a)
        a = a.clamp(med / c, med * c) if is_t else np.clip(a, med / c, med * c)
    return a


def weights(input_states, start_planes_mm, nodes_of_terms, step_length_mm, constants):
    """The weight that multiplies the residual, shape (n, J, 4)."""
    is_t = torch.is_tensor(input_states)
    lev = lever_arms_mm(nodes_of_terms, step_length_mm, start_planes_mm,
                        constants["last_plane_mm"])
    if not constants["lever_arm_is_on"]:
        lev = lev * 0 + 1.0
    a = factor_of_the_track(input_states[:, 4], constants)
    ones = lev * 0 + 1.0
    g = (torch.stack if is_t else np.stack)([ones, ones, lev, lev], axis=-1)
    return (a[:, None, None] * g) / constants["reference_bend_mm"]


@register("loss", "cost_weighted")
class CostWeighted:
    name = "cost_weighted"
    needs_labels = False

    def __init__(self, *, equation_of_motion, terms, first_plane_mm, last_plane_mm,
                 momentum_window_gev, roll_off, floor, clamp,
                 samples_of_the_field_integral, lever_arm_is_on, track_bend_is_on,
                 momentum_window_is_on, reference_bend_mm):
        """
        reference_bend_mm  "median_of_first_round_states", or a number in mm
        """
        self.equation_of_motion = equation_of_motion
        self.settings = {
            "terms": check_terms(terms),
            "first_plane_mm": float(first_plane_mm),
            "last_plane_mm": float(last_plane_mm),
            "momentum_window_gev": [float(momentum_window_gev[0]),
                                    float(momentum_window_gev[1])],
            "roll_off": float(roll_off), "floor": float(floor), "clamp": float(clamp),
            "samples_of_the_field_integral": int(samples_of_the_field_integral),
            "lever_arm_is_on": bool(lever_arm_is_on),
            "track_bend_is_on": bool(track_bend_is_on),
            "momentum_window_is_on": bool(momentum_window_is_on),
        }
        if reference_bend_mm != "median_of_first_round_states":
            reference_bend_mm = float(reference_bend_mm)
        self.reference_bend_mm = reference_bend_mm

    def constants(self, first_round_states) -> dict:
        """The constants of a run: the settings, the field integral, and the
        reference bend measured on the first round's states."""
        settings = self.settings
        length = settings["last_plane_mm"] - settings["first_plane_mm"]
        integral = mean_field_integral(
            self.equation_of_motion.field.field_map, settings["first_plane_mm"],
            settings["last_plane_mm"], settings["samples_of_the_field_integral"])
        if self.reference_bend_mm == "median_of_first_round_states":
            qop = np.asarray(first_round_states)[:, 4]
            reference = float(np.median(track_bend_mm(qop, integral, length)))
        else:
            reference = self.reference_bend_mm
        return dict(settings, name=self.name, length_mm=float(length),
                    field_integral_tesla_mm=integral, reference_bend_mm=reference)

    def value(self, predicted_track, target, constants):
        residual = stage_residual(predicted_track, self.equation_of_motion,
                                  constants["terms"])
        w = weights(residual.input_states, residual.start_planes_mm,
                    residual.nodes_of_terms, residual.step_length_mm, constants)
        return ((residual.residual * w) ** 2).mean()
