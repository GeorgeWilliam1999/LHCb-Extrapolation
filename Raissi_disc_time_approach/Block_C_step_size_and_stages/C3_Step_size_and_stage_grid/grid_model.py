#!/usr/bin/env python
"""The straight-line-residual network, made grid-capable for Block C step C3.

This is `../../Block_A_technique_works/A3a_General_leg_network/residual_model.py` with three changes and
nothing else. That file belongs to another experiment and is not edited; this
one is a copy so that the two studies cannot drift into each other.

## What is the same

The parametrisation, exactly:

    output_j = straight_j  +  scale (x) net_j

    straight_j = (x + tx*(z_j - z0),  y + ty*(z_j - z0),  tx,  ty)

for the Gauss-Legendre nodes z_j = z0 + c_j*dz (j = 1..q) and the endpoint
z0 + dz, with the label-free per-sample scale

    scale_slope = kappa * |qop| * I_B                    (dimensionless)
    scale_pos   = kappa * |qop| * I_B * |dz| / 2         (mm)

    I_B = INT_{z0}^{z0+dz} |B(x_s(z), y_s(z), z)| dz     [T mm]

taken along the straight line with a 16-point midpoint rule. Every symbol on
the right is an input: the network is never shown a label.

## What is different

1. **The floors are lowered** from 1e-3 mm / 1e-6 to **1e-9 mm / 1e-12**.
   The old floors were set for legs of 70 mm and up. This dataset's shortest
   stratum is |dz| ~ 0.1 mm, where the true deviation from a straight line is
   of order

       kappa * |qop| * |B| * |dz|^2 / 2  ~  1e-3 * 0.03 * 1 T * 0.01 / 2
                                        ~  1.5e-7 mm

   i.e. four orders of magnitude *below* the old position floor. Left alone,
   the floor - not the physics - would set the scale on the three shortest
   strata, the network would be asked to emit a number of order 1e-4 there,
   and the whole point of the residual redesign would be lost on exactly the
   steps this experiment exists to study. The new floors sit far below the
   smallest real scale in the set (measured in `prepare_grid.py`) and only
   guard against an exactly field-free or exactly straight sample.

2. **q, width and depth come from the command line**, so one module serves the
   whole 4 x 3 x 10 architecture grid; the network emits 4*(q+1) numbers as
   before, and the (z0, dz) extra inputs are unchanged.

3. **The field twin defaults to MagUp**, the polarity of the simulated sample
   (`../C0_Magnet_tracks_dataset/README.md`). Passing a field explicitly still
   overrides it.

The state dict is the base network's, key for key, so the shared trainer's
checkpoint-every-restart and resume behaviour work untouched.
"""
from __future__ import annotations

import numpy as np
import torch

import use_shared                                        # noqa: F401
from _shared.model import OneStepNetwork
from _shared.reference import KAPPA, make_field

torch.set_default_dtype(torch.float64)

# Lowered for the short strata; see the module docstring.
POS_FLOOR_MM = 1e-9        # mm
SLOPE_FLOOR = 1e-12        # dimensionless
N_FIELD_SAMPLES = 16
FIELD_WHICH = "up"         # the polarity of the simulated sample


# --------------------------------------------------------------- the scale --
def field_integral_numpy(S, z0, dz, field, n_samples=N_FIELD_SAMPLES):
    """I_B = INT |B| dz along the straight line, midpoint rule, numpy.

    S  (N, 5) start states in physical units;  z0, dz  (N,)
    Returns (N,) in T mm. The torch twin is `GridResidualNetwork.field_integral`
    and `prepare_grid.py` checks the two against each other.
    """
    S = np.asarray(S, dtype=np.float64)
    z0 = np.asarray(z0, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    u = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples   # midpoints
    s = u[None, :] * dz[:, None]                                     # (N, k)
    x = S[:, 0:1] + S[:, 2:3] * s
    y = S[:, 1:2] + S[:, 3:4] * s
    z = z0[:, None] + s
    Bx, By, Bz = field(x.ravel(), y.ravel(), z.ravel())
    B = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2).reshape(x.shape)
    return B.mean(axis=1) * np.abs(dz)


def scales_from_field_integral(S, dz, IB, pos_floor=POS_FLOOR_MM,
                               slope_floor=SLOPE_FLOOR):
    """(scale_pos (N,), scale_slope (N,)) from the field integral."""
    S = np.asarray(S, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    IB = np.asarray(IB, dtype=np.float64)
    base = KAPPA * np.abs(S[:, 4]) * IB
    slope = np.maximum(base, slope_floor)
    pos = np.maximum(base * np.abs(dz) / 2.0, pos_floor)
    return pos, slope


def straight_line_states(S, dz, c):
    """(N, q+1, 4) straight-line stage states and endpoint, numpy."""
    S = np.asarray(S, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    off = np.concatenate([np.asarray(c)[None, :] * dz[:, None],
                          dz[:, None]], axis=1)                      # (N, q+1)
    out = np.empty((len(S), off.shape[1], 4))
    out[:, :, 0] = S[:, 0:1] + S[:, 2:3] * off
    out[:, :, 1] = S[:, 1:2] + S[:, 3:4] * off
    out[:, :, 2] = S[:, 2:3]
    out[:, :, 3] = S[:, 3:4]
    return out


def straight_line_endpoint(S, dz):
    """(N, 4) the straight-line endpoint alone - the C4 baseline column."""
    S = np.asarray(S, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    return np.stack([S[:, 0] + S[:, 2] * dz, S[:, 1] + S[:, 3] * dz,
                     S[:, 2], S[:, 3]], axis=1)


# --------------------------------------------------------------- the model --
class GridResidualNetwork(OneStepNetwork):
    """The base network, reinterpreted as the deviation from a straight line.

    Constructed exactly like `OneStepNetwork` - so `torch.manual_seed(seed)`
    followed by this constructor draws the same initial parameters the shared
    trainer would have drawn - plus the leg bookkeeping needed to build the
    straight line and the scale from the inputs alone:

        c            (q,)  the Gauss-Legendre nodes of the dataset
        extra_mean, extra_scale  (2,) the dataset's normalisation of (z0, dz)
        field        a FieldV8R1; None loads the MagUp map
    """

    def __init__(self, q, in_scale, out_scale, width=50, depth=4, n_extra=2,
                 c=None, extra_mean=None, extra_scale=None, field=None,
                 n_field_samples=N_FIELD_SAMPLES, pos_floor=POS_FLOOR_MM,
                 slope_floor=SLOPE_FLOOR, node_profile="flat"):
        super().__init__(q, in_scale, out_scale, width=width, depth=depth,
                         n_extra=n_extra)
        if c is None or extra_mean is None or extra_scale is None:
            raise ValueError("the grid network needs the dataset's c, "
                             "extra_mean and extra_scale")
        if node_profile not in ("flat", "poly"):
            raise ValueError("node_profile must be 'flat' or 'poly'")
        if len(np.asarray(c)) != int(q):
            raise ValueError("c has %d nodes but q = %d"
                             % (len(np.asarray(c)), int(q)))
        from _shared.field_torch import FieldTorch
        if field is None:
            field = make_field(FIELD_WHICH)
        # The field twin is held inside a tuple ON PURPOSE. Assigned directly it
        # would become a submodule and its three 81x81x146 grid buffers (23 MB)
        # would be written into <tag>.pt on every L-BFGS restart and demanded
        # back on every resume.
        self._field_holder = (FieldTorch(field),)
        self.n_field_samples = int(n_field_samples)
        self.pos_floor = float(pos_floor)
        self.slope_floor = float(slope_floor)
        self.node_profile = node_profile
        cc = torch.as_tensor(np.asarray(c, dtype=np.float64))
        self.register_buffer("c", cc)
        self.register_buffer("cout", torch.cat([cc, torch.ones(1, dtype=cc.dtype)]))
        self.register_buffer("extra_mean",
                             torch.as_tensor(np.asarray(extra_mean, dtype=np.float64)))
        self.register_buffer("extra_scale",
                             torch.as_tensor(np.asarray(extra_scale, dtype=np.float64)))
        u = (torch.arange(self.n_field_samples, dtype=torch.float64) + 0.5) \
            / self.n_field_samples
        self.register_buffer("u", u)
        self._cache = None            # (S object, extra object, scale tensor)

    @property
    def field(self):
        """The differentiable field twin, kept out of the module tree."""
        return self._field_holder[0]

    # -- the leg, read back out of the normalised extra inputs ---------------
    def z0_dz(self, extra):
        z0 = extra[:, 0] * self.extra_scale[0] + self.extra_mean[0]
        dz = extra[:, 1] * self.extra_scale[1] + self.extra_mean[1]
        return z0, dz

    def field_integral(self, S, z0, dz):
        """I_B along the straight line, (N,) in T mm; no gradient is taken."""
        with torch.no_grad():
            s = self.u[None, :] * dz[:, None]
            x = S[:, 0:1] + S[:, 2:3] * s
            y = S[:, 1:2] + S[:, 3:4] * s
            z = z0[:, None] + s
            Bx, By, Bz = self.field(x, y, z)
            B = torch.sqrt(Bx * Bx + By * By + Bz * Bz)
            return B.mean(dim=1) * dz.abs()

    def residual_scale(self, S, extra):
        """(N, q+1, 4) the per-sample, per-component scale of the deviation.

        Cached on the identity of the two input tensors: during training the
        same S_train / extra_train objects are handed in on every L-BFGS
        evaluation, so the 16 field lookups are paid once, while any new tensor
        recomputes. Identity, not content, so a recycled buffer can never serve
        a stale scale.
        """
        cached = self._cache
        if cached is not None and cached[0] is S and cached[1] is extra:
            return cached[2]
        with torch.no_grad():
            z0, dz = self.z0_dz(extra)
            IB = self.field_integral(S, z0, dz)
            base = KAPPA * S[:, 4].abs() * IB                       # (N,)
            slope = base.clamp_min(self.slope_floor)
            pos = (base * dz.abs() / 2.0).clamp_min(self.pos_floor)
            if self.node_profile == "poly":
                fp = self.cout ** 2                                  # (q+1,)
                fs = self.cout
            else:
                fp = torch.ones_like(self.cout)
                fs = torch.ones_like(self.cout)
            sc = torch.stack([
                pos[:, None] * fp[None, :], pos[:, None] * fp[None, :],
                slope[:, None] * fs[None, :], slope[:, None] * fs[None, :],
            ], dim=-1)                                               # (N,q+1,4)
        self._cache = (S, extra, sc)
        return sc

    def straight(self, S, extra):
        """(N, q+1, 4) the straight-line propagation of the start state."""
        _, dz = self.z0_dz(extra)
        off = self.cout[None, :] * dz[:, None]                       # (N, q+1)
        x = S[:, 0:1] + S[:, 2:3] * off
        y = S[:, 1:2] + S[:, 3:4] * off
        tx = S[:, 2:3].expand_as(off)
        ty = S[:, 3:4].expand_as(off)
        return torch.stack([x, y, tx, ty], dim=-1)

    # -- the forward pass -----------------------------------------------------
    def raw(self, S, extra):
        """(N, q+1, 4) the network's own output, before any scale is applied."""
        return self.net(torch.cat([S / self.in_scale, extra], dim=1)) \
            .reshape(-1, self.q + 1, 4)

    def forward(self, S, extra=None):
        if extra is None:
            raise ValueError("the grid network needs the normalised (z0, dz) "
                             "extra inputs")
        return self.straight(S, extra) \
            + self.residual_scale(S, extra) * self.raw(S, extra)


# ---------------------------------------------------------------- the twin --
def grid_data_loss(model, S, ref, extra=None):
    """The supervised twin, normalised by the residual scale.

    `_shared.model.data_loss` divides by the population-wide `out_scale`; that
    is the quantity this parametrisation replaces, so the deviation is compared
    on its own scale instead - the same O(1) target the physics loss sees.
    """
    out = model(S, extra)
    sc = model.residual_scale(S, extra)
    return (((out - ref[:, :, :4]) / sc) ** 2).mean()


def build_grid_model(data, width, depth, field=None, node_profile="flat"):
    """The constructor the trainer and the analysis scripts share."""
    return GridResidualNetwork(
        int(data["q"]), data["in_scale"], data["out_scale"],
        width=width, depth=depth, n_extra=2,
        c=np.asarray(data["c"]), extra_mean=np.asarray(data["extra_mean"]),
        extra_scale=np.asarray(data["extra_scale"]), field=field,
        node_profile=node_profile)
