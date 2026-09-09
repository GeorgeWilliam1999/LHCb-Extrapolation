#!/usr/bin/env python
"""The straight-line-residual network: the deviation, not the state.

Wave 1 asked one network to emit the ABSOLUTE stage states and endpoint of any
leg, with a single population-wide output scale (`out_scale`, the standard
deviation of the straight-line-propagated training states: 602 mm in x, 341 mm
in y). On the 5.2 m cross-magnet leg the answer is hundreds of millimetres away
from the start state and that scale is right; on the 70 mm plane-to-plane leg
the whole correction to a straight line is 6.6 um, i.e. one part in 10^8 of the
scale the network's last layer works in. Wave 1 duly gave 300 um on that leg -
fifty times worse than doing nothing at all.

This file makes the network predict the DEVIATION from a straight line and
gives every sample its own scale, so that the quantity the last layer has to
represent is O(1) on all three leg types at once.

    output_j = straight_j  +  scale (x) net_j

with, for a start state S = (x, y, tx, ty, qop) at z0, a step dz, and the
Gauss-Legendre nodes z_j = z0 + c_j*dz (j = 1..q) plus the endpoint z0 + dz,

    straight_j = (x + tx*(z_j - z0),  y + ty*(z_j - z0),  tx,  ty)

i.e. the magnet switched off, and `net_j` the raw network output for node j
(the same 4*(q+1) numbers wave 1 produced, but now O(1) rather than carried in
millimetres).

## The scale, and why it is label-free

The whole point of the exercise is that the network never sees a label, so the
scale may only use the inputs. Along the straight line the equation of motion
gives, to first order in the field,

    |d tx| ~ kappa * |qop| * INT |B| dz          (dimensionless)
    |d x | ~ kappa * |qop| * INT |B| dz * |dz|/2 (mm, the linear ramp of that
                                                  slope over the step)

with kappa = 1e-3 and qop = 0.299792458 q/p[GeV] the Allen convention of
`_shared/reference.py`, and

    I_B = INT_{z0}^{z0+dz} |B(x_s(z), y_s(z), z)| dz    [T mm]

the field integral taken ALONG THE STRAIGHT LINE x_s(z) = x + tx*(z - z0),
y_s(z) = y + ty*(z - z0), evaluated with a 16-point midpoint rule. Nothing in
that expression needs the answer: it is a property of the input state and the
leg. A floor (1e-3 mm on positions, 1e-6 on slopes) keeps the scale away from
zero for a straight track in a field-free region.

The same scale is used for all q+1 outputs of a sample (a "flat" node profile).
The alternative - multiplying node j by c_j^2 for positions and c_j for slopes,
so that the early nodes are scaled by how far along the step they sit - is
implemented as `node_profile="poly"`; `prepare_residual.py` measures both and
records which one was chosen.

## Why this class can be dropped into every existing helper

`_shared/evaluate.py` (`predict`, `score_split`, `chain`) and
`_shared/model.py` (`reconstruction_residuals`, `physics_loss`) only ever call
`model(S, extra)`. The residual network therefore reconstructs (z0, dz) from
the normalised `extra` it is handed, using the dataset's own `extra_mean` /
`extra_scale`, and computes I_B itself with the differentiable field twin. So
the physics loss is UNCHANGED - it is built from the reconstructed absolute
states and only needs this forward - and the chaining experiment, including the
leg-D geometry that appears in no dataset, works with no further plumbing.

The data twin does need one change: `_shared.model.data_loss` normalises by the
population-wide `out_scale`, which is exactly the thing being replaced, so
`residual_data_loss` below divides by the per-sample residual scale instead.

Both losses see O(1) numbers; the state dict is the base network's, key for
key, so a run resumes from its checkpoint exactly as the shared trainer expects.
"""
from __future__ import annotations

import numpy as np
import torch

import use_shared                                        # noqa: F401
from _shared.model import OneStepNetwork
from _shared.reference import KAPPA

torch.set_default_dtype(torch.float64)

POS_FLOOR_MM = 1e-3        # mm
SLOPE_FLOOR = 1e-6         # dimensionless
N_FIELD_SAMPLES = 16


# --------------------------------------------------------------- the scale --
def field_integral_numpy(S, z0, dz, field, n_samples=N_FIELD_SAMPLES):
    """I_B = INT |B| dz along the straight line, midpoint rule, numpy.

    S  (N, 5) start states in physical units;  z0, dz  (N,)
    Returns (N,) in T mm.  Used by prepare_residual.py and the checks; the
    model's own copy below is the torch twin of this function.
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
    """(scale_pos (N,), scale_slope (N,)) from the field integral.

    scale_pos   = kappa |qop| I_B |dz| / 2   [mm]
    scale_slope = kappa |qop| I_B            [-]
    """
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


# --------------------------------------------------------------- the model --
class ResidualOneStepNetwork(OneStepNetwork):
    """The base network, reinterpreted as the deviation from a straight line.

    Constructed exactly like `OneStepNetwork` (so `torch.manual_seed(seed)`
    followed by this constructor draws the same initial parameters as wave 1
    did), plus the leg bookkeeping it needs to build the straight line and the
    scale from the inputs alone:

        c            (q,)  the Gauss-Legendre nodes of the dataset
        extra_mean, extra_scale  (2,) the dataset's normalisation of (z0, dz),
                     so the physical leg can be read back out of `extra`
        field        a FieldV8R1 (or None for MagDown)

    `out_scale` is kept as a buffer for bookkeeping and for anything that reads
    it, but the forward pass does not use it: the residual scale replaces it.
    """

    def __init__(self, q, in_scale, out_scale, width=50, depth=4, n_extra=2,
                 c=None, extra_mean=None, extra_scale=None, field=None,
                 n_field_samples=N_FIELD_SAMPLES, pos_floor=POS_FLOOR_MM,
                 slope_floor=SLOPE_FLOOR, node_profile="flat"):
        super().__init__(q, in_scale, out_scale, width=width, depth=depth,
                         n_extra=n_extra)
        if c is None or extra_mean is None or extra_scale is None:
            raise ValueError("the residual network needs the dataset's c, "
                             "extra_mean and extra_scale")
        if node_profile not in ("flat", "poly"):
            raise ValueError("node_profile must be 'flat' or 'poly'")
        from _shared.field_torch import FieldTorch
        # The field twin is held inside a tuple ON PURPOSE. Assigned directly it
        # would become a submodule, and its three 81x81x146 grid buffers (23 MB)
        # would be written into `<tag>.pt` on every single L-BFGS restart and
        # demanded back on every resume. The tuple keeps it out of `state_dict`,
        # so the checkpoint stays the 34k network parameters it should be and is
        # interchangeable, key for key, with a wave-1 checkpoint.
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
        same `S_train` / `extra_train` objects are handed in on every L-BFGS
        evaluation, so the 16 field lookups are paid once, while any new tensor
        (a chaining step, a fresh split) recomputes. Identity, not content, so
        a recycled buffer can never serve a stale scale.
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
            raise ValueError("the residual network needs the normalised "
                             "(z0, dz) extra inputs")
        return self.straight(S, extra) \
            + self.residual_scale(S, extra) * self.raw(S, extra)


# ---------------------------------------------------------------- the twin --
def residual_data_loss(model, S, ref, extra=None):
    """The supervised twin, normalised by the residual scale.

    `_shared.model.data_loss` divides by the population-wide `out_scale`; that
    is the quantity this experiment is replacing, so the deviation is compared
    on its own scale instead. With the residual parametrisation this is exactly
    the MSE of `net_j` against the true deviation divided by the scale, i.e.
    the same O(1) target the physics loss now sees.
    """
    out = model(S, extra)
    sc = model.residual_scale(S, extra)
    return (((out - ref[:, :, :4]) / sc) ** 2).mean()


def build_residual_model(data, width, depth, field=None, node_profile="flat"):
    """The constructor the trainer and the analysis scripts share."""
    return ResidualOneStepNetwork(
        int(data["q"]), data["in_scale"], data["out_scale"],
        width=width, depth=depth, n_extra=2,
        c=np.asarray(data["c"]), extra_mean=np.asarray(data["extra_mean"]),
        extra_scale=np.asarray(data["extra_scale"]), field=field,
        node_profile=node_profile)
