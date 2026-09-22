#!/usr/bin/env python
"""The fixed-step straight-line-residual network of Block D.

One network per fixed leg [z0, z0 + dz] of the magnet crossing. It is the
shared `OneStepNetwork` (`_shared/model.py`) with NO extra inputs - the leg is a
property of the network, held in two buffers, never an input - and its output
reinterpreted as the deviation from a straight line:

    output_j = straight_j  +  scale (x) net_j          j = 1 .. q, endpoint

    straight_j = (x + tx*(z_j - z0),  y + ty*(z_j - z0),  tx,  ty)
    z_j        = z0 + c_j*dz  (the q Gauss-Legendre nodes), z_{q+1} = z0 + dz

with the per-sample, label-free scale of the residual redesign (the Block A
result that made short legs work, carried into Block C and now here):

    scale_slope = kappa * |qop| * I_B                  (dimensionless)
    scale_pos   = kappa * |qop| * I_B * |dz| / 2       (mm)
    I_B         = INT_{z0}^{z0+dz} |B(x_s(z), y_s(z), z)| dz   [T mm]

I_B is the field integral along the straight line through the input state,
taken with a 16-point midpoint rule, so every symbol on the right-hand side is
an input. The floors (1e-9 mm, 1e-12) are the Block C ones and only guard an
exactly field-free sample. `kappa` is the Allen convention of
`_shared/reference.py`.

The physics loss is untouched: `_shared.model.physics_loss` only calls
`model(S)` and reconstructs the input from the absolute stage states this
forward returns. The supervised twin of Block D (D2) uses the same class with
q = 0 - one output block, the endpoint - and `residual_data_loss` below, which
divides the error by the residual scale so that the twin sees the same O(1)
target as the physics networks.

The state dict is the base network's, key for key (the buffers c, cout, z0,
dz, u are constant and rebuilt by the constructor), so the shared trainer's
checkpoint-every-restart and resume work unchanged.

Provenance: the residual parameterisation is that of the removed
`A3a_General_leg_network/residual_model.py` (in git history at a062bd8),
with (z0, dz) moved from the inputs into buffers and nothing else changed.
"""
from __future__ import annotations

import numpy as np
import torch

import use_shared                                        # noqa: F401
from _shared.field_torch import FieldTorch
from _shared.model import OneStepNetwork
from _shared.reference import KAPPA

torch.set_default_dtype(torch.float64)

POS_FLOOR_MM = 1e-9
SLOPE_FLOOR = 1e-12
N_FIELD_SAMPLES = 16
WIDTH, DEPTH = 128, 2          # the one architecture of Block D (George 2026-09-14)


# ------------------------------------------------------- numpy twins (checks) --
def field_integral_numpy(S, z0, dz, field, n_samples=N_FIELD_SAMPLES):
    """I_B along the straight line, midpoint rule; S (N, 5), z0 and dz scalars."""
    S = np.asarray(S, dtype=np.float64)
    u = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples
    s = u[None, :] * dz
    x = S[:, 0:1] + S[:, 2:3] * s
    y = S[:, 1:2] + S[:, 3:4] * s
    z = z0 + s
    Bx, By, Bz = field(x.ravel(), y.ravel(), np.broadcast_to(z, x.shape).ravel())
    B = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2).reshape(x.shape)
    return B.mean(axis=1) * abs(dz)


def scales_from_field_integral(S, dz, IB, pos_floor=POS_FLOOR_MM,
                               slope_floor=SLOPE_FLOOR):
    base = KAPPA * np.abs(np.asarray(S)[:, 4]) * np.asarray(IB)
    return (np.maximum(base * abs(dz) / 2.0, pos_floor),
            np.maximum(base, slope_floor))


def straight_line_states(S, dz, c):
    """(N, q+1, 4) straight-line stage states and endpoint, numpy."""
    S = np.asarray(S, dtype=np.float64)
    off = np.append(np.asarray(c, dtype=np.float64), 1.0) * dz          # (q+1,)
    out = np.empty((len(S), len(off), 4))
    out[:, :, 0] = S[:, 0:1] + S[:, 2:3] * off[None, :]
    out[:, :, 1] = S[:, 1:2] + S[:, 3:4] * off[None, :]
    out[:, :, 2] = S[:, 2:3]
    out[:, :, 3] = S[:, 3:4]
    return out


# ---------------------------------------------------------------- the model --
class FrozenResidualNetwork(OneStepNetwork):
    """(x, y, tx, ty, qop) at z0 -> q stage states + endpoint on a FIXED leg.

    c      (q,) the Gauss-Legendre nodes (empty for the q = 0 twin)
    z0, dz the leg, in mm (dz < 0 would be a backward leg; Block D is forward)
    field  a FieldV8R1 (None -> MagDown; Block D always passes the MagUp map)
    """

    def __init__(self, q, in_scale, out_scale, width=WIDTH, depth=DEPTH,
                 n_extra=0, c=None, z0=None, dz=None, field=None,
                 n_field_samples=N_FIELD_SAMPLES, pos_floor=POS_FLOOR_MM,
                 slope_floor=SLOPE_FLOOR):
        if n_extra:
            raise ValueError("the fixed-step network takes no extra inputs")
        super().__init__(q, in_scale, out_scale, width=width, depth=depth, n_extra=0)
        if c is None or z0 is None or dz is None:
            raise ValueError("the fixed-step network needs c, z0 and dz")
        cc = torch.as_tensor(np.asarray(c, dtype=np.float64)).reshape(-1)
        if len(cc) != self.q:
            raise ValueError("c has %d nodes but q = %d" % (len(cc), self.q))
        # held in a tuple ON PURPOSE: as a submodule the field twin's 23 MB of
        # grid buffers would be written into every checkpoint
        self._field_holder = (FieldTorch(field),)
        self.n_field_samples = int(n_field_samples)
        self.pos_floor = float(pos_floor)
        self.slope_floor = float(slope_floor)
        self.register_buffer("c", cc)
        self.register_buffer("cout", torch.cat([cc, torch.ones(1, dtype=cc.dtype)]))
        self.register_buffer("z0", torch.tensor(float(z0)))
        self.register_buffer("dz", torch.tensor(float(dz)))
        u = (torch.arange(self.n_field_samples, dtype=torch.float64) + 0.5) \
            / self.n_field_samples
        self.register_buffer("u", u)
        self._cache = None

    @property
    def field(self):
        return self._field_holder[0]

    def field_integral(self, S):
        """I_B along the straight line through S over this leg, (N,) in T mm."""
        with torch.no_grad():
            s = self.u[None, :] * self.dz
            x = S[:, 0:1] + S[:, 2:3] * s
            y = S[:, 1:2] + S[:, 3:4] * s
            z = (self.z0 + s).expand_as(x)
            Bx, By, Bz = self.field(x, y, z)
            B = torch.sqrt(Bx * Bx + By * By + Bz * Bz)
            return B.mean(dim=1) * self.dz.abs()

    def residual_scale(self, S):
        """(N, q+1, 4) the per-sample scale; cached on the identity of S."""
        cached = self._cache
        if cached is not None and cached[0] is S:
            return cached[1]
        with torch.no_grad():
            base = KAPPA * S[:, 4].abs() * self.field_integral(S)
            slope = base.clamp_min(self.slope_floor)
            pos = (base * self.dz.abs() / 2.0).clamp_min(self.pos_floor)
            ones = torch.ones_like(self.cout)
            sc = torch.stack([pos[:, None] * ones, pos[:, None] * ones,
                              slope[:, None] * ones, slope[:, None] * ones], dim=-1)
        self._cache = (S, sc)
        return sc

    def straight(self, S):
        off = self.cout[None, :] * self.dz                         # (1, q+1)
        x = S[:, 0:1] + S[:, 2:3] * off
        y = S[:, 1:2] + S[:, 3:4] * off
        tx = S[:, 2:3].expand_as(x)
        ty = S[:, 3:4].expand_as(x)
        return torch.stack([x, y, tx, ty], dim=-1)

    def raw(self, S):
        return self.net(S / self.in_scale).reshape(-1, self.q + 1, 4)

    def forward(self, S, extra=None):
        return self.straight(S) + self.residual_scale(S) * self.raw(S)


def residual_data_loss(model, S, ref, extra=None):
    """The twin's loss: MSE of the deviation in units of the residual scale."""
    out = model(S)
    return (((out - ref[:, :, :4]) / model.residual_scale(S)) ** 2).mean()


def build_model(data, field, width=WIDTH, depth=DEPTH):
    """The constructor every Block D script shares, from a leg dataset."""
    return FrozenResidualNetwork(
        int(data["q"]), np.asarray(data["in_scale"]), np.asarray(data["out_scale"]),
        width=width, depth=depth, c=np.asarray(data["c"]),
        z0=float(data["z0"]), dz=float(data["z1"]) - float(data["z0"]), field=field)
