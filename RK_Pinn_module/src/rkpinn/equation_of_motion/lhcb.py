"""The equation of motion of a charged particle in the LHCb magnetic field.

A track on a plane of constant z is described by the state

    S = (x, y, tx, ty, q/p),

where x and y are the position in millimetres, tx = dx/dz and ty = dy/dz are
the slopes and have no unit, and q/p is the charge divided by the momentum.
With z as the independent variable the Lorentz force gives

    dx/dz  = tx
    dy/dz  = ty
    dtx/dz = kappa (q/p) sqrt(1 + tx^2 + ty^2) (tx ty Bx - (1 + tx^2) By + ty Bz)
    dty/dz = kappa (q/p) sqrt(1 + tx^2 + ty^2) ((1 + ty^2) Bx - tx ty By - tx Bz)
    d(q/p)/dz = 0

where (Bx, By, Bz) is the field in tesla at (x, y, z) and kappa fixes the
units. The rates depend on z through the field, so the equation is not
autonomous. q/p is conserved: it is an input to every step, never predicted.

Units: q/p is stored as 0.299792458 * charge / momentum in GeV, and kappa is
1e-3. These are the conventions of the Allen reconstruction software.

Ported from the frozen code, with the arithmetic unchanged and in the same
order, so that the rates are identical to the last bit:

    LhcbEquationOfMotion                from  _shared/reference.py  (deriv)
    DifferentiableLhcbEquationOfMotion  from  _shared/model.py      (LHCbRates)

The field map is an argument of both. Neither has a default polarity.
Gates: tests/test_equation_of_motion_matches_the_frozen_code.py.
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.equation_of_motion.field_map import DifferentiableFieldMap, FieldMap
from rkpinn.registry import register

KAPPA = 1.0e-3
CHARGE_OVER_MOMENTUM_PER_INVERSE_GEV = 0.299792458


def charge_over_momentum(charge, momentum_gev):
    """q/p in the units of the state, from the charge and the momentum in GeV."""
    return CHARGE_OVER_MOMENTUM_PER_INVERSE_GEV * charge / momentum_gev


@register("equation_of_motion", "lhcb")
class LhcbEquationOfMotion:
    """The rates of change of the state with z, in numpy.

    number_of_components is the number of components that change along the
    track: x, y, tx and ty. The state itself has one more entry, q/p, whose
    rate is zero.
    """

    number_of_components = 4
    width_of_state = 5

    def __init__(self, field_map: FieldMap):
        self.field_map = field_map

    def rates(self, state, z):
        """dS/dz for states of shape (n, 5); z is one plane or one plane per row.

        Returns shape (n, 5). The last column, the rate of q/p, is zero.
        """
        x, y, tx, ty, qop = state.T
        if np.isscalar(z):
            z = np.full_like(x, z)
        Bx, By, Bz = self.field_map(x, y, z)
        k = KAPPA * qop
        N = np.sqrt(1 + tx * tx + ty * ty)
        return np.stack(
            [
                tx,
                ty,
                k * N * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz),
                k * N * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz),
                np.zeros_like(x),
            ],
            axis=1,
        )


@register("equation_of_motion", "lhcb_differentiable")
class DifferentiableLhcbEquationOfMotion(torch.nn.Module):
    """The same rates in torch, in double precision, differentiable with
    respect to the state. The field is evaluated at the positions given, so a
    gradient flows through the field to those positions.
    """

    number_of_components = 4
    width_of_state = 5

    def __init__(self, field_map: FieldMap):
        super().__init__()
        self.field = DifferentiableFieldMap(field_map)

    def rates(self, state_without_charge_over_momentum, charge_over_momentum, z):
        """The rates of (x, y, tx, ty) at the states given.

        state_without_charge_over_momentum: shape (n, q, 4)
        charge_over_momentum:               shape (n, 1)
        z: the planes the states lie on, shape (q,) when every row shares
           them, or (n, q) when each row has its own
        Returns shape (n, q, 4).
        """
        S4 = state_without_charge_over_momentum
        x, y, tx, ty = S4[..., 0], S4[..., 1], S4[..., 2], S4[..., 3]
        zz = z[None, :].expand_as(x) if z.dim() == 1 else z
        Bx, By, Bz = self.field(x, y, zz)
        k = KAPPA * charge_over_momentum
        N = torch.sqrt(1 + tx * tx + ty * ty)
        return torch.stack(
            [tx, ty,
             k * N * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz),
             k * N * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz)],
            dim=-1)

    def forward(self, state_without_charge_over_momentum, charge_over_momentum, z):
        return self.rates(state_without_charge_over_momentum, charge_over_momentum, z)
