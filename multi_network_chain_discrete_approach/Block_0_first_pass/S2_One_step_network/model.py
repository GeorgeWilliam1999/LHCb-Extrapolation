"""The paper's discrete-time network, for one LHCb magnet crossing.

Mirrors Van_Der_Pole/discrete_time_network/model.py (the clean-start Raissi
rebuild), adapted where the problem is genuinely different:

    (x, y, tx, ty, qop) at z0  ->  the q stage states + endpoint, (N, q+1, 4)

- the ODE is non-autonomous: the stage planes z_j = z0 + c_j*dz are FIXED
  constants of the frozen leg, so f evaluates the (differentiable) field at
  the network's proposed (x_j, y_j) on those planes;
- qop is a conserved input, not an output (the scheme's qop equations are
  trivial);
- inputs and outputs carry fixed physical scales (measured label-free in
  prepare_data); the reconstruction residuals are normalised per component by
  the input scale, making the loss dimensionless — the van der Pol state was
  O(1) and needed none of this.

The loss is the paper's alone: every one of the q+1 outputs must reconstruct
the input through the implicit Runge-Kutta equations. The data twin swaps
that for MSE against RK4 stage/endpoint labels — identical everything else.
"""
from __future__ import annotations

import torch

from field_torch import FieldTorch

torch.set_default_dtype(torch.float64)

KAPPA = 1.0e-3   # Allen convention, qop = 0.299792458 * q / p[GeV]


class LHCbRates(torch.nn.Module):
    """f(S, z) for S = (x, y, tx, ty) at fixed qop; z fixed per stage plane."""

    def __init__(self):
        super().__init__()
        self.field = FieldTorch()

    def forward(self, S4, qop, z):
        # S4: (N, q, 4), qop: (N, 1), z: (q,) fixed planes
        x, y, tx, ty = S4[..., 0], S4[..., 1], S4[..., 2], S4[..., 3]
        zz = z[None, :].expand_as(x)
        Bx, By, Bz = self.field(x, y, zz)
        k = KAPPA * qop
        N = torch.sqrt(1 + tx * tx + ty * ty)
        return torch.stack(
            [tx, ty,
             k * N * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz),
             k * N * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz)],
            dim=-1)


class OneStepNetwork(torch.nn.Module):
    """(5 inputs, scaled) -> (q+1, 4) stage states + endpoint (scaled out)."""

    def __init__(self, q: int, in_scale, out_scale, width: int = 50, depth: int = 4):
        super().__init__()
        self.q = int(q)
        self.register_buffer("in_scale", torch.as_tensor(in_scale))
        self.register_buffer("out_scale", torch.as_tensor(out_scale[:4]))
        layers, n_in = [], 5
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
            n_in = width
        layers += [torch.nn.Linear(n_in, 4 * (self.q + 1))]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, S):                      # S: (N, 5) physical units
        out = self.net(S / self.in_scale)
        return out.reshape(-1, self.q + 1, 4) * self.out_scale


def reconstruction_residuals(model, rates, S, dz, znodes, A, b):
    """(N, q+1, 4): every output's reconstruction of the input, minus the input,
    normalised per component by the input scale."""
    out = model(S)
    stages, endpoint = out[:, :-1, :], out[:, -1, :]
    qop = S[:, 4:5]
    F = rates(stages, qop, znodes)                              # (N, q, 4)
    rec_stages = stages - dz * torch.einsum("jk,nkd->njd", A, F)
    rec_end = endpoint - dz * torch.einsum("j,njd->nd", b, F)
    rec = torch.cat([rec_stages, rec_end[:, None, :]], dim=1)
    return (rec - S[:, None, :4]) / model.in_scale[:4]


def physics_loss(model, rates, S, dz, znodes, A, b):
    return (reconstruction_residuals(model, rates, S, dz, znodes, A, b) ** 2).mean()


def data_loss(model, S, ref):
    """The twin: MSE against RK4 stage+endpoint labels, same normalisation.
    ref: (N, q+1, 5) reference states at nodes + endpoint."""
    out = model(S)
    return (((out - ref[:, :, :4]) / model.out_scale) ** 2).mean()
