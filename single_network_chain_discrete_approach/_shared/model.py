"""The paper's discrete-time network for an LHCb magnet crossing, generalised.

This is `S2_One_step_network/model.py` (the verified baseline, used unchanged by
S2b_One_step_network_v2) with the frozen-leg assumptions lifted and nothing else
touched:

  * the network may take `n_extra` further inputs, already normalised by the
    caller - the general-leg experiments pass (z0, dz) there;
  * the stage planes may be per sample: `znodes` may be the (q,) vector of the
    frozen leg or an (N, q) tensor of per-sample planes z0 + c_j*dz;
  * the step length `dz` may be a python float (frozen leg) or an (N,) tensor.

With `n_extra = 0`, a (q,) `znodes` and a float `dz`, every arithmetic
operation is the one the baseline performed, in the same order, so the losses
reproduce the baseline bit-for-bit (checked in `smoke_tests.py`).

    (x, y, tx, ty, qop) at z0  [+ extras]  ->  q stage states + endpoint,
                                               shape (N, q+1, 4)

- the ODE is non-autonomous: the rates evaluate the differentiable field at the
  network's proposed (x_j, y_j) on the stage planes;
- qop is a conserved input, not an output (the scheme's qop equations are
  trivial);
- inputs and outputs carry fixed physical scales (measured label-free in
  `prepare.py`); the reconstruction residuals are normalised per component by
  the input scale, making the loss dimensionless - the van der Pol state was
  O(1) and needed none of this.

The physics loss is the paper's alone: every one of the q+1 outputs must
reconstruct the input through the implicit Runge-Kutta equations. The data twin
swaps that for MSE against RK4 stage/endpoint labels - identical everything
else.
"""
from __future__ import annotations

import torch

try:
    from .field_torch import FieldTorch
except ImportError:                       # pragma: no cover  (run as a script)
    from field_torch import FieldTorch

torch.set_default_dtype(torch.float64)

KAPPA = 1.0e-3   # Allen convention, qop = 0.299792458 * q / p[GeV]


class LHCbRates(torch.nn.Module):
    """f(S, z) for S = (x, y, tx, ty) at fixed qop.

    z is the stage-plane tensor: shape (q,) when the planes are constants of a
    frozen leg, or (N, q) when every sample has its own leg.
    """

    def __init__(self, field=None):
        super().__init__()
        self.field = FieldTorch(field)

    def forward(self, S4, qop, z):
        # S4: (N, q, 4), qop: (N, 1), z: (q,) or (N, q)
        x, y, tx, ty = S4[..., 0], S4[..., 1], S4[..., 2], S4[..., 3]
        zz = z[None, :].expand_as(x) if z.dim() == 1 else z
        Bx, By, Bz = self.field(x, y, zz)
        k = KAPPA * qop
        N = torch.sqrt(1 + tx * tx + ty * ty)
        return torch.stack(
            [tx, ty,
             k * N * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz),
             k * N * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz)],
            dim=-1)


class OneStepNetwork(torch.nn.Module):
    """(5 + n_extra inputs, scaled) -> (q+1, 4) stage states + endpoint (scaled).

    n_extra: how many already-normalised extra inputs the forward pass expects
    (0 for a frozen leg; 2 for the general leg, carrying z0 and dz).
    """

    def __init__(self, q: int, in_scale, out_scale, width: int = 50,
                 depth: int = 4, n_extra: int = 0):
        super().__init__()
        self.q = int(q)
        self.n_extra = int(n_extra)
        self.register_buffer("in_scale", torch.as_tensor(in_scale))
        self.register_buffer("out_scale", torch.as_tensor(out_scale[:4]))
        layers, n_in = [], 5 + self.n_extra
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
            n_in = width
        layers += [torch.nn.Linear(n_in, 4 * (self.q + 1))]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, S, extra=None):          # S: (N, 5) physical units
        if self.n_extra == 0:
            out = self.net(S / self.in_scale)
        else:
            if extra is None:
                raise ValueError("this network expects %d extra normalised "
                                 "inputs; none were given" % self.n_extra)
            out = self.net(torch.cat([S / self.in_scale, extra], dim=1))
        return out.reshape(-1, self.q + 1, 4) * self.out_scale


def _dz_views(dz):
    """(stage view, endpoint view) of the step length.

    A python float or 0-dim tensor is returned untouched, so the frozen-leg
    arithmetic is byte-identical to the baseline; an (N,) tensor is broadcast
    to (N, 1, 1) for the stages and (N, 1) for the endpoint.
    """
    if torch.is_tensor(dz) and dz.dim() > 0:
        return dz.reshape(-1, 1, 1), dz.reshape(-1, 1)
    return dz, dz


def reconstruction_residuals(model, rates, S, dz, znodes, A, b, extra=None):
    """(N, q+1, 4): every output's reconstruction of the input, minus the input,
    normalised per component by the input scale."""
    out = model(S) if model.n_extra == 0 else model(S, extra)
    stages, endpoint = out[:, :-1, :], out[:, -1, :]
    qop = S[:, 4:5]
    F = rates(stages, qop, znodes)                              # (N, q, 4)
    dz_stage, dz_end = _dz_views(dz)
    rec_stages = stages - dz_stage * torch.einsum("jk,nkd->njd", A, F)
    rec_end = endpoint - dz_end * torch.einsum("j,njd->nd", b, F)
    rec = torch.cat([rec_stages, rec_end[:, None, :]], dim=1)
    return (rec - S[:, None, :4]) / model.in_scale[:4]


def physics_loss(model, rates, S, dz, znodes, A, b, extra=None):
    """The paper's loss: the mean squared normalised reconstruction residual."""
    return (reconstruction_residuals(model, rates, S, dz, znodes, A, b,
                                     extra) ** 2).mean()


def data_loss(model, S, ref, extra=None):
    """The twin: MSE against RK4 stage+endpoint labels, same normalisation.
    ref: (N, q+1, 5) reference states at nodes + endpoint."""
    out = model(S) if model.n_extra == 0 else model(S, extra)
    return (((out - ref[:, :, :4]) / model.out_scale) ** 2).mean()
