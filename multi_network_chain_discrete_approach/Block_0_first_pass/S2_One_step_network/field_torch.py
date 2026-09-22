"""A differentiable torch fp64 twin of the canonical v8r1 trilinear field.

The physics loss evaluates f at the network's proposed stage states, and
autograd must flow through B(x, y, z) with respect to the stage positions.
This module re-implements the archive loader's trilinear interpolation
(core/field_v8r1.py, mirrored index arithmetic) on torch tensors; the grid
data are loaded once via the numpy loader, so the two share bytes.

GATE (run this file): parity vs the numpy original on 200k random points in
the detector volume, required < 1e-12 T. Gradients exist a.e. (the field is
C0: piecewise-trilinear — gradient jumps at cell faces, the measured ~30 um
scheme floor; torch returns the one-sided value there, which is fine for
optimisation).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "S0_Baseline_data_exploration"))
from reference_card import FIELD  # noqa: E402  (the numpy loader, already gated)

torch.set_default_dtype(torch.float64)


class FieldTorch(torch.nn.Module):
    def __init__(self, np_field=FIELD):
        super().__init__()
        self.register_buffer("invD", torch.tensor(np_field.invD))
        self.register_buffer("mn", torch.tensor(np_field.min))
        self.register_buffer("Bx", torch.tensor(np_field.Bx * np_field.scale))
        self.register_buffer("By", torch.tensor(np_field.By * np_field.scale))
        self.register_buffer("Bz", torch.tensor(np_field.Bz * np_field.scale))
        self.N = tuple(int(n) for n in np_field.N)

    def forward(self, x, y, z):
        fx = (x - self.mn[0]) * self.invD[0]
        fy = (y - self.mn[1]) * self.invD[1]
        fz = (z - self.mn[2]) * self.invD[2]
        ix = fx.detach().floor().long().clamp(0, self.N[0] - 2)
        iy = fy.detach().floor().long().clamp(0, self.N[1] - 2)
        iz = fz.detach().floor().long().clamp(0, self.N[2] - 2)
        tx = (fx - ix).clamp(0.0, 1.0)
        ty = (fy - iy).clamp(0.0, 1.0)
        tz = (fz - iz).clamp(0.0, 1.0)

        def tri(G):
            c000 = G[iz, iy, ix];         c100 = G[iz, iy, ix + 1]
            c010 = G[iz, iy + 1, ix];     c110 = G[iz, iy + 1, ix + 1]
            c001 = G[iz + 1, iy, ix];     c101 = G[iz + 1, iy, ix + 1]
            c011 = G[iz + 1, iy + 1, ix]; c111 = G[iz + 1, iy + 1, ix + 1]
            c00 = c000 * (1 - tx) + c100 * tx
            c10 = c010 * (1 - tx) + c110 * tx
            c01 = c001 * (1 - tx) + c101 * tx
            c11 = c011 * (1 - tx) + c111 * tx
            c0 = c00 * (1 - ty) + c10 * ty
            c1 = c01 * (1 - ty) + c11 * ty
            return c0 * (1 - tz) + c1 * tz

        return tri(self.Bx), tri(self.By), tri(self.Bz)


if __name__ == "__main__":
    ft = FieldTorch()
    rng = np.random.default_rng(0)
    n = 200_000
    x = rng.uniform(-3000, 3000, n)
    y = rng.uniform(-2500, 2500, n)
    z = rng.uniform(-500, 12000, n)
    Bn = np.stack(FIELD(x, y, z), axis=1)
    with torch.no_grad():
        Bt = torch.stack(ft(torch.tensor(x), torch.tensor(y), torch.tensor(z)), dim=1).numpy()
    worst = np.abs(Bn - Bt).max()
    print("parity gate: worst |numpy - torch| = %.3e T on %d points" % (worst, n))
    assert worst < 1e-12, "torch field twin disagrees with the canonical loader"
    # gradient existence smoke test
    xt = torch.tensor([100.0, -50.0], requires_grad=True)
    _, By, _ = ft(xt, torch.tensor([50.0, 10.0]), torch.tensor([5000.0, 4000.0]))
    By.sum().backward()
    print("grad smoke: dBy/dx =", xt.grad.numpy(), "-> GATE PASS")
