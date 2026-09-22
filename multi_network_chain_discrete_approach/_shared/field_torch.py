"""A differentiable torch fp64 twin of the canonical v8r1 trilinear field.

The physics loss evaluates the rates at the network's proposed stage states,
and autograd must flow through B(x, y, z) with respect to those positions. This
module re-implements the vendored numpy loader's trilinear interpolation
(`field_v8r1.py`, mirrored index arithmetic) on torch tensors; the grid data are
loaded once via the numpy loader, so the two share bytes.

Generalised from `S2_One_step_network/field_torch.py` in exactly one way: the
constructor takes ANY `FieldV8R1` instance, so the MagDown and MagUp maps can
both be used in one process. Called with no argument it uses MagDown, as
before, and the arithmetic is unchanged.

GATE (`python field_torch.py`, or `parity(...)`): agreement with the numpy
original on 200k random points in the map, required < 1e-12 T. Gradients exist
almost everywhere (the field is C0: piecewise-trilinear, so the gradient jumps
at cell faces - the measured ~30 um scheme floor; torch returns the one-sided
value there, which is fine for optimisation).
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from .reference import make_field, field_bounds
except ImportError:                       # pragma: no cover  (run as a script)
    from reference import make_field, field_bounds

torch.set_default_dtype(torch.float64)


class FieldTorch(torch.nn.Module):
    """B(x, y, z) in Tesla, differentiable, matching the numpy loader exactly."""

    def __init__(self, np_field=None):
        super().__init__()
        if np_field is None:
            np_field = make_field("down")
        self.np_field = np_field
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


def parity(which: str = "down", n: int = 200_000, seed: int = 0,
           tol: float = 1e-12, verbose: bool = True) -> dict:
    """The gate: torch twin vs the numpy loader on n random in-map points.

    Returns the measurement; raises AssertionError if it fails.
    """
    np_field = make_field(which)
    ft = FieldTorch(np_field)
    lo, hi = field_bounds(np_field)
    rng = np.random.default_rng(seed)
    pts = lo + rng.random((n, 3)) * (hi - lo)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    Bn = np.stack(np_field(x, y, z), axis=1)
    with torch.no_grad():
        Bt = torch.stack(ft(torch.tensor(x), torch.tensor(y), torch.tensor(z)),
                         dim=1).numpy()
    worst = float(np.abs(Bn - Bt).max())

    # gradient existence smoke test
    xt = torch.tensor([100.0, -50.0], requires_grad=True)
    _, By, _ = ft(xt, torch.tensor([50.0, 10.0]), torch.tensor([5000.0, 4000.0]))
    By.sum().backward()
    grad = xt.grad.numpy().tolist()

    out = {"field": which, "n_points": n, "seed": seed,
           "max_abs_diff_T": worst, "tol_T": tol,
           "passes": bool(worst < tol), "dBy_dx_smoke": grad}
    if verbose:
        print("parity gate [%s]: worst |numpy - torch| = %.3e T on %d points -> %s"
              % (which, worst, n, "PASS" if out["passes"] else "FAIL"))
        print("grad smoke: dBy/dx =", grad)
    assert out["passes"], "torch field twin disagrees with the canonical loader"
    return out


if __name__ == "__main__":
    for which in ("down", "up"):
        parity(which)
