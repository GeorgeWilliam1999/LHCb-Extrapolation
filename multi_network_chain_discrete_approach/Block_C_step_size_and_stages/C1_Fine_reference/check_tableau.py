#!/usr/bin/env python
"""C1.1 - the verification battery for the sixth-order reference tableau.

`_shared/reference.py` now carries `rk6_rows`: Butcher's seven-stage explicit
Runge-Kutta method of order six, vectorised over rows exactly like `rk4_rows`.
The coefficients were copied from the van der Pol study's reference machinery
(`/data/bfys/gscriven/Van_Der_Pole/RK_Truth/rk6.py`, read-only). A copied
tableau is worth nothing until it has been checked on the machine that will
use it, so this script re-runs that file's battery here and adds two checks it
could not make, because they are about the new implementation rather than
about the numbers:

  1. PROVENANCE. The seven arrays in `_shared/reference.py` are compared
     element by element with the arrays in the source file, imported directly
     from it. Exact equality is required, not a tolerance.

  2. IDENTITIES. Every row of A sums to its node c, the weights b sum to one,
     and A is strictly lower triangular (the method is explicit). This is
     `rk6.py`'s `verify_tableau`, re-run on this node.

  3. ORDER ON THE THREE EXACT-SOLUTION PROBLEMS. `rk6.py`'s own test set: a
     linear problem, a nonlinear one, and one whose right-hand side depends on
     time explicitly (a linear autonomous problem alone cannot catch a wrong
     node vector). The global error at t = 1 is measured against the exact
     solution over a ladder of step sizes and the slope of log(error) against
     log(h) is fitted, excluding the points that have fallen to roundoff. The
     slope must be 6 to within 0.35. The stepper used here is a plain generic
     one driven by the SAME coefficient arrays that `rk6_rows` uses, so this
     tests the numbers.

  4. ORDER OF `rk6_rows` ITSELF, ON THE LHCb EQUATION OF MOTION. The three
     problems above cannot be run through `rk6_rows`, which is hard-wired to
     the LHCb ODE. So the ODE is kept and the FIELD is replaced by a smooth
     analytic stand-in with the same call signature (a solenoid-like blob,
     infinitely differentiable, of about the right size in Tesla and about the
     right extent in z). On a smooth field the scheme must show its order, and
     any mistake in the masked stepping, the shortened last step or the
     direction handling shows up as a lost order. The error is measured
     against the same routine at a 0.5 mm step, over a ladder of coarse steps
     (640 -> 80 mm) chosen because on a smooth field this scheme is already at
     the fp64 floor by a 40 mm step. This tests the code.

     The real v8r1 map is trilinear on a 100 mm grid, i.e. only C0: its
     derivative jumps at every cell face. That is a property of the map, not
     of the scheme, and it is what C1.2 measures. It is why the order check
     has to be done on a smooth field.

  5. MECHANICS. Four properties of the implementation that are not about
     order: the last step lands exactly on z1 for a span that is not a whole
     number of steps; a backward leg (z1 < z0) integrates and closes on the
     forward one; q/p passes through bit-exactly; and rows with different
     (z0, z1) in one call get the same answers as when run one at a time (the
     masking is correct).

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python check_tableau.py

Output: results/tableau_checks.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import importlib.util
import json
import platform
import sys
import time

import numpy as np

import use_shared                                            # noqa: F401
from _shared.reference import (KAPPA, RK6_A, RK6_B, RK6_C, RK6_ORDER,
                               RK6_STAGES, deriv, rk6_rows)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
RK6_SOURCE = "/data/bfys/gscriven/Van_Der_Pole/RK_Truth/rk6.py"
ORDER_TOL = 0.35            # rk6.py's own tolerance on the fitted slope


# --------------------------------------------------------------- 1. source ---
def load_source_module(path):
    spec = importlib.util.spec_from_file_location("_rk6_source", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_provenance(src):
    same = {
        "C": bool(np.array_equal(RK6_C, src.C)),
        "A": bool(np.array_equal(RK6_A, src.A)),
        "B": bool(np.array_equal(RK6_B, src.B)),
        "ORDER": RK6_ORDER == src.ORDER,
        "N_STAGES": RK6_STAGES == src.N_STAGES,
    }
    return {
        "source_file": RK6_SOURCE,
        "compared": "element by element, exact equality (no tolerance)",
        "identical": same,
        "max_abs_difference": {
            "C": float(np.abs(RK6_C - src.C).max()),
            "A": float(np.abs(RK6_A - src.A).max()),
            "B": float(np.abs(RK6_B - src.B).max()),
        },
        "passes": bool(all(same.values())),
    }


# ----------------------------------------------------------- 2. identities ---
def check_identities(tol=1e-15):
    row_sums = float(np.abs(RK6_A.sum(axis=1) - RK6_C).max())
    weight_sum = float(abs(RK6_B.sum() - 1.0))
    upper = float(np.abs(np.triu(RK6_A)).max())
    return {
        "what": "each row of A sums to its node c; b sums to 1; A strictly "
                "lower triangular (explicit)",
        "max_row_sum_error": row_sums,
        "weight_sum_error": weight_sum,
        "upper_triangle_max": upper,
        "tolerance": tol,
        "passes": bool(row_sums < tol and weight_sum < tol and upper == 0.0),
    }


# ------------------------------------- 3. order on the exact-solution set ----
def generic_step(f, t, y, h):
    """One RK6 step of a generic f(t, y), driven by the shared coefficients."""
    k = np.empty((RK6_STAGES,) + y.shape)
    k[0] = f(t, y)
    for i in range(1, RK6_STAGES):
        k[i] = f(t + RK6_C[i] * h,
                 y + h * np.tensordot(RK6_A[i, :i], k[:i], axes=(0, 0)))
    return y + h * np.tensordot(RK6_B, k, axes=(0, 0))


def generic_integrate_to(f, y0, t_end, h, t0=0.0):
    y = np.atleast_2d(np.asarray(y0, dtype=float))
    n = max(1, int(round((t_end - t0) / h)))
    h = (t_end - t0) / n
    for i in range(n):
        y = generic_step(f, t0 + i * h, y, h)
    return y


TEST_PROBLEMS = {
    "linear:  dy/dt = -y": (
        lambda t, y: -y, np.array([[1.0]]), lambda t: np.exp(-t)),
    "nonlinear:  dy/dt = -y^2": (
        lambda t, y: -y ** 2, np.array([[1.0]]), lambda t: 1.0 / (1.0 + t)),
    "non-autonomous:  dy/dt = y cos t": (
        lambda t, y: y * np.cos(t), np.array([[1.0]]),
        lambda t: np.exp(np.sin(t))),
}


def fitted_slope(hs, errs, floor=1e-13):
    usable = errs > floor
    slope = (float(np.polyfit(np.log(hs[usable]), np.log(errs[usable]), 1)[0])
             if usable.sum() >= 2 else float("nan"))
    return slope, usable


def check_order_exact_problems():
    hs = np.geomspace(0.2, 0.01, 8)
    out = {"what": "global error at t = 1 against the exact solution; the "
                   "slope of log(error) vs log(h) is the order",
           "step_sizes": hs.tolist(), "expected_order": RK6_ORDER,
           "tolerance_on_slope": ORDER_TOL, "problems": {}}
    ok = True
    for name, (f, y0, exact) in TEST_PROBLEMS.items():
        errs = np.array([abs(float(generic_integrate_to(f, y0, 1.0, h)[0, 0])
                             - exact(1.0)) for h in hs])
        slope, usable = fitted_slope(hs, errs)
        good = bool(abs(slope - RK6_ORDER) < ORDER_TOL)
        ok &= good
        out["problems"][name] = {
            "errors": errs.tolist(),
            "points_above_roundoff": int(usable.sum()),
            "fitted_order": slope, "passes": good}
    out["passes"] = bool(ok)
    return out


# ------------------------------- 4. order of rk6_rows on a smooth field ------
class SmoothField:
    """A smooth analytic stand-in for the map, with the map's call signature.

    B = (Bx, By, Bz) in Tesla, infinitely differentiable everywhere, peaking at
    about one Tesla near z = 4700 mm and falling away over about a metre - the
    right size and roughly the right extent, so that the trajectories it makes
    are LHCb-like without the trilinear map's kinks at the cell faces.
    """

    Z0, W = 4700.0, 1200.0
    XY = 3000.0

    def __call__(self, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        g = np.exp(-((z - self.Z0) / self.W) ** 2) * np.exp(
            -((x / self.XY) ** 2 + (y / self.XY) ** 2))
        by = -1.05 * g
        bx = 0.08 * g * (y / self.XY)
        bz = 0.15 * g * ((z - self.Z0) / self.W) * (x / self.XY)
        return bx, by, bz


def _legs(n=6, seed=11):
    rng = np.random.default_rng(seed)
    S = np.column_stack([
        rng.normal(0.0, 40.0, n), rng.normal(0.0, 25.0, n),
        rng.normal(0.0, 0.12, n), rng.normal(0.0, 0.04, n),
        0.299792458 * rng.choice([-1.0, 1.0], n) / rng.uniform(3.0, 60.0, n)])
    return S


def check_order_rk6_rows_smooth():
    field = SmoothField()
    S = _legs()
    z0, z1 = 2648.2, 7826.0
    # the ladder has to sit in the asymptotic band: on this field rk6_rows is
    # already at the fp64 floor (~2e-12 mm) by a 40 mm step, so a finer ladder
    # would measure rounding, not the method.
    hs = np.array([640.0, 320.0, 160.0, 80.0])
    fine_step = 0.5
    fine = rk6_rows(S, z0, z1, step=fine_step, field=field)
    errs = np.array([
        float(np.abs(rk6_rows(S, z0, z1, step=h, field=field)[:, :4]
                     - fine[:, :4]).max()) for h in hs])
    slope, usable = fitted_slope(hs, errs, floor=1e-10)
    good = bool(abs(slope - RK6_ORDER) < ORDER_TOL)
    return {
        "what": "the LHCb ODE with a smooth analytic field in place of the "
                "trilinear map, integrated by rk6_rows itself; error against "
                "the same routine at step %g mm" % fine_step,
        "why": "the v8r1 map is trilinear, hence only C0, so its own "
               "convergence flattens at the order the kinks allow (C1.2). A "
               "smooth field is the only way to measure the SCHEME's order "
               "through this code path.",
        "field": "analytic gaussian blob, peak |B| ~ 1.05 T at z = 4700 mm",
        "leg_mm": [z0, z1], "n_rows": int(len(S)),
        "step_sizes_mm": hs.tolist(),
        "max_abs_error_mm": errs.tolist(),
        "points_above_floor": int(usable.sum()),
        "fitted_order": slope, "expected_order": RK6_ORDER,
        "tolerance_on_slope": ORDER_TOL, "passes": good,
    }


# ------------------------------------------------------------ 5. mechanics ---
def check_mechanics():
    field = SmoothField()
    S = _legs(n=5, seed=3)
    z0, z1 = 2648.2, 7826.0                    # 5177.8 mm: not a whole
    step = 1.0                                 # number of 1 mm steps
    fwd = rk6_rows(S, z0, z1, step=step, field=field)
    back = rk6_rows(fwd, z1, z0, step=step, field=field)
    closure = float(np.abs(back[:, :4] - S[:, :4]).max())

    # a single row on its own must equal the same row inside a mixed batch
    zz0 = np.array([z0, z0, 3000.0, 7826.0, 5000.0])
    zz1 = np.array([z1, 5000.0, 7826.0, 2648.2, 2648.2])
    batch = rk6_rows(S, zz0, zz1, step=step, field=field)
    one_by_one = np.vstack([rk6_rows(S[i:i + 1], zz0[i], zz1[i], step=step,
                                     field=field) for i in range(len(S))])
    masking = float(np.abs(batch - one_by_one).max())

    # a leg shorter than one step, and a zero-length leg
    tiny = rk6_rows(S, z0, z0 + 0.03, step=step, field=field)
    zero = rk6_rows(S, z0, z0, step=step, field=field)

    return {
        "span_not_a_whole_number_of_steps_mm": z1 - z0,
        "forward_then_back_closure_mm": closure,
        "backward_leg_runs": bool(np.isfinite(back).all()),
        "batch_vs_one_at_a_time_max_abs_diff": masking,
        "batch_legs": [list(map(float, zz0)), list(map(float, zz1))],
        "leg_shorter_than_one_step_finite": bool(np.isfinite(tiny).all()),
        "zero_length_leg_returns_start_exactly":
            bool(np.array_equal(zero, S)),
        "qop_passthrough_exact": bool(np.array_equal(fwd[:, 4], S[:, 4])),
        "passes": bool(closure < 1e-9 and masking == 0.0
                       and np.isfinite(tiny).all()
                       and np.array_equal(zero, S)
                       and np.array_equal(fwd[:, 4], S[:, 4])),
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    src = load_source_module(RK6_SOURCE)
    checks = {
        "1_provenance": check_provenance(src),
        "2_identities": check_identities(),
        "3_order_on_exact_solution_problems": check_order_exact_problems(),
        "4_order_of_rk6_rows_on_a_smooth_field": check_order_rk6_rows_smooth(),
        "5_mechanics": check_mechanics(),
    }
    out = {
        "what": "C1.1 verification battery for _shared.reference.rk6_rows",
        "tableau": "Butcher, seven stages, order six (explicit)",
        "kappa": KAPPA,
        "node": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "wall_s": None,
        "checks": checks,
        "all_pass": bool(all(c["passes"] for c in checks.values())),
    }
    out["wall_s"] = round(time.time() - t0, 2)
    with open(os.path.join(RESULTS, "tableau_checks.json"), "w") as f:
        json.dump(out, f, indent=1)
    for name, c in checks.items():
        print("%-42s %s" % (name, "ok" if c["passes"] else "FAIL"))
    print("orders:",
          {k: round(v["fitted_order"], 3)
           for k, v in checks["3_order_on_exact_solution_problems"]
           ["problems"].items()},
          "| rk6_rows on a smooth field:",
          round(checks["4_order_of_rk6_rows_on_a_smooth_field"]["fitted_order"], 3))
    print("all pass:", out["all_pass"], " (%.1f s)" % out["wall_s"])
    assert out["all_pass"], "the RK6 tableau battery did not pass"


if __name__ == "__main__":
    main()
