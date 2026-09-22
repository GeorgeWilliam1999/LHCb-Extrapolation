"""The exact Gauss-Legendre scheme on the LHCb ODE, with no network in it.

For a step from a state S0 at plane z0 to the plane z0 + dz, the q-stage
implicit Runge-Kutta scheme defines q *stage states* Y_j, each living at the
plane z0 + c_j*dz, through

    Y_j = S0 + dz * sum_k a_jk f(Y_k, z0 + c_k*dz),     j = 1 .. q      (1)
    S1  = S0 + dz * sum_j b_j  f(Y_j, z0 + c_j*dz),                     (2)

where (c, A, b) is the q-stage Gauss-Legendre tableau (`_shared/irk.py`) and f
is the LHCb equation of motion (`_shared/reference.deriv`). Equation (1) is
implicit: the unknown stage states appear on both sides. This module solves it
directly with a root-finder, so the numbers it produces are *the scheme's own
answer* -- exactly what a network trained on the same equations would produce
if its physics loss reached zero. Its error against the fine sixth-order
reference is therefore the **ceiling**: the best any such network could do.

This is `../../Block_0_first_pass/S1_Simple_first_pass/exact_scheme.py`'s `solve_leg` generalised in
three ways and in no others:

  * **the field is an argument.** That file calls `deriv(S, znodes)` with no
    field, i.e. the MagDown map. The sample is a MagUp sample
    (`../C0_Magnet_tracks_dataset/README.md`), so the polarity has to be chosen
    by the caller rather than inherited from a default.
  * **z0 and dz are per sample.** That file took a leg (z0, z1) pair; the
    Block C dataset carries a start plane and a step length per row.
  * **the residual tolerance is 1e-9** (it used 1e-8), tightened because the
    strata reach down to steps of 0.05 mm where the scheme's own error is
    small enough for a loose tolerance to be visible in the answer.

Everything else is character for character the same: qop is held fixed so the
unknowns are 4 per stage rather than 5; the residual is scaled to [mm, mm,
mrad, mrad] so that the mixed-unit system is well conditioned for the solver;
the initial guess is the straight line through the input state, which uses only
what a network would be given; and convergence is judged **on the residual**,
not on the solver's own flag -- `scipy.optimize.root` reports failure when
asked for a tighter *step* tolerance than it can deliver even though the
equations are already satisfied to roundoff.

Nothing in `_shared/` is changed by this module: `_shared/irk.py`'s generic
`exact_step` already takes the right-hand side as a callable, so a caller binds
whichever field it wants into that closure. This file is the LHCb-specific
binding, kept in the experiment folder so that the experiments that import it
all get the same code (as `../../Block_A_technique_works/A1_Stage_count_sweep` imports the first pass's).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import root

import use_shared  # noqa: F401  (puts _shared on sys.path)
from _shared.reference import deriv, make_field

# The residual is formed in (mm, mm, rad, rad) and divided by this, so that
# every component of the vector the solver sees is in mm or mrad.
SCALE = np.array([1.0, 1.0, 1e-3, 1e-3])

# The residual tolerance, in those units: a solve counts as converged when
# every component of the residual is below it.
RESIDUAL_TOL = 1e-9

# Handed to `root` as its *step* tolerance. It is deliberately far below
# RESIDUAL_TOL: the solver is asked to keep going as long as it can make
# progress, and whether it got there is decided afterwards on the residual.
STEP_TOL = 1e-12


def straight_line_nodes(S0, znodes, z0):
    """The initial guess: the input state carried along its own slopes.

    (x, y) advance linearly and (tx, ty) do not change -- the trajectory a
    particle would follow with the magnet switched off. It uses nothing but the
    input state, which is the same information a network is given.
    """
    dzn = znodes - z0
    return np.column_stack([
        S0[0] + S0[2] * dzn,
        S0[1] + S0[3] * dzn,
        np.full(len(znodes), S0[2]),
        np.full(len(znodes), S0[3]),
    ])


def solve_state(S0, z0, dz, tab, field, tol=RESIDUAL_TOL, maxfev=0):
    """Solve the q-stage implicit equations for one state and one step.

    S0     (5,) the start state (x, y, tx, ty, qop), fp64, mm and rad
    z0     the start plane in mm
    dz     the step length in mm; negative integrates backwards
    tab    (c, A, b) from `_shared.reference.gauss_legendre`
    field  a FieldV8R1 -- the polarity is the caller's choice, not a default
    tol    the residual tolerance in mm and mrad
    maxfev 0 leaves scipy's default of 200*(4q + 1) function evaluations

    Returns (stages, S1, converged, n_eval, residual):
      stages    (q, 5) the stage states, qop carried through
      S1        (5,) the end state at z0 + dz
      converged True when max |residual| < tol
      n_eval    residual evaluations the root-finder used (its cost measure)
      residual  max |residual| at the returned point, in mm and mrad
    """
    c, A, b = tab
    q = len(c)
    znodes = z0 + c * dz
    qop = S0[4]
    S_work = np.empty((q, 5))
    S_work[:, 4] = qop

    n_eval = [0]

    def rates(Y4):
        """f(Y_j, z_j) for all q stages, as a (q, 4) array. One deriv call."""
        n_eval[0] += 1
        S_work[:, :4] = Y4
        return deriv(S_work, znodes, field)[:, :4]

    def residual(v):
        Y4 = v.reshape(q, 4)
        R = Y4 - S0[None, :4] - dz * (A @ rates(Y4))
        return (R / SCALE[None, :]).ravel()

    guess = straight_line_nodes(S0, znodes, z0)
    options = dict(maxfev=maxfev) if maxfev else {}
    sol = root(residual, guess.ravel(), method="hybr", tol=STEP_TOL,
               options=options)
    used = n_eval[0]                       # the solver's own cost, not the check
    Y4 = sol.x.reshape(q, 4)
    res = float(np.abs(residual(sol.x)).max())

    stages = np.empty((q, 5))
    stages[:, :4] = Y4
    stages[:, 4] = qop
    S1 = S0.copy()
    S1[:4] = S0[:4] + dz * (b @ deriv(stages, znodes, field)[:, :4])
    return stages, S1, res < tol, used, res


def straight_line_end(S0, dz):
    """The end state with the magnet switched off: the null step to beat."""
    S1 = np.asarray(S0, dtype=np.float64).copy()
    S1[0] = S0[0] + S0[2] * dz
    S1[1] = S0[1] + S0[3] * dz
    return S1


def field_for(which="up"):
    """The map for one polarity. Named here so every script in this folder
    reads the polarity from one place; the dataset's own `field` array says
    which one its labels were built with."""
    return make_field(which)
