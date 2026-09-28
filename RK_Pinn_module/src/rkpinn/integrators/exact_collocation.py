"""The exact collocation scheme: the stage equations solved with a root finder.

For a step from a state S_0 on the plane z_0 to the plane z_0 + dz, the
Gauss-Legendre method with q stages defines q stage states through

    S_j   = S_0 + dz * sum_k A_jk f(S_k, z_0 + c_k dz),      j = 1 .. q      (1)
    S_end = S_0 + dz * sum_k b_k  f(S_k, z_0 + c_k dz).                      (2)

Equation (1) is implicit: the unknown stage states are on both sides. This
module solves it directly, with no network. The answer is what a network with
a loss of exactly zero would produce, so its distance from the reference is
the ceiling for that number of steps and stages.

How the solve is set up:

- q/p is conserved, so there are 4 unknowns per stage, not 5;
- the residual of (1) is divided by (1, 1, 1e-3, 1e-3), so the solver sees
  millimetres and thousandths of a slope, which conditions the mixed units;
- the first guess of the root finder is the input state carried along its own
  slopes. This is a starting point for the solver only. It is not part of the
  answer and is not an output form of any network;
- convergence is judged on the residual, not on the flag of the root finder,
  which reports failure when asked for a smaller step than it can deliver
  although the equations already hold to rounding.

Ported, with the arithmetic unchanged and in the same order:

    solve_one_step     from  C2_Exact_scheme_table/exact_solver.py  (solve_state)
    solve_whole_track  from  E2_Comparators/exact_chain.py          (the loop of main)

Gates: tests/test_exact_collocation_matches_the_frozen_code.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from rkpinn.integrators.gauss_legendre_tableau import GaussLegendreTableau
from rkpinn.registry import register

# The residual is formed in (mm, mm, slope, slope) and divided by this.
SCALE_OF_THE_RESIDUAL = np.array([1.0, 1.0, 1e-3, 1e-3])

# A solve has converged when every component of the scaled residual is below this.
TOLERANCE_OF_THE_RESIDUAL = 1e-9

# Handed to the root finder as its step tolerance. It is far below the
# tolerance of the residual on purpose: the solver keeps going while it can
# make progress, and convergence is decided afterwards on the residual.
STEP_TOLERANCE_OF_THE_ROOT_FINDER = 1e-12


@dataclass
class ExactStep:
    """The exact scheme's answer for one state and one step."""

    stage_states: np.ndarray        # shape (q, 5), q/p carried through
    end_state: np.ndarray           # shape (5,), on the plane z_0 + dz
    converged: bool                 # largest residual below the tolerance
    residual_evaluations: int       # what the root finder used; its cost
    largest_residual: float         # in millimetres and thousandths of a slope


@dataclass
class ExactTrack:
    """The exact scheme chained over a whole track, for many tracks."""

    planes_mm: np.ndarray               # shape (N + 1,)
    states: np.ndarray                  # shape (n, N + 1, 5): the input and every end state
    stage_states: np.ndarray            # shape (n, N, q, 5)
    solves_not_converged: int
    residual_evaluations: int


def first_guess_of_the_stage_states(state, stage_planes_mm, start_plane_mm):
    """The starting point of the root finder: the input state carried along
    its own slopes to the stage planes. Shape (q, 4)."""
    distance = stage_planes_mm - start_plane_mm
    return np.column_stack([
        state[0] + state[2] * distance,
        state[1] + state[3] * distance,
        np.full(len(stage_planes_mm), state[2]),
        np.full(len(stage_planes_mm), state[3]),
    ])


@register("integrator", "exact_collocation")
def solve_one_step(equation_of_motion, state, start_plane_mm, step_length_mm,
                   tableau: GaussLegendreTableau,
                   tolerance_of_the_residual=TOLERANCE_OF_THE_RESIDUAL,
                   most_residual_evaluations=0) -> ExactStep:
    """Solve the stage equations for one state and one step.

    equation_of_motion         gives rates(state, z) for states of shape (n, 5)
    state                      shape (5,): x, y, tx, ty, q/p
    start_plane_mm             the plane the step starts on
    step_length_mm             negative integrates backwards
    tableau                    the Gauss-Legendre tableau
    most_residual_evaluations  0 leaves the root finder's own limit
    """
    c, A, b = tableau.nodes, tableau.stage_matrix, tableau.weights
    q = len(c)
    S0, z0, dz = state, start_plane_mm, step_length_mm
    znodes = z0 + c * dz
    qop = S0[4]
    S_work = np.empty((q, 5))
    S_work[:, 4] = qop

    n_eval = [0]

    def rates(Y4):
        n_eval[0] += 1
        S_work[:, :4] = Y4
        return equation_of_motion.rates(S_work, znodes)[:, :4]

    def residual(v):
        Y4 = v.reshape(q, 4)
        R = Y4 - S0[None, :4] - dz * (A @ rates(Y4))
        return (R / SCALE_OF_THE_RESIDUAL[None, :]).ravel()

    guess = first_guess_of_the_stage_states(S0, znodes, z0)
    options = dict(maxfev=most_residual_evaluations) if most_residual_evaluations else {}
    sol = root(residual, guess.ravel(), method="hybr",
               tol=STEP_TOLERANCE_OF_THE_ROOT_FINDER, options=options)
    used = n_eval[0]                       # the solver's own cost, not the check
    Y4 = sol.x.reshape(q, 4)
    res = float(np.abs(residual(sol.x)).max())

    stages = np.empty((q, 5))
    stages[:, :4] = Y4
    stages[:, 4] = qop
    S1 = S0.copy()
    S1[:4] = S0[:4] + dz * (b @ equation_of_motion.rates(stages, znodes)[:, :4])
    return ExactStep(stage_states=stages, end_state=S1,
                     converged=bool(res < tolerance_of_the_residual),
                     residual_evaluations=used, largest_residual=res)


def solve_whole_track(equation_of_motion, states, first_plane_mm, length_mm,
                      number_of_steps, tableau: GaussLegendreTableau) -> ExactTrack:
    """The exact step applied `number_of_steps` times, each end state starting
    the next step. This is the same chaining a network goes through.

    states  shape (n, 5): the states on the first plane
    """
    Z0, L, N = first_plane_mm, length_mm, number_of_steps
    dz = L / N
    S = np.asarray(states, dtype=np.float64)
    n = len(S)
    q = tableau.number_of_stages
    all_states = np.empty((n, N + 1, 5))
    all_stages = np.empty((n, N, q, 5))
    all_states[:, 0] = S
    cur, n_fail, n_eval = S.copy(), 0, 0
    for k in range(N):
        nxt = np.empty_like(cur)
        for i in range(n):
            step = solve_one_step(equation_of_motion, cur[i], Z0 + k * dz, dz, tableau)
            nxt[i] = step.end_state
            all_stages[i, k] = step.stage_states
            n_fail += (not step.converged)
            n_eval += step.residual_evaluations
        cur = nxt
        all_states[:, k + 1] = cur
    planes = Z0 + np.arange(N + 1) * dz
    return ExactTrack(planes_mm=planes, states=all_states, stage_states=all_stages,
                      solves_not_converged=int(n_fail), residual_evaluations=int(n_eval))
