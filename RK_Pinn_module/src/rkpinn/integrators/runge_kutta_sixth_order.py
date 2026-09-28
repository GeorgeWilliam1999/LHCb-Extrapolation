"""The sixth-order reference: Butcher's explicit method with seven stages.

Everything is scored against a numerical solution of the equation of motion.
The reference is this method in double precision at a fixed step of 0.1 mm.
It contains the field and nothing else: no scattering, no energy loss.

The coefficients are those of the frozen code, which copied them from the van
der Pol study. They are not derived again here. The gates check the order
conditions they must satisfy.

Ported from `_shared/reference.py` (`rk6_rows` and the RK6 arrays) with the
arithmetic unchanged and in the same order, so the end states are identical to
the last bit.
Gates: tests/test_sixth_order_reference_matches_the_frozen_code.py.
"""
from __future__ import annotations

import numpy as np

from rkpinn.registry import register

ORDER = 6
NUMBER_OF_STAGES = 7

# The step of the reference. It is a named value, not a default: every caller
# passes the step it uses, so the step a run used is in its configuration.
REFERENCE_STEP_LENGTH_MM = 0.1

NODES = np.array([0.0, 1 / 3, 2 / 3, 1 / 3, 5 / 6, 1 / 6, 1.0])

STAGE_MATRIX = np.zeros((7, 7))
STAGE_MATRIX[1, 0] = 1 / 3
STAGE_MATRIX[2, 0], STAGE_MATRIX[2, 1] = 0.0, 2 / 3
STAGE_MATRIX[3, 0], STAGE_MATRIX[3, 1], STAGE_MATRIX[3, 2] = 1 / 12, 1 / 3, -1 / 12
STAGE_MATRIX[4, 0], STAGE_MATRIX[4, 1], STAGE_MATRIX[4, 2], STAGE_MATRIX[4, 3] = \
    25 / 48, -55 / 24, 35 / 48, 15 / 8
STAGE_MATRIX[5, 0], STAGE_MATRIX[5, 1], STAGE_MATRIX[5, 2], STAGE_MATRIX[5, 3], \
    STAGE_MATRIX[5, 4] = 3 / 20, -11 / 24, -1 / 8, 1 / 2, 1 / 10
STAGE_MATRIX[6, 0], STAGE_MATRIX[6, 1], STAGE_MATRIX[6, 2] = -261 / 260, 33 / 13, 43 / 156
STAGE_MATRIX[6, 3], STAGE_MATRIX[6, 4], STAGE_MATRIX[6, 5] = -118 / 39, 32 / 195, 80 / 39

WEIGHTS = np.array([13 / 200, 0.0, 11 / 40, 11 / 40, 4 / 25, 4 / 25, 13 / 200])


@register("integrator", "runge_kutta_sixth_order")
def integrate_with_sixth_order(equation_of_motion, states, start_planes_mm,
                               end_planes_mm, *, step_length_mm):
    """Carry states from their start planes to their end planes.

    equation_of_motion  gives rates(state, z) for states of shape (n, 5)
    states              shape (n, 5)
    start_planes_mm     one plane, or one per row
    end_planes_mm       one plane, or one per row
    step_length_mm      the fixed step; REFERENCE_STEP_LENGTH_MM for the reference

    Every row marches in steps of `step_length_mm` towards its own end plane.
    The last step of each row is shortened so the row lands exactly on its end
    plane, and a row that has arrived takes no further step. An end plane
    below the start plane integrates backwards. A row whose two planes
    coincide is returned unchanged.
    """
    S = np.atleast_2d(np.asarray(states, dtype=np.float64)).copy()
    z = np.broadcast_to(np.asarray(start_planes_mm, dtype=np.float64),
                        (len(S),)).astype(np.float64).copy()
    z1 = np.broadcast_to(np.asarray(end_planes_mm, dtype=np.float64),
                         (len(S),)).astype(np.float64)
    step = abs(float(step_length_mm))
    sign = np.sign(z1 - z)
    sign[sign == 0] = 1.0
    k = np.empty((NUMBER_OF_STAGES, len(S), S.shape[1]), dtype=np.float64)
    remaining = (z1 - z) * sign
    # The march, plus a small allowance for the residue a shortened last step
    # can leave behind. It is a hard cap, so a row can never loop for ever.
    max_sweeps = int(np.ceil(remaining.max() / step)) + 8 if len(S) else 0
    for _ in range(max_sweeps):
        active = remaining > 1e-12
        if not active.any():
            break
        h = np.minimum(remaining[active], step) * sign[active]
        ha = h[:, None]
        Sa, za = S[active], z[active]
        ka = k[:, :len(Sa)]
        ka[0] = equation_of_motion.rates(Sa, za)
        for i in range(1, NUMBER_OF_STAGES):
            Y = Sa + ha * np.tensordot(STAGE_MATRIX[i, :i], ka[:i], axes=(0, 0))
            ka[i] = equation_of_motion.rates(Y, za + NODES[i] * h)
        S[active] = Sa + ha * np.tensordot(WEIGHTS, ka, axes=(0, 0))
        z[active] = za + h
        remaining = (z1 - z) * sign
    return S


def integrate_with_sixth_order_over_planes(equation_of_motion, states, planes_mm, *,
                                           step_length_mm):
    """Carry states from the first of `planes_mm` to each later plane in turn,
    keeping the state on every plane. Returns shape (n, number of planes, 5).

    Each leg starts from the state the leg before ended on.
    """
    planes = np.asarray(planes_mm, dtype=np.float64)
    S = np.atleast_2d(np.asarray(states, dtype=np.float64))
    kept = np.empty((len(S), len(planes), S.shape[1]))
    kept[:, 0] = S
    current = S
    for j in range(1, len(planes)):
        current = integrate_with_sixth_order(
            equation_of_motion, current, planes[j - 1], planes[j],
            step_length_mm=step_length_mm)
        kept[:, j] = current
    return kept
