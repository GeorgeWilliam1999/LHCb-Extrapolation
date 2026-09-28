"""Gate: the exact collocation scheme of the package is that of the frozen code.

Proves
  * one step is identical, to the last bit, to the frozen `solve_state`:
    stage states, end state, residual and the number of evaluations;
  * the chain is identical, to the last bit, to the frozen chain run on this
    machine;
  * the stored exact states are reproduced to 1e-11 mm in position and 1e-15
    in slope. They are not reproduced to the last bit by the package or by the
    frozen code, because they were written on other processors;
  * every solve converges to a residual below 1e-9;
  * the stage states satisfy the stage equations, and the end state is the
    weighted sum of the rates at the stages.

The stored files hold 1,452 test tracks. Reproducing all of them at 256 steps
takes most of an hour, so the gate reproduces the first tracks of each file.
The number of tracks is stated beside each case.
"""
from __future__ import annotations

import numpy as np
import pytest

from frozen_code import (
    frozen_exact_solver, frozen_module, frozen_tracks, identical, stored_exact_states)
from rkpinn.equation_of_motion.field_map import load_field_map
from rkpinn.equation_of_motion.lhcb import LhcbEquationOfMotion
from rkpinn.integrators import exact_collocation
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau

# (number of steps, number of stages, number of tracks reproduced)
STORED_CASES = (
    (2, 2, 40), (2, 4, 40), (2, 8, 40), (2, 16, 40),
    (256, 2, 3), (256, 4, 2), (256, 8, 1), (256, 16, 1),
)


def equation_of_the_tracks():
    polarity = str(frozen_tracks()["field"])
    return LhcbEquationOfMotion(load_field_map("v8r1_" + polarity)), polarity


def test_constants_equal_the_frozen_code():
    frozen = frozen_exact_solver()
    assert identical(exact_collocation.SCALE_OF_THE_RESIDUAL, frozen.SCALE)
    assert exact_collocation.TOLERANCE_OF_THE_RESIDUAL == frozen.RESIDUAL_TOL
    assert exact_collocation.STEP_TOLERANCE_OF_THE_ROOT_FINDER == frozen.STEP_TOL


@pytest.mark.parametrize("number_of_stages", (2, 4, 8, 16))
@pytest.mark.parametrize("step_length_mm", (20.2265625, 80.90625, 2588.9, -80.90625))
def test_one_step_is_identical_to_the_frozen_solver(number_of_stages, step_length_mm):
    frozen_solver = frozen_exact_solver()
    frozen_reference = frozen_module("reference")
    equation, polarity = equation_of_the_tracks()
    tracks = frozen_tracks()
    states = tracks["test_truth"][:12, 96]          # inside the magnet
    start_plane = float(tracks["planes"][96])
    ours_tableau = gauss_legendre_tableau(number_of_stages)
    their_tableau = frozen_reference.gauss_legendre(number_of_stages)
    their_field = frozen_reference.make_field(polarity)
    for state in states:
        ours = exact_collocation.solve_one_step(
            equation, state.copy(), start_plane, step_length_mm, ours_tableau)
        stages, end_state, converged, used, residual = frozen_solver.solve_state(
            state.copy(), start_plane, step_length_mm, their_tableau, their_field)
        assert identical(ours.stage_states, stages)
        assert identical(ours.end_state, end_state)
        assert ours.converged == bool(converged) and ours.converged
        assert ours.residual_evaluations == used
        assert ours.largest_residual == residual


def frozen_chain_run_now(states, first_plane_mm, length_mm, number_of_steps,
                         number_of_stages, polarity):
    """The loop of the frozen `exact_chain.py`, with the frozen solver, run here."""
    frozen_solver = frozen_exact_solver()
    frozen_reference = frozen_module("reference")
    tableau = frozen_reference.gauss_legendre(number_of_stages)
    field = frozen_reference.make_field(polarity)
    step_length = length_mm / number_of_steps
    current = states.copy()
    kept = [current.copy()]
    for k in range(number_of_steps):
        following = np.empty_like(current)
        for i in range(len(current)):
            following[i] = frozen_solver.solve_state(
                current[i], first_plane_mm + k * step_length, step_length, tableau, field)[1]
        current = following
        kept.append(current.copy())
    return np.stack(kept, axis=1)


@pytest.mark.parametrize("number_of_steps, number_of_stages, number_of_tracks", STORED_CASES)
def test_chain_is_identical_to_the_frozen_chain_run_here(number_of_steps, number_of_stages,
                                                         number_of_tracks):
    equation, polarity = equation_of_the_tracks()
    tracks = frozen_tracks()
    states = tracks["test_S0"][:number_of_tracks]
    first_plane, length = float(tracks["z0"]), float(tracks["L"])
    ours = exact_collocation.solve_whole_track(
        equation, states, first_plane, length, number_of_steps,
        gauss_legendre_tableau(number_of_stages))
    theirs = frozen_chain_run_now(
        states, first_plane, length, number_of_steps, number_of_stages, polarity)
    assert ours.solves_not_converged == 0
    assert identical(ours.states, theirs)
    assert ours.planes_mm[0] == first_plane
    assert abs(ours.planes_mm[-1] - (first_plane + length)) < 1e-9
    assert ours.stage_states.shape == (number_of_tracks, number_of_steps, number_of_stages, 5)


# The stored files were written on farm nodes, whose processors differ from
# this machine's. The root finder then rounds differently, and the stored
# states differ from a run here in the last digits: by at most 4.5e-13 mm in
# position and 5.6e-17 in slope on the tracks of this gate (measured
# 2026-09-28). The frozen code run here differs from its own stored files by
# exactly the same amounts. So the stored files are reproduced to a tolerance,
# and the bit for bit comparison is the one above.
STORED_POSITION_TOLERANCE_MM = 1e-11
STORED_SLOPE_TOLERANCE = 1e-15


@pytest.mark.parametrize("number_of_steps, number_of_stages, number_of_tracks", STORED_CASES)
def test_stored_exact_states_are_reproduced(number_of_steps, number_of_stages,
                                            number_of_tracks):
    equation, _ = equation_of_the_tracks()
    tracks = frozen_tracks()
    stored = stored_exact_states(number_of_steps, number_of_stages)
    assert stored.shape == (len(tracks["test_S0"]), number_of_steps + 1, 5)
    ours = exact_collocation.solve_whole_track(
        equation, tracks["test_S0"][:number_of_tracks], float(tracks["z0"]),
        float(tracks["L"]), number_of_steps, gauss_legendre_tableau(number_of_stages))
    assert ours.solves_not_converged == 0
    difference = np.abs(ours.states - stored[:number_of_tracks])
    assert difference[..., :2].max() < STORED_POSITION_TOLERANCE_MM
    assert difference[..., 2:4].max() < STORED_SLOPE_TOLERANCE
    assert identical(ours.states[..., 4], stored[:number_of_tracks][..., 4])


def test_stages_satisfy_the_stage_equations_and_the_end_is_their_sum():
    equation, _ = equation_of_the_tracks()
    tracks = frozen_tracks()
    tableau = gauss_legendre_tableau(4)
    start_plane, step_length = float(tracks["planes"][96]), 80.90625
    stage_planes = start_plane + tableau.nodes * step_length
    for state in tracks["test_truth"][:12, 96]:
        step = exact_collocation.solve_one_step(
            equation, state.copy(), start_plane, step_length, tableau)
        rates = equation.rates(step.stage_states, stage_planes)[:, :4]
        residual = (step.stage_states[:, :4] - state[None, :4]
                    - step_length * (tableau.stage_matrix @ rates))
        scaled = residual / exact_collocation.SCALE_OF_THE_RESIDUAL[None, :]
        assert np.abs(scaled).max() < exact_collocation.TOLERANCE_OF_THE_RESIDUAL
        summed = state[:4] + step_length * (tableau.weights @ rates)
        assert identical(step.end_state[:4], summed)
        assert step.end_state[4] == state[4]
        assert (step.stage_states[:, 4] == state[4]).all()
