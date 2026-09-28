"""Gate: the sixth-order reference of the package is that of the frozen code.

Proves
  * the coefficients are identical to the frozen ones and satisfy the order
    conditions of a method of order six that are checked here;
  * the end state is identical, to the last bit, to the frozen `rk6_rows`
    over 40 mm and over 2,589 mm forwards and over 40 mm and 400 mm
    backwards, with one plane for all rows and with one plane per row;
  * the states of the frozen tracks are reproduced, to the last bit, on the
    first 33 of their 257 planes (all 257 were checked once by hand on
    2026-09-28 and were identical; the whole file is the gate of phase 2);
  * on a smooth field the fitted order is six;
  * an integration forwards and then back returns to the start.
"""
from __future__ import annotations

import numpy as np
import pytest

from frozen_code import frozen_module, frozen_tracks, identical
from rkpinn.equation_of_motion.field_map import load_field_map
from rkpinn.equation_of_motion.lhcb import LhcbEquationOfMotion
from rkpinn.integrators import runge_kutta_sixth_order as sixth_order

NUMBER_OF_STATES = 60


class SmoothField:
    """An analytic field with no kinks, for measuring the order."""

    def __call__(self, x, y, z):
        bell = np.exp(-((z - 5000.0) / 1500.0) ** 2)
        return (1e-5 * y * bell, -1.0 * bell * (1.0 + 1e-8 * x * x), 1e-5 * x * bell)


def equation_and_frozen_field():
    frozen = frozen_module("reference")
    polarity = str(frozen_tracks()["field"])
    equation = LhcbEquationOfMotion(load_field_map("v8r1_" + polarity))
    return equation, frozen, frozen.make_field(polarity)


def test_coefficients_are_identical_to_the_frozen_code():
    frozen = frozen_module("reference")
    assert identical(sixth_order.NODES, frozen.RK6_C)
    assert identical(sixth_order.STAGE_MATRIX, frozen.RK6_A)
    assert identical(sixth_order.WEIGHTS, frozen.RK6_B)
    assert sixth_order.REFERENCE_STEP_LENGTH_MM == frozen.RK6_STEP
    assert sixth_order.ORDER == frozen.RK6_ORDER
    assert sixth_order.NUMBER_OF_STAGES == frozen.RK6_STAGES


def test_coefficients_satisfy_the_conditions_checked():
    c, A, b = sixth_order.NODES, sixth_order.STAGE_MATRIX, sixth_order.WEIGHTS
    assert np.abs(np.triu(A)).max() == 0.0, "the method is explicit"
    assert np.abs(A.sum(axis=1) - c).max() < 1e-15
    # the weights integrate polynomials exactly to degree five
    for power in range(6):
        assert abs((b * c ** power).sum() - 1.0 / (power + 1)) < 1e-15
    # the conditions of the form b A c^m, to order six
    for power in range(1, 5):
        assert abs(b @ A @ c ** power - 1.0 / ((power + 1) * (power + 2))) < 1e-15
    assert abs(b @ A @ A @ c - 1.0 / 24) < 1e-15
    assert abs(b @ A @ A @ A @ c - 1.0 / 120) < 1e-15
    assert abs(b @ A @ A @ A @ A @ c - 1.0 / 720) < 1e-15


@pytest.mark.parametrize("distance_mm", (40.0, 2589.0, -40.0, -400.0))
def test_end_state_is_identical_to_the_frozen_code(distance_mm):
    equation, frozen, frozen_field = equation_and_frozen_field()
    tracks = frozen_tracks()
    plane = 128
    states = tracks["test_truth"][:NUMBER_OF_STATES, 0 if distance_mm > 0 else plane]
    start = float(tracks["planes"][0 if distance_mm > 0 else plane])
    ours = sixth_order.integrate_with_sixth_order(
        equation, states, start, start + distance_mm,
        step_length_mm=sixth_order.REFERENCE_STEP_LENGTH_MM)
    theirs = frozen.rk6_rows(states, start, start + distance_mm,
                             step=frozen.RK6_STEP, field=frozen_field)
    assert np.abs(ours[:, :4] - states[:, :4]).max() > 0
    assert identical(ours, theirs)


def test_end_state_with_one_plane_per_row_is_identical_to_the_frozen_code():
    equation, frozen, frozen_field = equation_and_frozen_field()
    tracks = frozen_tracks()
    generator = np.random.default_rng(3)
    which_plane = generator.integers(0, 200, NUMBER_OF_STATES)
    states = tracks["test_truth"][np.arange(NUMBER_OF_STATES), which_plane]
    start = tracks["planes"][which_plane]
    end = start + generator.uniform(-100.0, 300.0, NUMBER_OF_STATES)
    end[0] = start[0]                       # a row that does not move
    ours = sixth_order.integrate_with_sixth_order(
        equation, states, start, end, step_length_mm=sixth_order.REFERENCE_STEP_LENGTH_MM)
    theirs = frozen.rk6_rows(states, start, end, step=frozen.RK6_STEP, field=frozen_field)
    assert identical(ours, theirs)
    assert identical(ours[0], states[0])


def test_a_coarser_step_is_identical_to_the_frozen_code():
    equation, frozen, frozen_field = equation_and_frozen_field()
    tracks = frozen_tracks()
    states = tracks["test_truth"][:NUMBER_OF_STATES, 0]
    start, end = float(tracks["z0"]), float(tracks["z1"])
    ours = sixth_order.integrate_with_sixth_order(
        equation, states, start, end, step_length_mm=5.0)
    theirs = frozen.rk6_rows(states, start, end, step=5.0, field=frozen_field)
    assert identical(ours, theirs)


def test_states_of_the_frozen_tracks_are_reproduced_on_their_planes():
    equation, _, _ = equation_and_frozen_field()
    tracks = frozen_tracks()
    number_of_tracks, number_of_planes = 20, 33      # the first eighth of the crossing
    assert float(tracks["rk6_step_mm"]) == sixth_order.REFERENCE_STEP_LENGTH_MM
    ours = sixth_order.integrate_with_sixth_order_over_planes(
        equation, tracks["test_S0"][:number_of_tracks], tracks["planes"][:number_of_planes],
        step_length_mm=sixth_order.REFERENCE_STEP_LENGTH_MM)
    assert ours.shape == (number_of_tracks, number_of_planes, 5)
    assert identical(ours, tracks["test_truth"][:number_of_tracks, :number_of_planes])


def test_fitted_order_on_a_smooth_field_is_six():
    equation = LhcbEquationOfMotion(SmoothField())
    tracks = frozen_tracks()
    momentum = tracks["test_P"][:400]
    states = tracks["test_S0"][:400][(momentum > 5.0) & (momentum < 50.0)][:40]
    start, end = 3000.0, 7000.0
    finest = sixth_order.integrate_with_sixth_order(
        equation, states, start, end, step_length_mm=2.5)
    steps = np.array([800.0, 400.0, 200.0, 100.0])
    errors = np.array([
        np.median(np.abs(sixth_order.integrate_with_sixth_order(
            equation, states, start, end, step_length_mm=step)[:, 0] - finest[:, 0]))
        for step in steps])
    assert (errors > 1e-11).all(), "the errors must stand clear of rounding: %r" % errors
    order = np.polyfit(np.log(steps), np.log(errors), 1)[0]
    assert 5.5 < order < 7.0, (order, errors)


def test_forwards_then_back_returns_to_the_start():
    equation, _, _ = equation_and_frozen_field()
    tracks = frozen_tracks()
    states = tracks["test_truth"][:NUMBER_OF_STATES, 96]        # inside the magnet
    start = float(tracks["planes"][96])
    end = start + 500.0
    step = sixth_order.REFERENCE_STEP_LENGTH_MM
    there = sixth_order.integrate_with_sixth_order(
        equation, states, start, end, step_length_mm=step)
    back = sixth_order.integrate_with_sixth_order(
        equation, there, end, start, step_length_mm=step)
    assert np.abs(back[:, :2] - states[:, :2]).max() < 1e-6      # millimetres: one nanometre
    assert np.abs(back[:, 2:4] - states[:, 2:4]).max() < 1e-9
    assert identical(back[:, 4], states[:, 4])
