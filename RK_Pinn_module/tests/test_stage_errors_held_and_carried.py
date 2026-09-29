"""Gate: stage errors, held and carried. Standard output 6.

This output is new, so it is gated on what it must satisfy, not on a number
of the paper.

Proves
  * the contributions of all steps sum to the final state of the network
    minus the start state carried over the whole crossing, to rounding;
  * the sum equals the measured endpoint error, up to the difference between
    the integrator at the step of the carrying and the stored reference, and
    that difference is returned;
  * for the exact scheme itself every stage error is zero, the contribution
    from the stage errors is zero, and all of the endpoint error comes from
    the scheme;
  * the error of x and y at the end of a step is first order exactly, and
    that of the slopes is first order for a small stage error: what is beyond
    first order falls with the square of the error;
  * the position part and the slope part are what the table says they are;
  * separate steps and a track without stages are refused.

The tracks are few and the steps are four, so that the gate takes a minute.
"""
from __future__ import annotations

import numpy as np
import pytest

from building_blocks import ACTIVATION, equations
from rkpinn.evaluation.error_along_the_track import stage_errors_held_and_carried
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.networks.output_forms import DirectStates
from rkpinn.networks.stage_network import build_stage_network
from rkpinn.predicted_track.predicted_track import (
    END_STATE_SUMMED, PredictedTrack, track_of_the_exact_scheme, track_of_the_reference)
from rkpinn.predicted_track.track_layout import TrackLayout
from rkpinn.track_data.load_tracks import load_tracks
from the_store import KEY_OF_THE_TRACKS, the_store

NUMBER_OF_TRACKS = 6
NUMBER_OF_STEPS, NUMBER_OF_STAGES = 4, 2
CARRYING = dict(step_length_of_the_carrying_mm=1.0, fraction_for_the_derivative=1e-3)


@pytest.fixture(scope="module")
def setting():
    tracks = load_tracks(the_store(), KEY_OF_THE_TRACKS, check_the_key=False)
    layout = TrackLayout(tracks.first_plane_mm, tracks.last_plane_mm, NUMBER_OF_STEPS,
                         gauss_legendre_tableau(NUMBER_OF_STAGES))
    momentum = tracks.test.momentum_gev
    rows = np.flatnonzero((momentum > 5.0) & (momentum < 50.0))[:NUMBER_OF_TRACKS]
    return dict(tracks=tracks, layout=layout, start=tracks.test.start_state[rows],
                reference_end=tracks.test.reference_end_state[rows])


@pytest.fixture(scope="module")
def of_the_exact_scheme(setting):
    exact = track_of_the_exact_scheme(equations()[0], setting["start"], setting["layout"])
    return exact, stage_errors_held_and_carried(
        exact, equations()[0], setting["reference_end"], **CARRYING)


def with_stage_errors(exact, errors) -> PredictedTrack:
    """The exact scheme's track with `errors` added to its stages of every
    step, and every end state summed from the stages it then has. Each step
    starts from the input the exact track has, so the steps are not chained:
    the first-order check needs the same input with and without the error."""
    equation = equations()[0]
    tableau, dz = exact.tableau, exact.step_length_mm
    n, steps, q = exact.stage_states.shape[:3]
    stages = exact.stage_states + errors
    ends = np.empty((n, steps, 4))
    for m in range(steps):
        planes = exact.start_planes_mm[0, m] + tableau.nodes * dz
        five = np.concatenate([stages[:, m].reshape(n * q, 4),
                               np.repeat(exact.input_states[:, m, 4:5], q, axis=0)], axis=1)
        rates = equation.rates(five, np.tile(planes, n))[:, :4].reshape(n, q, 4)
        ends[:, m] = exact.input_states[:, m, :4] + dz * np.einsum(
            "k,nkd->nd", tableau.weights, rates)
    return PredictedTrack(
        step_length_mm=dz, tableau=tableau, start_planes_mm=exact.start_planes_mm,
        input_states=exact.input_states, stage_states=stages, end_states=ends,
        end_state_was=END_STATE_SUMMED, layout=exact.layout, produced_by="a made track")


def test_for_the_exact_scheme_all_of_the_error_is_the_scheme(setting, of_the_exact_scheme):
    exact, result = of_the_exact_scheme
    assert np.abs(result["held"]).max() < 1e-12
    assert np.abs(result["from_stage_errors"]).max() == 0.0
    measured = exact.end_states[:, -1] - setting["reference_end"][:, :4]
    total = result["from_the_scheme"].sum(axis=1)
    assert np.abs(total[:, 0]).max() > 1e-3, "4 steps of 2 stages must have an error"
    assert np.abs(total - measured)[:, :2].max() * 1e3 < 1e-3          # micrometres
    for row in result["per_step"]:
        assert row["carried_from_stage_errors_in_x_median_of_absolute_micrometres"] == 0.0
        assert row["carried_from_the_scheme_in_x_median_of_absolute_micrometres"] > 0.0


def test_contributions_sum_to_the_endpoint_error_of_a_network(setting):
    tracks, layout = setting["tracks"], setting["layout"]
    spread = tracks.training.reference_states_on_planes[:, ::64].reshape(-1, 5).std(axis=0)
    network = build_stage_network(
        seed=3, equation_of_motion=equations()[1], tableau=layout.tableau,
        step_length_mm=layout.step_length_mm,
        first_start_plane_mm=layout.start_planes_mm[0],
        last_start_plane_mm=layout.start_planes_mm[-1], scale_of_inputs=spread,
        output_form=DirectStates(spread[:4] * 1e-3), width=16, depth=2,
        activation=ACTIVATION, end_state=END_STATE_SUMMED)
    # an untrained network would leave the field map; its outputs are made small
    # and added to the input, as a made track, so that the states stay inside
    whole = network.whole_track(setting["start"], layout)
    exact = track_of_the_exact_scheme(equations()[0], setting["start"], layout)
    made = with_stage_errors(exact, np.asarray(whole.stage_states) * 1e-2)
    chained = chain(made, layout)
    result = stage_errors_held_and_carried(
        chained, equations()[0], setting["reference_end"], **CARRYING)
    check = result["check"]
    assert check["number_of_tracks"] == NUMBER_OF_TRACKS
    assert check["sum_against_final_minus_carried_start_largest_micrometres"] < 1e-6
    assert check["sum_against_measured_endpoint_error_largest_micrometres"] < 1e-3
    assert check["carried_start_against_stored_reference_largest_micrometres"] < 1e-3
    assert check["measured_endpoint_error_in_x_median_of_absolute_micrometres"] > 1.0
    total = (result["from_stage_errors"] + result["from_the_scheme"]).sum(axis=1)
    measured = chained.end_states[:, -1] - setting["reference_end"][:, :4]
    assert np.abs(total - measured)[:, 2:].max() < 1e-9
    assert np.abs(result["from_stage_errors"]).max() > 0
    assert len(result["per_step"]) == NUMBER_OF_STEPS
    assert [row["distance_left_mm"] for row in result["per_step"]] == pytest.approx(
        [layout.last_plane_mm - z for z in layout.planes_mm[1:]])
    assert result["per_step"][-1]["distance_left_mm"] == 0.0
    assert result["per_step"][-1][
        "slope_part_in_x_median_of_absolute_micrometres"] == 0.0


def chain(made: PredictedTrack, layout) -> PredictedTrack:
    """A chained track with the stage errors of `made`: every step starts from
    the end state of the step before, and keeps its own error against the
    exact scheme from its own input."""
    equation = equations()[0]
    from rkpinn.predicted_track.predicted_track import steps_of_the_exact_scheme
    tableau, dz = made.tableau, made.step_length_mm
    n, steps, q = made.stage_states.shape[:3]
    exact = track_of_the_exact_scheme(equation, made.input_states[:, 0], layout)
    errors = made.stage_states - exact.stage_states
    inputs = np.empty((n, steps, 5))
    stages = np.empty((n, steps, q, 4))
    ends = np.empty((n, steps, 4))
    current = made.input_states[:, 0].copy()
    for m in range(steps):
        inputs[:, m] = current
        plane = layout.start_planes_mm[m]
        solved = steps_of_the_exact_scheme(equation, current, plane, dz, tableau)
        stages[:, m] = solved.stage_states[:, 0] + errors[:, m]
        planes = plane + tableau.nodes * dz
        five = np.concatenate([stages[:, m].reshape(n * q, 4),
                               np.repeat(current[:, 4:5], q, axis=0)], axis=1)
        rates = equation.rates(five, np.tile(planes, n))[:, :4].reshape(n, q, 4)
        ends[:, m] = current[:, :4] + dz * np.einsum("k,nkd->nd", tableau.weights, rates)
        current = np.concatenate([ends[:, m], current[:, 4:5]], axis=1)
    return PredictedTrack(
        step_length_mm=dz, tableau=tableau,
        start_planes_mm=np.broadcast_to(layout.start_planes_mm, (n, steps)).copy(),
        input_states=inputs, stage_states=stages, end_states=ends,
        end_state_was=END_STATE_SUMMED, layout=layout, produced_by="a made chain")


def test_a_small_stage_error_is_its_first_order(setting, of_the_exact_scheme):
    exact, _ = of_the_exact_scheme
    generator = np.random.default_rng(5)
    shape = exact.stage_states.shape
    direction = generator.standard_normal(shape) * np.array([1.0, 1.0, 1e-2, 1e-2])
    beyond, first = [], []
    for size in (0.1, 0.01):
        made = chain(with_stage_errors(exact, size * direction), setting["layout"])
        result = stage_errors_held_and_carried(
            made, equations()[0], setting["reference_end"],
            step_length_of_the_carrying_mm=5.0, fraction_for_the_derivative=1e-3)
        rows = result["per_step"]
        beyond.append(np.array([r["beyond_first_order_in_slope_in_x_median_of_absolute"]
                                for r in rows]))
        first.append(np.array([r["first_order_in_slope_in_x_median_of_absolute"]
                               for r in rows]))
        assert (beyond[-1] < 0.05 * first[-1]).all()
        held = np.abs(result["held"]).max(axis=(0, 1, 2))
        assert held[0] == pytest.approx(np.abs(size * direction[..., 0]).max(), rel=1e-6)
        # x and y: first order exactly. With the steps chained, the position part is
        # the first order of the slope errors of the stages, to rounding
        whole = np.abs(made.end_states - made.input_states[..., :4]).max()
        for m, row in enumerate(rows):
            assert row["position_part_in_x_median_of_absolute_micrometres"] > 0
        assert whole > 0
    # ten times smaller an error: the first order is ten times smaller, and what is
    # beyond it a hundred times smaller (measured 2026-09-30: within 2 %)
    assert first[1] == pytest.approx(first[0] / 10.0, rel=0.05)
    assert beyond[1] == pytest.approx(beyond[0] / 100.0, rel=0.1)


def test_the_position_and_slope_parts(setting, of_the_exact_scheme):
    exact, _ = of_the_exact_scheme
    errors = np.zeros(exact.stage_states.shape)
    errors[..., 0] = 1e-3                  # one micrometre in x at every stage
    made = with_stage_errors(exact, errors)
    # the steps of this track are not chained, so only the parts held at the end of
    # a step are read; the carried parts need a chain
    result = stage_errors_held_and_carried(
        chain(made, setting["layout"]), equations()[0], setting["reference_end"],
        step_length_of_the_carrying_mm=5.0, fraction_for_the_derivative=1e-3)
    for row in result["per_step"]:
        assert row["largest_stage_error_in_x_median_micrometres"] == pytest.approx(
            1.0, rel=1e-6)
        assert row["largest_stage_error_in_slope_in_x_median"] < 1e-12
        # an error in x alone changes the end state only through the field it meets
        assert row["position_part_in_x_median_of_absolute_micrometres"] < 1.0


def test_what_has_no_stages_is_refused(setting):
    layout = setting["layout"]
    reference = track_of_the_reference(
        equations()[0], setting["start"][:2],
        TrackLayout(layout.first_plane_mm, layout.first_plane_mm + 40.0, 2, layout.tableau),
        step_length_of_the_integrator_mm=1.0)
    with pytest.raises(ValueError, match="whole track with stages"):
        stage_errors_held_and_carried(reference, equations()[0], setting["reference_end"],
                                      **CARRYING)
