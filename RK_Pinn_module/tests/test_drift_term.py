"""Gate: the stage residual of a path with a drift term, before any training.

Proves
  * both drift terms, and the whole loss, are zero to rounding on the exact
    scheme chained along a track and on its separate steps (the check before
    training);
  * the x part of the local defect is h sum_i b_i r_tx exactly;
  * the endpoint sum, and the block sum at the scale of the whole track, are
    the carried-defect estimate of the endpoint error, computed here step by step;
  * the Newton endpoint term is the correction one sweep of
    evaluation/newton_sweep.py makes to the last plane;
  * the gradient of each loss matches finite differences on a small network;
  * the share fixes lambda so that the drift term is that share of the loss of
    the first round's states, and the share 0 gives the stage residual of a
    path to the last bit;
  * settings that make no sense are refused.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from building_blocks import equations
from rkpinn.evaluation.newton_sweep import one_sweep
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.losses.pooled import spread_of_components
from rkpinn.losses.stage_residual_of_a_path import StageResidualOfAPath, residual_of_a_path
from rkpinn.losses.stage_residual_of_a_path_with_a_drift_term import (
    StageResidualOfAPathWithADriftTerm, block_sums, endpoint_sum, local_defects,
    newton_endpoint_correction)
from rkpinn.networks.label_free_path_network import LabelFreePathNetwork
from rkpinn.networks.output_forms import StraightLinePlusScaledCorrection
from rkpinn.predicted_track.predicted_track import (
    steps_of_the_exact_scheme, track_of_the_exact_scheme)
from rkpinn.predicted_track.track_layout import TrackLayout
from rkpinn.track_data.load_tracks import load_tracks
from rkpinn.training.round_trainer import LARGEST_LOSS_AT_THE_EXACT_SOLUTION
from the_store import KEY_OF_THE_TRACKS, the_store

STEPS, STAGES = 8, 2
SCALES = [2, 4, 8]
ALL = ("endpoint_sum", "block_sums", "newton_endpoint")


@pytest.fixture(scope="module")
def tracks():
    return load_tracks(the_store(), KEY_OF_THE_TRACKS, check_the_key=False)


def layout_of(tracks):
    return TrackLayout(tracks.first_plane_mm, tracks.last_plane_mm, STEPS,
                       gauss_legendre_tableau(STAGES))


def network(tracks, seed=0, width=16):
    torch.manual_seed(seed)
    return LabelFreePathNetwork(
        equation_of_motion=equations()[1], layout=layout_of(tracks),
        scale_of_inputs=spread_of_components(tracks.training.start_state),
        output_form=StraightLinePlusScaledCorrection("per_track"),
        width=width, depth=2, activation="tanh")


def loss(drift, share=0.5):
    return StageResidualOfAPathWithADriftTerm(
        equation_of_motion=equations()[1], weights="uniform",
        gradient_through_plane_states=True, drift_term=drift,
        share_of_the_drift_term_at_round_one=share,
        block_scales_in_steps=SCALES if drift == "block_sums" else None)


def first_round(tracks):
    return SimpleNamespace(states=tracks.training.start_state[:64],
                           start_planes_mm=np.full(64, tracks.first_plane_mm))


def states(tracks, number=7):
    return torch.as_tensor(tracks.test.start_state[:number])


@pytest.mark.parametrize("drift", ALL)
def test_zero_at_the_exact_scheme(tracks, drift):
    the_loss = loss(drift)
    constants = the_loss.constants_with_the_network(first_round(tracks), network(tracks))
    exact = track_of_the_exact_scheme(equations()[0], tracks.test.start_state[:16],
                                      layout_of(tracks))
    separate = steps_of_the_exact_scheme(
        equations()[0], tracks.training.start_state[:8], np.full(8, tracks.first_plane_mm),
        layout_of(tracks).step_length_mm, layout_of(tracks).tableau)
    with torch.no_grad():
        for track in (exact, separate):
            assert float(the_loss.drift(track, constants)) < LARGEST_LOSS_AT_THE_EXACT_SOLUTION
            assert float(the_loss.value(track, None, constants)) \
                < LARGEST_LOSS_AT_THE_EXACT_SOLUTION
        # and far from zero off the scheme
        off = network(tracks).predict(states(tracks))
        assert float(the_loss.drift(off, constants)) > 1e-12


def test_block_sum_of_the_whole_track_is_the_carried_defect_estimate(tracks):
    track = network(tracks).whole_track(tracks.test.start_state[:16])
    layout = layout_of(tracks)
    h = layout.step_length_mm
    b = np.asarray(layout.tableau.weights)
    sigma = spread_of_components(tracks.training.start_state)[:4]
    with torch.no_grad():
        delta = local_defects(track, equations()[1]).numpy()
        r = residual_of_a_path(track, equations()[1]).numpy()
        whole = block_sums(track, equations()[1], sigma, [STEPS]).numpy()
    # the x and y parts of the defect are h sum_i b_i times the stage slope residual
    assert np.allclose(delta[..., 0], h * np.einsum("i,bni->bn", b, r[..., 2]),
                       rtol=1e-10, atol=1e-13)      # one rounding of slope - (slope - r)
    assert np.allclose(delta[..., 1], h * np.einsum("i,bni->bn", b, r[..., 3]),
                       rtol=1e-10, atol=1e-13)      # one rounding of slope - (slope - r)
    # carried step by step to the last plane
    sums = np.zeros((len(delta), 4))
    for n in range(STEPS):
        lever = layout.last_plane_mm - (layout.start_planes_mm[n] + h)
        sums[:, :2] += delta[:, n, :2] + delta[:, n, 2:] * lever
        sums[:, 2:] += delta[:, n, 2:]
    assert np.allclose(whole, ((sums / sigma) ** 2).mean(-1), rtol=1e-10, atol=0)
    with torch.no_grad():
        end = endpoint_sum(track, equations()[1], sigma).numpy()
    assert np.allclose(end, ((sums[:, :2] / sigma[:2]) ** 2).mean(-1), rtol=1e-10, atol=0)


def test_newton_term_is_the_correction_of_one_sweep(tracks):
    track = network(tracks).whole_track(tracks.test.start_state[:16])
    with torch.no_grad():
        correction = newton_endpoint_correction(track, equations()[1]).numpy()
    swept = one_sweep(track, equations()[1]).track
    moved = np.asarray(swept.end_states)[:, -1] - np.asarray(track.end_states)[:, -1]
    assert np.abs(moved).max() > 1e-6
    assert np.allclose(-correction, moved, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("drift", ALL)
def test_gradient_matches_finite_differences(tracks, drift):
    net = network(tracks, width=4)
    the_loss = loss(drift)
    constants = the_loss.constants_with_the_network(first_round(tracks), net)
    s = states(tracks, 5)
    parameter = net.body[-1].weight
    value = the_loss.value(net.predict(s), None, constants)
    gradient, = torch.autograd.grad(value, parameter)
    eps = 1e-5
    floor = 1e-6 * float(gradient.abs().max())
    for index in ((0, 0), (5, 1), (17, 3), (40, 2)):
        with torch.no_grad():
            parameter[index] += eps
            up = float(the_loss.value(net.predict(s), None, constants))
            parameter[index] -= 2 * eps
            down = float(the_loss.value(net.predict(s), None, constants))
            parameter[index] += eps
        assert (up - down) / (2 * eps) == pytest.approx(float(gradient[index]), rel=1e-4,
                                                        abs=floor)


@pytest.mark.parametrize("drift", ALL)
def test_the_share_fixes_lambda(tracks, drift):
    net = network(tracks)
    first = first_round(tracks)
    with torch.no_grad():
        track = net.predict(torch.as_tensor(first.states))
        the_loss = loss(drift, share=0.3)
        constants = the_loss.constants_with_the_network(first, net)
        residual = float(the_loss.residual_part(track, constants))
        drift_value = float(the_loss.drift(track, constants))
        whole = float(the_loss.value(track, None, constants))
        assert constants["weight_of_the_drift_term"] * drift_value / whole \
            == pytest.approx(0.3, rel=1e-9)
        assert constants["residual_at_round_one"] == pytest.approx(residual, rel=1e-12)
        none = loss(drift, share=0.0)
        constants_none = none.constants_with_the_network(first, net)
        plain = StageResidualOfAPath(equation_of_motion=equations()[1], weights="uniform",
                                     gradient_through_plane_states=True)
        plain_constants = plain.constants(first.states)
        assert constants_none["weight_of_the_drift_term"] == 0.0
        assert float(none.value(track, None, constants_none)) \
            == float(plain.value(track, None, plain_constants))
    with pytest.raises(ValueError):
        the_loss.constants(first.states)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        loss("blocks")
    with pytest.raises(ValueError):
        StageResidualOfAPathWithADriftTerm(
            equation_of_motion=equations()[1], weights="uniform",
            gradient_through_plane_states=True, drift_term="block_sums",
            share_of_the_drift_term_at_round_one=0.5, block_scales_in_steps=None)
    with pytest.raises(ValueError):
        StageResidualOfAPathWithADriftTerm(
            equation_of_motion=equations()[1], weights="uniform",
            gradient_through_plane_states=True, drift_term="newton_endpoint",
            share_of_the_drift_term_at_round_one=0.5, block_scales_in_steps=[4])
    with pytest.raises(ValueError):
        StageResidualOfAPathWithADriftTerm(
            equation_of_motion=equations()[1], weights="uniform",
            gradient_through_plane_states=True, drift_term="endpoint_sum",
            share_of_the_drift_term_at_round_one=0.5, block_scales_in_steps=[4])
    with pytest.raises(ValueError):
        loss("block_sums", share=1.0)


def test_a_block_that_does_not_divide_the_track_is_refused(tracks):
    track = network(tracks).whole_track(tracks.test.start_state[:4])
    with pytest.raises(ValueError):
        with torch.no_grad():
            block_sums(track, equations()[1], [1.0, 1.0, 1.0, 1.0], [3])
