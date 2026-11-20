"""Gate: the label-free path network and the stage residual of a path, before
any training.

Proves
  * the network emits N x q stage states, and fills a whole track with the
    layout set, the end states summed and the rates kept;
  * its plane states are the explicit sum s_{n+1} = s_n + h sum_i b_i F_ni;
  * with the last layer zero, the straight-line form gives the straight line
    from the start state to every stage plane;
  * the loss is zero to rounding on the exact scheme chained along a track,
    and on its separate steps (the check before training), and the largest
    residual it sees is the one the Newton sweep reports for the same track;
  * the gradient of the loss matches finite differences on a small network;
  * detaching the plane states keeps the value and changes the gradient;
  * the causal weights fall along the track, and eps gives the asked weight
    to the last step at round one;
  * the centre of the sliding window moves when blocks settle;
  * the weights are held through a restart, and the uniform weights do not
    change with start_a_restart.
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
from rkpinn.losses.stage_residual_of_a_path import (
    StageResidualOfAPath, centre_of_the_window, residual_of_a_path, sliding_window_weights)
from rkpinn.networks.label_free_path_network import LabelFreePathNetwork
from rkpinn.networks.output_forms import StraightLinePlusScaledCorrection
from rkpinn.predicted_track.predicted_track import (
    END_STATE_SUMMED, steps_of_the_exact_scheme, track_of_the_exact_scheme)
from rkpinn.predicted_track.track_layout import TrackLayout
from rkpinn.track_data.load_tracks import load_tracks
from rkpinn.training.round_trainer import LARGEST_LOSS_AT_THE_EXACT_SOLUTION
from the_store import KEY_OF_THE_TRACKS, the_store

STEPS, STAGES = 8, 2


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


def states(tracks, number=7):
    return torch.as_tensor(tracks.test.start_state[:number])


def loss(weights="uniform", through=True, **parameters):
    return StageResidualOfAPath(equation_of_motion=equations()[1], weights=weights,
                                gradient_through_plane_states=through, **parameters)


def constants_of(the_loss, tracks, net=None):
    first = SimpleNamespace(states=tracks.training.start_state[:64],
                            start_planes_mm=np.full(64, tracks.first_plane_mm))
    return the_loss.constants_with_the_network(first, net)


def test_shapes_and_the_track_it_fills(tracks):
    net = network(tracks)
    s = states(tracks)
    emitted = net(s)
    assert emitted.shape == (7, STEPS, STAGES, 4)
    assert net.body[-1].out_features == STEPS * STAGES * 4
    track = net.predict(s)
    assert track.layout is not None and track.end_state_was == END_STATE_SUMMED
    assert torch.equal(track.stage_states, emitted)
    assert track.end_states.shape == (7, STEPS, 4)
    assert track.stage_rates.shape == (7, STEPS, STAGES, 4)
    assert torch.equal(track.input_states[:, 0], s)
    assert torch.equal(track.input_states[:, 1:, :4], track.end_states[:, :-1])
    assert (track.input_states[:, :, 4] == s[:, 4:5]).all()
    whole = net.whole_track(s.numpy(), layout_of(tracks))
    assert isinstance(whole.stage_rates, np.ndarray)
    assert np.array_equal(whole.end_states, track.end_states.detach().numpy())
    assert "_equation_holder" not in "".join(net.state_dict())
    assert all(p.dtype == torch.float64 for p in net.parameters())


def test_plane_states_are_the_explicit_sum(tracks):
    net = network(tracks)
    s = states(tracks)
    with torch.no_grad():
        track = net.predict(s)
    layout = layout_of(tracks)
    b = torch.as_tensor(layout.tableau.weights)
    h = layout.step_length_mm
    plane = s[:, :4].clone()
    for n in range(STEPS):
        z = torch.as_tensor(layout.stage_planes_mm(n))
        F = equations()[1].rates(track.stage_states[:, n], s[:, 4:5], z)
        assert torch.allclose(F, track.stage_rates[:, n], rtol=0, atol=1e-15)
        plane = plane + h * torch.einsum("i,bid->bd", b, F)
        assert torch.allclose(track.end_states[:, n], plane, rtol=1e-13, atol=1e-10)


def test_zeroed_last_layer_gives_the_straight_line(tracks):
    net = network(tracks)
    with torch.no_grad():
        net.body[-1].weight.zero_()
        net.body[-1].bias.zero_()
    s = states(tracks)
    emitted = net(s)
    layout = layout_of(tracks)
    planes = np.stack([layout.stage_planes_mm(n) for n in range(STEPS)])
    dz = torch.as_tensor(planes - layout.first_plane_mm)[None]
    assert torch.allclose(emitted[..., 0], s[:, 0, None, None] + s[:, 2, None, None] * dz,
                          rtol=0, atol=1e-9)
    assert torch.allclose(emitted[..., 1], s[:, 1, None, None] + s[:, 3, None, None] * dz,
                          rtol=0, atol=1e-9)
    assert (emitted[..., 2] == s[:, 2, None, None]).all()
    assert (emitted[..., 3] == s[:, 3, None, None]).all()


def test_zero_at_the_exact_scheme_and_as_the_newton_sweep_sees_it(tracks):
    the_loss = loss()
    constants = constants_of(the_loss, tracks)
    exact = track_of_the_exact_scheme(equations()[0], tracks.test.start_state[:16],
                                      layout_of(tracks))
    with torch.no_grad():
        assert float(the_loss.value(exact, None, constants)) < LARGEST_LOSS_AT_THE_EXACT_SOLUTION
        separate = steps_of_the_exact_scheme(
            equations()[0], tracks.training.start_state[:8],
            np.full(8, tracks.first_plane_mm), layout_of(tracks).step_length_mm,
            layout_of(tracks).tableau)
        assert float(the_loss.value(separate, None, constants)) \
            < LARGEST_LOSS_AT_THE_EXACT_SOLUTION
        # a track off the scheme: the largest residual is the Newton sweep's
        track = network(tracks).whole_track(tracks.test.start_state[:16])
        R = residual_of_a_path(track, equations()[1])
        sweep = one_sweep(track, equations()[1])
    assert float(R.abs().max()) > 1e-6
    assert float(R.abs().max()) == pytest.approx(sweep.largest_residual_before, rel=1e-9)


def test_gradient_matches_finite_differences(tracks):
    net = network(tracks, width=4)
    the_loss = loss()
    constants = constants_of(the_loss, tracks)
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


def test_detached_plane_states_keep_the_value_and_change_the_gradient(tracks):
    net = network(tracks)
    s = states(tracks)
    through, detached = loss(through=True), loss(through=False)
    values, gradients = [], []
    for the_loss in (through, detached):
        constants = constants_of(the_loss, tracks)
        net.zero_grad()
        value = the_loss.value(net.predict(s), None, constants)
        value.backward()
        values.append(float(value.detach()))
        gradients.append(torch.cat([p.grad.reshape(-1) for p in net.parameters()]))
    assert values[0] == values[1]
    assert not torch.allclose(gradients[0], gradients[1], rtol=1e-6, atol=0)


@pytest.mark.parametrize("w_last", (0.1, 0.001))
def test_causal_weights_fall_and_give_the_asked_last_weight(tracks, w_last):
    net = network(tracks)
    the_loss = loss("causal", weight_of_the_last_step_at_round_one=w_last)
    constants = constants_of(the_loss, tracks, net)
    assert constants["causal_eps"] > 0
    first = torch.as_tensor(tracks.training.start_state[:64])
    with torch.no_grad():
        per_step = the_loss.per_step(net.predict(first), constants)
        w = the_loss.weights_of_the_steps(per_step, constants)
    assert float(w[0]) == 1.0
    assert (w[1:] < w[:-1]).all()
    assert float(w[-1]) == pytest.approx(w_last, rel=1e-9)
    with pytest.raises(ValueError):
        the_loss.constants(tracks.training.start_state[:64])


def test_the_sliding_window_moves_when_blocks_settle():
    high, low = 1.0, 1e-6
    per_step = torch.full((16,), high, dtype=torch.float64)
    assert centre_of_the_window(per_step, 4, 1e-3) == 4
    per_step[:4] = low
    assert centre_of_the_window(per_step, 4, 1e-3) == 8
    per_step[8:12] = low                     # not leading: the window does not jump
    assert centre_of_the_window(per_step, 4, 1e-3) == 8
    per_step[4:8] = low
    assert centre_of_the_window(per_step, 4, 1e-3) == 16
    w = sliding_window_weights(per_step, 4, 2.0, 1e-3)
    assert float(w[16 - 1]) > 0.5 and (w[1:] <= w[:-1]).all()


def test_weights_are_held_through_a_restart(tracks):
    s = states(tracks)
    uniform = loss()
    constants = constants_of(uniform, tracks)
    net = network(tracks)
    with torch.no_grad():
        before = float(uniform.value(net.predict(s), None, constants))
        uniform.start_a_restart()
        after = float(uniform.value(net.predict(s), None, constants))
        assert before == after
        assert torch.equal(uniform.weights_of_the_steps(torch.rand(STEPS), constants),
                           torch.ones(STEPS, dtype=torch.float64))

    window = loss("sliding_window", block_of_steps=2, width_in_steps=1.0, settled_below=1e-3)
    constants = constants_of(window, tracks)
    small, large = torch.full((STEPS,), 1e-6, dtype=torch.float64), \
        torch.full((STEPS,), 1.0, dtype=torch.float64)
    fresh_small = window.weights_of_the_steps(small, constants)
    assert not torch.equal(fresh_small, window.weights_of_the_steps(large, constants))
    window.start_a_restart()
    held = window.weights_of_the_steps(large, constants)
    assert torch.equal(held, window.weights_of_the_steps(small, constants))
    assert len(window.weights_of_the_steps(torch.ones(3, dtype=torch.float64), constants)) == 3
    window.stop_holding()
    assert torch.equal(window.weights_of_the_steps(small, constants), fresh_small)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        loss("causal")
    with pytest.raises(ValueError):
        loss("uniform", block_of_steps=4)
    with pytest.raises(ValueError):
        loss("sliding_window", block_of_steps=4, width_in_steps=2.0)
    with pytest.raises(ValueError):
        loss("causal", weight_of_the_last_step_at_round_one=1.5)
    with pytest.raises(ValueError):
        loss("weighted")
