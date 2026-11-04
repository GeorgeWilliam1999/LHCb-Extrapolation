"""Gate: the path network, before any training.

Proves
  * it emits every stage state and every plane state of the layout, shape
    (n, steps, stages + 1, 4), and fills a whole track with the layout set,
    each step's input being the previous step's emitted end state;
  * with `extra_outputs` 4 it emits a log-sigma per state and component, and
    with 0 none; any other number is refused;
  * the same seed gives the same weights twice;
  * with the last layer zero, the straight-line form gives the straight line
    from the start state to the plane of every output, and direct states give
    zero;
  * a start plane other than the first is refused, and `whole_track` is
    `predict` without a gradient, as numpy.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from building_blocks import equations
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.losses.pooled import spread_of_components
from rkpinn.networks.output_forms import DirectStates, StraightLinePlusScaledCorrection
from rkpinn.networks.path_network import PathNetwork
from rkpinn.predicted_track.track_layout import TrackLayout
from rkpinn.track_data.load_tracks import load_tracks
from the_store import KEY_OF_THE_TRACKS, the_store

STEPS, STAGES = 8, 2


@pytest.fixture(scope="module")
def tracks():
    return load_tracks(the_store(), KEY_OF_THE_TRACKS, check_the_key=False)


def layout_of(tracks):
    return TrackLayout(tracks.first_plane_mm, tracks.last_plane_mm, STEPS,
                       gauss_legendre_tableau(STAGES))


def network(tracks, form="straight_line", extra_outputs=0, seed=0, width=16):
    start = tracks.training.start_state
    output_form = (StraightLinePlusScaledCorrection("per_track") if form == "straight_line"
                   else DirectStates(spread_of_components(start)[:4]))
    torch.manual_seed(seed)
    return PathNetwork(equation_of_motion=equations()[1], layout=layout_of(tracks),
                       scale_of_inputs=spread_of_components(start), output_form=output_form,
                       width=width, depth=2, activation="tanh", extra_outputs=extra_outputs)


def states(tracks, number=7):
    return torch.as_tensor(tracks.test.start_state[:number])


def test_shapes_and_the_track_it_fills(tracks):
    net = network(tracks, extra_outputs=0)
    s = states(tracks)
    emitted, extra = net(s)
    assert emitted.shape == (7, STEPS, STAGES + 1, 4) and extra is None
    track = net.predict(s)
    assert track.layout is not None and track.layout.number_of_steps == STEPS
    assert track.stage_states.shape == (7, STEPS, STAGES, 4)
    assert track.end_states.shape == (7, STEPS, 4)
    assert torch.equal(track.stage_states, emitted[:, :, :STAGES])
    assert torch.equal(track.end_states, emitted[:, :, STAGES])
    assert torch.equal(track.input_states[:, 0], s)
    assert torch.equal(track.input_states[:, 1:, :4], track.end_states[:, :-1])
    assert (track.input_states[:, :, 4] == s[:, 4:5]).all()
    assert track.extra_outputs is None
    with_sigma = network(tracks, extra_outputs=4)
    emitted, extra = with_sigma(s)
    assert extra.shape == (7, STEPS, STAGES + 1, 4)
    assert with_sigma.predict(s).extra_outputs.shape == (7, STEPS, STAGES + 1, 4)
    with pytest.raises(ValueError):
        network(tracks, extra_outputs=3)


def test_the_same_seed_gives_the_same_weights(tracks):
    first, second = network(tracks, seed=3), network(tracks, seed=3)
    other = network(tracks, seed=4)
    for (name, a), (_, b), (_, c) in zip(first.state_dict().items(),
                                         second.state_dict().items(),
                                         other.state_dict().items()):
        assert torch.equal(a, b), name
    assert any(not torch.equal(a, c) for a, c in zip(first.parameters(), other.parameters()))


@pytest.mark.parametrize("form", ("straight_line", "direct_states"))
def test_zeroed_last_layer_gives_the_declared_answer(tracks, form):
    net = network(tracks, form=form)
    last = net.body[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    s = states(tracks)
    emitted, _ = net(s)
    if form == "direct_states":
        assert (emitted == 0).all()
        return
    layout = layout_of(tracks)
    planes = np.empty((STEPS, STAGES + 1))
    for n in range(STEPS):
        planes[n, :STAGES] = layout.stage_planes_mm(n)
        planes[n, STAGES] = layout.planes_mm[n + 1]
    dz = torch.as_tensor(planes - layout.first_plane_mm)[None]
    x = s[:, 0, None, None] + s[:, 2, None, None] * dz
    y = s[:, 1, None, None] + s[:, 3, None, None] * dz
    assert torch.allclose(emitted[..., 0], x, rtol=0, atol=1e-9)
    assert torch.allclose(emitted[..., 1], y, rtol=0, atol=1e-9)
    assert (emitted[..., 2] == s[:, 2, None, None]).all()
    assert (emitted[..., 3] == s[:, 3, None, None]).all()


def test_only_the_first_plane_and_whole_track(tracks):
    net = network(tracks)
    s = states(tracks)
    with pytest.raises(ValueError):
        net.predict(s, torch.full((len(s),), tracks.first_plane_mm + 1.0, dtype=torch.float64))
    with torch.no_grad():
        predicted = net.predict(s)
    whole = net.whole_track(s.numpy(), layout_of(tracks))
    assert isinstance(whole.end_states, np.ndarray)
    assert np.array_equal(whole.end_states, predicted.end_states.numpy())
    assert np.array_equal(whole.final_state(), predicted.final_state().numpy())
    assert all(p.dtype == torch.float64 for p in net.parameters())
