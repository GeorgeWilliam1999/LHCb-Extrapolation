"""Gate: the networks, before any training.

Proves
  * with a predicted end state, the stage network is identical, to the last
    bit, to the frozen one-step network built from the same seed and scales,
    in its outputs and in the gradient of its outputs;
  * the whole-crossing network is identical to the frozen one-step network
    with no stages;
  * with the last layer zero, a network returns what its output form declares;
  * one batch that mixes every start plane gives the outputs each plane's
    states give on their own: the start plane is an input per row;
  * the networks are in double precision whatever the default of torch is;
  * a summed end state is the input plus the step length times the weighted
    sum of the rates at the stages, and the network then has 4q outputs;
  * with the exact scheme's stages, the summed end state is the exact scheme's
    end state;
  * the self chain is the network applied step after step, each end state
    being the next input, and equals the frozen chaining;
  * the field map is not written into a snapshot of the weights, and a
    network built again from its settings and weights gives the same outputs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from building_blocks import CELLS, DEPTH, WIDTH, crossing, equations
from frozen_code import frozen_module, frozen_tracks, identical
from rkpinn.networks.collect_and_sum import collect_and_sum, rates_at_the_stages
from rkpinn.networks.output_forms import DirectStates
from rkpinn.networks.stage_network import StageNetwork, build_stage_network
from rkpinn.networks.whole_crossing_network import build_whole_crossing_network
from rkpinn.predicted_track.predicted_track import (
    END_STATE_PREDICTED, END_STATE_SUMMED, steps_of_the_exact_scheme)
from rkpinn.predicted_track.track_layout import TrackLayout

NUMBER_OF_STATES = 2000


def frozen_one_step_network(cell, seed, number_of_extra_inputs=1):
    frozen = frozen_module("model")
    torch.manual_seed(seed)
    return frozen.OneStepNetwork(cell.number_of_stages, cell.spread, cell.spread,
                                 width=WIDTH, depth=DEPTH, n_extra=number_of_extra_inputs)


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
@pytest.mark.parametrize("seed", (0, 3))
def test_stage_network_is_identical_to_the_frozen_network(number_of_steps,
                                                          number_of_stages, seed):
    cell = crossing(number_of_steps, number_of_stages)
    ours = cell.network(END_STATE_PREDICTED, seed=seed)
    theirs = frozen_one_step_network(cell, seed)
    states, planes = cell.states(NUMBER_OF_STATES)

    def outputs_and_gradient(emit):
        for parameter in emit.__self__.parameters():
            parameter.grad = None
        out = emit()
        weights = torch.linspace(0.5, 1.5, out.numel(), dtype=torch.float64).reshape(out.shape)
        (out * weights).sum().backward()
        gradient = torch.cat([p.grad.reshape(-1) for p in emit.__self__.parameters()])
        return out.detach().numpy(), gradient.numpy()

    class Ours:
        __self__ = ours
        def __call__(self):
            return ours(states, planes)

    class Theirs:
        __self__ = theirs
        def __call__(self):
            return theirs(states, ours.start_plane_as_input(planes)[:, None])

    out_ours, gradient_ours = outputs_and_gradient(Ours())
    out_theirs, gradient_theirs = outputs_and_gradient(Theirs())
    assert out_ours.shape == (NUMBER_OF_STATES, number_of_stages + 1, 4)
    assert identical(out_ours, out_theirs)
    assert identical(gradient_ours, gradient_theirs)


def test_whole_crossing_network_is_identical_to_the_frozen_network():
    cell = crossing(2, 2)
    tracks = frozen_tracks()
    states = torch.as_tensor(tracks["train_S0"][:NUMBER_OF_STATES])
    spread = tracks["train_S0"].std(axis=0)
    ours = build_whole_crossing_network(
        seed=5, first_plane_mm=cell.first_plane_mm, last_plane_mm=cell.last_plane_mm,
        scale_of_inputs=spread, output_form=DirectStates(spread[:4]),
        width=WIDTH, depth=DEPTH)
    frozen = frozen_module("model")
    torch.manual_seed(5)
    theirs = frozen.OneStepNetwork(0, spread, spread, width=WIDTH, depth=DEPTH, n_extra=0)
    with torch.no_grad():
        assert identical(ours(states).numpy(), theirs(states)[:, 0].numpy())
    track = ours.predict(states)
    assert track.number_of_stages == 0 and track.stage_states is None
    assert track.end_state_was == END_STATE_PREDICTED
    assert track.step_length_mm == cell.length_mm
    assert identical(track.final_state()[:, 4].detach().numpy(), states[:, 4].numpy())
    whole = ours.whole_track(states.numpy())
    assert identical(whole.end_states, track.end_states.detach().numpy())
    with pytest.raises(ValueError):
        ours.predict(states, torch.full((len(states),), 3000.0, dtype=torch.float64))


@pytest.mark.parametrize("end_state", (END_STATE_SUMMED, END_STATE_PREDICTED))
def test_zeroed_last_layer_gives_the_declared_answer(end_state):
    cell = crossing(64, 2)
    network = cell.network(end_state)
    with torch.no_grad():
        network.body[-1].weight.zero_()
        network.body[-1].bias.zero_()
        states, planes = cell.states(200)
        emitted = network(states, planes)
        declared = network.output_form.answer_with_zeroed_network(
            states, network.planes_of_outputs_mm(planes))
    assert identical(emitted.numpy(), declared.numpy())
    assert (emitted == 0).all()


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
def test_the_start_plane_is_an_input_per_row(number_of_steps, number_of_stages):
    cell = crossing(number_of_steps, number_of_stages)
    network = cell.network(END_STATE_SUMMED)
    states, planes = cell.states(600)
    with torch.no_grad():
        mixed = network.predict(states, planes)
        for plane in torch.unique(planes)[:6]:
            rows = planes == plane
            alone = network.predict(states[rows], planes[rows])
            for a, b in ((mixed.stage_states[rows], alone.stage_states),
                         (mixed.end_states[rows], alone.end_states)):
                scale = b.abs().amax(dim=tuple(range(b.dim() - 1))).clamp_min(1e-300)
                assert ((a - b).abs() / scale).max() < 1e-12


def test_networks_are_in_double_precision_whatever_the_default_is():
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        cell = crossing(64, 2)
        network = cell.network(END_STATE_SUMMED)
        states, planes = cell.states(50)
        track = network.predict(states, planes)
        assert all(p.dtype == torch.float64 for p in network.parameters())
        assert all(b.dtype == torch.float64 for b in network.buffers())
        assert track.stage_states.dtype == torch.float64
        assert track.end_states.dtype == torch.float64
    finally:
        torch.set_default_dtype(default)


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
def test_a_summed_end_state_is_the_sum_over_the_stages(number_of_steps, number_of_stages):
    cell = crossing(number_of_steps, number_of_stages)
    network = cell.network(END_STATE_SUMMED)
    assert network.number_of_outputs == number_of_stages
    assert network.body[-1].out_features == 4 * number_of_stages
    assert cell.network(END_STATE_PREDICTED).body[-1].out_features == 4 * (number_of_stages + 1)
    states, planes = cell.states(300, as_tensors=False)
    with torch.no_grad():
        track = network.predict(torch.as_tensor(states), torch.as_tensor(planes))
    assert track.end_state_was == END_STATE_SUMMED
    stages = track.stage_states[:, 0].numpy()
    # the sum, written again with numpy and the numpy equation of motion
    rates = np.empty_like(stages)
    for k in range(number_of_stages):
        at_stage = np.concatenate([stages[:, k], states[:, 4:5]], axis=1)
        rates[:, k] = equations()[0].rates(
            at_stage, planes + cell.tableau.nodes[k] * cell.step_length_mm)[:, :4]
    summed = states[:, :4] + cell.step_length_mm * np.einsum(
        "k,nkd->nd", cell.tableau.weights, rates)
    scale = np.abs(summed).max(axis=0)
    assert (np.abs(track.end_states[:, 0].numpy() - summed).max(axis=0) < 1e-13 * scale).all()


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
def test_exact_stages_sum_to_the_exact_end_state(number_of_steps, number_of_stages):
    cell = crossing(number_of_steps, number_of_stages)
    states, planes = cell.states(40, as_tensors=False)
    exact = steps_of_the_exact_scheme(equations()[0], states, planes,
                                      cell.step_length_mm, cell.tableau)
    S = torch.as_tensor(states)
    stages = torch.as_tensor(exact.stage_states[:, 0])
    stage_planes = (torch.as_tensor(planes)[:, None]
                    + torch.as_tensor(cell.tableau.nodes)[None, :] * cell.step_length_mm)
    rates = rates_at_the_stages(equations()[1], S, stages, stage_planes)
    summed = collect_and_sum(S, rates, cell.step_length_mm,
                             torch.as_tensor(cell.tableau.weights)).numpy()
    difference = np.abs(summed - exact.end_states[:, 0])
    assert difference[:, :2].max() < 1e-11          # millimetres
    assert difference[:, 2:].max() < 1e-15


@pytest.mark.parametrize("end_state", (END_STATE_SUMMED, END_STATE_PREDICTED))
def test_self_chain_applies_the_network_step_after_step(end_state):
    cell = crossing(64, 2)
    network = cell.network(end_state)
    layout = TrackLayout(cell.first_plane_mm, cell.last_plane_mm, 64, cell.tableau)
    start = frozen_tracks()["test_S0"][:50]
    track = network.whole_track(start, layout)
    assert track.is_a_whole_track and track.end_state_was == end_state
    assert track.stage_states.shape == (50, 64, 2, 4)
    current = start.copy()
    with torch.no_grad():
        for k in range(64):
            plane = cell.first_plane_mm + k * cell.step_length_mm
            assert identical(track.input_state(k), current)
            step = network.predict(torch.as_tensor(current),
                                   torch.full((50,), plane, dtype=torch.float64))
            assert identical(track.stage_states_of(k), step.stage_states[:, 0].numpy())
            current = np.concatenate([step.end_states[:, 0].numpy(), current[:, 4:5]], axis=1)
    assert identical(track.final_state(), current)
    assert identical(track.states_on_planes()[:, :, 4], np.repeat(start[:, 4:5], 65, axis=1))


def test_self_chain_is_identical_to_the_frozen_chaining():
    cell = crossing(64, 2)
    ours = cell.network(END_STATE_PREDICTED)
    theirs = frozen_one_step_network(cell, 0)
    layout = TrackLayout(cell.first_plane_mm, cell.last_plane_mm, 64, cell.tableau)
    start = frozen_tracks()["test_S0"][:50]
    track = ours.whole_track(start, layout)
    # the loop of the frozen `carry`, with the frozen network
    current = start.copy()
    kept = [current.copy()]
    with torch.no_grad():
        for k in range(64):
            planes = torch.full((50,), cell.first_plane_mm + k * cell.step_length_mm,
                                dtype=torch.float64)
            end = theirs(torch.as_tensor(current),
                         ours.start_plane_as_input(planes)[:, None])[:, -1, :].numpy()
            current = np.concatenate([end, current[:, 4:5]], axis=1)
            kept.append(current.copy())
    assert identical(track.states_on_planes(), np.stack(kept, axis=1))


def test_a_layout_that_is_not_the_network_s_is_refused():
    cell = crossing(64, 2)
    network = cell.network(END_STATE_SUMMED)
    start = frozen_tracks()["test_S0"][:5]
    with pytest.raises(ValueError):
        network.whole_track(start, TrackLayout(cell.first_plane_mm, cell.last_plane_mm,
                                               128, cell.tableau))
    with pytest.raises(ValueError):
        network.whole_track(start, TrackLayout(cell.first_plane_mm, cell.last_plane_mm,
                                               64, crossing(256, 16).tableau))
    with pytest.raises(ValueError):
        cell.network("guessed")


@pytest.mark.parametrize("end_state", (END_STATE_SUMMED, END_STATE_PREDICTED))
def test_a_network_is_built_again_from_its_settings_and_weights(end_state):
    cell = crossing(64, 2)
    network = cell.network(end_state, seed=4)
    snapshot = network.state_dict()
    assert not [name for name in snapshot if "field" in name], \
        "the field map must not be written into a snapshot"
    assert sum(v.numel() for v in snapshot.values()) < 60_000
    settings = network.settings()
    again = build_stage_network(
        seed=99, equation_of_motion=equations()[1], tableau=cell.tableau,
        step_length_mm=settings["step_length_mm"],
        first_start_plane_mm=settings["first_start_plane_mm"],
        last_start_plane_mm=settings["last_start_plane_mm"],
        scale_of_inputs=settings["scale_of_inputs"],
        output_form=DirectStates(settings["scale_of_outputs"]),
        width=settings["width"], depth=settings["depth"], end_state=settings["end_state"])
    again.load_state_dict(snapshot)
    states, planes = cell.states(100)
    with torch.no_grad():
        assert identical(again(states, planes).numpy(), network(states, planes).numpy())
    assert isinstance(again, StageNetwork) and again.settings() == settings


def test_a_scale_must_be_positive():
    with pytest.raises(ValueError):
        DirectStates([1.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        DirectStates([1.0, 1.0, 1.0])
    assert DirectStates([1.0, 1.0, 1.0, 1.0]).name == "direct_states"
