"""Gate: the networks, before any training.

Proves
  * with a predicted end state, the stage network is identical, to the last
    bit, to the frozen one-step network built from the same seed and scales,
    in its outputs and in the gradient of its outputs;
  * with the output form `straight_line_plus_scaled_correction`, the scale
    `per_track` and a predicted end state, the stage network is identical, to
    the last bit, to the frozen `ChainNetwork` of the second mini-paper built
    from the same seed and scales: outputs, gradient, the per-track scale, and
    the chain over the crossing;
  * the global scale of that form is the median of the per-track scale over
    the first round's states, and the form refuses what it cannot be built with;
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

from building_blocks import ACTIVATION, CELLS, DEPTH, WIDTH, crossing, equations
from frozen_code import frozen_chain_network, frozen_module, frozen_tracks, identical
from rkpinn.networks.collect_and_sum import collect_and_sum, rates_at_the_stages
from rkpinn.networks.output_forms import (
    MEDIAN_OF_PER_TRACK_SCALE, PER_TRACK, DirectStates, StraightLinePlusScaledCorrection,
    per_track_scale)
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


def frozen_chain_network_of(cell, seed):
    """The frozen `ChainNetwork` of the cell, drawn from the seed, on the up
    polarity, and the frozen module it comes from."""
    chain = frozen_chain_network()
    field = frozen_module("reference").make_field("up")
    return chain, chain.build(cell.number_of_stages, cell.number_of_steps, cell.length_mm,
                              cell.first_plane_mm, cell.spread, field, seed=seed)


def straight_line_form(scale=PER_TRACK):
    return StraightLinePlusScaledCorrection(scale)


def outputs_and_gradient(network, emit):
    """The outputs `emit()` gives and the gradient of their weighted sum with
    respect to the parameters of `network`."""
    for parameter in network.parameters():
        parameter.grad = None
    out = emit()
    weights = torch.linspace(0.5, 1.5, out.numel(), dtype=torch.float64).reshape(out.shape)
    (out * weights).sum().backward()
    gradient = torch.cat([p.grad.reshape(-1) for p in network.parameters()])
    return out.detach().numpy(), gradient.numpy()


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
@pytest.mark.parametrize("seed", (0, 3))
def test_stage_network_is_identical_to_the_frozen_network(number_of_steps,
                                                          number_of_stages, seed):
    cell = crossing(number_of_steps, number_of_stages)
    ours = cell.network(END_STATE_PREDICTED, seed=seed)
    theirs = frozen_one_step_network(cell, seed)
    states, planes = cell.states(NUMBER_OF_STATES)
    out_ours, gradient_ours = outputs_and_gradient(ours, lambda: ours(states, planes))
    out_theirs, gradient_theirs = outputs_and_gradient(
        theirs, lambda: theirs(states, ours.start_plane_as_input(planes)[:, None]))
    assert out_ours.shape == (NUMBER_OF_STATES, number_of_stages + 1, 4)
    assert identical(out_ours, out_theirs)
    assert identical(gradient_ours, gradient_theirs)


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
@pytest.mark.parametrize("seed", (0, 3))
def test_stage_network_with_the_straight_line_form_is_identical_to_the_frozen_chain_network(
        number_of_steps, number_of_stages, seed):
    """The network of the second mini-paper is in the package: with a predicted
    end state and the per-track scale, outputs and gradient to the last bit."""
    cell = crossing(number_of_steps, number_of_stages)
    ours = cell.network(END_STATE_PREDICTED, seed=seed, output_form=straight_line_form())
    chain, theirs = frozen_chain_network_of(cell, seed)
    states, planes = cell.states(NUMBER_OF_STATES)
    out_ours, gradient_ours = outputs_and_gradient(ours, lambda: ours(states, planes))
    out_theirs, gradient_theirs = outputs_and_gradient(theirs, lambda: theirs(states, planes))
    assert out_ours.shape == (NUMBER_OF_STATES, number_of_stages + 1, 4)
    assert identical(out_ours, out_theirs)
    assert identical(gradient_ours, gradient_theirs)
    # the per-track scale: identical to the frozen one, and within rounding of
    # the frozen numpy twin
    scale = per_track_scale(states, planes, cell.step_length_mm, equations()[1].field)
    assert identical(scale.numpy(), theirs.residual_scale(states, planes)[:, 0, :].numpy())
    I_B, I_y = chain.field_integrals_numpy(states.numpy(), planes.numpy(), cell.step_length_mm,
                                           frozen_module("reference").make_field("up"))
    twin = np.stack(chain.scales_numpy(states.numpy(), cell.step_length_mm, I_B, I_y), axis=1)
    assert np.abs(scale.numpy() / twin - 1).max() < 1e-14
    assert (scale > 0).all()


def test_self_chain_with_the_straight_line_form_is_identical_to_the_frozen_carry():
    """The paper's chaining: the frozen `carry` of the frozen `ChainNetwork`
    and the package's self chain of the same network give the same states on
    every plane, to the last bit."""
    cell = crossing(64, 2)
    ours = cell.network(END_STATE_PREDICTED, seed=1, output_form=straight_line_form())
    chain, theirs = frozen_chain_network_of(cell, 1)
    layout = TrackLayout(cell.first_plane_mm, cell.last_plane_mm, 64, cell.tableau)
    start = frozen_tracks()["test_S0"][:50]
    track = ours.whole_track(start, layout)
    assert identical(track.states_on_planes(), chain.carry(theirs, start, cell.first_plane_mm, 64))


@pytest.mark.parametrize("seed", (0, 2))
def test_the_global_scale_is_the_median_of_the_per_track_scale(seed):
    """The form built from a configuration: `per_track` stays per track; the
    median name becomes four numbers, the median over the first round's
    states; four numbers pass through. With four numbers the states are the
    straight line plus those numbers times the raw outputs."""
    cell = crossing(64, 2)

    class Context:
        """What the run context gives the form."""
        first_round_states = cell.first_round_states
        first_round_start_planes_mm = cell.first_round_start_planes_mm
        equation_of_motion = equations()[1]

        class layout:
            step_length_mm = cell.step_length_mm

        @staticmethod
        def scale(value, number):
            return [float(v) for v in value]

    per_track = StraightLinePlusScaledCorrection.from_configuration(PER_TRACK, Context)
    assert per_track.scale_setting() == PER_TRACK and per_track.scale_is_per_track
    median = StraightLinePlusScaledCorrection.from_configuration(
        MEDIAN_OF_PER_TRACK_SCALE, Context)
    expected = np.median(per_track_scale(
        torch.as_tensor(cell.first_round_states),
        torch.as_tensor(cell.first_round_start_planes_mm), cell.step_length_mm,
        equations()[1].field).numpy(), axis=0)
    assert median.scale_setting() == expected.tolist()
    assert not median.scale_is_per_track and (expected > 0).all()
    given = StraightLinePlusScaledCorrection.from_configuration([1.0, 2.0, 3.0, 4.0], Context)
    assert given.scale_setting() == [1.0, 2.0, 3.0, 4.0]

    network = cell.network(END_STATE_SUMMED, seed=seed, output_form=straight_line_form(expected))
    states, planes = cell.states(300)
    with torch.no_grad():
        raw = network.raw(states, planes)
        emitted = network(states, planes)
        straight = network.output_form.straight_line(states)
        assert identical(emitted.numpy(), (straight + torch.as_tensor(expected) * raw).numpy())
        # the straight line of the form is that of the frozen code, on the end plane
        assert identical(
            cell.network(END_STATE_PREDICTED, seed=seed, output_form=straight_line_form(expected))
            .output_form.straight_line(states)[:, -1].numpy(),
            frozen_chain_network().straight_numpy(states.numpy(), cell.step_length_mm))


def test_the_straight_line_form_refuses_what_it_cannot_be_built_with():
    with pytest.raises(ValueError):
        StraightLinePlusScaledCorrection("guessed")
    with pytest.raises(ValueError):
        StraightLinePlusScaledCorrection([1.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        StraightLinePlusScaledCorrection([1.0, 1.0, 1.0])
    # the whole-crossing network has no equation of motion, so no field map
    tracks = frozen_tracks()
    spread = tracks["train_S0"].std(axis=0)
    cell = crossing(2, 2)
    with pytest.raises(ValueError, match="field map"):
        build_whole_crossing_network(
            seed=5, first_plane_mm=cell.first_plane_mm, last_plane_mm=cell.last_plane_mm,
            scale_of_inputs=spread, output_form=straight_line_form(),
            width=WIDTH, depth=DEPTH, activation=ACTIVATION)


def test_whole_crossing_network_is_identical_to_the_frozen_network():
    cell = crossing(2, 2)
    tracks = frozen_tracks()
    states = torch.as_tensor(tracks["train_S0"][:NUMBER_OF_STATES])
    spread = tracks["train_S0"].std(axis=0)
    ours = build_whole_crossing_network(
        seed=5, first_plane_mm=cell.first_plane_mm, last_plane_mm=cell.last_plane_mm,
        scale_of_inputs=spread, output_form=DirectStates(spread[:4]),
        width=WIDTH, depth=DEPTH, activation=ACTIVATION)
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
@pytest.mark.parametrize("form", ("direct_states", "straight_line_plus_scaled_correction"))
def test_zeroed_last_layer_gives_the_declared_answer(end_state, form):
    cell = crossing(64, 2)
    network = cell.network(end_state, output_form=(
        DirectStates(cell.spread[:4]) if form == "direct_states" else straight_line_form()))
    with torch.no_grad():
        network.body[-1].weight.zero_()
        network.body[-1].bias.zero_()
        states, planes = cell.states(200)
        emitted = network(states, planes)
        declared = network.output_form.answer_with_zeroed_network(states, planes)
    assert identical(emitted.numpy(), declared.numpy())
    if form == "direct_states":
        assert (emitted == 0).all()
    else:
        # the straight line: x + tx (z_j - z_start), y + ty (z_j - z_start), tx, ty
        offsets = network.nodes_of_outputs[None, :] * cell.step_length_mm
        assert torch.allclose(emitted[:, :, 0], states[:, 0:1] + states[:, 2:3] * offsets,
                              rtol=0, atol=1e-12)
        assert torch.allclose(emitted[:, :, 1], states[:, 1:2] + states[:, 3:4] * offsets,
                              rtol=0, atol=1e-12)
        assert identical(emitted[:, :, 2].numpy(),
                         states[:, 2:3].expand(-1, emitted.shape[1]).numpy())
        assert identical(emitted[:, :, 3].numpy(),
                         states[:, 3:4].expand(-1, emitted.shape[1]).numpy())


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
@pytest.mark.parametrize("form", (
    "direct_states", "straight_line_plus_scaled_correction, per_track",
    "straight_line_plus_scaled_correction, four numbers"))
def test_a_network_is_built_again_from_its_settings_and_weights(end_state, form):
    from rkpinn.registry import component
    cell = crossing(64, 2)
    output_form = {
        "direct_states": lambda: DirectStates(cell.spread[:4]),
        "straight_line_plus_scaled_correction, per_track": lambda: straight_line_form(),
        "straight_line_plus_scaled_correction, four numbers":
            lambda: straight_line_form([0.1, 0.001, 0.002, 0.00002]),
    }[form]()
    network = cell.network(end_state, seed=4, output_form=output_form)
    snapshot = network.state_dict()
    assert not [name for name in snapshot if "field" in name], \
        "the field map must not be written into a snapshot"
    assert sum(v.numel() for v in snapshot.values()) < 60_000
    settings = network.settings()
    assert settings["output_form"] == form.split(",")[0]
    assert settings["scale_of_outputs"] == (
        PER_TRACK if "per_track" in form else [float(v) for v in output_form.scale_of_outputs])
    again = build_stage_network(
        seed=99, equation_of_motion=equations()[1], tableau=cell.tableau,
        step_length_mm=settings["step_length_mm"],
        first_start_plane_mm=settings["first_start_plane_mm"],
        last_start_plane_mm=settings["last_start_plane_mm"],
        scale_of_inputs=settings["scale_of_inputs"],
        output_form=component("output_form", settings["output_form"])(settings["scale_of_outputs"]),
        width=settings["width"], depth=settings["depth"],
        activation=settings["activation"], end_state=settings["end_state"])
    again.load_state_dict(snapshot)
    states, planes = cell.states(100)
    with torch.no_grad():
        assert identical(again(states, planes).numpy(), network(states, planes).numpy())
    assert isinstance(again, StageNetwork) and again.settings() == settings


def test_an_activation_that_is_not_built_is_refused():
    cell = crossing(2, 2)
    with pytest.raises(ValueError):
        build_stage_network(
            seed=0, equation_of_motion=equations()[1], tableau=cell.tableau,
            step_length_mm=cell.step_length_mm,
            first_start_plane_mm=cell.start_planes_mm[0],
            last_start_plane_mm=cell.start_planes_mm[-1], scale_of_inputs=cell.spread,
            output_form=DirectStates(cell.spread[:4]), width=8, depth=1,
            activation="no_such_activation", end_state=END_STATE_SUMMED)
    assert cell.network(END_STATE_SUMMED).settings()["activation"] == "tanh"


def test_a_scale_must_be_positive():
    with pytest.raises(ValueError):
        DirectStates([1.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        DirectStates([1.0, 1.0, 1.0])
    assert DirectStates([1.0, 1.0, 1.0, 1.0]).name == "direct_states"
