"""Gate: the losses, before any training.

The five gates every loss must pass (PACKAGE_PLAN.md, section 3.3):

  1. the exact collocation solution gives machine zero, under every loss and
     both choices of terms;
  2. the weights in torch equal an independent implementation in numpy, and
     are identical, to the last bit, to the weights of the frozen code;
  3. with every weight switched off, the weighted loss equals the unweighted
     one to the last bit;
  4. the constants recorded are the constants used: the loss reads the
     momentum window from the constants of the run and from nowhere else;
  5. a loss of zero implies the summed end state equals the exact scheme's:
     tests/test_networks.py, test_exact_stages_sum_to_the_exact_end_state.

And
  * the pooled loss and its gradient are identical, to the last bit, to the
    frozen loss, with a predicted end state and the terms of the paper;
  * the cost-weighted loss equals the frozen one to a relative 1e-12. It is
    not identical: the frozen code divides the residual by the scale of the
    inputs and multiplies it back, which rounds;
  * the combinations of terms and end state that make no sense are refused;
  * with a summed end state, the term of the end state is zero to rounding;
  * no loss imports a network.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pytest
import torch

import rkpinn
from building_blocks import (
    CELLS, SETTINGS_OF_THE_COST_WEIGHTED_LOSS, crossing, equations)
from frozen_code import (
    frozen_chain_network, frozen_module, frozen_tracks, frozen_windowed_loss, identical)
from rkpinn.losses import cost_weighted
from rkpinn.losses.cost_weighted import CostWeighted
from rkpinn.losses.pooled import Pooled
from rkpinn.losses.stage_residual import (
    TERMS_STAGES, TERMS_STAGES_AND_END_STATE, stage_residual)
from rkpinn.losses.supervised_endpoint import SupervisedEndpoint
from rkpinn.losses.unweighted import Unweighted
from rkpinn.predicted_track.predicted_track import (
    END_STATE_PREDICTED, END_STATE_SUMMED, steps_of_the_exact_scheme)

NUMBER_OF_STATES = 2000
WINDOWS = ((10.0, 50.0), (3.0, 8.0))
BOTH_TERMS = (TERMS_STAGES, TERMS_STAGES_AND_END_STATE)


def cost_weighted_loss(cell, terms, **changes):
    settings = dict(SETTINGS_OF_THE_COST_WEIGHTED_LOSS, **changes)
    return CostWeighted(equation_of_motion=equations()[1], terms=terms,
                        first_plane_mm=cell.first_plane_mm,
                        last_plane_mm=cell.last_plane_mm, **settings)


def all_losses(cell, terms):
    """(loss, constants) for the three label-free losses."""
    made = []
    for loss in (Unweighted(equation_of_motion=equations()[1], terms=terms),
                 Pooled(equation_of_motion=equations()[1], terms=terms),
                 cost_weighted_loss(cell, terms)):
        made.append((loss, loss.constants(cell.first_round_states)))
    return made


def value_and_gradient(network, value):
    for parameter in network.parameters():
        parameter.grad = None
    value.backward()
    return float(value.detach()), torch.cat(
        [p.grad.reshape(-1) for p in network.parameters()]).numpy()


class AsTheFrozenCodeSeesIt:
    """A network of the package, presented to the frozen losses, which call
    `model(S, extra)` and read `model.in_scale` and `model.n_extra`."""

    n_extra = 1

    def __init__(self, network, divisor):
        self.network = network
        self.in_scale = torch.as_tensor(np.asarray(divisor, dtype=np.float64))

    def __call__(self, states, start_planes_mm):
        return self.network(states, start_planes_mm)


# ------------------------------------------------------------------ the pooled loss
@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
def test_pooled_loss_is_identical_to_the_frozen_loss(number_of_steps, number_of_stages):
    cell = crossing(number_of_steps, number_of_stages)
    network = cell.network(END_STATE_PREDICTED)
    states, planes = cell.states(NUMBER_OF_STATES)
    loss = Pooled(equation_of_motion=equations()[1], terms=TERMS_STAGES_AND_END_STATE)
    constants = loss.constants(cell.first_round_states)
    assert identical(np.array(constants["divisor_per_component"]), cell.spread[:4])
    ours = value_and_gradient(
        network, loss.value(network.predict(states, planes), None, constants))

    frozen_model = frozen_module("model")
    frozen_reference = frozen_module("reference")
    rates = frozen_model.LHCbRates(frozen_reference.make_field("up"))
    stage_planes = planes[:, None] + torch.as_tensor(cell.tableau.nodes)[None, :] \
        * cell.step_length_mm
    theirs = value_and_gradient(network, frozen_model.physics_loss(
        AsTheFrozenCodeSeesIt(network, cell.spread), rates, states, cell.step_length_mm,
        stage_planes, torch.as_tensor(cell.tableau.stage_matrix),
        torch.as_tensor(cell.tableau.weights), planes))
    assert np.isfinite(ours[0]) and ours[0] > 0 and np.isfinite(ours[1]).all()
    assert ours[0] == theirs[0]
    assert identical(ours[1], theirs[1])


# ------------------------------------------------------------ gate 1: machine zero
@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
@pytest.mark.parametrize("terms", BOTH_TERMS)
def test_exact_solution_gives_machine_zero(number_of_steps, number_of_stages, terms):
    cell = crossing(number_of_steps, number_of_stages)
    states, planes = cell.states(40, as_tensors=False)
    exact = steps_of_the_exact_scheme(equations()[0], states, planes,
                                      cell.step_length_mm, cell.tableau)
    network = cell.network(END_STATE_SUMMED)
    with torch.no_grad():
        untrained = network.predict(torch.as_tensor(states), torch.as_tensor(planes))
    for loss, constants in all_losses(cell, terms):
        at_the_solution = float(loss.value(exact, None, constants))
        if terms == TERMS_STAGES_AND_END_STATE:
            untrained.end_state_was = END_STATE_SUMMED
        of_a_network = float(loss.value(untrained, None, constants))
        assert 0 <= at_the_solution < 1e-20, (loss.name, at_the_solution)
        assert at_the_solution < 1e-20 * of_a_network, (loss.name, of_a_network)


# ----------------------------------------------------------- gate 2: the weights
def independent_weights(states, start_planes, nodes, step_length, constants):
    """The weights of equation 20, written again from the formula, row by row."""
    a = np.empty(len(states))
    for n, state in enumerate(states):
        p = 0.299792458 / max(abs(state[4]), 1e-12)
        low, high = constants["momentum_window_gev"]
        if p < low:
            window = np.exp(-(np.log(p / low) / constants["roll_off"]) ** 2)
        elif p > high:
            window = np.exp(-(np.log(p / high) / constants["roll_off"]) ** 2)
        else:
            window = 1.0
        window = max(window, constants["floor"])
        bend = 1e-3 * abs(state[4]) * constants["field_integral_tesla_mm"] \
            * constants["length_mm"]
        a[n] = np.sqrt(window) * constants["reference_bend_mm"] / bend
    middle = np.median(a)
    a = np.clip(a, middle / constants["clamp"], middle * constants["clamp"])
    out = np.empty((len(states), len(nodes), 4))
    for n in range(len(states)):
        for j, node in enumerate(nodes):
            lever = max(constants["last_plane_mm"]
                        - (start_planes[n] + node * step_length), 0.0) + step_length
            out[n, j] = a[n] * np.array([1.0, 1.0, lever, lever]) \
                / constants["reference_bend_mm"]
    return out


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
@pytest.mark.parametrize("window", WINDOWS)
def test_weights_equal_an_independent_implementation(number_of_steps, number_of_stages,
                                                     window):
    cell = crossing(number_of_steps, number_of_stages)
    loss = cost_weighted_loss(cell, TERMS_STAGES, momentum_window_gev=window)
    constants = loss.constants(cell.first_round_states)
    states, planes = cell.states(1500, as_tensors=False)
    nodes = cell.tableau.nodes
    in_torch = cost_weighted.weights(torch.as_tensor(states), torch.as_tensor(planes),
                                     torch.as_tensor(nodes), cell.step_length_mm,
                                     constants).numpy()
    in_numpy = cost_weighted.weights(states, planes, nodes, cell.step_length_mm, constants)
    independent = independent_weights(states, planes, nodes, cell.step_length_mm, constants)
    assert in_torch.shape == (1500, number_of_stages, 4)
    assert (in_torch > 0).all()
    assert np.abs(in_torch / independent - 1).max() < 1e-13
    assert np.abs(in_numpy / independent - 1).max() < 1e-13


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
@pytest.mark.parametrize("window", WINDOWS)
def test_weights_are_identical_to_the_frozen_code(number_of_steps, number_of_stages, window):
    cell = crossing(number_of_steps, number_of_stages)
    frozen = frozen_windowed_loss()
    frozen_reference = frozen_module("reference")
    field = frozen_reference.make_field("up")
    their_network = frozen_chain_network().build(
        number_of_stages, number_of_steps, cell.length_mm, cell.first_plane_mm,
        cell.spread, field)
    field_integral = frozen.i_bar(field, cell.first_plane_mm, cell.last_plane_mm)
    theirs = frozen.reference_constants(
        their_network, cell.first_round_states, cell.first_round_start_planes_mm,
        cell.last_plane_mm, field_integral, cell.length_mm, mode="full",
        clamp=SETTINGS_OF_THE_COST_WEIGHTED_LOSS["clamp"], p_lo=window[0], p_hi=window[1],
        rolloff=SETTINGS_OF_THE_COST_WEIGHTED_LOSS["roll_off"],
        w_floor=SETTINGS_OF_THE_COST_WEIGHTED_LOSS["floor"])
    loss = cost_weighted_loss(cell, TERMS_STAGES_AND_END_STATE, momentum_window_gev=window)
    ours = loss.constants(cell.first_round_states)
    assert ours["reference_bend_mm"] == theirs["D_ref"]
    assert ours["field_integral_tesla_mm"] == theirs["i_bar"]
    assert ours["length_mm"] == theirs["L"] and ours["last_plane_mm"] == theirs["z1"]

    states, planes = cell.states(NUMBER_OF_STATES)
    nodes = torch.cat([torch.as_tensor(cell.tableau.nodes),
                       torch.ones(1, dtype=torch.float64)])
    assert identical(nodes.numpy(), their_network.cout.numpy())
    in_torch = cost_weighted.weights(states, planes, nodes, cell.step_length_mm, ours)
    assert identical(in_torch.numpy(),
                     frozen.weights(their_network, states, planes, theirs).numpy())
    in_numpy = cost_weighted.weights(states.numpy(), planes.numpy(), nodes.numpy(),
                                     cell.step_length_mm, ours)
    assert identical(in_numpy, frozen.weights(their_network, states.numpy(),
                                              planes.numpy(), theirs))


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
def test_cost_weighted_loss_equals_the_frozen_loss(number_of_steps, number_of_stages):
    cell = crossing(number_of_steps, number_of_stages)
    network = cell.network(END_STATE_PREDICTED)
    states, planes = cell.states(NUMBER_OF_STATES)
    loss = cost_weighted_loss(cell, TERMS_STAGES_AND_END_STATE)
    constants = loss.constants(cell.first_round_states)
    ours = value_and_gradient(
        network, loss.value(network.predict(states, planes), None, constants))

    frozen = frozen_windowed_loss()
    frozen_model = frozen_module("model")
    rates = frozen_model.LHCbRates(frozen_module("reference").make_field("up"))
    nodes = network.nodes_of_outputs
    weights = cost_weighted.weights(states, planes, nodes, cell.step_length_mm, constants)
    stage_planes = planes[:, None] + torch.as_tensor(cell.tableau.nodes)[None, :] \
        * cell.step_length_mm
    theirs = value_and_gradient(network, frozen.weighted_loss(
        AsTheFrozenCodeSeesIt(network, cell.spread), rates, states, cell.step_length_mm,
        stage_planes, torch.as_tensor(cell.tableau.stage_matrix),
        torch.as_tensor(cell.tableau.weights), planes, weights))
    assert abs(ours[0] / theirs[0] - 1) < 1e-12
    assert np.abs(ours[1] - theirs[1]).max() < 1e-10 * np.abs(theirs[1]).max()


# ------------------------------------------------ gate 3: every weight switched off
@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
@pytest.mark.parametrize("end_state, terms", ((END_STATE_SUMMED, TERMS_STAGES),
                                              (END_STATE_PREDICTED,
                                               TERMS_STAGES_AND_END_STATE)))
def test_with_every_weight_off_it_is_the_unweighted_loss(number_of_steps, number_of_stages,
                                                         end_state, terms):
    cell = crossing(number_of_steps, number_of_stages)
    network = cell.network(end_state)
    states, planes = cell.states(NUMBER_OF_STATES)
    off = cost_weighted_loss(cell, terms, lever_arm_is_on=False, track_bend_is_on=False,
                             momentum_window_is_on=False, reference_bend_mm=1.0)
    constants = off.constants(cell.first_round_states)
    assert constants["reference_bend_mm"] == 1.0
    weighted = value_and_gradient(
        network, off.value(network.predict(states, planes), None, constants))
    plain = Unweighted(equation_of_motion=equations()[1], terms=terms)
    unweighted = value_and_gradient(network, plain.value(
        network.predict(states, planes), None, plain.constants()))
    assert weighted[0] == unweighted[0]
    assert identical(weighted[1], unweighted[1])
    on = cost_weighted_loss(cell, terms)
    with torch.no_grad():
        assert float(on.value(network.predict(states, planes), None,
                              on.constants(cell.first_round_states))) != unweighted[0]


@pytest.mark.parametrize("switch", ("lever_arm_is_on", "track_bend_is_on",
                                    "momentum_window_is_on"))
def test_each_switch_changes_the_loss(switch):
    cell = crossing(64, 2)
    network = cell.network(END_STATE_SUMMED)
    states, planes = cell.states(NUMBER_OF_STATES)
    with torch.no_grad():
        track = network.predict(states, planes)
        values = []
        for is_on in (True, False):
            loss = cost_weighted_loss(cell, TERMS_STAGES, **{switch: is_on})
            constants = loss.constants(cell.first_round_states)
            assert constants[switch] is is_on
            values.append(float(loss.value(track, None, constants)))
    assert values[0] != values[1]


# ------------------------------------- gate 4: the constants recorded are those used
def test_the_loss_reads_its_constants_from_the_run_and_nowhere_else():
    cell = crossing(64, 2)
    network = cell.network(END_STATE_SUMMED)
    states, planes = cell.states(NUMBER_OF_STATES)
    with torch.no_grad():
        track = network.predict(states, planes)
    built_with_one_window = cost_weighted_loss(cell, TERMS_STAGES,
                                               momentum_window_gev=(10.0, 50.0))
    built_with_another = cost_weighted_loss(cell, TERMS_STAGES,
                                            momentum_window_gev=(3.0, 8.0))
    of_one = built_with_one_window.constants(cell.first_round_states)
    of_another = built_with_another.constants(cell.first_round_states)
    assert of_one["momentum_window_gev"] == [10.0, 50.0]
    assert of_another["momentum_window_gev"] == [3.0, 8.0]
    # the value follows the constants it is given, not the loss that computes it
    a = float(built_with_one_window.value(track, None, of_another))
    b = float(built_with_another.value(track, None, of_another))
    c = float(built_with_one_window.value(track, None, of_one))
    assert a == b and a != c
    # and it is the value of the formula with the window of the constants
    residual = stage_residual(track, equations()[1], TERMS_STAGES)
    weights = independent_weights(states.numpy(), planes.numpy(), cell.tableau.nodes,
                                  cell.step_length_mm, of_another)
    by_hand = float(((residual.residual.numpy() * weights) ** 2).mean())
    assert abs(a / by_hand - 1) < 1e-12


def test_constants_hold_every_setting():
    cell = crossing(64, 2)
    loss = cost_weighted_loss(cell, TERMS_STAGES)
    constants = loss.constants(cell.first_round_states)
    for name, value in SETTINGS_OF_THE_COST_WEIGHTED_LOSS.items():
        if name == "reference_bend_mm":
            assert isinstance(constants[name], float) and constants[name] > 0
        elif name == "momentum_window_gev":
            assert constants[name] == list(value)
        else:
            assert constants[name] == value
    assert constants["terms"] == TERMS_STAGES and constants["name"] == "cost_weighted"
    with pytest.raises(TypeError):
        CostWeighted(equation_of_motion=equations()[1], terms=TERMS_STAGES,
                     first_plane_mm=0.0, last_plane_mm=1.0)      # no setting has a default


# -------------------------------------------------------- terms and the end state
def test_combinations_that_make_no_sense_are_refused():
    cell = crossing(64, 2)
    states, planes = cell.states(100)
    with torch.no_grad():
        predicted = cell.network(END_STATE_PREDICTED).predict(states, planes)
    with pytest.raises(ValueError):
        stage_residual(predicted, equations()[1], TERMS_STAGES)
    with pytest.raises(ValueError):
        stage_residual(predicted, equations()[1], "everything")
    with pytest.raises(ValueError):
        Pooled(equation_of_motion=equations()[1], terms="everything")


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
def test_term_of_a_summed_end_state_is_zero_to_rounding(number_of_steps, number_of_stages):
    cell = crossing(number_of_steps, number_of_stages)
    states, planes = cell.states(NUMBER_OF_STATES)
    with torch.no_grad():
        track = cell.network(END_STATE_SUMMED).predict(states, planes)
        with_end = stage_residual(track, equations()[1], TERMS_STAGES_AND_END_STATE)
        without = stage_residual(track, equations()[1], TERMS_STAGES)
    q = number_of_stages
    assert with_end.residual.shape == (NUMBER_OF_STATES, q + 1, 4)
    assert identical(with_end.residual[:, :q].numpy(), without.residual.numpy())
    assert with_end.residual[:, q, :2].abs().max() < 1e-11        # millimetres
    assert with_end.residual[:, q, 2:].abs().max() < 1e-14
    assert float(with_end.nodes_of_terms[-1]) == 1.0
    loss = Pooled(equation_of_motion=equations()[1], terms=TERMS_STAGES)
    constants = loss.constants(cell.first_round_states)
    stages_only = float(loss.value(track, None, constants))
    both = float(loss.value(track, None, dict(constants, terms=TERMS_STAGES_AND_END_STATE)))
    assert abs(both / (stages_only * q / (q + 1)) - 1) < 1e-12


def test_rates_kept_in_the_track_are_the_rates_the_loss_would_compute():
    cell = crossing(64, 2)
    states, planes = cell.states(500)
    with torch.no_grad():
        track = cell.network(END_STATE_SUMMED).predict(states, planes)
        kept = stage_residual(track, equations()[1], TERMS_STAGES).residual.numpy()
        track.stage_rates = None
        computed = stage_residual(track, equations()[1], TERMS_STAGES).residual.numpy()
    assert identical(kept, computed)


# ------------------------------------------------------ the supervised endpoint loss
def test_supervised_endpoint_loss():
    from rkpinn.networks.output_forms import DirectStates
    from rkpinn.networks.whole_crossing_network import build_whole_crossing_network
    tracks = frozen_tracks()
    start, end = tracks["train_S0"][:3000], tracks["train_truth"][:3000, 256]
    spread = start.std(axis=0)
    network = build_whole_crossing_network(
        seed=0, first_plane_mm=float(tracks["z0"]), last_plane_mm=float(tracks["z1"]),
        scale_of_inputs=spread, output_form=DirectStates(end.std(axis=0)[:4]),
        width=128, depth=2, activation="tanh")
    loss = SupervisedEndpoint(
        divisor=SupervisedEndpoint.MEASURED_WHEN_CONSTANTS_ARE_TAKEN)
    given = SupervisedEndpoint(divisor=[1.0, 2.0, 3.0, 4.0])
    assert given.constants()["divisor_per_component"] == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError):
        SupervisedEndpoint(divisor=[1.0, 2.0, 0.0, 4.0])
    assert loss.needs_labels is True
    constants = loss.constants(end)
    assert identical(np.array(constants["divisor_per_component"]), end.std(axis=0)[:4])
    track = network.predict(torch.as_tensor(start))
    value, gradient = value_and_gradient(network, loss.value(track, end, constants))
    by_hand = (((track.end_states[:, 0].detach().numpy() - end[:, :4])
                / end.std(axis=0)[:4]) ** 2).mean()
    assert abs(value / by_hand - 1) < 1e-13
    assert np.isfinite(gradient).all() and np.abs(gradient).max() > 0
    perfect = network.predict(torch.as_tensor(start))
    perfect.end_states = torch.as_tensor(end[:, None, :4].copy())
    assert float(loss.value(perfect, end, constants)) == 0.0
    with pytest.raises(ValueError):
        loss.value(track, None, constants)
    # it reads the end of the last step, so it applies to a chained network too
    cell = crossing(64, 2)
    chained = cell.network(END_STATE_SUMMED).whole_track(
        start[:20], __import__("rkpinn.predicted_track.track_layout", fromlist=["x"])
        .TrackLayout(cell.first_plane_mm, cell.last_plane_mm, 64, cell.tableau))
    of_the_chain = float(loss.value(chained, end[:20], constants))
    by_hand = (((chained.final_state()[:, :4] - end[:20, :4])
                / end.std(axis=0)[:4]) ** 2).mean()
    assert abs(of_the_chain / by_hand - 1) < 1e-13


# ------------------------------------------------------------------ the two rules
@pytest.mark.parametrize("directory", ("losses", "evaluation"))
def test_no_loss_and_no_evaluation_imports_a_network(directory):
    folder = os.path.join(os.path.dirname(rkpinn.__file__), directory)
    importing = re.compile(r"^\s*(from|import)\s+rkpinn\.networks|^\s*from\s+rkpinn\s+import\s+networks",
                           re.MULTILINE)
    for name in sorted(os.listdir(folder)):
        if name.endswith(".py"):
            with open(os.path.join(folder, name)) as handle:
                assert not importing.search(handle.read()), \
                    "%s/%s imports a network" % (directory, name)
