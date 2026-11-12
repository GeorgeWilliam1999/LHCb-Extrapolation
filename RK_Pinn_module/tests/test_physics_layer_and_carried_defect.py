"""Gate: the physics layer, the carried-defect loss and the whole-track protocol
(added 2026-11-12; plan "RK-PINN: plan for the next phase", options A and B).

Proves
  the physics layer (networks/stage_network_with_a_physics_layer.py)
  * with no sweep inside it is the stage network, to the last bit;
  * with one sweep its stages are S0 + dz A f(body stages), its end state is
    summed from the rates at those stages, and the body's stages are kept;
  * the gradient of the pooled loss reaches the body through the layer;
  * each sweep inside lowers the stage residual of a moved network;
  * a predicted end state and a wrong number of sweeps are refused;
  * a run trains with it and the chain uses the layer;
  the whole-track protocol (training/training_protocols.py)
  * a round holds whole tracks: every start plane, track after track, and in
    round 1 the reference states of those tracks;
  the carried-defect loss (losses/pooled_with_a_carried_defect_term.py)
  * the defect of a step is the end state summed from the stages minus the end
    state summed from the swept stages;
  * the carry is delta_x + delta_tx * lever, and the same in y;
  * with share 0 it is the pooled loss, to the last bit;
  * lambda makes the term its share of round one; the loss is zero at the
    exact scheme's stages;
  * on whole tracks the term is the squared sum over the steps of a track, and
    on other states the mean square over the states;
  * a run trains on it and records lambda.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from frozen_code import identical
from rkpinn.losses.pooled_with_a_carried_defect_term import (
    are_whole_tracks, carried_to_the_last_plane, defect_of_each_step)
from rkpinn.losses.stage_residual import stage_residual
from rkpinn.networks.collect_and_sum import collect_and_sum
from rkpinn.predicted_track.predicted_track import steps_of_the_exact_scheme
from rkpinn.run_record.configuration import run_key
from rkpinn.training import round_trainer
from rkpinn.training.checkpoints import RunFolder
from small_run import small_configuration, small_store

QUIET = dict(allow_uncommitted_changes=True, report=lambda text: None)
WHOLE_TRACKS = {"name": "rounds_on_whole_tracks_of_own_predictions", "states_per_round": 400}


def layer_network(sweeps):
    return {"kind": "stage_network_with_a_physics_layer", "output_form": "direct_states",
            "width": 16, "depth": 2, "activation": "tanh",
            "end_state": "summed_from_the_stages",
            "scale_of_inputs": "spread_of_first_round_states",
            "scale_of_outputs": "spread_of_first_round_states", "sweeps_inside": sweeps}


def carried_loss(share=0.5):
    return {"name": "pooled_with_a_carried_defect_term", "terms": "stages",
            "share_of_the_carried_term_at_round_one": share}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return small_store(tmp_path_factory.mktemp("store_of_the_physics_layer"))


def moved(network, seed=1):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in network.parameters():
            p.add_(1e-3 * torch.randn_like(p))
    return network


def first_round(run, number=64):
    return (torch.as_tensor(run.first_round.states[:number]),
            torch.as_tensor(run.first_round.start_planes_mm[:number]))


# ---------------------------------------------------------------- physics layer
def test_with_no_sweep_it_is_the_stage_network(store):
    store, tracks_key = store
    plain = round_trainer.build_run(small_configuration(tracks_key), store)
    layer = round_trainer.build_run(
        small_configuration(tracks_key, network=layer_network(0)), store)
    states, planes = first_round(plain)
    with torch.no_grad():
        a, b = plain.network.predict(states, planes), layer.network.predict(states, planes)
    assert identical(a.stage_states, b.stage_states) and identical(a.end_states, b.end_states)
    assert b.first_call is None


def test_one_sweep_is_the_stage_map_of_the_body_stages(store):
    store, tracks_key = store
    run = round_trainer.build_run(
        small_configuration(tracks_key, network=layer_network(1)), store)
    network = moved(run.network)
    states, planes = first_round(run)
    with torch.no_grad():
        track = network.predict(states, planes)
        body = network.forward(states, planes)
        dz, tableau = network.step_length_mm, network.tableau
        A, b, c = (torch.as_tensor(v) for v in (tableau.stage_matrix, tableau.weights,
                                                 tableau.nodes))
        zs = planes[:, None] + c[None, :] * dz
        F = network.equation_of_motion.rates(body, states[:, 4:5], zs)
        swept = states[:, None, :4] + dz * torch.einsum("jk,nkd->njd", A, F)
        F2 = network.equation_of_motion.rates(swept, states[:, 4:5], zs)
    assert identical(track.first_call.stage_states[:, 0], body)
    assert float((track.stage_states[:, 0] - swept).abs().max()) < 1e-13
    assert float((track.end_states[:, 0] - collect_and_sum(states, F2, dz, b)).abs().max()) < 1e-13
    assert network.cost_per_step()["field_lookups_per_step"] == 4
    assert network.settings()["sweeps_inside"] == 1


def test_the_gradient_reaches_the_body_through_the_layer(store):
    store, tracks_key = store
    run = round_trainer.build_run(
        small_configuration(tracks_key, network=layer_network(1)), store)
    loss = round_trainer.loss_of_a_round(run, run.first_round)()
    loss.backward()
    grads = [p.grad for p in run.network.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_each_sweep_inside_lowers_the_residual(store):
    store, tracks_key = store
    values = []
    for sweeps in (0, 1, 2):
        run = round_trainer.build_run(
            small_configuration(tracks_key, network=layer_network(sweeps)), store)
        moved(run.network)
        with torch.no_grad():
            values.append(float(round_trainer.loss_of_a_round(run, run.first_round)()))
    assert values[1] < 0.1 * values[0] and values[2] < 0.1 * values[1]


def test_the_layer_refuses_what_it_cannot_do(store):
    store, tracks_key = store
    predicted = dict(layer_network(1), end_state="predicted")
    with pytest.raises(ValueError):
        round_trainer.build_run(small_configuration(
            tracks_key, network=predicted,
            loss={"name": "pooled", "terms": "stages_and_end_state"}), store)
    for wrong in (-1, 1.5, True):
        with pytest.raises(ValueError):
            round_trainer.build_run(
                small_configuration(tracks_key, network=layer_network(wrong)), store)


def test_a_run_trains_with_the_layer(store):
    store, tracks_key = store
    configuration = small_configuration(tracks_key, training_rounds_at_most=2,
                                        network=layer_network(1))
    state = round_trainer.train(configuration, store, if_the_run_exists="refuse", **QUIET)
    assert state["rounds_done"] == 2
    folder = RunFolder(store, run_key(configuration))
    run = round_trainer.build_run(configuration, store, folder.constants())
    run.network.load_state_dict(folder.weights_of_round(2))
    whole = run.network.whole_track(run.tracks.validation.start_state[:5], run.layout)
    with torch.no_grad():
        one = run.network.predict(torch.as_tensor(np.asarray(whole.input_states)[:, 0]),
                                  torch.as_tensor(np.asarray(whole.start_planes_mm)[:, 0]))
    assert identical(np.asarray(whole.stage_states)[:, 0], one.stage_states[:, 0].numpy())


# ---------------------------------------------------------------- whole tracks
def test_a_round_of_the_protocol_holds_whole_tracks(store):
    store, tracks_key = store
    run = round_trainer.build_run(
        small_configuration(tracks_key, training_protocol=WHOLE_TRACKS), store)
    N = run.layout.number_of_steps
    first = run.first_round
    assert len(first.states) == 400 and first.source == "reference states, whole tracks"
    assert are_whole_tracks(torch.as_tensor(first.start_planes_mm), run.layout.start_planes_mm)
    on = run.tracks.planes_a_step_starts_on(N)
    reference = run.tracks.training.reference_states_on_planes[:, on]
    rows = first.states.reshape(-1, N, 5)
    # every row of the round is one training track on its N start planes
    for track in rows[:10]:
        match = np.flatnonzero((reference[:, 0] == track[0]).all(axis=1))
        assert len(match) == 1 and identical(reference[match[0]], track)
    later = run.protocol.states_of_round(2, 0, run.tracks, run.layout, run.network, run.target)
    assert later.source == "own predictions, whole tracks"
    assert are_whole_tracks(torch.as_tensor(later.start_planes_mm), run.layout.start_planes_mm)
    assert identical(later.states.reshape(-1, N, 5)[:, 0, 4], later.states.reshape(-1, N, 5)[:, -1, 4])
    assert not are_whole_tracks(torch.as_tensor(first.start_planes_mm[:-1]),
                                run.layout.start_planes_mm)
    assert not are_whole_tracks(torch.as_tensor(first.start_planes_mm[::-1].copy()),
                                run.layout.start_planes_mm)


# ---------------------------------------------------------------- carried defect
def test_the_defect_is_summed_minus_swept_and_the_carry_is_a_straight_line(store):
    store, tracks_key = store
    run = round_trainer.build_run(small_configuration(
        tracks_key, loss=carried_loss(), training_protocol=WHOLE_TRACKS), store)
    network = moved(run.network)
    states, planes = first_round(run)
    with torch.no_grad():
        track = network.predict(states, planes)
        residual = stage_residual(track, network.equation_of_motion, "stages")
        defect = defect_of_each_step(residual, track, network.equation_of_motion)
        dz, tableau = network.step_length_mm, network.tableau
        A, b, c = (torch.as_tensor(v) for v in (tableau.stage_matrix, tableau.weights,
                                                 tableau.nodes))
        zs = planes[:, None] + c[None, :] * dz
        Y = track.stage_states[:, 0]
        F = network.equation_of_motion.rates(Y, states[:, 4:5], zs)
        swept = states[:, None, :4] + dz * torch.einsum("jk,nkd->njd", A, F)
        F2 = network.equation_of_motion.rates(swept, states[:, 4:5], zs)
        expected = track.end_states[:, 0] - collect_and_sum(states, F2, dz, b)
        carried = carried_to_the_last_plane(defect, planes, dz, run.layout.last_plane_mm)
    assert float((defect - expected).abs().max()) < 1e-13 * float(expected.abs().max())
    lever = run.layout.last_plane_mm - planes - dz
    assert identical(carried[:, 0], defect[:, 0] + defect[:, 2] * lever)
    assert identical(carried[:, 1], defect[:, 1] + defect[:, 3] * lever)
    assert float(lever.min()) >= -1e-9


def test_with_share_zero_it_is_the_pooled_loss(store):
    store, tracks_key = store
    old = round_trainer.build_run(
        small_configuration(tracks_key, training_protocol=WHOLE_TRACKS), store)
    new = round_trainer.build_run(small_configuration(
        tracks_key, loss=carried_loss(share=0.0), training_protocol=WHOLE_TRACKS), store)
    assert new.constants["loss"]["weight_of_the_carried_term"] == 0.0
    values, grads = [], []
    for run in (old, new):
        loss = round_trainer.loss_of_a_round(run, run.first_round)()
        run.network.zero_grad()
        loss.backward()
        values.append(loss.detach())
        grads.append(torch.cat([p.grad.reshape(-1) for p in run.network.parameters()]))
    assert identical(values[0], values[1]) and identical(grads[0], grads[1])


def test_lambda_makes_the_carried_term_its_share_and_the_exact_scheme_gives_zero(store):
    store, tracks_key = store
    run = round_trainer.build_run(small_configuration(
        tracks_key, loss=carried_loss(0.5), training_protocol=WHOLE_TRACKS), store)
    c = run.constants["loss"]
    weight = c["weight_of_the_carried_term"]
    assert c["first_round_states_are_whole_tracks"] is True
    assert weight > 0 and c["carried_at_round_one"] > 0
    assert abs(weight * c["carried_at_round_one"]
               / (c["pooled_at_round_one"] + weight * c["carried_at_round_one"]) - 0.5) < 1e-12
    with torch.no_grad():
        total = float(round_trainer.loss_of_a_round(run, run.first_round)())
    assert abs(total / (c["pooled_at_round_one"] + weight * c["carried_at_round_one"]) - 1) < 1e-12
    exact = steps_of_the_exact_scheme(run.equation_of_motion, run.first_round.states[:8],
                                      run.first_round.start_planes_mm[:8],
                                      run.layout.step_length_mm, run.layout.tableau)
    with torch.no_grad():
        assert float(run.loss.value(exact, None, c)) < 1e-18
    loss = round_trainer.loss_of_a_round(run, run.first_round)()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in run.network.parameters())


def test_the_term_sums_over_the_steps_of_a_track(store):
    store, tracks_key = store
    run = round_trainer.build_run(small_configuration(
        tracks_key, loss=carried_loss(0.5), training_protocol=WHOLE_TRACKS), store)
    network, c, N = moved(run.network), run.constants["loss"], run.layout.number_of_steps
    states = torch.as_tensor(run.first_round.states)
    planes = torch.as_tensor(run.first_round.start_planes_mm)
    with torch.no_grad():
        track = network.predict(states, planes)
        _, carried = run.loss.parts(track, c)
        residual = stage_residual(track, network.equation_of_motion, "stages")
        defect = defect_of_each_step(residual, track, network.equation_of_motion)
        each = carried_to_the_last_plane(defect, planes, network.step_length_mm,
                                         run.layout.last_plane_mm)
        each = each / torch.as_tensor(c["divisor_per_component"], dtype=torch.float64)[0:2]
        by_track = (each.reshape(-1, N, 2).sum(dim=1) ** 2).mean()
        # the same states in another order are not whole tracks: each state alone
        order = torch.randperm(len(states), generator=torch.Generator().manual_seed(0))
        _, alone = run.loss.parts(network.predict(states[order], planes[order]), c)
    assert abs(float(carried) / float(by_track) - 1) < 1e-12
    assert abs(float(alone) / float((each ** 2).mean()) - 1) < 1e-12
    assert float(carried) != float(alone)


def test_a_run_trains_on_the_carried_loss_and_records_lambda(store):
    store, tracks_key = store
    configuration = small_configuration(tracks_key, training_rounds_at_most=2,
                                        loss=carried_loss(), training_protocol=WHOLE_TRACKS)
    state = round_trainer.train(configuration, store, if_the_run_exists="refuse", **QUIET)
    assert state["rounds_done"] == 2
    recorded = RunFolder(store, run_key(configuration)).constants()["loss"]
    assert recorded["weight_of_the_carried_term"] > 0
    again = round_trainer.build_run(configuration, store,
                                    RunFolder(store, run_key(configuration)).constants())
    assert again.constants["loss"]["weight_of_the_carried_term"] \
        == recorded["weight_of_the_carried_term"]
