"""Gate: the two changes of Part B of the per-step bias study, added on 2026-10-28.

Proves
  * B.3, `rounds_on_own_predictions_with_residual_draws`: round 1 is that of
    `rounds_on_own_predictions` to the last bit; with the share 0 every round is;
    a later round has the budget, the uniform share drawn as the standard
    protocol draws, and the residual share taken from a pool by residual: the
    states it takes have a larger mean residual than the pool; the weight of a
    z-face crossing is applied where a step crosses a face; a run trains on it;
  * B.4, `pooled_with_a_bias_term`: the bins and the bias term have the right
    shape; the bias term is zero for a residual of zero mean in every bin and
    equals the square of a constant residual; with the share 0 the loss is the
    pooled loss to the last bit, value and gradient; lambda makes the bias term
    the share of the loss of round 1 and is recorded in the constants; the
    exact scheme gives zero; a run trains on it;
  * the old components are untouched: a configuration of the pooled loss and
    of `rounds_on_own_predictions` keeps its key (test_default_configuration.py
    holds those keys).

The runs are small and are made in a temporary store. They prove the
mechanics, not what the changes reach.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from frozen_code import identical
from rkpinn.losses.pooled_with_a_bias_term import bias_term, bins_of_states
from rkpinn.predicted_track.predicted_track import steps_of_the_exact_scheme
from rkpinn.run_record.configuration import run_key
from rkpinn.training import round_trainer
from rkpinn.training.checkpoints import RunFolder
from rkpinn.training.training_protocols import (
    FROM_OWN_PREDICTIONS_BY_RESIDUAL, RoundsOnOwnPredictionsWithResidualDraws,
    stage_residual_of_states, steps_across_a_z_face)
from small_run import small_configuration, small_store

QUIET = dict(allow_uncommitted_changes=True, report=lambda text: None)


def residual_draws(share=0.5, candidates=4, face=2.0):
    return {"name": "rounds_on_own_predictions_with_residual_draws", "states_per_round": 400,
            "share_drawn_by_residual": share, "candidates_per_residual_state": candidates,
            "weight_of_a_step_across_a_z_face": face}


def bias_loss(share=0.1):
    return {"name": "pooled_with_a_bias_term", "terms": "stages",
            "share_of_the_bias_term_at_round_one": share,
            "number_of_bands_of_start_planes": 2, "momentum_band_edges_gev": [3.0, 8.0, 20.0]}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return small_store(tmp_path_factory.mktemp("store_of_the_bias_study"))


def moved(network, seed=1):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in network.parameters():
            p.add_(1e-3 * torch.randn_like(p))
    return network


# ---------------------------------------------------------------- B.3
def test_residual_draws_keep_round_one_and_share_zero(store):
    store, tracks_key = store
    old = round_trainer.build_run(small_configuration(tracks_key), store)
    new = round_trainer.build_run(
        small_configuration(tracks_key, training_protocol=residual_draws()), store)
    assert identical(new.first_round.states, old.first_round.states)
    assert identical(new.first_round.start_planes_mm, old.first_round.start_planes_mm)
    zero = round_trainer.build_run(
        small_configuration(tracks_key, training_protocol=residual_draws(share=0.0)), store)
    moved(old.network)
    zero.network.load_state_dict(old.network.state_dict())
    a = old.protocol.states_of_round(2, 0, old.tracks, old.layout, old.network, old.target)
    b = zero.protocol.states_of_round(2, 0, zero.tracks, zero.layout, zero.network,
                                      zero.target)
    assert identical(a.states, b.states) and identical(a.start_planes_mm, b.start_planes_mm)
    assert a.source == b.source


def test_residual_draws_take_the_states_of_large_residual(store):
    store, tracks_key = store
    run = round_trainer.build_run(
        small_configuration(tracks_key, training_protocol=residual_draws()), store)
    moved(run.network)
    second = run.protocol.states_of_round(2, 0, run.tracks, run.layout, run.network,
                                          run.target)
    assert second.source == FROM_OWN_PREDICTIONS_BY_RESIDUAL
    assert second.states.shape == (400, 5) and second.start_planes_mm.shape == (400,)
    divisor = run.constants["loss"]["divisor_per_component"]
    r = stage_residual_of_states(run.network, second.states, second.start_planes_mm, divisor)
    uniform, by_residual = r[:200], r[200:]
    assert by_residual.mean() > 1.5 * uniform.mean()
    # the states drawn by residual are states of the network's own chain
    whole = run.network.whole_track(run.tracks.training.start_state, run.layout)
    own = np.asarray(whole.input_states).reshape(-1, 5)
    for state in second.states[200:220]:
        assert np.abs(own - state).max(axis=1).min() < 1e-12


def test_a_z_face_crossing_is_found_where_a_step_crosses_a_face():
    class Field:
        lowest_corner = torch.tensor([-4000.0, -4000.0, -500.0])
        inverse_cell_size = torch.tensor([0.01, 0.01, 0.01])
    crosses = steps_across_a_z_face(np.array([2550.0, 2610.0, 2599.0, 2630.0]), 80.0, Field())
    assert crosses.tolist() == [True, False, True, True]


def test_residual_draws_refuse_settings_out_of_range():
    for bad in (dict(share=1.5), dict(candidates=0), dict(face=0.0)):
        settings = residual_draws(**bad)
        settings.pop("name")
        with pytest.raises(ValueError):
            RoundsOnOwnPredictionsWithResidualDraws(**settings)


def test_a_run_trains_on_residual_draws(store):
    store, tracks_key = store
    configuration = small_configuration(tracks_key, training_rounds_at_most=2,
                                        training_protocol=residual_draws())
    state = round_trainer.train(configuration, store, if_the_run_exists="refuse", **QUIET)
    assert state["rounds_done"] == 2
    rows = RunFolder(store, run_key(configuration)).rounds()
    assert rows[1]["source_of_states"] == FROM_OWN_PREDICTIONS_BY_RESIDUAL


# ---------------------------------------------------------------- B.4
def test_bins_and_bias_term_have_their_shapes():
    states = torch.tensor([[0, 0, 0, 0, 0.3], [0, 0, 0, 0, -0.3], [0, 0, 0, 0, 0.01],
                           [0, 0, 0, 0, -0.01]], dtype=torch.float64)
    planes = torch.tensor([2648.2, 2648.2, 7745.0, 7745.0], dtype=torch.float64)
    bins, number = bins_of_states(states, planes, 2648.2, 7745.1, 8, [3.0, 8.0, 20.0, 50.0])
    assert number == 8 * 5 * 2 and bins.shape == (4,)
    assert len(set(bins.tolist())) == 4 and int(bins.max()) < number
    residual = torch.randn(4, 2, 4, dtype=torch.float64)
    assert bias_term(residual, bins, number).shape == ()


def test_bias_term_is_zero_for_a_residual_of_zero_mean_in_every_bin():
    bins = torch.tensor([0, 0, 3, 3, 3, 7], dtype=torch.long)
    r = torch.randn(6, 2, 4, dtype=torch.float64)
    for b in (0, 3, 7):
        r[bins == b] -= r[bins == b].mean(dim=0)
    assert float(bias_term(r, bins, 10)) < 1e-30
    constant = torch.full((6, 2, 4), 0.7, dtype=torch.float64)
    assert abs(float(bias_term(constant, bins, 10)) - 0.49) < 1e-15


def test_with_share_zero_it_is_the_pooled_loss(store):
    store, tracks_key = store
    old = round_trainer.build_run(small_configuration(tracks_key), store)
    new = round_trainer.build_run(
        small_configuration(tracks_key, loss=bias_loss(share=0.0)), store)
    assert new.constants["loss"]["weight_of_the_bias_term"] == 0.0
    assert identical(new.constants["loss"]["divisor_per_component"],
                     old.constants["loss"]["divisor_per_component"])
    values, grads = [], []
    for run in (old, new):
        loss = round_trainer.loss_of_a_round(run, run.first_round)()
        run.network.zero_grad()
        loss.backward()
        values.append(loss.detach())
        grads.append(torch.cat([p.grad.reshape(-1) for p in run.network.parameters()]))
    assert identical(values[0], values[1]) and identical(grads[0], grads[1])


def test_lambda_makes_the_bias_term_its_share_of_round_one(store):
    store, tracks_key = store
    run = round_trainer.build_run(small_configuration(tracks_key, loss=bias_loss()), store)
    c = run.constants["loss"]
    weight = c["weight_of_the_bias_term"]
    assert weight > 0 and c["bias_at_round_one"] > 0
    assert abs(weight * c["bias_at_round_one"]
               / (c["pooled_at_round_one"] + weight * c["bias_at_round_one"]) - 0.1) < 1e-12
    with torch.no_grad():
        total = float(round_trainer.loss_of_a_round(run, run.first_round)())
    assert abs(total / (c["pooled_at_round_one"] + weight * c["bias_at_round_one"]) - 1) < 1e-12
    exact = steps_of_the_exact_scheme(run.equation_of_motion, run.first_round.states[:8],
                                      run.first_round.start_planes_mm[:8],
                                      run.layout.step_length_mm, run.layout.tableau)
    with torch.no_grad():
        assert float(run.loss.value(exact, None, c)) < 1e-18


def test_a_run_trains_on_the_bias_loss_and_records_lambda(store):
    store, tracks_key = store
    configuration = small_configuration(tracks_key, training_rounds_at_most=2,
                                        loss=bias_loss())
    state = round_trainer.train(configuration, store, if_the_run_exists="refuse", **QUIET)
    assert state["rounds_done"] == 2
    recorded = RunFolder(store, run_key(configuration)).constants()["loss"]
    assert recorded["weight_of_the_bias_term"] > 0
    # resumed from the record, the run uses the recorded lambda, not a new one
    again = round_trainer.build_run(configuration, store, RunFolder(
        store, run_key(configuration)).constants())
    assert again.constants["loss"]["weight_of_the_bias_term"] == \
        recorded["weight_of_the_bias_term"]
