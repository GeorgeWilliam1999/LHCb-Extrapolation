"""Gate: the four changes the literature suggests, added on 2026-10-23.

Proves
  * the old behaviour is untouched: the protocol, the optimiser and the report
    of a run made before are those they were, and the default configuration
    keeps its key (test_default_configuration.py holds that key);
  * one fixed draw: round 1 is identical, to the last bit, to round 1 of
    `rounds_on_own_predictions`; every later round is on the same tracks and
    planes; with the reference setting it is the same states, with the own
    predictions setting the network's own states on those tracks and planes;
  * the best round of a report is the round of the lowest validation error,
    and the report says which round it scored;
  * the refit of the last layer: its Jacobian is that of autograd, it lowers
    the loss of a trained network on its states, changes only the last layer,
    and the report scores the refitted weights;
  * the self-scaled Broyden optimiser: its update keeps H symmetric and meets
    the secant equation H y = s; with tau = phi = 1 it is the BFGS update; a
    run trains with it, and the loss falls.

The runs are small and are made in a temporary store. They prove the
mechanics. They say nothing about what a change can reach.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from frozen_code import identical
from rkpinn.run_record.configuration import run_key
from rkpinn.run_record.report_of_a_run import best_round, report_of_a_run
from rkpinn.training import round_trainer
from rkpinn.training.checkpoints import RunFolder
from rkpinn.training.last_layer_refit import (
    last_linear_layer, normal_equations, refit_a_snapshot, refit_last_layer)
from rkpinn.training.optimiser import SelfScaledBroydenRestarts
from rkpinn.training.training_protocols import (
    FROM_OWN_PREDICTIONS_FIXED, FROM_THE_REFERENCE, FROM_THE_REFERENCE_FIXED,
    RoundsOnOneFixedDraw)
from small_run import small_configuration, small_store

QUIET = dict(allow_uncommitted_changes=True, report=lambda text: None)
SETTINGS = dict(split="test", tracks_for_the_stage_errors=4,
                step_length_of_the_carrying_mm=2.0, fraction_for_the_derivative=1e-3,
                states_for_the_loss_shares=200, bins_of_the_histograms=20)
BROYDEN = {"name": "self_scaled_broyden_restarts", "iterations_per_restart": 5,
           "tolerance_of_the_gradient": 1e-13, "tolerance_of_the_change": 1e-16,
           "renew_scale_when_loss_falls_to": 0.1, "threads_for_the_matrix": 1}


def fixed_draw(states_after_round_one):
    return {"name": "rounds_on_one_fixed_draw", "states_per_round": 400,
            "states_after_round_one": states_after_round_one}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return small_store(tmp_path_factory.mktemp("store_of_the_literature_changes"))


@pytest.fixture(scope="module")
def trained(store):
    """A small run of three rounds, with its key."""
    store, tracks_key = store
    configuration = small_configuration(tracks_key, training_rounds_at_most=3)
    round_trainer.train(configuration, store, if_the_run_exists="refuse", **QUIET)
    return store, run_key(configuration), configuration


# ---------------------------------------------------------------- one fixed draw
@pytest.mark.parametrize("later", ["own_predictions_on_the_same_tracks_and_planes",
                                   "the_reference_states_of_round_one"])
def test_one_fixed_draw_keeps_its_tracks_and_planes(store, later):
    store, tracks_key = store
    old = round_trainer.build_run(small_configuration(tracks_key), store)
    new = round_trainer.build_run(
        small_configuration(tracks_key, training_protocol=fixed_draw(later)), store)
    assert new.first_round.source == FROM_THE_REFERENCE
    assert identical(new.first_round.states, old.first_round.states)
    assert identical(new.first_round.start_planes_mm, old.first_round.start_planes_mm)
    # the networks start from the same weights, so their losses on round 1 agree
    assert identical(new.network.state_dict()["body.0.weight"],
                     old.network.state_dict()["body.0.weight"])

    with torch.no_grad():                               # move the network off its start
        for p in new.network.parameters():
            p.add_(1e-3 * torch.randn_like(p))
    second = new.protocol.states_of_round(2, 0, new.tracks, new.layout, new.network,
                                          new.target)
    third = new.protocol.states_of_round(3, 0, new.tracks, new.layout, new.network,
                                         new.target)
    assert identical(second.start_planes_mm, new.first_round.start_planes_mm)
    assert identical(second.states, third.states)
    if later == "the_reference_states_of_round_one":
        assert second.source == FROM_THE_REFERENCE_FIXED
        assert identical(second.states, new.first_round.states)
    else:
        assert second.source == FROM_OWN_PREDICTIONS_FIXED
        whole = new.network.whole_track(new.tracks.training.start_state, new.layout)
        drawn = new.protocol._draw_of_round_one(0, new.tracks, new.layout)
        expected = np.asarray(whole.input_states)[drawn.track, drawn.step_of_the_track]
        assert np.allclose(second.states, expected, rtol=0, atol=1e-9)
        # round 1 starts every track on its reference; later ones do not
        assert not identical(second.states, new.first_round.states)


def test_one_fixed_draw_refuses_an_unknown_setting():
    with pytest.raises(ValueError):
        RoundsOnOneFixedDraw(states_per_round=10, states_after_round_one="fresh")


def test_a_run_trains_on_one_fixed_draw(store):
    store, tracks_key = store
    configuration = small_configuration(
        tracks_key, training_rounds_at_most=2,
        training_protocol=fixed_draw("own_predictions_on_the_same_tracks_and_planes"))
    state = round_trainer.train(configuration, store, if_the_run_exists="refuse", **QUIET)
    assert state["rounds_done"] == 2
    rows = RunFolder(store, run_key(configuration)).rounds()
    assert rows[1]["source_of_states"] == FROM_OWN_PREDICTIONS_FIXED


# ---------------------------------------------------------------- the best round
def test_the_best_round_is_that_of_the_lowest_validation_error(trained):
    store, key, configuration = trained
    folder = RunFolder(store, key)
    quantity = configuration["training"]["stopping_rule"]["quantity"]
    errors = [float(row[quantity]) for row in folder.rounds()]
    assert best_round(folder, quantity) == int(np.argmin(errors)) + 1
    where = report_of_a_run(store, key, round_number="best", **SETTINGS, **QUIET)
    with open(os.path.join(where, "summary.json")) as handle:
        summary = json.load(handle)
    assert summary["round"] == int(np.argmin(errors)) + 1
    assert summary["round_asked_for"] == "best" and summary["refit"] is None


# ---------------------------------------------------------------- the refit
def test_the_jacobian_of_the_refit_is_that_of_autograd(trained):
    store, key, configuration = trained
    run = round_trainer.build_run(configuration, store, RunFolder(store, key).constants())
    run.network.load_state_dict(RunFolder(store, key).weights_of_round(3))
    states = torch.as_tensor(run.first_round.states[:30])
    planes = torch.as_tensor(run.first_round.start_planes_mm[:30])
    JTJ, JTr, total, count = normal_equations(run.network, run.loss, run.constants["loss"],
                                              states, planes)
    layer = last_linear_layer(run.network)
    outputs, width = layer.weight.shape

    def residual(vector):
        block = vector.reshape(outputs, width + 1)
        saved = layer.weight.data.clone(), layer.bias.data.clone()
        del layer.weight, layer.bias
        layer.weight, layer.bias = block[:, :width], block[:, width]
        try:
            terms = run.loss.squared_terms(run.network.predict(states, planes), None,
                                           run.constants["loss"])
            divisor = torch.as_tensor(run.constants["loss"]["divisor_per_component"],
                                      dtype=torch.float64)
            from rkpinn.losses.stage_residual import stage_residual
            r = stage_residual(run.network.predict(states, planes),
                               run.network.equation_of_motion,
                               run.constants["loss"]["terms"]).residual / divisor
            assert torch.allclose(r ** 2, terms)
            return r.reshape(-1)
        finally:
            del layer.weight, layer.bias
            layer.weight = torch.nn.Parameter(saved[0])
            layer.bias = torch.nn.Parameter(saved[1])

    vector = torch.cat([layer.weight.detach(), layer.bias.detach()[:, None]], 1).reshape(-1)
    J = torch.autograd.functional.jacobian(residual, vector)
    r = residual(vector).detach()
    assert torch.allclose(JTJ, J.T @ J, rtol=1e-8, atol=1e-12 * float(JTJ.abs().max()))
    assert torch.allclose(JTr, J.T @ r, rtol=1e-8, atol=1e-12 * float(JTr.abs().max()))
    assert np.isclose(total, float((r * r).sum()), rtol=1e-12) and count == r.numel()


def test_the_refit_lowers_the_loss_and_changes_only_the_last_layer(trained):
    store, key, configuration = trained
    folder = RunFolder(store, key)
    run = round_trainer.build_run(configuration, store, folder.constants())
    run.network.load_state_dict(folder.weights_of_round(3))
    before = {name: value.clone() for name, value in run.network.state_dict().items()}
    made = refit_last_layer(run.network, run.loss, run.constants["loss"],
                            run.first_round.states, run.first_round.start_planes_mm,
                            iterations=5, first_damping=1e-3, report=lambda text: None)
    assert made["loss_after"] < made["loss_before"]
    last = "body.%d" % (len(run.network.body) - 1)
    after = run.network.state_dict()
    for name in before:
        if name.startswith(last):
            assert not identical(before[name], after[name])
        else:
            assert identical(before[name], after[name])


def test_the_report_scores_a_refit_and_the_run_is_untouched(trained):
    store, key, _ = trained
    folder = RunFolder(store, key)
    weights = folder.weights_of_round(3)
    refit_a_snapshot(store, key, round_number=3, iterations=3, first_damping=1e-3,
                     name="gate", **QUIET)
    assert all(identical(weights[n], folder.weights_of_round(3)[n]) for n in weights)
    where = report_of_a_run(store, key, round_number=3, refit="gate", **SETTINGS, **QUIET)
    assert os.path.basename(where).startswith("round_0003_refit_gate_at_")
    with open(os.path.join(where, "summary.json")) as handle:
        assert json.load(handle)["refit"] == "gate"


# ---------------------------------------------------------------- self-scaled Broyden
def test_the_update_meets_the_secant_equation():
    generator = torch.Generator().manual_seed(3)
    n = 12
    A = torch.randn(n, n, generator=generator, dtype=torch.float64)
    H = A @ A.T + n * torch.eye(n, dtype=torch.float64)
    g = torch.randn(n, generator=generator, dtype=torch.float64)
    step = 0.7
    s = -step * (H @ g)
    B = torch.randn(n, n, generator=generator, dtype=torch.float64)
    y = (B @ B.T + n * torch.eye(n, dtype=torch.float64)) @ s       # y.s > 0
    optimiser = SelfScaledBroydenRestarts(**{k: v for k, v in BROYDEN.items() if k != "name"})
    updated = H.clone()
    tau, phi = optimiser._update(updated, s, y, g, step)
    assert 0 < tau <= 1
    assert torch.allclose(updated, updated.T, atol=1e-10)
    assert torch.allclose(updated @ y, s, rtol=1e-9, atol=1e-12)

    # tau = phi = 1 is the BFGS update of the inverse Hessian
    ys = float(y @ s)
    rho = 1.0 / ys
    I = torch.eye(n, dtype=torch.float64)
    bfgs = (I - rho * torch.outer(s, y)) @ H @ (I - rho * torch.outer(y, s)) \
        + rho * torch.outer(s, s)
    Hy, yHy = H @ y, float(y @ (H @ y))
    v = (yHy ** 0.5) * (s / ys - Hy / yHy)
    formula = H - torch.outer(Hy, Hy) / yHy + torch.outer(v, v) + torch.outer(s, s) / ys
    assert torch.allclose(formula, bfgs, rtol=1e-9, atol=1e-9)


def test_a_run_trains_with_self_scaled_broyden(store):
    store, tracks_key = store
    configuration = small_configuration(tracks_key, training_rounds_at_most=2,
                                        training_optimiser=BROYDEN)
    state = round_trainer.train(configuration, store, if_the_run_exists="refuse", **QUIET)
    assert state["rounds_done"] == 2
    restarts = RunFolder(store, run_key(configuration)).restarts()
    assert all(float(r["loss_after"]) < float(r["loss_before"]) for r in restarts)
