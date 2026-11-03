"""Gate: the learned gain (C1a) and the error-conditioned stage network (C2),
added on 2026-11-03 for Part C of the experiment "does a Kalman-like gain
remove the bias".

Proves
  * the analytic gain of the package is that of the Part B script, to the last
    bit, on a toy step; the gain matrix times r is K r; the exact scheme's
    stages give K r = 0;
  * both networks have their shapes; a residual of zero gives a correction of
    zero, whatever the weights; with the last layer zero, as built, the
    learned gain gives the summed end state and the corrected stages are the
    first stages, to the last bit;
  * the distilled loss is zero when the correction is K r, is the formula
    when it is not, and is zero to rounding at the exact scheme; the loss on
    both calls is the pooled loss for a track of one call, and the mean of the
    two pooled losses for two;
  * a configuration of each is valid, keyed, read back to the same key, and
    trains in a temporary store: the loss falls, the frozen stage network
    does not move, a run stopped and resumed equals one not stopped, and the
    report holds the counts of field lookups, Jacobians and network calls.

The stage network both are built on is a small run trained in a temporary
store. The runs prove the mechanics, not what the networks reach.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest
import torch

from building_blocks import equations
from frozen_code import identical
from rkpinn.integrators import linearised_gain
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.losses.pooled import Pooled
from rkpinn.predicted_track.predicted_track import (END_STATE_PREDICTED, PredictedTrack,
                                                    steps_of_the_exact_scheme)
from rkpinn.run_record.configuration import read_configuration, run_key, write_configuration
from rkpinn.run_record.report_of_a_run import report_of_a_run
from rkpinn.training import round_trainer
from rkpinn.training.checkpoints import RunFolder
from small_run import small_configuration, small_store

QUIET = dict(allow_uncommitted_changes=True, report=lambda text: None)
REPORT = dict(split="test", tracks_for_the_stage_errors=4,
              step_length_of_the_carrying_mm=2.0, fraction_for_the_derivative=1e-3,
              states_for_the_loss_shares=200, bins_of_the_histograms=20)
COUNTS = ("field_lookups_per_step", "field_lookups_of_the_output_scales_per_step",
          "jacobian_evaluations_per_step", "linear_solves_per_step",
          "calls_of_the_stage_network_per_step", "calls_of_the_small_network_per_step",
          "jacobian_evaluations_per_step_in_training")


def gain_network(stage_key, **changes):
    block = {"kind": "stage_network_with_a_learned_gain",
             "stage_network_of_the_run": {"key": stage_key, "round": 2},
             "train_the_stage_network": False, "inputs_of_the_gain": "state_and_residual",
             "width": 8, "depth": 2, "activation": "tanh",
             "scale_of_inputs": "spread_of_first_round_states",
             "scale_of_outputs": "same_as_the_stage_network",
             "scale_of_the_residual_and_rates": "rms_over_first_round_states"}
    block.update(changes)
    return block


def corrector_network(stage_key, **changes):
    block = gain_network(stage_key, kind="error_conditioned_stage_network", **changes)
    block.pop("inputs_of_the_gain")
    return block


def c1a(tracks_key, stage_key, **changes):
    return small_configuration(tracks_key, network=gain_network(stage_key, **changes),
                               loss={"name": "distilled_analytic_gain",
                                     "with_the_stage_loss": False})


def c2(tracks_key, stage_key, **changes):
    return small_configuration(tracks_key, network=corrector_network(stage_key, **changes),
                               loss={"name": "pooled_on_both_calls", "terms": "stages"})


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """A small store with one trained stage network (straight-line form,
    per-track scale) of 2 rounds."""
    store, tracks_key = small_store(tmp_path_factory.mktemp("store_of_the_learned_gain"))
    stage = small_configuration(
        tracks_key, network={"kind": "stage_network",
                             "output_form": "straight_line_plus_scaled_correction",
                             "width": 16, "depth": 2, "activation": "tanh",
                             "end_state": "summed_from_the_stages",
                             "scale_of_inputs": "spread_of_first_round_states",
                             "scale_of_outputs": "per_track"},
        training_rounds_at_most=2)
    round_trainer.train(stage, store, if_the_run_exists="refuse", **QUIET)
    return store, tracks_key, run_key(stage)


def built(store, configuration):
    return round_trainer.build_run(configuration, store)


def moved(module, seed=3, size=0.3):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(size * torch.randn_like(p))


# ------------------------------------------------------------ the analytic gain
def gain_update_of_part_b(S0, z, Y, rates_and_jacobian, A, B, STEP_MM):
    """The arithmetic of `gain_update` of the Part B script
    (experiments/does_a_kalman_like_gain_remove_the_bias/analysis/part_b.py),
    copied line for line, for q = 2, one block of rows."""
    C = torch.as_tensor(gauss_legendre_tableau(2).nodes)
    f, J = rates_and_jacobian(Y, S0[:, 4:5], z[:, None] + C[None, :] * STEP_MM)
    with torch.no_grad():
        r = Y - S0[:, None, :4] - STEP_MM * torch.einsum("jk,nkd->njd", A, f)
        m = len(Y)
        M = torch.eye(8, dtype=torch.float64).repeat(m, 1, 1)
        for j in range(2):
            for k in range(2):
                M[:, 4 * j:4 * j + 4, 4 * k:4 * k + 4] -= STEP_MM * A[j, k] * J[:, k]
        w = torch.linalg.solve(M, r.reshape(m, 8, 1))[..., 0].reshape(m, 2, 4)
        Kr = STEP_MM * torch.einsum("k,mkab,mkb->ma", B, J, w)
        end = S0[:, :4] + STEP_MM * torch.einsum("k,mkd->md", B, f)
    return end, end - Kr, r


def toy_step(made, rows=64):
    store, tracks_key, stage_key = made
    run = built(store, c1a(tracks_key, stage_key))
    S0 = torch.as_tensor(run.first_round.states[:rows])
    z = torch.as_tensor(run.first_round.start_planes_mm[:rows])
    Y = run.network.stage_network(S0, z)[:, :2].detach()
    return run, S0, z, Y


def test_the_analytic_gain_is_that_of_part_b_to_the_last_bit(made):
    run, S0, z, Y = toy_step(made)
    tableau = run.layout.tableau
    A, B = torch.as_tensor(tableau.stage_matrix), torch.as_tensor(tableau.weights)
    equation = run.differentiable_equation
    end, corrected, Kr, r = linearised_gain.analytic_correction(
        equation, S0, Y, z, run.layout.step_length_mm, tableau)
    jacobian = lambda Y, qop, zs: linearised_gain.rates_and_jacobian(equation, Y, qop, zs)
    end_b, corrected_b, r_b = gain_update_of_part_b(S0, z, Y, jacobian, A, B,
                                                     run.layout.step_length_mm)
    assert identical(end.numpy(), end_b.numpy())
    assert identical(corrected.numpy(), corrected_b.numpy())
    assert identical(r.numpy(), r_b.numpy())
    assert float(Kr.abs().max()) > 0


def test_the_gain_matrix_times_the_residual_is_k_r(made):
    run, S0, z, Y = toy_step(made)
    tableau, dz = run.layout.tableau, run.layout.step_length_mm
    A, B = torch.as_tensor(tableau.stage_matrix), torch.as_tensor(tableau.weights)
    planes = z[:, None] + torch.as_tensor(tableau.nodes)[None, :] * dz
    f, J = linearised_gain.rates_and_jacobian(run.differentiable_equation, Y, S0[:, 4:5], planes)
    r = linearised_gain.residual_of_the_stages(S0, Y, f, dz, A)
    K = linearised_gain.gain_matrix(J, dz, A, B)
    assert K.shape == (len(S0), 4, 8)
    Kr = linearised_gain.gain_times_residual(J, r, dz, A, B)
    assert torch.allclose(torch.einsum("nam,nm->na", K, r.reshape(-1, 8)), Kr,
                          rtol=1e-10, atol=1e-18)


def test_the_exact_scheme_gives_no_correction(made):
    run, S0, z, _ = toy_step(made, rows=8)
    exact = steps_of_the_exact_scheme(run.equation_of_motion, S0.numpy(), z.numpy(),
                                      run.layout.step_length_mm, run.layout.tableau)
    end, corrected, Kr, r = linearised_gain.analytic_correction(
        run.differentiable_equation, S0, torch.as_tensor(exact.stage_states[:, 0]), z,
        run.layout.step_length_mm, run.layout.tableau)
    assert float(r[..., :2].abs().max()) < 1e-9 and float(Kr[:, :2].abs().max()) < 1e-9
    assert float(Kr[:, 2:].abs().max()) < 1e-12


# ------------------------------------------------------------ the learned gain, C1
def test_the_learned_gain_has_its_shapes_and_starts_as_the_stage_network(made):
    run, S0, z, _ = toy_step(made)
    network = run.network
    assert network.gain[0].in_features == 6 + 8 and network.gain[-1].out_features == 32
    track = network.predict(S0, z)
    assert track.end_states.shape == (len(S0), 1, 4)
    assert track.stage_states.shape == (len(S0), 1, 2, 4)
    assert track.end_state_was == END_STATE_PREDICTED
    plain = network.stage_network.predict(S0, z)
    assert identical(track.end_states.detach().numpy(), plain.end_states.detach().numpy())
    assert identical(track.stage_states.detach().numpy(), plain.stage_states.detach().numpy())
    with_rates = built(made[0], c1a(made[1], made[2],
                                    inputs_of_the_gain="state_residual_and_rates")).network
    assert with_rates.gain[0].in_features == 6 + 16


def test_a_zero_residual_gives_a_zero_correction_whatever_the_weights(made):
    run, S0, z, _ = toy_step(made)
    for configuration in (c1a, c2):
        network = built(made[0], configuration(made[1], made[2])).network
        moved(network)
        stages, rates, residual = network.first_call(S0, z)
        zero = torch.zeros_like(residual)
        if configuration is c1a:
            assert float(network.correction(S0, z, residual, rates).abs().max()) > 0
            assert identical(network.correction(S0, z, zero, rates).detach().numpy(),
                             np.zeros((len(S0), 4)))
        else:
            assert float(network.correction(S0, z, residual).abs().max()) > 0
            assert identical(network.correction(S0, z, zero).detach().numpy(),
                             np.zeros((len(S0), 2, 4)))


def track_with_end(network_track, end):
    return PredictedTrack(
        step_length_mm=network_track.step_length_mm, tableau=network_track.tableau,
        start_planes_mm=network_track.start_planes_mm,
        input_states=network_track.input_states, stage_states=network_track.stage_states,
        end_states=end[:, None], end_state_was=END_STATE_PREDICTED,
        stage_rates=network_track.stage_rates)


def test_the_distilled_loss_is_zero_at_k_r_and_is_its_formula(made):
    run, S0, z, Y = toy_step(made)
    loss, constants = run.loss, run.constants["loss"]
    track = run.network.predict(S0, z)
    end, corrected, Kr, r = linearised_gain.analytic_correction(
        run.differentiable_equation, S0, Y, z, run.layout.step_length_mm, run.layout.tableau)
    assert float(loss.value(track_with_end(track, corrected), None, constants)) < 1e-30
    shifted = corrected + torch.tensor([1e-3, -2e-3, 1e-6, 3e-6], dtype=torch.float64)
    divisor = np.asarray(constants["divisor_per_component"])
    wanted = np.mean(((end - shifted - Kr).numpy() / divisor) ** 2)
    assert np.isclose(float(loss.value(track_with_end(track, shifted), None, constants)),
                      wanted, rtol=1e-9)
    exact = steps_of_the_exact_scheme(run.equation_of_motion, S0[:8].numpy(), z[:8].numpy(),
                                      run.layout.step_length_mm, run.layout.tableau)
    assert float(loss.value(exact, None, constants)) < 1e-18
    with_stages = built(made[0], small_configuration(
        made[1], network=gain_network(made[2]),
        loss={"name": "distilled_analytic_gain", "with_the_stage_loss": True}))
    terms = with_stages.loss.squared_terms(track, None, with_stages.constants["loss"])
    assert terms.shape == (len(S0), 3, 4)


# ------------------------------------------------------------ the corrector, C2
def test_the_corrector_has_its_shapes_and_starts_as_the_stage_network(made):
    run = built(made[0], c2(made[1], made[2]))
    S0 = torch.as_tensor(run.first_round.states[:64])
    z = torch.as_tensor(run.first_round.start_planes_mm[:64])
    network = run.network
    assert network.corrector[0].in_features == 6 + 8
    assert network.corrector[-1].out_features == 64
    track = network.predict(S0, z)
    plain = network.stage_network.predict(S0, z)
    assert track.first_call is not None
    assert identical(track.stage_states.detach().numpy(), plain.stage_states.detach().numpy())
    assert identical(track.end_states.detach().numpy(), plain.end_states.detach().numpy())
    moved(network.corrector)
    track = network.predict(S0, z)
    assert not identical(track.stage_states.detach().numpy(),
                         track.first_call.stage_states.detach().numpy())


def test_the_loss_on_both_calls_is_the_mean_of_two_pooled_losses(made):
    run = built(made[0], c2(made[1], made[2]))
    S0 = torch.as_tensor(run.first_round.states[:64])
    z = torch.as_tensor(run.first_round.start_planes_mm[:64])
    moved(run.network.corrector)
    track = run.network.predict(S0, z)
    pooled = Pooled(equation_of_motion=run.differentiable_equation, terms="stages")
    constants = run.constants["loss"]
    second = pooled.value(track, None, constants)
    first = pooled.value(track.first_call, None, constants)
    assert torch.isclose(run.loss.value(track, None, constants), 0.5 * (first + second),
                         rtol=1e-12)
    one_call = run.network.stage_network.predict(S0, z)
    assert identical(run.loss.value(one_call, None, constants).detach().numpy(),
                     pooled.value(one_call, None, constants).detach().numpy())


# ------------------------------------------------------------ configurations and runs
@pytest.mark.parametrize("configuration", [c1a, c2])
def test_a_configuration_is_read_back_to_the_same_key(made, configuration, tmp_path):
    written = configuration(made[1], made[2])
    write_configuration(written, str(tmp_path / "run.yaml"))
    assert run_key(read_configuration(str(tmp_path / "run.yaml"))) == run_key(written)
    assert run_key(configuration(made[1], made[2], width=9)) != run_key(written)


@pytest.mark.parametrize("configuration", [c1a, c2])
def test_a_run_trains_resumes_and_reports_its_counts(made, configuration, tmp_path):
    store, tracks_key, stage_key = made
    run = configuration(tracks_key, stage_key)
    key = run_key(run)
    round_trainer.train(run, store, if_the_run_exists="refuse", rounds_in_this_call=1,
                        **QUIET)
    round_trainer.train(run, store, if_the_run_exists="resume", **QUIET)
    folder = RunFolder(store, key)
    rows = folder.restarts()
    assert float(rows[-1]["loss_after"]) < float(rows[0]["loss_before"])
    weights = folder.weights_of_round(3)
    for name, value in RunFolder(store, stage_key).weights_of_round(2).items():
        assert identical(weights["stage_network." + name].numpy(), value.numpy())
    small = "gain." if configuration is c1a else "corrector."
    assert any(float(v.abs().max()) > 0 for n, v in weights.items()
               if n.startswith(small) and n.endswith("weight"))

    # the same run, not stopped, in a second store with the same stage network
    other, other_tracks = small_store(tmp_path / "second_store")
    assert other_tracks == tracks_key
    stage = read_configuration(RunFolder(store, stage_key).path("configuration.yaml"))
    round_trainer.train(stage, other, if_the_run_exists="refuse", **QUIET)
    round_trainer.train(run, other, if_the_run_exists="refuse", **QUIET)
    not_stopped = RunFolder(other, key).weights_of_round(3)
    assert set(not_stopped) == set(weights)
    for name in weights:
        assert identical(weights[name].numpy(), not_stopped[name].numpy())

    where = report_of_a_run(store, key, round_number="last", **REPORT, **QUIET)
    with open(os.path.join(where, "cost.csv")) as handle:
        cost, = list(csv.DictReader(handle))
    for name in COUNTS:
        assert name in cost
    assert int(cost["field_lookups_per_step"]) == (2 if configuration is c1a else 4)
    assert int(cost["field_lookups_of_the_output_scales_per_step"]) == 16
    assert int(cost["calls_of_the_small_network_per_step"]) == 1
    assert int(cost["jacobian_evaluations_per_step"]) == 0
    assert int(cost["jacobian_evaluations_per_step_in_training"]) == (
        2 if configuration is c1a else 0)
