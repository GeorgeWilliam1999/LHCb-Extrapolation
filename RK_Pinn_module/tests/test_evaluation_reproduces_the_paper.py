"""Gate: the evaluation, run on the stored states of the old runs, reproduces
the numbers of the second mini-paper.

This is the done-condition of the evaluation in `PACKAGE_PLAN.md`, section 9.

The old runs are six: 64 steps of 2 stages, 128 steps of 8 stages and 256
steps of 16 stages, each trained with the pooled loss and with the
cost-weighted loss. The tracks are the 1,452 test tracks of the store.

Proves, for every one of the six runs unless said, to the last digit the
paper's tables hold:

  outputs 1, 2, 3   the endpoint error per component and momentum band, all
                    five statistics; the radial error per band
  output 4          the single-step error per step, for the two runs of 64
                    steps, with the old network applied by the frozen code
  output 5          the error along the track, on every plane
  output 7          the three references: the prediction against the
                    reference, the prediction against the true state, the
                    reference against the true state, per band
  output 8          convergence: the headline of the validation error, the
                    verdict of the rule and the first round it held
  output 9          where the loss puts its weight, for the pooled and the
                    cost-weighted loss on the old network of 64 steps, the
                    draw of the paper played again

and that the conventions are those of the paper: the bands divide the tracks
with none left out and none counted twice, and slopes carry no factor.

Outputs 6 and 10 have no number in the paper. Output 6 has its own gate.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from frozen_code import frozen_chain_network, frozen_module, frozen_weighted_loss
from old_runs import (
    BAND_OF_THE_PAPER, COMPONENT_OF_THE_PAPER, OLD_RUNS, STATISTIC_OF_THE_PAPER,
    WINDOW_OF_THE_PAPER_GEV, folder_of_run, rounds_of_run, states_of_run,
    table_of_the_paper)
from rkpinn.equation_of_motion.field_map import load_field_map
from rkpinn.equation_of_motion.lhcb import LhcbEquationOfMotion
from rkpinn.evaluation import conventions, convergence, endpoint_error
from rkpinn.evaluation.error_along_the_track import error_along_the_track, single_step_error
from rkpinn.evaluation.standard_report import standard_report, where_the_loss_puts_its_weight
from rkpinn.evaluation.three_references import carry_to_own_planes, three_references
from rkpinn.integrators.runge_kutta_sixth_order import REFERENCE_STEP_LENGTH_MM
from rkpinn.track_data.load_tracks import load_tracks
from the_store import KEY_OF_THE_TRACKS, the_store

WINDOW = WINDOW_OF_THE_PAPER_GEV


@pytest.fixture(scope="module")
def tracks():
    return load_tracks(the_store(), KEY_OF_THE_TRACKS, check_the_key=False)


def reference_on_planes_of(tracks, number_of_steps):
    on = np.append(tracks.planes_a_step_starts_on(number_of_steps),
                   tracks.number_of_steps_between_planes)
    return tracks.test.reference_states_on_planes[:, on]


# ------------------------------------------------------------------ the conventions
def test_the_bands_divide_the_tracks(tracks):
    momentum = tracks.test.momentum_gev
    bands = conventions.momentum_bands(momentum, WINDOW)
    assert [name for name, _ in bands] == list(conventions.MOMENTUM_BANDS) + [
        "10 to 50 GeV (window of the loss)"]
    counted = np.sum([mask for _, mask in bands[:5]], axis=0)
    assert (counted == 1).all()
    paper = {r["band"]: int(r["n"]) for r in table_of_the_paper("tab_headline_bands", 64, 2,
                                                                "pooled")}
    for name, mask in bands:
        assert int(mask.sum()) == paper[{v: k for k, v in BAND_OF_THE_PAPER.items()}[name]]
    assert len(conventions.momentum_bands(momentum)) == 5
    assert conventions.momentum_bands(momentum, with_all=True)[0][1].all()


def test_slopes_carry_no_factor_and_positions_are_in_micrometres():
    predicted = np.array([[1.001, 2.0, 0.1000007, 0.2, 0.01]])
    reference = np.array([[1.000, 2.0005, 0.1, 0.2, 0.01]])
    e = conventions.errors(predicted, reference)
    assert e["x"][0] == pytest.approx(1.0) and e["y"][0] == pytest.approx(-0.5)
    assert e["slope in x"][0] == pytest.approx(7e-7, rel=1e-6)
    assert e["radial"][0] == pytest.approx(np.hypot(1.0, 0.5))
    assert e["larger_of_x_and_y"][0] == pytest.approx(1.0)
    assert [unit for _, _, unit in conventions.COMPONENTS] == [
        "micrometres", "micrometres", "no unit", "no unit"]
    assert conventions.name_of_network(64, 2, 80.903125) == "64 steps of 80.9 mm, 2 stages"


# ------------------------------------------------------------- outputs 1, 2 and 3
@pytest.mark.parametrize("number_of_steps, number_of_stages, loss", OLD_RUNS)
def test_endpoint_error_per_component_and_band(tracks, number_of_steps, number_of_stages,
                                               loss):
    end = states_of_run(number_of_steps, number_of_stages, loss)[:, -1]
    ours = {(r["band"], r["component"]): r for r in endpoint_error.errors_per_component(
        end, tracks.test.reference_end_state, tracks.test.momentum_gev, WINDOW)}
    paper = table_of_the_paper("tab_headline_components", number_of_steps,
                               number_of_stages, loss)
    assert len(paper) == len(ours) == 28
    for row in paper:
        mine = ours[(BAND_OF_THE_PAPER[row["band"]], COMPONENT_OF_THE_PAPER[row["component"]])]
        assert mine["number_of_tracks"] == int(row["n"])
        assert mine["unit"] == {"um": "micrometres", "slope": "no unit"}[row["unit"]]
        for theirs, name in STATISTIC_OF_THE_PAPER.items():
            assert mine[name] == float(row[theirs]), (row["band"], row["component"], name)


@pytest.mark.parametrize("number_of_steps, number_of_stages, loss", OLD_RUNS)
def test_radial_endpoint_error_per_band(tracks, number_of_steps, number_of_stages, loss):
    end = states_of_run(number_of_steps, number_of_stages, loss)[:, -1]
    ours = {(r["band"], r["magnitude"]): r
            for r in endpoint_error.magnitudes_of_position_error(
                end, tracks.test.reference_end_state, tracks.test.momentum_gev, WINDOW)}
    for row in table_of_the_paper("tab_headline_bands", number_of_steps, number_of_stages,
                                  loss):
        mine = ours[(BAND_OF_THE_PAPER[row["band"]], "radial")]
        assert mine["median_micrometres"] == float(row["radial_med_um"])
        assert mine["percentile_95_micrometres"] == float(row["radial_p95_um"])
        assert mine["fraction_above_1_mm"] == float(row["frac_above_1mm"])
    headline, = table_of_the_paper("tab_headline_pairs", number_of_steps, number_of_stages,
                                   loss)
    overall = ours[("all momenta", "radial")]
    assert overall["median_micrometres"] == float(headline["radial_med_um"])
    assert overall["number_above_1_mm"] == int(headline["n_above_1mm"])
    assert ours[("10 to 50 GeV (window of the loss)", "radial")]["median_micrometres"] \
        == float(headline["band_radial_med_um"])


def test_signed_distributions_agree_with_the_statistics(tracks):
    end = states_of_run(64, 2, "pooled")[:, -1]
    reference, momentum = tracks.test.reference_end_state, tracks.test.momentum_gev
    quantiles = {(r["band"], r["component"]): r for r in endpoint_error.signed_distributions(
        end, reference, momentum, WINDOW)}
    statistics = endpoint_error.errors_per_component(end, reference, momentum, WINDOW)
    for row in statistics:
        q = quantiles[(row["band"], row["component"])]
        assert q["percentile_50"] == row["signed_median"]
        assert (q["percentile_84"] - q["percentile_16"]) / 2.0 == pytest.approx(
            row["half_width_68"], rel=1e-14)
        assert q["percentile_01"] <= q["percentile_16"] <= q["percentile_99"]
    histograms = endpoint_error.histograms_of_signed_error(end, reference, 40)
    assert histograms["x"]["counts"].sum() <= len(end)
    assert len(histograms["slope in y"]["edges"]) == 41


# ------------------------------------------------------------------------ output 5
@pytest.mark.parametrize("number_of_steps, number_of_stages, loss", OLD_RUNS)
def test_error_along_the_track(tracks, number_of_steps, number_of_stages, loss):
    states = states_of_run(number_of_steps, number_of_stages, loss)
    step_length = tracks.length_mm / number_of_steps
    planes = tracks.first_plane_mm + np.arange(number_of_steps + 1) * step_length
    ours = error_along_the_track(states, reference_on_planes_of(tracks, number_of_steps),
                                 planes, tracks.test.momentum_gev, WINDOW)
    paper = table_of_the_paper("tab_along_z", number_of_steps, number_of_stages, loss)
    assert len(ours) == len(paper) == number_of_steps + 1
    for mine, row in zip(ours, paper):
        assert mine["plane"] == int(row["plane"])
        assert mine["z_mm"] == float(row["z_mm"])
        assert mine["radial_median_micrometres"] == float(row["chain_radial_med_um"])
        assert mine["radial_percentile_95_micrometres"] == float(row["chain_radial_p95_um"])
        assert mine["radial_median_in_the_window_micrometres"] \
            == float(row["chain_radial_band_med_um"])
    assert ours[0]["radial_median_micrometres"] == 0.0


# ------------------------------------------------------------------------ output 4
@pytest.mark.parametrize("loss", ("pooled", "reweighted"))
def test_single_step_error(tracks, loss):
    number_of_steps, number_of_stages = 64, 2
    frozen = frozen_chain_network()
    field = frozen_module("reference").make_field("up")
    model = frozen.load_network(folder_of_run(number_of_steps, number_of_stages, loss), field)
    reference = reference_on_planes_of(tracks, number_of_steps)
    n = len(reference)
    states = reference[:, :number_of_steps].reshape(-1, 5)
    after = reference[:, 1:].reshape(-1, 5)
    step = np.repeat(np.arange(number_of_steps)[None, :], n, axis=0).reshape(-1)
    start_planes = tracks.start_planes_mm(number_of_steps)
    ends = frozen.step_outputs(model, states, tracks.first_plane_mm
                               + step * (tracks.length_mm / number_of_steps))[:, -1, :]
    ours = single_step_error(ends, after, step, start_planes)
    paper = table_of_the_paper("tab_single_step_vs_z", number_of_steps, number_of_stages,
                               loss)
    assert len(ours["per_step"]) == len(paper) == number_of_steps
    for mine, row in zip(ours["per_step"], paper):
        assert mine["step"] == int(row["plane"])
        assert mine["z_of_start_plane_mm"] == pytest.approx(float(row["z_mm"]), abs=1e-9)
        assert mine["radial_median_micrometres"] == float(row["single_step_radial_med_um"])
    assert ours["radial"]["number_of_tracks"] == n * number_of_steps
    assert [r["component"] for r in ours["per_component"]] == [
        "x", "y", "slope in x", "slope in y"]


# ------------------------------------------------------------------------ output 7
@pytest.mark.parametrize("number_of_steps, number_of_stages, loss", OLD_RUNS)
def test_three_references(tracks, number_of_steps, number_of_stages, loss):
    test = tracks.test
    end = states_of_run(number_of_steps, number_of_stages, loss)[:, -1]
    equation = LhcbEquationOfMotion(load_field_map("v8r1_up"))
    carried = carry_to_own_planes(
        equation, end, tracks.last_plane_mm, test.own_fibre_plane_mm,
        step_length_of_the_integrator_mm=REFERENCE_STEP_LENGTH_MM)
    ours = {(r["band"], r["comparison"]): r for r in three_references(
        predicted_end_states=end, reference_end_states=test.reference_end_state,
        predicted_on_own_planes=carried,
        reference_on_own_planes=test.reference_state_on_own_fibre_plane,
        true_states_on_own_planes=test.true_state_on_own_fibre_plane,
        momentum_gev=test.momentum_gev, momentum_window_gev=WINDOW)}
    names = {"nn_vs_rk6": "prediction against reference",
             "nn_vs_true": "prediction against true state",
             "rk6_vs_true": "reference against true state"}
    paper = table_of_the_paper("tab_against_true_state", number_of_steps, number_of_stages,
                               loss)
    assert len(paper) == 7
    for row in paper:
        for theirs, comparison in names.items():
            mine = ours[(BAND_OF_THE_PAPER[row["band"]], comparison)]
            assert mine["number_of_tracks"] == int(row["n"])
            assert mine["position_larger_of_x_and_y_median_micrometres"] \
                == float(row[theirs + "_pos_max_med_um"])
            assert mine["position_larger_of_x_and_y_percentile_95_micrometres"] \
                == float(row[theirs + "_pos_max_p95_um"])
            assert mine["slope_larger_of_the_two_median"] \
                == float(row[theirs + "_slope_max_med"])


def test_three_references_with_the_exact_scheme(tracks):
    from rkpinn.track_data.exact_states import load_exact_states
    test = tracks.test
    exact = load_exact_states(the_store(), KEY_OF_THE_TRACKS, "test", 64, 2)[0][:, -1]
    end = states_of_run(64, 2, "pooled")[:, -1]
    rows = {(r["band"], r["comparison"]): r for r in three_references(
        predicted_end_states=end, reference_end_states=test.reference_end_state,
        predicted_on_own_planes=end, reference_on_own_planes=end,
        true_states_on_own_planes=end, momentum_gev=test.momentum_gev,
        exact_end_states=exact)}
    ceiling = rows[("all momenta", "exact scheme against reference")]
    against = rows[("all momenta", "prediction against reference")]
    assert 0 < ceiling["position_radial_median_micrometres"] \
        < against["position_radial_median_micrometres"]
    assert ("all momenta", "prediction against exact scheme") in rows


# ------------------------------------------------------------------------ output 8
@pytest.mark.parametrize("number_of_steps, number_of_stages, loss", OLD_RUNS)
def test_convergence(number_of_steps, number_of_stages, loss):
    rounds = rounds_of_run(number_of_steps, number_of_stages, loss)
    errors = [float(row["val_z1_pos_med_um"]) for row in rounds]
    ours = convergence.convergence(errors, window_in_rounds=10, tolerance=0.05,
                                   rounds_held=3)
    paper, = table_of_the_paper("tab_headline_pairs", number_of_steps, number_of_stages,
                                loss)
    assert ours["rounds"] == int(paper["rounds"])
    assert ours["plateaued_now"] == bool(int(paper["plateaued_now"]))
    assert (ours["first_round_the_rule_held"] or -1) == int(paper["first_plateau_round"])
    assert ours["median_of_last_window_micrometres"] == float(paper["val_headline_med_um"])
    assert ours["spread_percent"] == float(paper["val_headline_spread_pct"])
    assert ours["full_range_over_median_percent"] \
        == float(paper["val_headline_full_range_over_median_pct"])


# ------------------------------------------------------------------------ output 9
def test_where_the_loss_puts_its_weight(tracks):
    """The draw of the paper, played again on the old network of 64 steps and
    2 stages, with the frozen residual and the frozen weights."""
    frozen_model = frozen_module("model")
    reference = frozen_module("reference")
    chain = frozen_chain_network()
    frozen_weights = frozen_weighted_loss()
    number_of_steps, number_of_stages = 64, 2
    on_planes = tracks.training.reference_states_on_planes
    finest = tracks.number_of_steps_between_planes
    field = reference.make_field("up")

    def draw(steps, number, generator, even=False):
        stride = finest // steps
        if even:
            per = max(1, number // steps)
            track = generator.integers(0, len(on_planes), per * steps)
            k = np.repeat(np.arange(steps), per)
        else:
            track = generator.integers(0, len(on_planes), number)
            k = generator.integers(0, steps, number)
        return on_planes[track, k * stride], k

    generator = np.random.default_rng(20260918)
    for number in (2000, 500, 40):
        for steps in (2, 256):
            draw(steps, number, generator)
    states, step = draw(number_of_steps, 8000, generator, even=True)

    step_length = tracks.length_mm / number_of_steps
    start_planes = tracks.first_plane_mm + step * step_length
    model = chain.load_network(folder_of_run(number_of_steps, number_of_stages, "pooled"),
                               field)
    St, zt = torch.as_tensor(states), torch.as_tensor(start_planes)
    c, A, b = reference.gauss_legendre(number_of_stages)
    rates = frozen_model.LHCbRates(field)
    with torch.no_grad():
        residual = (frozen_model.reconstruction_residuals(
            model, rates, St, step_length, chain.znodes_for(model, zt), torch.tensor(A),
            torch.tensor(b), zt) * model.in_scale[:4]).numpy()
    constants = frozen_weights.reference_constants(
        model, states, start_planes, tracks.last_plane_mm,
        frozen_weights.i_bar(field, tracks.first_plane_mm, tracks.last_plane_mm),
        tracks.length_mm)
    momentum = frozen_weights.QOP_TO_GEV / np.abs(states[:, 4])
    paper = table_of_the_paper("tab_preflight_shares")
    names = dict(BAND_OF_THE_PAPER, **{"quarter %d" % j: "quarter %d" % j
                                       for j in (1, 2, 3, 4)})
    for mode in ("blockE", "full"):
        w = frozen_weights.weights(model, St, zt, constants, mode)
        w = w.detach().numpy() if hasattr(w, "detach") else np.asarray(w)
        of_each_state = ((residual * w) ** 2).mean(axis=(1, 2))
        ours = {r["where"]: r for r in where_the_loss_puts_its_weight(
            of_each_state, momentum, step, number_of_steps, WINDOW)}
        rows = [r for r in paper if r["mode"] == mode]
        assert len(rows) == 10
        for row in rows:
            mine = ours[names[row["key"]]]
            assert mine["number_of_states"] == int(row["n"])
            assert mine["share_of_the_loss_percent"] == pytest.approx(
                float(row["share_pct"]), rel=1e-12)
            assert mine["share_of_the_states_percent"] == pytest.approx(
                float(row["states_pct"]), rel=1e-12)


# ----------------------------------------------------------------- the whole report
def test_the_report_of_an_old_run_says_what_it_left_out(tracks):
    number_of_steps = 64
    states = states_of_run(number_of_steps, 2, "reweighted")
    planes = tracks.first_plane_mm + np.arange(number_of_steps + 1) * (
        tracks.length_mm / number_of_steps)
    report = standard_report(
        name=conventions.name_of_network(number_of_steps, 2, planes[1] - planes[0]),
        momentum_gev=tracks.test.momentum_gev, momentum_window_gev=WINDOW,
        states_on_planes=states,
        reference_states_on_planes=reference_on_planes_of(tracks, number_of_steps),
        planes_mm=planes)
    assert set(report["tables"]) == {
        "endpoint_error_per_component", "endpoint_error_magnitudes",
        "signed_distributions", "error_along_the_track"}
    left_out = report["summary"]["outputs_left_out"]
    assert [text.split(",")[0] for text in left_out] == ["4", "6", "7", "8", "9", "10"]
    headline, = table_of_the_paper("tab_headline_pairs", number_of_steps, 2, "reweighted")
    assert report["summary"]["endpoint_radial_median_micrometres"] \
        == float(headline["radial_med_um"])
    assert report["summary"]["name"] == "64 steps of 80.9 mm, 2 stages"
