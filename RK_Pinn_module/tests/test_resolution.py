"""Gate: output 11, the resolution against momentum.

Proves
  * the resolution is |predicted - reference| per component, in micrometres
    for positions and with no unit for slopes, and is never negative;
  * the table per band holds the median and the 16th and 84th percentiles of
    exactly the tracks of the p band, with the window of the loss as the last
    row;
  * the histogram against momentum counts every track with a resolution
    above zero, once, and its percentiles per bin are those of the tracks of
    the bin;
  * the momentum of each single step is that of its track, in the order the
    report lays the steps out;
  * the standard report holds the table for the endpoint error alone, and for
    the single-step error too when separate steps are given, and the figure
    against p is written.

The data are made up. The numbers say nothing about any network.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from rkpinn.evaluation import conventions, resolution
from rkpinn.evaluation.standard_report import (histograms_of_resolution, standard_report,
                                               write_figures, write_report)

NUMBER_OF_TRACKS = 400
STEPS = 3


@pytest.fixture(scope="module")
def made_up():
    draw = np.random.default_rng(11)
    reference = draw.normal(size=(NUMBER_OF_TRACKS, 5))
    reference[:, 2:4] *= 0.1
    signed = draw.normal(size=(NUMBER_OF_TRACKS, 4)) * np.array([1e-3, 2e-3, 1e-5, 2e-5])
    predicted = reference.copy()
    predicted[:, :4] += signed
    momentum = np.exp(draw.uniform(np.log(1.5), np.log(120.0), NUMBER_OF_TRACKS))
    return predicted, reference, signed, momentum


def test_the_resolution_is_the_absolute_error_in_the_units_of_the_conventions(made_up):
    predicted, reference, signed, _ = made_up
    r = resolution.resolution(predicted, reference)
    assert r["x"] == pytest.approx(np.abs(signed[:, 0]) * 1e3)
    assert r["y"] == pytest.approx(np.abs(signed[:, 1]) * 1e3)
    assert r["slope in x"] == pytest.approx(np.abs(signed[:, 2]))
    assert r["slope in y"] == pytest.approx(np.abs(signed[:, 3]))
    assert all((v >= 0).all() for v in r.values())


def test_the_table_per_band_holds_the_percentiles_of_the_tracks_of_the_band(made_up):
    predicted, reference, _, momentum = made_up
    window = (10.0, 50.0)
    rows = resolution.resolution_per_momentum_band(
        predicted, reference, momentum, resolution.ENDPOINT, window)
    bands = [r["band"] for r in rows if r["component"] == "x"]
    assert bands == [conventions.ALL_MOMENTA, *conventions.MOMENTUM_BANDS,
                     conventions.name_of_window(window)]
    assert {r["kind_of_error"] for r in rows} == {resolution.ENDPOINT}
    edges = conventions.EDGES_OF_MOMENTUM_BANDS_GEV
    r_x = np.abs(predicted[:, 0] - reference[:, 0]) * 1e3
    for low, high, name in [*zip(edges[:-1], edges[1:], conventions.MOMENTUM_BANDS),
                            (*window, conventions.name_of_window(window))]:
        row, = [r for r in rows if r["band"] == name and r["component"] == "x"]
        mine = r_x[(momentum >= low) & (momentum < high)]
        assert row["number_of_tracks"] == mine.size
        if mine.size:
            assert row["resolution_median"] == pytest.approx(np.median(mine))
            assert row["resolution_percentile_16"] == pytest.approx(np.quantile(mine, 0.16))
            assert row["resolution_percentile_84"] == pytest.approx(np.quantile(mine, 0.84))
        else:
            assert np.isnan(row["resolution_median"])
    assert sum(r["number_of_tracks"] for r in rows if r["component"] == "x"
               and r["band"] in conventions.MOMENTUM_BANDS) == NUMBER_OF_TRACKS


def test_the_histogram_counts_every_track_once_and_its_percentiles_are_of_the_bin(made_up):
    predicted, reference, _, momentum = made_up
    h = resolution.histograms_of_resolution(predicted, reference, momentum, 12)
    assert set(h) == {name for name, _, _ in conventions.COMPONENTS}
    for name, _, unit in conventions.COMPONENTS:
        mine = h[name]
        assert mine["unit"] == unit
        assert mine["counts"].shape == (12, 12)
        assert len(mine["momentum_edges_gev"]) == 13 and len(mine["resolution_edges"]) == 13
        # the edges span the whole range, so every track is in the histogram
        assert mine["counts"].sum() == NUMBER_OF_TRACKS
        assert mine["number_in_each_bin"].sum() == NUMBER_OF_TRACKS
        # the top edge belongs to the last bin
        which = np.minimum(np.digitize(momentum, mine["momentum_edges_gev"]) - 1, 11)
        r = resolution.resolution(predicted, reference)[name]
        for k in range(12):
            here = r[which == k]
            assert mine["number_in_each_bin"][k] == here.size
            if here.size:
                assert mine["median"][k] == pytest.approx(np.median(here))
                assert mine["percentile_16"][k] == pytest.approx(np.quantile(here, 0.16))
                assert mine["percentile_84"][k] == pytest.approx(np.quantile(here, 0.84))
            else:
                assert np.isnan(mine["median"][k])


def test_the_momentum_of_each_single_step_is_that_of_its_track():
    momentum = np.array([1.0, 2.0, 3.0])
    assert resolution.momentum_of_each_single_step(momentum, 6).tolist() == [
        1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    with pytest.raises(ValueError, match="whole number of steps"):
        resolution.momentum_of_each_single_step(momentum, 7)


def test_the_standard_report_holds_the_table_and_writes_the_figures(made_up, tmp_path):
    predicted, reference, _, momentum = made_up
    planes = np.array([0.0, 100.0, 200.0, 300.0])
    reference_on_planes = np.repeat(reference[:, None, :], STEPS + 1, axis=1)
    predicted_on_planes = reference_on_planes.copy()
    predicted_on_planes[:, -1] = predicted
    single_steps = dict(
        end_states=np.repeat(predicted[:, None, :], STEPS, axis=1).reshape(-1, 5),
        reference_after_the_step=np.repeat(reference[:, None, :], STEPS, axis=1).reshape(-1, 5),
        step_of_the_track=np.tile(np.arange(STEPS), NUMBER_OF_TRACKS),
        start_planes_mm=planes[:-1])
    shared = dict(name="3 steps of 100.0 mm, 2 stages", momentum_gev=momentum,
                  momentum_window_gev=(10.0, 50.0), states_on_planes=predicted_on_planes,
                  reference_states_on_planes=reference_on_planes, planes_mm=planes)

    with_steps = standard_report(single_steps=single_steps, **shared)
    rows = with_steps["tables"]["resolution_per_momentum_band"]
    assert {r["kind_of_error"] for r in rows} == set(resolution.KINDS_OF_ERROR)
    # 2 kinds x 4 components x (all momenta + 5 bands + window)
    assert len(rows) == 2 * 4 * 7
    assert not any(text.startswith("11") for text in with_steps["summary"]["outputs_left_out"])
    r_x = np.abs(predicted[:, 0] - reference[:, 0]) * 1e3
    assert with_steps["summary"]["endpoint_resolution_x_median"] == pytest.approx(np.median(r_x))
    assert with_steps["summary"]["single_step_resolution_x_median"] == pytest.approx(
        np.median(r_x))

    endpoint_only = standard_report(**shared)
    rows = endpoint_only["tables"]["resolution_per_momentum_band"]
    assert {r["kind_of_error"] for r in rows} == {resolution.ENDPOINT}
    assert len(rows) == 4 * 7
    assert any(text.startswith("11") for text in endpoint_only["summary"]["outputs_left_out"])

    folder = str(tmp_path / "report")
    written = write_report(with_steps, folder)
    assert os.path.join(folder, "resolution_per_momentum_band.csv") in written
    figures = write_figures(
        with_steps, folder,
        histograms_of_the_resolution=histograms_of_resolution(
            predicted, reference, momentum, single_steps, 10))
    assert os.path.join(folder, "resolution_against_momentum.png") in figures
    assert os.path.getsize(os.path.join(folder, "resolution_against_momentum.png")) > 5000
