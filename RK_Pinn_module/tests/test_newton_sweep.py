"""Gate: one Newton sweep of the stage equations of a whole track.

Proves
  * from the exact scheme's states, solved again now, one sweep changes no
    stage state and no plane state beyond rounding (1e-11 mm, 1e-14 in slope),
    and the residual stays at rounding;
  * the forward-Euler path is millimetres off at the end of the region; one
    sweep from it brings the median end state within a micrometre of the exact
    scheme's (the worst of 20 tracks: 43 mm to 45 um), and a second sweep
    reaches the exact scheme's end state to 1 nm (worst 1.4e-7 mm) (measured 2026-11-04 on 60 test tracks: Euler 10.8 mm, one sweep 0.60
    um, two sweeps 0.1489 um, the exact scheme 0.1489 um, all as the median
    radial error against the reference);
  * a sweep costs N q field lookups and N q Jacobians per track, and refuses a
    track with no layout.

One sweep from the Euler path does not reach the floor on the real field.
The 0.146 um of the Kalman-like gain study is from a correction made step by
step along the chain, not from one sweep of a whole path.
"""
from __future__ import annotations

import numpy as np
import pytest

from building_blocks import equations
from rkpinn.evaluation.newton_sweep import forward_euler_path, one_sweep
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.predicted_track.predicted_track import track_of_the_exact_scheme
from rkpinn.predicted_track.track_layout import TrackLayout
from rkpinn.track_data.load_tracks import load_tracks
from the_store import KEY_OF_THE_TRACKS, the_store

NUMBER_OF_TRACKS = 20
POSITION_TOLERANCE_MM = 1e-11
SLOPE_TOLERANCE = 1e-14


@pytest.fixture(scope="module")
def cell():
    tracks = load_tracks(the_store(), KEY_OF_THE_TRACKS, check_the_key=False)
    layout = TrackLayout(tracks.first_plane_mm, tracks.last_plane_mm, 64,
                         gauss_legendre_tableau(2))
    start = tracks.test.start_state[:NUMBER_OF_TRACKS]
    exact = track_of_the_exact_scheme(equations()[0], start, layout)
    return layout, start, exact


def largest_differences(a, b):
    """The largest difference in position and in slope, over states (..., 4)."""
    d = np.abs(np.asarray(a)[..., :4] - np.asarray(b)[..., :4])
    return d[..., :2].max(), d[..., 2:].max()


def test_the_exact_scheme_is_left_as_it_is(cell):
    layout, _, exact = cell
    sweep = one_sweep(exact, equations()[1])
    for name in ("stage_states", "end_states"):
        position, slope = largest_differences(getattr(sweep.track, name), getattr(exact, name))
        assert position < POSITION_TOLERANCE_MM and slope < SLOPE_TOLERANCE, name
    assert sweep.largest_residual_before < 1e-11
    assert sweep.largest_residual_after < 1e-11
    assert sweep.field_lookups_per_track == sweep.jacobians_per_track == 64 * 2
    assert np.array_equal(sweep.track.input_states[:, 0], exact.input_states[:, 0])


def test_two_sweeps_from_the_euler_path_reach_the_exact_scheme(cell):
    layout, start, exact = cell
    _, differentiable = equations()
    euler = forward_euler_path(differentiable, start, layout)
    end = exact.final_state()[:, :2]
    off = np.abs(euler.final_state()[:, :2] - end).max()
    assert off > 1.0                                               # millimetres
    first = one_sweep(euler, differentiable)
    after_one = np.abs(first.track.final_state()[:, :2] - end).max()
    assert after_one < 1e-2 * off and after_one < 0.1               # under 100 um, worst track
    assert np.median(np.abs(first.track.final_state()[:, :2] - end)) < 1e-3   # 1 um
    second = one_sweep(first.track, differentiable)
    assert np.abs(second.track.final_state()[:, :2] - end).max() < 1e-6
    assert second.largest_residual_after < 1e-9
    assert first.largest_residual_after < first.largest_residual_before


def test_a_sweep_needs_a_layout(cell):
    layout, start, exact = cell
    exact.layout, kept = None, exact.layout
    try:
        with pytest.raises(ValueError):
            one_sweep(exact, equations()[1])
    finally:
        exact.layout = kept
