"""Gates of the Python copies of the LHCb integrators (fastex.lhcb_integrators).

The copies are the reference for the compiled copies, so they are checked on
their own:

  - the inverse fifth root of LHCbMath is accurate to 1e-8 over the range that
    the step control uses;
  - each integrator converges to the reference of the present package when its
    tolerance or its step falls;
  - with the default settings each integrator takes the number of field reads
    that its algorithm fixes (6 for a Cash-Karp step and 5 after a failed step;
    1 + 2 a step for the adaptive Nystrom; 1 a step for the fast Nystrom).

    PYTHONNOUSERSITE=1 PYTHONPATH=<present package>/src:<this folder>/.. \\
        python -m pytest python/tests/test_lhcb_integrators.py
"""
from __future__ import annotations

import numpy as np
import pytest

from fastex import STORE_OF_THE_PRESENT_RUNS, TRACK_SETS, lhcb_integrators
from fastex.export import flat_in_momentum

Z0, Z1 = 2648.2, 7826.0


@pytest.fixture(scope="module")
def tracks():
    from rkpinn import registry
    from rkpinn.run_record.store import Store
    from rkpinn.track_data.load_tracks import load_tracks
    loaded = load_tracks(Store(STORE_OF_THE_PRESENT_RUNS), TRACK_SETS["linear"])
    rows = flat_in_momentum(loaded.test.momentum_gev)[::25]          # 40 tracks
    field_map = registry.component("field_map", "v8r1_up")()
    return loaded.test.start_state[rows], loaded.test.reference_end_state[rows], field_map


def radial_um(finals, reference):
    return np.hypot(finals[:, 0] - reference[:, 0], finals[:, 1] - reference[:, 1]) * 1e3


def test_inverse_fifth_root():
    for x in np.logspace(-12, 6, 400):
        assert abs(lhcb_integrators.inverse_fifth_root(x) * x ** 0.2 - 1.0) < 1e-8


def test_the_rec_integrator_converges(tracks):
    start, reference, field_map = tracks
    default, _ = lhcb_integrators.carry_tracks("rec_cash_karp", start, Z0, Z1, field_map)
    fine, _ = lhcb_integrators.carry_tracks("rec_cash_karp", start, Z0, Z1, field_map,
                                            tolerance=1e-7, max_step=20.0, min_step=1.0)
    assert np.median(radial_um(fine, reference)) < 0.05
    assert np.median(radial_um(fine, reference)) < 0.01 * np.median(radial_um(default, reference))


def test_the_adaptive_nystrom_converges(tracks):
    start, reference, field_map = tracks
    default, _ = lhcb_integrators.carry_tracks("allen_adaptive_nystrom", start, Z0, Z1, field_map)
    fine, _ = lhcb_integrators.carry_tracks("allen_adaptive_nystrom", start, Z0, Z1, field_map,
                                            tolerance=1e-5, min_step=1.0, max_step=50.0)
    assert np.median(radial_um(fine, reference)) < 0.1 * np.median(radial_um(default, reference))


def test_the_fast_nystrom_converges(tracks):
    start, reference, field_map = tracks
    errors = [np.median(radial_um(lhcb_integrators.carry_tracks(
        "allen_fast_nystrom", start, Z0, Z1, field_map, step=step, max_steps=100000)[0],
        reference)) for step in (500.0, 100.0, 20.0)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 5.0


def test_the_counts_of_field_reads(tracks):
    start, _, field_map = tracks
    _, counts = lhcb_integrators.carry_tracks("rec_cash_karp", start, Z0, Z1, field_map)
    assert (counts[:, 0] == 6 * counts[:, 1] + 5 * counts[:, 2]).all()
    _, counts = lhcb_integrators.carry_tracks("allen_adaptive_nystrom", start, Z0, Z1, field_map)
    assert (counts[:, 0] == 1 + 2 * (counts[:, 1] + counts[:, 2])).all()
    _, counts = lhcb_integrators.carry_tracks("allen_fast_nystrom", start, Z0, Z1, field_map)
    assert (counts[:, 0] == 11).all() and (counts[:, 1] == 11).all()
