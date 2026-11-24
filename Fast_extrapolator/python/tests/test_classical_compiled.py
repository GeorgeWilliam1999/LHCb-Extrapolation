"""Gates of the classical integrators of the compiled program that the present
package does not have (cpp/classical/more_integrators.h).

  - `nystrom:N` and `sweeps:N:q:K` of the compiled program give the final
    states of their Python versions (fastex.classical) to 1e-9 mm, on both
    field maps;
  - with many sweeps, `sweeps` gives the exact collocation scheme of the
    present package (integrators/exact_collocation.py), which solves the same
    stage equations with a root finder.

The compiled program of this node is used (cpp/build/<node>/fx).

    PYTHONNOUSERSITE=1 PYTHONPATH=<present package>/src:python \\
        python -m pytest python/tests/test_classical_compiled.py
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from fastex import FOLDER_OF_EXPORTS, classical
from fastex.compiled import run

Z0, Z1 = 2648.2, 7826.0
MAPS = {"linear": "v8r1_up", "cubic": "v8r1_up_cubic"}


@pytest.fixture(scope="module")
def world():
    from rkpinn import registry
    found = {}
    for of_map, name in MAPS.items():
        folder = os.path.join(FOLDER_OF_EXPORTS, "tracks_" + of_map)
        start = np.fromfile(os.path.join(folder, "start_state.f64")).reshape(-1, 5)
        sample = np.fromfile(os.path.join(folder, "flat_sample.i64"), dtype=np.int64)[::5]
        found[of_map] = (start[sample], registry.component("field_map", name)())
    return found


@pytest.mark.parametrize("of_map", ["linear", "cubic"])
@pytest.mark.parametrize("steps", [6, 16])
def test_nystrom(world, of_map, steps):
    start, field_map = world[of_map]
    planes = Z0 + np.arange(steps + 1) * ((Z1 - Z0) / steps)
    python = classical.nystrom3(start, planes, field_map)
    compiled, counts = run("nystrom:%d" % steps, start, Z0, Z1, reader=of_map)
    assert np.abs(compiled - python)[:, :2].max() < 1e-9
    assert np.abs(compiled - python)[:, 2:4].max() < 1e-12
    assert (counts[:, 0] == 2 * steps + 1).all()


@pytest.mark.parametrize("of_map", ["linear", "cubic"])
@pytest.mark.parametrize("layout", [(8, 2, 3), (4, 4, 5)])
def test_sweeps(world, of_map, layout):
    from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
    steps, stages, sweeps = layout
    start, field_map = world[of_map]
    python = classical.stage_sweeps(start, Z0, Z1, steps, gauss_legendre_tableau(stages), sweeps,
                                    field_map)
    compiled, counts = run("sweeps:%d:%d:%d" % layout, start, Z0, Z1, reader=of_map)
    assert np.abs(compiled - python)[:, :2].max() < 1e-9
    assert np.abs(compiled - python)[:, 2:4].max() < 1e-12
    assert (counts[:, 0] == (sweeps + 1) * steps * stages).all()


def test_many_sweeps_give_the_exact_scheme(world):
    from rkpinn import registry
    from rkpinn.integrators.exact_collocation import solve_whole_track
    from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
    start, field_map = world["cubic"]
    start = start[:25]
    equation = registry.component("equation_of_motion", "lhcb")(field_map)
    exact = solve_whole_track(equation, start, Z0, Z1 - Z0, 16, gauss_legendre_tableau(2))
    assert exact.solves_not_converged == 0
    compiled, _ = run("sweeps:16:2:40", start, Z0, Z1, reader="cubic")
    assert np.abs(compiled[:, :4] - exact.states[:, -1, :4])[:, :2].max() < 1e-7
