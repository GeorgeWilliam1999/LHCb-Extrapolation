"""Gates of the polynomial correction (fastex/correction.py) and of its compiled
form (cpp/corrections/corrected_start.h).

    PYTHONNOUSERSITE=1 PYTHONPATH=<RK_Pinn_module>/src:<Fast_extrapolator>/python \\
        python -m pytest python/tests/test_correction.py
"""
from __future__ import annotations

import math
import shutil

import numpy as np
from numpy.polynomial import legendre as numpy_legendre

from fastex import correction
from fastex.compiled import run

Z0, Z1 = 2648.2, 7826.0


def some_states(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 5)) * np.array([150.0, 150.0, 0.06, 0.06, 0.05])


def test_the_number_of_terms():
    for degree in (1, 3, 6, 8):
        assert len(correction.terms_of(degree)) == math.comb(degree + 5, 5)


def test_the_legendre_table_is_that_of_numpy():
    u = np.linspace(-1.3, 1.3, 41)
    table = correction.legendre(u, 8)
    for k in range(9):
        expected = numpy_legendre.legval(u, [0.0] * k + [1.0])
        assert np.allclose(table[k], expected, rtol=0, atol=1e-12)


def test_the_fit_finds_a_polynomial_again():
    states = some_states(4000, 1)
    scale = correction.scale_of_inputs(states)
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(len(correction.terms_of(3)), 4)) * 1e-3
    target = correction.correction(states, truth, scale, 3)
    found, _ = correction.fit(states, target, 3, scale)
    assert np.abs(correction.correction(states, found, scale, 3) - target).max() < 1e-9


def test_the_correction_is_zero_for_a_straight_track():
    states = some_states(50, 3)
    states[:, 4] = 0.0
    scale = np.array([300.0, 300.0, 0.15, 0.15, 0.15])
    coefficients = np.random.default_rng(4).normal(size=(len(correction.terms_of(4)), 4))
    assert np.abs(correction.correction(states, coefficients, scale, 4)).max() == 0.0


def test_the_compiled_correction_is_the_numpy_one():
    """A random polynomial on the straight line: the compiled method in double
    precision against numpy, for a degree below and a degree above one vector."""
    states = some_states(300, 5)
    scale = np.array([400.0, 400.0, 0.2, 0.2, 0.2])
    for degree in (2, 6, 9):
        name = "gate_of_the_tests_degree_%d" % degree
        coefficients = np.random.default_rng(degree).normal(size=(len(correction.terms_of(degree)), 4))
        folder = correction.export(name, coefficients, scale, degree, start="straight_line", steps=0,
                                   reference="none", reader="rec", first_plane_mm=Z0, last_plane_mm=Z1)
        try:
            compiled, _ = run("corrected:" + name, states, Z0, Z1, reader="rec", precision="double")
        finally:
            shutil.rmtree(folder)
        length = Z1 - Z0
        line = np.stack([states[:, 0] + length * states[:, 2], states[:, 1] + length * states[:, 3],
                         states[:, 2], states[:, 3]], 1)
        expected = line + correction.correction(states, coefficients, scale, degree)
        assert np.abs(compiled[:, :4] - expected).max() < 1e-9 * max(1.0, np.abs(expected).max())
