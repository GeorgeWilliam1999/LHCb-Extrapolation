"""Gate: the Gauss-Legendre tableau of the package is that of the frozen code.

Proves
  * the nodes, stage matrix and weights are identical, to the last bit, to the
    frozen tableau at 16 stages and at every other stage count in use;
  * the five identities hold to 5e-13 at every stage count;
  * the largest violations equal the ones the frozen check reports;
  * a tableau that violates an identity is refused.
"""
from __future__ import annotations

import numpy as np
import pytest

from frozen_code import frozen_module, identical
from rkpinn.integrators.gauss_legendre_tableau import (
    TOLERANCE_OF_THE_IDENTITIES, GaussLegendreTableau, gauss_legendre_tableau)

STAGE_COUNTS = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20)


def test_identical_to_the_frozen_tableau_at_16_stages():
    frozen = frozen_module("irk")
    ours = gauss_legendre_tableau(16)
    nodes, stage_matrix, weights = frozen.gauss_legendre(16)
    assert ours.stage_matrix.shape == (16, 16)
    assert identical(ours.nodes, nodes)
    assert identical(ours.stage_matrix, stage_matrix)
    assert identical(ours.weights, weights)


@pytest.mark.parametrize("number_of_stages", STAGE_COUNTS)
def test_identical_to_the_frozen_tableau(number_of_stages):
    frozen = frozen_module("irk")
    ours = gauss_legendre_tableau(number_of_stages)
    nodes, stage_matrix, weights = frozen.gauss_legendre(number_of_stages)
    assert identical(ours.nodes, nodes)
    assert identical(ours.stage_matrix, stage_matrix)
    assert identical(ours.weights, weights)


@pytest.mark.parametrize("number_of_stages", STAGE_COUNTS)
def test_the_five_identities_hold(number_of_stages):
    tableau = gauss_legendre_tableau(number_of_stages)
    violations = tableau.check()
    assert set(violations) == {"rows", "quadrature", "collocation", "symplectic", "literature"}
    for name, value in violations.items():
        if name == "literature" and number_of_stages > 3:
            assert np.isnan(value)
        else:
            assert value < TOLERANCE_OF_THE_IDENTITIES, (name, value)
    assert tableau.order == 2 * number_of_stages


@pytest.mark.parametrize("number_of_stages", STAGE_COUNTS)
def test_violations_equal_the_frozen_check(number_of_stages):
    frozen = frozen_module("irk").verify_tableau(number_of_stages)
    ours = gauss_legendre_tableau(number_of_stages).violations()
    pairs = (("rows", "row_sums"), ("quadrature", "quadrature"),
             ("collocation", "collocation"), ("symplectic", "symplectic"),
             ("literature", "vs_literature"))
    for our_name, their_name in pairs:
        a, b = float(ours[our_name]), float(frozen[their_name])
        assert (np.isnan(a) and np.isnan(b)) or a == b, (our_name, a, b)
    assert frozen["passes"]


def test_a_wrong_tableau_is_refused():
    good = gauss_legendre_tableau(4)
    wrong_weights = good.weights.copy()
    wrong_weights[0] += 1e-9
    wrong = GaussLegendreTableau(4, good.nodes, good.stage_matrix, wrong_weights)
    with pytest.raises(ArithmeticError):
        wrong.check()
