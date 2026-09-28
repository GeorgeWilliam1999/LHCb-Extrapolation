"""The Gauss-Legendre tableau for any number of stages.

A Runge-Kutta method with q stages advances a state S_0 by a step dz through

    S_j   = S_0 + dz * sum_k A_jk f(S_k, z_0 + c_k dz),      j = 1 .. q
    S_end = S_0 + dz * sum_k b_k  f(S_k, z_0 + c_k dz),

where the S_j are the stage states, the c_k are the nodes that place the stage
planes inside the step, A is the stage matrix and the b_k are the weights. The
three together are the tableau.

In the Gauss-Legendre family the nodes are the roots of the Legendre
polynomial of degree q, mapped onto the interval from 0 to 1. The matrix A is
full, so the method is implicit. Its order is 2q.

The tableau is built, not looked up: the nodes and weights come from the
Legendre roots, and A from the integrals of the Lagrange basis polynomials
from 0 to each node. It is checked against five identities every time it is
built.

Ported from `_shared/irk.py` (`tableau`, `verify_tableau`, `gauss_legendre`)
with the arithmetic unchanged, so the tableau is identical to the last bit.
Gates: tests/test_gauss_legendre_tableau_matches_the_frozen_code.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rkpinn.registry import register

# The largest violation of an identity that is accepted.
TOLERANCE_OF_THE_IDENTITIES = 5e-13

# The closed forms for 1, 2 and 3 stages (Hairer and Wanner, Solving Ordinary
# Differential Equations II, table IV.5.2). Used only to check the construction.
_ROOT_3, _ROOT_15 = np.sqrt(3.0), np.sqrt(15.0)
TABLEAUX_FROM_THE_LITERATURE = {
    1: (np.array([0.5]), np.array([[0.5]]), np.array([1.0])),
    2: (np.array([0.5 - _ROOT_3 / 6, 0.5 + _ROOT_3 / 6]),
        np.array([[0.25, 0.25 - _ROOT_3 / 6],
                  [0.25 + _ROOT_3 / 6, 0.25]]),
        np.array([0.5, 0.5])),
    3: (np.array([0.5 - _ROOT_15 / 10, 0.5, 0.5 + _ROOT_15 / 10]),
        np.array([[5 / 36, 2 / 9 - _ROOT_15 / 15, 5 / 36 - _ROOT_15 / 30],
                  [5 / 36 + _ROOT_15 / 24, 2 / 9, 5 / 36 - _ROOT_15 / 24],
                  [5 / 36 + _ROOT_15 / 30, 2 / 9 + _ROOT_15 / 15, 5 / 36]]),
        np.array([5 / 18, 4 / 9, 5 / 18])),
}


def _barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    q = len(nodes)
    w = np.empty(q)
    for j in range(q):
        w[j] = 1.0 / np.prod(nodes[j] - np.delete(nodes, j))
    return w


def _lagrange_basis(x: np.ndarray, nodes: np.ndarray, w: np.ndarray) -> np.ndarray:
    """L[i, k] is the k-th Lagrange basis polynomial on the nodes, at x[i].

    The barycentric form is well conditioned where solving a Vandermonde
    system is not. A point that falls exactly on a node is handled.
    """
    d = x[:, None] - nodes[None, :]
    hit = np.isclose(d, 0.0, atol=1e-14)
    d = np.where(hit, 1.0, d)
    tmp = w[None, :] / d
    L = tmp / tmp.sum(axis=1, keepdims=True)
    rows = hit.any(axis=1)
    L[rows] = hit[rows].astype(float)
    return L


def build_nodes_stage_matrix_and_weights(number_of_stages: int):
    """The nodes c, the stage matrix A and the weights b, unchecked."""
    q = number_of_stages
    x, wq = np.polynomial.legendre.leggauss(q)
    c, b = (x + 1.0) / 2.0, wq / 2.0

    wbar = _barycentric_weights(c)
    A = np.empty((q, q))
    for j in range(q):
        tau = (x + 1.0) * (c[j] / 2.0)             # the same rule, on [0, c_j]
        wtau = wq * (c[j] / 2.0)
        A[j] = wtau @ _lagrange_basis(tau, c, wbar)
    return c, A, b


def violations_of_the_identities(nodes, stage_matrix, weights) -> dict:
    """The largest violation of each of the five identities.

    - rows:         each row of A sums to its node;
    - quadrature:   the weights integrate polynomials exactly to degree 2q - 1;
    - collocation:  A satisfies the collocation conditions to degree q - 1;
    - symplectic:   b_j A_jk + b_k A_kj = b_j b_k, which the construction
                    never imposed;
    - literature:   for 1, 2 and 3 stages, agreement with the closed forms
                    (not a number for other stage counts).
    """
    c, A, b = nodes, stage_matrix, weights
    q = len(c)
    rows = np.abs(A.sum(axis=1) - c).max()
    ms = np.arange(1, 2 * q + 1)
    quadrature = np.abs([(b * c ** (m - 1)).sum() - 1.0 / m for m in ms]).max()
    collocation = max(np.abs(A @ c ** (m - 1) - c ** m / m).max()
                      for m in range(1, q + 1))
    symplectic = np.abs(b[:, None] * A + (b[:, None] * A).T
                        - np.outer(b, b)).max()
    literature = np.nan
    if q in TABLEAUX_FROM_THE_LITERATURE:
        ck, Ak, bk = TABLEAUX_FROM_THE_LITERATURE[q]
        literature = max(np.abs(c - ck).max(), np.abs(A - Ak).max(),
                         np.abs(b - bk).max())
    return dict(rows=rows, quadrature=quadrature, collocation=collocation,
                symplectic=symplectic, literature=literature)


@dataclass(frozen=True)
class GaussLegendreTableau:
    """The nodes, stage matrix and weights of one Gauss-Legendre method."""

    number_of_stages: int
    nodes: np.ndarray          # c, shape (q,)
    stage_matrix: np.ndarray   # A, shape (q, q)
    weights: np.ndarray        # b, shape (q,)

    @property
    def order(self) -> int:
        return 2 * self.number_of_stages

    def violations(self) -> dict:
        return violations_of_the_identities(self.nodes, self.stage_matrix, self.weights)

    def check(self, tolerance: float = TOLERANCE_OF_THE_IDENTITIES) -> dict:
        """Check the five identities. Raises if any is violated."""
        violations = self.violations()
        wrong = {name: value for name, value in violations.items()
                 if not np.isnan(value) and not value < tolerance}
        if wrong:
            raise ArithmeticError(
                "the Gauss-Legendre tableau with %d stages violates %r (tolerance %g)"
                % (self.number_of_stages, wrong, tolerance))
        return violations


@register("tableau", "gauss_legendre")
def gauss_legendre_tableau(number_of_stages: int) -> GaussLegendreTableau:
    """The Gauss-Legendre tableau with this many stages, checked as it is built."""
    number_of_stages = int(number_of_stages)
    if number_of_stages < 1:
        raise ValueError("a tableau needs at least one stage")
    nodes, stage_matrix, weights = build_nodes_stage_matrix_and_weights(number_of_stages)
    tableau = GaussLegendreTableau(number_of_stages, nodes, stage_matrix, weights)
    tableau.check()
    return tableau
