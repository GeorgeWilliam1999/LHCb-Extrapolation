"""Gate: the supervised path loss.

Proves
  * with no uncertainty, the value on a known pair is the mean over states and
    components of the squared error over the divisor, computed by hand;
  * with the diagonal uncertainty, the loss at the right sigma is lower than at
    a sigma half or twice as large, and equals the formula by hand; the bounded
    log-sigma keeps the loss above -2B;
  * `squared_terms` has one term per state and component, and the value is
    their mean;
  * the diagonal uncertainty without extra outputs, a wrong divisor and an
    unknown uncertainty are refused.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rkpinn.losses.supervised_path import BOUND_OF_THE_LOG_SIGMA, SupervisedPath
from rkpinn.predicted_track.predicted_track import END_STATE_PREDICTED, PredictedTrack

N, STEPS, STAGES = 5, 3, 2
DIVISOR = [2.0, 4.0, 0.5, 0.25]


def track(emitted, extra=None):
    """A whole track whose stages and end states are `emitted` (n, steps, q + 1, 4)."""
    emitted = torch.as_tensor(emitted)
    inputs = torch.zeros((len(emitted), STEPS, 5), dtype=torch.float64)
    return PredictedTrack(
        step_length_mm=1.0, tableau=_tableau(), start_planes_mm=torch.zeros((len(emitted), STEPS)),
        input_states=inputs, stage_states=emitted[:, :, :STAGES], end_states=emitted[:, :, STAGES],
        end_state_was=END_STATE_PREDICTED, extra_outputs=extra)


def _tableau():
    from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
    return gauss_legendre_tableau(STAGES)


@pytest.fixture
def pair():
    generator = np.random.default_rng(1)
    labels = generator.normal(size=(N, STEPS, STAGES + 1, 4))
    error = generator.normal(size=labels.shape)
    return labels, error


def test_value_on_a_known_pair(pair):
    labels, error = pair
    loss = SupervisedPath(divisor=DIVISOR, uncertainty="none")
    constants = loss.constants()
    value = float(loss.value(track(labels + error), labels, constants))
    by_hand = np.mean((error / np.asarray(DIVISOR)) ** 2)
    assert abs(value - by_hand) < 1e-14 * by_hand
    terms = loss.squared_terms(track(labels + error), labels, constants)
    assert terms.shape == (N, STEPS, STAGES + 1, 4)
    assert abs(float(terms.mean()) - value) < 1e-15
    assert float(loss.value(track(labels), labels, constants)) == 0.0


def test_the_gaussian_loss_is_lowest_at_the_right_sigma(pair):
    labels, _ = pair
    sigma = np.array([0.3, 0.7, 0.05, 0.02])
    # errors of exactly one sigma, of both signs, in every state and component
    signs = np.where(np.random.default_rng(2).random(labels.shape) < 0.5, -1.0, 1.0)
    emitted = labels + signs * sigma
    loss = SupervisedPath(divisor=DIVISOR, uncertainty="diagonal")
    constants = loss.constants()

    B = BOUND_OF_THE_LOG_SIGMA

    def at(sigma_claimed):
        # the raw output whose bounded log-sigma is log(sigma_claimed / divisor)
        wanted = np.log(sigma_claimed / np.asarray(DIVISOR))
        extra = torch.as_tensor(np.broadcast_to(B * np.arctanh(wanted / B), labels.shape).copy())
        return float(loss.value(track(emitted, extra), labels, constants))

    right = at(sigma)
    assert right < at(sigma / 2) and right < at(sigma * 2)
    by_hand = np.mean(1.0 + 2.0 * np.log(sigma / np.asarray(DIVISOR)))
    assert abs(right - by_hand) < 1e-12
    # bounded below: a raw output of -1e6 everywhere, with no error, gives -2B
    floor = torch.full(labels.shape, -1e6, dtype=torch.float64)
    assert abs(float(loss.value(track(labels, floor), labels, constants)) + 2 * B) < 1e-9


def test_what_is_refused(pair):
    labels, error = pair
    loss = SupervisedPath(divisor=DIVISOR, uncertainty="diagonal")
    with pytest.raises(ValueError):
        loss.value(track(labels + error), labels, loss.constants())
    with pytest.raises(ValueError):
        SupervisedPath(divisor=[1.0, 1.0, 1.0], uncertainty="none")
    with pytest.raises(ValueError):
        SupervisedPath(divisor=[1.0, 1.0, 1.0, -1.0], uncertainty="none")
    with pytest.raises(ValueError):
        SupervisedPath(divisor=DIVISOR, uncertainty="full")
    with pytest.raises(ValueError):
        SupervisedPath(divisor=DIVISOR, uncertainty="none").value(track(labels), None,
                                                                  loss.constants())
