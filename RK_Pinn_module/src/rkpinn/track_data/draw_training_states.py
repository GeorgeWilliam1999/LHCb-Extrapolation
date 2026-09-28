"""Draw the states one round trains on: the same number on every start plane.

The states are pairs of (state, plane the step starts on). When the budget
covers every track on every start plane, all are taken. Otherwise the same
number of tracks is drawn on each start plane, without repetition within a
plane, and independently from plane to plane.

This file only draws. Which states it draws from, the reference states in the
first round or the network's own predictions later, is decided by the training
protocol.

Ported from `E1_Network_grid/train_network.py` (`draw_states`) with the
arithmetic and the order of the random draws unchanged, so the same seed gives
the same states.
Gates: tests/test_draw_of_training_states_matches_the_frozen_code.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DrawnStates:
    states: np.ndarray              # shape (m, 5)
    start_planes_mm: np.ndarray     # shape (m,): z of the plane each step starts on
    step_of_the_track: np.ndarray   # shape (m,): which step of the track, from 0
    track: np.ndarray               # shape (m,): which track


def generator_of_a_round(seed: int, round_number: int) -> np.random.Generator:
    """The random generator of one round of one run. Rounds count from 1."""
    return np.random.default_rng([seed, round_number])


def draw_training_states(states_on_start_planes, start_planes_mm, states_per_round,
                         generator: np.random.Generator) -> DrawnStates:
    """
    states_on_start_planes  shape (n, N, 5): every track's state on the plane
                            each of its N steps starts on
    start_planes_mm         shape (N,)
    states_per_round        the budget of states
    """
    n_tr, N = states_on_start_planes.shape[:2]
    if n_tr * N <= states_per_round:
        tr = np.repeat(np.arange(n_tr)[None, :], N, axis=0).ravel()
        pl = np.repeat(np.arange(N), n_tr)
    else:
        per = min(n_tr, states_per_round // N)
        tr = np.concatenate([generator.choice(n_tr, per, replace=False) for _ in range(N)])
        pl = np.repeat(np.arange(N), per)
    return DrawnStates(states=states_on_start_planes[tr, pl],
                       start_planes_mm=start_planes_mm[pl],
                       step_of_the_track=pl, track=tr)
