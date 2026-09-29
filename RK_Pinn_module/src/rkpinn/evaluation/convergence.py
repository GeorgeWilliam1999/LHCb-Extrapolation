"""Convergence: the validation error round by round, and when it stopped falling.

Standard output 8 of `PACKAGE_PLAN.md`, section 5.

Convergence is judged on the validation error, never on the loss. From the
validation error of every round this gives:

  plateaued_now          whether the rule holds as of the latest round: the
                         verdict
  first_round_it_held    the first round the rule ever held: a record, not a
                         verdict. On the frozen runs the rule held early and
                         the error then fell again
  headline               the median of the last window of rounds, with its
                         spread, (largest - smallest) / 2 over the median

The rule itself is `validation_has_plateaued` of the training; it is used
here, not written again.

Ported from `Self_chained_paper/scripts/common.py` (`first_plateau_round`,
`validation_headline`), with the arithmetic unchanged.
Gates: tests/test_evaluation_reproduces_the_paper.py.
"""
from __future__ import annotations

import numpy as np

from rkpinn.training.stopping_rule import validation_has_plateaued


def first_round_the_rule_held(errors, window_in_rounds, tolerance, rounds_held):
    """The number of the first round the rule held, counting from 1; None if never."""
    v = np.asarray(errors, dtype=float)
    window, held = int(window_in_rounds), 0
    for r in range(2 * window, len(v) + 1):
        a, b = np.median(v[r - window:r]), np.median(v[r - 2 * window:r - window])
        held = held + 1 if a >= (1.0 - tolerance) * b else 0
        if held >= rounds_held:
            return r
    return None


def headline_of_validation(errors, window_in_rounds) -> dict:
    """The median of the last window of rounds, and how steady it is."""
    v = np.asarray(errors, dtype=float)
    window = int(window_in_rounds)
    last, before = v[-window:], v[-2 * window:-window]
    middle = float(np.median(last))
    return {
        "median_of_last_window_micrometres": middle,
        "spread_percent": float(100.0 * (last.max() - last.min()) / 2.0 / middle),
        "full_range_over_median_percent": float(
            100.0 * (last.max() - last.min()) / middle),
        "smallest_micrometres": float(last.min()),
        "largest_micrometres": float(last.max()),
        "last_window_against_the_one_before_percent": (
            float(100.0 * (middle - np.median(before)) / np.median(before))
            if len(before) == window else float("nan")),
        "rounds_used": int(len(last)),
    }


def convergence(errors, window_in_rounds, tolerance, rounds_held) -> dict:
    """The verdict and the record of a run, from its validation errors."""
    v = np.asarray(errors, dtype=float)
    out = {
        "rounds": int(len(v)),
        "plateaued_now": bool(validation_has_plateaued(
            v, window_in_rounds, tolerance, rounds_held)),
        "first_round_the_rule_held": first_round_the_rule_held(
            v, window_in_rounds, tolerance, rounds_held),
    }
    out.update(headline_of_validation(v, window_in_rounds))
    return out


def validation_per_round(rows_of_rounds, quantity) -> list:
    """One row per round: the round, where its states came from, its loss and
    its validation error."""
    return [dict(round=int(row["round"]), source_of_states=row["source_of_states"],
                 restarts=int(row["restarts"]), loss_first=float(row["loss_first"]),
                 loss_last=float(row["loss_last"]), validation_error=float(row[quantity]))
            for row in rows_of_rounds]
