"""The stopping rule: the plateau of the validation error.

Convergence is judged on the validation error, never on the loss: the loss
can be flat while the error is still falling. The test split is never used.

The rule, with v the validation error of every round so far:

    at round r,  a = median of v over the last `window_in_rounds` rounds
                 b = median of v over the `window_in_rounds` rounds before those
    the error has stopped falling at r when  a >= (1 - tolerance) * b

and the run has plateaued when that holds at each of the last `rounds_held`
rounds. It therefore needs 2 * window_in_rounds + rounds_held - 1 rounds
before it can hold at all.

The values used on the frozen runs were a window of 10 rounds, a tolerance of
0.05 and 3 rounds held. Here each is a setting with no default, and so is the
quantity the rule is applied to.

Ported from `F2_Analysis/compare_to_blockE.py` (`plateaued_now`), with the
arithmetic unchanged.
Gates: tests/test_training.py.
"""
from __future__ import annotations

import numpy as np

from rkpinn.registry import register
from rkpinn.training.validation_error import QUANTITIES


def validation_has_plateaued(errors, window_in_rounds, tolerance, rounds_held) -> bool:
    v = np.asarray(errors, dtype=np.float64)
    window, hold = int(window_in_rounds), int(rounds_held)
    if len(v) < 2 * window + hold - 1:
        return False
    for r in range(len(v) - hold + 1, len(v) + 1):
        a, b = np.median(v[r - window:r]), np.median(v[r - 2 * window:r - window])
        if a < (1.0 - tolerance) * b:
            return False
    return True


@register("stopping_rule", "validation_plateau")
class ValidationPlateau:
    name = "validation_plateau"
    settings_in_a_configuration = ("quantity", "window_in_rounds", "tolerance",
                                   "rounds_held")

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(**settings)

    def __init__(self, *, quantity, window_in_rounds, tolerance, rounds_held):
        if quantity not in QUANTITIES:
            raise ValueError("the rule is applied to one of %s, not %r"
                             % (", ".join(QUANTITIES), quantity))
        if int(window_in_rounds) < 1 or int(rounds_held) < 1 or not 0 <= tolerance < 1:
            raise ValueError("the window and the rounds held are at least 1, and the "
                             "tolerance is from 0 to below 1")
        self.quantity = quantity
        self.window_in_rounds = int(window_in_rounds)
        self.tolerance = float(tolerance)
        self.rounds_held = int(rounds_held)

    def holds(self, errors) -> bool:
        """True when the run has plateaued as of its latest round."""
        return validation_has_plateaued(errors, self.window_in_rounds, self.tolerance,
                                        self.rounds_held)
