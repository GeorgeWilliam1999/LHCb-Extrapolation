"""The validation error of a round: what the stopping rule is applied to.

After every round the network carries the validation tracks from the first
plane to the last, and its end states are compared with the reference end
states. Three numbers are recorded, each a median over the validation tracks
of an absolute error in micrometres:

    validation_endpoint_error_in_x_median_micrometres
    validation_endpoint_error_in_y_median_micrometres
    validation_endpoint_error_larger_of_x_and_y_median_micrometres
        for each track the larger of its errors in x and in y; this is the
        number the frozen runs were judged on

Which of the three the stopping rule uses is a setting of the rule.

These are endpoint errors: errors after the whole chain. The full evaluation,
per component, per momentum band and signed, is the standard report of phase 5.
"""
from __future__ import annotations

import numpy as np

MICROMETRES_PER_MILLIMETRE = 1e3

QUANTITIES = (
    "validation_endpoint_error_in_x_median_micrometres",
    "validation_endpoint_error_in_y_median_micrometres",
    "validation_endpoint_error_larger_of_x_and_y_median_micrometres",
)


def validation_errors(final_states, reference_end_states) -> dict:
    """The three numbers, from end states of shape (n, 4 or 5)."""
    predicted = np.asarray(final_states, dtype=np.float64)
    reference = np.asarray(reference_end_states, dtype=np.float64)
    in_x = np.abs(predicted[:, 0] - reference[:, 0]) * MICROMETRES_PER_MILLIMETRE
    in_y = np.abs(predicted[:, 1] - reference[:, 1]) * MICROMETRES_PER_MILLIMETRE
    # as the frozen code: np.abs(st[:, :2] - tr[:, :2]).max(axis=1) * 1e3
    larger = (np.abs(predicted[:, :2] - reference[:, :2]).max(axis=1)
              * MICROMETRES_PER_MILLIMETRE)
    return {
        QUANTITIES[0]: float(np.median(in_x)),
        QUANTITIES[1]: float(np.median(in_y)),
        QUANTITIES[2]: float(np.median(larger)),
    }
