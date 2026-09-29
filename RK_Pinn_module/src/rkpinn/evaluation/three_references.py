"""The three references: a prediction against the reference integrator,
against the exact scheme, and against the true state.

Standard output 7 of `PACKAGE_PLAN.md`, section 5.

The three contain different things, so they answer different questions.

  the reference integrator   the field and nothing else. Against it a
                             prediction is scored.
  the exact scheme           what a network with a loss of zero would give.
                             Its distance from the reference is the ceiling of
                             that number of steps and stages.
  the true state             the field, the scattering, the energy loss and
                             the interactions. Its distance from the reference
                             is the floor no method with the field alone can
                             beat.

The true state is on the particle's own plane of the fibre tracker, which is
not the last plane of the crossing. So a prediction is first carried on from
the last plane to that plane, with the reference integrator.

The position error is the larger of the errors in x and y, and the slope
error the larger of those of the two slopes, as in the second mini-paper.

Ported from `Self_chained_paper/scripts/numbers.py` (`sec_against_true`),
with the arithmetic unchanged. The comparison with the exact scheme is added.
Gates: tests/test_evaluation_reproduces_the_paper.py.
"""
from __future__ import annotations

import numpy as np

from rkpinn.evaluation import conventions
from rkpinn.integrators.runge_kutta_sixth_order import integrate_with_sixth_order


def carry_to_own_planes(equation_of_motion, end_states, last_plane_mm, own_planes_mm, *,
                        step_length_of_the_integrator_mm):
    """End states on the last plane, carried to each particle's own plane."""
    return integrate_with_sixth_order(
        equation_of_motion, np.asarray(end_states, dtype=np.float64), last_plane_mm,
        own_planes_mm, step_length_mm=step_length_of_the_integrator_mm)


def _distance(first, second, mask) -> dict:
    e = conventions.errors(first[mask], second[mask])
    return {
        "number_of_tracks": int(mask.sum()),
        "position_larger_of_x_and_y_median_micrometres": conventions.median(
            e["larger_of_x_and_y"]),
        "position_larger_of_x_and_y_percentile_95_micrometres": conventions.quantile(
            e["larger_of_x_and_y"], 0.95),
        "position_radial_median_micrometres": conventions.median(e["radial"]),
        "slope_larger_of_the_two_median": conventions.median(e["larger_of_slopes"]),
        "slope_in_x_median_of_absolute": conventions.median_of_absolute(e["slope in x"]),
        "slope_in_y_median_of_absolute": conventions.median_of_absolute(e["slope in y"]),
    }


def three_references(*, predicted_end_states, reference_end_states,
                     predicted_on_own_planes, reference_on_own_planes,
                     true_states_on_own_planes, momentum_gev, momentum_window_gev=None,
                     exact_end_states=None):
    """One row per band and comparison.

    Comparisons on the last plane:
      prediction against reference
      prediction against exact scheme     (if the exact end states are given)
      exact scheme against reference      (the ceiling; if given)
    Comparisons on the particle's own plane:
      prediction against true state
      reference against true state        (the floor)
    """
    pairs = [("prediction against reference", predicted_end_states, reference_end_states)]
    if exact_end_states is not None:
        pairs += [("prediction against exact scheme", predicted_end_states,
                   exact_end_states),
                  ("exact scheme against reference", exact_end_states,
                   reference_end_states)]
    pairs += [("prediction against true state", predicted_on_own_planes,
               true_states_on_own_planes),
              ("reference against true state", reference_on_own_planes,
               true_states_on_own_planes)]
    rows = []
    for band, mask in conventions.momentum_bands(momentum_gev, momentum_window_gev,
                                                 with_all=True):
        for name, first, second in pairs:
            rows.append(dict(band=band, comparison=name,
                             **_distance(np.asarray(first), np.asarray(second), mask)))
    return rows
