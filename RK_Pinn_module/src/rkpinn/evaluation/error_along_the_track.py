"""The error along the track: plane by plane, step by step, and which step's
error becomes how much endpoint error.

Standard outputs 4, 5 and 6 of `PACKAGE_PLAN.md`, section 5.

  4  the single-step error: one application of the network from a reference
     state, against the reference one step later. How good one application
     is, before chaining.
  5  the error along the track: the chained prediction against the reference,
     on every plane a step ends on. How the error grows.
  6  stage errors, held and carried: which step's stage error becomes how
     much endpoint error.

Output 6, in full. For step m of a track, with input state s_m on plane m:

  held      the stage error against the exact scheme started from the same
            input, stage by stage:   delta[k] = S_hat[k] - S_exact[k]

  at the end of the step, against the exact scheme from the same input:
            e = end state of the network - end state of the exact scheme
            To first order in the stage errors,
                e ~ dz * sum over k of  b_k * J_k delta[k],
            with J_k the derivative of the rates f with respect to the state
            at stage k. J_k delta[k] is taken as a central difference of f
            along delta[k]. What is left of e after the first order is given
            beside it, for the slopes. For x and y nothing is left: their
            rates are the slopes themselves, which are linear in the state,
            so the error of x and y at the end of a step is first order
            exactly.

  carried   every state is carried from the end of the step to the last
            plane with the reference integrator. The step then contributes,
            on the last plane,
                from its stage errors   carried(network end) - carried(exact end)
                from the scheme itself  carried(exact end) - carried(reference end)
            where the reference end is s_m advanced one step by the reference
            integrator.

  the check The two contributions of all steps sum to
                final state of the network - s_0 carried over the whole crossing,
            because every carried state appears once with each sign. This is
            the measured endpoint error, up to the difference between the
            integrator at the step used for carrying and the stored reference.
            Both the sum and that difference are returned.

  position and slope parts
            e in x is the position part. e in the slope, times the distance
            left to the last plane, is the slope part: what a slope error
            costs in position if it is carried along a straight line.

Outputs 4 and 5 are ported from `Self_chained_paper/scripts/numbers.py`
(`sec_along_z`), with the arithmetic unchanged. Output 6 is new.
Gates: tests/test_evaluation_reproduces_the_paper.py,
tests/test_stage_errors_held_and_carried.py.
"""
from __future__ import annotations

import numpy as np

from rkpinn.evaluation import conventions
from rkpinn.integrators.runge_kutta_sixth_order import integrate_with_sixth_order
from rkpinn.predicted_track.predicted_track import steps_of_the_exact_scheme

UM = conventions.MICROMETRES_PER_MILLIMETRE


# ------------------------------------------------------------------ output 5
def error_along_the_track(states_on_planes, reference_states_on_planes, planes_mm,
                          momentum_gev, momentum_window_gev=None):
    """One row per plane: the radial error of the chained prediction, its
    median and 95th percentile, its median in the window of the loss if one
    is given, and the median of the absolute error of every component.

    states_on_planes, reference_states_on_planes   shape (n, S + 1, 4 or more)
    """
    predicted = np.asarray(states_on_planes)
    reference = np.asarray(reference_states_on_planes)
    if predicted.shape[:2] != reference.shape[:2]:
        raise ValueError("the prediction and the reference are not on the same planes")
    window = None
    if momentum_window_gev is not None:
        p = np.asarray(momentum_gev, dtype=float)
        window = (p >= momentum_window_gev[0]) & (p < momentum_window_gev[1])
    rows = []
    for k in range(predicted.shape[1]):
        e = conventions.errors(predicted[:, k], reference[:, k])
        row = dict(plane=k, z_mm=float(planes_mm[k]),
                   radial_median_micrometres=conventions.median(e["radial"]),
                   radial_percentile_95_micrometres=conventions.quantile(e["radial"], 0.95))
        if window is not None:
            row["radial_median_in_the_window_micrometres"] = conventions.median(
                e["radial"][window])
        row["x_median_of_absolute_micrometres"] = conventions.median_of_absolute(e["x"])
        row["y_median_of_absolute_micrometres"] = conventions.median_of_absolute(e["y"])
        row["slope_in_x_median_of_absolute"] = conventions.median_of_absolute(e["slope in x"])
        row["slope_in_y_median_of_absolute"] = conventions.median_of_absolute(e["slope in y"])
        rows.append(row)
    return rows


# ------------------------------------------------------------------ output 4
def single_step_error(end_states_of_steps, reference_states_after_the_step,
                      step_of_the_track, start_planes_mm):
    """The error of one application, from separate steps.

    end_states_of_steps               shape (m, 4 or more): the end state of each step
    reference_states_after_the_step   shape (m, 4 or more): the reference one step
                                      after the state the step started from
    step_of_the_track                 shape (m,): which step of the track each is
    start_planes_mm                   shape (S,): z of the plane each step starts on

    Returns {"per_component": rows, "per_step": rows, "radial": statistics}.
    """
    e = conventions.errors(end_states_of_steps, reference_states_after_the_step)
    step = np.asarray(step_of_the_track)
    per_component = [dict(component=name, unit=unit,
                          **conventions.statistics_of_signed_error(e[name]))
                     for name, _, unit in conventions.COMPONENTS]
    per_step = []
    for k, plane in enumerate(start_planes_mm):
        here = step == k
        per_step.append(dict(
            step=k, z_of_start_plane_mm=float(plane),
            radial_median_micrometres=conventions.median(e["radial"][here]),
            slope_in_x_median_of_absolute=conventions.median_of_absolute(
                e["slope in x"][here]),
            slope_in_y_median_of_absolute=conventions.median_of_absolute(
                e["slope in y"][here])))
    return {"per_component": per_component, "per_step": per_step,
            "radial": conventions.statistics_of_magnitude(e["radial"])}


# ------------------------------------------------------------------ output 6
def _derivative_of_rates_along(equation_of_motion, states, planes, direction, fraction):
    """J delta: the central difference of the rates along `direction`.
    states (m, 5), direction (m, 4), planes (m,). Returns (m, 4)."""
    step = np.zeros_like(states)
    step[:, :4] = fraction * direction
    forward = equation_of_motion.rates(states + step, planes)[:, :4]
    backward = equation_of_motion.rates(states - step, planes)[:, :4]
    return (forward - backward) / (2.0 * fraction)


def stage_errors_held_and_carried(track, equation_of_motion, reference_end_states, *,
                                  step_length_of_the_carrying_mm,
                                  fraction_for_the_derivative):
    """Which step's stage error becomes how much endpoint error.

    track                  a whole track with stages, as numpy
    equation_of_motion     the numpy form
    reference_end_states   shape (n, 4 or more): the stored reference on the last plane
    step_length_of_the_carrying_mm   the step of the reference integrator used
                           to carry states to the last plane
    fraction_for_the_derivative      the fraction of a stage error the rates
                           are differenced over, to take J delta

    Returns a dict:
      per_step      one row per step
      check         the sum of the contributions against the measured error
      held          the stage errors, shape (n, S, q, 4), in mm and slopes
      from_stage_errors, from_the_scheme
                    the contributions on the last plane, shape (n, S, 4)
    """
    if not track.is_a_whole_track or track.number_of_stages == 0:
        raise ValueError("stage errors need a whole track with stages")
    inputs = np.asarray(track.input_states, dtype=np.float64)
    stages = np.asarray(track.stage_states, dtype=np.float64)
    ends = np.asarray(track.end_states, dtype=np.float64)
    n, steps = inputs.shape[:2]
    q = track.number_of_stages
    dz = track.step_length_mm
    tableau = track.tableau
    planes = track.layout.planes_mm
    last_plane = float(planes[-1])
    charge_over_momentum = inputs[:, 0, 4:5]

    # -- held: against the exact scheme from the same input -------------------------
    exact_stages = np.empty((n, steps, q, 4))
    exact_ends = np.empty((n, steps, 4))
    first_order = np.empty((n, steps, 4))
    for m in range(steps):
        exact = steps_of_the_exact_scheme(
            equation_of_motion, inputs[:, m], planes[m], dz, tableau)
        exact_stages[:, m] = exact.stage_states[:, 0]
        exact_ends[:, m] = exact.end_states[:, 0]
        stage_planes = planes[m] + tableau.nodes * dz
        flat = np.concatenate(
            [exact_stages[:, m].reshape(n * q, 4), np.repeat(charge_over_momentum, q, axis=0)],
            axis=1)
        direction = (stages[:, m] - exact_stages[:, m]).reshape(n * q, 4)
        along = _derivative_of_rates_along(
            equation_of_motion, flat, np.tile(stage_planes, n), direction,
            fraction_for_the_derivative).reshape(n, q, 4)
        first_order[:, m] = dz * np.einsum("k,nkd->nd", tableau.weights, along)
    held = stages - exact_stages
    at_the_end_of_the_step = ends - exact_ends
    beyond_first_order = at_the_end_of_the_step - first_order

    # -- carried: one march over the planes, the states joining as it passes ----------
    # from the inputs:     the input of step m, from plane m to the last plane
    # from the exact ends: the exact end of step m, from plane m + 1 to the last plane
    from_inputs = np.empty((steps, n, 5))
    from_exact_ends = np.empty((max(steps - 1, 0), n, 5))
    for k in range(steps):
        from_inputs[k] = inputs[:, k]
        if k >= 1:
            from_exact_ends[k - 1, :, :4] = exact_ends[:, k - 1]
            from_exact_ends[k - 1, :, 4] = charge_over_momentum[:, 0]
        moving = np.concatenate([from_inputs[:k + 1].reshape(-1, 5),
                                 from_exact_ends[:k].reshape(-1, 5)], axis=0)
        moved = integrate_with_sixth_order(
            equation_of_motion, moving, planes[k], planes[k + 1],
            step_length_mm=step_length_of_the_carrying_mm)
        from_inputs[:k + 1] = moved[:(k + 1) * n].reshape(k + 1, n, 5)
        from_exact_ends[:k] = moved[(k + 1) * n:].reshape(k, n, 5)

    from_stage_errors = np.empty((n, steps, 4))
    from_the_scheme = np.empty((n, steps, 4))
    for m in range(steps):
        last = m == steps - 1
        network_end = ends[:, m] if last else from_inputs[m + 1][:, :4]
        exact_end = exact_ends[:, m] if last else from_exact_ends[m][:, :4]
        reference_end = from_inputs[m][:, :4]
        from_stage_errors[:, m] = network_end - exact_end
        from_the_scheme[:, m] = exact_end - reference_end

    # -- the check ---------------------------------------------------------------------
    total = (from_stage_errors + from_the_scheme).sum(axis=1)
    against_the_carried_reference = ends[:, -1] - from_inputs[0][:, :4]
    measured = ends[:, -1] - np.asarray(reference_end_states, dtype=np.float64)[:, :4]
    check = {
        "sum_against_final_minus_carried_start_largest_micrometres": float(
            np.abs(total - against_the_carried_reference)[:, :2].max() * UM),
        "sum_against_measured_endpoint_error_largest_micrometres": float(
            np.abs(total - measured)[:, :2].max() * UM),
        "sum_against_measured_endpoint_error_largest_slope": float(
            np.abs(total - measured)[:, 2:].max()),
        "carried_start_against_stored_reference_largest_micrometres": float(
            np.abs(from_inputs[0][:, :4]
                   - np.asarray(reference_end_states)[:, :4])[:, :2].max() * UM),
        "measured_endpoint_error_in_x_median_of_absolute_micrometres":
            conventions.median_of_absolute(measured[:, 0] * UM),
        "step_length_of_the_carrying_mm": float(step_length_of_the_carrying_mm),
        "number_of_tracks": int(n),
    }

    # -- the table ---------------------------------------------------------------------
    med = conventions.median_of_absolute
    rows = []
    for m in range(steps):
        distance_left = last_plane - float(planes[m + 1])
        e = at_the_end_of_the_step[:, m]
        row = dict(step=m, z_of_start_plane_mm=float(planes[m]),
                   distance_left_mm=distance_left)
        for name, place in (("x", 0), ("y", 1)):
            slope = place + 2
            row.update({
                "largest_stage_error_in_%s_median_micrometres" % name: conventions.median(
                    np.abs(held[:, m, :, place]).max(axis=1) * UM),
                "largest_stage_error_in_slope_in_%s_median" % name: conventions.median(
                    np.abs(held[:, m, :, slope]).max(axis=1)),
                "position_part_in_%s_median_of_absolute_micrometres" % name:
                    med(e[:, place] * UM),
                "slope_part_in_%s_median_of_absolute_micrometres" % name:
                    med(e[:, slope] * distance_left * UM),
                "beyond_first_order_in_slope_in_%s_median_of_absolute" % name:
                    med(beyond_first_order[:, m, slope]),
                "first_order_in_slope_in_%s_median_of_absolute" % name:
                    med(first_order[:, m, slope]),
                "carried_from_stage_errors_in_%s_median_of_absolute_micrometres" % name:
                    med(from_stage_errors[:, m, place] * UM),
                "carried_from_stage_errors_in_%s_signed_median_micrometres" % name:
                    conventions.signed_median(from_stage_errors[:, m, place] * UM),
                "carried_from_the_scheme_in_%s_median_of_absolute_micrometres" % name:
                    med(from_the_scheme[:, m, place] * UM),
            })
        rows.append(row)
    return {"per_step": rows, "check": check, "held": held,
            "from_stage_errors": from_stage_errors, "from_the_scheme": from_the_scheme}
