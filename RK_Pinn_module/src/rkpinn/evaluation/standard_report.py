"""The standard report: the ten standard outputs of `PACKAGE_PLAN.md`,
section 5, for one prediction.

It reads states and targets. It imports no network and knows of none: the
same report scores a network, the exact scheme or the reference integrator.

   1  endpoint error per component                 endpoint_error_per_component.csv
   2  endpoint error per component and band        (the same table, by band)
                                                   endpoint_error_magnitudes.csv
   3  signed distributions per component           signed_distributions.csv
   4  single-step error per component              single_step_error_per_component.csv
                                                   single_step_error_per_step.csv
   5  error along the track                        error_along_the_track.csv
   6  stage errors, held and carried               stage_errors_held_and_carried.csv
   7  the three references                         three_references.csv
   8  convergence                                  convergence.csv, validation_per_round.csv
   9  where the loss puts its weight               where_the_loss_puts_its_weight.csv
  10  cost                                         cost.csv

An output whose input is not given is left out, and the summary says so. It
is never filled with a placeholder.

Output 9, where the loss puts its weight. For every state of a draw the loss
has a value, the mean of its squared weighted residuals. The share of a state
is its value over the sum of all. The table gives the share of the loss in
each momentum band and in each quarter of the crossing, beside the share of
the states there. Ported from `Self_chained_paper/scripts/numbers.py`
(`sec_preflight`), with the arithmetic unchanged.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from rkpinn.evaluation import conventions, convergence, endpoint_error
from rkpinn.evaluation.error_along_the_track import error_along_the_track, single_step_error
from rkpinn.evaluation.three_references import three_references

NUMBER_OF_QUARTERS = 4


# ------------------------------------------------------------------ output 9
def where_the_loss_puts_its_weight(loss_of_each_state, momentum_gev, step_of_the_track,
                                   number_of_steps, momentum_window_gev=None):
    """One row per momentum band and per quarter of the crossing: the share of
    the loss and the share of the states, in percent."""
    value = np.asarray(loss_of_each_state, dtype=float)
    step = np.asarray(step_of_the_track)
    share = value / value.sum()
    rows = []
    for band, mask in conventions.momentum_bands(momentum_gev, momentum_window_gev):
        rows.append(dict(kind="momentum band", where=band, number_of_states=int(mask.sum()),
                         share_of_the_loss_percent=100.0 * float(share[mask].sum()),
                         share_of_the_states_percent=100.0 * float(mask.mean())))
    for quarter in range(NUMBER_OF_QUARTERS):
        mask = ((step >= quarter * number_of_steps // NUMBER_OF_QUARTERS)
                & (step < (quarter + 1) * number_of_steps // NUMBER_OF_QUARTERS))
        rows.append(dict(kind="quarter of the crossing", where="quarter %d" % (quarter + 1),
                         number_of_states=int(mask.sum()),
                         share_of_the_loss_percent=100.0 * float(share[mask].sum()),
                         share_of_the_states_percent=100.0 * float(mask.mean())))
    top = max(1, len(share) // 100)
    rows.append(dict(kind="the 1 % of states with the largest loss", where="all",
                     number_of_states=int(top),
                     share_of_the_loss_percent=100.0 * float(np.sort(share)[::-1][:top].sum()),
                     share_of_the_states_percent=100.0 * top / len(share)))
    return rows


# ------------------------------------------------------------------ the report
def standard_report(*, name, momentum_gev, momentum_window_gev,
                    states_on_planes, reference_states_on_planes, planes_mm,
                    single_steps=None, stage_errors=None, references=None,
                    rounds=None, loss_shares=None, cost=None) -> dict:
    """The tables of the standard report.

    name                      how the prediction is named: by steps, stages and step length
    states_on_planes          shape (n, S + 1, 5): the prediction on the first plane
                              and at the end of every step
    reference_states_on_planes  the reference on the same planes
    single_steps    dict(end_states, reference_after_the_step, step_of_the_track,
                    start_planes_mm), or None
    stage_errors    what `stage_errors_held_and_carried` returned, or None
    references      dict(predicted_on_own_planes, reference_on_own_planes,
                    true_states_on_own_planes, exact_end_states or None), or None
    rounds          dict(rows, quantity, window_in_rounds, tolerance, rounds_held), or None
    loss_shares     dict(loss_of_each_state, momentum_gev, step_of_the_track,
                    number_of_steps), or None
    cost            dict of numbers, or None

    Returns {"tables": {name: rows}, "summary": dict}.
    """
    predicted = np.asarray(states_on_planes)
    reference = np.asarray(reference_states_on_planes)
    end, reference_end = predicted[:, -1], reference[:, -1]
    tables, left_out = {}, []

    tables["endpoint_error_per_component"] = endpoint_error.errors_per_component(
        end, reference_end, momentum_gev, momentum_window_gev)
    tables["endpoint_error_magnitudes"] = endpoint_error.magnitudes_of_position_error(
        end, reference_end, momentum_gev, momentum_window_gev)
    tables["signed_distributions"] = endpoint_error.signed_distributions(
        end, reference_end, momentum_gev, momentum_window_gev)
    tables["error_along_the_track"] = error_along_the_track(
        predicted, reference, planes_mm, momentum_gev, momentum_window_gev)

    summary = {"name": name, "number_of_tracks": int(len(end)),
               "kind_of_error": "endpoint error, after the whole chain, unless a table "
                                "says single step"}
    for row in tables["endpoint_error_magnitudes"]:
        if row["band"] == conventions.ALL_MOMENTA:
            summary["endpoint_%s_median_micrometres" % row["magnitude"]] = \
                row["median_micrometres"]

    if single_steps is not None:
        one = single_step_error(
            single_steps["end_states"], single_steps["reference_after_the_step"],
            single_steps["step_of_the_track"], single_steps["start_planes_mm"])
        tables["single_step_error_per_component"] = one["per_component"]
        tables["single_step_error_per_step"] = one["per_step"]
        summary["single_step_radial_median_micrometres"] = one["radial"]["median_micrometres"]
    else:
        left_out.append("4, the single-step error: no separate steps were given")

    if stage_errors is not None:
        tables["stage_errors_held_and_carried"] = stage_errors["per_step"]
        summary["check_of_the_stage_errors"] = stage_errors["check"]
    else:
        left_out.append("6, stage errors held and carried: not given; it needs a "
                        "prediction with stages")

    if references is not None:
        tables["three_references"] = three_references(
            predicted_end_states=end, reference_end_states=reference_end,
            momentum_gev=momentum_gev, momentum_window_gev=momentum_window_gev,
            **references)
    else:
        left_out.append("7, the three references: not given")

    if rounds is not None:
        series = [float(row[rounds["quantity"]]) for row in rounds["rows"]]
        verdict = convergence.convergence(
            series, rounds["window_in_rounds"], rounds["tolerance"], rounds["rounds_held"])
        verdict["quantity"] = rounds["quantity"]
        tables["convergence"] = [verdict]
        tables["validation_per_round"] = convergence.validation_per_round(
            rounds["rows"], rounds["quantity"])
        summary["plateaued_now"] = verdict["plateaued_now"]
    else:
        left_out.append("8, convergence: no rounds were given")

    if loss_shares is not None:
        tables["where_the_loss_puts_its_weight"] = where_the_loss_puts_its_weight(
            momentum_window_gev=momentum_window_gev, **loss_shares)
    else:
        left_out.append("9, where the loss puts its weight: not given")

    if cost is not None:
        tables["cost"] = [dict(cost)]
    else:
        left_out.append("10, cost: not given")

    summary["outputs_left_out"] = left_out
    return {"tables": tables, "summary": summary}


# --------------------------------------------------------------------- writing
def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    return value


def write_report(report: dict, folder: str) -> list:
    """Write every table as a CSV file and the summary as JSON. The folder
    must not exist: a report is never written over another."""
    os.makedirs(folder)
    written = []
    for name, rows in report["tables"].items():
        if not rows:
            continue
        columns = []
        for row in rows:
            columns += [c for c in row if c not in columns]
        path = os.path.join(folder, name + ".csv")
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(_plain(rows))
        written.append(path)
    path = os.path.join(folder, "summary.json")
    with open(path, "w") as handle:
        json.dump(_plain(report["summary"]), handle, indent=1)
    written.append(path)
    return written


def write_figures(report: dict, folder: str, histograms=None) -> list:
    """The figures of the report, as PNG files in a folder that exists.

    Every figure names the prediction by its steps, stages and step length,
    and says whether the error is an endpoint error or a single-step error.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tables, name = report["tables"], report["summary"]["name"]
    written = []

    def save(figure, file):
        figure.tight_layout()
        path = os.path.join(folder, file)
        figure.savefig(path, dpi=130)
        plt.close(figure)
        written.append(path)

    rows = tables["error_along_the_track"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    z = [r["z_mm"] for r in rows]
    axis.plot(z, [r["radial_median_micrometres"] for r in rows], label="median")
    axis.plot(z, [r["radial_percentile_95_micrometres"] for r in rows],
              label="95th percentile")
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_xlabel("z of the plane [mm]")
    axis.set_ylabel("radial error of the chained prediction [micrometres]")
    axis.set_title("Error along the track\n%s" % name)
    axis.legend()
    save(figure, "error_along_the_track.png")

    rows = [r for r in tables["endpoint_error_per_component"]
            if r["band"] != conventions.ALL_MOMENTA]
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    for axis, (component, _, unit) in zip(axes, conventions.COMPONENTS):
        mine = [r for r in rows if r["component"] == component]
        axis.bar(range(len(mine)), [r["median_of_absolute"] for r in mine])
        axis.set_xticks(range(len(mine)))
        axis.set_xticklabels([r["band"] for r in mine], rotation=60, ha="right", fontsize=7)
        axis.set_yscale("log")
        axis.set_title("%s [%s]" % (component, unit))
    figure.suptitle("Endpoint error, median of the absolute error, by momentum band\n%s"
                    % name)
    save(figure, "endpoint_error_by_momentum_band.png")

    if histograms is not None:
        figure, axes = plt.subplots(1, 4, figsize=(14, 3.6))
        for axis, (component, _, unit) in zip(axes, conventions.COMPONENTS):
            h = histograms[component]
            axis.stairs(h["counts"], h["edges"])
            axis.axvline(0.0, color="0.5", lw=0.8)
            axis.set_xlabel("signed endpoint error in %s [%s]" % (component, unit))
        figure.suptitle("Signed endpoint error, central 98 %% of the tracks\n%s" % name)
        save(figure, "signed_distributions.png")

    if "validation_per_round" in tables:
        rows = tables["validation_per_round"]
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot([r["round"] for r in rows], [r["validation_error"] for r in rows], "o-")
        held = tables["convergence"][0]["first_round_the_rule_held"]
        if held is not None:
            axis.axvline(held, color="0.5", ls="--", label="first round the rule held")
            axis.legend()
        axis.set_yscale("log")
        axis.set_xlabel("round")
        axis.set_ylabel("validation endpoint error [micrometres]")
        axis.set_title("Convergence, judged on the validation error\n%s" % name)
        save(figure, "convergence.png")

    if "stage_errors_held_and_carried" in tables:
        rows = tables["stage_errors_held_and_carried"]
        figure, axis = plt.subplots(figsize=(8, 4.5))
        z = [r["z_of_start_plane_mm"] for r in rows]
        axis.plot(z, [r["carried_from_stage_errors_in_x_median_of_absolute_micrometres"]
                      for r in rows], label="from the stage errors of the step")
        axis.plot(z, [r["carried_from_the_scheme_in_x_median_of_absolute_micrometres"]
                      for r in rows], label="from the scheme itself")
        axis.set_yscale("log")
        axis.set_xlabel("z of the plane the step starts on [mm]")
        axis.set_ylabel("contribution to the endpoint error in x [micrometres]")
        axis.set_title("Stage errors, held and carried to the last plane\n%s" % name)
        axis.legend()
        save(figure, "stage_errors_held_and_carried.png")
    return written
