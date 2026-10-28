"""The standard report: the eleven standard outputs of `PACKAGE_PLAN.md`,
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
  11  resolution against momentum                  resolution_per_momentum_band.csv
                                                   resolution_against_momentum.png
                                                   signed_resolution_per_momentum_band.csv
                                                   signed_resolution_against_momentum.png

An output whose input is not given is left out, and the summary says so. It
is never filled with a placeholder.

Output 11, the resolution. |predicted - reference| per component, for the
endpoint error and, when separate steps are given, for the single-step error,
against the total momentum p. The table holds the median and the 16th and 84th percentiles per band; the
figures are two-dimensional histograms with those percentiles drawn over
them. The signed resolution, predicted - reference, is given the same way,
with half the 16th to 84th percentile range beside the percentiles, on a
symmetric-log axis (George, 2026-10-28). See `resolution.py`.

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

from rkpinn.evaluation import conventions, convergence, endpoint_error, resolution
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
    momentum_gev              shape (n,): the total momentum p of each track
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

    tables["resolution_per_momentum_band"] = resolution_per_momentum_band(
        end, reference_end, momentum_gev, momentum_window_gev, single_steps)
    for row in tables["resolution_per_momentum_band"]:
        if row["band"] == conventions.ALL_MOMENTA:
            summary["%s_resolution_%s_median" % (row["kind_of_error"].replace(" ", "_"),
                                                 row["component"].replace(" ", "_"))] = \
                row["resolution_median"]
    tables["signed_resolution_per_momentum_band"] = signed_resolution_per_momentum_band(
        end, reference_end, momentum_gev, momentum_window_gev, single_steps)
    for row in tables["signed_resolution_per_momentum_band"]:
        if row["band"] == conventions.ALL_MOMENTA:
            summary["%s_signed_resolution_%s_median" % (row["kind_of_error"].replace(" ", "_"),
                                                        row["component"].replace(" ", "_"))] = \
                row["signed_median"]
    if single_steps is None:
        left_out.append("11, the single-step resolution: no separate steps were given; "
                        "the endpoint resolution is in the table")

    summary["outputs_left_out"] = left_out
    return {"tables": tables, "summary": summary}


# ----------------------------------------------------------------- output 11
def _errors_for_the_resolution(end, reference_end, momentum_gev, single_steps):
    """[(kind of error, predicted, reference, momentum of each row)]: the
    endpoint error, then the single-step error when separate steps are given."""
    out = [(resolution.ENDPOINT, end, reference_end, momentum_gev)]
    if single_steps is not None:
        ends = np.asarray(single_steps["end_states"])
        out.append((resolution.SINGLE_STEP, ends, single_steps["reference_after_the_step"],
                    resolution.momentum_of_each_single_step(momentum_gev, len(ends))))
    return out


def resolution_per_momentum_band(end, reference_end, momentum_gev, momentum_window_gev,
                                 single_steps):
    """The table of output 11: one row per kind of error, band and component."""
    rows = []
    for kind, predicted, reference, p in _errors_for_the_resolution(
            end, reference_end, momentum_gev, single_steps):
        rows += resolution.resolution_per_momentum_band(
            predicted, reference, p, kind, momentum_window_gev)
    return rows


def signed_resolution_per_momentum_band(end, reference_end, momentum_gev,
                                        momentum_window_gev, single_steps):
    """The signed table of output 11: one row per kind of error, band and
    component."""
    rows = []
    for kind, predicted, reference, p in _errors_for_the_resolution(
            end, reference_end, momentum_gev, single_steps):
        rows += resolution.signed_resolution_per_momentum_band(
            predicted, reference, p, kind, momentum_window_gev)
    return rows


def histograms_of_signed_resolution(end, reference_end, momentum_gev, single_steps,
                                    number_of_bins):
    """The histograms of the signed figure of output 11: {kind of error: what
    `resolution.histograms_of_signed_resolution` returns}."""
    return {kind: resolution.histograms_of_signed_resolution(predicted, reference, p,
                                                             number_of_bins)
            for kind, predicted, reference, p in _errors_for_the_resolution(
                end, reference_end, momentum_gev, single_steps)}


def histograms_of_resolution(end, reference_end, momentum_gev, single_steps,
                             number_of_bins):
    """The histograms of output 11 for the figure: {kind of error: what
    `resolution.histograms_of_resolution` returns}."""
    return {kind: resolution.histograms_of_resolution(predicted, reference, p, number_of_bins)
            for kind, predicted, reference, p in _errors_for_the_resolution(
                end, reference_end, momentum_gev, single_steps)}


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


def write_figures(report: dict, folder: str, histograms=None,
                  histograms_of_the_resolution=None,
                  histograms_of_the_signed_resolution=None) -> list:
    """The figures of the report, as PNG files in a folder that exists.

    histograms                     what `endpoint_error.histograms_of_signed_error` returned
    histograms_of_the_resolution   what `histograms_of_resolution` returned: one
                                   row of panels per kind of error and one panel
                                   per component
    histograms_of_the_signed_resolution   what `histograms_of_signed_resolution`
                                   returned, laid out the same way

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

    if histograms_of_the_resolution:
        written += write_figure_of_the_resolution(name, folder, histograms_of_the_resolution)
    if histograms_of_the_signed_resolution:
        written += write_figure_of_the_resolution(name, folder,
                                                  histograms_of_the_signed_resolution, signed=True)
    return written


def write_figure_of_the_resolution(name: str, folder: str, histograms_of_the_resolution,
                                   signed: bool = False) -> list:
    """A figure of output 11: one row of panels per kind of error, one panel
    per component, the resolution of every track against p as a
    two-dimensional histogram with the median and the 16th and 84th
    percentiles per bin of p drawn over it.

    signed False: resolution_against_momentum.png, |predicted - reference| on
                  a logarithmic axis, from `histograms_of_resolution`
           True:  signed_resolution_against_momentum.png, predicted -
                  reference on a symmetric-log axis with zero marked, from
                  `histograms_of_signed_resolution`
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors
    import matplotlib.pyplot as plt

    by_kind = histograms_of_the_resolution
    kinds = [kind for kind in resolution.KINDS_OF_ERROR if kind in by_kind]
    figure, axes = plt.subplots(len(kinds), 4, figsize=(15, 3.8 * len(kinds)),
                                squeeze=False)
    for row, kind in zip(axes, kinds):
        for axis, (component, _, unit) in zip(row, conventions.COMPONENTS):
            h = by_kind[kind][component]
            counts = np.ma.masked_equal(h["counts"].T, 0)
            axis.pcolormesh(h["momentum_edges_gev"],
                            h["signed_edges"] if signed else h["resolution_edges"], counts,
                            cmap="Blues", norm=matplotlib.colors.LogNorm(), rasterized=True)
            centres = h["momentum_centres_gev"]
            axis.plot(centres, h["median"], color="k", lw=1.2, label="median")
            axis.plot(centres, h["percentile_16"], color="k", lw=0.8, ls="--",
                      label="16th and 84th percentiles")
            axis.plot(centres, h["percentile_84"], color="k", lw=0.8, ls="--")
            axis.set_xscale("log")
            if signed:
                axis.axhline(0.0, color="r", lw=0.6)
                axis.set_yscale("symlog", linthresh=h["linear_threshold"])
                axis.set_ylabel("predicted - reference, %s [%s]" % (component, unit))
            else:
                axis.set_yscale("log")
                axis.set_ylabel("resolution in %s [%s]" % (component, unit))
            axis.set_xlabel("p [GeV]")
            axis.set_title("%s error, %s" % (kind, component))
        row[0].legend(fontsize=7, loc="upper right")
    figure.suptitle("%s against p\n%s" % (
        "Signed resolution, predicted - reference," if signed
        else "Resolution |predicted - reference|", name))
    figure.tight_layout()
    path = os.path.join(folder, "%sresolution_against_momentum.png" % ("signed_" if signed else ""))
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return [path]
