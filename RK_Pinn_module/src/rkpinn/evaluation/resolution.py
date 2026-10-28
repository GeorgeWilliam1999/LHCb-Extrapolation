"""The resolution: the absolute and the signed error of every component, against momentum.

Standard output 11 of `PACKAGE_PLAN.md`, section 5.

The resolution of a component is the absolute difference between the
predicted state and the reference state,

    resolution = | s_predicted - s_reference |,

per component x, y, tx and ty. It is defined for two errors:

  endpoint      the whole track across the magnet region, from the first
                plane to the last plane of the crossing: the chained
                prediction against the reference on the last plane
  single step   one application of the network from the reference state on a
                start plane, against the reference one step later

and it is shown against momentum: as a two-dimensional histogram of the
resolution against the momentum, with the median and the 16th and 84th
percentiles in every bin of momentum drawn over it, and as a table with one
row per momentum band and component holding the median and the 16th and 84th
percentiles.

The signed resolution is the difference itself,

    signed resolution = s_predicted - s_reference,

with the same two errors, the same table (the median and the 16th and 84th
percentiles, and half their range, the 68 % half-width) and the same figure,
on an axis that is linear within plus or minus the 5th percentile of
|signed resolution| and logarithmic outside it (George, 2026-10-28). Where
the absolute resolution gives the size of the error, the signed resolution
shows whether it is a bias or a spread, and at which momenta.

The momentum is the total momentum p of the track, the momentum of the
bands of every other output (George, 2026-10-27: p, not pT).

The reference is the sixth-order Runge-Kutta integrator at its step of
0.1 mm, the same reference every other output uses.

Gates: tests/test_resolution.py, tests/test_report_of_a_run.py.
"""
from __future__ import annotations

import numpy as np

from rkpinn.evaluation import conventions

ENDPOINT = "endpoint"
SINGLE_STEP = "single step"
KINDS_OF_ERROR = (ENDPOINT, SINGLE_STEP)

PERCENTILES_OF_THE_RESOLUTION = (0.16, 0.50, 0.84)


def resolution(predicted, reference) -> dict:
    """The resolution of every component: |predicted - reference|, in the
    units of the conventions. predicted, reference: shape (n, 4 or more)."""
    e = conventions.errors(predicted, reference)
    return {name: np.abs(e[name]) for name, _, _ in conventions.COMPONENTS}


def signed_resolution(predicted, reference) -> dict:
    """The signed resolution of every component: predicted - reference, in
    the units of the conventions."""
    e = conventions.errors(predicted, reference)
    return {name: np.asarray(e[name], dtype=float) for name, _, _ in conventions.COMPONENTS}


def resolution_per_momentum_band(predicted, reference, momentum_gev, kind_of_error,
                                 momentum_window_gev=None):
    """One row per momentum band and component: the median and the 16th and
    84th percentiles of the resolution, and the number of tracks. All momenta
    come first; the window of the loss, when given, comes last."""
    r = resolution(predicted, reference)
    rows = []
    for band, mask in conventions.momentum_bands(
            momentum_gev, momentum_window_gev, with_all=True):
        for name, _, unit in conventions.COMPONENTS:
            v = r[name][mask]
            rows.append(dict(
                kind_of_error=kind_of_error, band=band, component=name, unit=unit, number_of_tracks=int(v.size),
                resolution_percentile_16=conventions.quantile(v, 0.16),
                resolution_median=conventions.median(v),
                resolution_percentile_84=conventions.quantile(v, 0.84)))
    return rows


def signed_resolution_per_momentum_band(predicted, reference, momentum_gev, kind_of_error,
                                        momentum_window_gev=None):
    """One row per momentum band and component: the median and the 16th and
    84th percentiles of the signed resolution, half their range, and the
    number of tracks. All momenta come first; the window of the loss, when
    given, comes last."""
    r = signed_resolution(predicted, reference)
    rows = []
    for band, mask in conventions.momentum_bands(
            momentum_gev, momentum_window_gev, with_all=True):
        for name, _, unit in conventions.COMPONENTS:
            v = r[name][mask]
            low, high = conventions.quantile(v, 0.16), conventions.quantile(v, 0.84)
            rows.append(dict(
                kind_of_error=kind_of_error, band=band, component=name, unit=unit,
                number_of_tracks=int(v.size), signed_percentile_16=low,
                signed_median=conventions.median(v), signed_percentile_84=high,
                signed_half_width_68=(high - low) / 2.0))
    return rows


def _momentum_bins(momentum_gev, bins):
    """Logarithmic edges over the whole range of p, and the bin of each track;
    the top edge belongs to the last bin, as in np.histogram2d."""
    p = np.asarray(momentum_gev, dtype=float)
    positive = p > 0
    low, high = (p[positive].min(), p[positive].max()) if positive.any() else (1.0, 10.0)
    if not high > low:
        high = low * 10.0
    edges = np.geomspace(low, high, bins + 1)
    return p, positive, edges, np.minimum(np.digitize(p, edges) - 1, bins - 1)


def _percentiles_in_each_bin(v, which, bins):
    profile = {fraction: np.full(bins, np.nan) for fraction in PERCENTILES_OF_THE_RESOLUTION}
    number = np.zeros(bins, dtype=int)
    for k in range(bins):
        here = which == k
        number[k] = int(here.sum())
        if number[k]:
            for fraction in PERCENTILES_OF_THE_RESOLUTION:
                profile[fraction][k] = np.quantile(v[here], fraction)
    return profile, number


def histograms_of_signed_resolution(predicted, reference, momentum_gev, number_of_bins) -> dict:
    """For a figure: per component, a two-dimensional histogram of the
    signed resolution against the momentum, with the 16th, 50th and 84th
    percentiles of the signed resolution in every bin of momentum.

    The bins of p are logarithmic. The bins of the signed resolution are
    even in asinh(v / t), where t, the linear threshold, is the 5th
    percentile of |v|: about even within plus or minus t and logarithmic
    outside it, the same as a symmetric-log axis with that threshold. Both
    span the whole range, so every track is in the histogram.

    Returns {component: dict(unit, linear_threshold, momentum_edges_gev,
    signed_edges, counts, momentum_centres_gev, percentile_16, median,
    percentile_84, number_in_each_bin)}.
    """
    r = signed_resolution(predicted, reference)
    bins = int(number_of_bins)
    p, positive, momentum_edges, which = _momentum_bins(momentum_gev, bins)
    out = {}
    for name, _, unit in conventions.COMPONENTS:
        v = r[name]
        threshold = float(np.quantile(np.abs(v), 0.05)) if v.size else 1.0
        if not threshold > 0:
            above_zero = np.abs(v) > 0
            threshold = float(np.abs(v)[above_zero].min()) if above_zero.any() else 1.0
        low, high = (v.min(), v.max()) if v.size else (-1.0, 1.0)
        if not high > low:
            low, high = low - threshold, high + threshold
        signed_edges = threshold * np.sinh(np.linspace(np.arcsinh(low / threshold),
                                                       np.arcsinh(high / threshold), bins + 1))
        signed_edges[0], signed_edges[-1] = low, high
        counts, _, _ = np.histogram2d(p[positive], v[positive],
                                      bins=[momentum_edges, signed_edges])
        profile, number = _percentiles_in_each_bin(v, which, bins)
        out[name] = dict(
            unit=unit, linear_threshold=threshold, momentum_edges_gev=momentum_edges,
            signed_edges=signed_edges, counts=counts,
            momentum_centres_gev=np.sqrt(momentum_edges[:-1] * momentum_edges[1:]),
            percentile_16=profile[0.16], median=profile[0.50], percentile_84=profile[0.84],
            number_in_each_bin=number)
    return out


def histograms_of_resolution(predicted, reference, momentum_gev, number_of_bins) -> dict:
    """For a figure: per component, a two-dimensional histogram of the
    resolution against the momentum, both on logarithmic bins, with the
    16th, 50th and 84th percentiles of the resolution in every bin of
    momentum.

    The bins span the whole range of the momentum and of the resolution of
    that component, so every track is in the histogram. A resolution of
    exactly zero cannot be placed on a logarithmic axis and is left out of
    the histogram; it still counts in the percentiles.

    Returns {component: dict(unit, momentum_edges_gev, resolution_edges,
    counts, momentum_centres_gev, percentile_16, median, percentile_84,
    number_in_each_bin)}.
    """
    r = resolution(predicted, reference)
    bins = int(number_of_bins)
    p, positive, momentum_edges, which = _momentum_bins(momentum_gev, bins)
    out = {}
    for name, _, unit in conventions.COMPONENTS:
        v = r[name]
        above_zero = v > 0
        if above_zero.any():
            r_low, r_high = v[above_zero].min(), v[above_zero].max()
            if not r_high > r_low:
                r_high = r_low * 10.0
        else:
            r_low, r_high = 1.0, 10.0
        resolution_edges = np.geomspace(r_low, r_high, bins + 1)
        keep = positive & above_zero
        counts, _, _ = np.histogram2d(p[keep], v[keep],
                                      bins=[momentum_edges, resolution_edges])
        profile, number = _percentiles_in_each_bin(v, which, bins)
        out[name] = dict(
            unit=unit, momentum_edges_gev=momentum_edges, resolution_edges=resolution_edges,
            counts=counts, momentum_centres_gev=np.sqrt(momentum_edges[:-1] * momentum_edges[1:]),
            percentile_16=profile[0.16], median=profile[0.50], percentile_84=profile[0.84],
            number_in_each_bin=number)
    return out


def momentum_of_each_single_step(momentum_of_each_track, number_of_single_steps):
    """The momentum of the track each single step belongs to. The single
    steps are laid out track by track, every step of one track before the
    next track, as `report_of_a_run._separate_steps` lays them out."""
    p = np.asarray(momentum_of_each_track, dtype=float)
    if number_of_single_steps % len(p):
        raise ValueError("%d single steps are not a whole number of steps of %d tracks"
                         % (number_of_single_steps, len(p)))
    return np.repeat(p, number_of_single_steps // len(p))
