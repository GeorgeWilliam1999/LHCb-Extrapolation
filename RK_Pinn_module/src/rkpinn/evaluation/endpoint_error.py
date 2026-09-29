"""The endpoint error: the error on the last plane, after the whole chain.

Standard outputs 1, 2 and 3 of `PACKAGE_PLAN.md`, section 5:

  1  the endpoint error per component, all momenta
  2  the endpoint error per component and per momentum band
  3  the signed distribution per component: whether the error is a bias or a
     spread

All three come from one comparison, the predicted end states against the
reference end states, so they are one set of tables.

Ported from `Self_chained_paper/scripts/numbers.py` (`chain_block`,
`sec_headline`), with the arithmetic unchanged.
Gates: tests/test_evaluation_reproduces_the_paper.py.
"""
from __future__ import annotations

import numpy as np

from rkpinn.evaluation import conventions


def errors_per_component(predicted, reference, momentum_gev, momentum_window_gev=None):
    """One row per band and component: the five statistics of the signed error.
    All momenta come first."""
    e = conventions.errors(predicted, reference)
    rows = []
    for band, mask in conventions.momentum_bands(momentum_gev, momentum_window_gev,
                                                 with_all=True):
        for name, _, unit in conventions.COMPONENTS:
            rows.append(dict(band=band, component=name, unit=unit,
                             **conventions.statistics_of_signed_error(e[name][mask])))
    return rows


def magnitudes_of_position_error(predicted, reference, momentum_gev,
                                 momentum_window_gev=None):
    """One row per band and magnitude, radial and the larger of x and y."""
    e = conventions.errors(predicted, reference)
    rows = []
    for band, mask in conventions.momentum_bands(momentum_gev, momentum_window_gev,
                                                 with_all=True):
        for magnitude in conventions.MAGNITUDES:
            rows.append(dict(band=band, magnitude=magnitude,
                             **conventions.statistics_of_magnitude(e[magnitude][mask])))
    return rows


def signed_distributions(predicted, reference, momentum_gev, momentum_window_gev=None):
    """One row per band and component: quantiles of the signed error, from the
    1st to the 99th percentile."""
    e = conventions.errors(predicted, reference)
    rows = []
    for band, mask in conventions.momentum_bands(momentum_gev, momentum_window_gev,
                                                 with_all=True):
        for name, _, unit in conventions.COMPONENTS:
            v = e[name][mask]
            row = dict(band=band, component=name, unit=unit, number_of_tracks=int(v.size))
            for fraction in conventions.QUANTILES_OF_A_DISTRIBUTION:
                row["percentile_%02d" % round(100 * fraction)] = conventions.quantile(
                    v, fraction)
            row["fraction_positive"] = float((v > 0).mean()) if v.size else float("nan")
            rows.append(row)
    return rows


def histograms_of_signed_error(predicted, reference, number_of_bins, mask=None):
    """For a figure: per component, the edges and counts of a histogram of the
    signed error, over the central 98 % of the errors."""
    e = conventions.errors(predicted, reference)
    out = {}
    for name, _, unit in conventions.COMPONENTS:
        v = e[name] if mask is None else e[name][mask]
        low, high = np.quantile(v, [0.01, 0.99]) if v.size else (0.0, 1.0)
        counts, edges = np.histogram(v, bins=int(number_of_bins), range=(low, high))
        out[name] = dict(unit=unit, edges=edges, counts=counts)
    return out
