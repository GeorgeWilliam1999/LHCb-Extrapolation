"""The conventions of every result: units, momentum bands, statistics.

They are defined here and nowhere else.

Units
    positions       micrometres
    slopes          no unit: tx = dx/dz and ty = dy/dz are pure numbers. A slope
                    error is the raw difference. It is never multiplied by 1,000.

Momentum bands, in GeV
    below 3, 3 to 8, 8 to 20, 20 to 50, above 50.
    A track is in a band when  lower edge <= momentum < upper edge.
    The window of the loss of a run is reported beside the bands, as an extra
    row. It overlaps them on purpose: it is not part of the partition.

Statistics of a signed error v
    median_of_absolute          median of |v|
    percentile_95_of_absolute   95th percentile of |v|
    signed_median               median of v: a bias
    half_width_68               (84th percentile - 16th percentile) / 2 of v:
                                a spread that the tails do not pull
    root_mean_square            sqrt(mean of v squared): pulled by the tails,
                                quoted beside the median

Magnitudes of a position error
    radial                  sqrt(dx^2 + dy^2)
    larger_of_x_and_y       max(|dx|, |dy|)

Naming an error
    endpoint error      after the whole chain, on the last plane
    single-step error   after one application of the network

Ported from `Self_chained_paper/scripts/common.py`, with the arithmetic
unchanged. The bands and the units are those of the second mini-paper; the
frozen analysis code used other bands and multiplied slopes by 1,000.
Gates: tests/test_evaluation_reproduces_the_paper.py.
"""
from __future__ import annotations

import numpy as np

MICROMETRES_PER_MILLIMETRE = 1e3
ONE_MILLIMETRE_IN_MICROMETRES = 1000.0

EDGES_OF_MOMENTUM_BANDS_GEV = (0.0, 3.0, 8.0, 20.0, 50.0, float("inf"))
MOMENTUM_BANDS = ("below 3 GeV", "3 to 8 GeV", "8 to 20 GeV", "20 to 50 GeV",
                  "above 50 GeV")
ALL_MOMENTA = "all momenta"

# (name, place in the state, unit)
COMPONENTS = (("x", 0, "micrometres"), ("y", 1, "micrometres"),
              ("slope in x", 2, "no unit"), ("slope in y", 3, "no unit"))

STATISTICS = ("median_of_absolute", "percentile_95_of_absolute", "signed_median",
              "half_width_68", "root_mean_square")

MAGNITUDES = ("radial", "larger_of_x_and_y")

QUANTILES_OF_A_DISTRIBUTION = (0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99)


def name_of_window(momentum_window_gev) -> str:
    low, high = momentum_window_gev
    return "%g to %g GeV (window of the loss)" % (low, high)


def momentum_bands(momentum_gev, momentum_window_gev=None, with_all=False):
    """[(name, mask)] for the five bands, then the window of the loss if one
    is given. `with_all` puts all momenta first."""
    p = np.asarray(momentum_gev, dtype=float)
    bands = []
    if with_all:
        bands.append((ALL_MOMENTA, np.ones(p.shape, dtype=bool)))
    for low, high, name in zip(EDGES_OF_MOMENTUM_BANDS_GEV[:-1],
                               EDGES_OF_MOMENTUM_BANDS_GEV[1:], MOMENTUM_BANDS):
        bands.append((name, (p >= low) & (p < high)))
    if momentum_window_gev is not None:
        low, high = momentum_window_gev
        bands.append((name_of_window(momentum_window_gev), (p >= low) & (p < high)))
    return bands


# ---------------------------------------------------------------------- statistics
def _of(v):
    return np.asarray(v, dtype=float)


def median(v):
    v = _of(v)
    return float(np.median(v)) if v.size else float("nan")


def quantile(v, fraction):
    v = _of(v)
    return float(np.quantile(v, fraction)) if v.size else float("nan")


def median_of_absolute(v):
    v = _of(v)
    return float(np.median(np.abs(v))) if v.size else float("nan")


def percentile_95_of_absolute(v):
    v = _of(v)
    return float(np.quantile(np.abs(v), 0.95)) if v.size else float("nan")


def signed_median(v):
    return median(v)


def half_width_68(v):
    v = _of(v)
    if v.size == 0:
        return float("nan")
    low, high = np.quantile(v, [0.16, 0.84])
    return float((high - low) / 2.0)


def root_mean_square(v):
    v = _of(v)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")


def statistics_of_signed_error(v) -> dict:
    """The five statistics of a signed error, with the number of tracks."""
    return {
        "number_of_tracks": int(_of(v).size),
        "median_of_absolute": median_of_absolute(v),
        "percentile_95_of_absolute": percentile_95_of_absolute(v),
        "signed_median": signed_median(v),
        "half_width_68": half_width_68(v),
        "root_mean_square": root_mean_square(v),
    }


def statistics_of_magnitude(r) -> dict:
    """The statistics of a magnitude in micrometres, which is never negative."""
    r = _of(r)
    empty = r.size == 0
    return {
        "number_of_tracks": int(r.size),
        "median_micrometres": median(r),
        "percentile_95_micrometres": quantile(r, 0.95),
        "percentile_99_micrometres": quantile(r, 0.99),
        "mean_micrometres": float("nan") if empty else float(r.mean()),
        "largest_micrometres": float("nan") if empty else float(r.max()),
        "fraction_above_1_mm": float("nan") if empty else float(
            (r > ONE_MILLIMETRE_IN_MICROMETRES).mean()),
        "number_above_1_mm": int((r > ONE_MILLIMETRE_IN_MICROMETRES).sum()),
    }


# -------------------------------------------------------------------------- errors
def errors(predicted, reference) -> dict:
    """The signed errors of states, predicted minus reference, in the units
    of the conventions.

    predicted, reference   shape (n, 4 or more), in millimetres and slopes
    Returns the signed error of each component under its name, and the
    magnitudes "radial", "larger_of_x_and_y" and "larger_of_slopes".
    """
    predicted, reference = _of(predicted), _of(reference)
    dx = (predicted[:, 0] - reference[:, 0]) * MICROMETRES_PER_MILLIMETRE
    dy = (predicted[:, 1] - reference[:, 1]) * MICROMETRES_PER_MILLIMETRE
    dtx = predicted[:, 2] - reference[:, 2]
    dty = predicted[:, 3] - reference[:, 3]
    return {"x": dx, "y": dy, "slope in x": dtx, "slope in y": dty,
            "radial": np.hypot(dx, dy),
            "larger_of_x_and_y": np.maximum(np.abs(dx), np.abs(dy)),
            "larger_of_slopes": np.maximum(np.abs(dtx), np.abs(dty))}


def name_of_network(number_of_steps, number_of_stages, step_length_mm) -> str:
    """How a figure or a table names a network: by its steps, stages and step length."""
    if number_of_stages:
        return "%d steps of %.1f mm, %d stages" % (number_of_steps, step_length_mm,
                                                   number_of_stages)
    return "%d step of %.1f mm, no stages" % (number_of_steps, step_length_mm)
