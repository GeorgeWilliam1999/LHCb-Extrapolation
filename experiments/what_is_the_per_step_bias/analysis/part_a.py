"""Part A of the per-step bias study: what the single-step error e is made of.

    part_a.py errors      cache e of the three runs (training, validation, test; last
                          round, and the last 5 rounds on test) and of the exact
                          2-stage Gauss-Legendre scheme (test, training)
    part_a.py a1          smooth models of e on (x, y, tx, ty, q/p, z): R^2 on validation
    part_a.py a2          binned mean of e against each variable
    part_a.py a3          e by momentum band, and the power n of e ~ (q/p)^n
    part_a.py a4          sign structure: charge, sign of y, quadrant, bend toward or
                          away from the beam; endpoint along and across the bend
    part_a.py reconcile   per-track mean of e against the spread within a track
    part_a.py jacobians   cache the reference Jacobian of every step of the test tracks
    part_a.py a5          each step's e carried to the end with the reference Jacobian
    part_a.py a6          steps that cross a z face of the field map's grid
    part_a.py a9          bias maps: correlation between forms and between snapshots

Units: um and urad (see common.py). Results go to results/, figures to figures/.
"""
from __future__ import annotations

import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

from common import (BANDS, CACHE, COMPONENTS, EQUATION, FIGURES, FIELD, NUMBER_OF_STEPS,
                    PLANES_MM, RUNS, SHORT, SNAPSHOTS_AVERAGED, START_PLANES_MM, STEP_MM,
                    TO_REPORTED, band_masks, bend_sign_of_steps, bend_sign_of_tracks,
                    cached_single_step_errors, chain, endpoint_scores, lag_one_autocorrelation,
                    last_round, network_of, on_planes, plain_step, setup_threads, split,
                    step_features, write_rows)
import stage_equations as se

FEATURES = ("x", "y", "tx", "ty", "q/p", "z")
KEYS = [key for _, key in RUNS]


def momentum_of_steps(name):
    return np.repeat(split(name).momentum_gev[:, None], NUMBER_OF_STEPS, axis=1)


def exact_errors(name):
    file = os.path.join(CACHE, "e_exact_gl2_%s.npy" % name)
    if not os.path.exists(file):
        ref = on_planes(name)
        n = len(ref)
        ends, _, worst = se.exact_step(
            ref[:, :-1].reshape(-1, 5),
            np.broadcast_to(START_PLANES_MM, (n, NUMBER_OF_STEPS)).reshape(-1))
        print("exact single steps, %s: largest scaled residual %.2e" % (name, worst))
        e = (ends.reshape(n, NUMBER_OF_STEPS, 4) - ref[:, 1:, :4]) * TO_REPORTED
        np.save(file, e)
    return np.load(file)


def last_rounds(key):
    last = last_round(key)
    return list(range(last - SNAPSHOTS_AVERAGED + 1, last + 1))


# ------------------------------------------------------------------ errors
def errors():
    for key in KEYS:
        for name in ("training", "validation", "test"):
            cached_single_step_errors(key, name)
        for r in last_rounds(key):
            cached_single_step_errors(key, "test", r)
        print("cached:", key, flush=True)
    for name in ("test", "validation", "training"):
        exact_errors(name)


# ------------------------------------------------------------------ a1
def flat(name):
    X = step_features(name).reshape(-1, 6)
    return X


def r_squared(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def model_file(key, kind):
    return os.path.join(CACHE, "a1_%s_%s.pkl" % (kind, key))


# The settings of the trees tried. A first fit with the first setting alone
# reached R^2 0.8-0.9 on the training tracks and at most 0.3 on validation: it
# learned the tracks, not the state. The setting is now chosen on the last
# fifth of the training tracks, held out, then refitted on all of them.
TREE_SETTINGS = [dict(max_leaf_nodes=63, min_samples_leaf=20, max_iter=500, learning_rate=0.1),
                 dict(max_leaf_nodes=15, min_samples_leaf=500, max_iter=300, learning_rate=0.05),
                 dict(max_leaf_nodes=7, min_samples_leaf=5000, max_iter=200, learning_rate=0.05)]


def a1():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    Xt, Xv = flat("training"), flat("validation")
    held = np.arange(len(Xt)) >= int(0.8 * len(Xt)) // NUMBER_OF_STEPS * NUMBER_OF_STEPS
    rows = []
    for series, key in RUNS:
        et = cached_single_step_errors(key, "training").reshape(-1, 4)
        ev = cached_single_step_errors(key, "validation").reshape(-1, 4)
        trees, polys = [], []
        for i, c in enumerate(COMPONENTS):
            held_r2 = []
            for setting in TREE_SETTINGS:
                tree = HistGradientBoostingRegressor(early_stopping=False, random_state=1,
                                                     **setting).fit(Xt[~held], et[~held, i])
                held_r2.append(r_squared(et[held, i], tree.predict(Xt[held])))
            best = int(np.argmax(held_r2))
            tree = HistGradientBoostingRegressor(early_stopping=False, random_state=1,
                                                 **TREE_SETTINGS[best]).fit(Xt, et[:, i])
            poly = make_pipeline(StandardScaler(), PolynomialFeatures(3), Ridge(alpha=1e-6)) \
                .fit(Xt, et[:, i])
            trees.append(tree)
            polys.append(poly)
            rows.append(dict(
                series=series, run_key=key, component=c,
                training_rows=len(Xt), validation_rows=len(Xv),
                rms_of_e_validation=float(np.sqrt((ev[:, i] ** 2).mean())),
                r2_held_out_training_tracks_per_setting=" ".join("%.3f" % r for r in held_r2),
                tree_setting_chosen=best,
                r2_trees_training=float(r_squared(et[:, i], tree.predict(Xt))),
                r2_trees_validation=float(r_squared(ev[:, i], tree.predict(Xv))),
                r2_cubic_polynomial_validation=float(r_squared(ev[:, i], poly.predict(Xv))),
                removable_if_r2_ge_0_5=bool(r_squared(ev[:, i], tree.predict(Xv)) >= 0.5)))
            print(rows[-1], flush=True)
        with open(model_file(key, "trees"), "wb") as handle:
            pickle.dump(trees, handle)
        with open(model_file(key, "cubic"), "wb") as handle:
            pickle.dump(polys, handle)
    write_rows("a1_smooth_model_r2.csv", rows)


# ------------------------------------------------------------------ a2
def a2():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X = flat("training")
    rows = []
    edges = {f: np.unique(np.quantile(X[:, j], np.linspace(0, 1, 21))) for j, f in
             enumerate(FEATURES)}
    for series, key in RUNS:
        e = cached_single_step_errors(key, "training").reshape(-1, 4)
        fig, axes = plt.subplots(4, 6, figsize=(18, 10), sharex="col")
        for j, f in enumerate(FEATURES):
            b = np.clip(np.digitize(X[:, j], edges[f][1:-1]), 0, len(edges[f]) - 2)
            for i, c in enumerate(COMPONENTS):
                means, ses, centres = [], [], []
                for k in range(len(edges[f]) - 1):
                    m = b == k
                    if m.sum() < 10:
                        continue
                    means.append(e[m, i].mean())
                    ses.append(e[m, i].std() / np.sqrt(m.sum()))
                    centres.append(np.median(X[m, j]))
                    rows.append(dict(run_key=key, variable=f, component=c, bin=k,
                                     variable_median=centres[-1], rows=int(m.sum()),
                                     mean_e=means[-1], se_of_mean=ses[-1],
                                     rms_e=float(np.sqrt((e[m, i] ** 2).mean()))))
                ax = axes[i, j]
                ax.errorbar(centres, means, ses, fmt="o-", ms=3)
                ax.axhline(0, color="grey", lw=0.5)
                if j == 0:
                    ax.set_ylabel("mean e_%s (%s)" % (c, "um" if i < 2 else "urad"))
                if i == 3:
                    ax.set_xlabel(f + (" (mm)" if f in ("x", "y", "z") else ""))
        fig.suptitle("Binned mean of the single-step error, training tracks: %s (%s)"
                     % (SHORT[key], key))
        fig.tight_layout()
        os.makedirs(FIGURES, exist_ok=True)
        fig.savefig(os.path.join(FIGURES, "a2_binned_mean_%s.png" % key), dpi=110)
        plt.close(fig)
    write_rows("a2_binned_mean_of_e.csv", rows)


# ------------------------------------------------------------------ a3
def a3():
    test = split("test")
    p = momentum_of_steps("test")
    bend = bend_sign_of_steps("test")
    qop = np.abs(on_planes("test")[:, :-1, 4])
    rows, fits = [], []
    sources = [(SHORT[k], k, cached_single_step_errors(k, "test")) for k in KEYS]
    sources.append(("exact 2-stage Gauss-Legendre", "exact", exact_errors("test")))
    for series, key, e in sources:
        along = e[..., 0] * bend
        for label, m in band_masks(test.momentum_gev):
            row = dict(series=series, run_key=key, band=label, tracks=int(m.sum()))
            row["mean_e_x_along_bend_um"] = float(along[m].mean())
            row["se_e_x_along_bend_um"] = float(along[m].std() / np.sqrt(along[m].size))
            row["fraction_of_steps_overbending"] = float((along[m] > 0).mean())
            for i, c in enumerate(COMPONENTS):
                row["mean_e_%s" % c] = float(e[m, :, i].mean())
                row["mean_abs_e_%s" % c] = float(np.abs(e[m, :, i]).mean())
                row["rms_e_%s" % c] = float(np.sqrt((e[m, :, i] ** 2).mean()))
            rows.append(row)
        # the power of |q/p|: rms of e in 12 bins of log|q/p|, fitted on a log-log line
        edges = np.quantile(qop, np.linspace(0, 1, 13))
        b = np.clip(np.digitize(qop, edges[1:-1]), 0, 11)
        for i, c in enumerate(COMPONENTS):
            centre = np.array([np.median(qop[b == k]) for k in range(12)])
            rms = np.array([np.sqrt((e[..., i][b == k] ** 2).mean()) for k in range(12)])
            mabs = np.array([np.abs(e[..., i][b == k].mean()) for k in range(12)])
            n_rms, _ = np.polyfit(np.log(centre), np.log(rms), 1)
            n_mean, _ = np.polyfit(np.log(centre), np.log(np.maximum(mabs, 1e-30)), 1)
            fits.append(dict(series=series, run_key=key, component=c,
                             power_n_of_rms_e=float(n_rms),
                             power_n_of_abs_binned_mean_e=float(n_mean),
                             rms_lowest_momentum_bin=float(rms[-1]),
                             rms_highest_momentum_bin=float(rms[0])))
    write_rows("a3_e_by_momentum_band.csv", rows)
    write_rows("a3_power_of_q_over_p.csv", fits)


# ------------------------------------------------------------------ a4
def group_rows(series, key, e, test_groups):
    rows = []
    for group, value, m in test_groups:
        row = dict(series=series, run_key=key, grouping=group, group=value,
                   steps=int(m.sum()))
        for i, c in enumerate(COMPONENTS):
            v = e[..., i][m]
            row["mean_e_%s" % c] = float(v.mean())
            row["se_e_%s" % c] = float(v.std() / np.sqrt(v.size))
        rows.append(row)
    return rows


def a4():
    ref = on_planes("test")
    s = ref[:, :-1]
    bend = bend_sign_of_steps("test")
    charge = np.sign(s[..., 4])
    toward = bend * np.sign(s[..., 0]) < 0
    groups = [("charge", "+", charge > 0), ("charge", "-", charge < 0),
              ("sign of y", "y > 0", s[..., 1] > 0), ("sign of y", "y < 0", s[..., 1] < 0),
              ("bend", "toward the beam", toward), ("bend", "away from the beam", ~toward),
              ("bend in x", "tx grows (bend +x)", bend > 0),
              ("bend in x", "tx falls (bend -x)", bend < 0)]
    for sx in (1, -1):
        for sy in (1, -1):
            groups.append(("quadrant", "x%s y%s" % ("+" if sx > 0 else "-", "+" if sy > 0 else "-"),
                           (np.sign(s[..., 0]) == sx) & (np.sign(s[..., 1]) == sy)))
    rows = []
    for series, key in RUNS:
        rows += group_rows(series, key, cached_single_step_errors(key, "test"), groups)
    rows += group_rows("exact 2-stage Gauss-Legendre", "exact", exact_errors("test"), groups)
    write_rows("a4_sign_structure_of_steps.csv", rows)

    # the endpoint along and across the bend, per band (the addendum of the brief)
    test = split("test")
    tb = bend_sign_of_tracks("test")
    rows = []
    for series, key in RUNS:
        final = chain(plain_step(network_of(key)), test.start_state)[:, -1]
        _, d = endpoint_scores(final)
        along, across = d[:, 0] * tb, d[:, 1] * np.sign(test.reference_end_state[:, 1])
        for label, m in [("all", np.ones(len(d), bool))] + band_masks(test.momentum_gev):
            rows.append(dict(
                series=series, run_key=key, band=label, tracks=int(m.sum()),
                mean_x_along_bend_um=float(along[m].mean()),
                se_x_along_bend_um=float(along[m].std(ddof=1) / np.sqrt(m.sum())),
                median_x_along_bend_um=float(np.median(along[m])),
                fraction_bending_too_much=float((along[m] > 0).mean()),
                mean_y_away_from_midplane_um=float(across[m].mean()),
                se_y_away_from_midplane_um=float(across[m].std(ddof=1) / np.sqrt(m.sum())),
                mean_y_um=float(d[m, 1].mean()),
                se_y_um=float(d[m, 1].std(ddof=1) / np.sqrt(m.sum())),
                rms_x_um=float(np.sqrt((d[m, 0] ** 2).mean())),
                rms_y_um=float(np.sqrt((d[m, 1] ** 2).mean()))))
    write_rows("a4_endpoint_along_and_across_the_bend.csv", rows)


# ------------------------------------------------------------------ reconcile
def lag_autocorrelation(errors, lag):
    centred = errors - errors.mean(axis=1, keepdims=True)
    return (centred[:, :-lag] * centred[:, lag:]).sum(axis=1) / (centred ** 2).sum(axis=1)


def reconcile():
    """How a lag-1 autocorrelation of 0.85-0.90 coexists with a population
    median near zero: split the variance of e into a per-track mean and the
    spread within a track."""
    bend = bend_sign_of_steps("test")
    rows = []
    sources = [(SHORT[k], k, cached_single_step_errors(k, "test")) for k in KEYS]
    for series, key, e in sources:
        for i, c in enumerate(COMPONENTS):
            v = e[..., i] * (bend if c in ("x", "tx") else 1.0)
            m = v.mean(axis=1)
            # the standard error of a per-track mean, with the number of steps cut to
            # the effective number n (1 - r) / (1 + r) of the track's lag-1 autocorrelation r
            r1 = np.clip(lag_one_autocorrelation(e[..., i]), -0.99, 0.99)
            n_eff = np.maximum(v.shape[1] * (1 - r1) / (1 + r1), 1.0)
            se_m = v.std(axis=1, ddof=1) / np.sqrt(n_eff)
            rows.append(dict(
                series=series, run_key=key,
                component=c + (" along the bend" if c in ("x", "tx") else ""),
                population_mean=float(v.mean()), population_median=float(np.median(v)),
                population_rms=float(np.sqrt((v ** 2).mean())),
                share_of_variance_in_per_track_means=float(m.var() / v.var()),
                fraction_of_tracks_with_positive_mean=float((m > 0).mean()),
                fraction_of_tracks_mean_beyond_3_se=float((np.abs(m) > 3 * se_m).mean()),
                median_abs_per_track_mean=float(np.median(np.abs(m))),
                lag1_median=float(np.median(lag_one_autocorrelation(e[..., i]))),
                lag8_median=float(np.median(lag_autocorrelation(e[..., i], 8))),
                lag16_median=float(np.median(lag_autocorrelation(e[..., i], 16))),
                lag32_median=float(np.median(lag_autocorrelation(e[..., i], 32))),
                sign_changes_per_track_median=float(np.median(
                    (np.diff(np.sign(e[..., i]), axis=1) != 0).sum(axis=1)))))
            print(rows[-1], flush=True)
    write_rows("a4b_per_track_bias_reconciliation.csv", rows)


# ------------------------------------------------------------------ a5
def _jacobian_of_one_step(k):
    """J (n, 4, 5): d(reference state on plane k+1)/d(state on plane k), by
    central differences on the sixth-order method at 0.1 mm."""
    from rkpinn.integrators.runge_kutta_sixth_order import integrate_with_sixth_order
    s = on_planes("test")[:, k]
    n = len(s)
    h = np.array([1e-3, 1e-3, 1e-6, 1e-6, 0.0])
    h[4] = 1e-4
    states = []
    for j in range(5):
        for sign in (1.0, -1.0):
            t = s.copy()
            t[:, j] += sign * (h[j] * (np.abs(s[:, 4]) if j == 4 else 1.0))
            states.append(t)
    out = integrate_with_sixth_order(EQUATION, np.concatenate(states), PLANES_MM[k],
                                     PLANES_MM[k + 1], step_length_mm=0.1)
    out = out.reshape(10, n, 5)
    J = np.empty((n, 5, 5))
    for j in range(5):
        step = 2 * h[j] * (np.abs(s[:, 4]) if j == 4 else 1.0)
        J[:, :, j] = (out[2 * j] - out[2 * j + 1]) / (step[:, None] if np.ndim(step) else step)
    return J


def step_jacobians():
    """J (n, 64, 5, 5) of the reference over each step of the test tracks, cached."""
    file = os.path.join(CACHE, "jacobians_of_the_steps_test.npy")
    if not os.path.exists(file):
        with Pool(int(os.environ.get("WORKERS", "16"))) as pool:
            J = pool.map(_jacobian_of_one_step, range(NUMBER_OF_STEPS))
        np.save(file, np.stack(J, axis=1))
    return np.load(file)


def carried_to_the_end(J):
    """T (n, 64, 5, 5): from the end of step k to the last plane."""
    n = len(J)
    T = np.empty_like(J)
    T[:, -1] = np.eye(5)
    for k in range(NUMBER_OF_STEPS - 2, -1, -1):
        T[:, k] = T[:, k + 1] @ J[:, k + 1]
    return T


def a5():
    J = step_jacobians()
    T = carried_to_the_end(J)[..., :4, :4]                       # (n, 64, 4, 4)
    test = split("test")
    rows, zrows = [], []
    bins = np.arange(NUMBER_OF_STEPS) // 8
    for series, key in RUNS:
        e = cached_single_step_errors(key, "test") / TO_REPORTED       # mm, slope
        c = np.einsum("nkab,nkb->nka", T, e) * TO_REPORTED              # contributions (n,64,4)
        total = c.sum(axis=1)
        final = chain(plain_step(network_of(key)), test.start_state)[:, -1]
        _, d = endpoint_scores(final)
        row = dict(series=series, run_key=key)
        for i, comp in enumerate(("x", "y")):
            row["correlation_carried_sum_vs_chained_%s" % comp] = float(
                np.corrcoef(total[:, i], d[:, i])[0, 1])
            row["slope_chained_on_carried_%s" % comp] = float(
                np.polyfit(total[:, i], d[:, i], 1)[0])
        row["carried_sum_radial_median_um"] = float(np.median(np.hypot(total[:, 0], total[:, 1])))
        row["chained_radial_median_um"] = float(np.median(np.hypot(d[:, 0], d[:, 1])))
        row["difference_radial_median_um"] = float(
            np.median(np.hypot(total[:, 0] - d[:, 0], total[:, 1] - d[:, 1])))
        rows.append(row)
        print(row, flush=True)
        for b in range(8):
            cb = c[:, bins == b].sum(axis=1)
            zrow = dict(series=series, run_key=key, z_from_mm=float(START_PLANES_MM[8 * b]),
                        z_to_mm=float(PLANES_MM[8 * b + 8]))
            for i, comp in enumerate(("x", "y")):
                zrow["share_of_endpoint_variance_%s" % comp] = float(
                    np.cov(cb[:, i], total[:, i])[0, 1] / total[:, i].var(ddof=1))
                zrow["rms_contribution_%s_um" % comp] = float(np.sqrt((cb[:, i] ** 2).mean()))
            zrows.append(zrow)
    write_rows("a5_carried_sum_vs_chained_endpoint.csv", rows)
    write_rows("a5_endpoint_share_by_z.csv", zrows)


# ------------------------------------------------------------------ a6
def face_index(v, axis):
    return np.floor((v - FIELD.lowest_corner[axis]) * FIELD.inverse_cell_size[axis])


def a6():
    ref = on_planes("test")
    s, t = ref[:, :-1], ref[:, 1:]
    z_cross = face_index(PLANES_MM[:-1], 2) != face_index(PLANES_MM[1:], 2)
    z_cross = np.broadcast_to(z_cross, s.shape[:2])
    xy_cross = ((face_index(s[..., 0], 0) != face_index(t[..., 0], 0))
                | (face_index(s[..., 1], 1) != face_index(t[..., 1], 1)))
    sources = [(SHORT[k], k, cached_single_step_errors(k, "test")) for k in KEYS]
    sources.append(("exact 2-stage Gauss-Legendre", "exact", exact_errors("test")))
    rows = []
    for series, key, e in sources:
        radial = np.hypot(e[..., 0], e[..., 1])
        for what, m in (("crosses a z face", z_cross), ("crosses no z face", ~z_cross),
                        ("crosses an x or y face", xy_cross),
                        ("crosses no face at all", ~z_cross & ~xy_cross)):
            rows.append(dict(series=series, run_key=key, steps=what,
                             fraction_of_steps=float(m.mean()),
                             mean_radial_um=float(radial[m].mean()),
                             median_radial_um=float(np.median(radial[m])),
                             mean_abs_e_tx_urad=float(np.abs(e[..., 2][m]).mean()),
                             mean_abs_e_ty_urad=float(np.abs(e[..., 3][m]).mean())))
    write_rows("a6_cell_face_crossings.csv", rows)


# ------------------------------------------------------------------ a9
def bias_map(e, name="test", q_bins=8, z_bins=8):
    """Mean e in bins of signed q/p (quantiles) x z (8 groups of 8 steps); (cells, 4)."""
    feat = step_features(name)
    qop = feat[..., 4]
    edges = np.quantile(qop, np.linspace(0, 1, q_bins + 1))
    qb = np.clip(np.digitize(qop, edges[1:-1]), 0, q_bins - 1)
    zb = np.broadcast_to(np.arange(NUMBER_OF_STEPS) // (NUMBER_OF_STEPS // z_bins), qop.shape)
    cell = qb * z_bins + zb
    return np.stack([[e[..., i][cell == c].mean() for i in range(4)]
                     for c in range(q_bins * z_bins)])


def map_correlations(maps):
    return [float(np.corrcoef(maps[0][:, i], maps[1][:, i])[0, 1]) for i in range(4)]


def a9():
    rows = []
    maps = {k: bias_map(cached_single_step_errors(k, "test")) for k in KEYS}
    full = {k: cached_single_step_errors(k, "test") for k in KEYS}
    for a in range(3):
        for b in range(a + 1, 3):
            ka, kb = KEYS[a], KEYS[b]
            m = map_correlations([maps[ka], maps[kb]])
            for i, c in enumerate(COMPONENTS):
                rows.append(dict(comparison="between forms", first=SHORT[ka], second=SHORT[kb],
                                 component=c, bias_map_correlation=m[i],
                                 step_by_step_correlation=float(np.corrcoef(
                                     full[ka][..., i].ravel(), full[kb][..., i].ravel())[0, 1]),
                                 shared_if_above_0_7=bool(m[i] > 0.7)))
    for series, key in RUNS:
        es = [cached_single_step_errors(key, "test", r) for r in last_rounds(key)]
        ms = [bias_map(e) for e in es]
        for i, c in enumerate(COMPONENTS):
            pairs = [(a, b) for a in range(len(es)) for b in range(a + 1, len(es))]
            mc = [np.corrcoef(ms[a][:, i], ms[b][:, i])[0, 1] for a, b in pairs]
            sc = [np.corrcoef(es[a][..., i].ravel(), es[b][..., i].ravel())[0, 1]
                  for a, b in pairs]
            rows.append(dict(comparison="between the last 5 snapshots of a run",
                             first=SHORT[key], second="rounds %s" % " ".join(
                                 str(r) for r in last_rounds(key)), component=c,
                             bias_map_correlation=float(np.mean(mc)),
                             step_by_step_correlation=float(np.mean(sc)),
                             shared_if_above_0_7=bool(np.mean(mc) > 0.7),
                             lowest_pair_bias_map=float(np.min(mc)),
                             lowest_pair_step_by_step=float(np.min(sc))))
    # each run against the exact scheme, and the part of e that is not the exact
    # scheme's own error (network minus exact, both one step from the reference state)
    ex = exact_errors("test")
    exmap = bias_map(ex)
    own = {k: full[k] - ex for k in KEYS}
    ownmap = {k: bias_map(own[k]) for k in KEYS}
    for series, key in RUNS:
        for i, c in enumerate(COMPONENTS):
            rows.append(dict(comparison="run against the exact scheme", first=SHORT[key],
                             second="exact 2-stage Gauss-Legendre", component=c,
                             bias_map_correlation=float(np.corrcoef(maps[key][:, i],
                                                                    exmap[:, i])[0, 1]),
                             step_by_step_correlation=float(np.corrcoef(
                                 full[key][..., i].ravel(), ex[..., i].ravel())[0, 1]),
                             shared_if_above_0_7=bool(np.corrcoef(maps[key][:, i],
                                                                  exmap[:, i])[0, 1] > 0.7),
                             share_of_variance_of_e_not_exact=float(
                                 own[key][..., i].var() / full[key][..., i].var())))
    for a in range(3):
        for b in range(a + 1, 3):
            ka, kb = KEYS[a], KEYS[b]
            for i, c in enumerate(COMPONENTS):
                mc = float(np.corrcoef(ownmap[ka][:, i], ownmap[kb][:, i])[0, 1])
                rows.append(dict(comparison="between forms, e minus the exact scheme's error",
                                 first=SHORT[ka], second=SHORT[kb], component=c,
                                 bias_map_correlation=mc,
                                 step_by_step_correlation=float(np.corrcoef(
                                     own[ka][..., i].ravel(), own[kb][..., i].ravel())[0, 1]),
                                 shared_if_above_0_7=bool(mc > 0.7)))
    for series, key in RUNS:
        es = [cached_single_step_errors(key, "test", r) - ex for r in last_rounds(key)]
        ms = [bias_map(e) for e in es]
        pairs = [(a, b) for a in range(len(es)) for b in range(a + 1, len(es))]
        for i, c in enumerate(COMPONENTS):
            mc = [np.corrcoef(ms[a][:, i], ms[b][:, i])[0, 1] for a, b in pairs]
            sc = [np.corrcoef(es[a][..., i].ravel(), es[b][..., i].ravel())[0, 1]
                  for a, b in pairs]
            rows.append(dict(comparison="between the last 5 snapshots, e minus the exact "
                                        "scheme's error", first=SHORT[key],
                             second="rounds %s" % " ".join(str(r) for r in last_rounds(key)),
                             component=c, bias_map_correlation=float(np.mean(mc)),
                             step_by_step_correlation=float(np.mean(sc)),
                             shared_if_above_0_7=bool(np.mean(mc) > 0.7),
                             lowest_pair_bias_map=float(np.min(mc)),
                             lowest_pair_step_by_step=float(np.min(sc))))
    for row in rows:
        row.setdefault("share_of_variance_of_e_not_exact", "")
        row.setdefault("lowest_pair_bias_map", "")
        row.setdefault("lowest_pair_step_by_step", "")
    write_rows("a9_bias_map_correlations.csv", rows)


if __name__ == "__main__":
    setup_threads()
    {"errors": errors, "jacobians": step_jacobians, "a1": a1, "a2": a2, "a3": a3, "a4": a4, "reconcile": reconcile,
     "a5": a5, "a6": a6, "a9": a9}[sys.argv[1]]()
