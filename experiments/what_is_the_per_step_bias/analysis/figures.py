"""The figures of the results note, from the tables in results/ and the cache.

    figures.py      writes every figure of the note into figures/
    figures.py <name>...  only the figures named (the function names below)
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402

from common import (FIGURES, RESULTS, RUNS, SHORT, START_PLANES_MM, BANDS,   # noqa: E402
                    COMPONENTS, bend_sign_of_steps, cached_single_step_errors, split)
from part_a import bias_map, exact_errors, lag_autocorrelation          # noqa: E402

# categorical slots 1-4 of the dataviz palette; the exact scheme is grey
COLOURS = {"25dfa17ef168": "#2a78d6", "9cef9c160b7d": "#eb6834", "49e5e2450eaa": "#1baf7a"}
EXACT = "#5f5e5a"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
UNIT = {"x": "um", "y": "um", "tx": "urad", "ty": "urad"}
BAND_LABELS = [b[0] for b in BANDS]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 7,
                     "font.size": 10, "legend.frameon": False})


def read(file):
    with open(os.path.join(RESULTS, file)) as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name), dpi=120)
    plt.close(fig)
    print("written: figures/%s" % name)


def along_the_track():
    """e_x along the bend against z for six test tracks of 3-8 GeV, each run and the
    exact scheme: the error wanders slowly along a track, with a sign of its own."""
    test = split("test")
    pick = np.flatnonzero((test.momentum_gev > 3) & (test.momentum_gev < 8))[:6]
    bend = bend_sign_of_steps("test")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, t in zip(axes.ravel(), pick):
        for _, key in RUNS:
            e = cached_single_step_errors(key, "test")
            ax.plot(START_PLANES_MM, e[t, :, 0] * bend[t], color=COLOURS[key], label=SHORT[key])
        ax.plot(START_PLANES_MM, exact_errors("test")[t, :, 0] * bend[t], "k:", label="exact scheme")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title("test track %d, p = %.1f GeV" % (t, test.momentum_gev[t]))
    for ax in axes[1]:
        ax.set_xlabel("z of the start of the step (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("e_x along the bend (um)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Single-step error in x along the bend, along six tracks")
    fig.tight_layout()
    save(fig, "along_the_track.png")


def share_by_z():
    rows = read("a5_endpoint_share_by_z.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, comp in zip(axes, ("x", "y")):
        for _, key in RUNS:
            r = [x for x in rows if x["run_key"] == key]
            centres = [0.5 * (float(x["z_from_mm"]) + float(x["z_to_mm"])) for x in r]
            ax.plot(centres, [float(x["share_of_endpoint_variance_%s" % comp]) for x in r],
                    "o-", color=COLOURS[key], label=SHORT[key])
        ax.axhline(1 / 8, color="grey", ls="--", lw=0.8, label="even share (1/8)")
        ax.set_xlabel("z bin of 8 steps (mm)")
        ax.set_title("share of the variance of the endpoint error in %s" % comp)
    axes[0].set_ylabel("share")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "a5_share_by_z.png")


def toy_fit():
    rows = [r for r in read("a7_toy_fit_by_band.csv") if r["band"] != "all"]
    labels = [b[0] for b in BANDS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(labels))
    for j, (_, key) in enumerate(RUNS):
        r = {row["band"]: row for row in rows if row["run_key"] == key}
        ratio = [float(r[b]["mean_bias_over_resolution"]) for b in labels]
        width = [float(r[b]["pull_width_network"]) for b in labels]
        axes[0].bar(x + (j - 1) * 0.27, ratio, 0.27, color=COLOURS[key], label=SHORT[key])
        axes[1].bar(x + (j - 1) * 0.27, width, 0.27, color=COLOURS[key], label=SHORT[key])
    for s in (0.1, -0.1):
        axes[0].axhline(s, color="red", ls="--", lw=0.8)
    axes[0].set_ylabel("mean bias dp/p / momentum resolution")
    axes[1].axhline(1.0, color="black", lw=0.8)
    axes[1].set_ylabel("pull width of q/p, network fit")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("momentum band")
    axes[0].legend(fontsize=8)
    fig.suptitle("Toy UT + SciFi fit: the network against the reference (hits only, no scattering)")
    fig.tight_layout()
    save(fig, "a7_toy_fit.png")


def variants():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    names = []
    for j, (_, key) in enumerate(RUNS):
        file = "c_variants_%s.csv" % key
        if not os.path.exists(os.path.join(RESULTS, file)):
            continue
        rows = read(file)
        names = [r["variant"] for r in rows]
        y = np.arange(len(rows))
        ax.barh(y + (j - 1) * 0.27, [float(r["endpoint_radial_median_um"]) for r in rows], 0.27,
                color=COLOURS[key], label=SHORT[key])
    comparators = {r["series"]: float(r["endpoint_radial_median_um"]) for r in read("c_comparators.csv")}
    for name, value in comparators.items():
        ax.axvline(value, ls="--", color="black" if "RK4" in name else "grey", lw=0.8)
        ax.text(value, len(names) - 0.4, " " + name.split(" (")[0], fontsize=7, rotation=90,
                va="top")
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("test endpoint radial median (um)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "c_variants.png")


def a1_r_squared():
    """Validation R^2 of the trees model of e, per run and component, against the
    "removable" bar of 0.5."""
    rows = read("a1_smooth_model_r2.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(COMPONENTS))
    for j, (_, key) in enumerate(RUNS):
        r = {row["component"]: row for row in rows if row["run_key"] == key}
        ax.bar(x + (j - 1) * 0.26, [float(r[c]["r2_trees_validation"]) for c in COMPONENTS],
               0.24, color=COLOURS[key], label=SHORT[key])
    ax.axhline(0.5, color="#d03b3b", ls="--", lw=1.2)
    ax.text(len(COMPONENTS) - 0.5, 0.52, "removable if above", ha="right", fontsize=9,
            color="#d03b3b")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, ["e_" + c for c in COMPONENTS])
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel("R² on validation tracks")
    ax.set_title("A.1  A smooth model of the state does not predict e")
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.92))
    fig.tight_layout()
    save(fig, "a1_r_squared.png")


def a3_rms_by_band():
    """rms of e per momentum band, per run and the exact scheme (log scale)."""
    rows = read("a3_e_by_momentum_band.csv")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True)
    x = np.arange(len(BAND_LABELS))
    series = [(SHORT[k], k, COLOURS[k], "o-") for _, k in RUNS]
    for ax, c in zip(axes, COMPONENTS):
        for name, key, colour, style in series:
            r = {row["band"]: row for row in rows if row["run_key"] == key}
            ax.plot(x, [float(r[b]["rms_e_%s" % c]) for b in BAND_LABELS], style, color=colour,
                    label=name)
        r = {row["band"]: row for row in rows if row["series"].startswith("exact")}
        if r:
            ax.plot(x, [max(float(r[b]["rms_e_%s" % c]), 1e-4) for b in BAND_LABELS], "s:",
                    color=EXACT, label="exact scheme")
        ax.set_yscale("log")
        ax.set_xticks(x, BAND_LABELS, rotation=30)
        ax.set_title("rms e_%s (%s)" % (c, UNIT[c]))
    axes[0].legend(fontsize=8)
    fig.suptitle("A.3  Size of the single-step error by momentum band: in tx and ty every "
                 "network sits on the exact scheme")
    fig.tight_layout()
    save(fig, "a3_rms_by_band.png")


def a4_endpoint_along_the_bend():
    """Mean signed endpoint error along the bend and away from y = 0, ± standard
    error, per band: near zero against the spread of the endpoint error."""
    rows = [r for r in read("a4_endpoint_along_and_across_the_bend.csv") if r["band"] != "all"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
    x = np.arange(len(BAND_LABELS))
    for j, (_, key) in enumerate(RUNS):
        r = {row["band"]: row for row in rows if row["run_key"] == key}
        for ax, m, se in ((axes[0], "mean_x_along_bend_um", "se_x_along_bend_um"),
                          (axes[1], "mean_y_away_from_midplane_um", "se_y_away_from_midplane_um")):
            ax.errorbar(x + (j - 1) * 0.15, [float(r[b][m]) for b in BAND_LABELS],
                        yerr=[float(r[b][se]) for b in BAND_LABELS], fmt="o", capsize=3,
                        color=COLOURS[key], label=SHORT[key])
    for ax, title in zip(axes, ("x along the bend (+ = bends too much)",
                                "y away from y = 0 (+ = too far out)")):
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, BAND_LABELS)
        ax.set_xlabel("momentum band")
        ax.set_title(title)
    axes[0].set_ylabel("mean signed endpoint error ± SE (um)")
    axes[0].legend(fontsize=8)
    fig.suptitle("A.4  Mean signed endpoint error per momentum band, ± standard error")
    fig.tight_layout()
    save(fig, "a4_endpoint_along_the_bend.png")


def a4b_autocorrelation():
    """Autocorrelation of e along a track against the lag: high at lag 1, gone by
    lag 8. A slow wave, not a constant offset."""
    bend = bend_sign_of_steps("test")
    lags = np.arange(1, 33)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for _, key in RUNS:
        e = cached_single_step_errors(key, "test")
        for ax, values in ((axes[0], e[..., 0] * bend), (axes[1], e[..., 1])):
            ax.plot(lags, [np.median(lag_autocorrelation(values, k)) for k in lags],
                    color=COLOURS[key], label=SHORT[key])
    ex = exact_errors("test")
    for ax, values in ((axes[0], ex[..., 0] * bend), (axes[1], ex[..., 1])):
        ax.plot(lags, [np.median(lag_autocorrelation(values, k)) for k in lags], ":",
                color=EXACT, label="exact scheme")
    for ax, title in zip(axes, ("e_x along the bend", "e_y")):
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("lag (steps of 80.9 mm)")
        ax.set_title(title)
    axes[0].set_ylabel("autocorrelation along a track (median over tracks)")
    axes[0].legend(fontsize=8)
    fig.suptitle("A.4b  Neighbouring steps err the same way; steps 8 apart do not")
    fig.tight_layout()
    save(fig, "a4b_autocorrelation.png")


def a6_cell_faces():
    """Mean |e| on steps that cross a z face of the field map and on steps that do
    not: the slope error of every network follows the faces, like the exact scheme;
    the position error does not."""
    rows = read("a6_cell_face_crossings.csv")
    keys = ["exact"] + [k for _, k in RUNS]
    series = ["exact scheme"] + [SHORT[k] for _, k in RUNS]
    colours = [EXACT] + [COLOURS[k] for _, k in RUNS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(2)
    for j, (key, name, colour) in enumerate(zip(keys, series, colours)):
        r = {row["steps"]: row for row in rows if row["run_key"] == key}
        groups = ["crosses a z face", "crosses no z face"]
        for ax, column in ((axes[0], "mean_radial_um"), (axes[1], "mean_abs_e_tx_urad")):
            ax.bar(x + (j - 1.5) * 0.2, [float(r[g][column]) for g in groups], 0.18,
                   color=colour, label=name)
    for ax, title in zip(axes, ("mean radial position error (um)", "mean |e_tx| (urad)")):
        ax.set_yscale("log")
        ax.set_xticks(x, ["step crosses a z face", "step crosses no z face"])
        ax.set_title(title)
    axes[1].legend(fontsize=8)
    fig.suptitle("A.6  Cell faces of the field map make the slope error, not the position error")
    fig.tight_layout()
    save(fig, "a6_cell_faces.png")


def a8_seed_correlations():
    """Bias-map correlation of every pair of seeds, per form and component: of e,
    and of e minus the exact scheme's error (the network's own part)."""
    rows = read("a8_seed_correlations_last.csv")
    forms = ["straight line, per-track scale", "direct states"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, form in zip(axes, forms):
        for i, c in enumerate(COMPONENTS):
            r = [row for row in rows if row["form"] == form and row["component"] == c]
            for dx, column, colour, label in (
                    (-0.17, "bias_map_correlation", "#9c9b95", "e"),
                    (0.17, "bias_map_correlation_without_exact_scheme_error", SLOTS[0],
                     "e minus the exact scheme's error (the network's own part)")):
                v = [float(row[column]) for row in r]
                ax.scatter(i + dx + rng.uniform(-0.06, 0.06, len(v)), v, s=22, color=colour,
                           label=label if i == 0 else None, edgecolor="white", linewidth=0.5)
                ax.plot([i + dx - 0.12, i + dx + 0.12], [np.median(v)] * 2, color="black", lw=2)
        ax.axhline(0.7, color="#d03b3b", ls="--", lw=1.2)
        ax.text(3.45, 0.72, "shared if above", ha="right", fontsize=9, color="#d03b3b")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(4), ["e_" + c for c in COMPONENTS])
        ax.set_title("%s: %d pairs of seeds" % (form, len(r)))
    axes[0].set_ylabel("correlation of the bias maps of two seeds")
    axes[0].set_ylim(-0.7, 1.05)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("A.8  Seeds share only the exact scheme's error; the network's own error is "
                 "private (bar = median)")
    fig.tight_layout()
    save(fig, "a8_seed_correlations.png")


def a8_bias_maps():
    """The bias maps themselves for three L-BFGS seeds of the straight line
    per-track form and the exact scheme: e_x differs from seed to seed, e_tx is
    the exact scheme's map in every seed."""
    seeds = [("seed 1", "25dfa17ef168"), ("seed 2", "dedaab236b3d"), ("seed 3", "b68802fcce0e")]
    maps = [(name, bias_map(cached_single_step_errors(key, "test"))) for name, key in seeds]
    ex = bias_map(exact_errors("test"))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), layout="constrained")
    for row, (i, c) in enumerate(((0, "x"), (2, "tx"))):
        own = [m[:, i] for _, m in maps]
        limit = max(np.abs(np.concatenate(own + [ex[:, i]])).max(), 1e-9)
        panels = [(name, m[:, i]) for name, m in maps] + [("exact scheme", ex[:, i])]
        for ax, (name, values) in zip(axes[row], panels):
            image = ax.imshow(values.reshape(8, 8), cmap="RdBu_r", vmin=-limit, vmax=limit,
                              origin="lower", aspect="auto")
            ax.set_title("%s, mean e_%s" % (name, c))
            if row == 1:
                ax.set_xlabel("z (8 groups of 8 steps)")
            if ax is axes[row][0]:
                ax.set_ylabel("signed q/p (8 quantile bins)")
            ax.grid(False)
        fig.colorbar(image, ax=axes[row].tolist(), label="mean e_%s (%s)" % (c, UNIT[c]))
    fig.suptitle("A.8  Bias maps of three seeds (straight line per-track, L-BFGS): "
                 "e_x is different in each seed; e_tx is the exact scheme's map")
    save(fig, "a8_bias_maps.png")


PART_B = [("direct states", "49e5e2450eaa", "5cb2aeb6778b", "6aba11cc7475", 2.85),
          ("straight line, per-track scale", "25dfa17ef168", "8a0f3b8f422f", "f0f219df805e",
           0.92)]


def b_bands():
    """Test endpoint per momentum band: the baseline, B.3 and B.4, per form."""
    rows = {r["run_key"]: r for r in read("b_part_b_scores.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(BAND_LABELS))
    for ax, (form, base, b3, b4, target) in zip(axes, PART_B):
        for j, (key, name, colour) in enumerate(((base, "baseline", EXACT),
                                                 (b3, "B.3 residual draws", SLOTS[1]),
                                                 (b4, "B.4 bias term", SLOTS[2]))):
            r = rows[key]
            ax.plot(x, [float(r["median_um_%s" % b]) for b in BAND_LABELS], "o-", color=colour,
                    label="%s: %.2f um overall" % (name, float(r["endpoint_radial_median_um"])))
        ax.axhline(target, color="#d03b3b", ls="--", lw=1.2, label="target overall, %.2f um" % target)
        ax.set_yscale("log")
        ax.set_xticks(x, BAND_LABELS)
        ax.set_xlabel("momentum band")
        ax.set_title(form)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("test endpoint radial median (um)")
    fig.suptitle("Part B  Neither training change reaches its target")
    fig.tight_layout()
    save(fig, "b_bands.png")


def b_training():
    """Validation endpoint against the round for the Part B runs and their baselines."""
    from common import STORE
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for ax, (form, base, b3, b4, target) in zip(axes, PART_B):
        for key, name, colour in ((base, "baseline", EXACT), (b3, "B.3 residual draws", SLOTS[1]),
                                  (b4, "B.4 bias term", SLOTS[2])):
            with open(os.path.join(STORE, "runs", key, "rounds.csv")) as handle:
                r = list(csv.DictReader(handle))
            ax.plot([int(x["round"]) for x in r],
                    [float(x["validation_endpoint_error_larger_of_x_and_y_median_micrometres"])
                     for x in r], color=colour, lw=1.5, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("round")
        ax.set_title(form)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("validation endpoint, larger of x and y median (um)")
    fig.suptitle("Part B  Training: the validation score per round")
    fig.tight_layout()
    save(fig, "b_training.png")


def overview():
    """Every way tried to remove the error, per form, on one log scale."""
    variants = {}
    for _, key in RUNS[::2]:
        for r in read("c_variants_%s.csv" % key):
            variants[(key, r["variant"])] = float(r["endpoint_radial_median_um"])
    partb = {r["run_key"]: float(r["endpoint_radial_median_um"]) for r in read("b_part_b_scores.csv")}
    comparators = {r["series"]: float(r["endpoint_radial_median_um"]) for r in read("c_comparators.csv")}
    rk4 = [v for k, v in comparators.items() if "RK4" in k][0]
    exact = [v for k, v in comparators.items() if "Gauss" in k][0]
    names = list(dict.fromkeys(v for _, v in variants))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, (form, base, b3, b4, target) in zip(axes, PART_B):
        items = [(n, variants[(base, n)]) for n in names if (base, n) in variants]
        items += [("B.3 residual draws (retrained)", partb[b3]), ("B.4 bias term (retrained)", partb[b4])]
        y = np.arange(len(items))
        values = [v for _, v in items]
        colours = [SLOTS[2] if v <= target else SLOTS[0] for v in values]
        ax.hlines(y, 0.05, values, color="#d6d5d0", lw=1)
        ax.scatter(values, y, s=60, color=colours, zorder=3)
        for yy, v in zip(y, values):
            ax.text(v * 1.12, yy, "%.3g" % v, va="center", fontsize=8)
        ax.axvline(target, color="#d03b3b", ls="--", lw=1.2)
        ax.axvline(rk4, color="black", ls=":", lw=1.2)
        ax.axvline(exact, color=EXACT, ls="-.", lw=1.2)
        for value, text, colour in ((target, "target", "#d03b3b"), (rk4, "RK4", "black"),
                                    (exact, "exact scheme", EXACT)):
            ax.text(value, -0.7, text, ha="center", fontsize=8, color=colour)
        ax.set_yticks(y, [n for n, _ in items])
        ax.set_xscale("log")
        ax.set_xlim(0.05, 200)
        ax.set_xlabel("test endpoint radial median (um); green = meets the target")
        ax.set_title("%s (target %.2f um)" % (form, target), pad=18)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle("Every correction tried: only a sweep or a Newton step of the stage equations "
                 "reaches the target\n(RK4 = classic Runge-Kutta at 256 field calls, %.2f um; "
                 "exact scheme = the 2-stage Gauss-Legendre solution, %.3f um)" % (rk4, exact))
    fig.tight_layout()
    save(fig, "overview.png")


ALL = [along_the_track, share_by_z, toy_fit, variants, a1_r_squared, a3_rms_by_band,
       a4_endpoint_along_the_bend, a4b_autocorrelation, a6_cell_faces, a8_seed_correlations,
       a8_bias_maps, b_bands, b_training, overview]


if __name__ == "__main__":
    import sys
    asked = sys.argv[1:]
    for function in ALL:
        if not asked or function.__name__ in asked:
            function()
