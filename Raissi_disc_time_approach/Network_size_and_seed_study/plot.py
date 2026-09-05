#!/usr/bin/env python
"""The four figures of the size-and-seed study, from the aggregated tables.

Reads results/summary.csv and results/by_architecture.csv (written by
aggregate.py) and recomputes nothing.  Writes

  figures/floor_vs_architecture.png  median test endpoint error per architecture
                                     against parameter count, min-max seed bars,
                                     colour by depth, with the exact-scheme
                                     ceiling, the One_step_network_v2 4x50
                                     result and the data twin at the winner
  figures/loss_vs_error.png          final training loss vs test endpoint error
                                     over the converged runs, log-log, Spearman
                                     coefficient in the title
  figures/seed_spread.png            the ten seeds of each architecture as
                                     points, max/min ratio annotated
  figures/val_vs_test_selection.png  validation vs test endpoint error per run:
                                     does validation pick the best seed?

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python plot.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
from scipy.stats import spearmanr          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

# The exact scheme's own endpoint error at q = 8, solved without any network:
# no network trained on this scheme can beat it.  The like-for-like number is
# the one measured on THESE 2018 frozen-leg test states by
# ../Stage_count_sweep/measure_scheme_ceiling.py (22.5 um).  The 29 um quoted
# in ../One_step_network* was measured on a different population (32
# momentum-stratified legs) and is not the ceiling for this test set.
CEILING_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Stage_count_sweep",
    "results", "scheme_ceiling_same_population_q08.json")


def _ceiling_um(default=22.5):
    try:
        with open(CEILING_JSON) as f:
            return float(json.load(f)["endpoint_med_um"])
    except Exception:
        return default


CEILING_UM = _ceiling_um()
# The verified baseline, ../One_step_network_v2/results/summary.csv (physics,
# seeds 0-2, 4x50 network, four BLAS threads).
V2_SUMMARY = os.path.join(HERE, "..", "One_step_network_v2", "results",
                          "summary.csv")

DEPTH_COLOUR = {2: "#4C72B0", 4: "#DD8452", 6: "#55A868"}


def v2_reference():
    """Median test endpoint error of the v2 physics runs, or None."""
    if not os.path.exists(V2_SUMMARY):
        return None
    d = pd.read_csv(V2_SUMMARY)
    d = d[(d["mode"] == "physics") & d["converged"]]
    return float(np.median(d["endpoint_med_um"])) if len(d) else None


def load():
    runs = pd.read_csv(os.path.join(RESULTS, "summary.csv"))
    arch = pd.read_csv(os.path.join(RESULTS, "by_architecture.csv"))
    return runs, arch


def fig_floor(runs, arch):
    phys = arch[(arch["mode"] == "physics") & (arch["n_converged"] > 0)]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for depth, g in phys.groupby("depth"):
        g = g.sort_values("n_params")
        lo = g["test_endpoint_med_um_median"] - g["test_endpoint_med_um_min"]
        hi = g["test_endpoint_med_um_max"] - g["test_endpoint_med_um_median"]
        ax.errorbar(g["n_params"], g["test_endpoint_med_um_median"],
                    yerr=np.vstack([lo, hi]), fmt="o-", capsize=3,
                    color=DEPTH_COLOUR.get(int(depth), None),
                    label="depth %d" % depth, zorder=3)
        for _, r in g.iterrows():
            ax.annotate("%dx%d" % (r["depth"], r["width"]),
                        (r["n_params"], r["test_endpoint_med_um_median"]),
                        textcoords="offset points", xytext=(5, 6), fontsize=7.5)

    # y in data units, x in axes fraction, so the labels sit inside the frame
    # whatever the log limits end up being.
    tr = ax.get_yaxis_transform()
    ax.axhline(CEILING_UM, color="k", ls="--", lw=1.2, zorder=2)
    ax.text(0.015, CEILING_UM * 1.06,
            "exact-scheme ceiling %.0f um (q = 8, these 2018 test states)"
            % CEILING_UM, transform=tr, fontsize=8.5, va="bottom")

    v2 = v2_reference()
    if v2 is not None:
        ax.axhline(v2, color="grey", ls=":", lw=1.2, zorder=2)
        ax.text(0.015, v2 * 1.04,
                "One_step_network_v2, 4x50, 3 seeds, 4 threads: %.0f um" % v2,
                transform=tr, fontsize=8.5, va="bottom", color="grey")

    twin = arch[(arch["mode"] == "data") & (arch["n_converged"] > 0)]
    for _, r in twin.iterrows():
        ax.errorbar([r["n_params"]], [r["test_endpoint_med_um_median"]],
                    yerr=np.array([[r["test_endpoint_med_um_median"]
                                    - r["test_endpoint_med_um_min"]],
                                   [r["test_endpoint_med_um_max"]
                                    - r["test_endpoint_med_um_median"]]]),
                    fmt="D", ms=9, capsize=3, color="crimson", zorder=4,
                    label="data twin, %dx%d" % (r["depth"], r["width"]))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("network parameters")
    ax.set_ylabel("test endpoint error, median over seeds (um)")
    ax.set_title("Does the endpoint floor move with network size?\n"
                 "physics loss, q = 8, frozen leg; bars span the 10 seeds",
                 fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    path = os.path.join(FIGURES, "floor_vs_architecture.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_loss_vs_error(runs):
    c = runs[runs["converged"] & (runs["mode"] == "physics")]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    rho = np.nan
    if len(c) > 2:
        rho = float(spearmanr(c["final_loss"], c["test_endpoint_med_um"]).statistic)
    for depth, g in c.groupby("depth"):
        ax.scatter(g["final_loss"], g["test_endpoint_med_um"], s=26,
                   color=DEPTH_COLOUR.get(int(depth), None), alpha=0.85,
                   label="depth %d" % depth)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(CEILING_UM, color="k", ls="--", lw=1.0)
    ax.set_xlabel("final physics training loss")
    ax.set_ylabel("test endpoint error (um)")
    ax.set_title("Is the floor set by optimisation or by capacity?\n"
                 "Spearman rho = %.2f over %d converged runs" % (rho, len(c)),
                 fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    path = os.path.join(FIGURES, "loss_vs_error.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, rho, len(c)


def fig_seed_spread(runs, arch):
    phys = runs[runs["mode"] == "physics"].copy()
    order = (arch[arch["mode"] == "physics"]
             .sort_values(["n_params", "depth"])[["width", "depth", "n_params"]])
    labels, xs = [], {}
    for i, (_, r) in enumerate(order.iterrows()):
        labels.append("%dx%d\n%dk" % (r["depth"], r["width"],
                                      round(r["n_params"] / 1000.0)))
        xs[(int(r["width"]), int(r["depth"]))] = i

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for (w, d), g in phys.groupby(["width", "depth"]):
        i = xs.get((int(w), int(d)))
        if i is None:
            continue
        conv = g[g["converged"]]
        unconv = g[~g["converged"]]
        jitter = np.linspace(-0.18, 0.18, max(len(g), 1))
        if len(conv):
            ax.scatter(np.full(len(conv), i) + jitter[:len(conv)],
                       conv["test_endpoint_med_um"], s=30,
                       color=DEPTH_COLOUR.get(int(d), None), zorder=3)
            t = conv["test_endpoint_med_um"].to_numpy(float)
            ax.annotate("x%.1f" % (t.max() / t.min()),
                        (i, t.max()), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        if len(unconv):
            ax.scatter(np.full(len(unconv), i) + jitter[-len(unconv):],
                       unconv["test_endpoint_med_um"], s=34, facecolors="none",
                       edgecolors="grey", zorder=3)

    ax.axhline(CEILING_UM, color="k", ls="--", lw=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("test endpoint error (um)")
    ax.set_xlabel("architecture (depth x width, parameters)")
    ax.set_title("Seed-to-seed spread within each architecture\n"
                 "filled = converged, open = not converged; "
                 "annotation = max/min over the converged seeds", fontsize=11)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    path = os.path.join(FIGURES, "seed_spread.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_val_vs_test(runs):
    c = runs[runs["converged"] & (runs["mode"] == "physics")]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    rho_all = np.nan
    if len(c) > 2:
        rho_all = float(spearmanr(c["val_endpoint_med_um"],
                                  c["test_endpoint_med_um"]).statistic)
    for depth, g in c.groupby("depth"):
        ax.scatter(g["val_endpoint_med_um"], g["test_endpoint_med_um"], s=26,
                   color=DEPTH_COLOUR.get(int(depth), None), alpha=0.85,
                   label="depth %d" % depth)
    lims = [min(c["val_endpoint_med_um"].min(), c["test_endpoint_med_um"].min()),
            max(c["val_endpoint_med_um"].max(), c["test_endpoint_med_um"].max())]
    ax.plot(lims, lims, color="grey", ls=":", lw=1.0)

    # Within each architecture: does the validation-best seed have the best test
    # error?  Report the hit rate and the per-architecture rank correlation.
    hits, n_arch, rhos = 0, 0, []
    for _, g in c.groupby(["width", "depth"]):
        if len(g) < 3:
            continue
        n_arch += 1
        pick = g.loc[g["val_endpoint_med_um"].idxmin()]
        best = g.loc[g["test_endpoint_med_um"].idxmin()]
        hits += int(pick["seed"] == best["seed"])
        rhos.append(float(spearmanr(g["val_endpoint_med_um"],
                                    g["test_endpoint_med_um"]).statistic))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("validation endpoint error (um)")
    ax.set_ylabel("test endpoint error (um)")
    sub = "" if not n_arch else (
        "\nvalidation-best seed is also test-best in %d/%d architectures; "
        "median within-architecture rho = %.2f"
        % (hits, n_arch, float(np.median(rhos))))
    ax.set_title("Does validation pick the seed?  Spearman rho = %.2f "
                 "over %d runs%s" % (rho_all, len(c), sub), fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    path = os.path.join(FIGURES, "val_vs_test_selection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, rho_all, hits, n_arch, (float(np.median(rhos)) if rhos else np.nan)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    runs, arch = load()
    print(fig_floor(runs, arch))
    p, rho, n = fig_loss_vs_error(runs)
    print(p, "spearman(loss, test error) = %.3f over %d runs" % (rho, n))
    print(fig_seed_spread(runs, arch))
    p, rho_all, hits, n_arch, rho_med = fig_val_vs_test(runs)
    print(p, "spearman(val, test) = %.3f; val-best = test-best in %d/%d "
             "architectures; median within-architecture rho = %.3f"
          % (rho_all, hits, n_arch, rho_med))


if __name__ == "__main__":
    main()
