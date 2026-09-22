#!/usr/bin/env python
"""E3 - the Block E error tables: error(q, dz) at the SciFi plane, and the tails.

The error of a chain is the RADIAL distance at the end plane z1 = 7,826 mm
between the network's endpoint and the RK6 endpoint of the same track,
r = sqrt(dx^2 + dy^2), on the 1,452 test tracks (George 2026-09-18). Every
comparator is measured the same way, from its own saved end states:

    exact scheme     the q-stage collocation scheme chained without a network
                     (E2 at N = 2 and 256; Block D's runs at N = 64 and 128,
                     which are the same test particles - the E0 gate)
    straight line    the magnet ignored
    material floor   the particle's REAL first-SciFi state against the RK6
                     track carried to the same plane: what no field-only
                     method can beat
    Block D          the one-network-per-step chains at the same (N, q)

Also written: the tails (95th, 99th percentile, the fraction beyond 1 mm), the
cost per track against the error, and how much the error moved from round to
round near the end of training - training was stopped on a plateau, so a single
checkpoint's error carries that scatter with it.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python tables.py
Outputs: results/error_qdz_chain.csv, results/tails.csv, results/cost_accuracy.csv,
         results/comparators.csv, figures/error_qdz.png
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, "..", "E1_Network_grid", "results")
E2 = os.path.join(HERE, "..", "E2_Comparators", "results")
BD = os.path.join(HERE, "..", "..", "..", "multi_network_chain_discrete_approach",
                  "Block_D_fixed_step_crossing")
TRACKS = os.path.join(HERE, "..", "E0_Track_dataset", "results", "tracks.npz")
SPLIT = "test"


def radial_um(pred, truth):
    return np.hypot(pred[:, 0] - truth[:, 0], pred[:, 1] - truth[:, 1]) * 1e3


def summary(r):
    s = np.sort(r)
    return dict(med_um=float(np.median(r)), p95_um=float(np.quantile(r, 0.95)),
                p99_um=float(np.quantile(r, 0.99)), mean_um=float(r.mean()),
                max_um=float(r.max()), frac_above_1mm=float((r > 1000).mean()),
                worst5pct_share=float(s[int(0.95 * len(s)):].sum() / s.sum()), n=int(len(r)))


def val_scatter(run, last=8):
    """How much the validation error moved from round to round at the end."""
    p = os.path.join(run, "rounds.csv")
    if not os.path.exists(p):
        return {}
    v = np.array([float(x["val_z1_pos_med_um"]) for x in csv.DictReader(open(p))])
    v = v[-last:]
    return dict(val_rounds_mean_um=float(v.mean()), val_rounds_std_um=float(v.std()),
                val_rounds_min_um=float(v.min()), val_rounds_max_um=float(v.max()),
                val_rounds_rel_spread=float(v.std() / v.mean()), val_rounds_used=int(len(v)))


def main():
    D = np.load(TRACKS)
    n_max = int(D["n_max"])
    truth_end = np.asarray(D["%s_truth" % SPLIT])[:, n_max]
    S0 = np.asarray(D["%s_S0" % SPLIT])
    L = float(D["L"])
    straight = np.stack([S0[:, 0] + S0[:, 2] * L, S0[:, 1] + S0[:, 3] * L], axis=1)
    material = radial_um(np.asarray(D["%s_S_post" % SPLIT]), np.asarray(D["%s_truth_zpost" % SPLIT]))
    comparators = [dict(what="straight line", **summary(radial_um(straight, truth_end))),
                   dict(what="material floor (real SciFi state vs the RK6 track)", **summary(material))]

    rows, tails, cost = [], [], []
    for run in sorted(glob.glob(os.path.join(E1, "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        rec = json.load(open(os.path.join(run, "record.json")))
        N, q = rec["N"], rec["q"]
        end = np.load(os.path.join(run, "chain_states.npz"))["%s_states" % SPLIT][:, -1]
        r = radial_um(end, truth_end)
        s = summary(r)
        # the exact scheme at the same (N, q), radial, from its own states
        ex = os.path.join(E2, "exact_N%03d_q%02d_states.npz" % (N, q))
        exd = os.path.join(BD, "D2_Comparators", "results", "exact_N%03d_q%02d_states.npz" % (N, q))
        f = ex if os.path.exists(ex) else exd
        e_med = float(np.median(radial_um(np.load(f)["states"][:, -1], truth_end))) if os.path.exists(f) else float("nan")
        # Block D's one-network-per-step chain at the same (N, q)
        bd = os.path.join(BD, "D1_Chain_grid", "results", "N%03d_q%02d" % (N, q), "states.npz")
        b_med = float(np.median(radial_um(np.load(bd)["%s_states" % SPLIT][:, -1], truth_end))) if os.path.exists(bd) else float("nan")
        row = dict(N=N, q=q, dz_mm=round(rec["dz_mm"], 2), **s, exact_med_um=e_med,
                   block_d_med_um=b_med, restarts=rec["restarts"], rounds=rec["rounds"],
                   final_loss=rec["final_loss"], stopped_by_hand=rec.get("stopped_by_hand"),
                   us_per_track=round(rec["carry_us_per_track"][SPLIT], 1), **val_scatter(run))
        rows.append(row)
        tails.append(dict(N=N, q=q, med_um=s["med_um"], p95_um=s["p95_um"], p99_um=s["p99_um"],
                          max_um=s["max_um"], frac_above_1mm=s["frac_above_1mm"],
                          worst5pct_share=s["worst5pct_share"]))
        cost.append(dict(N=N, q=q, us_per_track=row["us_per_track"], med_um=s["med_um"],
                         n_parameters=rec["n_parameters"]))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for name, data in (("error_qdz_chain.csv", rows), ("tails.csv", tails),
                       ("cost_accuracy.csv", cost), ("comparators.csv", comparators)):
        with open(os.path.join(HERE, "results", name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    print("Radial error at z1 on the %d test tracks [um]" % rows[0]["n"])
    print("  N   q    dz |  median     p95     p99 | exact | Block D | round scatter | us/track")
    for x in sorted(rows, key=lambda d: (d["N"], d["q"])):
        print("%4d %3d %6.1f | %7.1f %7.0f %7.0f | %5.1f | %7s | %5.0f +- %-4.0f | %6.0f"
              % (x["N"], x["q"], x["dz_mm"], x["med_um"], x["p95_um"], x["p99_um"], x["exact_med_um"],
                 ("%.0f" % x["block_d_med_um"]) if x["block_d_med_um"] == x["block_d_med_um"] else "-",
                 x.get("val_rounds_mean_um", float("nan")), x.get("val_rounds_std_um", float("nan")),
                 x["us_per_track"]))
    for c in comparators:
        print("%-52s median %8.0f um, p95 %8.0f" % (c["what"], c["med_um"], c["p95_um"]))

    # -- the figure: error(q, dz) ------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = sorted({x["N"] for x in rows})
    qs = sorted({x["q"] for x in rows})
    grid = np.full((len(Ns), len(qs)), np.nan)
    ex_grid = np.full_like(grid, np.nan)
    for x in rows:
        grid[Ns.index(x["N"]), qs.index(x["q"])] = x["med_um"]
        ex_grid[Ns.index(x["N"]), qs.index(x["q"])] = x["exact_med_um"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    im = ax[0].imshow(np.log10(grid), cmap="viridis_r", aspect="auto")
    for i in range(len(Ns)):
        for j in range(len(qs)):
            ax[0].text(j, i, "%.0f" % grid[i, j], ha="center", va="center", color="w", fontsize=10)
    ax[0].set_xticks(range(len(qs)), ["q = %d" % q for q in qs])
    ax[0].set_yticks(range(len(Ns)), ["N = %d\ndz = %.0f mm" % (N, 5177.8 / N) for N in Ns])
    ax[0].set_title("Block E: median radial error at the SciFi plane [µm]")
    fig.colorbar(im, ax=ax[0], label="log10 median error [µm]")
    for i, N in enumerate(Ns):
        ax[1].plot(qs, grid[i], "-o", label="N = %d (dz = %.0f mm)" % (N, 5177.8 / N))
        ax[1].plot(qs, ex_grid[i], ":", color=ax[1].lines[-1].get_color(), lw=1)
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].set_xticks(qs, [str(q) for q in qs])
    ax[1].set_xlabel("stage count q")
    ax[1].set_ylabel("median radial error at z1 [µm]")
    ax[1].set_title("solid: the network; dotted: the exact scheme at the same N and q")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "error_qdz.png"), dpi=130)
    print("wrote results/error_qdz_chain.csv, tails.csv, cost_accuracy.csv, comparators.csv, "
          "figures/error_qdz.png")


if __name__ == "__main__":
    main()
