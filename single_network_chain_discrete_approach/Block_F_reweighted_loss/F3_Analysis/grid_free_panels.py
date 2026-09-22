#!/usr/bin/env python
"""F3 - the two E3 figures that assume the N x q grid, drawn one panel per run.

E3's `convergence.py` and `errors_vs_momentum.py` lay out len(Ns) x len(qs)
panels and index every (N, q) cell. Block F has three runs on a diagonal
(N = 64 q = 2, N = 128 q = 8, N = 256 q = 16), so the same panels are drawn
here in a single row, with the same per-panel code, and the same CSVs are
written (`results/convergence.csv` keeps E3's columns, including the
`val_last8_mean` that the case study and the anatomy use to pick the best
network).

Called by run_e3_for_block_f.py; can also be run on its own.
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.abspath(os.path.join(HERE, "..", "..", "Block_E_single_network_chain", "E3_Analysis"))
RUNS = os.path.join(HERE, "runs", "results")
TRACKS = os.path.join(E3, "..", "E0_Track_dataset", "results", "tracks.npz")
COMPONENTS = (("x", 0, 1e3, "µm"), ("y", 1, 1e3, "µm"), ("tx", 2, 1e3, "mrad"), ("ty", 3, 1e3, "mrad"))
P_EDGES = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200.0])


def _e3(name):
    spec = importlib.util.spec_from_file_location("e3_" + name, os.path.join(E3, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def runs():
    return sorted(glob.glob(os.path.join(RUNS, "N[0-9][0-9][0-9]_q[0-9][0-9]")))


def convergence():
    series = _e3("convergence").series                      # E3's own reader
    data, rows = {}, []
    for run in runs():
        rec = json.load(open(os.path.join(run, "record.json")))
        N, q = rec["N"], rec["q"]
        data[(N, q)] = series(run)
        restart, loss, _, cum, val, loss_end = data[(N, q)]
        last = min(8, len(val))
        rows.append(dict(N=N, q=q, restarts=len(restart), rounds=len(val),
                         loss_first=float(loss[0]), loss_final=float(loss[-1]),
                         loss_end_first=float(loss_end[0]), loss_end_last=float(loss_end[-1]),
                         loss_end_last8_ratio=float(loss_end[-last] / loss_end[-1]),
                         val_first=float(val[0]), val_final=float(val[-1]),
                         val_last8_mean=float(val[-last:].mean()), val_last8_std=float(val[-last:].std()),
                         val_best=float(val.min()), val_best_round=int(np.argmin(val) + 1),
                         test_final=rec["test"]["vs_rk6_endpoint"]["pos_med_um"]))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "convergence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt
    keys = sorted(data)
    fig, ax = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 3.6), squeeze=False)
    for a, (N, q) in zip(ax[0], keys):
        restart, loss, rounds, cum, val, loss_end = data[(N, q)]
        a.plot(restart, loss, "-", color="#1f77b4", lw=0.9)
        a.plot(cum - 1, loss_end, "o", color="#08306b", ms=3, label="round end")
        a.set_yscale("log")
        a.set_ylabel("training loss", color="#1f77b4", fontsize=8)
        a.tick_params(axis="y", labelcolor="#1f77b4", labelsize=7)
        a.tick_params(axis="x", labelsize=7)
        b = a.twinx()
        b.plot(cum - 1, val, "-s", color="#d62728", ms=3, lw=1.2)
        b.set_ylabel("val error at z1 [µm]", color="#d62728", fontsize=8)
        b.tick_params(axis="y", labelcolor="#d62728", labelsize=7)
        a.set_title("N = %d, q = %d" % (N, q), fontsize=9)
        a.set_xlabel("L-BFGS restart", fontsize=8)
    fig.suptitle("Block F: training loss (blue, left) and validation error at the SciFi plane "
                 "(red, right) against restart", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "convergence_grid.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(keys):
        restart, loss, rounds, cum, val, loss_end = data[k]
        col = cmap(i / max(len(keys) - 1, 1))
        ax[0].plot(cum, loss_end, "-o", ms=3, color=col, label="N=%d q=%d" % k)
        ax[1].plot(cum, val, "-o", ms=3, color=col)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("L-BFGS restarts")
    ax[0].set_ylabel("loss at the end of each round")
    ax[0].set_title("The loss on freshly drawn states, round by round", fontsize=10)
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("L-BFGS restarts")
    ax[1].set_ylabel("validation error at z1 [µm]")
    ax[1].set_yscale("log")
    ax[1].set_title("and the error it buys", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "convergence_summary.png"), dpi=130)
    plt.close(fig)
    print("wrote figures/convergence_grid.png, convergence_summary.png, results/convergence.csv")
    for x in rows:
        print("%4d %3d | %8d %6d | loss end %9.2e -> %.2e | val %6.0f -> %5.0f (best %4.0f at round %2d) | last-8 %5.0f +- %.0f"
              % (x["N"], x["q"], x["restarts"], x["rounds"], x["loss_end_first"], x["loss_end_last"],
                 x["val_first"], x["val_final"], x["val_best"], x["val_best_round"],
                 x["val_last8_mean"], x["val_last8_std"]))


def errors_vs_momentum():
    D = np.load(TRACKS)
    n_max = int(D["n_max"])
    truth_end = np.asarray(D["test_truth"])[:, n_max]
    P = np.asarray(D["test_P"])
    err = {}
    for run in runs():
        rec = json.load(open(os.path.join(run, "record.json")))
        end = np.load(os.path.join(run, "chain_states.npz"))["test_states"][:, -1]
        err[(rec["N"], rec["q"])] = end[:, :4] - truth_end[:, :4]
    rows = []
    for (N, q), d in sorted(err.items()):
        for name, i, scale, unit in COMPONENTS:
            a = np.abs(d[:, i]) * scale
            for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
                m = (P >= lo) & (P < hi)
                if m.sum() < 5:
                    continue
                rows.append(dict(N=N, q=q, component=name, unit=unit, p_lo=lo, p_hi=hi,
                                 n=int(m.sum()), med=float(np.median(a[m])),
                                 p95=float(np.quantile(a[m], 0.95)),
                                 signed_mean=float(np.mean(d[m, i]) * scale)))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "error_vs_p.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    centres = np.sqrt(P_EDGES[:-1] * P_EDGES[1:])
    keys = sorted(err)
    for name, i, scale, unit in COMPONENTS:
        fig, ax = plt.subplots(1, len(keys), figsize=(5.4 * len(keys), 4.0), sharey=True, squeeze=False)
        vals = np.concatenate([np.abs(d[:, i]) * scale for d in err.values()])
        hi = np.quantile(vals, 0.995)
        for a, (N, q) in zip(ax[0], keys):
            d = err[(N, q)]
            v = np.abs(d[:, i]) * scale
            hb = a.hexbin(P, np.clip(v, 1e-4 * hi, hi), xscale="log", yscale="log", gridsize=34,
                          mincnt=1, norm=LogNorm(), cmap="Blues")
            cb = fig.colorbar(hb, ax=a, pad=0.02)
            cb.set_label("test tracks per cell", fontsize=7)
            cb.ax.tick_params(labelsize=6)
            med = [np.median(v[(P >= lo) & (P < h2)]) if ((P >= lo) & (P < h2)).sum() > 4 else np.nan
                   for lo, h2 in zip(P_EDGES[:-1], P_EDGES[1:])]
            p95 = [np.quantile(v[(P >= lo) & (P < h2)], 0.95) if ((P >= lo) & (P < h2)).sum() > 4 else np.nan
                   for lo, h2 in zip(P_EDGES[:-1], P_EDGES[1:])]
            a.plot(centres, med, "r-o", ms=3, lw=1.4, label="median per momentum bin")
            a.plot(centres, p95, "r--", lw=1, label="95th percentile")
            a.set_title("N = %d, q = %d   (median %.3g %s)" % (N, q, np.median(v), unit), fontsize=9)
            a.grid(alpha=0.25, which="both")
            a.set_xlabel("momentum [GeV]")
        ax[0, 0].set_ylabel("|%s error| [%s]" % (name, unit))
        ax[0, 0].legend(fontsize=7)
        fig.suptitle("Block F: |%s| error at the SciFi plane against momentum. Each hexagon is coloured by "
                     "how many of the 1,452 test tracks fall in it (log scale)." % name, fontsize=11)
        fig.tight_layout()
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "figures", "error_vs_p_%s.png" % name), dpi=120)
        plt.close(fig)
        print("wrote figures/error_vs_p_%s.png" % name)
    print("wrote results/error_vs_p.csv")


if __name__ == "__main__":
    convergence()
    errors_vs_momentum()


def along_z():
    """E3's along_z: the same per-plane table and figure; the closing summary
    reads each run's own q instead of assuming every N has the first q."""
    D = np.load(TRACKS)
    Z0, L, n_max = float(D["z0"]), float(D["L"]), int(D["n_max"])
    truth = np.asarray(D["test_truth"])
    rows = []
    for run in runs():
        rec = json.load(open(os.path.join(run, "record.json")))
        N, q = rec["N"], rec["q"]
        stride = n_max // N
        st = np.load(os.path.join(run, "chain_states.npz"))["test_states"]
        for k in range(N + 1):
            t = truth[:, k * stride]
            r = np.hypot(st[:, k, 0] - t[:, 0], st[:, k, 1] - t[:, 1]) * 1e3
            dtx = np.abs(st[:, k, 2] - t[:, 2]) * 1e3
            rows.append(dict(N=N, q=q, plane=k, z_mm=float(Z0 + k * L / N),
                             med_um=float(np.median(r)), p95_um=float(np.quantile(r, 0.95)),
                             tx_med_mrad=float(np.median(dtx))))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "error_vs_z.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    import matplotlib.pyplot as plt
    keys = sorted({(x["N"], x["q"]) for x in rows})
    fig, ax = plt.subplots(1, len(keys), figsize=(4.3 * len(keys), 4.2), sharey=True, squeeze=False)
    for a, (N, q) in zip(ax[0], keys):
        s = [x for x in rows if x["N"] == N and x["q"] == q]
        a.plot([x["z_mm"] / 1000 for x in s], [x["med_um"] for x in s], "-", lw=1.3, label="median")
        a.plot([x["z_mm"] / 1000 for x in s], [x["p95_um"] for x in s], "--", lw=1.0, label="95th percentile")
        a.set_title("N = %d, q = %d (dz = %.0f mm)" % (N, q, 5177.8 / N), fontsize=10)
        a.set_xlabel("z [m]")
        a.set_yscale("log")
        a.grid(alpha=0.3, which="both")
    ax[0, 0].set_ylabel("radial error against the RK6 track [µm]")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Block F: how the error grows along the crossing, test tracks", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "error_vs_z.png"), dpi=130)
    plt.close(fig)
    print("wrote results/error_vs_z.csv, figures/error_vs_z.png")
    for N, q in keys:
        s = [x for x in rows if x["N"] == N and x["q"] == q]
        q1 = s[len(s) // 4]["med_um"]; h = s[len(s) // 2]["med_um"]; e = s[-1]["med_um"]
        print("  N=%3d q=%d: quarter way %6.1f um, half way %6.1f, end %6.1f  (end/half %.2f)"
              % (N, q, q1, h, e, e / h))


def single_step_tables():
    """E3's single-step tables (same computation, via E3's own imports), and
    the figure drawn per run rather than per (N, q) grid line."""
    import sys
    import time
    E1 = os.path.join(E3, "..", "E1_Network_grid")
    if E1 not in sys.path:
        sys.path.insert(0, E1)
    if E3 not in sys.path:
        sys.path.insert(0, E3)
    import use_shared                                          # noqa: F401
    from _shared.reference import make_field
    from chain_network import load_network, step_outputs, straight_numpy
    D = np.load(TRACKS)
    Z0, L, n_max = float(D["z0"]), float(D["L"]), int(D["n_max"])
    truth = np.asarray(D["test_truth"])
    P = np.asarray(D["test_P"])
    fld = make_field(str(D["field"]))
    rows, by_z = [], []
    for run in runs():
        rec = json.load(open(os.path.join(run, "record.json")))
        N, q = rec["N"], rec["q"]
        stride, dz = n_max // N, L / N
        model = load_network(run, fld)
        t0 = time.time()
        n_tr = len(truth)
        S = truth[:, 0:n_max:stride].reshape(-1, 5)
        nxt = truth[:, stride:n_max + 1:stride].reshape(-1, 5)
        plane = np.repeat(np.arange(N)[None, :], n_tr, 0).reshape(-1)
        z_start = Z0 + plane * dz
        out = step_outputs(model, S, z_start)[:, -1, :]
        r = np.hypot(out[:, 0] - nxt[:, 0], out[:, 1] - nxt[:, 1]) * 1e3
        line = straight_numpy(S, dz)
        r_line = np.hypot(line[:, 0] - nxt[:, 0], line[:, 1] - nxt[:, 1]) * 1e3
        p = np.repeat(P[:, None], N, axis=1).reshape(-1)
        rows.append(dict(N=N, q=q, dz_mm=round(dz, 2), n_pairs=int(len(r)),
                         step_med_um=float(np.median(r)), step_p95_um=float(np.quantile(r, 0.95)),
                         step_mean_um=float(r.mean()),
                         straight_line_step_med_um=float(np.median(r_line)),
                         step_over_straight=float(np.median(r) / np.median(r_line)),
                         step_med_um_p_above_10GeV=float(np.median(r[p >= 10])),
                         step_med_um_p_below_5GeV=float(np.median(r[p < 5])),
                         chain_med_um=rec["test"]["vs_rk6_endpoint"]["pos_med_um"],
                         wall_s=round(time.time() - t0, 1)))
        for k in range(N):
            m = plane == k
            by_z.append(dict(N=N, q=q, plane=k, z_mm=float(Z0 + k * dz),
                             step_med_um=float(np.median(r[m])),
                             step_p95_um=float(np.quantile(r[m], 0.95)),
                             straight_line_med_um=float(np.median(r_line[m]))))
        print("N=%3d q=%2d: single step median %7.2f um (p95 %8.1f), straight line %9.1f um, "
              "ratio %.4f | %.0f s" % (N, q, rows[-1]["step_med_um"], rows[-1]["step_p95_um"],
                                       rows[-1]["straight_line_step_med_um"],
                                       rows[-1]["step_over_straight"], rows[-1]["wall_s"]), flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for name, data in (("error_qdz_single_step.csv", rows), ("single_step_vs_z.csv", by_z)):
        with open(os.path.join(HERE, "results", name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    labels = ["N = %d, q = %d\n(dz = %.0f mm)" % (x["N"], x["q"], x["dz_mm"]) for x in rows]
    xs = np.arange(len(rows))
    ax[0].bar(xs, [x["step_med_um"] for x in rows], color="#4c72b0", label="network, one step")
    ax[0].plot(xs, [x["straight_line_step_med_um"] for x in rows], "k_", ms=18, mew=2, label="straight line, one step")
    ax[0].set_yscale("log")
    ax[1].bar(xs, [x["chain_med_um"] / x["step_med_um"] for x in rows], color="#dd8452", label="chain / single step")
    ax[1].plot(xs, [x["N"] for x in rows], "k_", ms=18, mew=2, label="N (errors adding up)")
    ax[1].set_yscale("log")
    for a in ax[:2]:
        a.set_xticks(xs, labels, fontsize=8)
    for x in rows:
        z = [b for b in by_z if b["N"] == x["N"] and b["q"] == x["q"]]
        ax[2].plot([b["z_mm"] / 1000 for b in z], [b["step_med_um"] for b in z], "-",
                   label="N = %d, q = %d" % (x["N"], x["q"]))
    for a, t, yl in ((ax[0], "One step, median radial error", "median radial error of one step [µm]"),
                     (ax[1], "Chain error / single-step error", "ratio (N if the step errors simply added)"),
                     (ax[2], "Where along the magnet the step error sits", "median radial error of one step [µm]")):
        a.set_title(t, fontsize=10)
        a.set_ylabel(yl, fontsize=9)
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    ax[2].set_xlabel("z of the step's start plane [m]")
    ax[2].set_yscale("log")
    fig.suptitle("Block F: one step of each network from the RK6 state on every start plane, test tracks", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "single_step.png"), dpi=130)
    plt.close(fig)
    print("wrote results/error_qdz_single_step.csv, single_step_vs_z.csv, figures/single_step.png")
