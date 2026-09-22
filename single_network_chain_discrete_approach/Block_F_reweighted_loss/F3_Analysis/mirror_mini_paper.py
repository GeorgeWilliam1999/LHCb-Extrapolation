#!/usr/bin/env python
"""F3 - two figures of the mini paper (Mini_paper/main.pdf, Figures 2 and 6) mirrored for
the reweighted-loss networks, on their own.

George (2026-09-21): "please can you make plots mirroring Mini_paper/main.pdf
figure 2, figure 6"; then: no block labels in the outputs, the networks are
looked at in isolation.

Neither figure maps one-to-one: Figure 2 (fig_baseline) compares three seeds
of two losses on two samples with a fiducial cut; Figure 6 (fig_magnet_up)
compares the two magnet polarities. Each panel here is the nearest analogue
for one set of networks, each named by its step count and stage count:

  Figure 2 (2 x 3)
    histogram           ENDPOINT radial error at z1 per network, the exact
                        scheme's ceiling and the straight line marked
    loss histories      training loss per L-BFGS restart, per network
    error vs momentum   binned medians of the endpoint error, per network
    per run             filled circles = endpoint median; squares = the
                        SINGLE-STEP error (one application from an RK6 state)
    by band             endpoint median for all / 10-50 GeV / below 5 GeV,
                        one bar per network
    trajectories        x(z) of test tracks carried by the N = 64, q = 2
                        network; the worst 5 percent in pink, 150 others on top
  Figure 6 (1 x 3)
    histogram           counts of the endpoint error per network (the
                        validation-best filled), straight line grey, exact
                        ceilings marked, each network's median as a thin line
    scatter vs momentum every test track, the validation-best network, with
                        the binned median on top
    three tracks        the sign check: three test tracks of BOTH charges (the
                        analogue of both polarities), RK6 reference solid, the
                        network's chain states as markers - the bend changes
                        sign with the charge and the network follows it

Every error here is an ENDPOINT error - the network applied N times from the
track's real state on the last UT plane, compared with the RK6 track at the
first SciFi plane - except the squares of the per-run panel, which are
single-step errors. A network whose record is a checkpoint of a run still
training is marked.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python mirror_mini_paper.py
Outputs: figures/mirror_fig2_baseline.png, figures/mirror_fig6_magnet_up.png,
         results/mirror_mini_paper.json
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E0 = os.path.join(HERE, "..", "..", "Block_E_single_network_chain", "E0_Track_dataset", "results", "tracks.npz")
RUNS = os.path.join(HERE, "..", "F1_Training", "results", "full")
SPLIT = "test"
BAND = (10.0, 50.0)
P_EDGES = np.geomspace(1.5, 200, 13)
GREEN, BLUE, PINK = "#2ca02c", "#1f77b4", "#e377c2"


def radial(end, truth):
    return np.hypot(end[:, 0] - truth[:, 0], end[:, 1] - truth[:, 1]) * 1e3


def binned_median(P, r):
    c = np.sqrt(P_EDGES[:-1] * P_EDGES[1:])
    m = [np.median(r[(P >= lo) & (P < hi)]) if ((P >= lo) & (P < hi)).sum() >= 5 else np.nan
         for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:])]
    return c, np.array(m)


def networks():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        if not (os.path.exists(os.path.join(d, "record.json")) and os.path.exists(os.path.join(d, "chain_states.npz"))):
            continue
        rec = json.load(open(os.path.join(d, "record.json")))
        prog = json.load(open(os.path.join(d, "progress.json")))
        st = np.load(os.path.join(d, "chain_states.npz"))["%s_states" % SPLIT]
        hist = list(csv.DictReader(open(os.path.join(d, "history.csv"))))
        out.append(dict(tag=os.path.basename(d), N=rec["N"], q=rec["q"], rec=rec, states=st, end=st[:, -1],
                        loss=np.array([float(h["loss_after"]) for h in hist]),
                        checkpoint=prog.get("phase") != "done"))
    return out


def name(n):
    s = "N = %d, q = %d" % (n["N"], n["q"])
    if n["checkpoint"]:
        s += " (checkpoint at restart %d; still training)" % n["rec"]["restarts"]
    return s


def single_step_medians():
    p = os.path.join(HERE, "results", "error_qdz_single_step.csv")
    if not os.path.exists(p):
        return {}
    return {(int(r["N"]), int(r["q"])): float(r["step_med_um"]) for r in csv.DictReader(open(p))}


def exact_ceiling(N, q):
    p = os.path.join(HERE, "results", "error_qdz_chain.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if int(r["N"]) == N and int(r["q"]) == q and r["exact_med_um"] not in ("", "nan"):
                return float(r["exact_med_um"])
    return np.nan


def main():
    D = np.load(E0)
    Z0, Z1, L, n_max = float(D["z0"]), float(D["z1"]), float(D["L"]), int(D["n_max"])
    S0 = np.asarray(D["%s_S0" % SPLIT])
    truth = np.asarray(D["%s_truth" % SPLIT])
    truth_end = truth[:, n_max]
    P = np.asarray(D["%s_P" % SPLIT])
    charge = np.sign(S0[:, 4])
    straight = np.stack([S0[:, 0] + S0[:, 2] * L, S0[:, 1] + S0[:, 3] * L], axis=1)
    r_line = radial(straight, truth_end)
    nets = networks()
    if not nets:
        raise SystemExit("no network with a record and chain states")
    steps = single_step_medians()
    best = min(nets, key=lambda n: n["rec"]["val"]["vs_rk6_endpoint"]["pos_med_um"])
    ceilings = {n["tag"]: exact_ceiling(n["N"], n["q"]) for n in nets}
    foot = "; ".join("%s is a checkpoint of a run still training" % name(n).split(" (")[0]
                     for n in nets if n["checkpoint"]) or "all runs finished"
    rec_out = dict(networks=[n["tag"] for n in nets], best_on_validation=best["tag"],
                   checkpoints_of_runs_still_training=[n["tag"] for n in nets if n["checkpoint"]],
                   straight_line_med_um=float(np.median(r_line)),
                   errors="endpoint (after N steps, at z1) everywhere except the single-step squares of the per-run panel")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    bins = np.geomspace(0.3, 3e6, 70)

    # ================================================================ Figure 2 ==
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    a = ax[0, 0]
    for i, n in enumerate(nets):
        r = radial(n["end"], truth_end)
        a.hist(r, bins=bins, density=True, histtype="step", lw=2.0, color=cmap(i), label="%s (%.0f um)" % (name(n), np.median(r)))
    ex = ceilings[best["tag"]]
    if np.isfinite(ex):
        a.axvline(ex, color="k", ls=":", lw=1.2)
        a.text(ex * 1.3, a.get_ylim()[1] * 0.92, "exact-scheme\nceiling %.1f um" % ex, fontsize=8, color="0.3")
    a.axvline(np.median(r_line), color="0.5", lw=1.2)
    a.text(np.median(r_line) / 8, a.get_ylim()[1] * 0.6, "straight\nline", fontsize=8, color="0.3")
    a.set_xscale("log")
    a.set_xlabel("radial endpoint error vs RK6 [um]")
    a.set_ylabel("density")
    a.set_title("held-out endpoint error, per network", fontsize=10)
    a.legend(fontsize=7)

    a = ax[0, 1]
    for i, n in enumerate(nets):
        a.plot(np.arange(len(n["loss"])), n["loss"], "-", color=cmap(i), lw=1.0, label=name(n))
    a.set_yscale("log")
    a.set_xlabel("L-BFGS restart")
    a.set_ylabel("training loss")
    a.set_title("loss histories (every restart; the saw-teeth are the round boundaries)", fontsize=10)
    a.legend(fontsize=7)

    a = ax[0, 2]
    for i, n in enumerate(nets):
        c, mm = binned_median(P, radial(n["end"], truth_end))
        a.plot(c, mm, "-o", color=cmap(i), ms=4, label=name(n))
    if np.isfinite(ex):
        a.axhline(ex, color="k", ls=":", lw=1.2)
    a.axvspan(BAND[0], BAND[1], color="0.93", zorder=0)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("|p| [GeV]")
    a.set_ylabel("median radial endpoint error [um]")
    a.set_title("endpoint error vs momentum (shaded: 10-50 GeV, the band the loss weights up)", fontsize=10)
    a.legend(fontsize=7)

    a = ax[1, 0]
    for i, n in enumerate(nets):
        a.plot(i, np.median(radial(n["end"], truth_end)), "o", color=cmap(i), ms=10)
        a.plot(i, steps.get((n["N"], n["q"]), np.nan), "s", color=cmap(i), ms=7)
    if np.isfinite(ex):
        a.axhline(ex, color="k", ls=":", lw=1.2)
    a.set_xticks(range(len(nets)), ["N=%d\nq=%d%s" % (n["N"], n["q"], "\n(checkpoint)" if n["checkpoint"] else "") for n in nets], fontsize=8)
    a.set_yscale("log")
    a.set_ylabel("median error [um]")
    a.set_title("per network: circles = endpoint after N steps; squares = ONE step from the RK6 state", fontsize=10)

    a = ax[1, 1]
    groups = (("all", np.ones_like(P, dtype=bool)), ("10-50 GeV", (P >= BAND[0]) & (P < BAND[1])), ("below 5 GeV", P < 5))
    w = 0.8 / len(nets)
    for i, n in enumerate(nets):
        vals = [np.median(radial(n["end"][m], truth_end[m])) for _, m in groups]
        a.bar(np.arange(len(groups)) + (i - (len(nets) - 1) / 2) * w, vals, w, color=cmap(i), alpha=0.9, label=name(n))
        rec_out["band_medians_%s" % n["tag"]] = dict(zip([g[0] for g in groups], vals))
    a.set_xticks(range(len(groups)), [g[0] for g in groups])
    a.set_ylabel("median radial endpoint error [um]")
    a.set_title("endpoint error by momentum band, per network", fontsize=10)
    a.legend(fontsize=7)

    a = ax[1, 2]
    n = [x for x in nets if x["tag"] == best["tag"]][0]
    r = radial(n["end"], truth_end)
    worst = r >= np.quantile(r, 0.95)
    rng = np.random.default_rng(20260921)
    keep = rng.choice(np.flatnonzero(~worst), 150, replace=False)
    zs = Z0 + np.arange(n["N"] + 1) * (L / n["N"])
    for i in np.flatnonzero(worst):
        a.plot(zs, n["states"][i, :, 0], "-", color=PINK, lw=0.8, alpha=0.45)
    for i in keep:
        a.plot(zs, n["states"][i, :, 0], "-", color="0.35", lw=0.7, alpha=0.75)
    a.set_xlabel("z [mm]")
    a.set_ylabel("x [mm]")
    a.set_title("the worst 5%% of test tracks (pink, %d) under the N=%d q=%d chain;\n150 others in dark grey on top"
                % (worst.sum(), n["N"], n["q"]), fontsize=10)
    fig.suptitle("One network per step length, chained across the magnet, mirroring the mini paper's Figure 2: "
                 "test split, n=%d, radial ENDPOINT error at the SciFi plane\n(%s)" % (len(P), foot), fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "mirror_fig2_baseline.png"), dpi=130)
    plt.close(fig)

    # ================================================================ Figure 6 ==
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.2))
    a = ax[0]
    bins6 = np.geomspace(0.3, 1e7, 80)
    for i, n in enumerate(nets):
        r = radial(n["end"], truth_end)
        if n["tag"] == best["tag"]:
            a.hist(r, bins=bins6, color=GREEN, alpha=0.35, label=name(n) + " (validation-best)")
            a.hist(r, bins=bins6, histtype="step", color=GREEN, lw=1.6)
        else:
            a.hist(r, bins=bins6, histtype="step", color=cmap(i), lw=1.1, label=name(n))
        a.axvline(np.median(r), color=GREEN if n["tag"] == best["tag"] else cmap(i), lw=0.8, alpha=0.8)
    a.hist(r_line, bins=bins6, histtype="step", color="0.55", lw=1.5, label="straight line")
    fin = [v for v in ceilings.values() if np.isfinite(v)]
    if fin:
        cx = min(fin)
        a.axvline(cx, color="0.3", ls=":", lw=1.2)
        a.text(cx * 1.4, a.get_ylim()[1] * 0.62, "exact-scheme ceiling\non these chains:\n%s" % "\n".join(
            "%.2f um (N=%d, q=%d)" % (ceilings[n["tag"]], n["N"], n["q"]) for n in nets if np.isfinite(ceilings[n["tag"]])),
            fontsize=8, color="0.3", va="top")
    a.set_xscale("log")
    a.set_xlabel("radial endpoint error [um]")
    a.set_ylabel("test tracks")
    a.set_title("endpoint error on the test split, per network\n(thin vertical lines: each network's median)", fontsize=11)
    a.legend(fontsize=8)

    a = ax[1]
    n = best
    r = radial(n["end"], truth_end)
    a.scatter(P, r, s=5, color=GREEN, alpha=0.4, label=name(n))
    c, mm = binned_median(P, r)
    a.plot(c, mm, "-", color="#145214", lw=3, label="binned median")
    if np.isfinite(ceilings[n["tag"]]):
        a.axhline(ceilings[n["tag"]], color="0.5", ls=":", lw=1.2)
    a.axvspan(BAND[0], BAND[1], color="0.93", zorder=0)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("track momentum p [GeV]")
    a.set_ylabel("radial endpoint error [um]")
    a.set_title("where the endpoint error lives in momentum (shaded: the band the loss weights up)", fontsize=11)
    a.legend(fontsize=8)

    a = ax[2]
    want = ((2.3, -1), (5.9, +1), (22.0, -1))
    picks = []
    for p0, sgn in want:
        cand = np.flatnonzero((charge == sgn) & (np.abs(S0[:, 0]) < 150) & (np.abs(S0[:, 1]) < 150))
        picks.append(cand[np.argmin(np.abs(P[cand] - p0))])
    zt = Z0 + np.arange(n_max + 1) * (L / n_max)
    zs = Z0 + np.arange(n["N"] + 1) * (L / n["N"])
    every = max(1, n["N"] // 8)
    for i in picks:
        col = GREEN if charge[i] > 0 else BLUE
        a.plot(zt / 1000, truth[i, :, 0], "-", color=col, lw=1.5)
        a.plot(zs[::every] / 1000, n["states"][i, ::every, 0], "o" if charge[i] > 0 else "s", mfc="none", mec=col, ms=6, mew=1.5)
        a.text(zt[-1] / 1000 + 0.05, truth[i, -1, 0], "p = %.1f GeV, %s" % (P[i], "+" if charge[i] > 0 else "-"), fontsize=8, va="center")
    a.plot([], [], "-", color=GREEN, label="positive charge: RK6 reference")
    a.plot([], [], "-", color=BLUE, label="negative charge: RK6 reference")
    a.plot([], [], "o", mfc="none", mec=GREEN, label="network chain states (every %dth plane), N=%d q=%d" % (every, n["N"], n["q"]))
    a.plot([], [], "s", mfc="none", mec=BLUE, label="network chain states (every %dth plane)" % every)
    a.set_xlabel("z [m]")
    a.set_ylabel("x [mm]")
    a.set_xlim(Z0 / 1000 - 0.1, Z1 / 1000 + 1.0)
    a.set_title("three test tracks, both charges\n(the bend changes sign with the charge; the network follows it)", fontsize=11)
    a.legend(fontsize=8, loc="best")
    rec_out["three_tracks"] = [dict(index=int(i), p_GeV=float(P[i]), charge=int(charge[i]),
                                    x_end_ref_mm=float(truth[i, -1, 0]), x_end_net_mm=float(n["states"][i, -1, 0])) for i in picks]
    fig.suptitle("One network per step length, chained across the magnet, mirroring the mini paper's Figure 6 "
                 "(the reversed-polarity figure): the two charges play the two polarities\n(%s)" % foot, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "mirror_fig6_magnet_up.png"), dpi=130)
    plt.close(fig)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "mirror_mini_paper.json"), "w") as f:
        json.dump(rec_out, f, indent=1)
    print("wrote figures/mirror_fig2_baseline.png, figures/mirror_fig6_magnet_up.png (%s)" % foot)


if __name__ == "__main__":
    main()
