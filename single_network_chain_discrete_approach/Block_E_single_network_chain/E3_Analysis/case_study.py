#!/usr/bin/env python
"""E3 - one network in depth: the best of the sixteen, on its own.

The network is chosen on the VALIDATION tracks (the mean error over the last
eight training rounds, which is steadier than any single checkpoint), so the
test tracks stay untouched until the numbers below. Everything the grid tables
show is repeated here for that one network, plus what only makes sense for a
single model:

  1. the headline: radial error at the SciFi plane, its tails, its cost, and
     the comparators (the exact scheme at the same N and q, the straight line,
     the material floor, Block D's one-network-per-step chain);
  2. each component against momentum, and each component's distribution;
  3. the 10-20 GeV band on its own, signed as well as absolute, which shows
     whether the network bends too much or too little rather than just by how
     much it misses;
  4. how the error grows along the crossing, beside what one step costs at each
     plane;
  5. how training converged: the loss and the validation error;
  6. the tails: who the worst 5 percent of tracks are;
  7. error against pseudorapidity and charge;
  8. the loss itself against momentum - the residual the network was trained to
     minimise, per track. If the low-momentum tracks carry the error but not
     the loss, the loss weighting is the thing to change next.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python case_study.py [--N 64 --q 2]
Outputs: results/case_study_*.csv, figures/case_study_*.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse   # noqa: E402
import csv        # noqa: E402
import json       # noqa: E402
import sys        # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, "..", "E1_Network_grid")
E2 = os.path.join(HERE, "..", "E2_Comparators", "results")
BD = os.path.join(HERE, "..", "..", "..", "multi_network_chain_discrete_approach",
                  "Block_D_fixed_step_crossing")
TRACKS = os.path.join(HERE, "..", "E0_Track_dataset", "results", "tracks.npz")
sys.path.insert(0, E1)

import use_shared   # noqa: E402,F401
from _shared.model import LHCbRates, reconstruction_residuals   # noqa: E402
from _shared.reference import gauss_legendre, make_field        # noqa: E402
from chain_network import load_network, step_outputs, znodes_for  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

SPLIT = "test"
COMPONENTS = (("x", 0, 1e3, "µm"), ("y", 1, 1e3, "µm"), ("tx", 2, 1e3, "mrad"), ("ty", 3, 1e3, "mrad"))
P_EDGES = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200.0])
BAND = (10.0, 20.0)


def pick_best():
    rows = list(csv.DictReader(open(os.path.join(HERE, "results", "convergence.csv"))))
    best = min(rows, key=lambda r: float(r["val_last8_mean"]))
    return int(best["N"]), int(best["q"]), rows


def radial_um(a, b):
    return np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]) * 1e3


def write(name, rows):
    with open(os.path.join(HERE, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--q", type=int, default=None)
    a = ap.parse_args(argv)
    bestN, bestq, conv = pick_best()
    N, q = (a.N or bestN), (a.q or bestq)
    run = os.path.join(E1, "results", "N%03d_q%02d" % (N, q))
    rec = json.load(open(os.path.join(run, "record.json")))
    print("case study: N = %d, q = %d (chosen on validation: %s)"
          % (N, q, "yes" if (N, q) == (bestN, bestq) else "by hand"))

    D = np.load(TRACKS)
    Z0, L, n_max = float(D["z0"]), float(D["L"]), int(D["n_max"])
    stride, dz = n_max // N, L / N
    truth = np.asarray(D["%s_truth" % SPLIT])
    truth_end = truth[:, n_max]
    P, ETA = np.asarray(D["%s_P" % SPLIT]), np.asarray(D["%s_ETA" % SPLIT])
    S0 = np.asarray(D["%s_S0" % SPLIT])
    charge = np.sign(S0[:, 4])
    fld = make_field(str(D["field"]))
    st = np.load(os.path.join(run, "chain_states.npz"))["%s_states" % SPLIT]
    end = st[:, -1]
    d = end[:, :4] - truth_end[:, :4]
    r = radial_um(end, truth_end)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    # -- 1. the headline and the comparators ---------------------------------------
    ex = os.path.join(E2, "exact_N%03d_q%02d_states.npz" % (N, q))
    exd = os.path.join(BD, "D2_Comparators", "results", "exact_N%03d_q%02d_states.npz" % (N, q))
    exf = ex if os.path.exists(ex) else exd
    bd = os.path.join(BD, "D1_Chain_grid", "results", "N%03d_q%02d" % (N, q), "states.npz")
    line = np.stack([S0[:, 0] + S0[:, 2] * L, S0[:, 1] + S0[:, 3] * L], axis=1)
    head = [dict(what="this network (N = %d, q = %d)" % (N, q), med_um=float(np.median(r)),
                 p95_um=float(np.quantile(r, 0.95)), p99_um=float(np.quantile(r, 0.99)),
                 max_um=float(r.max()), frac_above_1mm=float((r > 1000).mean()))]
    if os.path.exists(exf):
        e = radial_um(np.load(exf)["states"][:, -1], truth_end)
        head.append(dict(what="the exact scheme at the same N and q", med_um=float(np.median(e)),
                         p95_um=float(np.quantile(e, 0.95)), p99_um=float(np.quantile(e, 0.99)),
                         max_um=float(e.max()), frac_above_1mm=float((e > 1000).mean())))
    if os.path.exists(bd):
        b = radial_um(np.load(bd)["%s_states" % SPLIT][:, -1], truth_end)
        head.append(dict(what="Block D: one network per step, same N and q", med_um=float(np.median(b)),
                         p95_um=float(np.quantile(b, 0.95)), p99_um=float(np.quantile(b, 0.99)),
                         max_um=float(b.max()), frac_above_1mm=float((b > 1000).mean())))
    ln = radial_um(line, truth_end)
    mat = radial_um(np.asarray(D["%s_S_post" % SPLIT]), np.asarray(D["%s_truth_zpost" % SPLIT]))
    head += [dict(what="the straight line", med_um=float(np.median(ln)), p95_um=float(np.quantile(ln, 0.95)),
                  p99_um=float(np.quantile(ln, 0.99)), max_um=float(ln.max()),
                  frac_above_1mm=float((ln > 1000).mean())),
             dict(what="the material floor (real SciFi state vs the RK6 track)", med_um=float(np.median(mat)),
                  p95_um=float(np.quantile(mat, 0.95)), p99_um=float(np.quantile(mat, 0.99)),
                  max_um=float(mat.max()), frac_above_1mm=float((mat > 1000).mean()))]
    write("case_study_headline.csv", head)
    for h in head:
        print("  %-52s median %8.1f um  p95 %9.1f  p99 %9.1f" % (h["what"], h["med_um"], h["p95_um"], h["p99_um"]))

    # -- 2, 3. components against momentum, and the 10-20 GeV band -----------------
    band = (P >= BAND[0]) & (P <= BAND[1])
    comp_rows = []
    for name, i, scale, unit in COMPONENTS:
        v, sgn = np.abs(d[:, i]) * scale, d[:, i] * scale
        for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
            m = (P >= lo) & (P < hi)
            if m.sum() < 5:
                continue
            comp_rows.append(dict(component=name, unit=unit, p_lo=lo, p_hi=hi, n=int(m.sum()),
                                  med=float(np.median(v[m])), p95=float(np.quantile(v[m], 0.95)),
                                  signed_mean=float(sgn[m].mean()), signed_median=float(np.median(sgn[m]))))
        comp_rows.append(dict(component=name, unit=unit, p_lo=BAND[0], p_hi=BAND[1], n=int(band.sum()),
                              med=float(np.median(v[band])), p95=float(np.quantile(v[band], 0.95)),
                              signed_mean=float(sgn[band].mean()), signed_median=float(np.median(sgn[band]))))
    write("case_study_components.csv", comp_rows)
    print("  10-20 GeV (%d tracks): " % band.sum() + ", ".join(
        "%s %.3g %s (bias %+.3g)" % (n, np.median(np.abs(d[band, i]) * s), u, np.mean(d[band, i]) * s)
        for n, i, s, u in COMPONENTS))

    # -- 6, 7. tails, pseudorapidity, charge ---------------------------------------
    worst = r >= np.quantile(r, 0.95)
    tail = [dict(group="worst 5 percent", n=int(worst.sum()), med_r_um=float(np.median(r[worst])),
                 med_p_GeV=float(np.median(P[worst])), med_eta=float(np.median(ETA[worst])),
                 frac_below_5GeV=float((P[worst] < 5).mean()), frac_above_25GeV=float((P[worst] > 25).mean())),
            dict(group="the rest", n=int((~worst).sum()), med_r_um=float(np.median(r[~worst])),
                 med_p_GeV=float(np.median(P[~worst])), med_eta=float(np.median(ETA[~worst])),
                 frac_below_5GeV=float((P[~worst] < 5).mean()), frac_above_25GeV=float((P[~worst] > 25).mean()))]
    for lo, hi in ((2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4, 4.5), (4.5, 5)):
        m = (ETA >= lo) & (ETA < hi)
        if m.sum() > 5:
            tail.append(dict(group="eta %.1f-%.1f" % (lo, hi), n=int(m.sum()), med_r_um=float(np.median(r[m])),
                             med_p_GeV=float(np.median(P[m])), med_eta=float(np.median(ETA[m])),
                             frac_below_5GeV=float((P[m] < 5).mean()), frac_above_25GeV=float((P[m] > 25).mean())))
    for c, lab in ((1, "positive"), (-1, "negative")):
        m = charge == c
        tail.append(dict(group="charge %s" % lab, n=int(m.sum()), med_r_um=float(np.median(r[m])),
                         med_p_GeV=float(np.median(P[m])), med_eta=float(np.median(ETA[m])),
                         frac_below_5GeV=float((P[m] < 5).mean()), frac_above_25GeV=float((P[m] > 25).mean())))
    write("case_study_tails.csv", tail)
    print("  worst 5%%: median %.0f um, median momentum %.1f GeV (%.0f%% below 5 GeV); "
          "the rest: %.0f um at %.1f GeV" % (tail[0]["med_r_um"], tail[0]["med_p_GeV"],
                                             100 * tail[0]["frac_below_5GeV"], tail[1]["med_r_um"],
                                             tail[1]["med_p_GeV"]))

    # -- 8. the loss against momentum, per track -----------------------------------
    model = load_network(run, fld)
    rates = LHCbRates(make_field(str(D["field"])))
    c, A_np, b_np = gauss_legendre(q)
    S = truth[:, 0:n_max:stride].reshape(-1, 5)
    plane = np.repeat(np.arange(N)[None, :], len(truth), 0).reshape(-1)
    z_start = Z0 + plane * dz
    with torch.no_grad():
        res = reconstruction_residuals(model, rates, torch.as_tensor(S), dz,
                                       znodes_for(model, torch.as_tensor(z_start)),
                                       torch.tensor(A_np), torch.tensor(b_np),
                                       torch.as_tensor(z_start)).numpy()
    per_state = (res ** 2).mean(axis=(1, 2))
    per_track = per_state.reshape(len(truth), N).mean(axis=1)
    step_out = step_outputs(model, S, z_start)[:, -1, :]
    nxt = truth[:, stride:n_max + 1:stride].reshape(-1, 5)
    step_r = np.hypot(step_out[:, 0] - nxt[:, 0], step_out[:, 1] - nxt[:, 1]) * 1e3
    step_track = step_r.reshape(len(truth), N)
    loss_rows = []
    for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
        m = (P >= lo) & (P < hi)
        if m.sum() < 5:
            continue
        loss_rows.append(dict(p_lo=lo, p_hi=hi, n=int(m.sum()),
                              loss_med=float(np.median(per_track[m])),
                              loss_share=float(per_track[m].sum() / per_track.sum()),
                              step_med_um=float(np.median(step_track[m])),
                              chain_med_um=float(np.median(r[m]))))
    write("case_study_loss_vs_p.csv", loss_rows)
    print("  loss share by momentum:", ", ".join("%g-%g: %.0f%%" % (x["p_lo"], x["p_hi"], 100 * x["loss_share"])
                                                 for x in loss_rows))

    # -- the figures ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    centres = np.sqrt(P_EDGES[:-1] * P_EDGES[1:])
    title = "Block E case study: N = %d, q = %d (dz = %.1f mm), %d test tracks" % (N, q, dz, len(r))

    fig, ax = plt.subplots(2, 4, figsize=(19, 8))
    for j, (name, i, scale, unit) in enumerate(COMPONENTS):
        v = np.abs(d[:, i]) * scale
        hb = ax[0, j].hexbin(P, np.clip(v, 1e-4, np.quantile(v, 0.999)), xscale="log", yscale="log",
                             gridsize=36, mincnt=1, norm=LogNorm(), cmap="Blues")
        cb = fig.colorbar(hb, ax=ax[0, j], pad=0.02)
        cb.set_label("test tracks per cell", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        med = [np.median(v[(P >= lo) & (P < h2)]) for lo, h2 in zip(P_EDGES[:-1], P_EDGES[1:])]
        ax[0, j].plot(centres, med, "r-o", ms=3, lw=1.4, label="median per momentum bin")
        ax[0, j].legend(fontsize=7, loc="upper right")
        ax[0, j].set_xlabel("momentum [GeV]")
        ax[0, j].set_ylabel("|%s error| [%s]" % (name, unit))
        ax[0, j].set_title("%s against momentum (median %.3g %s)" % (name, np.median(v), unit), fontsize=10)
        ax[0, j].grid(alpha=0.25, which="both")
        sgn = d[band, i] * scale
        lim = np.quantile(np.abs(sgn), 0.99)
        ax[1, j].hist(np.clip(sgn, -lim, lim), bins=60, color="#4c72b0")
        ax[1, j].axvline(0, color="k", lw=0.8)
        ax[1, j].axvline(np.median(sgn), color="r", lw=1.2,
                         label="median %+.3g %s" % (np.median(sgn), unit))
        ax[1, j].axvline(np.mean(sgn), color="darkorange", lw=1.2, ls="--",
                         label="mean %+.3g %s (tail-pulled)" % (np.mean(sgn), unit))
        ax[1, j].set_ylabel("tracks", fontsize=8)
        ax[1, j].set_xlabel("signed %s error, %g-%g GeV [%s]" % (name, BAND[0], BAND[1], unit))
        ax[1, j].legend(fontsize=8)
        ax[1, j].set_title("%d tracks at %g-%g GeV" % (band.sum(), BAND[0], BAND[1]), fontsize=10)
    fig.suptitle(title + " — top: every test track, coloured by how many fall in each cell; bottom: the 10-20 GeV band, signed", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "case_study_components.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    zz = [Z0 + k * dz for k in range(N + 1)]
    med_z = [float(np.median(np.hypot(st[:, k, 0] - truth[:, k * stride, 0],
                                      st[:, k, 1] - truth[:, k * stride, 1])) * 1e3) for k in range(N + 1)]
    ax[0].plot(np.array(zz) / 1000, med_z, "-", color="#1f77b4", label="chain error")
    ax[0].plot(np.array(zz[:-1]) / 1000, np.median(step_track, axis=0), "-", color="#d62728",
               label="one step from the RK6 track")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("z [m]")
    ax[0].set_ylabel("median radial error [µm]")
    ax[0].set_title("Growth along the crossing", fontsize=10)
    ax[0].legend(fontsize=8)
    hist = list(csv.DictReader(open(os.path.join(run, "history.csv"))))
    rounds = list(csv.DictReader(open(os.path.join(run, "rounds.csv"))))
    ax[1].plot([int(x["restart"]) for x in hist], [float(x["loss_after"]) for x in hist],
               color="#1f77b4", lw=0.9)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("L-BFGS restart")
    ax[1].set_ylabel("training loss", color="#1f77b4")
    b2 = ax[1].twinx()
    cum = np.cumsum([int(x["restarts"]) for x in rounds])
    b2.plot(cum - 1, [float(x["val_z1_pos_med_um"]) for x in rounds], "-s", color="#d62728", ms=3)
    b2.set_ylabel("validation error at z1 [µm]", color="#d62728")
    ax[1].set_title("Convergence: the loss fell, the error stopped", fontsize=10)
    for lo, hi, lab in ((1, 5, "1-5 GeV"), (5, 20, "5-20 GeV"), (20, 200, "20-200 GeV")):
        m = (P >= lo) & (P < hi)
        ax[2].hist(np.clip(r[m], 1, 5000), bins=np.logspace(0, np.log10(5000), 50), histtype="step",
                   lw=1.4, density=True, label="%s (median %.0f µm)" % (lab, np.median(r[m])))
    ax[2].set_xscale("log")
    ax[2].set_xlabel("radial error at the SciFi plane [µm]")
    ax[2].set_ylabel("density")
    ax[2].set_title("The distribution, by momentum", fontsize=10)
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "case_study_overview.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(centres, [x["loss_med"] for x in loss_rows], "-o", color="#1f77b4", label="median loss per track")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("momentum [GeV]")
    ax[0].set_ylabel("the trained residual, per track")
    a2 = ax[0].twinx()
    a2.plot(centres, [x["chain_med_um"] for x in loss_rows], "-s", color="#d62728")
    a2.set_ylabel("median chain error [µm]", color="#d62728")
    a2.set_yscale("log")
    ax[0].set_title("What the loss sees against what the chain costs", fontsize=10)
    ax[1].bar(range(len(loss_rows)), [100 * x["loss_share"] for x in loss_rows], color="#4c72b0")
    ax[1].set_xticks(range(len(loss_rows)), ["%g-%g" % (x["p_lo"], x["p_hi"]) for x in loss_rows],
                     rotation=45, fontsize=8)
    ax[1].set_xlabel("momentum band [GeV]")
    ax[1].set_ylabel("share of the total loss [%]")
    ax[1].set_title("Which tracks the loss is actually made of", fontsize=10)
    for a in (ax[0], ax[1]):
        a.grid(alpha=0.3, which="both")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "case_study_loss_vs_p.png"), dpi=130)
    print("wrote results/case_study_*.csv and figures/case_study_{components,overview,loss_vs_p}.png")


if __name__ == "__main__":
    main()
