#!/usr/bin/env python
"""F2 - the endpoint deviation around 5 GeV, signed, per component, per network.

Asked by George's supervisor (2026-09-21): for tracks of around 5 GeV, what is
the expected deviation - the RMS in x, or the |dx| of the tables - and the
SIGNED distributions, of the kind shown for 10-20 GeV in the case study.

For each network (endpoint after the full chain, test tracks, against the RK6
track at the first SciFi plane), in the bands 3-5, 4-6, 5-7 and 10-20 GeV:
n, median |d|, RMS (sqrt of the mean square, about zero), standard deviation
about the mean, the signed mean and median, the 68 percent half-width (half the
16th-84th percentile range, a tail-robust width), and the 95th percentile of
|d|. RMS and standard deviation are pulled by the tails; the median and the 68
percent half-width are not, and the two are reported side by side.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python errors_near_5gev.py
Outputs: results/errors_near_5gev.csv, figures/signed_errors_4-6GeV.png
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
COMP = (("x", 0, 1e3, "um"), ("y", 1, 1e3, "um"), ("tx", 2, 1e3, "mrad"), ("ty", 3, 1e3, "mrad"))
BANDS = ((3, 5), (4, 6), (5, 7), (10, 20))
FIG_BAND = (4, 6)


def networks():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        if not (os.path.exists(os.path.join(d, "record.json")) and os.path.exists(os.path.join(d, "chain_states.npz"))):
            continue
        rec = json.load(open(os.path.join(d, "record.json")))
        prog = json.load(open(os.path.join(d, "progress.json")))
        out.append(dict(N=rec["N"], q=rec["q"], end=np.load(os.path.join(d, "chain_states.npz"))["test_states"][:, -1],
                        checkpoint=prog.get("phase") != "done", restarts=rec["restarts"]))
    return out


def stats(v):
    lo, hi = np.percentile(v, [16, 84])
    return dict(n=int(len(v)), med_abs=float(np.median(np.abs(v))), rms=float(np.sqrt(np.mean(v ** 2))),
                std=float(np.std(v)), mean=float(np.mean(v)), signed_med=float(np.median(v)),
                hw68=float((hi - lo) / 2), p95_abs=float(np.percentile(np.abs(v), 95)))


def main():
    D = np.load(E0)
    n_max = int(D["n_max"])
    truth = np.asarray(D["test_truth"])[:, n_max]
    P = np.asarray(D["test_P"])
    nets = networks()
    rows = []
    for n in nets:
        d = n["end"][:, :4] - truth[:, :4]
        for lo, hi in BANDS:
            m = (P >= lo) & (P < hi)
            for name, i, s, unit in COMP:
                rows.append(dict(N=n["N"], q=n["q"], checkpoint=int(n["checkpoint"]), p_lo=lo, p_hi=hi,
                                 component=name, unit=unit, **stats(d[m, i] * s)))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "errors_near_5gev.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for lo, hi in BANDS:
        print("\n=== %d-%d GeV: endpoint deviation at z1, test tracks ===" % (lo, hi))
        print("%-26s %-3s %5s %9s %9s %9s %9s %9s %9s" % ("network", "", "n", "|median|", "RMS", "std", "mean", "sgn med", "68% hw"))
        for n in nets:
            for name, i, s, unit in COMP:
                r = [x for x in rows if x["N"] == n["N"] and x["q"] == n["q"] and x["p_lo"] == lo and x["component"] == name][0]
                fmt = "%9.3f" if unit == "mrad" else "%9.1f"
                print("%-26s %-3s %5d " % ("N=%d q=%d%s" % (n["N"], n["q"], " (ckpt)" if n["checkpoint"] else ""), name, r["n"])
                      + " ".join(fmt % r[k] for k in ("med_abs", "rms", "std", "mean", "signed_med", "hw68")) + "  " + unit)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = FIG_BAND
    m = (P >= lo) & (P < hi)
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    cmap = plt.get_cmap("tab10")
    for j, (name, i, s, unit) in enumerate(COMP):
        allv = np.concatenate([(n["end"][m, i] - truth[m, i]) * s for n in nets])
        lim = np.percentile(np.abs(allv), 98)
        bins = np.linspace(-lim, lim, 41)
        for k, n in enumerate(nets):
            v = (n["end"][m, i] - truth[m, i]) * s
            st = stats(v)
            ax[j].hist(np.clip(v, -lim, lim), bins=bins, histtype="step", lw=1.8, color=cmap(k),
                       label="N=%d q=%d%s: median %s, RMS %s, 68%% hw %s" % (
                           n["N"], n["q"], " (ckpt)" if n["checkpoint"] else "",
                           ("%+.3f" if unit == "mrad" else "%+.1f") % st["signed_med"],
                           ("%.3f" if unit == "mrad" else "%.0f") % st["rms"],
                           ("%.3f" if unit == "mrad" else "%.0f") % st["hw68"]))
        ax[j].axvline(0, color="k", lw=0.8)
        ax[j].set_xlabel("signed %s error at z1 [%s]  (clipped at the 98th percentile)" % (name, unit), fontsize=9)
        ax[j].set_ylabel("test tracks")
        ax[j].set_title("%s, %d-%d GeV (%d tracks)" % (name, lo, hi, m.sum()), fontsize=10)
        ax[j].legend(fontsize=7)
        ax[j].grid(alpha=0.3)
    fig.suptitle("Signed endpoint deviation around 5 GeV, per network, after the full chain from the last UT plane "
                 "(%d-%d GeV test tracks, against RK6)" % (lo, hi), fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "signed_errors_%d-%dGeV.png" % (lo, hi)), dpi=130)
    print("\nwrote results/errors_near_5gev.csv, figures/signed_errors_%d-%dGeV.png" % (lo, hi))


if __name__ == "__main__":
    main()
