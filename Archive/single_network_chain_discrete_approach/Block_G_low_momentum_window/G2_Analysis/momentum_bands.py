#!/usr/bin/env python
"""G2 - the endpoint error against momentum, band by band, per network.

The secondary measure of the study: moving the loss window down is meant to
buy accuracy around 5 GeV, and what it costs is read here, in the 10-50 GeV
band that the earlier window was aimed at.

For each network, on the test tracks, the RADIAL distance at the first SciFi
plane (z1 = 7,826 mm) between the network's endpoint after the full chain and
the RK6 endpoint of the same track, r = sqrt(dx^2 + dy^2): the median and the
95th percentile in the bands 2-3, 3-5, 5-8, 8-10, 10-20, 20-50, 50-100 and
100-200 GeV, with |dx| and |dy| medians beside them. Two extra rows per
network: the whole test set, and 10-50 GeV as one band (the secondary
number of the pre-registration).

These are ENDPOINT errors, after N applications of the network from the track's
real state on the last UT plane - not single-step errors.

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python momentum_bands.py [--runs DIR] [--out DIR]
Outputs: <out>/results/momentum_bands.csv, <out>/figures/momentum_bands.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys           # noqa: E402
# importing a script out of a read-only folder must not leave a .pyc behind in it
sys.dont_write_bytecode = True

import argparse   # noqa: E402
import csv        # noqa: E402

import numpy as np   # noqa: E402

import run_discovery as rd   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E0 = os.path.join(HERE, "..", "..", "Block_E_single_network_chain", "E0_Track_dataset",
                  "results", "tracks.npz")
DEFAULT_RUNS = os.path.join(HERE, "..", "G1_Training", "results", "p03-08", "full")
P_EDGES = np.array([2, 3, 5, 8, 10, 20, 50, 100, 200.0])
EXTRA = ((10.0, 50.0),)          # reported as one band as well as 10-20 + 20-50
SPLIT = "test"
TAIL_UM = 1000.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--only", default=None, help="comma-separated run folders, e.g. N064_q02")
    a = ap.parse_args(argv)
    nets = rd.require(rd.find_runs(a.runs, a.only), a.runs)
    print(rd.describe(nets, a.runs))

    D = np.load(E0)
    n_max = int(D["n_max"])
    truth = np.asarray(D["%s_truth" % SPLIT])[:, n_max]
    P = np.asarray(D["%s_P" % SPLIT])
    bands = list(zip(P_EDGES[:-1], P_EDGES[1:])) + list(EXTRA)
    rows = []
    for n in nets:
        end = np.load(os.path.join(n["path"], "chain_states.npz"))["%s_states" % SPLIT][:, -1]
        d = end[:, :4] - truth[:, :4]
        rad = np.hypot(d[:, 0], d[:, 1]) * 1e3
        for lo, hi in bands + [(float(P.min()), float(P.max()) + 1.0)]:
            whole = (lo, hi) not in bands
            m = (P >= lo) & (P < hi)
            if m.sum() < 5:
                continue
            rows.append(dict(N=n["N"], q=n["q"], dz_mm=round(n["dz_mm"], 1),
                             kind="endpoint (after N steps, at z1)",
                             p_lo=("all" if whole else "%g" % lo), p_hi=("all" if whole else "%g" % hi),
                             n=int(m.sum()),
                             rad_med_um=float(np.median(rad[m])),
                             rad_p95_um=float(np.quantile(rad[m], 0.95)),
                             x_med_abs_um=float(np.median(np.abs(d[m, 0])) * 1e3),
                             y_med_abs_um=float(np.median(np.abs(d[m, 1])) * 1e3),
                             frac_above_1mm=float((rad[m] > TAIL_UM).mean()),
                             checkpoint=int(n["checkpoint"])))
    res, figs = os.path.join(a.out, "results"), os.path.join(a.out, "figures")
    os.makedirs(res, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    with open(os.path.join(res, "momentum_bands.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nENDPOINT radial error at the first SciFi plane, after the full chain from the last UT plane; "
          "test tracks, median / 95th percentile [um]")
    print("%-42s" % "momentum band" + "".join("%30s" % rd.short_label(n).split(" (")[0] for n in nets))
    for lo, hi in bands + [("all", "all")]:
        cells = []
        lab = "all momenta" if lo == "all" else "%g-%g GeV" % (lo, hi)
        for n in nets:
            r = [x for x in rows if x["N"] == n["N"] and x["q"] == n["q"]
                 and x["p_lo"] == ("all" if lo == "all" else "%g" % lo)
                 and x["p_hi"] == ("all" if hi == "all" else "%g" % hi)]
            cells.append("%14.1f / %-13.0f" % (r[0]["rad_med_um"], r[0]["rad_p95_um"]) if r else "%30s" % "-")
        nn = [x for x in rows if x["p_lo"] == ("all" if lo == "all" else "%g" % lo)
              and x["p_hi"] == ("all" if hi == "all" else "%g" % hi)]
        print("%-28s (n = %4d) %s" % (lab, nn[0]["n"] if nn else 0, "".join(cells)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centres = np.sqrt(P_EDGES[:-1] * P_EDGES[1:])
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.2))
    cmap = plt.get_cmap("tab10")
    for k, n in enumerate(nets):
        rs = [x for x in rows if x["N"] == n["N"] and x["q"] == n["q"]
              and x["p_lo"] not in ("all",) and (float(x["p_lo"]), float(x["p_hi"])) not in EXTRA]
        ax.plot(centres[:len(rs)], [x["rad_med_um"] for x in rs], "-o", ms=4, color=cmap(k),
                label="%s: median" % rd.short_label(n))
        ax.plot(centres[:len(rs)], [x["rad_p95_um"] for x in rs], "--", lw=1, alpha=0.7, color=cmap(k),
                label="%s: 95th percentile" % rd.short_label(n))
    w = nets[0]["window"]
    if w and all(x["window"] == w for x in nets):
        ax.axvspan(w[0], w[1], color="0.92", zorder=0)
        ax.text(np.sqrt(w[0] * w[1]), ax.get_ylim()[0], "loss window", ha="center", va="bottom",
                fontsize=8, color="0.4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("momentum [GeV]")
    ax.set_ylabel("radial endpoint error at z1 [um]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax.set_title("ENDPOINT radial error against momentum, after the full chain from the last UT plane\n%s"
                 % rd.study_text(nets), fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(figs, "momentum_bands.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print("\nwrote %s, %s" % (os.path.join(res, "momentum_bands.csv"), out_png))
    return rows


if __name__ == "__main__":
    main()
