#!/usr/bin/env python
"""F2 - endpoint error(dz, q) at the SciFi plane, one table per state component, 5-30 GeV.

George (2026-09-21): the error(dz, q) table, but for the ENDPOINT errors in x,
y, tx and ty separately - one table each, every cell one network, so nothing
is averaged across architectures - for tracks of 5-30 GeV. And (later the same
day) the networks on their own, named by what they are: no block labels, no
comparison rows.

These are endpoint errors: each network is applied N times from the track's
real state on the last UT plane, and its state on the first SciFi plane
(z1 = 7,826 mm) is compared with the RK6 track carried there. They are NOT
single-step errors (one application from an RK6 state); those are in
`../F3_Analysis/results/error_qdz_single_step.csv`.

Per cell, over the test tracks with 5 <= p < 30 GeV: the median |error|, the
95th percentile, and the signed median. dx, dy in um; dtx, dty in mrad.

A network whose record is a checkpoint of a run still training is marked.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python error_tables_by_component.py [--p-lo 5 --p-hi 30]
Outputs: results/error_by_component_5-30GeV.csv, figures/error_by_component_5-30GeV.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E0 = os.path.join(HERE, "..", "..", "Block_E_single_network_chain", "E0_Track_dataset", "results", "tracks.npz")
RUNS = os.path.join(HERE, "..", "F1_Training", "results", "full")
SPLIT = "test"
L_MM = 5177.8
COMP = (("x", 0, 1e3, "um"), ("y", 1, 1e3, "um"), ("tx", 2, 1e3, "mrad"), ("ty", 3, 1e3, "mrad"))


def networks():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        if not (os.path.exists(os.path.join(d, "record.json")) and os.path.exists(os.path.join(d, "chain_states.npz"))):
            continue
        rec = json.load(open(os.path.join(d, "record.json")))
        prog = json.load(open(os.path.join(d, "progress.json")))
        out.append(dict(N=rec["N"], q=rec["q"], dz_mm=L_MM / rec["N"],
                        end=np.load(os.path.join(d, "chain_states.npz"))["%s_states" % SPLIT][:, -1],
                        checkpoint=prog.get("phase") != "done", restarts=rec["restarts"]))
    return out


def label(n):
    return "N = %d, q = %d (dz = %.0f mm)%s" % (n["N"], n["q"], n["dz_mm"],
                                                "  [checkpoint at restart %d; still training]" % n["restarts"] if n["checkpoint"] else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--p-lo", type=float, default=5.0)
    ap.add_argument("--p-hi", type=float, default=30.0)
    a = ap.parse_args(argv)
    D = np.load(E0)
    n_max = int(D["n_max"])
    truth = np.asarray(D["%s_truth" % SPLIT])[:, n_max]
    P = np.asarray(D["%s_P" % SPLIT])
    m = (P >= a.p_lo) & (P < a.p_hi)
    tag = "%g-%gGeV" % (a.p_lo, a.p_hi)
    nets = networks()
    rows = []
    for n in nets:
        d = n["end"][m, :4] - truth[m, :4]
        for name, i, s, unit in COMP:
            rows.append(dict(N=n["N"], q=n["q"], dz_mm=round(n["dz_mm"], 1), component=name, unit=unit,
                             kind="endpoint (after N steps, at z1)", n_tracks=int(m.sum()),
                             med_abs=float(np.median(np.abs(d[:, i])) * s),
                             p95_abs=float(np.quantile(np.abs(d[:, i]), 0.95) * s),
                             signed_med=float(np.median(d[:, i]) * s),
                             checkpoint=int(n["checkpoint"]), restarts=n["restarts"]))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out_csv = os.path.join(HERE, "results", "error_by_component_%s.csv" % tag)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("ENDPOINT error at the SciFi plane (the network applied N times from the last UT plane), "
          "test tracks with %g <= p < %g GeV (n = %d); every row one network" % (a.p_lo, a.p_hi, m.sum()))
    for name, i, s, unit in COMP:
        fmt = "%8.3f" if unit == "mrad" else "%8.1f"
        print("\n=== %s [%s] ===" % (name, unit))
        print("%-58s %8s %8s %9s" % ("network", "|median|", "p95", "signed med"))
        rr = [x for x in rows if x["component"] == name]
        best = min(rr, key=lambda x: x["med_abs"])
        for x, n in zip(rr, nets):
            print("%-58s " % label(n) + fmt % x["med_abs"] + " " + fmt % x["p95_abs"]
                  + " " + ("%+9.3f" if unit == "mrad" else "%+9.1f") % x["signed_med"] + ("   * best" if x is best else ""))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.2))
    xt = ["N = %d\nq = %d\ndz = %.0f mm%s" % (n["N"], n["q"], n["dz_mm"], "\n(checkpoint)" if n["checkpoint"] else "") for n in nets]
    for j, (name, i, s, unit) in enumerate(COMP):
        rr = [x for x in rows if x["component"] == name]
        v = np.array([[x["med_abs"] for x in rr]])
        a_ = ax[j]
        im = a_.imshow(v, cmap="viridis_r", aspect="auto")
        for c_, x in enumerate(rr):
            a_.text(c_, 0, (("%.3f" if unit == "mrad" else "%.1f") % x["med_abs"]) + "\n(p95 %s)" % (("%.3f" if unit == "mrad" else "%.0f") % x["p95_abs"]),
                    ha="center", va="center", color="w" if (v[0, c_] - v.min()) / max(v.max() - v.min(), 1e-12) > 0.5 else "k", fontsize=9)
        b = int(np.argmin(v[0]))
        a_.add_patch(plt.Rectangle((b - 0.5, -0.5), 1, 1, fill=False, ec="red", lw=2))
        a_.set_xticks(range(len(rr)), xt, fontsize=8)
        a_.set_yticks([])
        a_.set_title("median |%s error| at z1 [%s]" % (name, unit), fontsize=10)
        fig.colorbar(im, ax=a_, pad=0.02).set_label("[%s]" % unit, fontsize=8)
    fig.suptitle("Endpoint error at the SciFi plane per state component, after the full chain from the last UT plane; "
                 "test tracks with %g-%g GeV (n = %d); every cell one network, red box = best" % (a.p_lo, a.p_hi, m.sum()), fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out_png = os.path.join(HERE, "figures", "error_by_component_%s.png" % tag)
    fig.savefig(out_png, dpi=130)
    print("\nwrote %s, %s" % (os.path.relpath(out_csv, HERE), os.path.relpath(out_png, HERE)))


if __name__ == "__main__":
    main()
