#!/usr/bin/env python
"""F2 - Block F against Block E, per component and per momentum band.

For each finished Block F run and its Block E counterpart (the extended run, as
it stands on disk), from the stored chain states on the 1,452 test tracks:

  per component   median |dx|, |dy| [um], |dtx|, |dty| [mrad] at z1, overall
                  and in the 10-50 GeV band, with the signed median beside it;
  per momentum    the radial error's median and 95th percentile in the bands
                  1-2, 2-5, 5-10, 10-20, 20-50, 50-100, 100-200 GeV.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python components_and_momentum.py
Outputs: results/components.csv, results/momentum_bands.csv, figures/blockF_vs_blockE.png
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(HERE, "..", "..", "Block_E_single_network_chain")
F = os.path.join(HERE, "..", "F1_Training", "results", "full")
TRACKS = os.path.join(E, "E0_Track_dataset", "results", "tracks.npz")
SPLIT = "test"
COMP = (("x", 0, 1e3, "um"), ("y", 1, 1e3, "um"), ("tx", 2, 1e3, "mrad"), ("ty", 3, 1e3, "mrad"))
P_EDGES = np.array([1, 2, 5, 10, 20, 50, 100, 200.0])
BAND = (10.0, 50.0)


def load(run):
    st = os.path.join(run, "chain_states.npz")
    if not os.path.exists(st):
        return None
    return np.load(st)["%s_states" % SPLIT][:, -1]


def main():
    D = np.load(TRACKS)
    n_max = int(D["n_max"])
    truth = np.asarray(D["%s_truth" % SPLIT])[:, n_max]
    P = np.asarray(D["%s_P" % SPLIT])
    band = (P >= BAND[0]) & (P < BAND[1])
    pairs = []
    for f in sorted(glob.glob(os.path.join(F, "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        tag = os.path.basename(f)
        e = os.path.join(E, "E1_Network_grid", "results", tag)
        ef, ee = load(f), load(e)
        if ef is None:
            continue
        pairs.append((tag, ("Block F", ef), ("Block E", ee)))
    crows, mrows = [], []
    for tag, *runs in pairs:
        for label, end in runs:
            d = end[:, :4] - truth[:, :4]
            rad = np.hypot(d[:, 0], d[:, 1]) * 1e3
            for name, i, s, unit in COMP:
                crows.append(dict(run=tag, block=label, component=name, unit=unit,
                                  med_abs=float(np.median(np.abs(d[:, i])) * s),
                                  signed_med=float(np.median(d[:, i]) * s),
                                  band_med_abs=float(np.median(np.abs(d[band, i])) * s),
                                  band_signed_med=float(np.median(d[band, i]) * s)))
            for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
                m = (P >= lo) & (P < hi)
                mrows.append(dict(run=tag, block=label, p_lo=lo, p_hi=hi, n=int(m.sum()),
                                  rad_med=float(np.median(rad[m])),
                                  rad_p95=float(np.quantile(rad[m], 0.95)),
                                  x_med=float(np.median(np.abs(d[m, 0])) * 1e3),
                                  y_med=float(np.median(np.abs(d[m, 1])) * 1e3)))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for fn, rows in (("components.csv", crows), ("momentum_bands.csv", mrows)):
        with open(os.path.join(HERE, "results", fn), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # -- print -------------------------------------------------------------------
    for tag, *runs in pairs:
        print("=== %s ===" % tag)
        print("  %-8s %-4s %10s %10s   %10s %10s" % ("", "", "all |med|", "signed", "band |med|", "signed"))
        for name, i, s, unit in COMP:
            for label, _ in runs:
                r = [x for x in crows if x["run"] == tag and x["block"] == label and x["component"] == name][0]
                print("  %-8s %-4s %10.1f %+10.1f   %10.1f %+10.1f  %s" % (
                    label, name, r["med_abs"], r["signed_med"], r["band_med_abs"], r["band_signed_med"], unit))
        print("  radial median / p95 by momentum band:")
        for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
            cells = []
            for label, _ in runs:
                r = [x for x in mrows if x["run"] == tag and x["block"] == label and x["p_lo"] == lo][0]
                cells.append("%s %6.0f / %6.0f" % (label[-1], r["rad_med"], r["rad_p95"]))
            print("    %3g-%3g GeV (n=%4d):  %s" % (lo, hi, r["n"], "   ".join(cells)))

    # -- figure ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centres = np.sqrt(P_EDGES[:-1] * P_EDGES[1:])
    fig, axes = plt.subplots(1, len(pairs), figsize=(7.2 * len(pairs), 4.8), squeeze=False)
    for ax, (tag, *runs) in zip(axes[0], pairs):
        for (label, _), col in zip(runs, ("#d62728", "#1f77b4")):
            rs = [x for x in mrows if x["run"] == tag and x["block"] == label]
            ax.plot(centres, [r["rad_med"] for r in rs], "-o", color=col, label="%s median" % label)
            ax.plot(centres, [r["rad_p95"] for r in rs], "--", color=col, alpha=0.7, label="%s 95th pct" % label)
        ax.axvspan(BAND[0], BAND[1], color="0.92", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("momentum [GeV]")
        ax.set_ylabel("radial error at z1 [um]")
        ax.set_title("%s: radial test error against momentum (shaded: 10-50 GeV)" % tag.replace("_", ", "),
                     fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Block F (reweighted loss) against Block E (extended), 1,452 test tracks", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "blockF_vs_blockE.png"), dpi=130)
    print("wrote results/components.csv, results/momentum_bands.csv, figures/blockF_vs_blockE.png")


if __name__ == "__main__":
    main()
