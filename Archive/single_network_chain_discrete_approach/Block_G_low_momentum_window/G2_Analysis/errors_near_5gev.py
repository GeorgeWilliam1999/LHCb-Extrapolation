#!/usr/bin/env python
"""G2 - the endpoint deviation around 5 GeV, signed, per component, per network.

The primary measure of the study. For each network, the ENDPOINT error is what
is left after the network has been applied N times from the track's real state
on the last UT plane, compared with the RK6 track carried to the first SciFi
plane (z1 = 7,826 mm). It is NOT a single-step error (one application from an
RK6 state); those are in G3_Analysis/results/error_qdz_single_step.csv.

Per network and per component (x, y, tx, ty), in the momentum bands 3-5, 4-6,
5-7, 5-8 and 10-20 GeV, on the test tracks:

  n, median |d|, RMS (about zero), standard deviation, the signed mean and
  median, the 68 percent half-width (half the 16th-84th percentile range, a
  tail-robust width) and the 95th percentile of |d|.

RMS and standard deviation are pulled by the tails; the median and the 68
percent half-width are not, and the two are printed side by side.

Beyond that (George, 2026-09-21): a few percent of the tracks around 5 GeV
land more than 1 mm from the RK6 endpoint and carry most of the RMS. Those
tracks start at the edge of the acceptance, and a momentum window in the loss
is not expected to touch them, so this script also reports

  the RMS with those > 1 mm-radial tracks removed, beside the full RMS;
  what fraction of the band they are; and the median |x0| (the distance of the
  track's starting point from the beam line, on the last UT plane) of that tail
  against the median |x0| of the rest of the band.

The shared statistics come from the Block F script by import, so the columns
that exist in both are the same arithmetic, not a retyping of it.

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python errors_near_5gev.py [--runs DIR] [--out DIR]
Outputs: <out>/results/errors_near_5gev.csv, <out>/figures/signed_errors_4-6GeV.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys           # noqa: E402
# importing a script out of a read-only folder must not leave a .pyc behind in it
sys.dont_write_bytecode = True

import argparse          # noqa: E402
import csv               # noqa: E402
import importlib.util    # noqa: E402

import numpy as np       # noqa: E402

import run_discovery as rd   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E0 = os.path.join(HERE, "..", "..", "Block_E_single_network_chain", "E0_Track_dataset",
                  "results", "tracks.npz")
F2 = os.path.abspath(os.path.join(HERE, "..", "..", "Block_F_reweighted_loss", "F2_Analysis"))
DEFAULT_RUNS = os.path.join(HERE, "..", "G1_Training", "results", "p03-08", "full")
BANDS = ((3, 5), (4, 6), (5, 7), (5, 8), (10, 20))
FIG_BAND = (4, 6)
TAIL_UM = 1000.0                 # "the tail": radial endpoint error beyond 1 mm
SPLIT = "test"


def _f2(name):
    """Import a Block F analysis module by path, without modifying it."""
    spec = importlib.util.spec_from_file_location("f2_" + name, os.path.join(F2, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SRC = _f2("errors_near_5gev")
stats = _SRC.stats                      # n, med_abs, rms, std, mean, signed_med, hw68, p95_abs
COMP = _SRC.COMP                        # (name, index, scale, unit) for x, y, tx, ty


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="folder holding the N<NNN>_q<qq> run folders")
    ap.add_argument("--out", default=HERE, help="where results/ and figures/ are written")
    ap.add_argument("--only", default=None, help="comma-separated run folders, e.g. N064_q02")
    a = ap.parse_args(argv)
    nets = rd.require(rd.find_runs(a.runs, a.only), a.runs)
    print(rd.describe(nets, a.runs))
    study = rd.study_text(nets)

    D = np.load(E0)
    n_max = int(D["n_max"])
    truth = np.asarray(D["%s_truth" % SPLIT])[:, n_max]
    P = np.asarray(D["%s_P" % SPLIT])
    x0 = np.abs(np.asarray(D["%s_S0" % SPLIT])[:, 0])          # |x| on the last UT plane, mm
    for n in nets:
        n["end"] = np.load(os.path.join(n["path"], "chain_states.npz"))["%s_states" % SPLIT][:, -1]

    rows, tails = [], []
    for n in nets:
        d = n["end"][:, :4] - truth[:, :4]
        rad = np.hypot(d[:, 0], d[:, 1]) * 1e3                  # um
        for lo, hi in BANDS:
            m = (P >= lo) & (P < hi)
            tail = m & (rad > TAIL_UM)
            rest = m & (rad <= TAIL_UM)
            t = dict(n_tail=int(tail.sum()),
                     tail_frac=float(tail.sum() / max(m.sum(), 1)),
                     x0_med_tail_mm=float(np.median(x0[tail])) if tail.any() else float("nan"),
                     x0_med_rest_mm=float(np.median(x0[rest])) if rest.any() else float("nan"))
            tails.append(dict(N=n["N"], q=n["q"], dz_mm=round(n["dz_mm"], 1), p_lo=lo, p_hi=hi,
                              kind="endpoint (after N steps, at z1)", n=int(m.sum()), **t,
                              rad_med_um=float(np.median(rad[m])),
                              rad_rms_um=float(np.sqrt(np.mean(rad[m] ** 2))),
                              rad_rms_no_tail_um=float(np.sqrt(np.mean(rad[rest] ** 2))) if rest.any() else float("nan"),
                              checkpoint=int(n["checkpoint"])))
            for name, i, s, unit in COMP:
                st = stats(d[m, i] * s)
                rows.append(dict(N=n["N"], q=n["q"], checkpoint=int(n["checkpoint"]),
                                 p_lo=lo, p_hi=hi, component=name, unit=unit, **st,
                                 rms_no_tail=float(np.sqrt(np.mean((d[rest, i] * s) ** 2))) if rest.any() else float("nan"),
                                 dz_mm=round(n["dz_mm"], 1),
                                 kind="endpoint (after N steps, at z1)", **t))

    res = os.path.join(a.out, "results")
    figs = os.path.join(a.out, "figures")
    os.makedirs(res, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    for fn, data in (("errors_near_5gev.csv", rows), ("tail_near_5gev.csv", tails)):
        with open(os.path.join(res, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    # -- the printed tables ---------------------------------------------------------
    for lo, hi in BANDS:
        print("\n=== %d-%d GeV: ENDPOINT deviation at the first SciFi plane, after the full chain "
              "from the last UT plane; test tracks ===" % (lo, hi))
        print("%-46s %-4s %5s %9s %9s %9s %9s %9s %9s %11s" % (
            "network", "", "n", "|median|", "RMS", "std", "mean", "sgn med", "68% hw", "RMS no tail"))
        for n in nets:
            for name, i, s, unit in COMP:
                r = [x for x in rows if x["N"] == n["N"] and x["q"] == n["q"]
                     and x["p_lo"] == lo and x["p_hi"] == hi and x["component"] == name][0]
                fmt = "%9.3f" if unit == "mrad" else "%9.1f"
                print("%-46s %-4s %5d " % (rd.short_label(n), name, r["n"])
                      + " ".join(fmt % r[k] for k in ("med_abs", "rms", "std", "mean", "signed_med", "hw68"))
                      + (" %11.3f" if unit == "mrad" else " %11.1f") % r["rms_no_tail"] + "  " + unit)
        print("  the tail (radial endpoint error beyond %.0f mm), which the RMS-no-tail column drops:"
              % (TAIL_UM / 1000))
        for n in nets:
            t = [x for x in tails if x["N"] == n["N"] and x["q"] == n["q"]
                 and x["p_lo"] == lo and x["p_hi"] == hi][0]
            print("    %-46s %3d of %4d tracks (%4.1f%%), median |x0| %6.1f mm against %6.1f mm "
                  "for the rest; radial median %7.1f um" % (
                      rd.short_label(n), t["n_tail"], t["n"], 100 * t["tail_frac"],
                      t["x0_med_tail_mm"], t["x0_med_rest_mm"], t["rad_med_um"]))

    # -- the signed distributions at 4-6 GeV ----------------------------------------
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
                       label="%s: median %s, RMS %s, 68%% hw %s" % (
                           rd.short_label(n),
                           ("%+.3f" if unit == "mrad" else "%+.1f") % st["signed_med"],
                           ("%.3f" if unit == "mrad" else "%.0f") % st["rms"],
                           ("%.3f" if unit == "mrad" else "%.0f") % st["hw68"]))
        ax[j].axvline(0, color="k", lw=0.8)
        ax[j].set_xlabel("signed %s error at z1 [%s]  (clipped at the 98th percentile)" % (name, unit), fontsize=9)
        ax[j].set_ylabel("test tracks")
        ax[j].set_title("%s, %d-%d GeV (%d tracks)" % (name, lo, hi, m.sum()), fontsize=10)
        ax[j].legend(fontsize=7)
        ax[j].grid(alpha=0.3)
    fig.suptitle("Signed ENDPOINT deviation around 5 GeV, per network, after the full chain from the last UT "
                 "plane to the first SciFi plane (%d-%d GeV test tracks, against RK6)\n%s"
                 % (lo, hi, study), fontsize=12)
    fig.tight_layout()
    out_png = os.path.join(figs, "signed_errors_%d-%dGeV.png" % (lo, hi))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print("\nwrote %s, %s, %s" % (os.path.join(res, "errors_near_5gev.csv"),
                                  os.path.join(res, "tail_near_5gev.csv"), out_png))
    return rows


if __name__ == "__main__":
    main()
