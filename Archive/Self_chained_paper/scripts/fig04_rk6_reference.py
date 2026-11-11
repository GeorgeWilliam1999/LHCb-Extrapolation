#!/usr/bin/env python
"""Figure 4 - how good the reference integrator is, and how it was shown to be.

The labels this paper scores against are produced by Butcher's seven-stage
explicit Runge-Kutta method of order six at a 0.1 mm step.  The three panels
are the three separate measurements that fix its accuracy.

WHAT THE FIGURE SHOWS

  (a) the SCHEME's own order.  The LHCb equation of motion is integrated with
      the trilinear field map replaced by a smooth analytic stand-in, so the
      method is not asked to integrate across a kink.  Endpoint error against
      the same routine at a 0.5 mm step, at steps of 640, 320, 160 and 80 mm,
      with the fitted slope printed and an ideal order-6 guide drawn through
      the coarsest point.  A smooth field is the only way to measure the order
      through this code path.
  (b) the step ladder on the REAL v8r1 map: the median, the 95th percentile and
      the worst of |S(h) - S(h/2)| over 200 momentum-stratified cross-magnet
      legs, at h = 0.8, 0.4, 0.2 and 0.1 mm.  An order-6 guide is drawn through
      the coarsest median for reference and the three successive halving ratios
      are printed beside the median line.  They are nothing like 64, because
      the map is trilinear on a 100 mm cell and so only C0 in its derivative.
  (c) the levels that matter, all on the same logarithmic axis: what one more
      halving moves the endpoint at the chosen 0.1 mm step, what the
      forward-then-back round trip fails to close by at that step, and how far
      the incumbent fp64 RK4 engine at 5 mm and at 1 mm sits from the fine
      reference.  Each entry is drawn as median, p95 and worst.

INPUT FILES (all opened read-only; every number is read from the files, none
is copied from the folder's README)

  ../../multi_network_chain_discrete_approach/Block_C_step_size_and_stages/
    C1_Fine_reference/results/reference_convergence.csv
      rows keyed by (measurement, step_mm, reference_step_mm, integrator,
      field, group_kind, group); this figure uses group = "all" and
      field = "v8r1.up" for the ladder, the closure and the RK4 comparison.
      pos_med_um / pos_p95_um / pos_max_um are the endpoint position
      differences, worst of x and y, over the 200 legs.
    C1_Fine_reference/results/reference_convergence_meta.json
      the polarity, the leg count, the step ladder and the successive ratios.
    C1_Fine_reference/results/tableau_checks.json
      checks.4_order_of_rk6_rows_on_a_smooth_field: step_sizes_mm,
      max_abs_error_mm (in mm) and fitted_order, for panel (a).

OUTPUT

  ../figures/fig_rk6_reference.png
  ../results/fig04_rk6_reference.csv   every plotted value, long format

RUN

  cd Self_chained_paper/scripts
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig04_rk6_reference.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys                                            # noqa: E402
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("", ".", HERE):
    while _p in sys.path:
        sys.path.remove(_p)

import importlib.util                                 # noqa: E402

import numpy as np                                    # noqa: E402
import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

FIGURES = os.path.join(C.PAPER, "figures")
PNG = os.path.join(FIGURES, "fig_rk6_reference.png")
CSV = os.path.join(C.RESULTS, "fig04_rk6_reference.csv")

CONV = os.path.join(C.C1_RESULTS, "reference_convergence.csv")
CONV_META = os.path.join(C.C1_RESULTS, "reference_convergence_meta.json")
TABLEAU = os.path.join(C.C1_RESULTS, "tableau_checks.json")


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "figure.facecolor": C.SURFACE, "axes.facecolor": C.SURFACE,
        "savefig.facecolor": C.SURFACE, "savefig.edgecolor": C.SURFACE,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "axes.edgecolor": C.TEXT2, "axes.labelcolor": C.TEXT1,
        "text.color": C.TEXT1, "xtick.color": C.TEXT2, "ytick.color": C.TEXT2,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "grid.color": C.NEUTRAL, "grid.alpha": 0.6, "grid.linewidth": 0.6,
    })


def fit_order(h, e):
    """The slope of log(error) against log(step): the observed order."""
    h = np.asarray(h, float)
    e = np.asarray(e, float)
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


def main():
    style()
    rows_csv = C.read_csv(CONV)
    meta = C.read_json(CONV_META)
    tab = C.read_json(TABLEAU)
    field = "v8r1." + meta["field"]["which"]
    nlegs = int(meta["legs"]["n"])
    out = []

    def sel(**kw):
        got = rows_csv
        for k, v in kw.items():
            got = [r for r in got if r[k] == v]
        return got

    fig = plt.figure(figsize=(6.3, 5.9))
    gs_top = fig.add_gridspec(1, 2, left=0.135, right=0.985, top=0.955,
                              bottom=0.545, wspace=0.28)
    gs_bot = fig.add_gridspec(1, 1, left=0.245, right=0.985, top=0.435,
                              bottom=0.095)
    axa = fig.add_subplot(gs_top[0, 0])
    axb = fig.add_subplot(gs_top[0, 1])
    axc = fig.add_subplot(gs_bot[0, 0])

    # --------------------------------- (a) the order on a smooth field ------
    sm = tab["checks"]["4_order_of_rk6_rows_on_a_smooth_field"]
    hs = np.asarray(sm["step_sizes_mm"], float)
    es = np.asarray(sm["max_abs_error_mm"], float) * 1e3          # mm -> um
    axa.plot(hs, es, "o-", lw=1.6, ms=6, color=C.BLUE,
             label="max over %d legs" % int(sm["n_rows"]))
    guide = es[0] * (hs / hs[0]) ** 6.0
    axa.plot(hs, guide, "--", lw=1.2, color=C.TEXT2,
             label=r"order 6 ($\div$64 a halving)")
    axa.set_xscale("log", base=2)
    axa.set_yscale("log")
    axa.set_xticks(hs)
    axa.set_xticklabels(["%d" % h for h in hs])
    axa.invert_xaxis()
    axa.set_xlabel("step h  (mm)")
    axa.set_ylabel(r"endpoint error  ($\mu$m)")
    axa.grid(True, which="both")
    axa.set_title("(a)  smooth analytic field", loc="left", pad=4)
    axa.legend(loc="upper right", handlelength=1.8, labelspacing=0.25)
    # the stored fit uses only the points above the fp64 floor (3 of the 4:
    # on a smooth field the scheme is already at the rounding floor by 80 mm)
    k = int(sm["points_above_floor"])
    refit = fit_order(hs[:k], es[:k])
    assert abs(refit - sm["fitted_order"]) < 1e-9
    axa.annotate("fitted order %.2f\n(first %d points; the rest\nis at the fp64 floor)"
                 % (refit, k), xy=(0.03, 0.03),
                 xycoords="axes fraction", ha="left", va="bottom", fontsize=8.5,
                 color=C.TEXT1)
    for h, e in zip(hs, es):
        out.append(dict(panel="a", series="smooth analytic field, max error",
                        x=h, x_unit="mm", value=e, unit="um"))
    out.append(dict(panel="a", series="fitted order (first %d points)" % k,
                    x="", x_unit="", value=refit, unit="dimensionless"))

    # --------------------------------- (b) the step ladder on the real map --
    lad = sorted(sel(measurement="step halving", field=field, group="all"),
                 key=lambda r: -float(r["step_mm"]))
    h2 = np.array([float(r["step_mm"]) for r in lad])
    med = np.array([float(r["pos_med_um"]) for r in lad])
    p95 = np.array([float(r["pos_p95_um"]) for r in lad])
    mx = np.array([float(r["pos_max_um"]) for r in lad])
    axb.plot(h2, mx, "^-", lw=1.3, ms=5, color=C.NEUTRAL, label="worst")
    axb.plot(h2, p95, "s-", lw=1.3, ms=5, color=C.GREEN, label="p95")
    axb.plot(h2, med, "o-", lw=1.8, ms=6, color=C.BLUE, label="median")
    axb.plot(h2, med[0] * (h2 / h2[0]) ** 6.0, "--", lw=1.2, color=C.TEXT2,
             label="order 6 guide")
    ratios = med[:-1] / med[1:]
    assert np.allclose(np.round(ratios, 3), meta["successive_halving_ratios"],
                       rtol=0, atol=1e-3)
    for i, r in enumerate(ratios):
        xm = np.sqrt(h2[i] * h2[i + 1])
        ym = np.sqrt(med[i] * med[i + 1])
        axb.annotate(r"$\div$%.1f" % r, xy=(xm, ym), xytext=(4, 4),
                     textcoords="offset points", ha="left", va="bottom",
                     fontsize=8.5, color=C.TEXT1)
    axb.set_xscale("log", base=2)
    axb.set_yscale("log")
    axb.set_xticks(h2)
    axb.set_xticklabels(["%g" % h for h in h2])
    axb.invert_xaxis()
    axb.set_xlabel("step h  (mm)")
    axb.set_ylim(0.35 * med.min(), 6.0 * mx.max())
    axb.grid(True, which="both")
    axb.set_title("(b)  real v8r1 map, %d legs" % nlegs, loc="left", pad=4)
    leg = axb.legend(loc="upper right", handlelength=1.6, labelspacing=0.2,
                     title="|S(h) - S(h/2)|")
    plt.setp(leg.get_title(), fontsize=9, color=C.TEXT2)
    for h, a, b, c_ in zip(h2, med, p95, mx):
        out.append(dict(panel="b", series="%s |S(h)-S(h/2)| median" % field,
                        x=h, x_unit="mm", value=a, unit="um"))
        out.append(dict(panel="b", series="%s |S(h)-S(h/2)| p95" % field,
                        x=h, x_unit="mm", value=b, unit="um"))
        out.append(dict(panel="b", series="%s |S(h)-S(h/2)| worst" % field,
                        x=h, x_unit="mm", value=c_, unit="um"))
    for i, r in enumerate(ratios):
        out.append(dict(panel="b", series="successive halving ratio",
                        x="%g->%g" % (h2[i], h2[i + 1]), x_unit="mm",
                        value=float(r), unit="dimensionless"))

    # ------------------------------ (c) the levels: closure and the RK4 engine --
    def one(**kw):
        r = sel(group="all", field=field, **kw)
        assert len(r) == 1, (kw, len(r))
        r = r[0]
        return (float(r["pos_med_um"]), float(r["pos_p95_um"]),
                float(r["pos_max_um"]), int(float(r["n"])))

    entries = [
        ("RK6, 0.1 vs 0.05 mm",
         dict(measurement="step halving", step_mm="0.1"), C.BLUE),
        ("RK6 closure, 0.1 mm",
         dict(measurement="forward-then-back closure", step_mm="0.1"), C.GREEN),
        ("RK4 at 1 mm vs RK6",
         dict(measurement="RK4 against RK6 at 0.05 mm", step_mm="1.0"),
         C.MAGENTA),
        ("RK4 at 5 mm vs RK6",
         dict(measurement="RK4 against RK6 at 0.05 mm", step_mm="5.0"),
         C.MAGENTA),
    ]
    ys = np.arange(len(entries))
    for y, (label, kw, colour) in zip(ys, entries):
        m, p, w, n = one(**kw)
        axc.plot([m, w], [y, y], "-", lw=2.2, color=colour, alpha=0.35,
                 solid_capstyle="butt")
        axc.plot([m], [y], "o", ms=7, color=colour, zorder=3)
        axc.plot([p], [y], "s", ms=5, color=colour, zorder=3)
        axc.plot([w], [y], "|", ms=10, mew=1.6, color=colour, zorder=3)
        axc.annotate("%.3g / %.3g / %.3g" % (m, p, w), xy=(m, y),
                     xytext=(0, 7), textcoords="offset points", ha="left",
                     va="bottom", fontsize=8.5, color=C.TEXT2)
        out.append(dict(panel="c", series=label.replace("\n", " "), x="median",
                        x_unit="", value=m, unit="um"))
        out.append(dict(panel="c", series=label.replace("\n", " "), x="p95",
                        x_unit="", value=p, unit="um"))
        out.append(dict(panel="c", series=label.replace("\n", " "), x="worst",
                        x_unit="", value=w, unit="um"))
    axc.set_yticks(ys)
    axc.set_yticklabels([e[0] for e in entries], fontsize=9)
    axc.set_ylim(-0.6, len(entries) - 0.15)
    axc.set_xscale("log")
    axc.set_xlim(8e-7, 4.0)
    axc.set_xlabel(r"endpoint difference, worst of x and y  ($\mu$m)")
    axc.grid(True, axis="x", which="both")
    axc.tick_params(axis="y", length=0)
    axc.set_title("(c)  the levels, as median / p95 / worst over %d legs"
                  % nlegs, loc="left", pad=4)

    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(PNG, dpi=150, facecolor=C.SURFACE)
    plt.close(fig)

    C.write_csv(CSV, out, ["panel", "series", "x", "x_unit", "value", "unit"])
    print("wrote %s" % PNG)
    print("wrote %s  (%d rows)" % (CSV, len(out)))
    print("(a) fitted order on the smooth field: %.3f over the first %d points "
          "(stored %.3f); all four points give %.3f"
          % (refit, k, sm["fitted_order"], fit_order(hs, es)))
    print("(b) successive halving ratios recomputed from the CSV: %s  "
          "(stored %s); ideal for order 6 is 64"
          % (", ".join("%.2f" % r for r in ratios),
             meta["successive_halving_ratios"]))


if __name__ == "__main__":
    main()
