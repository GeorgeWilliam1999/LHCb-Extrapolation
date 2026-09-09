#!/usr/bin/env python
"""C1.2 - the convergence figure, drawn from the csv tables only.

Four panels:

  1. the step ladder. |S(h) - S(h/2)| against h on the real map and on the
     smooth control, with a 2^6 guide line. Where the real-map series leaves
     that line is where the trilinear grid, not the scheme, is setting the
     error.
  2. the same, split by momentum band - the bend goes as 1/p, so the soft
     tracks are where the reference is worst.
  3. forward-then-back closure at the chosen step, per leg against momentum.
  4. RK4 at 5 mm and 1 mm against the fine reference: how far the ground moves.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python plot.py

Output: figures/reference_convergence.png
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("measurement", "integrator", "field", "group_kind",
                     "group", "p_band"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def main():
    os.makedirs(FIGURES, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = read(os.path.join(RESULTS, "reference_convergence.csv"))
    L = read(os.path.join(RESULTS, "reference_convergence_per_leg.csv"))
    meta = json.load(open(os.path.join(RESULTS,
                                       "reference_convergence_meta.json")))
    main_field = "v8r1." + meta["field"]["which"]

    def sel(**kw):
        out = S
        for k, v in kw.items():
            out = [r for r in out if r[k] == v]
        return out

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # ---- 1. the ladder ------------------------------------------------------
    ax = axes[0, 0]
    real = sorted(sel(measurement="step halving", field=main_field,
                      group="all"), key=lambda r: -r["step_mm"])
    smooth = sorted(sel(measurement="step halving", field="smooth analytic",
                        group="all"), key=lambda r: -r["step_mm"])
    hs = [r["step_mm"] for r in real]
    es = [r["pos_med_um"] for r in real]
    ax.plot(hs, es, "o-", lw=2, ms=8, color="#c44e52",
            label="%s  (median over %d legs)" % (main_field, int(real[0]["n"])))
    ax.fill_between(hs, [r["pos_med_um"] for r in real],
                    [r["pos_p95_um"] for r in real], color="#c44e52",
                    alpha=0.15, lw=0, label="median to p95")
    if smooth:
        ax.plot([r["step_mm"] for r in smooth],
                [r["pos_med_um"] for r in smooth], "s-", lw=2, ms=7,
                color="#4c72b0", label="smooth analytic field (control)")
    guide = np.array(hs, dtype=float)
    ax.plot(guide, es[0] * (guide / guide[0]) ** 6, "k--", lw=1.2,
            label=r"order 6:  each halving $\div\,64$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("step h  [mm]   (each point is |S(h) - S(h/2)|)")
    ax.set_ylabel("endpoint move  [um]")
    ax.set_title("The step ladder: where the answer stops moving")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # ---- 2. per momentum band ----------------------------------------------
    ax = axes[0, 1]
    bands = [r["group"] for r in sel(measurement="step halving",
                                     field=main_field,
                                     group_kind="p_band",
                                     step_mm=max(hs))]
    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(bands):
        rows = sorted(sel(measurement="step halving", field=main_field,
                          group=b), key=lambda r: -r["step_mm"])
        ax.plot([r["step_mm"] for r in rows], [r["pos_med_um"] for r in rows],
                "o-", lw=1.8, ms=6, color=cmap(i / max(1, len(bands) - 1)),
                label=b)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("step h  [mm]")
    ax.set_ylabel("endpoint move  [um]")
    ax.set_title("The same, by momentum band  (the bend goes as 1/p)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # ---- 3. closure ---------------------------------------------------------
    ax = axes[1, 0]
    cl = [r for r in L if r["measurement"] == "forward-then-back closure"
          and r["field"] == main_field]
    p = np.array([r["P_GeV"] for r in cl])
    g = np.array([max(r["pos_um"], 1e-9) for r in cl])
    d = np.array([r["direction"] for r in cl])
    for dd, colour, nm in ((1, "#4c72b0", "UT -> SciFi"),
                           (-1, "#c44e52", "SciFi -> UT")):
        m = d == dd
        ax.plot(p[m], g[m], "o", ms=4, alpha=0.65, color=colour, label=nm)
    ax.axhline(float(np.median(g)), color="k", ls="--", lw=1.2,
               label="median %.3g um" % np.median(g))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("truth momentum p  [GeV]")
    ax.set_ylabel("|start - (forward then back)|  [um]")
    ax.set_title("Forward-then-back closure at %g mm" % cl[0]["step_mm"])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # ---- 4. RK4 -------------------------------------------------------------
    ax = axes[1, 1]
    rk4 = [r for r in S if r["integrator"] == "RK4" and r["group_kind"] == "p_band"]
    steps = sorted({r["step_mm"] for r in rk4})
    bandnames = sorted({r["group"] for r in rk4},
                       key=lambda s: float(s.split("-")[0]))
    xs = np.arange(len(bandnames))
    w = 0.8 / max(1, len(steps))
    for i, h in enumerate(steps):
        vals = [next((r["pos_med_um"] for r in rk4
                      if r["step_mm"] == h and r["group"] == b), np.nan)
                for b in bandnames]
        ax.bar(xs + (i - (len(steps) - 1) / 2) * w, vals, w,
               label="RK4 at %g mm" % h)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(bandnames, fontsize=8, rotation=15)
    ax.set_ylabel("|RK4 - RK6 fine|  median  [um]")
    ax.set_title("The incumbent 5 mm RK4 engine against the fine reference")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=8)

    fig.suptitle("The fine RK6 reference on %d cross-magnet legs (%s), "
                 "momentum-stratified, both directions"
                 % (meta["legs"]["n"], main_field), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES, "reference_convergence.png"), dpi=140)
    plt.close(fig)
    print("wrote figures/reference_convergence.png")


if __name__ == "__main__":
    main()
