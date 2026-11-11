#!/usr/bin/env python
"""Figure 8 - where the loss actually goes (figures/fig_loss_shares.png).

The pre-flight measurement: 8,000 (track, plane) states drawn evenly over the
64 start planes of the extended pooled-loss N = 64, q = 2, dz = 80.9 mm
network, and the share of the loss each momentum band and each quarter of the
crossing carries under the pooled loss and under the cost-weighted loss.

WHAT THE FIGURE SHOWS, panel by panel.

(a) The share of the loss carried by each momentum band, pooled loss against
    cost-weighted loss, as grouped bars in per cent.  The 10-50 GeV loss window
    overlaps two of the five bands, so it is drawn as a separate hatched pair to
    the right of the partition rather than as a sixth band.  The track count n
    of each band is printed under its label.  Behind each pair, three thin grey
    bars give the same share with one factor of the weight switched off - the
    lever arm, the per-track factor, the momentum window - which are diagnostic
    ablations only and were never trained.

(b) The same, by quarter of the crossing in z.

(c) The clamp scan: the share the 10-50 GeV window carries and the share the
    > 50 GeV band carries, as the clamp on the per-track factor is tightened
    from off to a factor of 2.  The clamp of 5 that the runs used is marked.

(d) How well each loss ranks the states by what their error actually costs at
    the endpoint: the rank (Spearman) correlation of a state's share of the
    loss with its true endpoint cost, overall and within the 10-50 GeV window,
    and the share of the whole loss that the worst 1 % of states carries.  The
    cost is a diagnostic computed against the reference and never enters either
    loss.

INPUTS READ (all read-only).

  results/paper_numbers.json    group `preflight`: `modes` (the five weighting
                                modes' by_band and by_quarter_of_z shares, the
                                two rank correlations and the top-1 % share),
                                `clamp_scan`, and
                                `reproduced_with_the_18_Sept_snapshot`, whose
                                values go into the companion CSV as extra
                                columns.
  scripts/common.py             paths, palette, band labels, CSV writer.

The figure uses the EXTENDED network's numbers throughout, per the paper's
convention 5; the 18 September snapshot values are carried in the CSV only.

OUTPUTS.

  figures/fig_loss_shares.png
  results/fig08_loss_shares.csv

Run:
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig08_loss_shares.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys                                   # noqa: E402
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("", ".", HERE):                   # numbers.py's stdlib-shadowing guard
    while _p in sys.path:
        sys.path.remove(_p)

import importlib.util                        # noqa: E402

import numpy as np                           # noqa: E402
import matplotlib                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.lines import Line2D          # noqa: E402
from matplotlib.patches import Patch         # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["sc_paper_common"] = C
_spec.loader.exec_module(C)

FIGURES = os.path.join(C.PAPER, "figures")
PNG = os.path.join(FIGURES, "fig_loss_shares.png")
CSV = os.path.join(C.RESULTS, "fig08_loss_shares.csv")

P = C.PALETTE
FAINT = "#e5e4df"
GREY = "#9a9a94"
BAND_FILL = "#dfe7f2"

plt.rcParams.update({
    "figure.facecolor": P["SURFACE"], "axes.facecolor": P["SURFACE"],
    "savefig.facecolor": P["SURFACE"], "axes.edgecolor": P["NEUTRAL"],
    "axes.labelcolor": P["TEXT2"], "text.color": P["TEXT1"],
    "xtick.color": P["TEXT2"], "ytick.color": P["TEXT2"], "font.size": 9,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": FAINT, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 150,
})

# the paper's names for the five weighting modes; no folder or block words
NAME = {"blockE": "pooled loss",
        "full": "cost-weighted loss",
        "no_lever": "lever arm off",
        "no_track": "track factor off",
        "no_window": "window off"}
ABLATIONS = ("no_lever", "no_track", "no_window")
HATCH = {"no_lever": "///", "no_track": "\\\\\\", "no_window": "..."}

ROWS = []


def rec(**kw):
    row = dict(panel="", quantity="", group="", series="", n="", value="",
               snapshot_18Sept_value="", note="")
    row.update(kw)
    ROWS.append(row)


def main():
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    pf = J["preflight"]
    modes = pf["modes"]
    scan = pf["clamp_scan"]
    snap = {(r["mode"], r["quantity"]): r["recomputed"]
            for r in pf["reproduced_with_the_18_Sept_snapshot"]["rows"]}
    net = "$N$ = 64, $q$ = 2, $\\Delta z$ = 80.9 mm"

    fig = plt.figure(figsize=(7.6, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92],
                          hspace=0.66, wspace=0.28,
                          left=0.085, right=0.985, top=0.845, bottom=0.085)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axCc = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # =========================================================== (a) by band ==
    bands = list(C.BAND_LABELS)
    win = C.LOSS_WINDOW_LABEL
    keys = bands + [win]
    xs = np.array([0, 1, 2, 3, 4, 5.55])         # the window sits apart
    wmain, wabl = 0.30, 0.088

    def draw_group(ax, keys, xs, getter, getn, xlabels):
        for i, k in enumerate(keys):
            hatched = (k == win)
            for j, mode in enumerate(("blockE", "full")):
                col = P["BLUE"] if mode == "blockE" else P["MAGENTA"]
                ax.bar(xs[i] + (j - 0.5) * wmain, getter(mode, k), width=wmain,
                       color=col, edgecolor=P["TEXT1"], linewidth=0.5, zorder=3,
                       hatch="//" if hatched else None)
            for a, mode in enumerate(ABLATIONS):
                ax.bar(xs[i] + (a - 1.0) * 0.095, getter(mode, k), width=wabl,
                       color=GREY, alpha=0.75, edgecolor="none", zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=6.2)

    def gshare(mode, k):
        return modes[mode]["by_band"][k]["share_pct"]

    nband = {k: modes["blockE"]["by_band"][k]["n"] for k in keys}
    labA = ["%s\n(%s)" % (k.replace(" (loss window)", "\n(loss window)"),
                          "{:,}".format(nband[k])) for k in keys]
    draw_group(axA, keys, xs, gshare, nband, labA)
    axA.axvspan(4.75, 6.35, color=BAND_FILL, zorder=0)
    axA.set_ylim(0.0, 100.0)
    axA.set_xlim(-0.62, 6.25)
    axA.set_ylabel("share of the loss [%]")
    axA.set_xlabel("momentum band [GeV], with the state count n in brackets")
    axA.set_title("(a)  by momentum band", loc="left", pad=15.0)
    axA.text(0.015, 1.01, "8,000 states, " + net, transform=axA.transAxes,
             fontsize=7.5, color=P["TEXT2"], va="bottom")
    axA.grid(axis="x", visible=False)
    for i, k in enumerate(keys):
        for mode in ("blockE", "full"):
            v = gshare(mode, k)
            if v >= 3.0:
                axA.text(xs[i] + (0 if mode == "blockE" else 1) * wmain - 0.5 * wmain,
                         v + 1.6, "%.0f" % v, ha="center", fontsize=6.8,
                         color=P["TEXT2"])
    for k in keys:
        for mode in ("blockE", "full") + ABLATIONS:
            rec(panel="a", quantity="share of the loss [%]", group=k,
                series=NAME[mode], n=nband[k], value=gshare(mode, k),
                snapshot_18Sept_value="",
                note=("diagnostic only, not trained" if mode in ABLATIONS else ""))
        rec(panel="a", quantity="share of the states [%]", group=k,
            series="states", n=nband[k],
            value=modes["blockE"]["by_band"][k]["states_pct"],
            note="what an unweighted share would be if every state counted alike")

    # ======================================================= (b) by quarter ==
    qkeys = ["quarter %d" % i for i in (1, 2, 3, 4)]
    xq = np.arange(4, dtype=float)

    def gq(mode, k):
        return modes[mode]["by_quarter_of_z"][k]["share_pct"]

    nq = {k: modes["blockE"]["by_quarter_of_z"][k]["n"] for k in qkeys}
    labB = ["%s\n(n = %s)" % (k, "{:,}".format(nq[k])) for k in qkeys]
    draw_group(axB, qkeys, xq, gq, nq, labB)
    axB.set_ylim(0.0, 100.0)
    axB.set_xlim(-0.62, 3.62)
    axB.set_ylabel("share of the loss [%]")
    axB.set_xlabel("quarter of the crossing in $z$")
    axB.set_title("(b)  by quarter of $z$", loc="left", pad=15.0)
    axB.text(0.015, 1.01, "first quarter leaves $z_0$, fourth arrives at $z_1$",
             transform=axB.transAxes, fontsize=7.5, color=P["TEXT2"], va="bottom")
    axB.grid(axis="x", visible=False)
    for i, k in enumerate(qkeys):
        for j, mode in enumerate(("blockE", "full")):
            v = gq(mode, k)
            if v >= 3.0:
                axB.text(xq[i] + (j - 0.5) * wmain, v + 1.6, "%.0f" % v,
                         ha="center", fontsize=6.8, color=P["TEXT2"])
    qsnap = {"quarter 1": "share of the loss in the first quarter of z [%]",
             "quarter 4": "share of the loss in the last quarter of z [%]"}
    for k in qkeys:
        for mode in ("blockE", "full") + ABLATIONS:
            rec(panel="b", quantity="share of the loss [%]", group=k,
                series=NAME[mode], n=nq[k], value=gq(mode, k),
                snapshot_18Sept_value=snap.get((mode, qsnap.get(k, "")), ""),
                note=("diagnostic only, not trained" if mode in ABLATIONS else ""))

    # ======================================================== (c) clamp scan ==
    order = ["off", "2", "3", "5", "10"]
    xc = np.arange(len(order), dtype=float)
    inband = [scan[k]["share_10-50 (loss window)_pct"] for k in order]
    above = [scan[k]["share_>50_pct"] for k in order]
    axCc.plot(xc, inband, "-o", ms=5, lw=1.8, color=P["MAGENTA"], zorder=3,
              label="10-50 GeV loss window")
    axCc.plot(xc, above, "-s", ms=4.5, lw=1.8, color=P["GREEN"], zorder=3,
              label="> 50 GeV")
    i5 = order.index("5")
    axCc.axvline(i5, color=P["TEXT2"], lw=0.9, ls=(0, (3, 3)), zorder=2)
    axCc.annotate("the clamp the runs used", xy=(i5, 56.0), xytext=(i5 + 0.55, 63.0),
                  fontsize=7.5, color=P["TEXT2"], ha="center",
                  arrowprops=dict(arrowstyle="->", lw=0.8, color=P["TEXT2"]))
    for x, v in zip(xc, inband):
        axCc.text(x, v + 2.8, "%.0f" % v, ha="center", fontsize=6.8, color=P["MAGENTA"])
    for x, v in zip(xc, above):
        axCc.text(x, v + 2.8, "%.1f" % v, ha="center", fontsize=6.8, color=P["GREEN"])
    axCc.set_xticks(xc)
    axCc.set_xticklabels(["off"] + ["%s" % k for k in order[1:]])
    axCc.set_xlim(-0.35, len(order) - 0.55)
    axCc.set_ylim(0.0, 70.0)
    axCc.set_xlabel("clamp on the per-track factor, as a factor of its median")
    axCc.set_ylabel("share of the loss [%]")
    axCc.set_title("(c)  the clamp scan", loc="left", pad=15.0)
    axCc.text(0.015, 1.01, "cost-weighted loss, clamp varied", transform=axCc.transAxes,
              fontsize=7.5, color=P["TEXT2"], va="bottom")
    axCc.legend(loc="upper left", fontsize=7.5)
    axCc.grid(axis="x", visible=False)
    for k, a, b in zip(order, inband, above):
        rec(panel="c", quantity="share of the loss [%]", group="clamp = %s" % k,
            series="10-50 GeV loss window", value=a)
        rec(panel="c", quantity="share of the loss [%]", group="clamp = %s" % k,
            series="> 50 GeV", value=b)
        rec(panel="c", quantity="rank correlation of share with the true endpoint cost",
            group="clamp = %s" % k, series="overall", value=scan[k]["spearman_share_vs_cost"])
        rec(panel="c", quantity="rank correlation of share with the true endpoint cost",
            group="clamp = %s" % k, series="in 10-50 GeV",
            value=scan[k]["spearman_share_vs_cost_in_band"])
        rec(panel="c", quantity="share carried by the worst 1 % of states [%]",
            group="clamp = %s" % k, series="cost-weighted loss",
            value=scan[k]["top_1pct_of_states_share_pct"])

    # ============================================= (d) does the loss rank cost ==
    groups = [("rank correlation\nwith the true cost,\noverall", "spearman_share_vs_cost",
               "rank correlation of share with true cost", 1.0),
              ("rank correlation\nwith the true cost,\nin 10-50 GeV", "spearman_share_vs_cost_in_band",
               "rank correlation of share with true cost, in 10-50 GeV", 1.0),
              ("share carried by\nthe worst 1 %\nof states", "top_1pct_of_states_share_pct",
               "top 1 per cent of states, share of the loss [%]", 100.0)]
    xd = np.arange(len(groups), dtype=float)
    wd = 0.32
    for j, mode in enumerate(("blockE", "full")):
        col = P["BLUE"] if mode == "blockE" else P["MAGENTA"]
        vals = [modes[mode][key] / sc for _, key, _, sc in groups]
        axD.bar(xd + (j - 0.5) * wd, vals, width=wd, color=col,
                edgecolor=P["TEXT1"], linewidth=0.5, zorder=3, label=NAME[mode])
        for x, v, (_, key, qsn, sc) in zip(xd, vals, groups):
            axD.text(x + (j - 0.5) * wd, v + 0.022,
                     ("%.2f" % v) if sc == 1.0 else ("%.0f %%" % (v * 100.0)),
                     ha="center", fontsize=6.9, color=P["TEXT2"])
            rec(panel="d", quantity=qsn, group=qsn.split(",")[0], series=NAME[mode],
                n=pf["n_states"], value=modes[mode][key],
                snapshot_18Sept_value=snap.get((mode, qsn), ""))
    axD.set_xticks(xd)
    axD.set_xticklabels([g[0] for g in groups], fontsize=7.0)
    axD.set_ylim(0.0, 1.22)
    axD.set_xlim(-0.62, len(groups) - 0.38)
    axD.set_ylabel("correlation, or fraction of the loss")
    axD.set_title("(d)  does the loss follow the cost?", loc="left", pad=15.0)
    axD.text(0.015, 1.01, "the endpoint cost is a diagnostic, in neither loss",
             transform=axD.transAxes, fontsize=7.5, color=P["TEXT2"], va="bottom")
    axD.legend(loc="upper left", fontsize=7.5)
    axD.grid(axis="x", visible=False)

    # ----------------------------------------------------------- the legend --
    handles = [Patch(facecolor=P["BLUE"], edgecolor=P["TEXT1"], lw=0.5, label=NAME["blockE"]),
               Patch(facecolor=P["MAGENTA"], edgecolor=P["TEXT1"], lw=0.5, label=NAME["full"]),
               Patch(facecolor=P["MAGENTA"], edgecolor=P["TEXT1"], lw=0.5, hatch="//",
                     label="the 10-50 GeV window, which overlaps two bands"),
               Line2D([0], [0], color=GREY, lw=5, alpha=0.75,
                      label="one factor of the weight switched off "
                            "(lever arm, track factor, window):\ndiagnostic only, not trained")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0),
               ncol=2, fontsize=7.4, handlelength=2.2, columnspacing=1.4,
               labelspacing=0.5, borderaxespad=0.2)

    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(PNG)
    plt.close(fig)

    rec(panel="-", quantity="note", group="18 September snapshot",
        series="-", value="",
        note="the snapshot values exist only for the quantities the original "
             "gate stored, whose momentum binning was the old 1/2/5/10/25/200 GeV "
             "set; per-band shares in the paper's bands are available for the "
             "extended network only")
    C.write_csv(CSV, [C.jsonable(r) for r in ROWS],
                ["panel", "quantity", "group", "series", "n", "value",
                 "snapshot_18Sept_value", "note"])
    print("wrote %s" % PNG)
    print("wrote %s  (%d rows)" % (CSV, len(ROWS)))


if __name__ == "__main__":
    main()
