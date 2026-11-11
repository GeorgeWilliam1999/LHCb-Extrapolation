#!/usr/bin/env python
"""Figure 9 - how the six runs converged (figures/fig_convergence.png).

Three matched pairs of networks, one column each: N = 64, q = 2, dz = 80.9 mm;
N = 128, q = 8, dz = 40.5 mm; N = 256, q = 16, dz = 20.2 mm.  In every column
the run trained under the pooled loss and the run trained under the
cost-weighted loss are drawn together.

WHAT THE FIGURE SHOWS.

Top row.  The validation endpoint error - the median over the 1,463 validation
    tracks of max(|dx|, |dy|) at z1 - round by round, as points, with the
    ten-round running median over them as a line.  Where the paper's plateau
    rule holds at the last round (the median of the last ten rounds no more
    than 5 per cent below the median of the ten before, at each of the last
    three rounds) the round at which the rule first held is marked with a
    vertical tick.  Where it does not hold the curve is annotated "not flat":
    two of the three pooled-loss runs were still falling when they were
    stopped.  The dotted vertical line on each pooled-loss curve is the round
    at which the 18 September 2026 snapshot was taken; everything to the right
    of it is the extension this paper uses.

    The round axis is the ORDINAL round, 1 .. rounds, which is what the paper's
    tables count.  Each of the three pooled-loss runs skips exactly one number
    in its stored round counter where the run was picked up and extended past
    the snapshot, so its stored counter runs one ahead of the ordinal from
    there on; the ordinal is used so that the marked rounds are the same
    numbers the tables and paper_numbers.json carry.

Bottom row.  The training loss at the end of every restart, on a logarithmic
    axis, for the same two runs.  Alternate rounds are shaded so the round
    boundaries are visible; every round is 25 restarts, and each round redraws
    its 32,000 training states, which is the jump at each boundary.  The two
    losses are different objectives and their absolute sizes are not
    comparable: only the shape of each curve is.

INPUTS READ (all read-only).

  <S>/Block_F_reweighted_loss/F1_Training/results/full/{N064_q02,N128_q08,N256_q16}/
      rounds.csv    round, val_z1_pos_med_um
      history.csv   restart, round, loss_after
  <S>/Block_E_single_network_chain/E1_Network_grid/results/{N064_q02,N128_q08,N256_q16}/
      the same two files, for the extended pooled-loss runs
  results/tab_extended_vs_snapshot.csv   snapshot_rounds, the 18 September round
  results/paper_numbers.json             group `headline`, for the recorded
                                         plateau verdict and first-hold round
                                         each panel is checked against
  scripts/common.py                      loaders, the de-duplication rule, the
                                         plateau rule, the palette

Both files of every run go through common.dedup (last row written kept): the
N = 256, q = 16 cost-weighted run was trained by two farm jobs at once from
restart 1,198 on, so its history.csv holds 302 repeated restart numbers and its
rounds.csv 13 repeated round numbers.

OUTPUTS.

  figures/fig_convergence.png
  results/fig09_convergence.csv   every plotted point, and the marked rounds.

Run:
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig09_convergence.py
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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["sc_paper_common"] = C
_spec.loader.exec_module(C)

FIGURES = os.path.join(C.PAPER, "figures")
PNG = os.path.join(FIGURES, "fig_convergence.png")
CSV = os.path.join(C.RESULTS, "fig09_convergence.csv")

P = C.PALETTE
FAINT = "#e5e4df"
GREY = "#9a9a94"

plt.rcParams.update({
    "figure.facecolor": P["SURFACE"], "axes.facecolor": P["SURFACE"],
    "savefig.facecolor": P["SURFACE"], "axes.edgecolor": P["NEUTRAL"],
    "axes.labelcolor": P["TEXT2"], "text.color": P["TEXT1"],
    "xtick.color": P["TEXT2"], "ytick.color": P["TEXT2"], "font.size": 8.5,
    "axes.titlesize": 9.0, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": FAINT, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 150,
})

COL = {"pooled": P["BLUE"], "reweighted": P["MAGENTA"]}
LOSSNAME = {"pooled": "pooled loss", "reweighted": "cost-weighted loss"}
WINDOW = C.PLATEAU_WINDOW                    # 10 rounds

ROWS = []


def rec(**kw):
    row = dict(panel="", N="", q="", loss="", quantity="", x="", y="", note="")
    row.update(kw)
    ROWS.append(row)


def running_median(v, w=WINDOW):
    """The trailing median of the last `w` values, defined from index w-1 on."""
    v = np.asarray(v, dtype=float)
    idx = np.arange(w - 1, len(v))
    return idx, np.array([np.median(v[i - w + 1:i + 1]) for i in idx])


def main():
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    pairs = J["headline"]["pairs"]
    snap_rounds = {(int(r["N"]), int(r["q"])): int(r["snapshot_rounds"])
                   for r in C.read_csv(os.path.join(C.RESULTS,
                                                    "tab_extended_vs_snapshot.csv"))}

    # the two rows carry DIFFERENT x quantities (round above, restart below),
    # so no axis is shared
    fig, axes = plt.subplots(2, 3, figsize=(7.9, 5.6))
    fig.subplots_adjust(left=0.105, right=0.972, top=0.775, bottom=0.085,
                        hspace=0.46, wspace=0.28)

    for c, (N, q) in enumerate(C.PAIRS):
        axT, axB = axes[0, c], axes[1, c]
        title = "$N$ = %d, $q$ = %d, $\\Delta z$ = %.1f mm" % (N, q, C.dz_mm(N))
        axT.set_title(title, loc="center", fontsize=8.7, pad=14.0)

        max_round, max_restart = 0, 0
        bounds = set()
        notes, ticks, allval = [], [], []
        for loss in ("pooled", "reweighted"):
            d = C.pooled_run(N, q) if loss == "pooled" else C.reweighted_run(N, q)
            rounds, rdup, rraw = C.load_rounds(d)
            hist, hdup, hraw = C.load_history(d)
            stored = np.array([int(float(r["round"])) for r in rounds])
            # The x axis is the ORDINAL round, 1 .. rounds, which is what the
            # paper's tables count.  The three pooled-loss runs' stored round
            # counter skips exactly one number where the run was picked up and
            # extended past the 18 September snapshot, so the stored counter
            # runs one ahead of the ordinal after that point.
            rn = np.arange(1, len(rounds) + 1)
            skipped = stored[-1] - len(rounds)
            val = np.array([float(r["val_z1_pos_med_um"]) for r in rounds])
            rs = np.array([int(float(h["restart"])) for h in hist])
            lo = np.array([float(h["loss_after"]) for h in hist])
            hr = np.array([int(float(h["round"])) for h in hist])
            note = []
            if rdup or hdup:
                note.append("de-duplicated: %d repeated restart numbers (%d rows), "
                            "%d repeated round numbers (%d rows)"
                            % (len(hdup), hraw - len(hist), len(rdup), rraw - len(rounds)))
            if skipped:
                note.append("the stored round counter skips %d number(s) at the "
                            "extension; x is the ordinal round" % skipped)
            dupnote = "; ".join(note)

            # ---------------------------------------------- top: validation --
            axT.plot(rn, val, "o", ms=2.6, color=COL[loss], alpha=0.55,
                     mec="none", zorder=3)
            ii, rm = running_median(val)
            axT.plot(rn[ii], rm, "-", lw=1.7, color=COL[loss], zorder=4)
            for x, y in zip(rn, val):
                rec(panel="top", N=N, q=q, loss=LOSSNAME[loss],
                    quantity="validation endpoint error, median max(|dx|,|dy|) [um]",
                    x=int(x), y=float(y), note=dupnote)
            for x, y in zip(rn[ii], rm):
                rec(panel="top", N=N, q=q, loss=LOSSNAME[loss],
                    quantity="ten-round running median of that error [um]",
                    x=int(x), y=float(y))

            held = C.plateaued_now(val)
            first = C.first_plateau_round(val)
            # cross-check against the verdict already recorded in the JSON
            conv = pairs[C.run_key(N, q)][loss]["convergence"]
            assert bool(conv["plateaued_now"]) == bool(held), (N, q, loss)
            assert conv["first_plateau_round"] == first, (N, q, loss)
            assert conv["rounds"] == len(rounds), (N, q, loss)
            assert conv["unique_restarts"] == len(hist), (N, q, loss)
            assert conv["rounds_rows_read"] == rraw and conv["history_rows_read"] == hraw, (N, q, loss)
            allval.append(val)
            if held and first is not None:
                axT.axvline(rn[first - 1], color=COL[loss], lw=1.0,
                            ls=(0, (2, 2)), alpha=0.9, zorder=2)
                ticks.append((int(rn[first - 1]), COL[loss]))
                rec(panel="top", N=N, q=q, loss=LOSSNAME[loss],
                    quantity="round at which the plateau rule first held",
                    x=int(rn[first - 1]), y="", note="the rule still holds at the last round")
            else:
                notes.append((loss, "not flat at the last round"))
                rec(panel="top", N=N, q=q, loss=LOSSNAME[loss],
                    quantity="round at which the plateau rule first held",
                    x=(int(rn[first - 1]) if first is not None else ""), y="",
                    note="the rule does NOT hold at the last round: still falling")

            if loss == "pooled":
                sr = snap_rounds[(N, q)]
                axT.axvline(sr, color=P["TEXT2"], lw=0.9, ls=(0, (1, 2.4)), zorder=2)
                axT.text(sr, 1.005, "18 Sept", transform=axT.get_xaxis_transform(),
                         fontsize=6.4, color=P["TEXT2"], ha="center", va="bottom")
                rec(panel="top", N=N, q=q, loss=LOSSNAME[loss],
                    quantity="round of the 18 September 2026 snapshot",
                    x=int(sr), y="", note="everything after it is the extension")

            # ------------------------------------------------ bottom: loss --
            axB.plot(rs, lo, "-", lw=0.6, color=COL[loss], alpha=0.85, zorder=3)
            for x, y in zip(rs, lo):
                rec(panel="bottom", N=N, q=q, loss=LOSSNAME[loss],
                    quantity="training loss at the end of the restart",
                    x=int(x), y=float(y), note=dupnote)
            max_round = max(max_round, int(rn.max()))
            max_restart = max(max_restart, int(rs.max()))
            for b in np.flatnonzero(np.diff(hr)) + 1:
                bounds.add(int(rs[b]))
            bounds.add(int(rs[0]))

        # alternate-round shading, so the round boundaries are visible
        bd = sorted(bounds) + [max_restart + 1]
        for k in range(0, len(bd) - 1, 2):
            axB.axvspan(bd[k], bd[k + 1], color=FAINT, alpha=0.55, lw=0, zorder=0)

        axT.set_yscale("log")
        axT.set_xlim(0, max_round + 1)
        vall = np.concatenate(allval)
        ylo, yhi = vall.min() * 0.86, vall.max() * 1.12
        axT.set_ylim(ylo, yhi)
        for xt, col in ticks:                      # the plateau ticks, on the axis
            axT.plot([xt], [ylo], "^", ms=5.5, color=col, zorder=6, clip_on=False)
        axT.yaxis.set_major_locator(FixedLocator([80, 100, 150, 200, 300, 400, 500]))
        axT.yaxis.set_minor_locator(NullLocator())
        axT.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
        axT.set_xlabel("round")
        axB.set_yscale("log")
        # a little headroom past the last restart, so the last tick label is
        # not clipped against the axes edge
        axB.set_xlim(0, max_restart * 1.05)
        axB.set_xlabel("restart")
        if c == 0:
            axT.set_ylabel("validation endpoint error [$\\mu$m]\nmedian max(|d$x$|, |d$y$|)")
            axB.set_ylabel("training loss")
        for loss, txt in notes:
            axT.text(0.97, 0.95 if loss == "pooled" else 0.83, txt,
                     transform=axT.transAxes, ha="right", va="top",
                     fontsize=7.0, color=COL[loss])

    handles = [Line2D([0], [0], color=COL["pooled"], marker="o", ms=3.5, lw=1.7,
                      label="pooled loss"),
               Line2D([0], [0], color=COL["reweighted"], marker="o", ms=3.5, lw=1.7,
                      label="cost-weighted loss"),
               Line2D([0], [0], color=GREY, lw=1.0, ls=(0, (2, 2)),
                      label="round the plateau rule first held (tick)"),
               Line2D([0], [0], color=P["TEXT2"], lw=0.9, ls=(0, (1, 2.4)),
                      label="round of the 18 September 2026 snapshot")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0),
               ncol=2, fontsize=7.5, handlelength=2.4, columnspacing=1.8,
               labelspacing=0.5, borderaxespad=0.2)
    fig.text(0.075, 0.915,
             "Above: one point per round; the line is the ten-round running median.\n"
             "Below: the loss at the end of every restart, alternate rounds shaded "
             "(25 restarts each).\nThe two losses are different objectives: their "
             "absolute sizes are not comparable.",
             fontsize=6.9, color=P["TEXT2"], va="top", ha="left", linespacing=1.45)

    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(PNG)
    plt.close(fig)

    C.write_csv(CSV, [C.jsonable(r) for r in ROWS],
                ["panel", "N", "q", "loss", "quantity", "x", "y", "note"])
    print("wrote %s" % PNG)
    print("wrote %s  (%d rows)" % (CSV, len(ROWS)))


if __name__ == "__main__":
    main()
