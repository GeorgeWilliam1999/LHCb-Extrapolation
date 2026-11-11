"""Per-run oscillation statistics from rounds.csv and restarts.csv (read-only).

Writes round_stats.csv, restart_stats.csv and the plots per_round_*.png,
late_restarts.png into this folder.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STORE = "/data/bfys/gscriven/rkpinn_store/runs"
HERE = os.path.dirname(os.path.abspath(__file__))
SCORE = "validation_endpoint_error_larger_of_x_and_y_median_micrometres"
LAST = 30

RUNS = [  # key, label (steps/stages, loss)
    ("49e5e2450eaa", "64/2 pooled"),
    ("da3a1d00b33d", "64/2 cost-w 10-50"),
    ("f977c7b61049", "64/2 cost-w 3-8"),
    ("75200e48408c", "128/8 pooled"),
    ("237dc4101ad5", "128/8 cost-w 10-50"),
    ("521415f2c534", "128/8 cost-w 3-8"),
    ("f38ec3e29819", "256/16 pooled"),
    ("aec34bbd4934", "256/16 cost-w 10-50 (ext. 160)"),
    ("9c3c5d86784b", "256/16 cost-w 3-8"),
]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def stats(key, label):
    rounds = read(os.path.join(STORE, key, "rounds.csv"))
    restarts = read(os.path.join(STORE, key, "restarts.csv"))
    n = len(rounds)
    r = np.array([int(x["round"]) for x in rounds])
    score = np.array([float(x[SCORE]) for x in rounds])
    lf = np.array([float(x["loss_first"]) for x in rounds])
    ll = np.array([float(x["loss_last"]) for x in rounds])
    # late window: last 30 rounds, never including round 1 (reference states)
    lo = max(1, n - LAST)
    w = slice(lo, n)
    s, a, b = score[w], lf[w], ll[w]
    dls = np.diff(np.log(s))
    # loss on the round's own draw, before and after the round's training
    dll = np.diff(np.log(b))
    # jump at the change of draw: loss_first(r+1) / loss_last(r)
    draw_jump = np.log(lf[lo + 1:n] / ll[lo:n - 1])
    # what one round of training removes: loss_last(r) / loss_first(r)
    train_gain = np.log(b / a)
    corr_last = float(np.corrcoef(dll, dls)[0, 1])
    corr_first = float(np.corrcoef(np.diff(np.log(a)), dls)[0, 1])
    # does the score after round r track the loss_first of round r+1 (how well
    # the network fits a fresh draw)?
    late_restarts = [x for x in restarts if int(x["round"]) >= r[lo]]
    stopped = sum(int(x["stopped_early"]) for x in late_restarts)
    renewed = sum(int(x["factor_renewed"]) for x in late_restarts)
    it = np.array([int(x["iterations"]) for x in late_restarts])
    nonfinite = sum(not np.isfinite(float(x["loss_after"])) for x in late_restarts)
    rising = sum(float(x["loss_after"]) > float(x["loss_before"]) for x in late_restarts)
    first_gain = np.array([float(x["gain"]) for x in late_restarts if int(x["restart_in_round"]) == 1])
    last_gain = np.array([float(x["gain"]) for x in late_restarts if int(x["restart_in_round"]) == 25])
    return dict(
        key=key, label=label, rounds=n, window=f"{r[lo]}-{r[-1]}", n_window=n - lo,
        score_median=float(np.median(s)), score_min=float(s.min()), score_max=float(s.max()),
        std_dlog_score=float(np.std(dls, ddof=1)),
        largest_jump_pct=float(100 * (np.exp(np.abs(dls).max()) - 1)),
        frac_worse=float(np.mean(dls > 0)),
        std_dlog_loss_last=float(np.std(dll, ddof=1)),
        draw_jump_median_pct=float(100 * (np.exp(np.median(draw_jump)) - 1)),
        draw_jump_std=float(np.std(draw_jump, ddof=1)),
        train_gain_median_pct=float(100 * (np.exp(np.median(train_gain)) - 1)),
        corr_dloss_last_dscore=corr_last, corr_dloss_first_dscore=corr_first,
        late_restarts=len(late_restarts), stopped_early=stopped, factor_renewed=renewed,
        iterations_min=int(it.min()), nonfinite=nonfinite, restarts_that_raised_loss=rising,
        gain_restart1_median=float(np.median(first_gain)),
        gain_restart25_median=float(np.median(last_gain)),
    ), dict(r=r, score=score, lf=lf, ll=ll, lo=lo, restarts=restarts)


def main():
    rows, series = [], {}
    for key, label in RUNS:
        row, ser = stats(key, label)
        rows.append(row)
        series[key] = (label, ser)
    with open(os.path.join(HERE, "round_stats.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for row in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    # (a) per-round score and loss, last 30 highlighted
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, (key, (label, s)) in zip(axes.ravel(), series.items()):
        r, lo = s["r"], s["lo"]
        ax.plot(r, s["score"], color="0.6", lw=1)
        ax.plot(r[lo:], s["score"][lo:], color="C0", lw=1.6, marker=".", label="validation error")
        ax.set_yscale("log")
        ax.set_ylabel("validation error, larger of x,y [µm]", color="C0")
        ax.set_xlabel("round")
        ax.set_title(f"{label}  ({key})", fontsize=9)
        ax.axvspan(r[lo], r[-1], color="C0", alpha=0.06)
        ax2 = ax.twinx()
        ax2.plot(r, s["lf"], color="C3", lw=0.8, alpha=0.7, label="loss at round start (fresh draw)")
        ax2.plot(r, s["ll"], color="C1", lw=0.8, alpha=0.9, label="loss at round end")
        ax2.set_yscale("log")
        ax2.set_ylabel("training loss", color="C1")
        if key == RUNS[0][0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    fig.suptitle("Per-round validation error and training loss; shaded = last 30 rounds")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "per_round_score_and_loss.png"), dpi=120)

    # (c) per-restart loss within late rounds: 49e5 and aec3, last 6 rounds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key in zip(axes, ["49e5e2450eaa", "aec34bbd4934"]):
        label, s = series[key]
        rs = s["restarts"]
        last = sorted({int(x["round"]) for x in rs})[-6:]
        for rr in last:
            xs = [x for x in rs if int(x["round"]) == rr]
            loss = [float(xs[0]["loss_before"])] + [float(x["loss_after"]) for x in xs]
            idx = (rr - last[0]) * 26 + np.arange(26)
            ax.plot(idx, loss, marker=".", ms=3, label=f"round {rr}")
        ax.set_yscale("log")
        ax.set_xlabel("restart (0 = before training, each round on its own fresh draw)")
        ax.set_ylabel("training loss")
        ax.set_title(f"{label} ({key}): loss after every restart, last 6 rounds", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "late_restarts.png"), dpi=120)


if __name__ == "__main__":
    main()
