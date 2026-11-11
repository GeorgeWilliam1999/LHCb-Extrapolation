"""Plots (b) noise floor vs observed, (d) fixed-set loss vs score, (e) L-BFGS arms."""
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "round_stats.csv"))))
floors = {}
for p in glob.glob(os.path.join(HERE, "noise_floor_*.json")):
    d = json.load(open(p))
    floors[d["key"]] = d

# (b) observed round-to-round spread vs sampling-noise floor
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(rows))
obs_loss = [float(r["std_dlog_loss_last"]) for r in rows]
obs_score = [float(r["std_dlog_score"]) for r in rows]
ax.bar(x - 0.3, obs_loss, 0.2, label="observed: std of round-to-round Δlog(loss at round end)")
ax.bar(x - 0.1, obs_score, 0.2, label="observed: std of round-to-round Δlog(validation error)")
fl_c = [np.sqrt(2) * floors[r["key"]]["chain_std_log"] if r["key"] in floors else np.nan for r in rows]
fl_f = [floors[r["key"]]["fixed_set_std_dlog"] if r["key"] in floors else np.nan for r in rows]
ax.bar(x + 0.1, fl_c, 0.2, label="noise floor: √2 × std log(loss) over fresh draws, fixed weights")
ax.bar(x + 0.3, fl_f, 0.2, label="loss on one fixed 128k set: std of round-to-round Δlog")
ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in rows], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("std of Δlog per round")
ax.set_title("Round-to-round spread, last 30 rounds (floors measured where a bar is shown)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "spread_vs_noise_floor.png"), dpi=120)

# (d) loss on one fixed set vs score, per late snapshot
fig, axes = plt.subplots(1, len(floors), figsize=(7 * len(floors), 4.5), squeeze=False)
for ax, (key, d) in zip(axes[0], sorted(floors.items())):
    fs = d["fixed_set"]
    r = [f["round"] for f in fs]
    ax.plot(r, [f["loss_fixed_set"] for f in fs], "k-o", ms=3, label="loss on one fixed 128k set")
    ax.plot(r, [f["loss_last"] for f in fs], color="C1", lw=1, label="loss at round end (own draw)")
    ax.plot(r, [f["loss_first"] for f in fs], color="C3", lw=1, label="loss at round start (fresh draw)")
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("training loss")
    ax2 = ax.twinx()
    ax2.plot(r, [f["score"] for f in fs], color="C0", marker=".", label="validation error")
    ax2.set_ylabel("validation error [µm]", color="C0")
    ax.set_title("%s: corr(Δlog fixed-set loss, Δlog score) = %.2f"
                 % (key, d["corr_dlog_fixedloss_dlog_score"]), fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fixed_set_loss_vs_score.png"), dpi=120)

# (e) L-BFGS arms from one snapshot
arms = sorted(glob.glob(os.path.join(HERE, "arm_*.json")))
if arms:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for p in arms:
        d = json.load(open(p))
        rr = d["rows"]
        ls = "--" if d["arm"] == "base_repeat" else "-"
        lab = "%s (draw %s, lr %g, history %d)" % (d["arm"], d["draw"], d["lr"], d["history"])
        a1.plot([x["restart"] for x in rr], [x["loss"] for x in rr], ls, marker="o", label=lab)
        a2.plot([x["restart"] for x in rr], [x["score"] for x in rr], ls, marker="o", label=lab)
    a1.set_yscale("log")
    a1.set_xlabel("restart (200 L-BFGS iterations each)")
    a1.set_ylabel("loss on the arm's own draw")
    a2.set_xlabel("restart")
    a2.set_ylabel("validation error [µm]")
    a2.axhline(5.4629, color="0.5", ls=":", label="trainer, round 59 after 25 restarts")
    a2.legend(fontsize=7)
    a1.set_title("64/2 pooled: from snapshot 58, round 59 (3 of 25 restarts)", fontsize=9)
    a2.set_title("validation error after each restart", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "lbfgs_arms.png"), dpi=120)
