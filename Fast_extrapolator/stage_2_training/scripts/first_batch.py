"""Write the configurations of the first batch (DESIGN.md): 20 files in configurations/.

    first_batch.py
"""
from __future__ import annotations

import os

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The stopping rule of 2026-11-25: a run ends when its best validation median of the last
# 30 rounds is not 1 % below the best before them. The first rule (15 rounds, 2 %, at most
# 200 rounds) ended runs whose validation error still fell by 6 % in 30 rounds.
COMMON = dict(restarts=5, iterations=200, rounds_at_most=1000, window=30, fall_needed=0.01)


def run(reference, start, steps, width, depth=2, scale="times_qop", loss="huber", seed=1):
    name = "%s_%s_%dx%d" % (reference, "nystrom_%d" % steps if start == "nystrom" else start, depth, width)
    if scale != "times_qop" and start != "none":
        name += "_fixed_scale"
    if loss != "huber":
        name += "_squared_loss"
    if seed != 1:
        name += "_seed_%d" % seed
    return dict(name=name, reference=reference, start=start, steps=steps, width=width, depth=depth,
                scale=scale, loss=loss, seed=seed, **COMMON)


def batch():
    rows = [run("cubic", "straight_line", 0, w) for w in (32, 64, 128)]
    rows += [run("cubic", "straight_line", 0, 64, depth=3),
             run("cubic", "straight_line", 0, 64, loss="squared"),
             run("cubic", "straight_line", 0, 64, scale="fixed"),
             run("cubic", "none", 0, 64, scale="fixed")]
    rows += [run("linear", "straight_line", 0, w) for w in (64, 128)]
    rows += [run("cubic", "nystrom", n, w) for n in (4, 8) for w in (32, 64)]
    rows += [run("cubic", "nystrom", 12, 64)]
    rows += [run("linear", "nystrom", n, w) for n in (6, 8, 12) for w in (32, 64)]
    # the run of width 32 with seed 1 did not train (it stays at 10 mm); a second seed
    rows += [run("cubic", "straight_line", 0, 32, seed=2)]
    return rows


if __name__ == "__main__":
    rows = batch()
    assert len({r["name"] for r in rows}) == len(rows)
    for r in rows:
        with open(os.path.join(HERE, "configurations", r["name"] + ".yaml"), "w") as handle:
            yaml.safe_dump(r, handle)
    print("%d configurations" % len(rows))
    print("\n".join(r["name"] for r in rows))
