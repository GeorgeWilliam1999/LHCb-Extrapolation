"""Write the configurations of the first batch (DESIGN.md): 20 files in configurations/.

    first_batch.py
"""
from __future__ import annotations

import os

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = dict(seed=1, restarts=5, iterations=200, rounds_at_most=200, window=15)


def run(reference, start, steps, width, depth=2, scale="times_qop", loss="huber"):
    name = "%s_%s_%dx%d" % (reference, "nystrom_%d" % steps if start == "nystrom" else start, depth, width)
    if scale != "times_qop" and start != "none":
        name += "_fixed_scale"
    if loss != "huber":
        name += "_squared_loss"
    return dict(name=name, reference=reference, start=start, steps=steps, width=width, depth=depth,
                scale=scale, loss=loss, **COMMON)


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
    return rows


if __name__ == "__main__":
    rows = batch()
    assert len({r["name"] for r in rows}) == len(rows)
    for r in rows:
        with open(os.path.join(HERE, "configurations", r["name"] + ".yaml"), "w") as handle:
            yaml.safe_dump(r, handle)
    print("%d configurations" % len(rows))
    print("\n".join(r["name"] for r in rows))
