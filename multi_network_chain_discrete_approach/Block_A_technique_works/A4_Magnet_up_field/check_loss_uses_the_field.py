#!/usr/bin/env python
"""A4.3 pre-submit gate: does `--field up` really reach the physics loss?

`train.py --field` is passed to `make_field` and on into the `LHCbRates` object
the physics loss evaluates its stage rates with. Nothing downstream would fail
loudly if it did not: the loss would simply be built on the wrong map, the runs
would converge, and the scored numbers would be scored against the up-field
references in the npz - producing a plausible-looking but meaningless result.

So it is measured before the farm submission rather than assumed. One L-BFGS
restart is run twice from the same seed on the SAME up-field dataset, once with
`--field up` and once with `--field down`. Identical initial parameters,
identical data; only the field the loss is built on differs. The two losses must
differ.

    results/probe_up.json, results/probe_down.json   the two runs
    results/loss_field_probe.json                    the comparison
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json    # noqa: E402

import use_shared     # noqa: F401,E402
from _shared.train import main as train_main    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(RES, "frozen_leg_up.npz")


def main():
    rec = {}
    for which in ("up", "down"):
        tag = "probe_" + which
        for suffix in (".pt", "_history.csv", ".json"):
            p = os.path.join(RES, tag + suffix)
            if os.path.exists(p):
                os.remove(p)                 # train.py resumes otherwise
        rec[which] = train_main([
            "--data", DATA, "--mode", "physics", "--seed", "0", "--q", "8",
            "--width", "50", "--depth", "4", "--field", which,
            "--outer-cap", "1", "--no-confirm", "--out", RES, "--tag", tag])

    out = {
        "what": "one L-BFGS restart, same seed and same up-field dataset, "
                "with the physics loss built on each polarity in turn",
        "loss_after_one_restart": {w: rec[w]["final_loss"] for w in rec},
        "field_recorded_in_json": {w: rec[w]["field"] for w in rec},
        "test_endpoint_med_um_vs_up_references": {
            w: rec[w]["test"]["endpoint_med_um"] for w in rec},
        "losses_differ": bool(rec["up"]["final_loss"] != rec["down"]["final_loss"]),
        "ratio_down_over_up": rec["down"]["final_loss"] / rec["up"]["final_loss"],
    }
    with open(os.path.join(RES, "loss_field_probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    assert out["losses_differ"], "--field does not reach the physics loss"


if __name__ == "__main__":
    main()
