#!/usr/bin/env python
"""Build the frozen-leg dataset on the magnet-up field, and say how it differs
from the magnet-down one.

The input states are the same event-derived states the v2 baseline used; only
the field the reference trajectories are integrated with changes. This script
builds both polarities so the two fiducial-cut counts and the two populations
can be compared directly.

    results/frozen_leg_up.npz        the dataset the trainings consume
    results/frozen_leg_up_meta.json  its counts and scales (written by prepare)
    results/frozen_leg_down.npz      the same builder on the down field
    results/dataset_meta.json        the comparison: counts, fiducial removals,
                                     and whether the two input populations are
                                     the same set of states as v2's
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json      # noqa: E402
import numpy as np  # noqa: E402

import use_shared  # noqa: F401,E402
from _shared.prepare import frozen_leg_dataset   # noqa: E402
from _shared.reference import field_md5, field_path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
V2_NPZ = os.path.join(HERE, "..", "..", "Block_0_first_pass", "S2b_One_step_network_v2", "results",
                      "frozen_leg_data.npz")


def main():
    os.makedirs(RES, exist_ok=True)
    out = {}
    for which in ("up", "down"):
        npz = os.path.join(RES, "frozen_leg_%s.npz" % which)
        print("=== building the %s-field dataset ===" % which, flush=True)
        frozen_leg_dataset(q=8, field=which, n_train=2000, seed=20260718,
                           rebase_mm=60.0, fiducial=True, out_npz=npz)
        with open(os.path.splitext(npz)[0] + "_meta.json") as f:
            out[which] = json.load(f)
        out[which]["field_file"] = field_path(which)
        out[which]["field_md5"] = field_md5(which)

    # Are the two populations the same *input* states, and the same as v2's?
    up = np.load(os.path.join(RES, "frozen_leg_up.npz"))
    dn = np.load(os.path.join(RES, "frozen_leg_down.npz"))
    v2 = np.load(os.path.abspath(V2_NPZ)) if os.path.exists(V2_NPZ) else None

    comp = {}
    for split in ("train", "val", "test"):
        d = {"n_up": int(len(up["%s_S" % split])),
             "n_down": int(len(dn["%s_S" % split]))}
        if v2 is not None and "%s_S" % split in v2.files:
            d["n_v2"] = int(len(v2["%s_S" % split]))
            d["down_equals_v2_bitwise"] = bool(
                d["n_down"] == d["n_v2"]
                and np.array_equal(dn["%s_S" % split], v2["%s_S" % split]))
        # the input states are built with the polarity-specific rebase, so
        # compare the surviving sets as sets of rows
        su = {tuple(r) for r in up["%s_S" % split]}
        sd = {tuple(r) for r in dn["%s_S" % split]}
        d["n_shared_states"] = len(su & sd)
        d["only_up"] = len(su - sd)
        d["only_down"] = len(sd - su)
        d["identical_population"] = bool(su == sd)
        comp[split] = d

    meta = {
        "note": "same event-derived input states as S2b_One_step_network_v2; "
                "only the field the references are integrated with changes",
        "up": out["up"], "down": out["down"],
        "population_comparison": comp,
        "in_scale_up": out["up"]["in_scale"],
        "in_scale_down": out["down"]["in_scale"],
        "out_scale_up": out["up"]["out_scale"],
        "out_scale_down": out["down"]["out_scale"],
    }
    with open(os.path.join(RES, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(comp, indent=1))
    print("counts up  :", out["up"]["counts"], "dropped", out["up"]["dropped_by_fiducial"])
    print("counts down:", out["down"]["counts"], "dropped", out["down"]["dropped_by_fiducial"])


if __name__ == "__main__":
    main()
