#!/usr/bin/env python
"""A4.4 control - score the up-field networks on ONLY the tracks the magnet-down
experiment also kept.

The fiducial cut is the one place the two polarities do not see the same data:
on MagDown 21 / 20 / 15 states have reference trajectories that leave the field
map and are removed, while on MagUp none do (`results/dataset_meta.json`). The
up-field test split therefore contains 15 tracks the down-field one does not,
and those are exactly the hardest ones - soft tracks bending far off axis. A
median taken over 2033 tracks and one taken over 2018 are not the same
measurement, so the up numbers are recomputed here on the common 2018.

Both datasets come from the same ordered row selection, so the down split is the
up split with 15 rows deleted; the two are re-aligned by walking them in order
and matching states to within the (sub-micron to 70 um) rebase difference.

    results/matched_subset.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import glob   # noqa: E402
import json   # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

import use_shared    # noqa: F401,E402
from _shared.evaluate import predict         # noqa: E402
from _shared.model import OneStepNetwork     # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
V2 = os.path.join(HERE, "..", "..", "Block_0_first_pass", "S2b_One_step_network_v2", "results")
TOL = 1.0        # mm: far above the 0.07 mm worst rebase difference, far below
                 # the spacing between distinct tracks


def align(up_S, dn_S):
    """Indices into up_S of the rows that survive in dn_S, in order."""
    keep, j = [], 0
    for i in range(len(up_S)):
        if j < len(dn_S) and np.abs(up_S[i, :2] - dn_S[j, :2]).max() < TOL:
            keep.append(i)
            j += 1
    return np.array(keep), j


def main():
    up = np.load(os.path.join(RES, "frozen_leg_up.npz"))
    dn = np.load(os.path.join(RES, "frozen_leg_down.npz"))
    out = {"tolerance_mm": TOL}

    for split in ("train", "val", "test"):
        idx, matched = align(up["%s_S" % split], dn["%s_S" % split])
        out[split] = {"n_up": int(len(up["%s_S" % split])),
                      "n_down": int(len(dn["%s_S" % split])),
                      "n_matched": int(matched),
                      "all_down_rows_found": bool(matched == len(dn["%s_S" % split]))}
        if split == "test":
            keep = idx

    S = up["test_S"]
    ref = up["test_ref"]
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "up_*.json"))):
        with open(p) as f:
            r = json.load(f)
        ck = os.path.join(RES, r["tag"] + ".pt")
        if not os.path.exists(ck):
            continue
        m = OneStepNetwork(int(up["q"]), up["in_scale"], up["out_scale"],
                           width=r["width"], depth=r["depth"], n_extra=0)
        m.load_state_dict(torch.load(ck, weights_only=True))
        m.eval()
        o = np.asarray(predict(m, torch.tensor(S)))
        e = 1e3 * np.abs(o[:, -1, :2] - ref[:, -1, :2]).max(axis=1)   # the
        # shared metric: the larger of |dx|, |dy| (see _shared/evaluate.py)
        rows.append({"tag": r["tag"], "mode": r["mode"], "seed": r["seed"],
                     "med_all_um": float(np.median(e)),
                     "med_matched_um": float(np.median(e[keep])),
                     "n_all": int(len(e)), "n_matched": int(len(keep))})

    for mode in ("physics", "data"):
        v = [r for r in rows if r["mode"] == mode]
        if not v:
            continue
        a = np.array([r["med_all_um"] for r in v])
        b = np.array([r["med_matched_um"] for r in v])
        out["%s_up" % mode] = {
            "n_seeds": len(v),
            "median_over_seeds_all_tracks_um": float(np.median(a)),
            "range_all_tracks_um": [float(a.min()), float(a.max())],
            "median_over_seeds_matched_um": float(np.median(b)),
            "range_matched_um": [float(b.min()), float(b.max())],
        }
    out["per_run"] = rows
    with open(os.path.join(RES, "matched_subset.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_run"}, indent=1))


if __name__ == "__main__":
    main()
