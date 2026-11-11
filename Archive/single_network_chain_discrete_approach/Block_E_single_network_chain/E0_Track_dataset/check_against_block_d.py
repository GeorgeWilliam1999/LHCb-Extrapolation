#!/usr/bin/env python
"""E0 gate - the Block E tracks are Block D's particles, on a finer plane grid.

Block E reuses Block D's selection, and its comparisons with Block D (the
exact-scheme runs at N = 64 and 128, the one-network-per-step chains) are only
fair on the same particles. This gate compares `results/tracks.npz` with
Block D's `crossing_particles.npz`:

  1. validation and test: the same particles in the same order (event and
     particle keys), and bit-identical start states, real UT and SciFi states,
     planes of those states, momenta;
  2. validation and test: the RK6 states on the 129 planes both grids share
     (Block D plane k = Block E plane 2k) agree within 1e-3 um in position and
     1e-6 mrad in slope. They cannot be bit-identical: RK6 restarts its 0.1 mm
     steps at every stored plane, and Block E stores twice as many;
  3. training: Block D's 2,000 training particles are among Block E's 11,567,
     with bit-identical start states and the same agreement on the shared
     planes.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python check_against_block_d.py
Output:  results/check_against_block_d.json, prints PASS or FAIL
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E0 = os.path.join(HERE, "results", "tracks.npz")
D0 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "multi_network_chain_discrete_approach",
                                  "Block_D_fixed_step_crossing", "D0_Crossing_dataset", "results",
                                  "crossing_particles.npz"))
POS_TOL_UM = 1e-3
SLOPE_TOL_MRAD = 1e-6
EXACT = ("S0", "S_pre", "S_post", "z_pre", "z_post", "P", "ETA", "PID", "PBAND")


def diff_on_shared_planes(e_truth, d_truth):
    shared = e_truth[:, ::2]
    assert shared.shape == d_truth.shape, (shared.shape, d_truth.shape)
    pos = float(np.abs(shared[:, :, :2] - d_truth[:, :, :2]).max() * 1e3)
    slope = float(np.abs(shared[:, :, 2:4] - d_truth[:, :, 2:4]).max() * 1e3)
    qop = float(np.abs(shared[:, :, 4] - d_truth[:, :, 4]).max())
    return {"pos_max_um": pos, "slope_max_mrad": slope, "qop_max_abs": qop,
            "ok": pos <= POS_TOL_UM and slope <= SLOPE_TOL_MRAD and qop == 0.0}


def main():
    e = np.load(E0)
    d = np.load(D0)
    out = {"block_e": E0, "block_d": D0, "tolerances": {"pos_um": POS_TOL_UM, "slope_mrad": SLOPE_TOL_MRAD}}
    ok = True
    out["planes"] = {"block_e": int(len(e["planes"])), "block_d": int(len(d["planes"])),
                     "shared_identical": bool(np.array_equal(e["planes"][::2], d["planes"]))}
    ok &= out["planes"]["shared_identical"]
    for s in ("val", "test"):
        r = {"n_block_e": int(len(e["%s_S0" % s])), "n_block_d": int(len(d["%s_S0" % s]))}
        same_keys = (r["n_block_e"] == r["n_block_d"]
                     and np.array_equal(e["%s_EVT" % s], d["%s_EVT" % s])
                     and np.array_equal(e["%s_MCKEY" % s], d["%s_MCKEY" % s]))
        r["same_particles_same_order"] = bool(same_keys)
        r["bit_identical"] = {k: bool(same_keys and np.array_equal(e["%s_%s" % (s, k)], d["%s_%s" % (s, k)]))
                              for k in EXACT}
        r["truth_on_shared_planes"] = diff_on_shared_planes(e["%s_truth" % s], d["%s_truth" % s]) if same_keys else None
        tz = np.abs(e["%s_truth_zpost" % s] - d["%s_truth_zpost" % s]) if same_keys else None
        r["truth_zpost_pos_max_um"] = float(tz[:, :2].max() * 1e3) if same_keys else None
        r_ok = (same_keys and all(r["bit_identical"].values()) and r["truth_on_shared_planes"]["ok"]
                and r["truth_zpost_pos_max_um"] <= POS_TOL_UM)
        r["ok"] = bool(r_ok)
        ok &= r_ok
        out[s] = r

    ekey = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(e["train_EVT"], e["train_MCKEY"]))}
    idx = [ekey.get((int(a), int(b)), -1) for a, b in zip(d["train_EVT"], d["train_MCKEY"])]
    found = int(sum(i >= 0 for i in idx))
    r = {"n_block_e": int(len(e["train_S0"])), "n_block_d": int(len(d["train_S0"])),
         "block_d_found_in_block_e": found}
    if found == len(idx):
        idx = np.array(idx)
        r["S0_bit_identical"] = bool(np.array_equal(e["train_S0"][idx], d["train_S0"]))
        r["truth_on_shared_planes"] = diff_on_shared_planes(e["train_truth"][idx], d["train_truth"])
        r["ok"] = bool(r["S0_bit_identical"] and r["truth_on_shared_planes"]["ok"])
    else:
        r["ok"] = False
    ok &= r["ok"]
    out["train"] = r
    out["PASS"] = bool(ok)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "check_against_block_d.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("E0 GATE PASS" if ok else "E0 GATE FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
