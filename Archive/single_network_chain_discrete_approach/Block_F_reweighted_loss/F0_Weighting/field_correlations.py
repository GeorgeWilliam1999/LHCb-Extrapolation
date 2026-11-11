#!/usr/bin/env python
"""Provenance for the field-correlation numbers quoted in ../README.md and weighted_loss.py.

Block F does NOT weight the loss by the magnetic field. The reason is measured
here, on Block E's error anatomy (N = 64, q = 2, the 1,452 test tracks,
`E3_Analysis/results/error_anatomy.csv`): along the crossing, does a step's
slope error follow |B|, and does a step's cost at z1 follow |B| or the distance
it still has to travel?

  slope error      the median |dtx| the network adds on that step, against RK6
                   from the same state (column local_tx_med_mrad)
  cost at z1       that slope error times the distance left (carried_to_end_um)
  |B|              the field on the beam axis at the step's midpoint, from the
                   v8r1 map

Spearman rank correlations over the 64 steps, Pearson beside them.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python field_correlations.py
Outputs: results/field_correlations.json
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

import use_shared                                      # noqa: F401
from _shared.reference import make_field

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(HERE, "..", "..", "Block_E_single_network_chain")
ANATOMY = os.path.join(E, "E3_Analysis", "results", "error_anatomy.csv")
TRACKS = os.path.join(E, "E0_Track_dataset", "results", "tracks.npz")


def main():
    rows = list(csv.DictReader(open(ANATOMY)))
    D = np.load(TRACKS)
    fld = make_field(str(D["field"]))
    z = np.array([float(r["z_mm"]) for r in rows])
    dz = float(z[1] - z[0])
    zmid = z + dz / 2
    Bx, By, Bz = fld(np.zeros_like(zmid), np.zeros_like(zmid), zmid)
    B = np.sqrt(np.asarray(Bx) ** 2 + np.asarray(By) ** 2 + np.asarray(Bz) ** 2)
    slope = np.array([float(r["local_tx_med_mrad"]) for r in rows])
    lever = np.array([float(r["lever_mm"]) for r in rows])
    cost = np.array([float(r["carried_to_end_um"]) for r in rows])

    def both(a, b):
        return dict(spearman=float(spearmanr(a, b).statistic), pearson=float(pearsonr(a, b)[0]))

    out = dict(
        source=os.path.relpath(ANATOMY, HERE), n_steps=int(len(rows)), dz_mm=dz,
        B_on_axis_T=dict(min=float(B.min()), max=float(B.max())),
        slope_error_vs_B=both(slope, B),
        cost_at_z1_vs_lever=both(cost, lever),
        cost_at_z1_vs_B=both(cost, B),
        reading="the per-step slope error is largest where |B| is smallest (negative correlation), "
                "and a step's cost at z1 follows the distance left, not the field")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "field_correlations.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
