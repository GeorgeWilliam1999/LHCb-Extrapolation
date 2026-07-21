#!/usr/bin/env python
"""Why does the physics loss plateau ~20x higher on the v2 (official) states,
when the data twin reaches the same loss on both?

The one structural asymmetry between the losses: the physics loss evaluates the
field at the network's stage positions (and differentiates through it); the data
loss never touches the field. So this checks (a) whether the physics loss is
carried by a tail of pathological states, and (b) where the reference paths sit
relative to the field-map bounds, for v1 and v2.

Run:  /data/bfys/gscriven/conda/envs/TE/bin/python diagnose_physics_gap.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, "..", "One_step_network")
sys.path.insert(0, V1)
sys.path.insert(0, os.path.join(HERE, "..", "Baseline_data_exploration"))
sys.path.insert(0, os.path.join(HERE, "..", "Simple_first_pass"))

from irk import tableau  # noqa: E402
from model import LHCbRates, OneStepNetwork, reconstruction_residuals  # noqa: E402
from reference_card import FIELD  # noqa: E402

torch.set_default_dtype(torch.float64)
rates = LHCbRates()

lo = FIELD.min
hi = [FIELD.min[i] + (FIELD.N[i] - 1) / FIELD.invD[i] for i in range(3)]
print("field map bounds [mm]: x [%.0f, %.0f]  y [%.0f, %.0f]  z [%.0f, %.0f]"
      % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))

for name, res in (("v1 (our sample)", os.path.join(V1, "results")),
                  ("v2 (official)", os.path.join(HERE, "results"))):
    d = np.load(os.path.join(res, "frozen_leg_data.npz"))
    Q = int(d["q"]); DZ = float(d["z1"]) - float(d["z0"])
    c, A_np, b_np = tableau(Q)
    A, b = torch.tensor(A_np), torch.tensor(b_np)
    ZN = torch.tensor(d["znodes"])
    S = torch.tensor(d["train_S"])
    ref = d["train_ref"]

    print("\n=== %s ===  (%d train states)" % (name, len(S)))
    print("  reference path |x|: p99 %.0f mm, max %.0f mm | |y|: p99 %.0f, max %.0f"
          % (np.quantile(np.abs(ref[:, :, 0]), 0.99), np.abs(ref[:, :, 0]).max(),
             np.quantile(np.abs(ref[:, :, 1]), 0.99), np.abs(ref[:, :, 1]).max()))
    out = (((ref[:, :, 0] < lo[0]) | (ref[:, :, 0] > hi[0])
            | (ref[:, :, 1] < lo[1]) | (ref[:, :, 1] > hi[1])).any(axis=1))
    print("  reference paths leaving the field map: %d of %d" % (out.sum(), len(S)))

    for seed in (0, 1, 2):
        ck = os.path.join(res, "one_step_physics_seed%d.pt" % seed)
        if not os.path.exists(ck):
            continue
        m = OneStepNetwork(Q, d["in_scale"], d["out_scale"])
        m.load_state_dict(torch.load(ck, weights_only=True))
        with torch.no_grad():
            r = reconstruction_residuals(m, rates, S, DZ, ZN, A, b).numpy()
        per_state = (r ** 2).mean(axis=(1, 2))
        tot = per_state.sum()
        order = np.argsort(per_state)[::-1]
        top1 = per_state[order[:max(1, len(S) // 100)]].sum() / tot
        top10 = per_state[order[:len(S) // 10]].sum() / tot
        print("    physics seed %d: loss %.3e | median per-state %.3e | top 1%% of "
              "states carry %.0f%%, top 10%% carry %.0f%%"
              % (seed, per_state.mean(), np.median(per_state), 100 * top1, 100 * top10))
