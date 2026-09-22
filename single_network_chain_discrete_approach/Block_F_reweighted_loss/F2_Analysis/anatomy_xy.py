#!/usr/bin/env python
"""F2 - the error anatomy with BOTH slopes, for any run folder.

E3's `error_anatomy.py` carries the lever-arm sum for the x slope only, which
explained 72 percent of Block E's endpoint error but only 38 percent of Block
F's, because Block F's remaining error is now mostly in y. This is the same
decomposition with dx + dtx*lever and dy + dty*lever combined radially, on the
1,452 test tracks, against RK6 taken from the SAME state the network was given
at every step.

Run:     PYTHONNOUSERSITE=1 python anatomy_xy.py --run <run folder> --label <name>
Outputs: results/anatomy_xy_<label>.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse   # noqa: E402
import json       # noqa: E402
import sys        # noqa: E402
import time       # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(HERE, "..", "..", "Block_E_single_network_chain")
sys.path.insert(0, os.path.join(E, "E1_Network_grid"))
import use_shared   # noqa: E402,F401
from _shared.reference import make_field, rk6_rows   # noqa: E402
from chain_network import load_network, step_outputs  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
TRACKS = os.path.join(E, "E0_Track_dataset", "results", "tracks.npz")
BAND = (10.0, 50.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args(argv)
    D = np.load(TRACKS)
    Z0, Z1, L, n_max = float(D["z0"]), float(D["z1"]), float(D["L"]), int(D["n_max"])
    sc = json.load(open(os.path.join(a.run, "scale.json")))
    N = int(sc["N"])
    dz = L / N
    fld = make_field(str(D["field"]))
    model = load_network(a.run, fld)
    S0, truth, P = np.asarray(D["test_S0"]), np.asarray(D["test_truth"]), np.asarray(D["test_P"])
    n = len(S0)
    band = (P >= BAND[0]) & (P < BAND[1])
    t0 = time.time()
    S = S0.copy()
    lx, ly, ltx, lty = (np.empty((N, n)) for _ in range(4))
    for k in range(N):
        z = Z0 + k * dz
        out = step_outputs(model, S, np.full(n, z))[:, -1, :]
        ref = rk6_rows(S, z, z + dz, step=1.0, field=fld)
        lx[k], ly[k] = (out[:, 0] - ref[:, 0]) * 1e3, (out[:, 1] - ref[:, 1]) * 1e3        # um
        ltx[k], lty[k] = (out[:, 2] - ref[:, 2]) * 1e3, (out[:, 3] - ref[:, 3]) * 1e3      # mrad
        S = np.concatenate([out, S[:, 4:5]], axis=1)
    final = np.hypot(S[:, 0] - truth[:, n_max, 0], S[:, 1] - truth[:, n_max, 1]) * 1e3
    lever = (Z1 - (Z0 + (np.arange(N) + 1) * dz))[:, None]                                 # mm
    px = (lx + ltx * lever).sum(axis=0)          # um: mrad*mm = um
    py = (ly + lty * lever).sum(axis=0)
    pred_xy = np.hypot(px, py)
    pred_x_only = np.abs((ltx * lever).sum(axis=0))
    pos_only = np.hypot(lx.sum(axis=0), ly.sum(axis=0))

    def med(v, m=None):
        return float(np.median(v if m is None else v[m]))

    out = dict(
        label=a.label, run=os.path.relpath(a.run, HERE), N=N, q=int(sc["q"]), n_tracks=int(n),
        final_med_um=med(final), final_band_med_um=med(final, band),
        local_step=dict(x_um=med(np.abs(lx)), y_um=med(np.abs(ly)), tx_mrad=med(np.abs(ltx)), ty_mrad=med(np.abs(lty)),
                        radial_um=med(np.hypot(lx, ly))),
        local_step_band=dict(x_um=med(np.abs(lx[:, band])), y_um=med(np.abs(ly[:, band])),
                             tx_mrad=med(np.abs(ltx[:, band])), ty_mrad=med(np.abs(lty[:, band]))),
        predicted_endpoint=dict(both_slopes_med_um=med(pred_xy), x_slope_only_med_um=med(pred_x_only),
                                positions_only_med_um=med(pos_only)),
        explained=dict(both_slopes=med(pred_xy) / med(final), x_slope_only=med(pred_x_only) / med(final),
                       positions_only=med(pos_only) / med(final)),
        x_part_med_um=med(np.abs(px)), y_part_med_um=med(np.abs(py)),
        coherence=dict(tx=med(np.abs(ltx.sum(0)) / np.abs(ltx).sum(0)), ty=med(np.abs(lty.sum(0)) / np.abs(lty).sum(0)),
                       independent=float(1 / np.sqrt(N))),
        wall_s=round(time.time() - t0, 1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "anatomy_xy_%s.json" % a.label), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
