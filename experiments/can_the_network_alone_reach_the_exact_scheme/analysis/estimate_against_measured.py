"""Does the label-free estimate of the loss predict the measured endpoint error?

For an existing run, on the test tracks along the network's own chain: the
carried-defect estimate of losses/pooled_with_a_carried_defect_term.py, summed
over the steps of each track, against the measured (network endpoint - exact
chain endpoint). The chain and the exact chain are read from the cache of
experiments/how_does_the_stage_residual_become_the_endpoint_error.

    PYTHONNOUSERSITE=1 PYTHONPATH=../../../RK_Pinn_module/src \
        /data/bfys/gscriven/conda/envs/TE/bin/python estimate_against_measured.py <run key> ...

Reads the store and that cache. Writes results/estimate_against_measured.csv.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch

from rkpinn.losses.pooled_with_a_carried_defect_term import (
    carried_to_the_last_plane, defect_of_each_step)
from rkpinn.losses.stage_residual import stage_residual
from rkpinn.run_record.configuration import read_configuration
from rkpinn.run_record.store import Store
from rkpinn.training.checkpoints import RunFolder
from rkpinn.training.round_trainer import build_run

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = "/data/bfys/gscriven/rkpinn_store"
CACHE = "/data/bfys/gscriven/rkpinn_analysis_cache/how_does_the_stage_residual_become_the_endpoint_error"
BANDS = [("all momenta", 0, np.inf), ("< 3 GeV", 0, 3), ("3-8 GeV", 3, 8), ("8-20 GeV", 8, 20),
         ("20-50 GeV", 20, 50), ("> 50 GeV", 50, np.inf)]


def main(keys):
    store = Store(STORE)
    rows = []
    for key in keys:
        folder = RunFolder(store, key)
        run = build_run(read_configuration(folder.path("configuration.yaml")), store,
                        folder.constants())
        d = np.load(os.path.join(CACHE, "%s.npz" % key))
        assert int(d["round"]) == int(folder.state()["rounds_done"])
        run.network.load_state_dict(folder.weights_of_round(int(d["round"])))
        run.network.eval()
        chain = d["states"]                                   # (n, N + 1, 5)
        n, N = chain.shape[0], chain.shape[1] - 1
        states = torch.as_tensor(chain[:, :-1].reshape(-1, 5))
        planes = torch.as_tensor(np.tile(np.asarray(run.layout.start_planes_mm), n))
        with torch.no_grad():
            track = run.network.predict(states, planes)
            residual = stage_residual(track, run.network.equation_of_motion, "stages")
            defect = defect_of_each_step(residual, track, run.network.equation_of_motion)
            carried = carried_to_the_last_plane(defect, planes, run.layout.step_length_mm,
                                                run.layout.last_plane_mm)
        estimate = carried.reshape(n, N, 2).sum(dim=1).numpy() * 1e3          # um
        measured = (chain[:, -1, :2] - d["exact_chain_end"][:, :2]) * 1e3
        against_reference = (chain[:, -1, :2] - d["reference_end"][:, :2]) * 1e3
        left = measured - estimate
        p = d["momentum_gev"]
        for label, lo, hi in BANDS:
            m = (p >= lo) & (p < hi)
            radial = lambda e: float(np.median(np.hypot(e[m, 0], e[m, 1])))
            rows.append({
                "run": key, "steps": N, "band": label, "tracks": int(m.sum()),
                "measured_against_the_exact_chain_um": radial(measured),
                "estimate_um": radial(estimate),
                "measured_minus_estimate_um": radial(left),
                "against_the_reference_um": radial(against_reference),
                "against_the_reference_after_subtracting_the_estimate_um":
                    radial(against_reference - estimate),
                "correlation_x": float(np.corrcoef(estimate[m, 0], measured[m, 0])[0, 1]),
                "correlation_y": float(np.corrcoef(estimate[m, 1], measured[m, 1])[0, 1]),
            })
            print(rows[-1], flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "estimate_against_measured.csv"), "w", newline="") as h:
        w = csv.DictWriter(h, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    main(sys.argv[1:])
