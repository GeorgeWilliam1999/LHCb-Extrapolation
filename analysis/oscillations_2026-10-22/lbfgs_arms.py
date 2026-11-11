"""One reduced round of L-BFGS from a fixed snapshot, under one setting (read-only).

Start: snapshot of round R-1. Draw: either the exact draw of round R (the
generator of round R on the chain of snapshot R-1) or another draw from the same
chain. Optimiser: the trainer's LbfgsRestarts, with lr / history overridable by
a local subclass (the package is not edited). After every restart the loss and
the validation error are recorded.

usage: lbfgs_arms.py <run_key> <arm_name> <restarts> <draw: round|alt:N> <lr> <history>
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/data/bfys/gscriven/LHCb_Extrapolation_Project/RK_Pinn_module/src")
from rkpinn.run_record.configuration import read_configuration
from rkpinn.run_record.store import Store
from rkpinn.training.checkpoints import RunFolder
from rkpinn.training.optimiser import LbfgsRestarts
from rkpinn.training.round_trainer import build_run, loss_of_a_round, validation_of
from rkpinn.training.training_protocols import StatesOfARound
from rkpinn.track_data.draw_training_states import draw_training_states, generator_of_a_round

STORE = "/data/bfys/gscriven/rkpinn_store"
HERE = os.path.dirname(os.path.abspath(__file__))
SCORE = "validation_endpoint_error_larger_of_x_and_y_median_micrometres"


class LbfgsWithLr(LbfgsRestarts):
    def __init__(self, lr, **kw):
        super().__init__(**kw)
        self.lr = lr

    def _new_lbfgs(self):
        return torch.optim.LBFGS(
            self._parameters, lr=self.lr, max_iter=self.iterations_per_restart,
            history_size=self.history, tolerance_grad=self.tolerance_of_the_gradient,
            tolerance_change=self.tolerance_of_the_change, line_search_fn=self.line_search)


def main():
    key, arm, k, draw, lr, history = (sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4],
                                      float(sys.argv[5]), int(sys.argv[6]))
    store = Store(STORE)
    folder = RunFolder(store, key)
    configuration = read_configuration(folder.path("configuration.yaml"))
    run = build_run(configuration, store, folder.constants())
    torch.set_num_threads(1)
    R = int(folder.rounds()[-1]["round"])
    seed = configuration["seed"]
    n_states = configuration["training"]["protocol"]["states_per_round"]
    tracks, layout = run.tracks, run.layout
    start_planes = tracks.start_planes_mm(layout.number_of_steps)
    run.network.load_state_dict(folder.weights_of_round(R - 1))
    with torch.no_grad():
        pool = np.asarray(run.network.whole_track(tracks.training.start_state, layout).input_states)
    gen = (generator_of_a_round(seed, R) if draw == "round"
           else np.random.default_rng([seed, 900000 + int(draw.split(":")[1])]))
    d = draw_training_states(pool, start_planes, n_states, gen)
    states = StatesOfARound(states=d.states, start_planes_mm=d.start_planes_mm, labels=None,
                            source="")
    loss_now = loss_of_a_round(run, states)
    settings = dict(configuration["training"]["optimiser"])
    settings.pop("name")
    settings["history"] = history
    opt = LbfgsWithLr(lr, **settings)
    opt.start_round(run.network.parameters())
    rows = [{"restart": 0, "loss": float(loss_now().detach()), "score": validation_of(run)[SCORE]}]
    print(arm, rows[-1], flush=True)
    for i in range(1, k + 1):
        rec = opt.one_restart(loss_now)
        rows.append({"restart": i, "loss": rec["loss_after"], "score": validation_of(run)[SCORE],
                     "iterations": rec["iterations"], "evaluations": rec["evaluations"],
                     "stopped_early": rec["stopped_early"], "seconds": rec["seconds"]})
        print(arm, rows[-1], flush=True)
    out = {"key": key, "arm": arm, "round": R, "start_snapshot": R - 1, "draw": draw, "lr": lr,
           "history": history, "rows": rows}
    with open(os.path.join(HERE, "arm_%s_%s.json" % (key, arm)), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
