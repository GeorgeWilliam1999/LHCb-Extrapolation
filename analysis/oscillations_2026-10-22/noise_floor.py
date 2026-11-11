"""Sampling-noise floor of the training loss, with no training (read-only).

For a fixed snapshot: the loss on many fresh draws of 32,000 states, drawn
exactly as the trainer draws them (training_protocols.RoundsOnOwnPredictions):
from the network's own chain and from the reference states. Also checks that
the trainer's recorded numbers are reproduced (loss_last of the final round,
its validation error), and evaluates late snapshots on one large fixed set to
see the loss without draw noise.

usage: noise_floor.py <run_key> <n_draws> <rounds_on_fixed_set (e.g. 30)>
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
from rkpinn.training.round_trainer import build_run, loss_of_a_round, validation_of
from rkpinn.training.training_protocols import StatesOfARound
from rkpinn.track_data.draw_training_states import draw_training_states, generator_of_a_round

STORE = "/data/bfys/gscriven/rkpinn_store"
HERE = os.path.dirname(os.path.abspath(__file__))
SCORE = "validation_endpoint_error_larger_of_x_and_y_median_micrometres"


def main():
    key, n_draws, n_fixed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    store = Store(STORE)
    folder = RunFolder(store, key)
    configuration = read_configuration(folder.path("configuration.yaml"))
    run = build_run(configuration, store, folder.constants())
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))  # build_run sets 1
    rounds = folder.rounds()
    R = int(rounds[-1]["round"])
    seed = configuration["seed"]
    n_states = configuration["training"]["protocol"]["states_per_round"]
    tracks, layout = run.tracks, run.layout
    start_planes = tracks.start_planes_mm(layout.number_of_steps)
    on = tracks.planes_a_step_starts_on(layout.number_of_steps)
    reference_pool = tracks.training.reference_states_on_planes[:, on]

    def chain_pool():
        with torch.no_grad():
            whole = run.network.whole_track(tracks.training.start_state, layout)
        return np.asarray(whole.input_states)

    def loss_on(pool, generator):
        d = draw_training_states(pool, start_planes, n_states, generator)
        s = StatesOfARound(states=d.states, start_planes_mm=d.start_planes_mm, labels=None,
                           source="")
        with torch.no_grad():
            return float(loss_of_a_round(run, s)())

    out = {"key": key, "final_round": R}
    t0 = time.time()
    # reproduce loss_last(R): draw R from the chain of snapshot R-1, loss of snapshot R
    run.network.load_state_dict(folder.weights_of_round(R - 1))
    pool_prev = chain_pool()
    run.network.load_state_dict(folder.weights_of_round(R))
    out["reproduced_loss_last"] = loss_on(pool_prev, generator_of_a_round(seed, R))
    out["recorded_loss_last"] = float(rounds[-1]["loss_last"])
    out["reproduced_score"] = validation_of(run)[SCORE]
    out["recorded_score"] = float(rounds[-1][SCORE])
    print("reproduction", out, "%.0f s" % (time.time() - t0), flush=True)

    pool_now = chain_pool()
    # the draw the next round would get (generator of round R+1, chain of snapshot R)
    out["loss_next_round_draw"] = loss_on(pool_now, generator_of_a_round(seed, R + 1))
    chain = [loss_on(pool_now, np.random.default_rng([seed, 100000 + g])) for g in range(n_draws)]
    ref = [loss_on(reference_pool, np.random.default_rng([seed, 200000 + g])) for g in range(n_draws)]
    # same states as round R, but taken from the new chain (isolates the chain shift)
    out["loss_round_R_indices_new_chain"] = loss_on(pool_now, generator_of_a_round(seed, R))
    out["chain_draws"], out["reference_draws"] = chain, ref
    for name, v in (("chain", chain), ("reference", ref)):
        lv = np.log(v)
        out[name + "_median"] = float(np.median(v))
        out[name + "_std_log"] = float(np.std(lv, ddof=1))
        out[name + "_min_max"] = [float(min(v)), float(max(v))]
    print("draws done %.0f s" % (time.time() - t0), flush=True)

    # late snapshots on one large fixed set: 4 x 32,000 reference-drawn states
    big = [draw_training_states(reference_pool, start_planes, n_states,
                                np.random.default_rng([seed, 300000 + g])) for g in range(4)]
    big = StatesOfARound(states=np.concatenate([b.states for b in big]),
                         start_planes_mm=np.concatenate([b.start_planes_mm for b in big]),
                         labels=None, source="")
    fixed = []
    for row in rounds[-n_fixed:]:
        r = int(row["round"])
        run.network.load_state_dict(folder.weights_of_round(r))
        with torch.no_grad():
            fixed.append({"round": r, "loss_fixed_set": float(loss_of_a_round(run, big)()),
                          "score": float(row[SCORE]), "loss_first": float(row["loss_first"]),
                          "loss_last": float(row["loss_last"])})
    out["fixed_set"] = fixed
    lf = np.log([x["loss_fixed_set"] for x in fixed])
    sc = np.log([x["score"] for x in fixed])
    out["fixed_set_std_dlog"] = float(np.std(np.diff(lf), ddof=1))
    out["corr_dlog_fixedloss_dlog_score"] = float(np.corrcoef(np.diff(lf), np.diff(sc))[0, 1])
    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "noise_floor_%s.json" % key), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("fixed_set",)}, indent=1))


if __name__ == "__main__":
    main()
