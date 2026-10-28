"""Before submission: build each run of runs.txt, print its key, run the
package's checks before training and, for the bias-term loss, its lambda.
Writes nothing.

    check_before_submission.py      from RK_Pinn_module, with PYTHONPATH=src
"""
import os
import sys

from rkpinn.run_record.configuration import read_configuration
from rkpinn.run_record.store import Store
from rkpinn.training.round_trainer import build_run, checks_before_training

HERE = os.path.dirname(os.path.abspath(__file__))
store = Store(os.environ.get("RKPINN_STORE", "/data/bfys/gscriven/rkpinn_store"))
for line in open(os.path.join(HERE, "runs.txt")):
    name = line.split()[0]
    run = build_run(read_configuration(os.path.join(HERE, "..", "configurations", name)), store)
    measured = checks_before_training(run)
    loss = run.constants["loss"]
    print(name, run.key, "exists" if os.path.isdir(os.path.join(store.location, "runs", run.key))
          else "new", measured,
          {k: loss[k] for k in ("pooled_at_round_one", "bias_at_round_one",
                                "weight_of_the_bias_term") if k in loss}, flush=True)
    if run.configuration["training"]["protocol"]["name"].endswith("residual_draws"):
        import time
        started = time.time()
        second = run.protocol.states_of_round(2, run.configuration["seed"], run.tracks,
                                              run.layout, run.network, run.target)
        print("   round 2 draw:", second.states.shape, second.source,
              "%.0f s" % (time.time() - started), flush=True)
