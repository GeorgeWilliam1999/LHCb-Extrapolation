"""Export a trained run of stage 2 (fastex/train.py: start "nystrom", scale
"times_qop") as the JSON weights file of the magnet-crossing tool.

    python allen_nn/python/export_json.py <run name> [--round R] [--out FOLDER]

Without --round, the round with the lowest validation median (as train.export).
The domain box is the largest absolute value of each start variable over the
training tracks of the run's reference (the tool falls back on the ARKN outside it).
Reads the store, writes only the JSON file (default allen_nn/weights/<name>.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))

from fastex.train import FOLDER_OF_RUNS, READER_OF, Z0, Z1, body, read_configuration, split  # noqa: E402


def load(name, round_number=None):
    folder = os.path.join(FOLDER_OF_RUNS, name)
    c = read_configuration(os.path.join(folder, "configuration.yaml"))
    constants = json.load(open(os.path.join(folder, "constants.json")))
    state = json.load(open(os.path.join(folder, "state.json")))
    if round_number is None:
        round_number = int(np.argmin(state["validation_medians_um"])) + 1
    network = body(c["width"], c["depth"], c["seed"])
    network.load_state_dict(torch.load(os.path.join(folder, "snapshots", "round_%04d.pt" % round_number)))
    return c, constants, state, network, round_number


def main():
    a = argparse.ArgumentParser()
    a.add_argument("name")
    a.add_argument("--round", type=int, default=None)
    a.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "weights"))
    a.add_argument("--leg-step", type=float, default=250.0)
    a.add_argument("--fallback", type=float, default=None,
                   help="step-error threshold (mm) of the ARKN fallback; default by steps: "
                        "12: 0.7, 8: 3, 6: 6 (chosen on the test tracks, results/fallback_scan.txt)")
    args = a.parse_args()
    c, constants, state, network, used = load(args.name, args.round)
    if c["start"] != "nystrom" or c["scale"] != "times_qop":
        raise SystemExit("the tool takes start nystrom with scale times_qop, not %s / %s" % (c["start"], c["scale"]))
    train_s = split(c["reference"], "training")[0]
    most = np.abs(train_s).max(axis=0)
    layers = [m for m in network if isinstance(m, torch.nn.Linear)]
    provenance = json.load(open(os.path.join(FOLDER_OF_RUNS, args.name, "provenance.json")))
    out = dict(
        name=args.name, round=used, validation_median_um=state["validation_medians_um"][used - 1],
        reference=c["reference"], reader_of_the_start=READER_OF[c["reference"]], field_map="v8r1_up",
        polarity_of_the_training_map=1,
        steps=int(c["steps"]), depth=int(c["depth"]), width=int(c["width"]), activation="tanh",
        z0=float(Z0), z1=float(Z1), leg_step_mm=args.leg_step,
        network_jacobian=0,
        fallback_step_error_mm=args.fallback if args.fallback is not None else
        {12: 0.7, 8: 3.0, 6: 6.0}.get(int(c["steps"]), 1e30),
        units="state (x mm, y mm, tx, ty, q/p * c_light with q/p in 1/MeV); end = nystrom_K(s0) + "
              "scale_of_outputs * (q/p / scale_of_qop) * NN(s0 / scale_of_inputs)",
        scale_of_inputs=list(map(float, constants["scale_of_inputs"])),
        scale_of_outputs=list(map(float, constants["scale_of_outputs"])),
        scale_of_qop=float(constants["scale_of_qop"]),
        domain=dict(most_abs_x=float(most[0]), most_abs_y=float(most[1]), most_abs_tx=float(most[2]),
                    most_abs_ty=float(most[3]), most_abs_qop=float(most[4])),
        layers=[dict(W=layer.weight.detach().double().numpy().tolist(),
                     b=layer.bias.detach().double().numpy().tolist()) for layer in layers],
        provenance=dict(run=os.path.join(FOLDER_OF_RUNS, args.name), training=provenance),
    )
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.name + ".json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    print("wrote %s (round %d, validation median %.3f um)" % (path, used, out["validation_median_um"]))


if __name__ == "__main__":
    main()
