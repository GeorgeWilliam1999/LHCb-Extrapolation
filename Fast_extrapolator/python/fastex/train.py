"""The trainer of stage 2: a start plus a network correction.

    end state = start(start state) + scale * network(start state / scale of inputs)

    start     none            the network gives the end state itself
              straight_line   no field read
              nystrom         the Nystrom integrator with `steps` equal steps; it
                              reads the map of the reference (cubic, or as LHCb reads it)
    scale     fixed           four numbers
              times_qop       four numbers times q/p of the track over its scale
    loss      huber           smooth Huber loss of each component; its threshold
                              is the median error of the round before, and never
                              below 2 um (1e-6 for a slope)
              squared         the squared error over a fixed divisor

A configuration is one YAML file. A run is a folder of the store of this
experiment, <store>/runs/<name>: the configuration, constants.json (the scales,
measured once), rounds.csv, a snapshot of the weights a round, state.json and
provenance.json. A run that is started again goes on from its last round.

A round is `restarts` restarts of L-BFGS (`iterations` each) on all training
tracks, in double precision. After a round the validation tracks are scored
(radial error on the last plane). The run ends when the best validation median
of the last `window` rounds is not `fall_needed` (a fraction) below the best
before them, or at `rounds_at_most`. A run that is started again with other
values of these three settings is extended. The test tracks are not read here.

The start of every track is computed once by the compiled program in single
precision, as the compiled method computes it.

    python -m fastex.train <configuration.yaml> [--rounds N]
    python -m fastex.train --starts <configuration.yaml> ...   the starts of the integrators, on a login node
    python -m fastex.train --export <name> [--round R]     the correction for the compiled program
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from fastex import FOLDER_OF_EXPORTS, STORE_OF_THIS_EXPERIMENT
from fastex.compiled import run as compiled_run
from fastex.compiled import summary
from fastex.correction import FOLDER_OF_CORRECTIONS

Z0, Z1 = 2648.2, 7826.0
READER_OF = {"cubic": "cubic", "linear": "rec"}
FLOOR_OF_THE_THRESHOLD = np.array([2e-3, 2e-3, 1e-6, 1e-6])
FOLDER_OF_RUNS = os.path.join(STORE_OF_THIS_EXPERIMENT, "runs")
KEYS = {"name", "reference", "start", "steps", "width", "depth", "scale", "loss", "seed",
        "restarts", "iterations", "rounds_at_most", "window", "fall_needed"}
# The settings of the stopping rule. A run can be started again with other
# values of these: that extends it, and the change is written in its state.
STOPPING = ("rounds_at_most", "window", "fall_needed")


def read_configuration(path):
    with open(path) as handle:
        c = yaml.safe_load(handle)
    unknown, missing = set(c) - KEYS, KEYS - set(c)
    if unknown or missing:
        raise ValueError("configuration %s: unknown %s, missing %s" % (path, sorted(unknown), sorted(missing)))
    if c["reference"] not in READER_OF or c["start"] not in ("none", "straight_line", "nystrom") \
            or c["scale"] not in ("fixed", "times_qop") or c["loss"] not in ("huber", "squared"):
        raise ValueError("configuration %s names a choice that does not exist" % path)
    if c["start"] == "none" and c["scale"] == "times_qop":
        raise ValueError("with no start the end state is not zero at q/p = 0; the scale is fixed")
    return c


def split(reference, name):
    folder = os.path.join(FOLDER_OF_EXPORTS, "tracks_" + reference, "splits", name)
    return (np.fromfile(os.path.join(folder, "start_state.f64")).reshape(-1, 5),
            np.fromfile(os.path.join(folder, "reference_end_state.f64")).reshape(-1, 5)[:, :4],
            np.fromfile(os.path.join(folder, "momentum_gev.f64")))


def start_of(states, c, split_name=None):
    if c["start"] == "none":
        return np.zeros((len(states), 4))
    if c["start"] == "straight_line":
        length = Z1 - Z0
        return np.stack([states[:, 0] + length * states[:, 2], states[:, 1] + length * states[:, 3],
                         states[:, 2], states[:, 3]], 1)
    # The start of an integrator is kept in the store, <exports>/starts: a farm
    # node has no compiled program, so the file is made on a login node first
    # (python -m fastex.train --starts <configuration> ...).
    if split_name is not None:
        kept = os.path.join(FOLDER_OF_EXPORTS, "starts",
                            "%s_nystrom_%d_%s.f64" % (c["reference"], c["steps"], split_name))
        if os.path.exists(kept):
            return np.fromfile(kept).reshape(-1, 4)
    start = compiled_run("nystrom:%d" % c["steps"], states, Z0, Z1, reader=READER_OF[c["reference"]],
                         precision="float")[0][:, :4]
    if split_name is not None:
        os.makedirs(os.path.dirname(kept), exist_ok=True)
        start.tofile(kept)
    return start


def body(width, depth, seed):
    torch.manual_seed(int(seed))
    layers, n = [], 5
    for _ in range(depth):
        layers += [torch.nn.Linear(n, width), torch.nn.Tanh()]
        n = width
    layers.append(torch.nn.Linear(n, 4))
    return torch.nn.Sequential(*layers).double()


def spread(values):
    """A robust spread of each column: 1.4826 times the median distance from the median."""
    return 1.4826 * np.median(np.abs(values - np.median(values, axis=0)), axis=0)


def constants_of(c, states, target):
    scale_of_inputs = states.std(axis=0)
    scale_of_qop = float(scale_of_inputs[4])
    if c["scale"] == "times_qop":
        scale_of_outputs = spread(target / (states[:, 4:5] / scale_of_qop))
    else:
        scale_of_outputs = np.maximum(spread(target), target.std(axis=0) * 1e-3)
    return dict(scale_of_inputs=scale_of_inputs.tolist(), scale_of_outputs=scale_of_outputs.tolist(),
                scale_of_qop=scale_of_qop, divisor_of_the_squared_loss=target.std(axis=0).tolist())


def correction(network, states, c, constants):
    raw = network(states / torch.tensor(constants["scale_of_inputs"], dtype=torch.float64))
    out = raw * torch.tensor(constants["scale_of_outputs"], dtype=torch.float64)
    if c["scale"] == "times_qop":
        out = out * (states[:, 4:5] / constants["scale_of_qop"])
    return out


def provenance():
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*arguments):
        return subprocess.run(("git", "-C", here) + arguments, capture_output=True, text=True).stdout.strip()
    changes = git("status", "--porcelain", "--", os.path.dirname(os.path.dirname(here))).splitlines()
    return dict(commit=git("rev-parse", "HEAD"), uncommitted_changes_of_the_experiment_folder=len(changes),
                traceable_to_the_commit=not changes, machine=platform.node(),
                created=time.strftime("%Y-%m-%d %H:%M:%S"))


def train(path, rounds_in_this_call=None, report=print):
    c = read_configuration(path)
    folder = os.path.join(FOLDER_OF_RUNS, c["name"])
    os.makedirs(os.path.join(folder, "snapshots"), exist_ok=True)
    kept = os.path.join(folder, "configuration.yaml")
    if os.path.exists(kept):
        before = read_configuration(kept)
        if {k: v for k, v in before.items() if k not in STOPPING} != \
                {k: v for k, v in c.items() if k not in STOPPING}:
            raise ValueError("the run %s exists with another configuration" % c["name"])
        extended = {k: [before[k], c[k]] for k in STOPPING if before[k] != c[k]}
        if extended:
            with open(kept, "w") as handle:
                yaml.safe_dump(c, handle)
    else:
        extended = {}
        with open(kept, "w") as handle:
            yaml.safe_dump(c, handle)
        with open(os.path.join(folder, "provenance.json"), "w") as handle:
            json.dump(provenance(), handle, indent=1)

    train_s, train_ref, _ = split(c["reference"], "training")
    val_s, val_ref, val_p = split(c["reference"], "validation")
    target = train_ref - start_of(train_s, c, "training")
    start_val = start_of(val_s, c, "validation")
    file_of_constants = os.path.join(folder, "constants.json")
    if os.path.exists(file_of_constants):
        constants = json.load(open(file_of_constants))
    else:
        constants = constants_of(c, train_s, target)
        json.dump(constants, open(file_of_constants, "w"), indent=1)

    network = body(c["width"], c["depth"], c["seed"])
    file_of_state = os.path.join(folder, "state.json")
    state = json.load(open(file_of_state)) if os.path.exists(file_of_state) else \
        dict(rounds_done=0, ended=None, validation_medians_um=[])
    if extended and os.path.exists(file_of_state):
        state.setdefault("extensions", []).append(dict(
            at_round=state["rounds_done"], had_ended=state["ended"], changed=extended,
            when=time.strftime("%Y-%m-%d %H:%M:%S")))
        state["ended"] = None
        json.dump(state, open(file_of_state, "w"), indent=1)
    if state["rounds_done"]:
        network.load_state_dict(torch.load(os.path.join(folder, "snapshots", "round_%04d.pt" % state["rounds_done"])))
    x, y = torch.tensor(train_s), torch.tensor(target)
    xv = torch.tensor(val_s)
    divisor = torch.tensor(constants["divisor_of_the_squared_loss"], dtype=torch.float64)
    rounds_here = 0
    while state["ended"] is None and state["rounds_done"] < c["rounds_at_most"]:
        if rounds_in_this_call is not None and rounds_here >= rounds_in_this_call:
            break
        t0 = time.time()
        with torch.no_grad():
            residual = (correction(network, x, c, constants) - y).abs()
        threshold = torch.maximum(torch.tensor(FLOOR_OF_THE_THRESHOLD, dtype=torch.float64), residual.median(dim=0).values)

        def loss():
            error = correction(network, x, c, constants) - y
            if c["loss"] == "squared":
                return ((error / divisor) ** 2).mean()
            r = error / threshold
            return (torch.sqrt(1.0 + r * r) - 1.0).mean()

        with torch.no_grad():
            first = loss().item()
        for _ in range(c["restarts"]):
            optimiser = torch.optim.LBFGS(network.parameters(), max_iter=c["iterations"], history_size=100,
                                          tolerance_grad=1e-14, tolerance_change=1e-16,
                                          line_search_fn="strong_wolfe")

            def closure():
                optimiser.zero_grad()
                value = loss()
                value.backward()
                return value
            optimiser.step(closure)
        with torch.no_grad():
            last = loss().item()
            end_val = start_val + correction(network, xv, c, constants).numpy()
        scores = summary(end_val, val_ref, val_p)
        state["rounds_done"] += 1
        rounds_here += 1
        state["validation_medians_um"].append(scores["radial_median_um"])
        torch.save(network.state_dict(), os.path.join(folder, "snapshots", "round_%04d.pt" % state["rounds_done"]))
        row = dict(round=state["rounds_done"], loss_first=first, loss_last=last,
                   threshold_x_um=float(threshold[0]) * 1e3,
                   validation_radial_median_um=scores["radial_median_um"],
                   validation_radial_95_um=scores["radial_95_um"],
                   validation_worst_band_median_um=scores["worst_band_median_um"],
                   seconds=time.time() - t0)
        file_of_rounds = os.path.join(folder, "rounds.csv")
        new = not os.path.exists(file_of_rounds)
        with open(file_of_rounds, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        medians, window = state["validation_medians_um"], c["window"]
        if len(medians) > window and \
                min(medians[-window:]) > (1.0 - c["fall_needed"]) * min(medians[:-window]):
            state["ended"] = "plateau"
        elif state["rounds_done"] >= c["rounds_at_most"]:
            state["ended"] = "at its cap"
        json.dump(state, open(file_of_state, "w"), indent=1)
        report("%s round %3d: loss %.4g -> %.4g, validation %.4g / %.4g / %.4g um (median / 95 %% / worst band), %.0f s"
               % (c["name"], row["round"], first, last, row["validation_radial_median_um"],
                  row["validation_radial_95_um"], row["validation_worst_band_median_um"], row["seconds"]))
    return folder


def export(name, round_number=None):
    """The correction of a run for the compiled program, <exports>/corrections/<name>.
    Without a round, the round with the lowest validation median."""
    folder = os.path.join(FOLDER_OF_RUNS, name)
    c = read_configuration(os.path.join(folder, "configuration.yaml"))
    constants = json.load(open(os.path.join(folder, "constants.json")))
    state = json.load(open(os.path.join(folder, "state.json")))
    if round_number is None:
        round_number = int(np.argmin(state["validation_medians_um"])) + 1
    network = body(c["width"], c["depth"], c["seed"])
    network.load_state_dict(torch.load(os.path.join(folder, "snapshots", "round_%04d.pt" % round_number)))
    out = os.path.join(FOLDER_OF_CORRECTIONS, name)
    os.makedirs(out, exist_ok=True)
    layers = [m for m in network if isinstance(m, torch.nn.Linear)]
    pairs = [("name", name), ("start", c["start"]), ("steps", int(c["steps"])), ("correction", "network"),
             ("reference", c["reference"]), ("reader", READER_OF[c["reference"]]),
             ("first_plane_mm", repr(Z0)), ("last_plane_mm", repr(Z1)), ("round", round_number),
             ("times_qop", int(c["scale"] == "times_qop")), ("scale_of_qop", repr(constants["scale_of_qop"])),
             ("width", c["width"]), ("depth", c["depth"]), ("layers", len(layers))]
    for k, layer in enumerate(layers):
        pairs.append(("layer_%d" % k, "%d %d" % (layer.out_features, layer.in_features)))
        layer.weight.detach().numpy().astype(np.float64).tofile(os.path.join(out, "W%d.f64" % k))
        layer.bias.detach().numpy().astype(np.float64).tofile(os.path.join(out, "b%d.f64" % k))
    np.array(constants["scale_of_inputs"]).tofile(os.path.join(out, "scale_of_inputs.f64"))
    np.array(constants["scale_of_outputs"]).tofile(os.path.join(out, "scale_of_outputs.f64"))
    with open(os.path.join(out, "meta"), "w") as handle:
        for key, value in pairs:
            handle.write("%s %s\n" % (key, value))
    return c, constants, network, round_number


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[0] == "--starts":
        for path in arguments[1:]:
            c = read_configuration(path)
            for name in ("training", "validation"):
                start_of(split(c["reference"], name)[0], c, name)
            print("starts of %s are in the store" % c["name"])
    elif arguments[0] == "--export":
        _, _, _, used = export(arguments[1], int(arguments[3]) if len(arguments) > 3 else None)
        print("exported %s, round %d" % (arguments[1], used))
    else:
        train(arguments[0], int(arguments[2]) if len(arguments) > 2 else None,
              report=lambda text: print(text, flush=True))
