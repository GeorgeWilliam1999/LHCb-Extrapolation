"""The trainer of stage 3: the network of the crossing (start: 12 Nystrom steps, scale times_qop) as in
stage 2 (fastex.train, unchanged and imported), on the data of make_corner_data.py, with two new
choices:

    data         training          all training tracks
                 corner_training   corner tracks only (|y| at 5 m > 600 mm), real + extra
    validation   validation | corner_validation
    tail_weight  null, or {y_from, y_to, most}: the Huber loss of a track is weighted by
                 1 + (most - 1) * clip((|y at 5 m| - y_from) / (y_to - y_from), 0, 1)
    stop_on      median | p99: the validation score of the stopping rule and of the best round

A round, the Huber loss, its threshold (the median error of the round before) and the stopping rule
(no `fall_needed` fall of the best score in the last `window` rounds) are those of stage 2. A run is
<store>/runs/<name>; a job that is started again goes on from its last round. A call stops after
HOURS_AT_MOST hours (environment, default 22) so that a farm job ends in time.

    python train_corner.py <configuration.yaml>
    python train_corner.py --export <name> <out.json> [round]   the weights JSON of allen_nn (best round by default)
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

from fastex.compiled import summary
from fastex.train import FLOOR_OF_THE_THRESHOLD, Z0, Z1, body, constants_of, correction

STORE = "/data/bfys/gscriven/fast_extrapolator_store/stage_3_corner"
KEYS = {"name", "data", "validation", "tail_weight", "stop_on", "width", "depth", "seed", "restarts", "iterations",
        "rounds_at_most", "window", "fall_needed"}
STOPPING = ("rounds_at_most", "window", "fall_needed")
C = dict(start="nystrom", scale="times_qop", steps=12, loss="huber")  # the stage-2 choices, fixed here


def read_configuration(path):
    c = yaml.safe_load(open(path))
    unknown, missing = set(c) - KEYS, KEYS - set(c)
    if unknown or missing:
        raise ValueError("configuration %s: unknown %s, missing %s" % (path, sorted(unknown), sorted(missing)))
    if c["data"] not in ("training", "corner_training") or c["validation"] not in ("validation", "corner_validation") \
            or c["stop_on"] not in ("median", "p99"):
        raise ValueError("configuration %s names a choice that does not exist" % path)
    return c


def data(name):
    f = lambda a: np.fromfile(os.path.join(STORE, "data", name, a + ".f64"))  # noqa: E731
    return dict(s=f("start_state").reshape(-1, 5), ref=f("reference_end_state").reshape(-1, 5)[:, :4],
                start=f("start_nystrom_12").reshape(-1, 4), p=f("momentum_gev"), y5=f("abs_y_at_5m"))


def weights(c, y5):
    w = c["tail_weight"]
    if not w:
        return np.ones_like(y5)
    return 1.0 + (w["most"] - 1.0) * np.clip((y5 - w["y_from"]) / (w["y_to"] - w["y_from"]), 0.0, 1.0)


def provenance():
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*a):
        return subprocess.run(("git", "-C", here) + a, capture_output=True, text=True).stdout.strip()
    changes = git("status", "--porcelain", "--", os.path.dirname(here)).splitlines() + \
        git("status", "--porcelain", "--", os.path.join(os.path.dirname(os.path.dirname(here)), "python")).splitlines()
    return dict(commit=git("rev-parse", "HEAD"), uncommitted_changes=len(changes), traceable_to_the_commit=not changes,
                machine=platform.node(), created=time.strftime("%Y-%m-%d %H:%M:%S"))


def scores_of(end, ref, p, y5):
    r = np.hypot(end[:, 0] - ref[:, 0], end[:, 1] - ref[:, 1]) * 1e3
    s = summary(end, ref, p)
    out = dict(median=s["radial_median_um"], p95=s["radial_95_um"], p99=float(np.percentile(r, 99)),
               p999=float(np.percentile(r, 99.9)), largest=float(r.max()))
    for cut in (600, 700):
        m = y5 > cut
        out["above_%d_median" % cut] = float(np.median(r[m])) if m.any() else float("nan")
        out["above_%d_p99" % cut] = float(np.percentile(r[m], 99)) if m.sum() > 10 else float("nan")
        out["above_%d_largest" % cut] = float(r[m].max()) if m.any() else float("nan")
    return out


def train(path):
    c = read_configuration(path)
    folder = os.path.join(STORE, "runs", c["name"])
    os.makedirs(os.path.join(folder, "snapshots"), exist_ok=True)
    kept = os.path.join(folder, "configuration.yaml")
    if os.path.exists(kept):
        before = read_configuration(kept)
        if {k: v for k, v in before.items() if k not in STOPPING} != {k: v for k, v in c.items() if k not in STOPPING}:
            raise ValueError("the run %s exists with another configuration" % c["name"])
    else:
        yaml.safe_dump(c, open(kept, "w"))
        json.dump(provenance(), open(os.path.join(folder, "provenance.json"), "w"), indent=1)
    d, v = data(c["data"]), data(c["validation"])
    target = d["ref"] - d["start"]
    file_of_constants = os.path.join(folder, "constants.json")
    if os.path.exists(file_of_constants):
        constants = json.load(open(file_of_constants))
    else:
        constants = constants_of(C, d["s"], target)
        constants["box_of_the_training_tracks"] = np.abs(d["s"]).max(axis=0).tolist()
        json.dump(constants, open(file_of_constants, "w"), indent=1)
    network = body(c["width"], c["depth"], c["seed"])
    file_of_state = os.path.join(folder, "state.json")
    state = json.load(open(file_of_state)) if os.path.exists(file_of_state) else dict(rounds_done=0, ended=None, scores=[])
    if state["ended"] is None and state["rounds_done"] >= c["rounds_at_most"]:
        state["ended"] = "at its cap"
    if state["rounds_done"]:
        network.load_state_dict(torch.load(os.path.join(folder, "snapshots", "round_%04d.pt" % state["rounds_done"])))
    x, y = torch.tensor(d["s"]), torch.tensor(target)
    w = torch.tensor(weights(c, d["y5"]))[:, None]
    xv = torch.tensor(v["s"])
    hours = float(os.environ.get("HOURS_AT_MOST", "22"))
    t_call = time.time()
    while state["ended"] is None and state["rounds_done"] < c["rounds_at_most"]:
        if time.time() - t_call > hours * 3600:
            print("stopping: %.1f h in this call" % hours, flush=True)
            break
        t0 = time.time()
        with torch.no_grad():
            residual = (correction(network, x, C, constants) - y).abs()
        threshold = torch.maximum(torch.tensor(FLOOR_OF_THE_THRESHOLD, dtype=torch.float64), residual.median(dim=0).values)

        def loss():
            r = (correction(network, x, C, constants) - y) / threshold
            return ((torch.sqrt(1.0 + r * r) - 1.0) * w).sum() / (w.sum() * 4)

        with torch.no_grad():
            first = loss().item()
        for _ in range(c["restarts"]):
            opt = torch.optim.LBFGS(network.parameters(), max_iter=c["iterations"], history_size=100, tolerance_grad=1e-14,
                                    tolerance_change=1e-16, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                value = loss()
                value.backward()
                return value
            opt.step(closure)
        with torch.no_grad():
            last = loss().item()
            end_v = v["start"] + correction(network, xv, C, constants).numpy()
        sc = scores_of(end_v, v["ref"], v["p"], v["y5"])
        state["rounds_done"] += 1
        state["scores"].append(sc[c["stop_on"]])
        torch.save(network.state_dict(), os.path.join(folder, "snapshots", "round_%04d.pt" % state["rounds_done"]))
        row = dict(round=state["rounds_done"], loss_first=first, loss_last=last, seconds=time.time() - t0,
                   **{"validation_" + k: val for k, val in sc.items()})
        f = os.path.join(folder, "rounds.csv")
        new = not os.path.exists(f)
        with open(f, "a", newline="") as handle:
            wr = csv.DictWriter(handle, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
        s, window = state["scores"], c["window"]
        if len(s) > window and min(s[-window:]) > (1.0 - c["fall_needed"]) * min(s[:-window]):
            state["ended"] = "plateau"
        elif state["rounds_done"] >= c["rounds_at_most"]:
            state["ended"] = "at its cap"
        json.dump(state, open(file_of_state, "w"), indent=1)
        print("%s round %3d: loss %.4g -> %.4g, validation median %.3g 99%% %.4g largest %.4g, above 700: median %.3g 99%% %.4g"
              " largest %.4g um, %.0f s" % (c["name"], row["round"], first, last, sc["median"], sc["p99"], sc["largest"],
                                            sc["above_700_median"], sc["above_700_p99"], sc["above_700_largest"],
                                            row["seconds"]), flush=True)


def export(name, round_number=None):
    """The weights JSON of allen_nn (the format of allen_nn/python/export_json.py) for the best round."""
    folder = os.path.join(STORE, "runs", name)
    c = read_configuration(os.path.join(folder, "configuration.yaml"))
    constants = json.load(open(os.path.join(folder, "constants.json")))
    state = json.load(open(os.path.join(folder, "state.json")))
    if round_number is None:
        round_number = int(np.argmin(state["scores"])) + 1
    network = body(c["width"], c["depth"], c["seed"])
    network.load_state_dict(torch.load(os.path.join(folder, "snapshots", "round_%04d.pt" % round_number)))
    layers = [m for m in network if isinstance(m, torch.nn.Linear)]
    box = constants["box_of_the_training_tracks"]
    out = dict(name=name, round=round_number, reference="linear", reader_of_the_start="rec", field_map="v8r1_up",
               polarity_of_the_training_map=1, steps=12, depth=c["depth"], width=c["width"], activation="tanh", z0=Z0,
               z1=Z1, leg_step_mm=250.0, network_jacobian=0, scale_of_inputs=constants["scale_of_inputs"],
               scale_of_outputs=constants["scale_of_outputs"], scale_of_qop=constants["scale_of_qop"],
               domain=dict(most_abs_x=box[0], most_abs_y=box[1], most_abs_tx=box[2], most_abs_ty=box[3], most_abs_qop=box[4]),
               layers=[dict(W=l.weight.detach().numpy().tolist(), b=l.bias.detach().numpy().tolist()) for l in layers],
               provenance=dict(run=folder, training=json.load(open(os.path.join(folder, "provenance.json")))))
    return out, round_number


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--export":
        out, used = export(a[1], int(a[3]) if len(a) > 3 else None)
        json.dump(out, open(a[2], "w"))
        print("wrote %s (run %s, round %d)" % (a[2], a[1], used))
    else:
        train(a[0])
