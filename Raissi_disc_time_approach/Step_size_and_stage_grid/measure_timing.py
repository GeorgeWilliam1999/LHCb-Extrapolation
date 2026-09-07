#!/usr/bin/env python
"""C3.3 - how many training rows one L-BFGS restart can afford.

The grid is 720 runs, and a run is 40 to 130 restarts. The training-set size is
therefore not a free parameter: it is set by what one restart of the *heaviest*
architecture in the grid can do inside a sensible wall time. This script
measures exactly that.

**The rule, fixed before measuring: one restart of the heaviest point of the
grid - depth 8, width 256, q = 20, the physics loss, one thread - on 4,000 /
8,000 / 16,000 / 36,000 training rows; take the largest N whose restart is
under 120 s.** With 120 s per restart a 130-restart run is about four and a
half hours, which is what a 720-job cluster can carry overnight.

The heaviest point is heavy for two separate reasons and both are measured:
the network is 8 x 256 with 4 x 21 = 84 outputs (about 480,000 parameters, and
L-BFGS with history 120 stores 240 vectors of that length), and the physics
loss evaluates the differentiable field at N x 20 stage planes on every
function evaluation, of which a `max_iter 200` strong-Wolfe restart makes
several hundred.

Each N is measured in its **own process**, so the peak resident set reported is
that N's alone and not the high-water mark of everything measured before it.
`--single N` measures one rung and writes `results/timing_N<N>.json`, which is
how the ladder is run **on the farm**: the 720 jobs run on farm slots, not on
the shared interactive node, and the interactive node was three times
oversubscribed while this study was set up, so a wall time measured there says
more about the other users than about the grid. `--collect` then assembles
`results/timing.json` from the per-rung files.

Memory is read from `/proc/self/status` (`VmHWM`), which counts the field map,
the dataset, the parameters, the L-BFGS history and the autograd graph
together - i.e. what the farm's `request_memory` has to cover.

    PYTHONNOUSERSITE=1 python measure_timing.py --data results/grid_q20.npz
    -> results/timing.json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import torch

import use_shared                                        # noqa: F401
from _shared.model import LHCbRates, physics_loss
from _shared.reference import gauss_legendre, make_field
from grid_model import grid_data_loss
from train_grid import subset_train

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LADDER = (4000, 8000, 16000, 36000)
LIMIT_S = 120.0


def peak_rss_mb():
    """VmHWM in MB - the peak resident set of this process."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(float(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return None


def one_restart(data_path, n_train, width, depth, mode, seed, max_iter):
    """Time exactly one L-BFGS restart of one architecture."""
    from grid_model import GridResidualNetwork
    d = np.load(os.path.abspath(data_path))
    data = subset_train({k: d[k] for k in d.files}, n_train)
    q = int(data["q"])
    field_which = str(data["field"])
    fld = make_field(field_which)

    S = torch.tensor(np.asarray(data["train_S"]))
    REF = torch.tensor(np.asarray(data["train_ref"]))
    ZN = torch.tensor(np.asarray(data["train_znodes"]))
    DZ = torch.tensor(np.asarray(data["train_dz"]))
    EX = torch.tensor(np.asarray(data["train_extra"]))
    _, A_np, b_np = gauss_legendre(q)
    A, b = torch.tensor(A_np), torch.tensor(b_np)
    rates = LHCbRates(fld)

    torch.manual_seed(seed)
    model = GridResidualNetwork(
        q, data["in_scale"], data["out_scale"], width=width, depth=depth,
        n_extra=2, c=np.asarray(data["c"]),
        extra_mean=np.asarray(data["extra_mean"]),
        extra_scale=np.asarray(data["extra_scale"]), field=fld)

    def loss_now():
        if mode == "physics":
            return physics_loss(model, rates, S, DZ, ZN, A, b, EX)
        return grid_data_loss(model, S, REF, EX)

    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            history_size=120, tolerance_grad=1e-13,
                            tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")
    n_eval = [0]

    def closure():
        n_eval[0] += 1
        opt.zero_grad()
        loss = loss_now()
        loss.backward()
        return loss

    t_setup = time.time()
    l0 = float(loss_now())
    t0 = time.time()
    opt.step(closure)
    wall = time.time() - t0
    l1 = float(loss_now())
    return {
        "n_train": int(len(S)), "q": q, "width": width, "depth": depth,
        "mode": mode, "seed": seed, "max_iter": max_iter,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "field_evaluations_per_loss": int(len(S) * q),
        "restart_wall_s": round(wall, 1),
        "loss_before": l0, "loss_after": l1,
        "closure_calls": n_eval[0],
        "s_per_closure": round(wall / max(n_eval[0], 1), 3),
        "setup_s": round(t0 - t_setup, 1),
        "peak_rss_mb": peak_rss_mb(),
        "host": os.uname().nodename,
        "load_average_1_5_15": [round(x, 1) for x in os.getloadavg()],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(RESULTS, "grid_q20.npz"))
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--mode", default="physics", choices=("physics", "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--ladder", default=",".join(str(n) for n in LADDER))
    ap.add_argument("--limit-s", type=float, default=LIMIT_S)
    ap.add_argument("--single", type=int, default=None,
                    help="measure this one N, write results/timing_N<N>.json "
                         "and print it; this is what the farm jobs run")
    ap.add_argument("--label", default=None,
                    help="with --single: name the output "
                         "results/timing_<label>.json instead of "
                         "results/timing_N<N>.json, so a per-architecture "
                         "probe does not collide with a ladder rung")
    ap.add_argument("--collect", action="store_true",
                    help="build results/timing.json from the per-N files "
                         "already on disk instead of measuring")
    ap.add_argument("--out", default=os.path.join(RESULTS, "timing.json"))
    a = ap.parse_args(argv)

    if a.single is not None:
        rec = one_restart(a.data, a.single, a.width, a.depth, a.mode, a.seed,
                          a.max_iter)
        rec["requested_n_train"] = a.single
        os.makedirs(RESULTS, exist_ok=True)
        name = "timing_%s.json" % a.label if a.label else "timing_N%d.json" % a.single
        with open(os.path.join(RESULTS, name), "w") as f:
            json.dump(rec, f, indent=1)
        print("JSON " + json.dumps(rec))
        return rec

    ladder = [int(x) for x in a.ladder.split(",")]
    if a.collect:
        rows = []
        for n in ladder:
            path = os.path.join(RESULTS, "timing_N%d.json" % n)
            if not os.path.exists(path):
                print("missing %s" % os.path.basename(path))
                continue
            with open(path) as f:
                rows.append(json.load(f))
        return finish(rows, ladder, a)

    rows = []
    for n in ladder:
        print("== N = %d" % n, flush=True)
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--single", str(n),
             "--data", os.path.abspath(a.data), "--width", str(a.width),
             "--depth", str(a.depth), "--mode", a.mode, "--seed", str(a.seed),
             "--max-iter", str(a.max_iter)],
            cwd=HERE, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("JSON ")]
        if not line:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            raise SystemExit("timing subprocess failed at N = %d" % n)
        rec = json.loads(line[-1][5:])
        rows.append(rec)
        print("   %6d rows: %7.1f s, %d closures, %.1f MB peak"
              % (rec["n_train"], rec["restart_wall_s"], rec["closure_calls"],
                 rec["peak_rss_mb"] or -1), flush=True)

    return finish(rows, ladder, a)


def finish(rows, ladder, a):
    """The rule applied to whatever rungs were measured, and the record."""
    rows = sorted(rows, key=lambda r: r["n_train"])
    under = [r for r in rows if r["restart_wall_s"] < a.limit_s]
    fell_back = not under
    chosen = max((r["n_train"] for r in under),
                 default=min((r["n_train"] for r in rows), default=min(ladder)))
    out = {
        "rule": "one L-BFGS restart of the heaviest grid point (depth %d, "
                "width %d, q from the dataset, %s loss, one thread); take the "
                "largest N under %.0f s" % (a.depth, a.width, a.mode, a.limit_s),
        "dataset": os.path.relpath(os.path.abspath(a.data), HERE),
        "limit_s": a.limit_s,
        "ladder": rows,
        "chosen_n_train": int(chosen),
        "chosen_restart_wall_s": next(
            (r["restart_wall_s"] for r in rows if r["n_train"] == chosen), None),
        "chosen_peak_rss_mb": next(
            (r["peak_rss_mb"] for r in rows if r["n_train"] == chosen), None),
        "no_rung_under_the_limit": bool(fell_back),
        "note": ("no rung of the ladder came in under the limit, so the "
                 "smallest rung was taken and the wall time per restart is "
                 "reported as measured" if fell_back else
                 "the chosen rung is the largest of the ladder under the limit"),
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("chosen_n_train", "chosen_restart_wall_s",
                       "chosen_peak_rss_mb", "no_rung_under_the_limit")},
                     indent=1))
    print("wrote %s" % os.path.relpath(a.out, HERE))
    return out


if __name__ == "__main__":
    main()
