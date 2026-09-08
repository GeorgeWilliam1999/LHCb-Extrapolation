#!/usr/bin/env python
"""C6.3a - what one chained job costs, measured before any farm time is spent.

The 0.1 mm column dominates the whole experiment: a 5.2 m crossing at a 0.1 mm
step is about 52,000 network evaluations per track, against 5,200 at 1 mm and
one at the full crossing. Two things are measured.

**The plan's probe.** The whole 0.1 mm column on **10 tracks**, at the heaviest
point of the architecture grid (8 x 256, q = 20, 484,180 parameters) and at a
light one (4 x 32, q = 2). This is the run the plan asked for and it is
reported as such.

**The batch probe, which is what the projection is built on.** A chain is a
loop over steps with the whole batch of tracks inside it, so the per-track cost
falls steeply with the batch: at ten tracks the fp64 matrix multiplies are far
too small to amortise torch's own per-operation overhead. 200 steps are timed
at batches of 10, 500 and 1,000, and the full-job projection uses the rate at
the batch each column actually runs on. Projecting from the ten-track probe
instead overstates the job by about a factor four, which is recorded here so
the difference is visible rather than hidden.

**The rule, fixed before measuring:** if the projected wall time of a full job
at 8 x 256 exceeds 8 hours, the 0.1 mm column is cut from 250 + 250 tracks to
100 + 100 and the cut is stated in the README and in the report.

    PYTHONNOUSERSITE=1 python measure_timing.py

writes `results/timing.json`.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json
import resource
import time

import numpy as np
import torch

import use_shared                                        # noqa: F401
import chain as C
from chain_one import load_model

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PROBE_TAGS = ("w256_d8_q20_physics_s0", "w32_d4_q02_physics_s0")
PROBE_TRACKS = 10
BATCHES = (10, 500, 1000)
BATCH_STEPS = 200
FULL_JOB_HOURS_LIMIT = 8.0


def _pieces(tag, tracks):
    width, depth, q, mode, seed = C.parse_tag(tag)
    model, data = load_model(tag, q, width, depth)
    return (model, np.asarray(data["extra_mean"]),
            np.asarray(data["extra_scale"]), width, depth, q)


def column_probe(tag, n_tracks, tracks):
    """The plan's probe: the whole 0.1 mm column on a handful of tracks."""
    model, em, es, width, depth, q = _pieces(tag, tracks)
    DIR = np.asarray(tracks["DIRECTION"])
    idx = C.column_tracks(DIR, n_tracks // 2)
    t0 = time.time()
    _, _, _, n_steps, _ = C.chain_column(
        model, np.asarray(tracks["S0"])[idx], np.asarray(tracks["z0"])[idx],
        np.asarray(tracks["dz"])[idx], 0.1, em, es)
    wall = time.time() - t0
    work = int(n_steps.sum())
    return {"tag": tag, "architecture": "%dx%d" % (depth, width), "q": q,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "n_tracks": len(idx),
            "steps_per_track_median": int(np.median(n_steps)),
            "sample_steps": work, "wall_s": round(wall, 2),
            "us_per_sample_step": round(1e6 * wall / work, 2),
            "peak_rss_gb": round(resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2, 2)}


def batch_probe(tag, tracks, batches=BATCHES, n_steps=BATCH_STEPS):
    """The per-sample-step cost at each batch size the columns actually use."""
    model, em, es, width, depth, q = _pieces(tag, tracks)
    DIR = np.asarray(tracks["DIRECTION"])
    emt, est = torch.as_tensor(em), torch.as_tensor(es)
    out = []
    for nb in batches:
        idx = C.column_tracks(DIR, nb // 2)
        S = torch.as_tensor(np.asarray(tracks["S0"])[idx].copy())
        z = torch.as_tensor(np.asarray(tracks["z0"])[idx].copy())
        h = torch.full_like(z, 0.1)
        with torch.no_grad():
            t0 = time.time()
            for _ in range(n_steps):
                extra = torch.stack([(z - emt[0]) / est[0],
                                     (h - emt[1]) / est[1]], dim=1)
                S[:, :4] = model(S, extra)[:, -1, :]
                z = z + h
            dt = time.time() - t0
        out.append({"tag": tag, "architecture": "%dx%d" % (depth, width),
                    "q": q, "batch": int(len(idx)), "n_steps": n_steps,
                    "ms_per_step": round(1e3 * dt / n_steps, 3),
                    "us_per_sample_step":
                        round(1e6 * dt / n_steps / len(idx), 2)})
    return out


def step_cost_model(rows):
    """(a, b) in ms: one step of a batch of n tracks costs a + b*n.

    A least-squares line through the measured (batch, ms per step) points. The
    intercept is torch's own per-operation overhead, which a batch of ten
    tracks pays in full and a batch of a thousand amortises away; charging
    every column the ten-track rate is what makes the naive projection four
    times too pessimistic.
    """
    x = np.array([r["batch"] for r in rows], dtype=float)
    y = np.array([r["ms_per_step"] for r in rows], dtype=float)
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def project(rows, n_per_direction_01mm, median_steps):
    """Projected wall time of a whole job, in hours, and the work it does."""
    a, b = step_cost_model(rows)
    hours, work = 0.0, 0
    for name, nominal, n_per in C.COLUMNS:
        n_tracks = 2 * (n_per_direction_01mm if name == "0.1 mm" else n_per)
        n = 1 if nominal is None else max(round(median_steps * 0.1 / nominal), 1)
        work += n_tracks * n
        hours += n * (a + b * n_tracks) / 1e3 / 3600.0
    return hours, work


def reproject(path):
    """Redo the projection from the probes already measured."""
    with open(path) as f:
        out = json.load(f)
    med = out["median_steps_at_0.1mm"]
    out["projection"] = []
    for tag in PROBE_TAGS:
        rows = [r for r in out["batch_probes"] if r["tag"] == tag]
        naive = [p for p in out["column_probes"] if p["tag"] == tag][0]
        a, b = step_cost_model(rows)
        print("%-24s one step of a batch of n tracks costs %.3f + %.5f n ms"
              % (tag, a, b))
        for n_per in (250, 100):
            hours, work = project(rows, n_per, med)
            out["projection"].append(
                {"tag": tag, "architecture": rows[0]["architecture"],
                 "n_per_direction_0.1mm": n_per, "sample_steps": int(work),
                 "projected_hours": round(hours, 2),
                 "step_cost_ms_intercept": round(a, 3),
                 "step_cost_ms_per_track": round(b, 5),
                 "projected_hours_from_10_track_probe":
                     round(work * naive["us_per_sample_step"] / 1e6 / 3600, 2)})
            print("   full job, %d + %d tracks at 0.1 mm: %d sample-steps, "
                  "%.2f h  (the 10-track probe would have said %.2f h)"
                  % (n_per, n_per, work, hours,
                     out["projection"][-1]
                     ["projected_hours_from_10_track_probe"]))
    heavy = [r for r in out["projection"]
             if r["architecture"] == "8x256"
             and r["n_per_direction_0.1mm"] == 250][0]
    out["decision"] = {
        "projected_hours_8x256_at_250_plus_250": heavy["projected_hours"],
        "limit_hours": FULL_JOB_HOURS_LIMIT,
        "n_per_direction_0.1mm": (250 if heavy["projected_hours"]
                                  <= FULL_JOB_HOURS_LIMIT else 100),
        "peak_rss_gb": max(p["peak_rss_gb"] for p in out["column_probes"])}
    print("decision: %.2f h projected at 8 x 256, limit %.0f h -> the 0.1 mm "
          "column runs on %d + %d tracks"
          % (heavy["projected_hours"], FULL_JOB_HOURS_LIMIT,
             out["decision"]["n_per_direction_0.1mm"],
             out["decision"]["n_per_direction_0.1mm"]))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", default=os.path.join(RESULTS,
                                                     "chain_tracks.npz"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "timing.json"))
    ap.add_argument("--reproject", action="store_true",
                    help="redo the projection from the probes already in "
                         "timing.json, without re-measuring anything")
    a = ap.parse_args(argv)
    if a.reproject:
        return reproject(a.out)
    tracks = np.load(a.tracks)
    med = int(np.median(np.round(np.abs(np.asarray(tracks["dz"])) / 0.1)))

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "host": os.uname().nodename,
           "rule": "if the projected 8 x 256 job exceeds %.0f h the 0.1 mm "
                   "column is cut to 100 + 100 tracks" % FULL_JOB_HOURS_LIMIT,
           "median_steps_at_0.1mm": med,
           "columns": [{"name": n, "nominal_dz_mm": d, "n_per_direction": k}
                       for n, d, k in C.COLUMNS],
           "column_probes": [], "batch_probes": [], "projection": []}

    for tag in PROBE_TAGS:
        p = column_probe(tag, PROBE_TRACKS, tracks)
        out["column_probes"].append(p)
        print("%-24s %-7s q=%2d  %d tracks x %d steps = %d sample-steps in "
              "%.1f s  (%.1f us each, %.2f GB)"
              % (p["tag"], p["architecture"], p["q"], p["n_tracks"],
                 p["steps_per_track_median"], p["sample_steps"], p["wall_s"],
                 p["us_per_sample_step"], p["peak_rss_gb"]), flush=True)

    for tag in PROBE_TAGS:
        rows = batch_probe(tag, tracks)
        out["batch_probes"] += rows
        for r in rows:
            print("%-24s %-7s batch %4d : %6.2f ms/step, %7.2f us/sample-step"
                  % (r["tag"], r["architecture"], r["batch"], r["ms_per_step"],
                     r["us_per_sample_step"]), flush=True)
        a, b = step_cost_model(rows)
        print("   one step of a batch of n tracks costs %.3f + %.5f n ms"
              % (a, b), flush=True)
        for n_per in (250, 100):
            hours, work = project(rows, n_per, med)
            naive = [p for p in out["column_probes"] if p["tag"] == tag][0]
            out["projection"].append(
                {"tag": tag, "architecture": rows[0]["architecture"],
                 "n_per_direction_0.1mm": n_per, "sample_steps": int(work),
                 "projected_hours": round(hours, 2),
                 "step_cost_ms_intercept": round(a, 3),
                 "step_cost_ms_per_track": round(b, 5),
                 "projected_hours_from_10_track_probe":
                     round(work * naive["us_per_sample_step"] / 1e6 / 3600, 2)})
            print("   full job, %d + %d tracks at 0.1 mm: %d sample-steps, "
                  "%.2f h  (the 10-track probe would have said %.2f h)"
                  % (n_per, n_per, work, hours,
                     out["projection"][-1]
                     ["projected_hours_from_10_track_probe"]), flush=True)

    heavy = [r for r in out["projection"]
             if r["architecture"] == "8x256"
             and r["n_per_direction_0.1mm"] == 250][0]
    out["decision"] = {
        "projected_hours_8x256_at_250_plus_250": heavy["projected_hours"],
        "limit_hours": FULL_JOB_HOURS_LIMIT,
        "n_per_direction_0.1mm": (250 if heavy["projected_hours"]
                                  <= FULL_JOB_HOURS_LIMIT else 100),
        "peak_rss_gb": max(p["peak_rss_gb"] for p in out["column_probes"])}
    print("decision: %.2f h projected at 8 x 256, limit %.0f h -> the 0.1 mm "
          "column runs on %d + %d tracks"
          % (heavy["projected_hours"], FULL_JOB_HOURS_LIMIT,
             out["decision"]["n_per_direction_0.1mm"],
             out["decision"]["n_per_direction_0.1mm"]))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main()
