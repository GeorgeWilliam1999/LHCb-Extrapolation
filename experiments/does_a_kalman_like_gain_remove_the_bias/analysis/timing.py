"""The time of each part of a gain step, on the CPU, in torch, batched.

Not a measurement of the target code: the field map, the network and the
solve are those of this package in double precision. Only the ratios between
the parts are of use. Every part is timed on the same 92,928 steps (the 64
steps of the 1,452 test tracks from their reference states), the parts in
turn inside each repeat, so a change of load on the node falls on all parts.

Parts, each for all rows, per step:
  lookup         f at the 2 stages, no gradient (2 field lookups)
  lookup + J     f and J = df/dY at the 2 stages by autograd (2 lookups, 2 Jacobians)
  solve          build I - dz A (x) J, solve the 8 x 8 system, form K r
  network        one network call (the stages)
  euler stages   f at the start state and the Euler stages (1 lookup)

    timing.py run      writes results/timing.csv (THREADS sets the torch threads)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from common import (NUMBER_OF_STEPS, RUNS, START_PLANES_MM, STEP_MM, network_of, on_planes,
                    setup_threads, write_rows)
from part_b import euler_stages
from stage_equations import A, B, _t, rates, rates_and_jacobian, residual, stage_planes

REPEATS = 7


def run():
    setup_threads()
    threads = torch.get_num_threads()
    ref = on_planes("test")
    n = len(ref)
    s = ref[:, :-1].reshape(-1, 5)
    z = np.broadcast_to(START_PLANES_MM, (n, NUMBER_OF_STEPS)).reshape(-1).copy()
    S0, zt = _t(s), _t(z)
    qop, zs = S0[:, 4:5], stage_planes(zt)
    nets = {key: network_of(key) for _, key in RUNS}
    stages = {}
    with torch.no_grad():
        for key, net in nets.items():
            stages[key] = net.predict(S0, zt).stage_states[:, 0].detach().clone()
    Y = stages[RUNS[0][1]]
    f, J = rates_and_jacobian(Y, qop, zs)

    def lookup():
        with torch.no_grad():
            rates(Y, qop, zs)

    def lookup_and_jacobian():
        rates_and_jacobian(Y, qop, zs)

    def solve():
        with torch.no_grad():
            r = residual(S0, Y, f)
            m = len(Y)
            M = torch.eye(8, dtype=torch.float64).repeat(m, 1, 1)
            for j in range(2):
                for k in range(2):
                    M[:, 4 * j:4 * j + 4, 4 * k:4 * k + 4] -= STEP_MM * A[j, k] * J[:, k]
            w = torch.linalg.solve(M, r.reshape(m, 8, 1))[..., 0].reshape(m, 2, 4)
            STEP_MM * torch.einsum("k,mkab,mkb->ma", B, J, w)

    def network(key):
        def call():
            with torch.no_grad():
                nets[key].predict(S0, zt)
        return call

    def euler():
        euler_stages(s, z)

    parts = [("lookup (2 per step)", lookup), ("lookup + Jacobian (2 + 2 per step)", lookup_and_jacobian),
             ("8x8 solve and K r", solve), ("euler stages (1 lookup per step)", euler)]
    parts += [("network call, " + key, network(key)) for _, key in RUNS]
    times = {name: [] for name, _ in parts}
    for name, fn in parts:              # warm up
        fn()
    for _ in range(REPEATS):
        for name, fn in parts:
            t0 = time.perf_counter()
            fn()
            times[name].append(time.perf_counter() - t0)
    base = np.median(times["lookup (2 per step)"])
    rows = []
    for name, _ in parts:
        t = np.array(times[name])
        rows.append(dict(part=name, threads=threads, rows=len(s), repeats=REPEATS,
                         median_s=float(np.median(t)), min_s=float(t.min()), max_s=float(t.max()),
                         median_ns_per_row=float(np.median(t) / len(s) * 1e9),
                         ratio_to_two_lookups=float(np.median(t) / base)))
        print(rows[-1], flush=True)
    file = "timing_threads_%d.csv" % threads
    write_rows(file, rows)


if __name__ == "__main__":
    {"run": run}[sys.argv[1]]()
