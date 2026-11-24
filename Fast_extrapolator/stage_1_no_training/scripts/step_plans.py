"""Step plans: steps of unequal length for the fixed-step integrators.

A fixed-step integrator spends its field reads evenly along z. The error it
makes is not even: it is made where the field changes fast (the edges of the
magnet), and an error of the slope that is made early has a long way to grow.
A plan puts the planes of the steps so that each step makes about the same
contribution to the error on the last plane.

The contribution is measured, not assumed, on 2,000 training tracks:

  1. the crossing is cut into 64 equal steps of length h;
  2. on each step the Nystrom step is made once with h and once with two steps
     of h / 2; the difference is the local error of that step;
  3. the local error is carried to the last plane with a straight line (the
     error of the position plus the error of the slope times the distance
     left), and its mean length over the tracks is the cost c(z) of that step.

For a fourth-order method the local error grows as h ** 5, so the planes that
equalise the contributions hold equal shares of c(z) ** (1/5). The exponent is
left free: a plan holds equal shares of c(z) ** beta, and beta is chosen on the
training tracks (classical_frontier.py), never on the test tracks. beta = 0 is
equal steps, so a chosen plan is never worse than equal steps on those tracks.

    step_plans.py      writes <exports>/plans/<map>/N<steps>_beta<beta>.f64 and
                       results/cost_of_a_step_along_z.csv
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from fastex import FOLDER_OF_EXPORTS  # noqa: E402

STEPS = (4, 6, 8, 10, 12, 16, 20, 24, 32)
BETAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
FINE_STEPS = 64
Z0, Z1 = 2648.2, 7826.0
FIELD_MAPS = {"linear": "v8r1_up", "cubic": "v8r1_up_cubic"}


def file_of_plan(of_map, steps, beta):
    return os.path.join(FOLDER_OF_EXPORTS, "plans", of_map, "N%d_beta%g.f64" % (steps, beta))


def cost_along_z(of_map):
    """The edges of the 64 fine steps and the cost of each: its mean
    contribution to the error on the last plane, in mm."""
    from rkpinn import registry
    from fastex.classical import nystrom3
    field_map = registry.component("field_map", FIELD_MAPS[of_map])()
    states = np.fromfile(os.path.join(FOLDER_OF_EXPORTS, "tracks_" + of_map,
                                      "training_start_state.f64")).reshape(-1, 5)
    edges = Z0 + np.arange(FINE_STEPS + 1) * ((Z1 - Z0) / FINE_STEPS)
    cost = np.empty(FINE_STEPS)
    for j in range(FINE_STEPS):
        a, b = edges[j], edges[j + 1]
        one = nystrom3(states, [a, b], field_map)
        two = nystrom3(states, [a, 0.5 * (a + b), b], field_map)
        d = one - two
        left = Z1 - b
        cost[j] = np.hypot(d[:, 0] + d[:, 2] * left, d[:, 1] + d[:, 3] * left).mean()
        states = two
    return edges, cost


def planes_of_plan(edges, cost, steps, beta):
    w = np.maximum(cost, 1e-300) ** beta
    cumulative = np.concatenate([[0.0], np.cumsum(w * np.diff(edges))])
    planes = np.interp(np.linspace(0.0, cumulative[-1], steps + 1), cumulative, edges)
    planes[0], planes[-1] = Z0, Z1
    return planes


def main():
    rows = []
    for of_map in ("linear", "cubic"):
        edges, cost = cost_along_z(of_map)
        os.makedirs(os.path.join(FOLDER_OF_EXPORTS, "plans", of_map), exist_ok=True)
        for steps in STEPS:
            for beta in BETAS:
                planes_of_plan(edges, cost, steps, beta).tofile(file_of_plan(of_map, steps, beta))
        for j in range(FINE_STEPS):
            rows.append(dict(map=of_map, start_of_the_step_mm=edges[j], end_of_the_step_mm=edges[j + 1],
                             mean_contribution_to_the_final_error_um=cost[j] * 1e3))
        print("%s map: the lengths of 8 steps with beta 0.2: %s mm"
              % (of_map, " ".join("%.0f" % d for d in np.diff(planes_of_plan(edges, cost, 8, 0.2)))))
        print("   cost of the fine steps (um): %s" % " ".join("%.2g" % (c * 1e3) for c in cost[::4]))
    with open(os.path.join(HERE, "results", "cost_of_a_step_along_z.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
