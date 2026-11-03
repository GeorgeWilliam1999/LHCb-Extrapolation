"""A.8: is the bias map the same for another seed?

The tranche-1 runs of the experiment does_a_better_optimiser_improve_the_endpoint
train seeds 2 and 3 of both forms (and the self-scaled Broyden arm). This
script reads their snapshots, never writes to their folders, and correlates
the binned bias map of e (as in A.9) between seeds at one round that every run
of a form has finished. It also correlates the maps of e minus the exact
scheme's own error (as in A.9), the part of e that is the network's.

    seeds.py common      at the latest round that all the runs of a form have
                         finished (on 2026-10-28 that is round 1: the runs were
                         submitted that morning; usable from about 30 Oct)
    seeds.py last        each run at its own latest finished round
    seeds.py last <key>... as last, without the runs named (a run that is
                         still training)

A round counts as finished when its snapshot folder holds both network.pt and
scores.json.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from common import (COMPONENTS, STORE, SHORT, setup_threads, single_step_errors, network_of,
                    write_rows)
from part_a import bias_map, exact_errors

FORMS = {
    "straight line, per-track scale": {
        "seed 1, L-BFGS": "25dfa17ef168", "seed 2, L-BFGS": "dedaab236b3d",
        "seed 3, L-BFGS": "b68802fcce0e", "seed 1, self-scaled Broyden": "166cfc8cdfd6",
        "seed 2, self-scaled Broyden": "3400f9c572ef",
        "seed 3, self-scaled Broyden": "c8b210923994"},
    "direct states": {
        "seed 1, L-BFGS": "49e5e2450eaa", "seed 2, L-BFGS": "b6bd9570e6f2",
        "seed 3, L-BFGS": "5cae49832580", "seed 1, self-scaled Broyden": "f4c86e05eafe",
        "seed 2, self-scaled Broyden": "c3a1aa54c6fe",
        "seed 3, self-scaled Broyden": "c19552a6831c"},
}


def finished_rounds(key):
    folder = os.path.join(STORE, "runs", key, "snapshots")
    if not os.path.isdir(folder):
        return []
    return sorted(int(name.split("_")[1]) for name in os.listdir(folder)
                  if os.path.exists(os.path.join(folder, name, "network.pt"))
                  and os.path.exists(os.path.join(folder, name, "scores.json")))


def main(mode, left_out=()):
    rows = []
    exact = exact_errors("test")
    for form, runs in FORMS.items():
        runs = {label: key for label, key in runs.items() if key not in left_out}
        rounds = {label: finished_rounds(key) for label, key in runs.items()}
        usable = {label: r for label, r in rounds.items() if r}
        if len(usable) < 2:
            print(form, ": fewer than two runs with a finished round", flush=True)
            continue
        if mode == "common":
            common = min(max(r) for r in usable.values())
            chosen = {label: common for label, r in usable.items() if common in r}
        else:
            chosen = {label: max(r) for label, r in usable.items()}
        maps, es, own_maps, own = {}, {}, {}, {}
        for label, r in chosen.items():
            es[label] = single_step_errors(network_of(runs[label], [r]), "test")
            maps[label] = bias_map(es[label])
            own[label] = es[label] - exact
            own_maps[label] = bias_map(own[label])
        labels = list(chosen)
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                la, lb = labels[a], labels[b]
                for i, c in enumerate(COMPONENTS):
                    rows.append(dict(
                        form=form, mode=mode, first=la, first_key=runs[la],
                        first_round=chosen[la], second=lb, second_key=runs[lb],
                        second_round=chosen[lb], component=c,
                        bias_map_correlation=float(np.corrcoef(maps[la][:, i],
                                                               maps[lb][:, i])[0, 1]),
                        step_by_step_correlation=float(np.corrcoef(
                            es[la][..., i].ravel(), es[lb][..., i].ravel())[0, 1]),
                        bias_map_correlation_without_exact_scheme_error=float(np.corrcoef(
                            own_maps[la][:, i], own_maps[lb][:, i])[0, 1]),
                        step_by_step_correlation_without_exact_scheme_error=float(np.corrcoef(
                            own[la][..., i].ravel(), own[lb][..., i].ravel())[0, 1]),
                        rms_e_first=float(np.sqrt((es[la][..., i] ** 2).mean())),
                        rms_e_second=float(np.sqrt((es[lb][..., i] ** 2).mean()))))
                    print(rows[-1], flush=True)
    write_rows("a8_seed_correlations_%s.csv" % mode, rows)


if __name__ == "__main__":
    setup_threads()
    main(sys.argv[1], sys.argv[2:])
