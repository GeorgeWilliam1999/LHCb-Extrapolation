"""The refit of the last layer: Gauss-Newton on the stage residual, after training.

Added 2026-10-23 to test a change the literature suggests: with the body of a
trained network held fixed, the last layer is a small least-squares problem.
Its outputs are linear in the last layer's weights W and biases b,

    raw[n] = W h[n] + b,      h[n] the outputs of the last hidden layer,

and the pooled loss is the mean of the squared weighted residual r[n](raw[n]).
The residual is not linear in raw (the field and the rates are taken at the
stage states), so the problem is solved by Levenberg-Marquardt: at each
iteration, with D[n] = d r[n] / d raw[n] (4J x 4J per state, by forward-mode
differentiation),

    J^T J = sum over n of  (D[n]^T D[n]) kron (h~[n] h~[n]^T),   h~ = [h, 1]
    J^T r = sum over n of  (D[n]^T r[n]) kron  h~[n]

and the step solves (J^T J + lambda diag(J^T J)) delta = -J^T r. A step that
does not lower the loss is refused and lambda raised; one that does is taken
and lambda lowered. There are 4J (width + 1) unknowns: 1,032 for 2 stages and
a width of 128.

Nothing of a run is changed. The refitted weights are written beside the
snapshot, in

    <store>/runs/<key>/refits/round_<round>_<name of the refit>/
        network.pt     the weights of the whole network, the last layer refitted
        record.json    the settings, the loss before and after, every iteration

and the standard report of a run scores them when it is given the name of the
refit (report_of_a_run.py, --refit).

The states it fits on are those the run would train on next: the protocol of
the run, at the round after the snapshot, from the network as it is.

    PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python \\
        -m rkpinn.training.last_layer_refit --store /data/bfys/gscriven/rkpinn_store \\
        --run <key> --round last --iterations 30 --name gauss_newton_30

Gates: tests/test_literature_changes.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from rkpinn.run_record.configuration import read_configuration
from rkpinn.run_record.provenance import provenance
from rkpinn.run_record.store import Store
from rkpinn.training.checkpoints import RunFolder

ROWS_AT_A_TIME = 4000


def last_linear_layer(network) -> torch.nn.Linear:
    layer = network.body[-1]
    if not isinstance(layer, torch.nn.Linear):
        raise ValueError("the last layer of the body is not linear")
    return layer


def _hidden(network, states, start_planes):
    """The outputs of the last hidden layer, shape (n, width)."""
    inputs = torch.cat([states / network.scale_of_inputs,
                        network.start_plane_as_input(start_planes)[:, None]], dim=1)
    return network.body[:-1](inputs)


def _weighted_residual_of_raw(network, loss, constants, states, start_planes):
    """The function raw (n, 4J) -> weighted residual (n, 4J'), J' the terms."""
    divisor = torch.as_tensor(constants["divisor_per_component"], dtype=torch.float64)
    from rkpinn.losses.stage_residual import stage_residual

    def of(raw):
        emitted = network.output_form.states_from_raw(
            raw.reshape(len(states), network.number_of_outputs, 4), states, start_planes)
        track = network.track_from_emitted(states, start_planes, emitted)
        r = stage_residual(track, network.equation_of_motion, constants["terms"]).residual
        return (r / divisor).reshape(len(states), -1)

    return of


def normal_equations(network, loss, constants, states, start_planes):
    """(J^T J, J^T r, sum of r^2, number of residuals) over the last layer's
    parameters, ordered as [W row by row, then b] per output: index o*(w+1)+i."""
    layer = last_linear_layer(network)
    outputs, width = layer.weight.shape
    size = outputs * (width + 1)
    JTJ = torch.zeros(size, size, dtype=torch.float64)
    JTr = torch.zeros(size, dtype=torch.float64)
    total, count = 0.0, 0
    for i in range(0, len(states), ROWS_AT_A_TIME):
        S, P = states[i:i + ROWS_AT_A_TIME], start_planes[i:i + ROWS_AT_A_TIME]
        with torch.no_grad():
            h = _hidden(network, S, P)
            raw = layer(h)
        h1 = torch.cat([h, torch.ones(len(h), 1, dtype=h.dtype)], dim=1)     # (n, w+1)
        of = _weighted_residual_of_raw(network, loss, constants, S, P)
        columns = []
        r = None
        for o in range(outputs):
            tangent = torch.zeros_like(raw)
            tangent[:, o] = 1.0
            r, column = torch.func.jvp(of, (raw,), (tangent,))
            columns.append(column.detach())
        r = r.detach()
        D = torch.stack(columns, dim=2)                          # (n, R, outputs)
        DTD = torch.einsum("nro,nrp->nop", D, D)                 # (n, outputs, outputs)
        DTr = torch.einsum("nro,nr->no", D, r)                   # (n, outputs)
        for o in range(outputs):
            JTr[o * (width + 1):(o + 1) * (width + 1)] += h1.T @ DTr[:, o]
            for p in range(o, outputs):
                block = h1.T @ (DTD[:, o, p:p + 1] * h1)
                JTJ[o * (width + 1):(o + 1) * (width + 1),
                    p * (width + 1):(p + 1) * (width + 1)] += block
                if p != o:
                    JTJ[p * (width + 1):(p + 1) * (width + 1),
                        o * (width + 1):(o + 1) * (width + 1)] += block.T
        total += float((r * r).sum())
        count += r.numel()
    return JTJ, JTr, total, count


def _loss_value(network, loss, constants, states, start_planes):
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(states), ROWS_AT_A_TIME):
            terms = loss.squared_terms(network.predict(states[i:i + ROWS_AT_A_TIME],
                                                       start_planes[i:i + ROWS_AT_A_TIME]),
                                       None, constants)
            total += float(terms.sum())
            count += terms.numel()
    return total / count


def _parameters_as_vector(layer):
    return torch.cat([layer.weight.detach(), layer.bias.detach()[:, None]], dim=1).reshape(-1)


def _set_parameters(layer, vector):
    outputs, width = layer.weight.shape
    block = vector.reshape(outputs, width + 1)
    with torch.no_grad():
        layer.weight.copy_(block[:, :width])
        layer.bias.copy_(block[:, width])


def refit_last_layer(network, loss, constants, states, start_planes, *, iterations,
                     first_damping, report=print) -> dict:
    """Levenberg-Marquardt on the last layer. Changes the network in place.
    Returns the record of the refit."""
    if not hasattr(loss, "squared_terms") or loss.needs_labels:
        raise ValueError("the refit is for a loss built on the stage residual")
    states = torch.as_tensor(states, dtype=torch.float64)
    start_planes = torch.as_tensor(start_planes, dtype=torch.float64)
    layer = last_linear_layer(network)
    damping = float(first_damping)
    before = _loss_value(network, loss, constants, states, start_planes)
    now = before
    rows = []
    for k in range(int(iterations)):
        started = time.time()
        JTJ, JTr, _, _ = normal_equations(network, loss, constants, states, start_planes)
        diagonal = torch.diagonal(JTJ).clamp_min(1e-300)
        x = _parameters_as_vector(layer)
        taken = False
        for _ in range(12):
            step = torch.linalg.solve(JTJ + damping * torch.diag(diagonal), -JTr)
            _set_parameters(layer, x + step)
            trial = _loss_value(network, loss, constants, states, start_planes)
            if trial < now:
                taken = True
                now = trial
                damping = max(damping / 3.0, 1e-12)
                break
            _set_parameters(layer, x)
            damping *= 4.0
        rows.append({"iteration": k + 1, "loss": now, "damping": damping, "taken": taken,
                     "seconds": round(time.time() - started, 1)})
        report("  refit iteration %d: loss %.4e, damping %.1e, %s"
               % (k + 1, now, damping, "step taken" if taken else "no step lowers the loss"))
        if not taken:
            break
    return {"loss_before": before, "loss_after": now, "iterations": rows,
            "number_of_states": int(len(states)),
            "number_of_parameters": int(layer.weight.numel() + layer.bias.numel())}


def refit_a_snapshot(store: Store, key: str, *, round_number, iterations, first_damping,
                     name, allow_uncommitted_changes=False, report=print) -> str:
    """Refit the last layer of one snapshot of a run. Returns the folder written."""
    from rkpinn.training.round_trainer import build_run
    record = provenance(allow_uncommitted_changes)
    folder = RunFolder(store, key)
    state = folder.state()
    if round_number == "last":
        round_number = state["rounds_done"]
    round_number = int(round_number)
    where = folder.path("refits", "round_%04d_%s" % (round_number, name))
    Store.refuse_if_present(where)
    configuration = read_configuration(folder.path("configuration.yaml"))
    run = build_run(configuration, store, folder.constants())
    run.network.load_state_dict(folder.weights_of_round(round_number))
    states_of_round = run.protocol.states_of_round(
        round_number + 1, configuration["seed"], run.tracks, run.layout, run.network,
        run.target)
    made = refit_last_layer(run.network, run.loss, run.constants["loss"],
                            states_of_round.states, states_of_round.start_planes_mm,
                            iterations=iterations, first_damping=first_damping,
                            report=report)
    made.update(run_key=key, round=round_number, name=name,
                states_from_round=round_number + 1, source_of_states=states_of_round.source,
                settings=dict(iterations=int(iterations), first_damping=float(first_damping)),
                provenance=record)
    writing = where + ".writing"
    os.makedirs(writing, exist_ok=True)
    torch.save(run.network.state_dict(), os.path.join(writing, "network.pt"))
    with open(os.path.join(writing, "record.json"), "w") as handle:
        json.dump(made, handle, indent=1)
    os.replace(writing, where)
    report("%s: refit of round %d written to refits/%s; loss %.4e -> %.4e"
           % (key, round_number, os.path.basename(where), made["loss_before"],
              made["loss_after"]))
    return where


def weights_of_a_refit(folder: RunFolder, round_number: int, name: str) -> dict:
    return torch.load(folder.path("refits", "round_%04d_%s" % (round_number, name),
                                  "network.pt"), weights_only=True)


def main(arguments=None):
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    parser = argparse.ArgumentParser(description="Refit the last layer of a snapshot.")
    parser.add_argument("--store", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--round", required=True, help="the round of the snapshot, or last")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--first-damping", type=float, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--allow-uncommitted-changes", action="store_true")
    a = parser.parse_args(arguments)
    return refit_a_snapshot(Store(a.store), a.run, round_number=a.round,
                            iterations=a.iterations, first_damping=a.first_damping,
                            name=a.name, allow_uncommitted_changes=a.allow_uncommitted_changes,
                            report=lambda text: print(text, flush=True))


if __name__ == "__main__":
    main()
