#!/usr/bin/env python
"""Train ONE (mode, seed) residual network, on the shared trainer's own loop.

The optimiser protocol must not drift from wave 1's, or the comparison stops
meaning anything: same fp64, same full-batch L-BFGS (max_iter 200, strong
Wolfe, history 120), same stall criterion (two consecutive restarts each
improving by less than 1%), same checkpoint-every-restart and therefore the
same resume behaviour, same confirmation pass, same json fields. So this script
does not re-implement that loop - it calls `_shared/train.py`'s `main` and
replaces exactly two of its module-level names before doing so:

    OneStepNetwork -> a factory returning `ResidualOneStepNetwork` built with
                      this dataset's Gauss nodes and (z0, dz) normalisation
    data_loss      -> `residual_data_loss`, which normalises by the per-sample
                      residual scale instead of the population-wide out_scale

`physics_loss` is untouched: it is computed from the reconstructed ABSOLUTE
states, so it only ever needed the wrapper's forward.

Every argument is the shared trainer's; the defaults are the shared trainer's
(including `--outer-cap 400`). Two are added:

    --node-profile flat|poly   overrides the profile stored in the dataset by
                               prepare_residual.py
    --init-check               build the model, verify the algebra and the
                               finiteness of the loss and the gradients, write
                               results/residual_init_check.json, and stop

    PYTHONNOUSERSITE=1 python train_residual.py \\
        --data results/general_legs_residual.npz --mode physics --seed 0 \\
        --width 100 --depth 4 --out results --tag residual_w100_physics_s0
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json
import sys

import numpy as np
import torch

import use_shared                                        # noqa: F401
import _shared.train as shared_train
from _shared.model import physics_loss
from _shared.reference import gauss_legendre, make_field
from residual_model import (ResidualOneStepNetwork, residual_data_loss,
                            straight_line_states)

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    d = np.load(os.path.abspath(path))
    return {k: d[k] for k in d.files}


def install(data, field_which, node_profile):
    """Point the shared trainer at the residual model and the residual twin."""
    fld = make_field(field_which)

    def factory(q, in_scale, out_scale, width=50, depth=4, n_extra=0):
        if n_extra != 2:
            raise SystemExit("the residual arm needs a general-leg dataset "
                             "(n_extra = 2), not n_extra = %d" % n_extra)
        return ResidualOneStepNetwork(
            q, in_scale, out_scale, width=width, depth=depth, n_extra=2,
            c=np.asarray(data["c"]), extra_mean=np.asarray(data["extra_mean"]),
            extra_scale=np.asarray(data["extra_scale"]), field=fld,
            node_profile=node_profile)

    shared_train.OneStepNetwork = factory
    shared_train.data_loss = residual_data_loss
    return factory


def init_check(data, factory, width, depth, seed, out_dir, tag):
    """The wrapper's algebra, and the finiteness of the loss and the gradients.

    Three things are asserted before any farm time is spent:

    1. with the last layer's weight and bias set to zero the model output is
       the straight line EXACTLY - i.e. the wrapper's straight-line term and
       node bookkeeping are right, and nothing else leaks in;
    2. at the shared trainer's own initialisation the output is the straight
       line plus scale * (a small raw output), so the model starts within one
       residual scale of the straight line rather than hundreds of millimetres
       away as wave 1 did;
    3. the physics loss, the residual data loss and every gradient are finite.
    """
    q = int(data["q"])
    torch.manual_seed(seed)
    model = factory(q, data["in_scale"], data["out_scale"], width=width,
                    depth=depth, n_extra=2)
    S = torch.tensor(np.asarray(data["train_S"]))
    EX = torch.tensor(np.asarray(data["train_extra"]))
    REF = torch.tensor(np.asarray(data["train_ref"]))
    ZN = torch.tensor(np.asarray(data["train_znodes"]))
    DZ = torch.tensor(np.asarray(data["train_dz"]))
    c, A_np, b_np = gauss_legendre(q)
    A, b = torch.tensor(A_np), torch.tensor(b_np)
    rates = shared_train.LHCbRates(make_field(str(data["field"])))

    straight_np = straight_line_states(np.asarray(data["train_S"]),
                                       np.asarray(data["train_dz"]),
                                       np.asarray(data["c"]))
    with torch.no_grad():
        raw0 = model.raw(S, EX)
        out0 = model(S, EX)
        sc = model.residual_scale(S, EX)
        straight_t = model.straight(S, EX)
    dev0 = (out0 - straight_t).abs()
    rec = {
        "tag": tag, "seed": seed, "width": width, "depth": depth, "q": q,
        "n_train": int(len(S)),
        "straight_term_vs_numpy_max_abs_mm": float(
            np.abs(straight_t.numpy() - straight_np).max()),
        "raw_output_at_init": {
            "max_abs": float(raw0.abs().max()),
            "median_abs": float(raw0.abs().median()),
        },
        "init_offset_from_straight_line_in_units_of_scale": {
            "max": float((dev0 / sc).max()),
            "median": float((dev0 / sc).median()),
        },
        "init_offset_from_straight_line_um": {
            "median_endpoint_position": float(
                dev0[:, -1, :2].max(dim=1).values.median() * 1e3),
            "p95_endpoint_position": float(
                np.quantile(dev0[:, -1, :2].max(dim=1).values.numpy(), 0.95) * 1e3),
        },
    }

    # 1. the last layer zeroed -> the straight line, exactly
    zeroed = factory(q, data["in_scale"], data["out_scale"], width=width,
                     depth=depth, n_extra=2)
    zeroed.load_state_dict(model.state_dict())
    with torch.no_grad():
        zeroed.net[-1].weight.zero_()
        zeroed.net[-1].bias.zero_()
        out_z = zeroed(S, EX)
    rec["zeroed_last_layer_vs_straight_line_max_abs_mm"] = float(
        (out_z - straight_t).abs().max())

    # 3. the losses and the gradients
    lp = physics_loss(model, rates, S, DZ, ZN, A, b, EX)
    lp.backward()
    gp = [p.grad.abs().max().item() for p in model.parameters()
          if p.grad is not None]
    rec["physics_loss_at_init"] = float(lp)
    rec["physics_grad_max_abs"] = float(max(gp))
    rec["physics_grad_all_finite"] = bool(
        all(torch.isfinite(p.grad).all().item() for p in model.parameters()
            if p.grad is not None))
    model.zero_grad(set_to_none=True)
    ld = residual_data_loss(model, S, REF, EX)
    ld.backward()
    gd = [p.grad.abs().max().item() for p in model.parameters()
          if p.grad is not None]
    rec["data_loss_at_init"] = float(ld)
    rec["data_grad_max_abs"] = float(max(gd))
    rec["data_grad_all_finite"] = bool(
        all(torch.isfinite(p.grad).all().item() for p in model.parameters()
            if p.grad is not None))
    rec["all_finite"] = bool(np.isfinite([rec["physics_loss_at_init"],
                                          rec["data_loss_at_init"],
                                          rec["physics_grad_max_abs"],
                                          rec["data_grad_max_abs"]]).all())
    rec["passes"] = bool(
        rec["all_finite"]
        and rec["zeroed_last_layer_vs_straight_line_max_abs_mm"] < 1e-9
        and rec["straight_term_vs_numpy_max_abs_mm"] < 1e-9)
    path = os.path.join(out_dir, "residual_init_check.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))
    print("wrote %s" % path)
    return rec


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--node-profile", choices=("flat", "poly"), default=None)
    ap.add_argument("--init-check", action="store_true")
    ap.add_argument("--field", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="residual_init_check")
    known, _ = ap.parse_known_args(argv)

    data = load(known.data)
    if str(data.get("kind", "general")) != "general":
        raise SystemExit("the residual arm needs the general-leg dataset")
    stored = (str(data["residual_node_profile"])
              if "residual_node_profile" in data else "flat")
    profile = known.node_profile or stored
    field_which = known.field or str(data["field"])
    factory = install(data, field_which, profile)
    print("residual arm: node profile '%s', field '%s', dataset %s"
          % (profile, field_which, os.path.basename(known.data)), flush=True)

    if known.init_check:
        return init_check(data, factory, known.width, known.depth, known.seed,
                          known.out, known.tag)

    # everything the shared trainer does not know about is stripped here
    passthrough, skip = [], False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok == "--node-profile":
            skip = True
            continue
        if tok.startswith("--node-profile="):
            continue
        if tok == "--init-check":
            continue
        passthrough.append(tok)
    return shared_train.main(passthrough)


if __name__ == "__main__":
    main()
