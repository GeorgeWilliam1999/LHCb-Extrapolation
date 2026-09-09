#!/usr/bin/env python
"""Train ONE point of the width x depth x q x mode x seed grid, and score it
per stratum and per direction.

The optimiser protocol must not drift from the one every earlier discrete-time
experiment used, or the grid stops being comparable with them, so this script
does not re-implement it: it calls `../../_shared/train.py`'s `main` and replaces
three of its module-level names before doing so.

    OneStepNetwork -> a factory returning `grid_model.GridResidualNetwork`
                      built with this dataset's Gauss nodes, its (z0, dz)
                      normalisation and the MagUp field twin
    data_loss      -> `grid_model.grid_data_loss`, which normalises by the
                      per-sample residual scale instead of the population-wide
                      out_scale
    load_dataset   -> the same loader with the training split subsampled to
                      `--n-train` rows, evenly across the six strata

`physics_loss` is untouched: it is computed from the reconstructed ABSOLUTE
states, so it only ever needed the wrapper's forward.

So the protocol is, unchanged: fp64, full-batch L-BFGS with `max_iter` 200,
strong Wolfe, history 120; a checkpoint and a history row after every restart
(hence a resume on rerun); stall = two consecutive restarts each improving the
loss by less than 1%; then a confirmation pass with a fresh optimiser, which
counts as converged only if it re-stalls within two restarts with the endpoint
medians unchanged; `--outer-cap 400`.

## What is added

The json the shared trainer writes is reopened and extended, so that step C4
can read the error(dz, q) tables straight from the run records without
re-loading a single checkpoint:

    n_train                 the rows actually trained on
    by_stratum              a flat list of score rows, one per
                            (split, stratum, direction), covering val and test,
                            the six strata plus "all", and both directions plus
                            "all". Each row carries n, the endpoint median and
                            p95 in um, the stage median, the endpoint slope
                            median in mrad, rho (mean and median) and the
                            straight-line median on the same rows.
    straight_by_stratum     the straight-line endpoint median per stratum, for
                            val and test, pulled out on its own because it is
                            the baseline every cell is read against.

Every metric is `../../_shared/evaluate.score_against_reference`, the same
function the whole line has reported since the baseline, applied to a row mask.

## Extra arguments

    --n-train N       rows to train on, spread evenly over the six strata
                      (default: the whole training split of the dataset)
    --node-profile flat|poly    overrides the dataset's stored profile
    --init-check      build the model, verify the algebra and the finiteness of
                      the loss and the gradients, write
                      results/grid_init_check.json, and stop

    PYTHONNOUSERSITE=1 python train_grid.py --data results/grid_q20.npz \\
        --mode physics --seed 0 --q 20 --width 256 --depth 8 --n-train 4000 \\
        --out results --tag w256_d8_q20_physics_s0
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
from _shared.evaluate import predict, score_against_reference
from _shared.model import physics_loss
from _shared.prepare import STRATUM_NAMES
from _shared.reference import gauss_legendre, make_field
from grid_model import (FIELD_WHICH, GridResidualNetwork, grid_data_loss,
                        straight_line_states)

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
SUBSET_SEED = 20260907
EVAL_SPLITS = ("val", "test")
DIRECTIONS = (("all", None), ("forward", 1), ("backward", -1))


# ------------------------------------------------------------ the row pick --
def subset_train(data, n_train):
    """Cut the training split to `n_train` rows, evenly over the six strata.

    The draw is seeded per stratum and does not depend on q, so every point of
    the grid trains on the same rows and two runs differ only in the thing the
    grid varies.
    """
    if n_train is None:
        return data
    S = np.asarray(data["train_STRATUM"])
    if n_train >= len(S):
        return data
    per = int(np.ceil(n_train / len(STRATUM_NAMES)))
    keep = []
    for i in range(len(STRATUM_NAMES)):
        pool = np.flatnonzero(S == i)
        rng = np.random.default_rng(SUBSET_SEED + i)
        keep.append(pool if len(pool) <= per
                    else np.sort(pool[rng.permutation(len(pool))[:per]]))
    keep = np.concatenate(keep)
    out = dict(data)
    for k in list(data):
        if k.startswith("train_"):
            out[k] = np.asarray(data[k])[keep]
    return out


# ------------------------------------------------------------ the installer --
def install(data, field_which, node_profile, n_train):
    """Point the shared trainer at the grid model, the twin and the subset."""
    fld = make_field(field_which)

    def factory(q, in_scale, out_scale, width=50, depth=4, n_extra=0):
        if n_extra != 2:
            raise SystemExit("the grid arm needs a general-leg dataset "
                             "(n_extra = 2), not n_extra = %d" % n_extra)
        return GridResidualNetwork(
            q, in_scale, out_scale, width=width, depth=depth, n_extra=2,
            c=np.asarray(data["c"]), extra_mean=np.asarray(data["extra_mean"]),
            extra_scale=np.asarray(data["extra_scale"]), field=fld,
            node_profile=node_profile)

    original_loader = shared_train.load_dataset

    def loader(path):
        return subset_train(original_loader(path), n_train)

    shared_train.OneStepNetwork = factory
    shared_train.data_loss = grid_data_loss
    shared_train.load_dataset = loader
    return factory


# ------------------------------------------------------------- the scoring --
def by_stratum_scores(model, data):
    """One score row per (split, stratum, direction), val and test.

    The network is run **once** per split and the masks are applied to its
    output, rather than re-predicting for each of the twenty-one cells: the
    scores are identical and the farm pays one forward pass instead of
    twenty-one.
    """
    rows = []
    for split in EVAL_SPLITS:
        S = np.asarray(data["%s_S" % split])
        ref = np.asarray(data["%s_ref" % split])
        dz = np.asarray(data["%s_dz" % split])
        extra = np.asarray(data["%s_extra" % split])
        out = predict(model, S, extra)
        STRAT = np.asarray(data["%s_STRATUM" % split])
        DIR = np.asarray(data["%s_DIRECTION" % split])
        groups = [("all", -1, np.ones(len(STRAT), dtype=bool))]
        groups += [(name, i, STRAT == i) for i, name in enumerate(STRATUM_NAMES)]
        for name, index, base in groups:
            for dname, sign in DIRECTIONS:
                m = base if sign is None else (base & (DIR == sign))
                if not m.any():
                    continue
                sc = score_against_reference(out[m], S[m], ref[m], dz[m])
                rows.append({"split": split, "stratum": index,
                             "stratum_name": name, "direction": dname, **sc})
    return rows


def rebuild_model(data, factory, args):
    """The trained model, rebuilt from its checkpoint."""
    torch.manual_seed(args.seed)
    model = factory(int(data["q"]), data["in_scale"], data["out_scale"],
                    width=args.width, depth=args.depth, n_extra=2)
    ckpt = os.path.join(args.out, args.tag + ".pt")
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    return model


# ----------------------------------------------------------- the init check --
def init_check(data, factory, width, depth, seed, out_dir, tag, field_which):
    """The wrapper's algebra, and the finiteness of the loss and the gradients.

    1. with the last layer's weight and bias zeroed the model output is the
       straight line EXACTLY, so the straight-line term and the node
       bookkeeping are right and nothing else leaks in;
    2. at the shared trainer's own initialisation the output is the straight
       line plus scale * (a small raw output), i.e. the model starts within one
       residual scale of the straight line;
    3. the physics loss, the residual data loss and every gradient are finite.

    The last layer is NOT zeroed for the real runs - a zero last layer makes
    every earlier layer's gradient exactly zero on the first step. Zeroing it
    is the algebra check and nothing else.
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
    _, A_np, b_np = gauss_legendre(q)
    A, b = torch.tensor(A_np), torch.tensor(b_np)
    rates = shared_train.LHCbRates(make_field(field_which))

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
        "field": field_which, "n_train": int(len(S)),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "n_outputs": 4 * (q + 1),
        "straight_term_vs_numpy_max_abs_mm": float(
            np.abs(straight_t.numpy() - straight_np).max()),
        "residual_scale": {
            "min_pos_mm": float(sc[:, :, 0].min()),
            "median_pos_mm": float(sc[:, :, 0].median()),
            "min_slope": float(sc[:, :, 2].min()),
        },
        "raw_output_at_init": {
            "max_abs": float(raw0.abs().max()),
            "median_abs": float(raw0.abs().median()),
        },
        "init_offset_from_straight_line_in_units_of_scale": {
            "max": float((dev0 / sc).max()),
            "median": float((dev0 / sc).median()),
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
    rec["physics_loss_at_init"] = float(lp.detach())
    rec["physics_grad_max_abs"] = float(max(
        p.grad.abs().max().item() for p in model.parameters()
        if p.grad is not None))
    rec["physics_grad_all_finite"] = bool(all(
        torch.isfinite(p.grad).all().item() for p in model.parameters()
        if p.grad is not None))
    model.zero_grad(set_to_none=True)
    ld = grid_data_loss(model, S, REF, EX)
    ld.backward()
    rec["data_loss_at_init"] = float(ld.detach())
    rec["data_grad_max_abs"] = float(max(
        p.grad.abs().max().item() for p in model.parameters()
        if p.grad is not None))
    rec["data_grad_all_finite"] = bool(all(
        torch.isfinite(p.grad).all().item() for p in model.parameters()
        if p.grad is not None))
    rec["all_finite"] = bool(np.isfinite([
        rec["physics_loss_at_init"], rec["data_loss_at_init"],
        rec["physics_grad_max_abs"], rec["data_grad_max_abs"]]).all())
    rec["passes"] = bool(
        rec["all_finite"]
        and rec["zeroed_last_layer_vs_straight_line_max_abs_mm"] == 0.0
        and rec["straight_term_vs_numpy_max_abs_mm"] < 1e-9)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "grid_init_check.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))
    print("wrote %s" % path)
    return rec


# -------------------------------------------------------------------- main --
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--node-profile", choices=("flat", "poly"), default=None)
    ap.add_argument("--init-check", action="store_true")
    ap.add_argument("--field", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="grid_init_check")
    known, _ = ap.parse_known_args(argv)

    d = np.load(os.path.abspath(known.data))
    data = subset_train({k: d[k] for k in d.files}, known.n_train)
    if str(data.get("kind", "general")) != "general":
        raise SystemExit("the grid arm needs a general-leg dataset")
    stored = (str(data["residual_node_profile"])
              if "residual_node_profile" in data else "flat")
    profile = known.node_profile or stored
    field_which = known.field or (str(data["field"]) if "field" in data
                                  else FIELD_WHICH)
    factory = install(data, field_which, profile, known.n_train)
    print("grid arm: q %s, %dx%d, node profile '%s', field '%s', %d training "
          "rows, dataset %s"
          % (data["q"], known.depth, known.width, profile, field_which,
             len(data["train_S"]), os.path.basename(known.data)), flush=True)

    if known.init_check:
        return init_check(data, factory, known.width, known.depth, known.seed,
                          known.out, known.tag, field_which)

    # everything the shared trainer does not know about is stripped here
    passthrough, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in ("--node-profile", "--n-train"):
            skip = True
            continue
        if tok.startswith("--node-profile=") or tok.startswith("--n-train="):
            continue
        if tok == "--init-check":
            continue
        passthrough.append(tok)
    record = shared_train.main(passthrough)

    # -- the per-stratum and per-direction block ----------------------------
    model = rebuild_model(data, factory, known)
    json_path = os.path.join(known.out, known.tag + ".json")
    with open(json_path) as f:
        rec = json.load(f)
    rec["n_train"] = int(len(data["train_S"]))
    rec["n_train_requested"] = known.n_train
    rec["node_profile"] = profile
    rec["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    rec["by_stratum"] = by_stratum_scores(model, data)
    rec["straight_by_stratum"] = {
        r["split"]: {} for r in rec["by_stratum"]}
    for r in rec["by_stratum"]:
        if r["direction"] == "all":
            rec["straight_by_stratum"][r["split"]][r["stratum_name"]] = \
                r["straight_med_um"]
    with open(json_path, "w") as f:
        json.dump(rec, f, indent=1)
    print("STRATA " + " ".join(
        "%s=%.4g" % (r["stratum_name"].split()[0], r["endpoint_med_um"])
        for r in rec["by_stratum"]
        if r["split"] == "test" and r["direction"] == "all"), flush=True)
    return rec


if __name__ == "__main__":
    main()
