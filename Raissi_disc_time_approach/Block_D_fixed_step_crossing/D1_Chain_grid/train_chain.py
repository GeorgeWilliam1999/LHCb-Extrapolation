#!/usr/bin/env python
"""D1 - one chain of Block D: N fixed-step networks trained one after another
across the magnet, exactly as the paper's discrete-time scheme steps.

For a step count N and a stage count q, the crossing z0 -> z1 is cut into N
equal legs [z_k, z_k + dz], dz = L/N, and one network is trained per leg, in
order:

    leg 0 trains on the real start states at z0 (D0's `S0`);
    leg k trains on the states leg k-1 PREDICTED at z_k, for the same particles,
          so the training data of every later leg carries the error of the
          legs before it - Raissi, Perdikaris and Karniadakis (2019), section 3:
          "A Runge-Kutta time-stepping scheme would then use this prediction as
          initial data for the next step and proceed to train again".

Every leg is a `chain_model.FrozenResidualNetwork` (two hidden layers of 128,
q Gauss-Legendre stages, straight-line-residual output) trained with the
shared trainer's verified protocol - fp64 full-batch L-BFGS restarted to a
stall and confirmed with a fresh optimiser - on the label-free physics loss
alone. Nothing is retrained: `_shared/train.py` is called through its `main`
with the model class swapped in, so the optimiser, the stall rule, the
confirmation pass, the checkpoint-every-restart and the resume are the ones
every earlier experiment used.

The validation and test particles go through the same chain, leg after leg,
and are scored at every plane against D0's RK6 truth; at the end the chain's
prediction on z1 is scored against the RK6 endpoint (the fine reference), and,
carried from z1 to the particle's own SciFi plane by the same RK6, against the
particle's REAL state there (the data ground truth, material included).

Each leg also has its own record (`leg<k>.json`, written by the shared
trainer): the leg's error against the RK6 propagation of ITS OWN inputs, i.e.
how good the network is at the step it was given, separate from the error it
inherited.

Resumable: a leg whose record exists is not trained again (its checkpoint is
reloaded and its predictions rebuilt); a leg that was interrupted mid-training
resumes from its last restart. Rerun the identical command line to continue.

Usage (one farm job):
    python train_chain.py --N 4 --q 8 --out results

Outputs in <out>/N<NNN>_q<qq>/:
    leg<kkk>.pt, leg<kkk>_history.csv, leg<kkk>.json   per leg (the trainer's)
    leg<kkk>_scale.json                                the leg's input scale
    states.npz    the chain's predicted states at every plane, per split
    chain.json    the chain's scores, the per-leg summary, the cost
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json
import time

import numpy as np
import torch

import use_shared                                        # noqa: F401
from _shared import train as shared_train
from _shared.evaluate import predict
from _shared.reference import RK6_STEP, gauss_legendre, make_field, rk6_rows
from chain_model import DEPTH, WIDTH, FrozenResidualNetwork, straight_line_states
from metrics import chain_scores, pos_err_um, slope_err_mrad, stats

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "D0_Crossing_dataset", "results",
                            "crossing_particles.npz")
SPLITS = ("train", "val", "test")
SCORED = ("val", "test")


# ------------------------------------------------------------- the pieces --
def chain_dir(out, N, q):
    return os.path.join(out, "N%03d_q%02d" % (N, q))


def leg_tag(k):
    return "leg%03d" % k


def rk6_through(S_in, znodes_out, z_start, fld):
    """(n, q+1, 5): RK6 from S_in at z_start through each node in turn."""
    ref = np.empty((len(S_in), len(znodes_out), 5))
    cur, zprev = np.asarray(S_in, dtype=np.float64).copy(), float(z_start)
    for j, zt in enumerate(znodes_out):
        cur = rk6_rows(cur, zprev, float(zt), step=RK6_STEP, field=fld)
        ref[:, j] = cur
        zprev = float(zt)
    return ref


def leg_dataset(states, k, zk, dz, q, c, D, field):
    """The frozen-leg dataset of leg k, in the shared trainer's format."""
    znodes = zk + c * dz
    zout = np.append(znodes, zk + dz)
    fld = make_field(field)
    arrays = dict(kind="frozen", znodes=znodes, zout=zout, z0=zk, z1=zk + dz,
                  q=q, c=c, field=field)
    for s in SPLITS:
        S = states[s][:, k]
        arrays["%s_S" % s] = S
        arrays["%s_P" % s] = np.asarray(D["%s_P" % s])
        arrays["%s_ref" % s] = rk6_through(S, zout, zk, fld)
        arrays["%s_z0" % s] = np.full(len(S), zk)
        arrays["%s_dz" % s] = np.full(len(S), dz)
    S_tr = arrays["train_S"]
    arrays["in_scale"] = S_tr.std(axis=0)
    straight = straight_line_states(S_tr, dz, c)
    arrays["out_scale"] = np.append(straight.reshape(-1, 4).std(axis=0), S_tr[:, 4].std())
    return arrays


def install(factory):
    """Point the shared trainer at the fixed-step network."""
    shared_train.OneStepNetwork = factory


def make_factory(c, zk, dz, fld, width, depth):
    def factory(q, in_scale, out_scale, width=width, depth=depth, n_extra=0):
        return FrozenResidualNetwork(q, in_scale, out_scale, width=width,
                                     depth=depth, c=c, z0=zk, dz=dz, field=fld)
    return factory


def rebuild(factory, data, ckpt, seed, width, depth):
    torch.manual_seed(seed)
    model = factory(int(data["q"]), np.asarray(data["in_scale"]),
                    np.asarray(data["out_scale"]), width=width, depth=depth)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    return model











# ------------------------------------------------------------------ main --
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, required=True, help="steps across the crossing")
    ap.add_argument("--q", type=int, required=True, help="Gauss-Legendre stages")
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--seed", type=int, default=0, help="one seed, every leg")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--n-train", type=int, default=None,
                    help="cap on the training particles (default: D0's own cap)")
    ap.add_argument("--n-eval", type=int, default=None,
                    help="cap on val/test particles (smoke tests only)")
    ap.add_argument("--outer-cap", type=int, default=400)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--legs", type=int, default=None,
                    help="train only the first this-many legs (smoke tests only)")
    ap.add_argument("--no-confirm", action="store_true",
                    help="skip the confirmation pass (smoke tests only)")
    a = ap.parse_args(argv)
    t_start = time.time()

    D = {k: v for k, v in np.load(os.path.abspath(a.data), allow_pickle=False).items()}
    Z0, Z1, L = float(D["z0"]), float(D["z1"]), float(D["L"])
    n_max = int(D["n_max"])
    if n_max % a.N:
        raise SystemExit("N = %d does not divide the D0 plane grid (%d)" % (a.N, n_max))
    stride = n_max // a.N
    dz = L / a.N
    field = str(D["field"])
    fld = make_field(field)
    c, A, b = gauss_legendre(a.q)
    planes = Z0 + np.arange(a.N + 1) * dz

    for s, cap in (("train", a.n_train), ("val", a.n_eval), ("test", a.n_eval)):
        if cap is not None:
            for k in list(D):
                if k.startswith(s + "_"):
                    D[k] = D[k][:cap]

    cdir = chain_dir(a.out, a.N, a.q)
    os.makedirs(cdir, exist_ok=True)
    states_path = os.path.join(cdir, "states.npz")
    if os.path.exists(states_path):
        st = np.load(states_path)
        states = {s: st["%s_states" % s] for s in SPLITS}
        done = st["done"]
    else:
        states = {s: np.full((len(D["%s_S0" % s]), a.N + 1, 5), np.nan) for s in SPLITS}
        for s in SPLITS:
            states[s][:, 0] = D["%s_S0" % s]
        done = np.zeros(a.N + 1, dtype=bool)
        done[0] = True

    def save_states():
        np.savez_compressed(states_path, done=done,
                            **{"%s_states" % s: states[s] for s in SPLITS})

    legs = []
    n_legs = a.N if a.legs is None else min(a.N, a.legs)
    for k in range(n_legs):
        zk = planes[k]
        tag = leg_tag(k)
        ckpt = os.path.join(cdir, tag + ".pt")
        rec_path = os.path.join(cdir, tag + ".json")
        data_path = os.path.join(cdir, tag + "_data.npz")
        scale_path = os.path.join(cdir, tag + "_scale.json")
        t_leg = time.time()
        assert done[k], "plane %d has no states to start leg %d from" % (k, k)

        factory = make_factory(c, zk, dz, fld, a.width, a.depth)
        install(factory)

        if os.path.exists(rec_path) and os.path.exists(ckpt) and done[k + 1]:
            with open(rec_path) as f:
                record = json.load(f)
            print("leg %d/%d already trained and applied; skipping" % (k + 1, a.N), flush=True)
        else:
            if os.path.exists(scale_path) and os.path.exists(data_path):
                data = {kk: v for kk, v in np.load(data_path, allow_pickle=False).items()}
            else:
                t0 = time.time()
                data = leg_dataset(states, k, zk, dz, a.q, c, D, field)
                np.savez_compressed(data_path, **data)
                with open(scale_path, "w") as f:
                    json.dump({"in_scale": data["in_scale"].tolist(),
                               "out_scale": data["out_scale"].tolist(),
                               "z0": zk, "z1": zk + dz, "q": a.q,
                               "rk6_wall_s": round(time.time() - t0, 1)}, f, indent=1)
                print("leg %d/%d: dataset built in %.0f s (z %.1f -> %.1f mm)"
                      % (k + 1, a.N, time.time() - t0, zk, zk + dz), flush=True)
            if os.path.exists(rec_path):
                with open(rec_path) as f:
                    record = json.load(f)
                print("leg %d/%d: record exists (converged = %s); reusing the checkpoint"
                      % (k + 1, a.N, record.get("converged")), flush=True)
            else:
                record = shared_train.main([
                    "--data", data_path, "--mode", "physics", "--seed", str(a.seed),
                    "--q", str(a.q), "--width", str(a.width), "--depth", str(a.depth),
                    "--out", cdir, "--tag", tag, "--field", field,
                    "--outer-cap", str(a.outer_cap), "--max-iter", str(a.max_iter)]
                    + (["--no-confirm"] if a.no_confirm else []))
            model = rebuild(factory, data, ckpt, a.seed, a.width, a.depth)
            for s in SPLITS:
                S = states[s][:, k]
                out = predict(model, S)
                states[s][:, k + 1] = np.concatenate([out[:, -1, :], S[:, 4:5]], axis=1)
            done[k + 1] = True
            save_states()
            if os.path.exists(data_path):
                os.remove(data_path)

        # the chain-level score at the plane this leg lands on
        plane_idx = (k + 1) * stride
        leg_row = {"leg": k, "z0": float(zk), "z1": float(zk + dz),
                   "converged": bool(record.get("converged", False)),
                   "restarts": int(record.get("restarts", 0)),
                   "final_loss": float(record.get("final_loss", float("nan"))),
                   "wall_s": float(record.get("wall_s", 0.0)),
                   "own_step_test_endpoint_med_um": float(record["test"]["endpoint_med_um"]),
                   "own_step_test_straight_med_um": float(record["test"]["straight_med_um"]),
                   "own_step_val_endpoint_med_um": float(record["val"]["endpoint_med_um"])}
        for s in SCORED:
            truth = D["%s_truth" % s][:, plane_idx]
            pred = states[s][:, k + 1]
            leg_row["chain_%s_pos_med_um" % s] = float(np.median(pos_err_um(pred, truth)))
            leg_row["chain_%s_pos_p95_um" % s] = float(np.quantile(pos_err_um(pred, truth), 0.95))
            leg_row["chain_%s_slope_med_mrad" % s] = float(np.median(slope_err_mrad(pred, truth)))
        leg_row["leg_wall_s_this_run"] = round(time.time() - t_leg, 1)
        legs.append(leg_row)
        print("leg %d/%d done: own-step test median %.3g um, chain test median at plane "
              "%d: %.3g um (converged %s, %d restarts)"
              % (k + 1, a.N, leg_row["own_step_test_endpoint_med_um"], k + 1,
                 leg_row["chain_test_pos_med_um"], leg_row["converged"],
                 leg_row["restarts"]), flush=True)

    if n_legs < a.N:
        print("smoke run: %d of %d legs; no chain.json" % (n_legs, a.N))
        return None

    # -- the chain's verdict: metrics.chain_scores, per component ----------------
    result = {"N": a.N, "q": a.q, "dz_mm": dz, "seed": a.seed, "width": a.width,
              "depth": a.depth, "field": field, "planes": planes.tolist(),
              "n_parameters_per_leg": int(sum(p.numel() for p in model.parameters())),
              "n_train": int(len(D["train_S0"])),
              "legs": legs,
              "all_legs_converged": bool(all(r["converged"] for r in legs)),
              "total_restarts": int(sum(r["restarts"] for r in legs)),
              "total_train_wall_s": float(sum(r["wall_s"] for r in legs))}
    result.update(chain_scores(states, D, a.N, fld))
    result["wall_s_this_run"] = round(time.time() - t_start, 1)
    with open(os.path.join(cdir, "chain.json"), "w") as f:
        json.dump(result, f, indent=1)
    print("CHAIN N=%d q=%d: test endpoint vs RK6 median %.3g um (p95 %.3g), vs real "
          "SciFi state %.3g um, straight line %.3g um; converged legs %d/%d; "
          "%.0f s training in all"
          % (a.N, a.q, result["test"]["vs_rk6_endpoint"]["pos_med_um"],
             result["test"]["vs_rk6_endpoint"]["pos_p95_um"],
             result["test"]["vs_real_scifi_state"]["pos_med_um"],
             result["test"]["straight_line_vs_rk6_endpoint"]["pos_med_um"],
             sum(r["converged"] for r in legs), a.N, result["total_train_wall_s"]),
          flush=True)
    return result


if __name__ == "__main__":
    main()
