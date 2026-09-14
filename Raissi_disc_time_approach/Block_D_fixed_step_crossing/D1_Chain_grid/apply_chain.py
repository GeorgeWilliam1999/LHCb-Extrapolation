#!/usr/bin/env python
"""How to apply the Block D networks correctly. Read APPLYING_THE_NETWORKS.md first.

    from apply_chain import load_chain
    chain = load_chain("results", N=4, q=8)      # the four 1294 mm networks
    states = chain.extrapolate(S0)               # S0: (n, 5) at z0 = 2648.2 mm
    end = states[:, -1]                          # (n, 5) at z1 = 7826.0 mm

Self-test (reproduces the chain's own record on D0's test particles):
    python apply_chain.py --N 4 --q 8 --check
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json

import numpy as np
import torch

import use_shared                                        # noqa: F401
from _shared.evaluate import predict
from _shared.reference import FROZEN_LEG, gauss_legendre, make_field
from chain_model import FrozenResidualNetwork

HERE = os.path.dirname(os.path.abspath(__file__))
Z0 = float(FROZEN_LEG["z0"])
Z1 = float(FROZEN_LEG["z1"])
L = Z1 - Z0
FIELD = "up"          # the polarity of the sample every Block D network was trained on


class Chain:
    """N fixed-step networks in order, z0 -> z1, each applied on its own leg only."""

    def __init__(self, N, q, legs, planes, record):
        self.N, self.q, self.legs, self.planes, self.record = N, q, legs, planes, record

    def check_input(self, S0):
        S0 = np.asarray(S0, dtype=np.float64)
        if S0.ndim != 2 or S0.shape[1] != 5:
            raise ValueError("S0 must be (n, 5): x [mm], y [mm], tx, ty, q/p "
                             "[0.299792458 * q / p(GeV)] on the plane z0 = %.1f mm" % Z0)
        if not np.isfinite(S0).all():
            raise ValueError("S0 has non-finite entries")
        if np.abs(S0[:, 4]).max() > 0.31:
            raise ValueError("q/p above 0.31 means p below 1 GeV: outside the training "
                             "population (1 < p < 200 GeV)")
        return S0

    def extrapolate(self, S0, stages=False):
        """(n, N+1, 5) the state on every plane; stages=True also returns the
        list of (n, q, 5) stage states per leg."""
        S = self.check_input(S0).copy()
        out = np.empty((len(S), self.N + 1, 5))
        out[:, 0] = S
        stage_states = []
        for k, (zk, zk1, model) in enumerate(self.legs):
            pred = predict(model, S)                       # (n, q+1, 4)
            if stages:
                st = np.concatenate([pred[:, :-1, :], np.repeat(S[:, None, 4:5], self.q, 1)], 2)
                stage_states.append(st)
            S = np.concatenate([pred[:, -1, :], S[:, 4:5]], axis=1)   # q/p carried
            out[:, k + 1] = S
        return (out, stage_states) if stages else out

    def endpoint(self, S0):
        return self.extrapolate(S0)[:, -1]


def load_chain(results_dir, N, q, field=FIELD):
    cdir = os.path.join(results_dir, "N%03d_q%02d" % (N, q))
    with open(os.path.join(cdir, "chain.json")) as f:
        record = json.load(f)
    fld = make_field(field)
    c, _, _ = gauss_legendre(q)
    dz = L / N
    legs = []
    for k in range(N):
        with open(os.path.join(cdir, "leg%03d_scale.json" % k)) as f:
            sc = json.load(f)
        zk = Z0 + k * dz
        assert abs(sc["z0"] - zk) < 1e-6 and sc["q"] == q, "leg file does not match its slot"
        torch.manual_seed(record["seed"])
        model = FrozenResidualNetwork(q, np.array(sc["in_scale"]), np.array(sc["out_scale"]),
                                      width=record["width"], depth=record["depth"],
                                      c=c, z0=zk, dz=dz, field=fld)
        model.load_state_dict(torch.load(os.path.join(cdir, "leg%03d.pt" % k),
                                         weights_only=True))
        model.eval()
        legs.append((zk, zk + dz, model))
    return Chain(N, q, legs, Z0 + np.arange(N + 1) * dz, record)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--check", action="store_true",
                    help="reproduce the chain record's test median from D0")
    a = ap.parse_args(argv)
    chain = load_chain(a.results, a.N, a.q)
    print("loaded N = %d legs at q = %d; planes:" % (a.N, a.q), np.round(chain.planes, 1))
    if a.check:
        D = np.load(os.path.join(HERE, "..", "D0_Crossing_dataset", "results",
                                 "crossing_particles.npz"))
        n_max = int(D["n_max"])
        S0 = D["test_S0"]
        end = chain.endpoint(S0)
        truth = D["test_truth"][:, n_max]
        med = float(np.median(np.abs(end[:, :2] - truth[:, :2]).max(axis=1) * 1e3))
        rec = chain.record["test"]["vs_rk6_endpoint"]["pos_med_um"]
        print("test median vs RK6: recomputed %.6g um, recorded %.6g um -> %s"
              % (med, rec, "MATCH" if abs(med - rec) <= 1e-6 * max(rec, 1) else "MISMATCH"))


if __name__ == "__main__":
    main()
