#!/usr/bin/env python
"""D3 - can a leg that was declared converged keep training?

The shared trainer's L-BFGS has fixed absolute tolerances (gradient 1e-13,
change 1e-16). The physics loss is normalised by population spreads, so on a
40 mm leg it sits near 5e-9 and its gradient can fall below the tolerance
while the loss is still dropping about one percent per restart. The optimiser
then returns at once, the restart leaves the loss unchanged, and the stall
rule reads that as convergence.

This script takes one leg, rebuilds its training set exactly as the chain
trainer did (the same input states from the chain's saved states, the same
RK6 references, the same scales), reloads its final weights, and then:

  A  runs a fresh optimiser on the loss as trained, for two restarts - this
     should reproduce the instant stop if the leg was stopped by the tolerance;
  B  runs a fresh optimiser on the SAME loss multiplied by a constant so that
     it starts at 1, for --restarts restarts. A constant factor does not move
     the minimum; it only lifts the gradient clear of the fixed tolerance.

After every --score-every restarts it scores the network on its own leg
(train and test), exactly as the leg record was scored. Nothing in
D1_Chain_grid is written.

Run (one leg):
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python continue_legs.py --N 128 --q 7 --leg 120

Outputs:
    results/continuation/N<NNN>_q<qq>_leg<kkk>.csv    one row per restart
    results/continuation/N<NNN>_q<qq>_leg<kkk>.json   before/after summary
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D1 = os.path.abspath(os.path.join(HERE, "..", "D1_Chain_grid"))
sys.path.insert(0, D1)
import use_shared  # noqa: E402,F401  (puts _shared on the path)
import torch  # noqa: E402

import train_chain as tc  # noqa: E402
from _shared.evaluate import score_split  # noqa: E402
from _shared.model import LHCbRates, physics_loss  # noqa: E402
from _shared.reference import gauss_legendre, make_field  # noqa: E402
from _shared.train import tensors_for  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT = os.path.join(HERE, "results", "continuation")


def lbfgs(model, max_iter):
    """The shared trainer's optimiser, setting for setting."""
    return torch.optim.LBFGS(model.parameters(), max_iter=max_iter, history_size=120,
                             tolerance_grad=1e-13, tolerance_change=1e-16,
                             line_search_fn="strong_wolfe")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--leg", type=int, required=True)
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--score-every", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=200)
    a = ap.parse_args(argv)
    name = "N%03d_q%02d_leg%03d" % (a.N, a.q, a.leg)
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()

    D = {k: v for k, v in np.load(tc.DEFAULT_DATA, allow_pickle=False).items()}
    Z0, L = float(D["z0"]), float(D["L"])
    dz = L / a.N
    zk = (Z0 + np.arange(a.N + 1) * dz)[a.leg]
    field = str(D["field"])
    fld = make_field(field)
    c, A_np, b_np = gauss_legendre(a.q)
    cdir = tc.chain_dir(os.path.join(D1, "results"), a.N, a.q)
    tag = tc.leg_tag(a.leg)
    st = np.load(os.path.join(cdir, "states.npz"))
    states = {s: st["%s_states" % s] for s in tc.SPLITS}

    t0 = time.time()
    data = tc.leg_dataset(states, a.leg, zk, dz, a.q, c, D, field)
    t_data = time.time() - t0
    sc = json.load(open(os.path.join(cdir, tag + "_scale.json")))
    assert np.allclose(data["in_scale"], sc["in_scale"], rtol=1e-12, atol=0), "in_scale differs"
    assert np.allclose(data["out_scale"], sc["out_scale"], rtol=1e-12, atol=0), "out_scale differs"
    rec = json.load(open(os.path.join(cdir, tag + ".json")))

    factory = tc.make_factory(c, zk, dz, fld, tc.WIDTH, tc.DEPTH)
    model = tc.rebuild(factory, data, os.path.join(cdir, tag + ".pt"), 0, tc.WIDTH, tc.DEPTH)
    model.train()
    A, b = torch.tensor(A_np), torch.tensor(b_np)
    S_tr, _ref, DZ, ZN, EX_tr = tensors_for(data, "train")
    rates = LHCbRates(make_field(field))

    def loss_now():
        return physics_loss(model, rates, S_tr, DZ, ZN, A, b, EX_tr)

    def own():
        out = {}
        for s in ("train", "test"):
            r = score_split(model, data, s)[0]
            out[s] = dict(endpoint_med_um=r["endpoint_med_um"], straight_med_um=r["straight_med_um"])
        return out

    model.zero_grad()
    l0 = loss_now()
    l0.backward()
    grad_max = max(float(p.grad.abs().max()) for p in model.parameters())
    L0 = float(l0.item())
    before = own()
    rows = []

    def run(phase, factor, n):
        opt = lbfgs(model, a.max_iter)

        def closure():
            opt.zero_grad()
            loss = loss_now() * factor
            loss.backward()
            return loss

        prev = float(loss_now().item())
        for r in range(n):
            t = time.time()
            opt.step(closure)
            wall = time.time() - t
            loss = float(loss_now().item())
            row = dict(phase=phase, restart=r, loss=loss, rel_gain=(prev - loss) / prev,
                       loss_over_start=loss / L0, wall_s=round(wall, 2))
            if (r + 1) % a.score_every == 0 or r == n - 1:
                o = own()
                row.update(train_own_um=o["train"]["endpoint_med_um"],
                           test_own_um=o["test"]["endpoint_med_um"],
                           test_straight_um=o["test"]["straight_med_um"])
            rows.append(row)
            print("%s %-9s restart %2d  loss %.4e  gain %+.2e  wall %5.1f s%s"
                  % (name, phase, r, loss, row["rel_gain"], wall,
                     ("  test own %.3f um" % row["test_own_um"]) if "test_own_um" in row else ""),
                  flush=True)
            prev = loss

    run("as_trained", 1.0, 2)
    after_a = own()
    run("rescaled", 1.0 / L0, a.restarts)
    after_b = own()

    fields = ["phase", "restart", "loss", "rel_gain", "loss_over_start", "wall_s",
              "train_own_um", "test_own_um", "test_straight_um"]
    with open(os.path.join(OUT, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = dict(
        N=a.N, q=a.q, leg=a.leg, z0=float(zk), z1=float(zk + dz),
        recorded=dict(final_loss=rec["final_loss"], restarts=rec["restarts"],
                      converged=rec["converged"],
                      test_own_um=rec["test"]["endpoint_med_um"],
                      test_straight_um=rec["test"]["straight_med_um"]),
        rebuilt=dict(loss=L0, grad_max_abs=grad_max, own=before, dataset_wall_s=round(t_data, 1)),
        after_fresh_optimiser_as_trained=dict(loss=[r for r in rows if r["phase"] == "as_trained"][-1]["loss"],
                                              own=after_a),
        after_rescaled=dict(restarts=a.restarts,
                            loss=[r for r in rows if r["phase"] == "rescaled"][-1]["loss"],
                            loss_over_start=[r for r in rows if r["phase"] == "rescaled"][-1]["loss_over_start"],
                            own=after_b),
        wall_s=round(time.time() - t_all, 1),
    )
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
