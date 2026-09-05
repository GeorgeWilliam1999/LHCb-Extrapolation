#!/usr/bin/env python
"""The gates on this shared package: it must reproduce the verified baseline.

Five checks, each of which must pass before any experiment is built on top:

  1. `frozen_leg_dataset(q=8)` reproduces `One_step_network_v2`'s dataset -
     the same 1979 / 2062 / 2018 states per split and the same input and
     output scales to 1e-9.
  2. `train.py --mode physics --seed 0` reproduces the baseline's very first
     L-BFGS restart exactly.

     A note on what "exactly" means here, because the obvious comparison is
     the wrong one. The number recorded in
     `One_step_network_v2/results/hist_physics_seed0.csv` at restart 0 is
     4.924321332318405e-03, and the shared driver produces
     4.927638009365614e-03 - a relative difference of 6.7e-4, far outside any
     tolerance one would want. That difference is NOT a code difference. The
     v2 runs were launched with `OMP_NUM_THREADS=4`; this package pins every
     run to a single thread (the node's torch spin-waits on its workers and
     runs about 68x slower otherwise). A different thread count means a
     different reduction order inside the BLAS matrix products, so the loss
     and its gradient differ in the last bits, and 200 L-BFGS iterations with
     a strong-Wolfe line search amplify last-bit differences into the fourth
     digit. Running the ORIGINAL `One_step_network/model.py` on one thread
     gives 4.927638009365614e-03 - bit-for-bit what the shared driver gives.

     So this test recomputes the baseline reference on one thread and requires
     agreement to 1e-12 (in fact it is bitwise), and separately reports the
     distance to the recorded 4-thread number. The model, the loss and the
     gradients were also checked component by component: identical
     initialisation, identical loss, identical gradients.
  3. `general_leg_dataset` builds for legs A, B and C, and a one-restart
     physics run on it completes.
  4. `make_field('up')` loads and the torch twin agrees with it.
  5. `chain` runs a trained model leg after leg without error.

(The HTCondor gate is separate - it needs a submit host; see condor/README.md.)

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python smoke_tests.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

BASELINE = os.path.join(ROOT, "One_step_network_v2", "results")


def check_frozen_dataset(tmp):
    from _shared.prepare import frozen_leg_dataset
    ref_meta = json.load(open(os.path.join(BASELINE, "frozen_leg_meta.json")))
    out = os.path.join(tmp, "frozen_leg_data.npz")
    t0 = time.time()
    arrays = frozen_leg_dataset(q=8, out_npz=out, verbose=False)
    counts = {s: int(len(arrays["%s_S" % s])) for s in ("train", "val", "test")}
    ok_counts = counts == ref_meta["counts"]
    d_in = float(np.abs(arrays["in_scale"] - np.array(ref_meta["in_scale"])).max())
    d_out = float(np.abs(arrays["out_scale"] - np.array(ref_meta["out_scale"])).max())
    print("  counts %s (baseline %s) -> %s" % (counts, ref_meta["counts"],
                                               "match" if ok_counts else "DIFFER"))
    print("  max |d in_scale| = %.3e   max |d out_scale| = %.3e" % (d_in, d_out))
    print("  built in %.0f s" % (time.time() - t0))
    assert ok_counts, "state counts differ from the baseline dataset"
    assert d_in < 1e-9 and d_out < 1e-9, "scales differ from the baseline dataset"
    return out


def baseline_first_restart(data_npz):
    """Run the ORIGINAL One_step_network model/loss for one L-BFGS restart.

    This is the reference the shared driver must reproduce. It is recomputed
    rather than read from the recorded history because the recorded runs used
    four BLAS threads and we run on one - see `check_first_restart`.
    """
    import torch
    sys.path.insert(0, os.path.join(ROOT, "Baseline_data_exploration"))
    sys.path.insert(0, os.path.join(ROOT, "Simple_first_pass"))
    sys.path.insert(0, os.path.join(ROOT, "One_step_network"))
    import model as baseline_model                                # noqa: E402
    from irk import tableau                                       # noqa: E402

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    d = np.load(data_npz)
    q = int(d["q"])
    dz = float(d["z1"]) - float(d["z0"])
    _, A_np, b_np = tableau(q)
    A, b = torch.tensor(A_np), torch.tensor(b_np)
    zn = torch.tensor(d["znodes"])
    S = torch.tensor(d["train_S"])
    rates = baseline_model.LHCbRates()
    torch.manual_seed(0)
    m = baseline_model.OneStepNetwork(q, d["in_scale"], d["out_scale"])
    opt = torch.optim.LBFGS(m.parameters(), max_iter=200, history_size=120,
                            tolerance_grad=1e-13, tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = baseline_model.physics_loss(m, rates, S, dz, zn, A, b)
        loss.backward()
        return loss

    opt.step(closure)
    return float(baseline_model.physics_loss(m, rates, S, dz, zn, A, b).item())


def check_first_restart(tmp, data_npz):
    from _shared import train as train_mod
    recorded = float(list(csv.DictReader(
        open(os.path.join(BASELINE, "hist_physics_seed0.csv"))))[0]["loss"])
    t0 = time.time()
    ref_loss = baseline_first_restart(data_npz)
    out = os.path.join(tmp, "parity_run")
    train_mod.main(["--data", data_npz, "--mode", "physics", "--seed", "0",
                    "--out", out, "--tag", "physics_seed0",
                    "--outer-cap", "3", "--no-confirm"])
    got = float(list(csv.DictReader(
        open(os.path.join(out, "physics_seed0_history.csv"))))[0]["loss"])
    rel = abs(got - ref_loss) / ref_loss
    rel_recorded = abs(got - recorded) / recorded
    print("  restart 0 loss, shared driver          : %.15e" % got)
    print("  restart 0 loss, baseline code, 1 thread: %.15e" % ref_loss)
    print("  relative difference                    : %.3e (bitwise %s)"
          % (rel, got == ref_loss))
    print("  restart 0 loss recorded in v2 (4 BLAS threads): %.15e"
          % recorded)
    print("  relative difference vs that record     : %.3e  (thread-count "
          "effect, see the note in this file)" % rel_recorded)
    print("  four restarts in %.0f s" % (time.time() - t0))
    assert rel < 1e-12, ("the shared driver does not reproduce the baseline "
                         "model and loss (relative difference %.3e)" % rel)
    assert rel_recorded < 1e-2, ("the shared driver is far from the recorded "
                                 "baseline run: %.3e" % rel_recorded)
    return out


def check_model_equivalence(data_npz):
    """The generalised model IS the baseline model when n_extra = 0.

    Same initial parameters, same loss, same gradients - bitwise, on the same
    data and the same seed.
    """
    import torch
    sys.path.insert(0, os.path.join(ROOT, "Baseline_data_exploration"))
    sys.path.insert(0, os.path.join(ROOT, "Simple_first_pass"))
    sys.path.insert(0, os.path.join(ROOT, "One_step_network"))
    import model as baseline_model                                # noqa: E402
    from irk import tableau                                       # noqa: E402
    from _shared import model as shared_model                     # noqa: E402

    torch.set_default_dtype(torch.float64)
    d = np.load(data_npz)
    q = int(d["q"])
    dz = float(d["z1"]) - float(d["z0"])
    _, A_np, b_np = tableau(q)
    A, b = torch.tensor(A_np), torch.tensor(b_np)
    zn = torch.tensor(d["znodes"])
    S = torch.tensor(d["train_S"])
    ref = torch.tensor(d["train_ref"])

    br, sr = baseline_model.LHCbRates(), shared_model.LHCbRates()
    torch.manual_seed(0)
    bm = baseline_model.OneStepNetwork(q, d["in_scale"], d["out_scale"])
    torch.manual_seed(0)
    sm = shared_model.OneStepNetwork(q, d["in_scale"], d["out_scale"])

    params_equal = all(torch.equal(p, r) for p, r in zip(bm.parameters(),
                                                         sm.parameters()))
    lb = baseline_model.physics_loss(bm, br, S, dz, zn, A, b)
    ls = shared_model.physics_loss(sm, sr, S, dz, zn, A, b)
    db = baseline_model.data_loss(bm, S, ref).item()
    ds = shared_model.data_loss(sm, S, ref).item()
    lb.backward()
    ls.backward()
    grads_equal = all(torch.equal(p.grad, r.grad)
                      for p, r in zip(bm.parameters(), sm.parameters()))
    print("  parameters identical: %s" % params_equal)
    print("  physics loss  %.18e vs %.18e -> equal %s"
          % (lb.item(), ls.item(), lb.item() == ls.item()))
    print("  data loss     %.18e vs %.18e -> equal %s" % (db, ds, db == ds))
    print("  gradients identical: %s" % grads_equal)
    assert params_equal and grads_equal
    assert lb.item() == ls.item() and db == ds
    return {"params": params_equal, "physics_loss": lb.item(),
            "data_loss": db, "grads": grads_equal}


def check_general(tmp):
    from _shared.prepare import general_leg_dataset
    from _shared import train as train_mod
    out_npz = os.path.join(tmp, "general_leg_data.npz")
    t0 = time.time()
    arrays = general_leg_dataset(legs=("A", "B", "C"), q=8, n_train=2000,
                                 n_eval=500, out_npz=out_npz, verbose=False)
    counts = {s: int(len(arrays["%s_S" % s])) for s in ("train", "val", "test")}
    print("  general dataset counts %s, built in %.0f s" % (counts, time.time() - t0))
    out = os.path.join(tmp, "general_run")
    t0 = time.time()
    rec = train_mod.main(["--data", out_npz, "--mode", "physics", "--seed", "0",
                          "--out", out, "--tag", "general_physics_seed0",
                          "--outer-cap", "1", "--no-confirm"])
    print("  one physics restart on general legs in %.0f s, final loss %.4e"
          % (time.time() - t0, rec["final_loss"]))
    assert np.isfinite(rec["final_loss"])
    return out_npz, os.path.join(out, "general_physics_seed0.pt"), arrays


def check_up_field():
    from _shared.field_torch import parity
    from _shared.reference import make_field
    f = make_field("up")
    print("  up map:", f.info())
    return parity("up", n=200_000, seed=1)


def check_chain(general_npz, ckpt, arrays):
    import torch
    from _shared.evaluate import chain, chain_errors, chain_reference
    from _shared.model import OneStepNetwork
    model = OneStepNetwork(int(arrays["q"]), arrays["in_scale"],
                           arrays["out_scale"], n_extra=2)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    n = 64
    S0 = arrays["test_S"][:n]
    z0 = arrays["test_z0"][:n]
    dz = arrays["test_dz"][:n]
    # Each sample's own leg cut into three sub-legs: the chain has to start
    # where the state actually lives, or the "reference" is meaningless.
    legs = [(z0 + i * dz / 3.0, z0 + (i + 1) * dz / 3.0) for i in range(3)]
    pred = chain(model, S0, legs, arrays["extra_mean"], arrays["extra_scale"])
    ref = chain_reference(S0, legs)
    errs = chain_errors(pred, ref)
    print("  chained 3 sub-legs on %d states; per-leg median error (um): %s"
          % (n, [round(v, 1) for v in errs["per_leg_med_um"]]))
    assert pred.shape == (n, 3, 5)
    assert np.isfinite(pred).all(), "the chained prediction went non-finite"
    assert np.isfinite(ref).all(), "the chained reference went non-finite"
    return errs


def main():
    results = {}
    with tempfile.TemporaryDirectory(prefix="a0_smoke_") as tmp:
        print("(i) frozen_leg_dataset(q=8) vs the baseline dataset")
        frozen = check_frozen_dataset(tmp)
        results["frozen_dataset"] = "pass"

        print("(ii-a) the generalised model vs the baseline model, bitwise")
        results["model_equivalence"] = check_model_equivalence(frozen)

        print("(ii-b) train.py physics seed 0, first restart vs the baseline")
        check_first_restart(tmp, frozen)
        results["first_restart_parity"] = "pass"

        print("(iv) general_leg_dataset(A,B,C) + a one-restart physics run")
        general_npz, ckpt, arrays = check_general(tmp)
        results["general_dataset"] = "pass"

        print("(v) make_field('up') + the torch parity gate")
        results["up_field_parity"] = check_up_field()

        print("(extra) chain() over three legs")
        results["chain"] = check_chain(general_npz, ckpt, arrays)

    print("\nALL LOCAL SMOKE TESTS PASS")
    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
