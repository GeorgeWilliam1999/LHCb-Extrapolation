#!/usr/bin/env python
"""Train ONE (mode, seed) one-step network to genuine stall, then confirm it.

The optimiser protocol is the verified baseline's, unchanged: full-batch
L-BFGS (max_iter 200, strong Wolfe, history 120, fp64), restarted until the
stall criterion fires - two consecutive restarts each improving the loss by
less than 1%. A checkpoint and a history row are written after EVERY restart,
so a run is crash-safe and rerunning the same command continues where it
stopped rather than starting again.

Confirmation pass. A stall can be an artefact of the optimiser's curvature
history rather than a property of the loss, so unless `--no-confirm` is given
the run continues with a FRESH optimiser once it has stalled. The confirmation
holds only if the run stalls again within two restarts and the endpoint medians
on train, val and test are unchanged (less than 1% relative).

A confirmation that does not hold sends the run BACK TO THE STALL PHASE and the
cycle repeats - train to a stall, confirm it, train on if it does not hold -
until a confirmation holds or the `--outer-cap` restart cap is reached. That is
the honest reading of a failed confirmation: the fresh optimiser either kept
improving the loss or moved the endpoint medians, and either way the run had
not finished training. Before 2026-09-07 such a run stopped and was recorded
`converged = false`, and resubmitting it only repeated the confirmation from
where it stood; on the Block C grid that was 77 of the first 107 runs. A run
that confirms at its first attempt is unaffected, restart for restart. The
json records how many confirmation passes were started as `confirm_attempts`.

Usage:
    PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 python train.py \\
        --data <prepared.npz> --mode physics --seed 0 --out results --tag physics_seed0

Outputs in <out>/:
    <tag>.pt           the model state dict, rewritten every restart
    <tag>_history.csv  one row per restart: phase, outer, loss, wall_s
    <tag>.json         the final record (losses, convergence, all three splits)
"""
from __future__ import annotations

import os

# Torch spin-waits on its worker threads on this node (measured 68x slower);
# pin everything to one thread BEFORE torch is imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse    # noqa: E402
import csv         # noqa: E402
import json        # noqa: E402
import sys         # noqa: E402
import time        # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _shared.evaluate import score_split                        # noqa: E402
from _shared.model import (LHCbRates, OneStepNetwork,           # noqa: E402
                           data_loss, physics_loss)
from _shared.reference import gauss_legendre, make_field        # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HISTORY_FIELDS = ["mode", "seed", "phase", "outer", "loss", "wall_s"]


# ------------------------------------------------------------- the history --
def read_history(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def append_history(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def _improved(previous, loss):
    """The stall test: did this restart improve the loss by at least 1%?

    The one place the rule is written down, so the live loop and the history
    reader below cannot drift apart.
    """
    return (previous - loss) >= 1e-2 * max(loss, 1e-30)


def _trailing_confirm_block(rows):
    """The rows of the confirmation pass a stopped run was last in, if any."""
    if not rows or rows[-1]["phase"] != "confirm":
        return []
    k = len(rows)
    while k > 0 and rows[k - 1]["phase"] == "confirm":
        k -= 1
    return rows[k:]


def resume_phase(rows):
    """The phase a resumed run should carry on in, read off its own history.

    A run whose last history row says 'confirm' is in one of two quite
    different situations, and the row alone does not say which: either the
    confirmation pass was still running when the job stopped, or it had just
    finished and did not hold, so the run needs more training rather than
    another confirmation. The stall rule is therefore re-applied to the
    trailing block of confirmation restarts, exactly as the live loop applies
    it. If it has already fired there the confirmation is over and the run goes
    back to the stall phase; if it has not, the confirmation is picked up where
    it stopped.
    """
    block = _trailing_confirm_block(rows)
    if len(block) < 2:
        return "stall" if not block else "confirm"
    start = rows.index(block[0])
    previous = float(rows[start - 1]["loss"]) if start else float("inf")
    stalled = 0
    for r in block:
        loss = float(r["loss"])
        stalled = 0 if _improved(previous, loss) else stalled + 1
        previous = loss
    return "stall" if stalled >= 2 else "confirm"


def count_confirm_attempts(rows):
    """How many confirmation passes this run has already started."""
    return sum(1 for i, r in enumerate(rows)
               if r["phase"] == "confirm"
               and (i == 0 or rows[i - 1]["phase"] != "confirm"))


# ---------------------------------------------------------------- the data --
def load_dataset(path):
    d = np.load(os.path.abspath(path), allow_pickle=False)
    return {k: d[k] for k in d.files}


def tensors_for(data, split="train"):
    """(S, ref, dz, znodes, extra) as tensors for one split."""
    S = torch.tensor(np.asarray(data["%s_S" % split], dtype=np.float64))
    ref = torch.tensor(np.asarray(data["%s_ref" % split], dtype=np.float64))
    if "%s_znodes" % split in data:                       # general leg
        znodes = torch.tensor(np.asarray(data["%s_znodes" % split]))
        dz = torch.tensor(np.asarray(data["%s_dz" % split]))
        extra = torch.tensor(np.asarray(data["%s_extra" % split]))
    else:                                                 # frozen leg
        znodes = torch.tensor(np.asarray(data["znodes"]))
        dz = float(data["z1"]) - float(data["z0"])        # a python float, as
        extra = None                                      # the baseline had it
    return S, ref, dz, znodes, extra


# ------------------------------------------------------------- the training --
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="prepared dataset .npz")
    ap.add_argument("--mode", required=True, choices=("physics", "data"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--q", type=int, default=None,
                    help="stages; defaults to the dataset's own q")
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--tag", required=True, help="basename of the outputs")
    ap.add_argument("--field", choices=("down", "up"), default=None,
                    help="field polarity; defaults to the dataset's own")
    ap.add_argument("--outer-cap", type=int, default=400,
                    help="safety cap on total L-BFGS restarts (both phases)")
    ap.add_argument("--max-iter", type=int, default=200,
                    help="L-BFGS iterations inside one restart")
    ap.add_argument("--no-confirm", action="store_true",
                    help="stop at the first stall, skip the confirmation pass")
    a = ap.parse_args(argv)

    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.out, a.tag + ".pt")
    hist_path = os.path.join(a.out, a.tag + "_history.csv")
    json_path = os.path.join(a.out, a.tag + ".json")

    data = load_dataset(a.data)
    q = int(data["q"])
    if a.q is not None and a.q != q:
        raise SystemExit("--q %d does not match the dataset's q = %d" % (a.q, q))
    field = a.field or (str(data["field"]) if "field" in data else "down")
    kind = str(data["kind"]) if "kind" in data else "frozen"
    n_extra = 2 if kind == "general" else 0

    c, A_np, b_np = gauss_legendre(q)
    A, b = torch.tensor(A_np), torch.tensor(b_np)
    S_tr, REF_tr, DZ, ZN, EX_tr = tensors_for(data, "train")
    in_scale, out_scale = data["in_scale"], data["out_scale"]

    rates = LHCbRates(make_field(field))          # no RNG is consumed here
    torch.manual_seed(a.seed)
    model = OneStepNetwork(q, in_scale, out_scale, width=a.width,
                           depth=a.depth, n_extra=n_extra)

    rows = read_history(hist_path)
    if rows and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        outer = 1 + max(int(r["outer"]) for r in rows)
        previous = float(rows[-1]["loss"])
        phase = resume_phase(rows)
        print("resuming %s seed %d in phase '%s' at restart %d (loss %.6e)"
              % (a.mode, a.seed, phase, outer, previous), flush=True)
    else:
        outer, previous, phase = 0, float("inf"), "stall"

    def loss_now():
        if a.mode == "physics":
            return physics_loss(model, rates, S_tr, DZ, ZN, A, b, EX_tr)
        return data_loss(model, S_tr, REF_tr, EX_tr)

    def make_opt():
        return torch.optim.LBFGS(model.parameters(), max_iter=a.max_iter,
                                 history_size=120, tolerance_grad=1e-13,
                                 tolerance_change=1e-16,
                                 line_search_fn="strong_wolfe")

    opt = make_opt()

    def closure():
        opt.zero_grad()
        loss = loss_now()
        loss.backward()
        return loss

    def one_restart(tag_phase):
        nonlocal outer, previous
        t0 = time.time()
        opt.step(closure)
        loss = loss_now().item()
        torch.save(model.state_dict(), ckpt)
        wall = round(time.time() - t0, 1)
        append_history(hist_path, dict(mode=a.mode, seed=a.seed,
                                       phase=tag_phase, outer=outer,
                                       loss=loss, wall_s=wall))
        print("  %s seed %d %s restart %d: loss %.6e (%.0f s)"
              % (a.mode, a.seed, tag_phase, outer, loss, wall), flush=True)
        improved = _improved(previous, loss)
        previous = loss
        outer += 1
        return loss, improved

    def medians():
        return {s: score_split(model, data, s)[0]["endpoint_med_um"]
                for s in ("train", "val", "test")}

    # -- the stall / confirm cycle --------------------------------------------
    # Phase 1 restarts until the stall criterion fires - two consecutive
    # restarts each improving the loss by less than 1%. Phase 2 then confirms
    # that stall with a fresh optimiser: without it the honest claim is only
    # "it stalled"; with it, "it stalled, and a fresh optimiser could not move
    # it".
    #
    # A confirmation that does NOT hold is not the end of the run. It means one
    # of two things - the fresh optimiser went on improving the loss for more
    # than two restarts, or it left the loss alone but moved the endpoint
    # medians by more than 1% - and both say the same thing: the run had not
    # finished training. Until 2026-09-07 such a run was written
    # `converged = false` and stopped there, and because the phase was read
    # straight off the last history row, resubmitting it only re-ran the
    # confirmation from where it stood, which is the one thing that could not
    # help. On the Block C magnet-to-magnet grid that was 77 of the first 107
    # runs.
    #
    # So a failed confirmation now drops back to the stall phase and the cycle
    # repeats - train to a stall, confirm it, train on if it does not hold -
    # until a confirmation passes or the restart cap is reached. `--outer-cap`
    # remains the only bound on the total work. A run that confirms at its
    # first attempt follows exactly the path it followed before, restart for
    # restart, and `resume_phase` reads the cycle position back out of the
    # history so a job that is evicted mid-cycle carries on rather than
    # starting the confirmation again.
    loss = previous
    hit_cap = False
    converged = False
    confirm_attempts = count_confirm_attempts(rows)
    while True:
        if phase == "stall":
            stalled, fired = 0, False
            while outer < a.outer_cap:
                loss, improved = one_restart("stall")
                stalled = 0 if improved else stalled + 1
                if stalled >= 2 and outer >= 3:
                    print("  %s seed %d: stalled at restart %d"
                          % (a.mode, a.seed, outer - 1), flush=True)
                    fired = True
                    break
            if not fired:
                hit_cap = True
                print("  %s seed %d: hit the %d-restart cap without stalling"
                      % (a.mode, a.seed, a.outer_cap), flush=True)
                break
            phase = "confirm"

        converged = True                  # it stalled; the question is whether
        if a.no_confirm or outer >= a.outer_cap:
            break                         # that stall survives a fresh optimiser

        before = medians()
        opt = make_opt()                  # curvature history deliberately reset
        confirm_attempts += 1
        n_conf, stalled, restalled = 0, 0, False
        while outer < a.outer_cap:
            loss, improved = one_restart("confirm")
            n_conf += 1
            stalled = 0 if improved else stalled + 1
            if stalled >= 2:
                restalled = True
                break
        if not restalled:
            print("  %s seed %d: confirmation hit the restart cap"
                  % (a.mode, a.seed), flush=True)
            converged = False
            break
        after = medians()
        unchanged = all(abs(after[s] - before[s])
                        <= 1e-2 * max(before[s], 1e-30) for s in before)
        converged = bool(n_conf <= 2 and unchanged)
        print("  %s seed %d: re-stalled after %d confirmation restarts; "
              "medians unchanged = %s -> converged = %s"
              % (a.mode, a.seed, n_conf, unchanged, converged), flush=True)
        if converged:
            break
        print("  %s seed %d: confirmation %d did not hold; back to the stall "
              "phase (restart %d of %d)"
              % (a.mode, a.seed, confirm_attempts, outer, a.outer_cap),
              flush=True)
        phase = "stall"

    # -- the record -----------------------------------------------------------
    rows = read_history(hist_path)
    wall_s = round(sum(float(r["wall_s"]) for r in rows), 1)
    record = {
        "tag": a.tag, "mode": a.mode, "seed": a.seed, "q": q,
        "width": a.width, "depth": a.depth, "n_extra": n_extra,
        "dataset": os.path.abspath(a.data), "dataset_kind": kind,
        "field": field,
        "restarts": len(rows),
        "final_loss": float(rows[-1]["loss"]) if rows else float("nan"),
        "converged": bool(converged),
        "confirm_attempts": int(confirm_attempts),
        "wall_s": wall_s,
        "outer_cap": a.outer_cap, "max_iter": a.max_iter,
        "confirmed": not a.no_confirm,
    }
    for split in ("train", "val", "test"):
        record[split] = score_split(model, data, split)[0]
    with open(json_path, "w") as f:
        json.dump(record, f, indent=1)

    print("SUMMARY %s mode=%s seed=%d q=%d restarts=%d final_loss=%.4e "
          "converged=%s test_endpoint_med=%.1f um test_p95=%.1f um "
          "straight=%.1f um wall=%.0f s"
          % (a.tag, a.mode, a.seed, q, record["restarts"], record["final_loss"],
             record["converged"], record["test"]["endpoint_med_um"],
             record["test"]["endpoint_p95_um"], record["test"]["straight_med_um"],
             time.time() - t_start), flush=True)
    return record


if __name__ == "__main__":
    main()
