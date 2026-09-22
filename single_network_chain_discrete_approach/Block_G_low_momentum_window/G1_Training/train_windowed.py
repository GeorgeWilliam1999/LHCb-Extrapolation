#!/usr/bin/env python
"""G1 - Block F's round trainer with ONE change: where the momentum window sits.

This file is `Block_F_reweighted_loss/F1_Training/train_weighted.py` with the
loss import swapped for `G0_Weighting/windowed_loss.py` (the same weights, with
the window read from the run's own constants instead of the module defaults)
and `--p-lo` / `--p-hi` added. Nothing else is touched:
`results/train_windowed.diff` is the diff against it. With the default window
(10-50 GeV) it reproduces Block F's training bit for bit (gate I2), and
`--weighting blockE` still reproduces Block E's (gate I1). The network, the
metrics and the rest of the loss are imported, not copied, so they cannot drift.

  loss      the label-free RK-PINN residual of `_shared.model`, weighted: each
            residual is measured as the displacement it would cause at the
            SciFi plane, as a fraction of that track's total bend, with the
            band between `--p-lo` and `--p-hi` weighted up. `--weighting` picks
            the mode (`full`, the ablations `no_lever` / `no_track` /
            `no_window`, or `blockE` for Block E's own weights). See
            `windowed_loss.py` for the form and why each factor is there
  states    rounds of (track, start plane) states on the dz planes
            z0, z0 + dz, ..., z1 - dz:
              round 1       the RK6 track states of the training tracks
              round r > 1   the network's own predictions: every training
                            track is carried from its real start state at z0
                            through the N steps by the current network, and the
                            states it predicts on the start planes are used
            each round draws STATES_PER_ROUND pairs, the same number on every
            plane (all pairs if there are fewer); the states are fixed within
            the round and no gradient flows back through earlier steps
  optimiser full-batch L-BFGS (200 iterations a restart, history 120,
            strong-Wolfe line search), restarted until two consecutive
            restarts each improve the loss by less than 1 percent - that ends
            a round
  rescaling the loss is divided by a constant so that it starts every round
            at 1, and the constant is renewed (with a fresh optimiser, so the
            curvature history stays consistent with the objective) whenever
            the loss has fallen tenfold since it was set. A constant does not
            move the minimum; it keeps L-BFGS's fixed absolute tolerances from
            ending restarts early, which stopped Block D's training
  stopping  when the first two restarts of a round after round 1 each improve
            the loss by less than 1 percent (the refreshed states teach
            nothing new); then a confirmation pass with a fresh optimiser on
            the same states: two restarts, each under 1 percent, and the
            validation median at z1 unchanged within 1 percent.
            Caps: --round-cap rounds, --outer-cap restarts in all.
  logging   every restart: the loss before and after (unscaled), the scale
            factor, the L-BFGS iterations and function evaluations it used,
            and a flag when it stopped before either limit (an early stop);
            every round: where its states came from, how far the training
            tracks' predicted end states moved since the last round, and the
            validation median error at z1

Resumable: a checkpoint, the history and the round's states are written after
every restart; rerunning the same command continues from the last restart
(with a fresh optimiser). A finished run exits at once.

Usage:
    python train_windowed.py --N 64 --q 2 --p-lo 3 --p-hi 8 --out results/p03-08
    python train_windowed.py --N 64 --q 2 --weighting blockE --n-train 300 --stop-after 3
    python train_windowed.py --N 2 --q 2 --n-train 300 --n-eval 200 --round-cap 2 --out results/smoke

The window is recorded in `scale.json["weighting"]` when the run is created and
is checked against the command line on every resume, the way `--weighting` is:
a run cannot silently change objective half way through. Put the window in
`--out` (`results/p03-08`) so two windows can never share a run folder.

Outputs in <out>/<weighting>/N<NNN>_q<qq>/:
    scale.json      the network's fixed configuration (input spreads, step, planes)
    network.pt      the weights, after every restart
    history.csv     one row per restart
    rounds.csv      one row per round
    progress.json   where the run is (for resuming)
    round_states.npz the current round's (state, start plane) pairs
    record.json     the final record: the run, and the validation and test chains
                    scored at every plane, per component and momentum band
    timing.json     (--timing-restarts only)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse   # noqa: E402
import csv        # noqa: E402
import json       # noqa: E402
import time       # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

import sys         # noqa: E402

import use_shared    # noqa: E402,F401
from _shared.model import LHCbRates                            # noqa: E402
from _shared.reference import gauss_legendre, make_field       # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "Block_E_single_network_chain",
                                "E1_Network_grid"))
sys.path.insert(0, os.path.join(_HERE, "..", "G0_Weighting"))
from chain_network import (DEPTH, WIDTH, Y_FLOOR_REL, build, carry,  # noqa: E402
                           znodes_for)
from metrics import chain_scores                               # noqa: E402
from windowed_loss import (MODES, i_bar, loss_for,             # noqa: E402
                           reference_constants, weights)

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRACKS = os.path.join(HERE, "..", "..", "Block_E_single_network_chain",
                              "E0_Track_dataset", "results", "tracks.npz")
STATES_PER_ROUND = 32000
STALL_TOL = 1e-2
RESCALE_DROP = 0.1
HISTORY_FIELDS = ["restart", "round", "in_round", "phase", "loss_before", "loss_after", "gain",
                  "factor", "factor_renewed", "n_iter", "func_evals", "early_stop", "wall_s"]
ROUND_FIELDS = ["round", "source", "n_states", "per_plane", "restarts", "loss_first", "loss_last",
                "train_end_moved_med_um", "val_z1_pos_med_um", "val_z1_x_med_um", "val_z1_y_med_um",
                "wall_s"]


def append_csv(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def draw_states(states_on_planes, z_starts, budget, rng):
    """(S, z_start, plane index) - the same number of tracks on every start plane."""
    n_tr, N = states_on_planes.shape[:2]
    if n_tr * N <= budget:
        tr = np.repeat(np.arange(n_tr)[None, :], N, axis=0).ravel()
        pl = np.repeat(np.arange(N), n_tr)
    else:
        per = min(n_tr, budget // N)
        tr = np.concatenate([rng.choice(n_tr, per, replace=False) for _ in range(N)])
        pl = np.repeat(np.arange(N), per)
    return states_on_planes[tr, pl], z_starts[pl], pl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--tracks", default=DEFAULT_TRACKS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weighting", default="full", choices=tuple(MODES),
                    help="which weights go inside the loss (default: full)")
    ap.add_argument("--p-lo", type=float, default=10.0,
                    help="GeV: the bottom of the momentum window in the loss (default: 10)")
    ap.add_argument("--p-hi", type=float, default=50.0,
                    help="GeV: the top of the momentum window in the loss (default: 50)")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--states-per-round", type=int, default=STATES_PER_ROUND)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--outer-cap", type=int, default=400)
    ap.add_argument("--round-cap", type=int, default=20)
    ap.add_argument("--round-restarts", type=int, default=None,
                    help="also end a round after this many restarts (default: only the stall rule ends a round)")
    ap.add_argument("--no-confirm", action="store_true")
    ap.add_argument("--n-train", type=int, default=None, help="cap on training tracks (smoke runs)")
    ap.add_argument("--n-eval", type=int, default=None, help="cap on validation and test tracks (smoke runs)")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="stop this invocation after this many restarts (tests the resume)")
    ap.add_argument("--extend", action="store_true",
                    help="continue a finished run: open a new round and train on to the new --outer-cap")
    ap.add_argument("--timing-restarts", type=int, default=None,
                    help="run this many round-1 restarts, write timing.json and stop")
    a = ap.parse_args(argv)
    t_start = time.time()

    D = {k: v for k, v in np.load(os.path.abspath(a.tracks), allow_pickle=False).items()}
    Z0, Z1, L, n_max = float(D["z0"]), float(D["z1"]), float(D["L"]), int(D["n_max"])
    if n_max % a.N:
        raise SystemExit("N = %d does not divide the plane grid (%d)" % (a.N, n_max))
    stride = n_max // a.N
    dz = L / a.N
    z_starts = Z0 + np.arange(a.N) * dz
    field = str(D["field"])
    fld = make_field(field)
    rates = LHCbRates(make_field(field))
    c, A_np, b_np = gauss_legendre(a.q)
    A, b = torch.tensor(A_np), torch.tensor(b_np)
    for s, cap in (("train", a.n_train), ("val", a.n_eval), ("test", a.n_eval)):
        if cap is not None:
            for k in list(D):
                if k.startswith(s + "_"):
                    D[k] = D[k][:cap]
    S0_train = D["train_S0"]
    rk6_on_planes = D["train_truth"][:, 0:n_max:stride]            # (n, N, 5) start planes

    run = os.path.join(a.out, a.weighting, "N%03d_q%02d" % (a.N, a.q))
    os.makedirs(run, exist_ok=True)
    paths = {k: os.path.join(run, v) for k, v in dict(
        scale="scale.json", ckpt="network.pt", hist="history.csv", rounds="rounds.csv",
        prog="progress.json", states="round_states.npz", record="record.json",
        timing="timing.json").items()}

    if os.path.exists(paths["prog"]):
        prog = json.load(open(paths["prog"]))
        if prog["phase"] == "done" and os.path.exists(paths["record"]):
            if not (a.extend and prog["restart"] < a.outer_cap):
                print("N=%d q=%d already finished; record at %s" % (a.N, a.q, paths["record"]))
                return json.load(open(paths["record"]))
            print("N=%d q=%d: extending a finished run from restart %d to the %d-restart cap"
                  % (a.N, a.q, prog["restart"], a.outer_cap), flush=True)
            prog.update(phase="train", round=prog["round"] + 1, in_round=0, stalled=0,
                        first_gains=[], round_loss_first=None, round_wall_s=0.0,
                        hit_cap=None, converged=False, confirmed=None)
    else:
        prog = None

    # -- the network's fixed configuration --------------------------------------
    if os.path.exists(paths["scale"]):
        sc = json.load(open(paths["scale"]))
        in_scale = np.asarray(sc["in_scale"])
        if sc["weighting"]["mode"] != a.weighting:
            raise SystemExit("this run was started with --weighting %s, not %s"
                             % (sc["weighting"]["mode"], a.weighting))
        w_lo, w_hi = sc["weighting"].get("p_lo"), sc["weighting"].get("p_hi")
        if (w_lo, w_hi) != (None, None) and (float(w_lo), float(w_hi)) != (a.p_lo, a.p_hi):
            raise SystemExit("this run was started with the window %g-%g GeV, not %g-%g"
                             % (w_lo, w_hi, a.p_lo, a.p_hi))
    else:
        rng1 = np.random.default_rng([a.seed, 1])
        S1, _, _p1 = draw_states(rk6_on_planes, z_starts, a.states_per_round, rng1)
        in_scale = S1.std(axis=0)
        sc = dict(N=a.N, q=a.q, L=L, z0=Z0, dz=dz, seed=a.seed, width=a.width, depth=a.depth,
                  y_floor_rel=Y_FLOOR_REL, field=field, in_scale=in_scale.tolist(),
                  states_per_round=a.states_per_round, n_train_tracks=int(len(S0_train)),
                  tracks=os.path.abspath(a.tracks))
        # the weighting's own constants, fixed here and never recomputed, so the
        # objective cannot drift between rounds
        _m = build(a.q, a.N, L, Z0, in_scale, fld, width=a.width, depth=a.depth, seed=a.seed)
        sc["weighting"] = reference_constants(_m, S1, z_starts[np.asarray(_p1)], Z1,
                                              i_bar(fld, Z0, Z1), L, mode=a.weighting,
                                              p_lo=a.p_lo, p_hi=a.p_hi)
        with open(paths["scale"], "w") as f:
            json.dump(sc, f, indent=1)
    model = build(a.q, a.N, L, Z0, in_scale, fld, width=a.width, depth=a.depth, seed=a.seed)
    if prog is not None and os.path.exists(paths["ckpt"]):
        model.load_state_dict(torch.load(paths["ckpt"], weights_only=True))
        print("resuming N=%d q=%d at restart %d, round %d (%s)"
              % (a.N, a.q, prog["restart"], prog["round"], prog["phase"]), flush=True)
    else:
        prog = dict(restart=0, round=1, in_round=0, phase="train", stalled=0, first_gains=[],
                    round_loss_first=None, converged=False, confirmed=None, hit_cap=None,
                    round_wall_s=0.0, prev_train_end=None)

    def save_progress():
        with open(paths["prog"], "w") as f:
            json.dump(prog, f, indent=1)

    def val_at_z1():
        st = carry(model, D["val_S0"], Z0, a.N)[:, -1]
        tr = D["val_truth"][:, n_max]
        pos = np.abs(st[:, :2] - tr[:, :2]).max(axis=1) * 1e3
        return (float(np.median(pos)), float(np.median(np.abs(st[:, 0] - tr[:, 0]) * 1e3)),
                float(np.median(np.abs(st[:, 1] - tr[:, 1]) * 1e3)))

    # -- the round's states -------------------------------------------------------
    def new_round_states():
        rng = np.random.default_rng([a.seed, prog["round"]])
        moved = float("nan")
        if prog["round"] == 1:
            source, on_planes, end = "rk6", rk6_on_planes, None
        else:
            source = "own"
            carried = carry(model, S0_train, Z0, a.N)
            on_planes, end = carried[:, :a.N], carried[:, -1]
            prev = os.path.join(run, "prev_train_end.npy")
            if os.path.exists(prev):
                p = np.load(prev)
                moved = float(np.median(np.abs(end[:, :2] - p[:, :2]).max(axis=1)) * 1e3)
            np.save(prev, end)
        S, zs, pl = draw_states(on_planes, z_starts, a.states_per_round, rng)
        np.savez(paths["states"], S=S, z_start=zs, plane=pl, source=source, moved=moved)
        return S, zs, source, moved

    if prog["in_round"] > 0 and os.path.exists(paths["states"]):
        rs = np.load(paths["states"])
        S_np, z_np, source, moved = rs["S"], rs["z_start"], str(rs["source"]), float(rs["moved"])
    else:
        S_np, z_np, source, moved = new_round_states()

    St, zt = torch.as_tensor(S_np), torch.as_tensor(z_np)
    zn = znodes_for(model, zt)
    W = weights(model, St, zt, sc["weighting"], a.weighting)

    def loss_now():
        return loss_for(a.weighting, model, rates, St, dz, zn, A, b, zt, W)

    def make_opt():
        return torch.optim.LBFGS(model.parameters(), max_iter=a.max_iter, history_size=120,
                                 tolerance_grad=1e-13, tolerance_change=1e-16,
                                 line_search_fn="strong_wolfe")

    state = {"opt": make_opt(), "factor": None, "factor_loss": None, "iters": 0, "evals": 0}

    def renew(loss_value):
        state.update(opt=make_opt(), factor=1.0 / loss_value, factor_loss=loss_value, iters=0, evals=0)

    def one_restart(phase):
        with torch.no_grad():
            before = loss_now().item()
        renewed = False
        if state["factor"] is None or before < RESCALE_DROP * state["factor_loss"]:
            renew(before)
            renewed = True
        opt, factor = state["opt"], state["factor"]

        def closure():
            opt.zero_grad()
            loss = loss_now() * factor
            loss.backward()
            return loss

        t0 = time.time()
        opt.step(closure)
        wall = time.time() - t0
        st = opt.state[opt._params[0]]
        n_iter, evals = st["n_iter"] - state["iters"], st["func_evals"] - state["evals"]
        state["iters"], state["evals"] = st["n_iter"], st["func_evals"]
        with torch.no_grad():
            after = loss_now().item()
        gain = (before - after) / before
        early = bool(n_iter < a.max_iter and evals < int(a.max_iter * 1.25))
        torch.save(model.state_dict(), paths["ckpt"])
        append_csv(paths["hist"], HISTORY_FIELDS, dict(
            restart=prog["restart"], round=prog["round"], in_round=prog["in_round"], phase=phase,
            loss_before=before, loss_after=after, gain=gain, factor=factor, factor_renewed=int(renewed),
            n_iter=n_iter, func_evals=evals, early_stop=int(early), wall_s=round(wall, 2)))
        print("  N=%d q=%d %s round %d restart %d: loss %.4e -> %.4e (gain %+.2e)  iters %d evals %d%s  %.0f s"
              % (a.N, a.q, phase, prog["round"], prog["restart"], before, after, gain, n_iter, evals,
                 "  EARLY STOP" if early else "", wall), flush=True)
        if prog["round_loss_first"] is None:
            prog["round_loss_first"] = before
        prog["restart"] += 1
        prog["in_round"] += 1
        prog["round_wall_s"] += wall
        return after, gain

    # -- timing mode (gate 5) -----------------------------------------------------
    if a.timing_restarts is not None:
        rows = []
        for _ in range(a.timing_restarts):
            t0 = time.time()
            after, gain = one_restart("timing")
            r = read_csv(paths["hist"])[-1]
            rows.append({k: r[k] for k in ("loss_before", "loss_after", "n_iter", "func_evals", "wall_s")})
        evals = sum(int(r["func_evals"]) for r in rows)
        wall = sum(float(r["wall_s"]) for r in rows)
        tim = dict(N=a.N, q=a.q, n_states=int(len(S_np)), restarts=rows,
                   us_per_state_per_evaluation=1e6 * wall / (evals * len(S_np)),
                   host=os.uname().nodename, threads=1)
        with open(paths["timing"], "w") as f:
            json.dump(tim, f, indent=1)
        print(json.dumps(tim, indent=1))
        return tim

    # -- the rounds ----------------------------------------------------------------
    stop_budget = a.stop_after
    started = prog["restart"]

    def out_of_budget():
        return stop_budget is not None and prog["restart"] - started >= stop_budget

    while prog["phase"] in ("train", "confirm"):
        if prog["phase"] == "train":
            while (prog["stalled"] < 2 and prog["restart"] < a.outer_cap
                   and (a.round_restarts is None or prog["in_round"] < a.round_restarts)):
                if out_of_budget():
                    save_progress()
                    print("stopped after %d restarts this invocation (resume by rerunning)" % stop_budget)
                    return None
                _, gain = one_restart("train")
                prog["stalled"] = 0 if gain >= STALL_TOL else prog["stalled"] + 1
                if prog["in_round"] <= 2:
                    prog["first_gains"].append(gain)
                save_progress()
            with torch.no_grad():
                last = loss_now().item()
            vmed, vx, vy = val_at_z1()
            append_csv(paths["rounds"], ROUND_FIELDS, dict(
                round=prog["round"], source=source, n_states=int(len(S_np)),
                per_plane=int(len(S_np) // a.N), restarts=prog["in_round"],
                loss_first=prog["round_loss_first"], loss_last=last, train_end_moved_med_um=moved,
                val_z1_pos_med_um=vmed, val_z1_x_med_um=vx, val_z1_y_med_um=vy,
                wall_s=round(prog["round_wall_s"], 1)))
            print("ROUND %d done (%s states): %d restarts, loss %.4e -> %.4e, val median at z1 %.1f um"
                  % (prog["round"], source, prog["in_round"], prog["round_loss_first"], last, vmed), flush=True)
            if prog["restart"] >= a.outer_cap:
                prog.update(phase="finish", hit_cap="restarts")
            elif (prog["round"] >= 2 and len(prog["first_gains"]) >= 2 and prog["in_round"] == 2
                  and all(g < STALL_TOL for g in prog["first_gains"][:2])):
                prog.update(phase="confirm" if not a.no_confirm else "finish", converged=True)
            elif prog["round"] >= a.round_cap:
                prog.update(phase="finish", hit_cap="rounds")
            else:
                prog.update(round=prog["round"] + 1, in_round=0, stalled=0, first_gains=[],
                            round_loss_first=None, round_wall_s=0.0)
                save_progress()
                S_np, z_np, source, moved = new_round_states()
                St, zt = torch.as_tensor(S_np), torch.as_tensor(z_np)
                zn = znodes_for(model, zt)
                W = weights(model, St, zt, sc["weighting"], a.weighting)
                state["factor"] = None
                continue
            save_progress()
        if prog["phase"] == "confirm":
            before_med = val_at_z1()[0]
            state["factor"] = None                  # fresh optimiser, same states
            gains = []
            for _ in range(2):
                if prog["restart"] >= a.outer_cap:
                    break
                _, g = one_restart("confirm")
                gains.append(g)
            after_med = val_at_z1()[0]
            prog["confirmed"] = bool(len(gains) == 2 and all(g < STALL_TOL for g in gains)
                                     and abs(after_med - before_med) <= 1e-2 * before_med)
            print("CONFIRMATION: gains %s, val median %.2f -> %.2f um -> confirmed %s"
                  % (["%.2e" % g for g in gains], before_med, after_med, prog["confirmed"]), flush=True)
            prog["phase"] = "finish"
            save_progress()

    # -- the record ----------------------------------------------------------------
    hist = read_csv(paths["hist"])
    states, cost = {}, {}
    for s in ("val", "test"):
        t0 = time.time()
        states[s] = carry(model, D["%s_S0" % s], Z0, a.N)
        cost[s] = (time.time() - t0) / len(states[s]) * 1e6
    scores = chain_scores(states, D, a.N, fld)
    np.savez_compressed(os.path.join(run, "chain_states.npz"), **{"%s_states" % s: v for s, v in states.items()})
    record = dict(
        N=a.N, q=a.q, dz_mm=dz, seed=a.seed, width=a.width, depth=a.depth, field=field,
        weighting=a.weighting, weighting_constants=sc["weighting"],
        n_parameters=int(sum(p.numel() for p in model.parameters())),
        n_train_tracks=int(len(S0_train)), states_per_round=a.states_per_round,
        round_restarts=a.round_restarts, outer_cap=a.outer_cap, round_cap=a.round_cap,
        rounds=int(prog["round"]), restarts=len(hist), converged=bool(prog["converged"]),
        confirmed=prog["confirmed"], hit_cap=prog["hit_cap"],
        early_stop_restarts=int(sum(int(r["early_stop"]) for r in hist)),
        final_loss=float(hist[-1]["loss_after"]) if hist else float("nan"),
        train_wall_s=round(sum(float(r["wall_s"]) for r in hist), 1),
        carry_us_per_track=cost, **scores)
    with open(paths["record"], "w") as f:
        json.dump(record, f, indent=1)
    prog["phase"] = "done"
    save_progress()
    t = record["test"]["vs_rk6_endpoint"]
    print("DONE N=%d q=%d: %d rounds, %d restarts (%d early stops), converged %s, confirmed %s; "
          "test at z1 vs RK6 median %.1f um (p95 %.1f); straight line %.1f um; %.0f s"
          % (a.N, a.q, record["rounds"], record["restarts"], record["early_stop_restarts"],
             record["converged"], record["confirmed"], t["pos_med_um"], t["pos_p95_um"],
             record["test"]["straight_line_vs_rk6_endpoint"]["pos_med_um"], time.time() - t_start), flush=True)
    return record


if __name__ == "__main__":
    main()
