"""B.1 and Part C of the per-step bias study: averaging and inference-time
corrections of the trained networks. Nothing here changes the package or a
run; every variant is built in this file from a run's snapshots.

    part_bc.py references       the exact 2-stage Gauss-Legendre chain and classic RK4,
                                64 steps (the two comparators of the pass criteria)
    part_bc.py variants <key>   every variant of one run:
        plain                   the last round, as trained
        B.1 output averaging    the mean of the end states of the last 5 snapshots, each
                                chained on its own (the tranche-1 analysis)
        B.1 weight averaging    one network with the mean of the last 5 snapshots' weights
        C.1 trees / cubic       each step's end state minus the A.1 model of e
                                (supervised: the model was fitted to reference errors)
        C.2 1 / 2 sweeps        fixed-point sweeps of the stage equations from the
                                network's stages, end state summed
        C.3 Newton              one Newton step of the stage equations from the network's
                                stages, with the gradient of the trilinear map
        C.4 distilled           a small network, trained without reference labels, that
                                predicts the C.3 change of the end state from the state

For every variant: the test endpoint radial median and its value per momentum
band, the mean signed endpoint error +- standard error, the lag-1
autocorrelation of the corrected single-step errors, and the field and
network calls per track (64 steps).
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch

from common import (CACHE, EQUATION, NUMBER_OF_STEPS, RESULTS, RUNS, SHORT, SNAPSHOTS_AVERAGED,
                    START_PLANES_MM, STEP_MM, TO_REPORTED, chain, endpoint_scores,
                    lag_one_scores, last_round, network_of, on_planes, plain_step,
                    setup_threads, single_step_errors, split, stages_and_end, write_rows)
import stage_equations as se

DISTILLED_WIDTH, DISTILLED_DEPTH = 64, 2
DISTILLED_EPOCHS = 60
DISTILLED_BATCH = 4096


def last_rounds(key):
    last = last_round(key)
    return list(range(last - SNAPSHOTS_AVERAGED + 1, last + 1))


def row_of(series, key, variant, final, e, field_calls, network_calls, other_calls=""):
    scores, _ = endpoint_scores(final)
    row = dict(series=series, run_key=key, variant=variant,
               field_calls_per_track=field_calls, network_calls_per_track=network_calls,
               other_calls_per_track=other_calls)
    row.update(scores)
    row.update(lag_one_scores(e) if e is not None else {"lag1_x_median": "", "lag1_y_median": ""})
    print(row, flush=True)
    return row


# ------------------------------------------------------------------ comparators
def rk4_step(states, planes):
    h = STEP_MM
    z = planes
    k1 = EQUATION.rates(states, z)
    k2 = EQUATION.rates(states + 0.5 * h * k1, z + 0.5 * h)
    k3 = EQUATION.rates(states + 0.5 * h * k2, z + 0.5 * h)
    k4 = EQUATION.rates(states + h * k3, z + h)
    return (states + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))[:, :4]


def references():
    test = split("test")
    rows = []
    exact = lambda s, z: se.exact_step(s, z)[0]                     # noqa: E731
    for name, step, calls in (("exact 2-stage Gauss-Legendre (solved)", exact, ""),
                              ("classic RK4", rk4_step, 256)):
        final = chain(step, test.start_state)[:, -1]
        e = single_step_errors(None, "test", step)
        rows.append(row_of(name, "", "comparator", final, e, calls, 0))
    write_rows("c_comparators.csv", rows)


# ------------------------------------------------------------------ C.1
def model_correction(key, kind):
    with open(os.path.join(CACHE, "a1_%s_%s.pkl" % (kind, key)), "rb") as handle:
        models = pickle.load(handle)

    def predicted_e(states, planes):
        X = np.concatenate([states, planes[:, None]], axis=1)
        return np.stack([m.predict(X) for m in models], axis=1) / TO_REPORTED
    return predicted_e


# ------------------------------------------------------------------ C.4
class Distilled(torch.nn.Module):
    def __init__(self, mean_in, scale_in, scale_out):
        super().__init__()
        layers, n = [], 6
        for _ in range(DISTILLED_DEPTH):
            layers += [torch.nn.Linear(n, DISTILLED_WIDTH, dtype=torch.float64), torch.nn.Tanh()]
            n = DISTILLED_WIDTH
        layers += [torch.nn.Linear(n, 4, dtype=torch.float64)]
        self.body = torch.nn.Sequential(*layers)
        self.register_buffer("mean_in", torch.as_tensor(mean_in))
        self.register_buffer("scale_in", torch.as_tensor(scale_in))
        self.register_buffer("scale_out", torch.as_tensor(scale_out))

    def forward(self, X):
        return self.body((X - self.mean_in) / self.scale_in) * self.scale_out


def newton_change(net, states, planes, rows_at_a_time=100000):
    """The C.3 change of the end state, (n, 4), from the stage equations only."""
    out = np.empty((len(states), 4))
    for i in range(0, len(states), rows_at_a_time):
        s, z = states[i:i + rows_at_a_time], planes[i:i + rows_at_a_time]
        stages, end = stages_and_end(net, s, z)
        out[i:i + rows_at_a_time] = se.polish(s, z, stages, newton_steps=1)[0] - end
    return out


def own_states(net, name):
    """The network's own chained states at the start of every step, (n*64, 5), and planes."""
    states = chain(plain_step(net), split(name).start_state)[:, :-1]
    planes = np.broadcast_to(START_PLANES_MM, states.shape[:2])
    return states.reshape(-1, 5), planes.reshape(-1).copy()


def train_distilled(key, net):
    file = os.path.join(CACHE, "c4_distilled_%s.pt" % key)
    Xs, Ys = {}, {}
    for name in ("training", "validation"):
        s, z = own_states(net, name)
        Xs[name] = np.concatenate([s, z[:, None]], axis=1)
        Ys[name] = newton_change(net, s, z)
    X, Y = torch.as_tensor(Xs["training"]), torch.as_tensor(Ys["training"])
    model = Distilled(X.mean(0), X.std(0), Y.std(0))
    if os.path.exists(file):
        model.load_state_dict(torch.load(file, weights_only=True))
    else:
        torch.manual_seed(1)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, DISTILLED_EPOCHS * (len(X) // DISTILLED_BATCH + 1))
        for epoch in range(DISTILLED_EPOCHS):
            order = torch.randperm(len(X))
            for i in range(0, len(X), DISTILLED_BATCH):
                b = order[i:i + DISTILLED_BATCH]
                loss = (((model(X[b]) - Y[b]) / model.scale_out) ** 2).mean()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                schedule.step()
            if epoch % 10 == 9:
                print("distilled", key, epoch + 1, float(loss), flush=True)
        torch.save(model.state_dict(), file)
    with torch.no_grad():
        Xv, Yv = torch.as_tensor(Xs["validation"]), torch.as_tensor(Ys["validation"])
        r2 = 1 - ((model(Xv) - Yv) ** 2).sum(0) / ((Yv - Yv.mean(0)) ** 2).sum(0)
    print("distilled", key, "validation R^2 of the Newton change", r2.tolist(), flush=True)
    return model, r2.tolist()


# ------------------------------------------------------------------ variants
def variants(key):
    series = SHORT[key]
    test = split("test")
    start = test.start_state
    rows = []
    net = network_of(key)
    plain = plain_step(net)
    rows.append(row_of(series, key, "plain (last round)", chain(plain, start)[:, -1],
                       single_step_errors(net, "test"), 128, 64))

    # B.1 output averaging: each snapshot chained on its own, end states averaged
    finals, es = [], []
    for r in last_rounds(key):
        n_r = network_of(key, [r])
        finals.append(chain(plain_step(n_r), start)[:, -1])
        es.append(single_step_errors(n_r, "test"))
    rows.append(row_of(series, key, "B.1 output averaging, last 5", np.mean(finals, axis=0),
                       np.mean(es, axis=0), 128 * 5, 64 * 5))

    # B.1 weight averaging
    wnet = network_of(key, last_rounds(key))
    rows.append(row_of(series, key, "B.1 weight averaging, last 5",
                       chain(plain_step(wnet), start)[:, -1], single_step_errors(wnet, "test"),
                       128, 64))
    net = network_of(key)                                            # back to the last round

    # C.1 subtract the A.1 model of e
    for kind in ("trees", "cubic"):
        model = model_correction(key, kind)
        step = (lambda m: lambda s, z: plain(s, z) - m(s, z))(model)
        rows.append(row_of(series, key, "C.1 subtract A.1 %s model (supervised)" % kind,
                           chain(step, start)[:, -1], single_step_errors(None, "test", step),
                           128, 64, "64 %s-model calls" % kind))

    # C.2 fixed-point sweeps, C.3 one Newton step
    for label, sweeps, newton, calls in (("C.2 one fixed-point sweep", 1, 0, 256),
                                         ("C.2 two fixed-point sweeps", 2, 0, 384),
                                         ("C.3 one Newton step", 0, 1, 256)):
        def step(s, z, sweeps=sweeps, newton=newton):
            stages = stages_and_end(net, s, z)[0]
            return se.polish(s, z, stages, sweeps=sweeps, newton_steps=newton)[0]
        rows.append(row_of(series, key, label, chain(step, start)[:, -1],
                           single_step_errors(None, "test", step), calls, 64,
                           "128 of the field calls also give the field gradient" if newton else ""))

    # C.4 distilled Newton change
    model, r2 = train_distilled(key, net)

    def distilled_step(s, z):
        with torch.no_grad():
            X = torch.as_tensor(np.concatenate([s, z[:, None]], axis=1))
            return plain(s, z) + model(X).numpy()
    rows.append(row_of(series, key, "C.4 distilled Newton change", chain(distilled_step, start)
                       [:, -1], single_step_errors(None, "test", distilled_step), 128, 64,
                       "64 calls of a 2x64 network; validation R^2 of the change %s"
                       % " ".join("%.3f" % v for v in r2)))
    write_rows("c_variants_%s.csv" % key, rows)


if __name__ == "__main__":
    setup_threads()
    if sys.argv[1] == "references":
        references()
    else:
        variants(sys.argv[2])
