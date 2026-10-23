"""The optimiser: full-batch L-BFGS in restarts, with the loss rescaled.

One restart is one call of L-BFGS, of at most `iterations_per_restart`
iterations. The loss handed to it is the loss of the run times a constant
factor:

  * at the start of a round the factor is 1 / loss, so the scaled loss starts
    the round at 1;
  * whenever the loss has fallen to `renew_scale_when_loss_falls_to` of what
    it was when the factor was set, the factor is set again, with a new
    optimiser, so that its record of the curvature stays that of the
    objective it minimises.

A constant factor does not move the minimum. It keeps the fixed absolute
tolerances of L-BFGS from ending a restart early, which is what stopped the
training of the study with one network per step.

Every number is a setting with no default. The values of the frozen trainer
were: 200 iterations, a history of 120, tolerances 1e-13 and 1e-16, the
strong Wolfe line search, renewal at 0.1.

Ported from `E1_Network_grid/train_network.py` (`make_opt`, `renew`,
`one_restart`), with the arithmetic unchanged.

A second optimiser, `self_scaled_broyden_restarts`, was added on 2026-10-23
to test a change the literature suggests (Urban, Stefanou and Pons,
arXiv:2405.04230, section 3.2): a quasi-Newton method that keeps the whole
inverse Hessian, updated by the self-scaled Broyden formula, in place of the
limited memory of L-BFGS. `lbfgs_restarts` is unchanged.
Gates: tests/test_training.py, tests/test_literature_changes.py.
"""
from __future__ import annotations

import math
import time

import torch
from torch.optim.lbfgs import _strong_wolfe

from rkpinn.registry import register


@register("optimiser", "lbfgs_restarts")
class LbfgsRestarts:
    name = "lbfgs_restarts"
    settings_in_a_configuration = (
        "iterations_per_restart", "history", "tolerance_of_the_gradient",
        "tolerance_of_the_change", "line_search", "renew_scale_when_loss_falls_to")

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(**settings)

    def __init__(self, *, iterations_per_restart, history, tolerance_of_the_gradient,
                 tolerance_of_the_change, line_search, renew_scale_when_loss_falls_to):
        self.iterations_per_restart = int(iterations_per_restart)
        self.history = int(history)
        self.tolerance_of_the_gradient = float(tolerance_of_the_gradient)
        self.tolerance_of_the_change = float(tolerance_of_the_change)
        self.line_search = line_search
        self.renew_scale_when_loss_falls_to = float(renew_scale_when_loss_falls_to)
        self._parameters = None
        self._state = None

    def _new_lbfgs(self):
        return torch.optim.LBFGS(
            self._parameters, max_iter=self.iterations_per_restart,
            history_size=self.history, tolerance_grad=self.tolerance_of_the_gradient,
            tolerance_change=self.tolerance_of_the_change,
            line_search_fn=self.line_search)

    def start_round(self, parameters) -> None:
        """A new round: a new optimiser, and the factor set at the first restart."""
        self._parameters = list(parameters)
        self._state = {"lbfgs": self._new_lbfgs(), "factor": None, "loss_at_factor": None,
                       "iterations": 0, "evaluations": 0}

    def _renew(self, loss_value):
        self._state.update(lbfgs=self._new_lbfgs(), factor=1.0 / loss_value,
                           loss_at_factor=loss_value, iterations=0, evaluations=0)

    def one_restart(self, loss_now) -> dict:
        """One restart. `loss_now` takes nothing and returns the loss of the
        run, unscaled, with its graph. Returns the record of the restart."""
        state = self._state
        with torch.no_grad():
            before = loss_now().item()
        renewed = False
        if (state["factor"] is None
                or before < self.renew_scale_when_loss_falls_to * state["loss_at_factor"]):
            self._renew(before)
            renewed = True
        lbfgs, factor = state["lbfgs"], state["factor"]

        def closure():
            lbfgs.zero_grad()
            loss = loss_now() * factor
            loss.backward()
            return loss

        started = time.time()
        lbfgs.step(closure)
        seconds = time.time() - started
        inner = lbfgs.state[lbfgs._params[0]]
        iterations = inner["n_iter"] - state["iterations"]
        evaluations = inner["func_evals"] - state["evaluations"]
        state["iterations"], state["evaluations"] = inner["n_iter"], inner["func_evals"]
        with torch.no_grad():
            after = loss_now().item()
        return {
            "loss_before": before, "loss_after": after,
            "gain": (before - after) / before,
            "factor": factor, "factor_renewed": int(renewed),
            "iterations": iterations, "evaluations": evaluations,
            "stopped_early": int(iterations < self.iterations_per_restart
                                 and evaluations < int(self.iterations_per_restart * 1.25)),
            "seconds": round(seconds, 2),
        }


@register("optimiser", "self_scaled_broyden_restarts")
class SelfScaledBroydenRestarts:
    """Full-batch quasi-Newton with the dense inverse Hessian H, updated by the
    self-scaled Broyden formula of arXiv:2405.04230, equations 9 to 23:

        s = step, y = change of the gradient,
        v = sqrt(y.Hy) (s / y.s - Hy / y.Hy)
        H <- (H - Hy Hy^T / y.Hy + phi v v^T) / tau + s s^T / y.s

    with tau = tau^(2) and phi = phi^(1) of equations 13 and 14. The step is
    found by the strong Wolfe line search of torch's L-BFGS.

    The restarts and the factor on the loss are those of `lbfgs_restarts`: one
    restart is at most `iterations_per_restart` iterations; the factor is set
    to 1 / loss at the first restart of a round and set again whenever the
    loss has fallen to `renew_scale_when_loss_falls_to` of what it was then.
    When the factor changes, H is multiplied by the ratio of the old factor
    to the new, so that it stays the inverse Hessian of the objective it
    minimises. H is kept across the restarts of a round and made again at the
    start of a round, whose states are new.

    H has (number of parameters)^2 entries: 2.7 GB at 18,440 parameters. Its
    products are done on `threads_for_the_matrix` threads; the evaluation of
    the loss stays on the threads of the run.
    """

    name = "self_scaled_broyden_restarts"
    settings_in_a_configuration = (
        "iterations_per_restart", "tolerance_of_the_gradient", "tolerance_of_the_change",
        "renew_scale_when_loss_falls_to", "threads_for_the_matrix")

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(**settings)

    def __init__(self, *, iterations_per_restart, tolerance_of_the_gradient,
                 tolerance_of_the_change, renew_scale_when_loss_falls_to,
                 threads_for_the_matrix):
        self.iterations_per_restart = int(iterations_per_restart)
        self.tolerance_of_the_gradient = float(tolerance_of_the_gradient)
        self.tolerance_of_the_change = float(tolerance_of_the_change)
        self.renew_scale_when_loss_falls_to = float(renew_scale_when_loss_falls_to)
        self.threads_for_the_matrix = int(threads_for_the_matrix)
        self._parameters = None
        self._state = None

    def start_round(self, parameters) -> None:
        self._parameters = list(parameters)
        self._state = {"H": None, "factor": None, "loss_at_factor": None}

    # -- the parameters as one vector -------------------------------------------------
    def _vector(self):
        return torch.cat([p.detach().reshape(-1) for p in self._parameters])

    def _set(self, vector):
        offset = 0
        with torch.no_grad():
            for p in self._parameters:
                n = p.numel()
                p.copy_(vector[offset:offset + n].view_as(p))
                offset += n

    def _loss_and_gradient(self, loss_now, factor):
        for p in self._parameters:
            p.grad = None
        loss = loss_now() * factor
        gradient = torch.autograd.grad(loss, self._parameters)
        return float(loss.detach()), torch.cat([g.reshape(-1) for g in gradient])

    # -- the update of the inverse Hessian -----------------------------------------------
    @staticmethod
    def scaling_and_updating(s, y, g, step, Hy, n):
        """tau^(2) and phi^(1) of arXiv:2405.04230, equations 11 to 23, and
        appendix B for s.H^-1 s = -step s.g. Returns (tau, phi)."""
        ys = float(y @ s)
        yHy = float(y @ Hy)
        sBs = -step * float(s @ g)
        b = sBs / ys
        h = yHy / ys
        a = h * b - 1.0
        tau_1 = min(1.0, 1.0 / b)                                # y.s / s.H^-1 s
        if not (a > 1e-12 and math.isfinite(a)):
            return tau_1, 1.0                                    # BFGS: s and Hy parallel
        c = math.sqrt(a / (a + 1.0))
        rho_minus = min(1.0, h * (1.0 - c))
        theta_minus = (rho_minus - 1.0) / a
        theta_plus = 1.0 / rho_minus
        theta = max(theta_minus, min(theta_plus, (1.0 - b) / b))
        sigma = 1.0 + a * theta
        root = sigma ** (-1.0 / (n - 1))
        if theta > 0:
            tau = tau_1 * min(root, 1.0 / theta)
        else:
            tau = min(tau_1 * root, sigma)
        phi = (1.0 - theta) / (1.0 + a * theta)
        return tau, phi

    def _update(self, H, s, y, g, step):
        threads = torch.get_num_threads()
        torch.set_num_threads(self.threads_for_the_matrix)
        try:
            Hy = H @ y
            tau, phi = self.scaling_and_updating(s, y, g, step, Hy, len(s))
            ys, yHy = float(y @ s), float(y @ Hy)
            # v = sqrt(yHy) (s/ys - Hy/yHy); every term is in the span of s and Hy:
            # H <- H/tau + U M U^T with U = [s, Hy]
            M = torch.tensor([
                [phi * yHy / (ys * ys) / tau + 1.0 / ys, -phi / ys / tau],
                [-phi / ys / tau, (phi - 1.0) / yHy / tau]], dtype=H.dtype)
            U = torch.stack([s, Hy], dim=1)
            H.mul_(1.0 / tau).addmm_(U, M @ U.T)
        finally:
            torch.set_num_threads(threads)
        return tau, phi

    def _direction(self, H, g):
        threads = torch.get_num_threads()
        torch.set_num_threads(self.threads_for_the_matrix)
        try:
            return -(H @ g)
        finally:
            torch.set_num_threads(threads)

    def one_restart(self, loss_now) -> dict:
        state = self._state
        with torch.no_grad():
            before = loss_now().item()
        renewed = False
        if (state["factor"] is None
                or before < self.renew_scale_when_loss_falls_to * state["loss_at_factor"]):
            new = 1.0 / before
            if state["H"] is not None:
                state["H"].mul_(state["factor"] / new)
            state.update(factor=new, loss_at_factor=before)
            renewed = True
        factor = state["factor"]
        x = self._vector()
        started = time.time()
        f, g = self._loss_and_gradient(loss_now, factor)
        iterations = evaluations = 0
        evaluations += 1

        def objective(x0, t, d):
            self._set(x0 + t * d)
            return self._loss_and_gradient(loss_now, factor)

        for _ in range(self.iterations_per_restart):
            if float(g.abs().max()) <= self.tolerance_of_the_gradient:
                break
            H = state["H"]
            if H is None:
                d = -g
                t = min(1.0, 1.0 / float(g.abs().sum()))
            else:
                d = self._direction(H, g)
                t = 1.0
            gtd = float(g @ d)
            if gtd > -self.tolerance_of_the_change:
                if H is None:
                    break
                state["H"] = None                     # not a descent direction: start again
                continue
            f_new, g_new, t, used = _strong_wolfe(objective, x, t, d, f, g, gtd)
            evaluations += int(used)
            iterations += 1
            s = t * d
            y = g_new - g
            x = x + s
            self._set(x)
            if float(y @ s) > 1e-300:
                if state["H"] is None:
                    n = len(x)
                    state["H"] = torch.eye(n, dtype=x.dtype).mul_(float(y @ s) / float(y @ y))
                self._update(state["H"], s, y, g, t)
            change = abs(f_new - f)
            f, g = f_new, g_new
            if float(s.abs().max()) <= self.tolerance_of_the_change or \
                    change < self.tolerance_of_the_change:
                break
        self._set(x)
        seconds = time.time() - started
        with torch.no_grad():
            after = loss_now().item()
        return {
            "loss_before": before, "loss_after": after,
            "gain": (before - after) / before,
            "factor": factor, "factor_renewed": int(renewed),
            "iterations": iterations, "evaluations": evaluations,
            "stopped_early": int(iterations < self.iterations_per_restart),
            "seconds": round(seconds, 2),
        }
