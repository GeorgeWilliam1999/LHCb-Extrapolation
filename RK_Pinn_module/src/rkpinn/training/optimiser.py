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
Gates: tests/test_training.py.
"""
from __future__ import annotations

import time

import torch

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
