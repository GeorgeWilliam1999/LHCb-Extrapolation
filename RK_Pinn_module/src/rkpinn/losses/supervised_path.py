"""The supervised path loss: the loss of the path network.

    uncertainty none:
        loss = mean over n, k, d of  ( (emitted - label)[n, k, d] / s[d] ) ** 2

    uncertainty diagonal:
        sigma[n, k, d] = s[d] * exp(extra[n, k, d])
        loss = mean over n, k, d of  ( (emitted - label) / sigma ) ** 2 + 2 log(sigma / s[d])

over every emitted state k (the stages and the end plane of every step) and
the four components d. It needs labels, the reference path, and no equation
of motion. With `diagonal` the network emits one extra number per state and
component, and the loss is the Gaussian negative log-likelihood up to a
constant: the sigma is the error the network claims, and the report checks
how often the truth lies inside it.

The divisor s is one constant per component. In a configuration, `divisor`
is "spread_of_target_states" (the spread of the labels over all states,
measured once), or four numbers.

Ported onto the stbc checkout on 2026-11-04. Gates: tests/test_supervised_path.py.
"""
from __future__ import annotations

import torch

from rkpinn.registry import register

UNCERTAINTIES = ("none", "diagonal")


@register("loss", "supervised_path")
class SupervisedPath:
    name = "supervised_path"
    needs_labels = True
    settings_in_a_configuration = ("divisor", "uncertainty")

    @classmethod
    def from_configuration(cls, settings, context):
        # "spread_of_target_states" is measured by the run over the labels of its
        # first round, (n, steps, q + 1, 4), all states pooled (round_trainer.py)
        return cls(divisor=context.scale(settings["divisor"], 4),
                   uncertainty=settings["uncertainty"])

    def __init__(self, *, divisor, uncertainty, equation_of_motion=None):
        self.equation_of_motion = equation_of_motion      # accepted, not used
        self.divisor = [float(v) for v in divisor]
        if len(self.divisor) != 4 or not all(v > 0 for v in self.divisor):
            raise ValueError("the divisor has four positive components")
        if uncertainty not in UNCERTAINTIES:
            raise ValueError("uncertainty is %s, not %r" % (" or ".join(UNCERTAINTIES), uncertainty))
        self.uncertainty = uncertainty

    def constants(self, first_round_states=None) -> dict:
        return {"name": self.name, "uncertainty": self.uncertainty,
                "divisor_per_component": list(self.divisor)}

    def emitted_states(self, predicted_track):
        """Shape (n, steps, q + 1, 4): the stages, then the end state, of every step."""
        stages = torch.as_tensor(predicted_track.stage_states)
        ends = torch.as_tensor(predicted_track.end_states)
        return torch.cat([stages, ends[:, :, None, :]], dim=2)

    def squared_terms(self, predicted_track, target, constants):
        """The term of every state and component, shape (n, steps, q + 1, 4).
        The loss is its mean. With `diagonal` a term is the Gaussian one, which
        can be negative."""
        if target is None:
            raise ValueError("the supervised path loss needs the reference path")
        divisor = torch.as_tensor(constants["divisor_per_component"], dtype=torch.float64)
        labels = torch.as_tensor(target, dtype=torch.float64)
        scaled = (torch.as_tensor(self.emitted_states(predicted_track)) - labels) / divisor
        if constants["uncertainty"] == "none":
            return scaled ** 2
        extra = predicted_track.extra_outputs
        if extra is None or extra.shape[-1] != 4:
            raise ValueError("the diagonal uncertainty needs a network with extra_outputs 4")
        extra = torch.as_tensor(extra)
        return (scaled * torch.exp(-extra)) ** 2 + 2 * extra

    def value(self, predicted_track, target, constants):
        return self.squared_terms(predicted_track, target, constants).mean()
