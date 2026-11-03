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

Draft of 2026-11-03, written off GitHub 184cf50e. Gates:
tests/test_supervised_path.py (planned).
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.losses.pooled import spread_of_components
from rkpinn.registry import register

UNCERTAINTIES = ("none", "diagonal")
SPREAD_OF_TARGET_STATES = "spread_of_target_states"


@register("loss", "supervised_path")
class SupervisedPath:
    name = "supervised_path"
    needs_labels = True
    settings_in_a_configuration = ("divisor", "uncertainty")

    @classmethod
    def from_configuration(cls, settings, context):
        divisor = settings["divisor"]
        if divisor == SPREAD_OF_TARGET_STATES:
            labels = np.asarray(context.first_round_labels)        # (n, steps, q + 1, 4)
            divisor = spread_of_components(labels.reshape(-1, 4)).tolist()
        return cls(divisor=divisor, uncertainty=settings["uncertainty"])

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
        return torch.cat([predicted_track.stage_states,
                          predicted_track.end_states[:, :, None, :]], dim=2)

    def value(self, predicted_track, target, constants):
        divisor = torch.as_tensor(constants["divisor_per_component"], dtype=torch.float64)
        labels = torch.as_tensor(target, dtype=torch.float64)
        scaled = (self.emitted_states(predicted_track) - labels) / divisor
        if constants["uncertainty"] == "none":
            return (scaled ** 2).mean()
        extra = getattr(predicted_track, "extra_outputs", None)
        if extra is None:
            raise ValueError("the diagonal uncertainty needs a network with extra_outputs 4")
        return ((scaled * torch.exp(-extra)) ** 2 + 2 * extra).mean()
