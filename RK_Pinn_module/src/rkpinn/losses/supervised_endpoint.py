"""The supervised endpoint loss: the loss of the whole-crossing network.

    loss = mean over n, d of  ( (end state predicted - end state of the target)[n, d] / s[d] ) ** 2

over the four components x, y, slope in x and slope in y. It is not a residual
loss: it needs a label, the end state of the target, and no equation of motion.

The divisor s is one constant per component. It is the spread of the states
given to `constants`. Which states those are, the input states or the end
states of the target, is a setting of the experiment.

It reads the end state of the last step of the predicted track, so it applies
to any network.
"""
from __future__ import annotations

import torch

from rkpinn.losses.pooled import spread_of_components
from rkpinn.registry import register


@register("loss", "supervised_endpoint")
class SupervisedEndpoint:
    name = "supervised_endpoint"
    needs_labels = True

    def __init__(self, *, equation_of_motion=None):
        # accepted so that every loss is built the same way; it is not used
        self.equation_of_motion = equation_of_motion

    def constants(self, first_round_states) -> dict:
        return {"name": self.name,
                "divisor_per_component":
                    spread_of_components(first_round_states)[:4].tolist()}

    def value(self, predicted_track, target, constants):
        if target is None:
            raise ValueError("the supervised endpoint loss needs the end states of a target")
        predicted = torch.as_tensor(predicted_track.end_states)[:, -1]
        wanted = torch.as_tensor(target, dtype=torch.float64)[:, :4]
        divisor = torch.as_tensor(constants["divisor_per_component"], dtype=torch.float64)
        return (((predicted - wanted) / divisor) ** 2).mean()
