"""The supervised endpoint loss: the loss of the whole-crossing network.

    loss = mean over n, d of  ( (end state predicted - end state of the target)[n, d] / s[d] ) ** 2

over the four components x, y, slope in x and slope in y. It is not a residual
loss: it needs a label, the end state of the target, and no equation of motion.

The divisor s is one constant per component. It normalises the error, so that
millimetres and slopes count equally; nothing is subtracted and no penalty is
added. What it is is left to each experiment (George, 2026-09-29). In a
configuration, `divisor` is

  "spread_of_first_round_states"   the spread of the input states
  "spread_of_target_end_states"    the spread of the end states of the target
  four numbers                     given outright

Built directly, the divisor is four numbers, or
"spread_of_the_states_given_to_constants".

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
    settings_in_a_configuration = ("divisor",)
    MEASURED_WHEN_CONSTANTS_ARE_TAKEN = "spread_of_the_states_given_to_constants"

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(divisor=context.scale(settings["divisor"], 4))

    def __init__(self, *, divisor, equation_of_motion=None):
        # the equation of motion is accepted so that every loss is built the
        # same way; it is not used
        self.equation_of_motion = equation_of_motion
        if isinstance(divisor, str):
            if divisor != self.MEASURED_WHEN_CONSTANTS_ARE_TAKEN:
                raise ValueError("the divisor is four numbers or %r, not %r"
                                 % (self.MEASURED_WHEN_CONSTANTS_ARE_TAKEN, divisor))
            self.divisor = divisor
        else:
            self.divisor = [float(v) for v in divisor]
            if len(self.divisor) != 4 or not all(v > 0 for v in self.divisor):
                raise ValueError("the divisor has four positive components")

    def constants(self, first_round_states=None) -> dict:
        if self.divisor == self.MEASURED_WHEN_CONSTANTS_ARE_TAKEN:
            divisor = spread_of_components(first_round_states)[:4].tolist()
        else:
            divisor = list(self.divisor)
        return {"name": self.name, "divisor_per_component": divisor}

    def value(self, predicted_track, target, constants):
        if target is None:
            raise ValueError("the supervised endpoint loss needs the end states of a target")
        predicted = torch.as_tensor(predicted_track.end_states)[:, -1]
        wanted = torch.as_tensor(target, dtype=torch.float64)[:, :4]
        divisor = torch.as_tensor(constants["divisor_per_component"], dtype=torch.float64)
        return (((predicted - wanted) / divisor) ** 2).mean()
