"""Output forms: how the raw outputs of a network become states.

The body of a network ends in a linear layer, whose outputs are of order one.
An output form turns them into states in millimetres and slopes. The form is
a setting of a run, named in its configuration.

One form is built:

  direct_states   state = scale of the component * raw output.
                  The network emits the states themselves, as in Raissi,
                  Perdikaris and Karniadakis (2019). Ruled by George on
                  2026-09-28.

The scale is one fixed constant per component, (x, y, slope in x, slope in y).
It is an argument. How it is measured, or whether it is one, is a setting of
the experiment (George, 2026-09-28).

Every output form declares what it returns when the network's last layer is
zero, and a gate checks it.

Place kept open: the straight line plus a scaled correction would be a second
form here. It is not built.
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.registry import register


@register("output_form", "direct_states")
class DirectStates(torch.nn.Module):
    """state = scale * raw output, component by component."""

    name = "direct_states"

    def __init__(self, scale_of_outputs):
        super().__init__()
        scale = np.asarray(scale_of_outputs, dtype=np.float64).reshape(-1)
        if scale.shape != (4,):
            raise ValueError("the scale of the outputs has four components: "
                             "x, y, slope in x, slope in y")
        if not (np.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError("the scale of the outputs must be positive and finite")
        self.register_buffer("scale_of_outputs", torch.as_tensor(scale))

    def states_from_raw(self, raw, input_states, planes_mm):
        """raw: shape (n, number of outputs, 4). The input states and the planes
        are not used by this form; they are arguments because other forms need them."""
        return raw * self.scale_of_outputs

    def answer_with_zeroed_network(self, input_states, planes_mm):
        """With the last layer zero the raw outputs are zero, and so is every state."""
        return torch.zeros(planes_mm.shape + (4,), dtype=torch.float64)

    def forward(self, raw, input_states, planes_mm):
        return self.states_from_raw(raw, input_states, planes_mm)
