"""The whole-crossing network: the supervised twin.

A separate network, applied once and not chained. It takes the state on the
first plane of the crossing and emits the state on the last plane. It has no
stages and no equation of motion in it; it is trained on the reference end
state. It is the comparison that shows what labels buy.

Settings, all arguments, none with a default: the two planes, the scale of the
inputs, the output form, the width and the depth.

The body, its initialisation and its arithmetic are those of the frozen
`OneStepNetwork` (`_shared/model.py`) with no stages and no extra input, so
with the same seed and scales the two give identical numbers.
Gates: tests/test_networks.py.
"""
from __future__ import annotations

import torch

from rkpinn.networks.stage_network import body_of_tanh_layers, positive_scale
from rkpinn.predicted_track.predicted_track import END_STATE_PREDICTED, PredictedTrack
from rkpinn.predicted_track.track_layout import TrackLayout
from rkpinn.registry import register


@register("network", "whole_crossing_network")
class WholeCrossingNetwork(torch.nn.Module):
    kind = "whole_crossing_network"

    def __init__(self, *, first_plane_mm, last_plane_mm, scale_of_inputs, output_form,
                 width, depth):
        super().__init__()
        self.first_plane_mm = float(first_plane_mm)
        self.last_plane_mm = float(last_plane_mm)
        self.width, self.depth = int(width), int(depth)
        self.output_form = output_form
        self.register_buffer("scale_of_inputs",
                             positive_scale(scale_of_inputs, 5, "the scale of the inputs"))
        self.body = body_of_tanh_layers(5, 4, width, depth)
        self.layout = TrackLayout(self.first_plane_mm, self.last_plane_mm, 1, None)

    @property
    def step_length_mm(self) -> float:
        return self.last_plane_mm - self.first_plane_mm

    def forward(self, states):
        """The state on the last plane, shape (n, 4)."""
        raw = self.body(states / self.scale_of_inputs).reshape(-1, 1, 4)
        planes = torch.full((len(states), 1), self.last_plane_mm, dtype=torch.float64)
        return self.output_form.states_from_raw(raw, states, planes)[:, 0]

    def predict(self, states, start_planes_mm=None) -> PredictedTrack:
        """The whole crossing as one step with no stages. The start plane is
        the first plane for every row; the argument is there because every
        network has it, and anything else is refused."""
        first = torch.full((len(states),), self.first_plane_mm, dtype=torch.float64)
        if start_planes_mm is not None and not torch.equal(
                torch.as_tensor(start_planes_mm, dtype=torch.float64), first):
            raise ValueError("the whole-crossing network starts on its first plane, "
                             "%r mm, and nowhere else" % self.first_plane_mm)
        return PredictedTrack(
            step_length_mm=self.step_length_mm, tableau=None,
            start_planes_mm=first[:, None], input_states=states[:, None, :],
            stage_states=None, end_states=self.forward(states)[:, None],
            end_state_was=END_STATE_PREDICTED, layout=self.layout,
            produced_by=self.kind)

    def whole_track(self, states, layout=None) -> PredictedTrack:
        """The same as `predict`, without a gradient, from numpy states."""
        with torch.no_grad():
            track = self.predict(torch.as_tensor(states, dtype=torch.float64))
        track.start_planes_mm = track.start_planes_mm.numpy()
        track.input_states = track.input_states.numpy()
        track.end_states = track.end_states.numpy()
        return track

    def settings(self) -> dict:
        return {
            "kind": self.kind, "first_plane_mm": self.first_plane_mm,
            "last_plane_mm": self.last_plane_mm,
            "scale_of_inputs": self.scale_of_inputs.tolist(),
            "output_form": self.output_form.name,
            "scale_of_outputs": self.output_form.scale_of_outputs.tolist(),
            "width": self.width, "depth": self.depth,
        }


def build_whole_crossing_network(*, seed, **settings) -> WholeCrossingNetwork:
    """A whole-crossing network with its weights drawn from the seed."""
    torch.manual_seed(int(seed))
    return WholeCrossingNetwork(**settings)
