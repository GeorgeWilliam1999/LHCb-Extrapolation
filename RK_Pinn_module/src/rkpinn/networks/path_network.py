"""The path network: one call emits the whole crossing of a track.

It takes the state on the first plane, (x, y, tx, ty, q/p), and emits every
stage state and every plane state of the layout at once: N steps x (q + 1)
states x 4 components, 768 numbers at 64 steps and 2 stages. It is not
chained and it contains no equation of motion. It is trained on the reference
path (targets/reference_path.py), so it is supervised; the comparison with the
chain says this plainly.

Its output form is applied to every emitted state with the start state as
the input state, so with the straight-line form the network learns the bend
from the straight line through the whole region.

It fills a PredictedTrack with the layout set, so the standard report scores
it like a chain, and `evaluation/newton_sweep.py` corrects it with one Newton
step on the stage equations of every step at once.

`has_stages` is true, because the track it fills has stages; it trains on the
start states of the tracks, not on drawn states, which it declares with
`trains_on_start_states` (training_protocols.py, one line).

With `extra_outputs` = 4 the body emits one more number per state, read by the
Gaussian path loss as the logarithm of a sigma per component. With 0 it emits
the states only.

Draft of 2026-11-03, written off GitHub 184cf50e; to be ported onto the stbc
checkout, where the straight-line output forms live. Gates:
tests/test_path_network.py (planned).
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.networks.stage_network import body_of_layers, check_activation, positive_scale
from rkpinn.predicted_track.predicted_track import END_STATE_PREDICTED, PredictedTrack
from rkpinn.registry import register


@register("network", "path_network")
class PathNetwork(torch.nn.Module):
    kind = "path_network"
    has_stages = True
    trains_on_start_states = True
    settings_in_a_configuration = (
        "output_form", "width", "depth", "activation", "extra_outputs",
        "scale_of_inputs", "scale_of_outputs")

    @classmethod
    def from_configuration(cls, settings, context):
        from rkpinn.registry import component
        output_form = component("output_form", settings["output_form"])(
            context.scale(settings["scale_of_outputs"], 4))
        torch.manual_seed(int(context.seed))
        return cls(layout=context.layout,
                   scale_of_inputs=context.scale(settings["scale_of_inputs"], 5),
                   output_form=output_form, width=settings["width"], depth=settings["depth"],
                   activation=settings["activation"], extra_outputs=settings["extra_outputs"])

    def __init__(self, *, layout, scale_of_inputs, output_form, width, depth, activation,
                 extra_outputs):
        super().__init__()
        self.layout = layout
        self.tableau = layout.tableau
        self.number_of_steps = int(layout.number_of_steps)
        self.number_of_stages = int(layout.number_of_stages)
        self.step_length_mm = float(layout.step_length_mm)
        self.first_plane_mm = float(layout.first_plane_mm)
        self.width, self.depth = int(width), int(depth)
        self.activation = check_activation(activation)
        self.extra_outputs = int(extra_outputs)
        if self.extra_outputs not in (0, 4):
            raise ValueError("extra_outputs is 0 (states only) or 4 (a sigma per component)")
        self.output_form = output_form
        self.register_buffer("scale_of_inputs",
                             positive_scale(scale_of_inputs, 5, "the scale of the inputs"))
        # the plane of every emitted state: for step n its stages, then its end plane
        planes = np.empty((self.number_of_steps, self.number_of_stages + 1))
        for n in range(self.number_of_steps):
            planes[n, :self.number_of_stages] = layout.stage_planes_mm(n)
            planes[n, self.number_of_stages] = layout.planes_mm[n + 1]
        self.register_buffer("planes_of_outputs_mm", torch.as_tensor(planes.reshape(-1)))
        self.states_emitted = self.number_of_steps * (self.number_of_stages + 1)
        self.body = body_of_layers(5, self.states_emitted * (4 + self.extra_outputs),
                                   width, depth, activation)

    def settings(self) -> dict:
        return {"kind": self.kind, "width": self.width, "depth": self.depth,
                "activation": self.activation, "extra_outputs": self.extra_outputs,
                "scale_of_inputs": self.scale_of_inputs.tolist(),
                "output_form": self.output_form.name}

    def raw(self, states):
        """The outputs of the body, shape (n, states emitted, 4 + extra)."""
        return self.body(states / self.scale_of_inputs).reshape(
            -1, self.states_emitted, 4 + self.extra_outputs)

    def forward(self, states):
        """The emitted states, shape (n, steps, stages + 1, 4), and the extra
        outputs, shape (n, steps, stages + 1, extra) or None."""
        raw = self.raw(states)
        planes = self.planes_of_outputs_mm[None, :].expand(len(states), -1)
        emitted = self.output_form.states_from_raw(raw[..., :4], states, planes)
        shape = (len(states), self.number_of_steps, self.number_of_stages + 1)
        extra = raw[..., 4:].reshape(shape + (self.extra_outputs,)) if self.extra_outputs else None
        return emitted.reshape(shape + (4,)), extra

    def predict(self, states, start_planes_mm=None) -> PredictedTrack:
        """The whole crossing from every start state. Every row starts on the
        first plane; another start plane is refused."""
        first = torch.full((len(states),), self.first_plane_mm, dtype=torch.float64)
        if start_planes_mm is not None and not torch.equal(
                torch.as_tensor(start_planes_mm, dtype=torch.float64), first):
            raise ValueError("the path network starts on its first plane, %r mm, "
                             "and nowhere else" % self.first_plane_mm)
        emitted, extra = self.forward(states)
        q, N = self.number_of_stages, self.number_of_steps
        ends = emitted[:, :, q]                                     # (n, N, 4)
        qop = states[:, 4:5, None].expand(-1, N, 1)
        inputs = torch.cat([torch.cat([states[:, None, :4], ends[:, :-1]], dim=1), qop], dim=2)
        planes = torch.as_tensor(self.layout.start_planes_mm)[None, :].expand(len(states), -1)
        track = PredictedTrack(
            step_length_mm=self.step_length_mm, tableau=self.tableau,
            start_planes_mm=planes, input_states=inputs,
            stage_states=emitted[:, :, :q], end_states=ends,
            end_state_was=END_STATE_PREDICTED, layout=self.layout,
            stage_rates=None, produced_by=self.kind)
        track.extra_outputs = extra      # read by the Gaussian path loss; None otherwise
        return track

    def whole_track(self, states, layout) -> PredictedTrack:
        """The same as predict: the network emits the whole track in one call."""
        if layout is not self.layout and (layout.number_of_steps != self.number_of_steps
                                          or layout.number_of_stages != self.number_of_stages):
            raise ValueError("the layout asked for is not the network's")
        with torch.no_grad():
            track = self.predict(torch.as_tensor(np.asarray(states, dtype=np.float64)))
        return track
