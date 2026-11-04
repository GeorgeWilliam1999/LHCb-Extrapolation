"""The path network: one call emits the whole crossing of a track.

It takes the state on the first plane, (x, y, tx, ty, q/p), and emits every
stage state and every plane state of the layout at once: N steps x (q + 1)
states x 4 components, 768 numbers at 64 steps and 2 stages. It is not
chained and it contains no equation of motion. It is trained on the reference
path (targets/reference_path.py), so it is supervised; the comparison with the
chain says this plainly.

Its output form is applied to every emitted state with the start state as
the input state and the whole region as one step: the form is bound with the
length of the region as its step length and, as the node of each output, the
distance of its plane from the first plane over that length. With the
straight-line form the network therefore learns the bend from the straight
line through the whole region, and the per-track scale is measured over the
whole region (16 samples of the field along the straight line, 324 mm apart at
64 steps), not over one step of the chain.

It fills a PredictedTrack with the layout set, so the standard report scores
it like a chain, and `evaluation/newton_sweep.py` corrects it with one Newton
step on the stage equations of every step at once.

`has_stages` is true, because the track it fills has stages; it trains on the
start states of the tracks, not on drawn states, which it declares with
`trains_on_start_states` (training_protocols.py, configuration.py).

With `extra_outputs` = 4 the body emits four more numbers per state, read by
the Gaussian path loss as the logarithm of a sigma per component. With 0 it
emits the states only.

Ported onto the stbc checkout on 2026-11-04. Gates: tests/test_path_network.py.
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.networks.stage_network import body_of_layers, check_activation, positive_scale
from rkpinn.predicted_track.predicted_track import END_STATE_PREDICTED, PredictedTrack
from rkpinn.registry import register

EXTRA_OUTPUTS = (0, 4)


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
        """The network a configuration names, with its weights drawn from the
        seed of the run."""
        from rkpinn.registry import component
        output_form = component("output_form", settings["output_form"]) \
            .from_configuration(settings["scale_of_outputs"], context)
        torch.manual_seed(int(context.seed))
        return cls(equation_of_motion=context.equation_of_motion, layout=context.layout,
                   scale_of_inputs=context.scale(settings["scale_of_inputs"], 5),
                   output_form=output_form, width=settings["width"], depth=settings["depth"],
                   activation=settings["activation"], extra_outputs=settings["extra_outputs"])

    def __init__(self, *, equation_of_motion, layout, scale_of_inputs, output_form, width,
                 depth, activation, extra_outputs):
        super().__init__()
        if layout.tableau is None:
            raise ValueError("the path network needs a layout with stages")
        self.layout = layout
        self.tableau = layout.tableau
        self.number_of_steps = int(layout.number_of_steps)
        self.number_of_stages = int(layout.number_of_stages)
        self.step_length_mm = float(layout.step_length_mm)
        self.first_plane_mm = float(layout.first_plane_mm)
        self.width, self.depth = int(width), int(depth)
        self.activation = check_activation(activation)
        self.extra_outputs = int(extra_outputs)
        if self.extra_outputs not in EXTRA_OUTPUTS:
            raise ValueError("extra_outputs is 0 (states only) or 4 (a sigma per component)")
        self.register_buffer("scale_of_inputs",
                             positive_scale(scale_of_inputs, 5, "the scale of the inputs"))
        # the plane of every emitted state: for step n its stages, then its end plane
        planes = np.empty((self.number_of_steps, self.number_of_stages + 1))
        for n in range(self.number_of_steps):
            planes[n, :self.number_of_stages] = layout.stage_planes_mm(n)
            planes[n, self.number_of_stages] = layout.planes_mm[n + 1]
        self.states_emitted = self.number_of_steps * (self.number_of_stages + 1)
        self.output_form = output_form
        # the whole region is one step of the form; each output at its fraction of it
        output_form.bind(step_length_mm=layout.length_mm,
                         nodes_of_outputs=(planes.reshape(-1) - self.first_plane_mm)
                         / layout.length_mm,
                         field=equation_of_motion.field)
        self.body = body_of_layers(5, self.states_emitted * (4 + self.extra_outputs),
                                   width, depth, activation)

    def settings(self) -> dict:
        return {"kind": self.kind, "width": self.width, "depth": self.depth,
                "activation": self.activation, "extra_outputs": self.extra_outputs,
                "number_of_steps": self.number_of_steps,
                "number_of_stages": self.number_of_stages,
                "scale_of_inputs": self.scale_of_inputs.tolist(),
                "output_form": self.output_form.name,
                "scale_of_outputs": self.output_form.scale_setting()}

    def forward(self, states):
        """The emitted states, shape (n, steps, stages + 1, 4), and the extra
        outputs, shape (n, steps, stages + 1, extra) or None."""
        raw = self.body(states / self.scale_of_inputs).reshape(
            -1, self.states_emitted, 4 + self.extra_outputs)
        first = self._first_planes(states)
        emitted = self.output_form.states_from_raw(raw[..., :4], states, first)
        shape = (len(states), self.number_of_steps, self.number_of_stages + 1)
        extra = raw[..., 4:].reshape(shape + (self.extra_outputs,)) if self.extra_outputs else None
        return emitted.reshape(shape + (4,)), extra

    def _first_planes(self, states):
        """The first plane for every row. Kept with the states it was made for,
        so that the per-track scale of the form, cached on the identity of its
        inputs, is measured once per tensor of states and not once per call."""
        held = getattr(self, "_held_first_planes", None)
        if held is not None and held[0] is states:
            return held[1]
        first = torch.full((len(states),), self.first_plane_mm, dtype=torch.float64)
        self._held_first_planes = (states, first)
        return first

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
        qop = states[:, None, 4:5].expand(-1, N, 1)
        inputs = torch.cat([torch.cat([states[:, None, :4], ends[:, :-1]], dim=1), qop], dim=2)
        planes = torch.as_tensor(self.layout.start_planes_mm)[None, :].expand(len(states), -1)
        return PredictedTrack(
            step_length_mm=self.step_length_mm, tableau=self.tableau,
            start_planes_mm=planes, input_states=inputs,
            stage_states=emitted[:, :, :q], end_states=ends,
            end_state_was=END_STATE_PREDICTED, layout=self.layout,
            produced_by=self.kind, extra_outputs=extra)

    def whole_track(self, states, layout=None) -> PredictedTrack:
        """The same as `predict`, without a gradient, from numpy states, as numpy."""
        if layout is not None and (layout.number_of_steps != self.number_of_steps
                                   or layout.number_of_stages != self.number_of_stages):
            raise ValueError("the layout asked for is not the network's")
        with torch.no_grad():
            track = self.predict(torch.as_tensor(np.asarray(states, dtype=np.float64)))
        for name in ("start_planes_mm", "input_states", "stage_states", "end_states",
                     "extra_outputs"):
            value = getattr(track, name)
            if value is not None:
                setattr(track, name, value.numpy().copy())
        return track
