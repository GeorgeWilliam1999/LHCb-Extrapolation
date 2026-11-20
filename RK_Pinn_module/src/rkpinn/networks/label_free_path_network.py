"""The label-free path network: one call emits every stage state of the crossing.

It takes the state on the first plane, (x, y, tx, ty, q/p), and emits the q
stage states of every one of the N steps: N x q states x 4 components, 512
numbers at 64 steps and 2 stages. It emits no plane state. The plane states
are summed from the stages with the weights b of the tableau:

    s_0     = the start state
    F_ni    = f(S_ni, q/p, z_n + c_i h)          i = 1..q
    s_{n+1} = s_n + h sum_i b_i F_ni

where S_ni is the emitted stage state i of step n, z_n the start plane of the
step, c the nodes of the tableau, h the step length and f the rates of the
equation of motion. Every stage is known before the sum, so all N q rates are
computed in one batched call, and a cumulative sum over the steps gives the
plane states. The network is trained without labels, on the stage residual of
every step with the chaining inside the loss (losses/stage_residual_of_a_path.py).

Its output form is bound only at the stage planes, with the start state as the
input state and the whole region as one step, as in the path network
(path_network.py): the node of each output is (stage plane - first plane) /
length of the region. With the straight-line form the network learns the
bend from the straight line through the whole region.

The settings are those of the path network without `extra_outputs`: this
network emits states only.

It fills a PredictedTrack with the layout set, the end states summed, and the
rates at the stages kept, so the loss does not compute them again.

Added 2026-11-20 for the experiment can_a_label_free_path_network_reach_the_exact_scheme
(approved by George the same day). path_network.py is unchanged, so no run made
before changes its key. Gates: tests/test_label_free_path.py.
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.networks.path_network import PathNetwork
from rkpinn.networks.stage_network import body_of_layers, check_activation, positive_scale
from rkpinn.predicted_track.predicted_track import END_STATE_SUMMED, PredictedTrack
from rkpinn.registry import register


@register("network", "label_free_path_network")
class LabelFreePathNetwork(PathNetwork):
    kind = "label_free_path_network"
    has_stages = True
    trains_on_start_states = True
    settings_in_a_configuration = (
        "output_form", "width", "depth", "activation", "scale_of_inputs", "scale_of_outputs")

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
                   activation=settings["activation"])

    def __init__(self, *, equation_of_motion, layout, scale_of_inputs, output_form, width,
                 depth, activation):
        torch.nn.Module.__init__(self)
        if layout.tableau is None:
            raise ValueError("the label-free path network needs a layout with stages")
        self.layout = layout
        self.tableau = layout.tableau
        # held outside the module tree, so the field map is not in the state dict
        self._equation_holder = (equation_of_motion,)
        self.number_of_steps = int(layout.number_of_steps)
        self.number_of_stages = int(layout.number_of_stages)
        self.step_length_mm = float(layout.step_length_mm)
        self.first_plane_mm = float(layout.first_plane_mm)
        self.width, self.depth = int(width), int(depth)
        self.activation = check_activation(activation)
        self.extra_outputs = 0
        self.register_buffer("scale_of_inputs",
                             positive_scale(scale_of_inputs, 5, "the scale of the inputs"))
        # the plane of every emitted state: the stages of step n, n = 0..N-1
        planes = np.stack([layout.stage_planes_mm(n) for n in range(self.number_of_steps)])
        self.register_buffer("stage_planes_mm",
                             torch.as_tensor(planes.reshape(-1), dtype=torch.float64),
                             persistent=False)
        self.register_buffer("weights_of_the_tableau",
                             torch.as_tensor(np.asarray(self.tableau.weights, dtype=np.float64)),
                             persistent=False)
        self.states_emitted = self.number_of_steps * self.number_of_stages
        self.output_form = output_form
        # the whole region is one step of the form; each output at its fraction of it
        output_form.bind(step_length_mm=layout.length_mm,
                         nodes_of_outputs=(planes.reshape(-1) - self.first_plane_mm)
                         / layout.length_mm,
                         field=equation_of_motion.field)
        self.body = body_of_layers(5, self.states_emitted * 4, width, depth, activation)

    @property
    def equation_of_motion(self):
        return self._equation_holder[0]

    def settings(self) -> dict:
        return {"kind": self.kind, "width": self.width, "depth": self.depth,
                "activation": self.activation,
                "number_of_steps": self.number_of_steps,
                "number_of_stages": self.number_of_stages,
                "scale_of_inputs": self.scale_of_inputs.tolist(),
                "output_form": self.output_form.name,
                "scale_of_outputs": self.output_form.scale_setting()}

    def forward(self, states):
        """The emitted stage states, shape (n, steps, stages, 4)."""
        raw = self.body(states / self.scale_of_inputs).reshape(-1, self.states_emitted, 4)
        emitted = self.output_form.states_from_raw(raw, states, self._first_planes(states))
        return emitted.reshape(len(states), self.number_of_steps, self.number_of_stages, 4)

    def predict(self, states, start_planes_mm=None) -> PredictedTrack:
        """The whole crossing from every start state: the stages emitted, the
        plane states summed from them. Every row starts on the first plane;
        another start plane is refused."""
        first = torch.full((len(states),), self.first_plane_mm, dtype=torch.float64)
        if start_planes_mm is not None and not torch.equal(
                torch.as_tensor(start_planes_mm, dtype=torch.float64), first):
            raise ValueError("the label-free path network starts on its first plane, "
                             "%r mm, and nowhere else" % self.first_plane_mm)
        stages = self.forward(states)                                   # (n, N, q, 4)
        n, N, q = len(states), self.number_of_steps, self.number_of_stages
        qop = states[:, 4:5]
        # all N q rates in one call
        rates = self.equation_of_motion.rates(
            stages.reshape(n, N * q, 4), qop, self.stage_planes_mm).reshape(n, N, q, 4)
        increments = self.step_length_mm * torch.einsum(
            "i,bnid->bnd", self.weights_of_the_tableau, rates)          # (n, N, 4)
        ends = states[:, None, :4] + torch.cumsum(increments, dim=1)
        inputs = torch.cat([torch.cat([states[:, None, :4], ends[:, :-1]], dim=1),
                            qop[:, None, :].expand(-1, N, 1)], dim=2)
        planes = torch.as_tensor(self.layout.start_planes_mm)[None, :].expand(n, -1)
        return PredictedTrack(
            step_length_mm=self.step_length_mm, tableau=self.tableau,
            start_planes_mm=planes, input_states=inputs, stage_states=stages,
            end_states=ends, end_state_was=END_STATE_SUMMED, layout=self.layout,
            stage_rates=rates, produced_by=self.kind)

    def whole_track(self, states, layout=None) -> PredictedTrack:
        """The same as `predict`, without a gradient, from numpy states, as numpy."""
        if layout is not None and (layout.number_of_steps != self.number_of_steps
                                   or layout.number_of_stages != self.number_of_stages):
            raise ValueError("the layout asked for is not the network's")
        with torch.no_grad():
            track = self.predict(torch.as_tensor(np.asarray(states, dtype=np.float64)))
        for name in ("start_planes_mm", "input_states", "stage_states", "end_states",
                     "stage_rates"):
            setattr(track, name, getattr(track, name).numpy().copy())
        return track
