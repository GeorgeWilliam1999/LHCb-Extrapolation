"""The stage network: one network for one step length, used on every step.

It takes a state and the z of the plane the step starts on, and emits the
stage states of one Gauss-Legendre step of length dz:

    (x, y, tx, ty, q/p) on z_start,  z_start   ->   q stage states (x, y, tx, ty)

Stage k is the network's estimate of the state on the plane z_start + c_k dz.
Knowing z_start is what lets one network serve every step, because the field a
step crosses depends on where along the magnet it starts.

Settings, all arguments, none with a default:

  number_of_stages, step_length_mm
  first_start_plane_mm, last_start_plane_mm
        the range of planes a step can start on. z_start is mapped linearly
        onto [-1, 1] over it.
  scale_of_inputs     five constants; the state is divided by them
  output_form         how raw outputs become states; see output_forms.py
  width, depth        of the body: `depth` layers of `width` units, tanh
  end_state           "summed_from_the_stages": the network has 4q outputs and
                          the end of the step is collected and summed from the
                          stages (ruled by George, 2026-09-28);
                      "predicted": the network has 4(q + 1) outputs and the
                          last is the end state, as in the frozen code and in
                          equations 19 and 20 of the second mini-paper.
                      Both are kept because George ruled on 2026-09-28 that
                      this depends on the experiment.

The body, its initialisation and the order of its outputs are those of the
frozen `OneStepNetwork` (`_shared/model.py`), so that with a predicted end
state, the same seed and the same scales the two give identical numbers.
Gates: tests/test_networks.py.
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.integrators.gauss_legendre_tableau import GaussLegendreTableau
from rkpinn.networks.collect_and_sum import collect_and_sum, rates_at_the_stages
from rkpinn.predicted_track.predicted_track import (
    END_STATE_PREDICTED, END_STATE_SUMMED, WAYS_TO_FORM_THE_END_STATE, PredictedTrack)
from rkpinn.registry import register


def body_of_tanh_layers(number_of_inputs, number_of_outputs, width, depth):
    """`depth` hidden layers of `width` units with tanh, then a linear layer.
    In double precision whatever the default of torch is."""
    layers, n_in = [], int(number_of_inputs)
    for _ in range(int(depth)):
        layers += [torch.nn.Linear(n_in, int(width), dtype=torch.float64), torch.nn.Tanh()]
        n_in = int(width)
    layers += [torch.nn.Linear(n_in, int(number_of_outputs), dtype=torch.float64)]
    return torch.nn.Sequential(*layers)


def positive_scale(values, number, what):
    scale = np.asarray(values, dtype=np.float64).reshape(-1)
    if scale.shape != (number,):
        raise ValueError("%s has %d components" % (what, number))
    if not (np.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("%s must be positive and finite" % what)
    return torch.as_tensor(scale)


@register("network", "stage_network")
class StageNetwork(torch.nn.Module):
    kind = "stage_network"

    def __init__(self, *, equation_of_motion, tableau: GaussLegendreTableau,
                 step_length_mm, first_start_plane_mm, last_start_plane_mm,
                 scale_of_inputs, output_form, width, depth, end_state):
        super().__init__()
        if end_state not in WAYS_TO_FORM_THE_END_STATE:
            raise ValueError("the end state is %s, not %r"
                             % (" or ".join(WAYS_TO_FORM_THE_END_STATE), end_state))
        self.tableau = tableau
        self.number_of_stages = tableau.number_of_stages
        self.end_state = end_state
        self.number_of_outputs = self.number_of_stages + (end_state == END_STATE_PREDICTED)
        self.step_length_mm = float(step_length_mm)
        self.first_start_plane_mm = float(first_start_plane_mm)
        self.last_start_plane_mm = float(last_start_plane_mm)
        self.width, self.depth = int(width), int(depth)
        # Held in a tuple on purpose: as a submodule, the grid of the field map
        # would be written into every snapshot of the weights.
        self._equation_holder = (equation_of_motion,)
        self.output_form = output_form
        self.register_buffer("scale_of_inputs",
                             positive_scale(scale_of_inputs, 5, "the scale of the inputs"))
        self.register_buffer("nodes", torch.as_tensor(tableau.nodes))
        self.register_buffer("weights_of_the_tableau", torch.as_tensor(tableau.weights))
        nodes_of_outputs = (torch.cat([self.nodes, torch.ones(1, dtype=torch.float64)])
                            if end_state == END_STATE_PREDICTED else self.nodes.clone())
        self.register_buffer("nodes_of_outputs", nodes_of_outputs)
        self.register_buffer("middle_of_start_planes_mm", torch.tensor(
            0.5 * (self.first_start_plane_mm + self.last_start_plane_mm),
            dtype=torch.float64))
        self.register_buffer("half_range_of_start_planes_mm", torch.tensor(
            max(0.5 * (self.last_start_plane_mm - self.first_start_plane_mm), 1.0),
            dtype=torch.float64))
        self.body = body_of_tanh_layers(6, 4 * self.number_of_outputs, width, depth)

    @property
    def equation_of_motion(self):
        return self._equation_holder[0]

    def start_plane_as_input(self, start_planes_mm):
        """The start plane mapped linearly onto [-1, 1] over its range."""
        return ((start_planes_mm - self.middle_of_start_planes_mm)
                / self.half_range_of_start_planes_mm)

    def planes_of_outputs_mm(self, start_planes_mm):
        """z of the plane of every output, shape (n, number of outputs)."""
        return start_planes_mm[:, None] + self.nodes_of_outputs[None, :] * self.step_length_mm

    def raw(self, states, start_planes_mm):
        """The outputs of the body, shape (n, number of outputs, 4)."""
        inputs = torch.cat([states / self.scale_of_inputs,
                            self.start_plane_as_input(start_planes_mm)[:, None]], dim=1)
        return self.body(inputs).reshape(-1, self.number_of_outputs, 4)

    def forward(self, states, start_planes_mm):
        """The states the network emits, shape (n, number of outputs, 4): the
        stages, followed by the end state when it is predicted."""
        return self.output_form.states_from_raw(
            self.raw(states, start_planes_mm), states,
            self.planes_of_outputs_mm(start_planes_mm))

    def predict(self, states, start_planes_mm) -> PredictedTrack:
        """One step from every state on its own start plane: separate steps."""
        emitted = self.forward(states, start_planes_mm)
        q = self.number_of_stages
        stage_states = emitted[:, :q]
        stage_planes = start_planes_mm[:, None] + self.nodes[None, :] * self.step_length_mm
        stage_rates = rates_at_the_stages(
            self.equation_of_motion, states, stage_states, stage_planes)
        if self.end_state == END_STATE_SUMMED:
            end_states = collect_and_sum(states, stage_rates, self.step_length_mm,
                                         self.weights_of_the_tableau)
        else:
            end_states = emitted[:, q]
        return PredictedTrack(
            step_length_mm=self.step_length_mm, tableau=self.tableau,
            start_planes_mm=start_planes_mm[:, None],
            input_states=states[:, None, :],
            stage_states=stage_states[:, None],
            end_states=end_states[:, None],
            end_state_was=self.end_state, layout=None,
            stage_rates=stage_rates[:, None], produced_by=self.kind)

    def whole_track(self, states, layout):
        """The network applied to itself over the layout; see self_chain.py."""
        from rkpinn.networks.self_chain import whole_track
        return whole_track(self, states, layout)

    def settings(self) -> dict:
        """What is needed, with the weights, to build this network again."""
        return {
            "kind": self.kind, "number_of_stages": self.number_of_stages,
            "step_length_mm": self.step_length_mm,
            "first_start_plane_mm": self.first_start_plane_mm,
            "last_start_plane_mm": self.last_start_plane_mm,
            "scale_of_inputs": self.scale_of_inputs.tolist(),
            "output_form": self.output_form.name,
            "scale_of_outputs": self.output_form.scale_of_outputs.tolist(),
            "width": self.width, "depth": self.depth, "end_state": self.end_state,
        }


def build_stage_network(*, seed, **settings) -> StageNetwork:
    """A stage network with its weights drawn from the seed."""
    torch.manual_seed(int(seed))
    return StageNetwork(**settings)
