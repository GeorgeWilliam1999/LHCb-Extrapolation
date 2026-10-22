"""Output forms: how the raw outputs of a network become states.

The body of a network ends in a linear layer, whose outputs are of order one.
An output form turns them into states in millimetres and slopes. The form is
a setting of a run, named in its configuration, and the scale of the outputs
is a second setting that the form reads.

Two forms are built:

  direct_states
        state = scale of the component * raw output.
        The network emits the states themselves, as in Raissi, Perdikaris
        and Karniadakis (2019). Ruled by George on 2026-09-28. The scale is
        four fixed numbers, one per component (x, y, slope in x, slope in y);
        how they are measured is a setting of the experiment.

  straight_line_plus_scaled_correction
        state_j = straight-line extrapolation of the input to the plane of
                  output j  +  scale * raw output j.
        The network emits the correction to the straight line: the bending
        of the magnet over one step. With the last layer zero the network is
        the straight line. This is the form of every network of the second
        mini-paper, `E1_Network_grid/chain_network.py::ChainNetwork` (chosen
        by George on 2026-09-14), ported here on 2026-10-22 with the
        arithmetic unchanged, so that a stage network with this form and a
        predicted end state is identical to that network to the last bit.
        The scale is one of
          per_track   the scale of the frozen code: for each track, from its
                      input state and the field along the straight line
                      through it (16-point midpoint rule over the step),
                          base_x = kappa |q/p| INT |B| dz
                          base_y = max(kappa |q/p| I_y, 1e-3 base_x)
                          I_y    = INT sqrt(1 + tx^2 + ty^2)
                                   ((1 + ty^2)|Bx| + |tx ty By| + |tx Bz|) dz
                          x, y:   max(base |dz| / 2, 1e-9 mm)
                          tx, ty: max(base, 1e-12)
                      I_y bounds the y-rate of the equation of motion term by
                      term, so it does not vanish by cancellation;
          median_of_per_track_scale_over_first_round_states
                      four fixed numbers: the median of the per-track scale
                      over the first round's states, measured once when the
                      run is made and recorded in its constants;
          four numbers, given outright.

Why the second form was added (2026-10-22). The package runs of October 2026
were worse at the stages than the frozen networks by a factor 6 to 25, and
at the end of the region by a factor 3 to 37, for one reason: with direct
states a raw output of zero is the origin of the detector, and the network
has to reproduce x + tx dz through two tanh layers to two parts in a million
before the bending is visible. With the straight line a raw output of zero
is already within 0.1 mm of the truth. Isolated on the login node with the
same trainer, loss, states, seed and body, the straight line lowers the loss
190 to 390 times and the single-step error 12 to 15 times in the same number
of restarts (correction plan of 2026-10-20, section 4).

Every output form declares what it returns when the network's last layer is
zero, and a gate checks it. A form belongs to one network: the network binds
it to its step length, the planes of its outputs and its field map, once, when
the network is built. The field map is held outside the module's buffers, so
that it is not written into a snapshot of the weights.
Gates: tests/test_networks.py.
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.equation_of_motion.lhcb import KAPPA
from rkpinn.registry import register

PER_TRACK = "per_track"
MEDIAN_OF_PER_TRACK_SCALE = "median_of_per_track_scale_over_first_round_states"

# The constants of the frozen `ChainNetwork` (chain_network.py, lines 86 to 89).
SAMPLES_OF_THE_FIELD_INTEGRAL = 16
Y_FLOOR_RELATIVE = 1e-3
POSITION_FLOOR_MM = 1e-9
SLOPE_FLOOR = 1e-12


def four_positive_numbers(values, what="the scale of the outputs"):
    scale = np.asarray(values, dtype=np.float64).reshape(-1)
    if scale.shape != (4,):
        raise ValueError("%s has four components: x, y, slope in x, slope in y" % what)
    if not (np.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("%s must be positive and finite" % what)
    return torch.as_tensor(scale)


def per_track_scale(states, start_planes_mm, step_length_mm, field,
                    samples=SAMPLES_OF_THE_FIELD_INTEGRAL, y_floor_relative=Y_FLOOR_RELATIVE):
    """The scale of the correction of each track, shape (n, 4), from its input
    state (n, 5), the plane its step starts on (n,), the step length and the
    differentiable field map. The arithmetic of `ChainNetwork.field_integrals`
    and `ChainNetwork.residual_scale`, in the same order."""
    with torch.no_grad():
        dz = torch.tensor(float(step_length_mm), dtype=torch.float64)
        u = (torch.arange(int(samples), dtype=torch.float64) + 0.5) / int(samples)
        s = u[None, :] * dz
        x = states[:, 0:1] + states[:, 2:3] * s
        y = states[:, 1:2] + states[:, 3:4] * s
        z = start_planes_mm[:, None] + s
        Bx, By, Bz = field(x, y, z)
        tx, ty = states[:, 2:3], states[:, 3:4]
        norm = torch.sqrt(1.0 + tx * tx + ty * ty)
        I_B = torch.sqrt(Bx * Bx + By * By + Bz * Bz).mean(dim=1) * dz.abs()
        I_y = (norm * ((1.0 + ty * ty) * Bx.abs() + (tx * ty * By).abs()
                       + (tx * Bz).abs())).mean(dim=1) * dz.abs()
        qop = states[:, 4].abs()
        base_x = KAPPA * qop * I_B
        base_y = torch.maximum(KAPPA * qop * I_y, float(y_floor_relative) * base_x)
        half = dz.abs() / 2.0
        return torch.stack([(base_x * half).clamp_min(POSITION_FLOOR_MM),
                            (base_y * half).clamp_min(POSITION_FLOOR_MM),
                            base_x.clamp_min(SLOPE_FLOOR),
                            base_y.clamp_min(SLOPE_FLOOR)], dim=-1)


class OutputForm(torch.nn.Module):
    """What every output form gives. Not registered itself."""

    name = ""

    @classmethod
    def from_configuration(cls, scale_of_outputs, context):
        """The form a configuration names, from its `scale_of_outputs` setting
        and what the run has built (`context`)."""
        raise NotImplementedError

    def bind(self, *, step_length_mm, nodes_of_outputs, field):
        """Called once by the network that holds the form.

        step_length_mm    the length of a step
        nodes_of_outputs  the nodes c of the planes of the outputs, as fractions
                          of the step, shape (J,); 1 for a predicted end state
        field             the differentiable field map, or None for a network
                          with no equation of motion
        """
        self.step_length_mm = float(step_length_mm)
        nodes = torch.as_tensor(nodes_of_outputs, dtype=torch.float64).reshape(-1)
        # offsets of the planes of the outputs from the start plane: the
        # arithmetic of `ChainNetwork.straight`, cout * dz
        self.offsets_mm = nodes[None, :] * torch.tensor(self.step_length_mm, dtype=torch.float64)
        self.number_of_outputs = int(len(nodes))
        self._field_holder = (field,)

    @property
    def field(self):
        return self._field_holder[0]

    def states_from_raw(self, raw, input_states, start_planes_mm):
        """raw: shape (n, J, 4), the outputs of the body; input_states: shape
        (n, 5); start_planes_mm: shape (n,). Returns the states, shape (n, J, 4)."""
        raise NotImplementedError

    def answer_with_zeroed_network(self, input_states, start_planes_mm):
        """What the form returns when the last layer of the network is zero."""
        raise NotImplementedError

    def scale_setting(self):
        """What `settings()` of the network records as `scale_of_outputs`, and
        what builds the form again: four numbers, or the name `per_track`."""
        raise NotImplementedError

    def forward(self, raw, input_states, start_planes_mm):
        return self.states_from_raw(raw, input_states, start_planes_mm)


@register("output_form", "direct_states")
class DirectStates(OutputForm):
    """state = scale * raw output, component by component."""

    name = "direct_states"

    @classmethod
    def from_configuration(cls, scale_of_outputs, context):
        return cls(context.scale(scale_of_outputs, 4))

    def __init__(self, scale_of_outputs):
        super().__init__()
        self.register_buffer("scale_of_outputs", four_positive_numbers(scale_of_outputs))

    def states_from_raw(self, raw, input_states, start_planes_mm):
        """The input states and the start planes are not used by this form."""
        return raw * self.scale_of_outputs

    def answer_with_zeroed_network(self, input_states, start_planes_mm):
        """With the last layer zero the raw outputs are zero, and so is every state."""
        return torch.zeros((len(input_states), self.number_of_outputs, 4),
                           dtype=torch.float64)

    def scale_setting(self):
        return self.scale_of_outputs.tolist()


@register("output_form", "straight_line_plus_scaled_correction")
class StraightLinePlusScaledCorrection(OutputForm):
    """state_j = straight line from the input to the plane of output j
                 + scale * raw output j.

    scale_of_outputs  `per_track`, or four numbers. The median of the per-track
                      scale over the first round's states is resolved to four
                      numbers by `from_configuration`.
    """

    name = "straight_line_plus_scaled_correction"

    @classmethod
    def from_configuration(cls, scale_of_outputs, context):
        if scale_of_outputs == PER_TRACK:
            return cls(PER_TRACK)
        if scale_of_outputs == MEDIAN_OF_PER_TRACK_SCALE:
            scale = per_track_scale(
                torch.as_tensor(context.first_round_states, dtype=torch.float64),
                torch.as_tensor(context.first_round_start_planes_mm, dtype=torch.float64),
                context.layout.step_length_mm, context.equation_of_motion.field)
            return cls(np.median(scale.numpy(), axis=0).tolist())
        return cls(context.scale(scale_of_outputs, 4))

    def __init__(self, scale_of_outputs):
        super().__init__()
        if isinstance(scale_of_outputs, str):
            if scale_of_outputs != PER_TRACK:
                raise ValueError(
                    "the scale of the outputs of %s is %s, %s or four numbers, not %r"
                    % (self.name, PER_TRACK, MEDIAN_OF_PER_TRACK_SCALE, scale_of_outputs))
            self.scale_is_per_track = True
        else:
            self.scale_is_per_track = False
            self.register_buffer("scale_of_outputs", four_positive_numbers(scale_of_outputs))
        self._cache = None

    def bind(self, *, step_length_mm, nodes_of_outputs, field):
        if field is None:
            raise ValueError("the output form %s needs the field map of an equation of "
                             "motion, and this network has none" % self.name)
        super().bind(step_length_mm=step_length_mm, nodes_of_outputs=nodes_of_outputs,
                     field=field)

    def straight_line(self, input_states):
        """The straight-line extrapolation of every input to the planes of the
        outputs, shape (n, J, 4): the arithmetic of `ChainNetwork.straight`."""
        x = input_states[:, 0:1] + input_states[:, 2:3] * self.offsets_mm
        y = input_states[:, 1:2] + input_states[:, 3:4] * self.offsets_mm
        return torch.stack([x, y, input_states[:, 2:3].expand_as(x),
                            input_states[:, 3:4].expand_as(x)], dim=-1)

    def scale(self, input_states, start_planes_mm):
        """The scale of the correction, shape (n, 1, 4) per track or (4,)
        fixed. The per-track scale is cached on the identity of the inputs,
        because the loss calls the network hundreds of times a restart on the
        same tensors (as `ChainNetwork.residual_scale` does)."""
        if not self.scale_is_per_track:
            return self.scale_of_outputs
        cached = self._cache
        if cached is not None and cached[0] is input_states and cached[1] is start_planes_mm:
            return cached[2]
        scale = per_track_scale(input_states, start_planes_mm, self.step_length_mm,
                                self.field)[:, None, :]
        self._cache = (input_states, start_planes_mm, scale)
        return scale

    def states_from_raw(self, raw, input_states, start_planes_mm):
        return self.straight_line(input_states) + self.scale(input_states, start_planes_mm) * raw

    def answer_with_zeroed_network(self, input_states, start_planes_mm):
        """With the last layer zero the correction is zero: the straight line."""
        return self.straight_line(input_states)

    def scale_setting(self):
        return PER_TRACK if self.scale_is_per_track else self.scale_of_outputs.tolist()
