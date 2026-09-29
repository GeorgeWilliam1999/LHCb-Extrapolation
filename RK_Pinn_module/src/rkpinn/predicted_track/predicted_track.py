"""The predicted track: for each step, its input state, its stage states and
its end state. It says nothing about how those states were produced.

Every network fills it. So do the exact scheme and the reference integrator,
so that a comparator is scored by the same code as a network. Every loss and
every evaluation reads it, and reads nothing else of a network.

Two uses, one structure:

  a whole track    every row is a particle, and its steps follow one another
                   from the first plane to the last; `layout` is given
  separate steps   every row is one step on its own start plane, as drawn for
                   a round of training; `layout` is None and each row has one step

The arrays are torch tensors while training, so that a gradient flows through
them, and numpy arrays otherwise. The structure does not care which.

Shapes, with n rows, S steps per row and q stages:

    start_planes_mm   (n, S)
    input_states      (n, S, 5)     x, y, slope in x, slope in y, charge over momentum
    stage_states      (n, S, q, 4)  or None when there are no stages
    end_states        (n, S, 4)
    stage_rates       (n, S, q, 4)  or None: the rates at the stage states, kept
                                    when whoever filled the track had to compute them

Charge over momentum is conserved. It is in the input states and is carried
unchanged, so the stage and end states hold four components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from rkpinn.integrators import exact_collocation
from rkpinn.integrators.gauss_legendre_tableau import GaussLegendreTableau
from rkpinn.integrators.runge_kutta_sixth_order import integrate_with_sixth_order
from rkpinn.predicted_track.track_layout import TrackLayout

END_STATE_SUMMED = "summed_from_the_stages"
END_STATE_PREDICTED = "predicted"
WAYS_TO_FORM_THE_END_STATE = (END_STATE_SUMMED, END_STATE_PREDICTED)


@dataclass
class PredictedTrack:
    step_length_mm: float
    tableau: Optional[GaussLegendreTableau]
    start_planes_mm: Any
    input_states: Any
    stage_states: Any
    end_states: Any
    end_state_was: str
    layout: Optional[TrackLayout] = None
    stage_rates: Any = None
    produced_by: str = ""

    def __post_init__(self):
        if self.end_state_was not in WAYS_TO_FORM_THE_END_STATE:
            raise ValueError("an end state is %s, not %r"
                             % (" or ".join(WAYS_TO_FORM_THE_END_STATE), self.end_state_was))
        n, steps = self.input_states.shape[:2]
        if tuple(self.input_states.shape) != (n, steps, 5):
            raise ValueError("input states have the shape (rows, steps, 5)")
        if tuple(self.end_states.shape) != (n, steps, 4):
            raise ValueError("end states have the shape (rows, steps, 4)")
        if tuple(self.start_planes_mm.shape) != (n, steps):
            raise ValueError("start planes have the shape (rows, steps)")
        q = self.number_of_stages
        if q == 0:
            if self.stage_states is not None:
                raise ValueError("a track with no tableau has no stage states")
            if self.end_state_was == END_STATE_SUMMED:
                raise ValueError("an end state cannot be summed from stages that do not exist")
        elif tuple(self.stage_states.shape) != (n, steps, q, 4):
            raise ValueError("stage states have the shape (rows, steps, %d, 4)" % q)
        if self.layout is not None and self.layout.number_of_steps != steps:
            raise ValueError("the layout has %d steps and the track %d"
                             % (self.layout.number_of_steps, steps))

    # -- sizes -------------------------------------------------------------------
    @property
    def number_of_rows(self) -> int:
        return self.input_states.shape[0]

    @property
    def number_of_steps(self) -> int:
        return self.input_states.shape[1]

    @property
    def number_of_stages(self) -> int:
        return 0 if self.tableau is None else self.tableau.number_of_stages

    @property
    def is_a_whole_track(self) -> bool:
        return self.layout is not None

    # -- one step ----------------------------------------------------------------
    def input_state(self, step: int):
        """The state the step starts from, shape (n, 5)."""
        return self.input_states[:, step]

    def stage_states_of(self, step: int):
        """The stage states of the step, shape (n, q, 4); None without stages."""
        return None if self.stage_states is None else self.stage_states[:, step]

    def end_state(self, step: int):
        """The state at the end of the step, shape (n, 5), charge over momentum carried."""
        return _with_charge_over_momentum(self.end_states[:, step], self.input_states[:, step])

    def final_state(self):
        """The state at the end of the last step, shape (n, 5)."""
        return self.end_state(self.number_of_steps - 1)

    def stage_planes_mm(self):
        """z of every stage plane, shape (n, S, q)."""
        if self.tableau is None:
            return None
        nodes = _like(self.tableau.nodes, self.start_planes_mm)
        return self.start_planes_mm[..., None] + nodes * self.step_length_mm

    # -- the whole track ----------------------------------------------------------
    def states_on_planes(self):
        """The state on the first plane and at the end of every step, as numpy,
        shape (n, S + 1, 5). Only a whole track has this."""
        if not self.is_a_whole_track:
            raise ValueError("separate steps do not form a track")
        inputs, ends = _as_numpy(self.input_states), _as_numpy(self.end_states)
        n, steps = inputs.shape[:2]
        states = np.empty((n, steps + 1, 5))
        states[:, 0] = inputs[:, 0]
        states[:, 1:, :4] = ends
        states[:, 1:, 4] = inputs[:, :, 4]
        return states


def _as_numpy(values):
    return values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)


def _like(values, other):
    """`values` as the kind of array `other` is."""
    if hasattr(other, "detach"):
        import torch
        return torch.as_tensor(np.asarray(values), dtype=other.dtype, device=other.device)
    return np.asarray(values)


def _with_charge_over_momentum(state_of_four, input_state):
    if hasattr(state_of_four, "detach"):
        import torch
        return torch.cat([state_of_four, input_state[:, 4:5]], dim=1)
    return np.concatenate([state_of_four, input_state[:, 4:5]], axis=1)


# ---------------------------------------------------------------------------------
# The comparators fill the same structure.

def track_of_the_exact_scheme(equation_of_motion, start_states,
                              layout: TrackLayout) -> PredictedTrack:
    """The exact collocation scheme chained over the layout, from the start
    states on the first plane. What a network with a loss of zero would give."""
    if layout.tableau is None:
        raise ValueError("the exact scheme needs a tableau")
    solved = exact_collocation.solve_whole_track(
        equation_of_motion, start_states, layout.first_plane_mm, layout.length_mm,
        layout.number_of_steps, layout.tableau)
    if solved.solves_not_converged:
        raise ArithmeticError("%d solves of the exact scheme did not converge"
                              % solved.solves_not_converged)
    n = len(solved.states)
    return PredictedTrack(
        step_length_mm=layout.step_length_mm, tableau=layout.tableau,
        start_planes_mm=np.broadcast_to(layout.start_planes_mm,
                                        (n, layout.number_of_steps)).copy(),
        input_states=solved.states[:, :-1].copy(),
        stage_states=solved.stage_states[..., :4].copy(),
        end_states=solved.states[:, 1:, :4].copy(),
        end_state_was=END_STATE_SUMMED, layout=layout,
        produced_by="the exact collocation scheme")


def steps_of_the_exact_scheme(equation_of_motion, input_states, start_planes_mm,
                              step_length_mm, tableau) -> PredictedTrack:
    """One exact step from each input state on its own start plane: separate steps."""
    input_states = np.asarray(input_states, dtype=np.float64)
    start_planes_mm = np.broadcast_to(
        np.asarray(start_planes_mm, dtype=np.float64), (len(input_states),))
    q = tableau.number_of_stages
    stages = np.empty((len(input_states), 1, q, 4))
    ends = np.empty((len(input_states), 1, 4))
    for i, (state, plane) in enumerate(zip(input_states, start_planes_mm)):
        step = exact_collocation.solve_one_step(
            equation_of_motion, state.copy(), float(plane), step_length_mm, tableau)
        if not step.converged:
            raise ArithmeticError("a solve of the exact scheme did not converge")
        stages[i, 0] = step.stage_states[:, :4]
        ends[i, 0] = step.end_state[:4]
    return PredictedTrack(
        step_length_mm=step_length_mm, tableau=tableau,
        start_planes_mm=start_planes_mm[:, None].copy(),
        input_states=input_states[:, None, :].copy(), stage_states=stages,
        end_states=ends, end_state_was=END_STATE_SUMMED, layout=None,
        produced_by="the exact collocation scheme")


def track_of_the_reference(equation_of_motion, start_states, layout: TrackLayout, *,
                           step_length_of_the_integrator_mm) -> PredictedTrack:
    """The sixth-order method carried over the planes of the layout. It has no
    stages of the layout's tableau, so the track holds none."""
    planes = layout.planes_mm
    current = np.atleast_2d(np.asarray(start_states, dtype=np.float64))
    n = len(current)
    inputs = np.empty((n, layout.number_of_steps, 5))
    ends = np.empty((n, layout.number_of_steps, 4))
    for k in range(layout.number_of_steps):
        inputs[:, k] = current
        current = integrate_with_sixth_order(
            equation_of_motion, current, planes[k], planes[k + 1],
            step_length_mm=step_length_of_the_integrator_mm)
        ends[:, k] = current[:, :4]
    without_stages = TrackLayout(layout.first_plane_mm, layout.last_plane_mm,
                                 layout.number_of_steps, None)
    return PredictedTrack(
        step_length_mm=layout.step_length_mm, tableau=None,
        start_planes_mm=np.broadcast_to(layout.start_planes_mm,
                                        (n, layout.number_of_steps)).copy(),
        input_states=inputs, stage_states=None, end_states=ends,
        end_state_was=END_STATE_PREDICTED, layout=without_stages,
        produced_by="the sixth-order reference")
