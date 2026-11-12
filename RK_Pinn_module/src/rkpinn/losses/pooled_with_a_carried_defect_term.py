"""The pooled loss with a carried-defect term: an estimate, without labels, of
the error a track has on the last plane, made small with the pooled loss.

    loss = pooled + lambda * carried

    pooled   the pooled loss (pooled.py): the mean of (r[n, j, d] / s[d])^2
    carried  the mean over the tracks t and the components x and y of

                 ( sum over the steps n of track t of  c[n] / s )^2

The defect of a step and its carry, for a step from S0 on z0 with the stages
Y and the rates F = f(Y):

    swept stages      Y' = S0 + dz A F  =  Y - r        one fixed-point sweep
    rates again       F' = f(Y')                         q field lookups
    defect            delta = dz sum over k of b_k (F_k - F'_k)       (x, y, tx, ty)
    carried defect    c_x = delta_x + delta_tx (z_last - z0 - dz),  c_y the same in y

delta is the end state summed from the network's stages minus the end state
summed from the swept stages. It estimates the network's end state minus the
exact scheme's, because the stage error is the stage residual to a few per
cent (from 32 steps on) and the map from the stage error to the end state is
linear. The carry is a straight line from the end of the step to the last
plane; it reproduces the exact carry with a correlation above 0.98. Both are
measured in experiments/how_does_the_stage_residual_become_the_endpoint_error.

Why: the defects of consecutive steps have one sign over long parts of a
track, so they add up. The pooled loss does not see the sign. This term is the
squared sum, so it is small only when the defects of a track cancel or vanish.

Tracks. The term needs every step of a track. The states of a round are whole
tracks when their number is a multiple of the number of steps and their start
planes repeat the start planes of the layout, track after track (the protocol
`rounds_on_whole_tracks_of_own_predictions` gives them so). Then the sum runs
over the steps of each track. Otherwise each state is taken as a track of one
step: the term is then the mean of (c / s)^2 over the states. This is so for
the check at the exact scheme's stages, where the term is zero either way.

lambda is fixed once, when the run is made, so that the term is the share
`share_of_the_carried_term_at_round_one` of the whole loss of the first
round's states under the network as it is drawn from the seed, as in
pooled_with_a_bias_term.py. With the share 0 the loss is the pooled loss to
the last bit.

The gradient flows through F and F' to the weights. The input states are
those of the round and carry no gradient, as in every loss of the package.

`squared_terms` gives the pooled terms alone, as in pooled_with_a_bias_term.py.
`pooled.py` is unchanged, so no run made before changes its key.
Gates: tests/test_physics_layer_and_carried_defect.py.
"""
from __future__ import annotations

import torch

from rkpinn.losses.pooled import Pooled
from rkpinn.losses.stage_residual import stage_residual
from rkpinn.registry import register


def defect_of_each_step(residual, track, equation_of_motion):
    """delta, shape (m, 4): the end state summed from the stages of the track
    minus the end state summed from its stages after one fixed-point sweep."""
    q = track.number_of_stages
    stages = torch.as_tensor(track.stage_states).reshape(-1, q, 4)
    c = torch.as_tensor(track.tableau.nodes)
    b = torch.as_tensor(track.tableau.weights)
    dz = track.step_length_mm
    planes = residual.start_planes_mm[:, None] + c[None, :] * dz
    if track.stage_rates is not None:
        rates = torch.as_tensor(track.stage_rates).reshape(-1, q, 4)
    else:       # a track with no rates kept, such as that of the exact scheme
        rates = equation_of_motion.rates(stages, residual.input_states[:, 4:5], planes)
    swept = stages - residual.residual[:, :q]
    swept_rates = equation_of_motion.rates(swept, residual.input_states[:, 4:5], planes)
    return dz * torch.einsum("k,nkd->nd", b, rates - swept_rates)


def carried_to_the_last_plane(defect, start_planes_mm, step_length_mm, last_plane_mm):
    """(c_x, c_y), shape (m, 2): the defect carried on a straight line from the
    end of its step to the last plane."""
    lever = (last_plane_mm - start_planes_mm - step_length_mm)[:, None]
    return defect[:, 0:2] + defect[:, 2:4] * lever


def are_whole_tracks(start_planes_mm, start_planes_of_the_layout_mm) -> bool:
    """True when the states are whole tracks: the start planes of the layout,
    in order, repeated track after track."""
    planes = torch.as_tensor(start_planes_of_the_layout_mm, dtype=torch.float64)
    N = len(planes)
    if len(start_planes_mm) == 0 or len(start_planes_mm) % N:
        return False
    return bool(torch.allclose(start_planes_mm.reshape(-1, N), planes[None, :],
                               rtol=0.0, atol=1e-6))


@register("loss", "pooled_with_a_carried_defect_term")
class PooledWithACarriedDefectTerm(Pooled):
    name = "pooled_with_a_carried_defect_term"
    needs_labels = False
    settings_in_a_configuration = ("terms", "share_of_the_carried_term_at_round_one")

    @classmethod
    def from_configuration(cls, settings, context):
        layout = context.layout
        return cls(equation_of_motion=context.equation_of_motion, terms=settings["terms"],
                   share_of_the_carried_term_at_round_one=settings[
                       "share_of_the_carried_term_at_round_one"],
                   start_planes_mm=[float(z) for z in layout.start_planes_mm],
                   last_plane_mm=float(layout.last_plane_mm))

    def __init__(self, *, equation_of_motion, terms, share_of_the_carried_term_at_round_one,
                 start_planes_mm, last_plane_mm):
        super().__init__(equation_of_motion=equation_of_motion, terms=terms)
        share = float(share_of_the_carried_term_at_round_one)
        if not 0.0 <= share < 1.0:
            raise ValueError("the share of the carried term is in [0, 1), not %r" % share)
        self.share = share
        self.start_planes_mm = [float(z) for z in start_planes_mm]
        self.last_plane_mm = float(last_plane_mm)

    def parts(self, predicted_track, constants):
        """(pooled, carried) of a predicted track of separate steps."""
        residual = stage_residual(predicted_track, self.equation_of_motion, constants["terms"])
        divisor = torch.as_tensor(constants["divisor_per_component"], dtype=torch.float64)
        pooled = ((residual.residual / divisor) ** 2).mean()
        defect = defect_of_each_step(residual, predicted_track, self.equation_of_motion)
        carried = carried_to_the_last_plane(
            defect, residual.start_planes_mm, residual.step_length_mm,
            constants["last_plane_mm"]) / divisor[0:2]
        if are_whole_tracks(residual.start_planes_mm, constants["start_planes_mm"]):
            carried = carried.reshape(-1, len(constants["start_planes_mm"]), 2).sum(dim=1)
        return pooled, (carried ** 2).mean()

    def constants_with_the_network(self, first_round, network) -> dict:
        """The constants of the pooled loss, and lambda from the first round's
        states under the network as it is drawn from the seed."""
        constants = Pooled.constants(self, first_round.states)
        constants.update(name=self.name, start_planes_mm=self.start_planes_mm,
                         last_plane_mm=self.last_plane_mm)
        with torch.no_grad():
            track = network.predict(torch.as_tensor(first_round.states),
                                    torch.as_tensor(first_round.start_planes_mm))
            pooled, carried = (float(v) for v in self.parts(track, constants))
        weight = (0.0 if self.share == 0.0
                  else self.share * pooled / ((1.0 - self.share) * carried))
        constants.update(
            share_of_the_carried_term_at_round_one=self.share,
            first_round_states_are_whole_tracks=are_whole_tracks(
                torch.as_tensor(first_round.start_planes_mm), self.start_planes_mm),
            pooled_at_round_one=pooled, carried_at_round_one=carried,
            weight_of_the_carried_term=weight)
        return constants

    def constants(self, first_round_states):
        raise ValueError("the constants of %r need the network: the run measures them "
                         "with constants_with_the_network" % self.name)

    def value(self, predicted_track, target, constants):
        weight = float(constants["weight_of_the_carried_term"])
        if weight == 0.0:
            return self.squared_terms(predicted_track, target, constants).mean()
        pooled, carried = self.parts(predicted_track, constants)
        return pooled + weight * carried
