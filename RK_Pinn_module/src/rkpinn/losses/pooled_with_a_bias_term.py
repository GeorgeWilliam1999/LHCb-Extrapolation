"""The pooled loss with a bias term: item B.4 of the per-step bias study.

    loss = pooled + lambda * bias

    pooled  the pooled loss (pooled.py): the mean of (r[n, j, d] / s[d])^2
    bias    the mean, over the bins that hold a state, the stages j and the
            components d, of the squared mean signed residual in the bin:

                bias = mean over b, j, d of ( mean over n in b of r[n, j, d] / s[d] )^2

A bin is one cell of (band of start planes, band of momentum, charge): the
start planes of the steps are cut into `number_of_bands_of_start_planes` equal
bands, the momentum at `momentum_band_edges_gev`, and the charge by its sign.
The bias term asks the residual to average to zero in every bin, so that a
network cannot keep a residual of one sign in one region of the detector. It
is still a squared, label-free loss: zero at the exact scheme, and zero for a
residual whose mean in every bin is zero.

lambda is fixed once, when the run is made, so that the bias term is the share
`share_of_the_bias_term_at_round_one` of the whole loss of the first round's
states under the network as it is drawn from the seed:

    lambda * B1 = share * (P1 + lambda * B1),   lambda = share * P1 / ((1 - share) * B1)

It is recorded in the constants of the run (constants.json) with P1 and B1,
and it does not change during the run. With the share 0, lambda is 0 and the
loss is the pooled loss to the last bit.

`squared_terms` gives the pooled terms alone, so that the evaluation, which
reads where the loss puts its weight per state, sees the pooled loss.

Added 2026-10-28 for the per-step bias study (approved by George the same
day). `pooled.py` is unchanged, so no run made before changes its key.
Gates: tests/test_bias_study_changes.py.
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.losses.cost_weighted import momentum_gev
from rkpinn.losses.pooled import Pooled
from rkpinn.losses.stage_residual import stage_residual
from rkpinn.registry import register


def bins_of_states(input_states, start_planes_mm, first_start_plane_mm, last_start_plane_mm,
                   number_of_bands_of_start_planes, momentum_band_edges_gev):
    """The bin of every state, shape (m,), and the number of bins."""
    bands = int(number_of_bands_of_start_planes)
    span = max(float(last_start_plane_mm) - float(first_start_plane_mm), 1e-9)
    z_band = ((start_planes_mm - first_start_plane_mm) / span * bands).floor().long() \
        .clamp(0, bands - 1)
    edges = torch.as_tensor(np.asarray(momentum_band_edges_gev, dtype=np.float64))
    p_band = torch.bucketize(momentum_gev(input_states[:, 4]), edges)
    charge = (input_states[:, 4] < 0).long()
    number_of_p_bands = len(edges) + 1
    return (z_band * number_of_p_bands + p_band) * 2 + charge, bands * number_of_p_bands * 2


def bias_term(normalised_residual, bins, number_of_bins):
    """mean over the bins that hold a state, the stages and the components of
    the squared mean of the residual in the bin. normalised_residual (m, J, 4)."""
    m, J, D = normalised_residual.shape
    flat = normalised_residual.reshape(m, J * D)
    sums = torch.zeros(number_of_bins, J * D, dtype=flat.dtype).index_add_(0, bins, flat)
    counts = torch.zeros(number_of_bins, dtype=flat.dtype).index_add_(
        0, bins, torch.ones(m, dtype=flat.dtype))
    held = counts > 0
    means = sums[held] / counts[held, None]
    return (means ** 2).mean()


@register("loss", "pooled_with_a_bias_term")
class PooledWithABiasTerm(Pooled):
    name = "pooled_with_a_bias_term"
    needs_labels = False
    settings_in_a_configuration = ("terms", "share_of_the_bias_term_at_round_one",
                                   "number_of_bands_of_start_planes", "momentum_band_edges_gev")

    @classmethod
    def from_configuration(cls, settings, context):
        layout = context.layout
        return cls(equation_of_motion=context.equation_of_motion, terms=settings["terms"],
                   share_of_the_bias_term_at_round_one=settings[
                       "share_of_the_bias_term_at_round_one"],
                   number_of_bands_of_start_planes=settings["number_of_bands_of_start_planes"],
                   momentum_band_edges_gev=settings["momentum_band_edges_gev"],
                   first_start_plane_mm=float(layout.start_planes_mm[0]),
                   last_start_plane_mm=float(layout.start_planes_mm[-1]))

    def __init__(self, *, equation_of_motion, terms, share_of_the_bias_term_at_round_one,
                 number_of_bands_of_start_planes, momentum_band_edges_gev,
                 first_start_plane_mm, last_start_plane_mm):
        super().__init__(equation_of_motion=equation_of_motion, terms=terms)
        share = float(share_of_the_bias_term_at_round_one)
        if not 0.0 <= share < 1.0:
            raise ValueError("the share of the bias term is in [0, 1), not %r" % share)
        if int(number_of_bands_of_start_planes) < 1:
            raise ValueError("the number of bands of start planes is at least 1")
        edges = [float(e) for e in momentum_band_edges_gev]
        if edges != sorted(edges) or any(e <= 0 for e in edges):
            raise ValueError("the momentum band edges are positive and increasing")
        self.share = share
        self.number_of_bands_of_start_planes = int(number_of_bands_of_start_planes)
        self.momentum_band_edges_gev = edges
        self.first_start_plane_mm = float(first_start_plane_mm)
        self.last_start_plane_mm = float(last_start_plane_mm)

    def parts(self, predicted_track, constants):
        """(pooled, bias) of a predicted track of separate steps."""
        residual = stage_residual(predicted_track, self.equation_of_motion, constants["terms"])
        divisor = torch.as_tensor(constants["divisor_per_component"], dtype=torch.float64)
        normalised = residual.residual / divisor
        bins, number = bins_of_states(
            residual.input_states, residual.start_planes_mm, self.first_start_plane_mm,
            self.last_start_plane_mm, self.number_of_bands_of_start_planes,
            self.momentum_band_edges_gev)
        return (normalised ** 2).mean(), bias_term(normalised, bins, number)

    def constants_with_the_network(self, first_round, network) -> dict:
        """The constants of the pooled loss, and lambda from the first round's
        states under the network as it is drawn from the seed."""
        constants = Pooled.constants(self, first_round.states)
        constants["name"] = self.name
        with torch.no_grad():
            track = network.predict(torch.as_tensor(first_round.states),
                                    torch.as_tensor(first_round.start_planes_mm))
            pooled, bias = (float(v) for v in self.parts(track, constants))
        weight = (0.0 if self.share == 0.0
                  else self.share * pooled / ((1.0 - self.share) * bias))
        constants.update(
            share_of_the_bias_term_at_round_one=self.share,
            number_of_bands_of_start_planes=self.number_of_bands_of_start_planes,
            momentum_band_edges_gev=self.momentum_band_edges_gev,
            first_start_plane_mm=self.first_start_plane_mm,
            last_start_plane_mm=self.last_start_plane_mm,
            pooled_at_round_one=pooled, bias_at_round_one=bias,
            weight_of_the_bias_term=weight)
        return constants

    def constants(self, first_round_states):
        raise ValueError("the constants of %r need the network: the run measures them "
                         "with constants_with_the_network" % self.name)

    def value(self, predicted_track, target, constants):
        weight = float(constants["weight_of_the_bias_term"])
        if weight == 0.0:
            return self.squared_terms(predicted_track, target, constants).mean()
        pooled, bias = self.parts(predicted_track, constants)
        return pooled + weight * bias
