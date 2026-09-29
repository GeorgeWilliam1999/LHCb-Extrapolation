"""Collecting and summing the stages: the end of a step formed from its stages.

    end state = input state + step length * sum over k of  b_k * f(stage state k, stage plane k)

where b are the weights of the tableau and f the rates of the equation of
motion. The end state is then consistent with the stages by construction: it
is not a separate output of the network.

Whether the end state is formed this way or predicted by the network is a
setting of the run (George, 2026-09-28).
"""
from __future__ import annotations

import torch


def rates_at_the_stages(equation_of_motion, input_states, stage_states, stage_planes_mm):
    """f at every stage state; shapes (n, 5), (n, q, 4), (n, q) -> (n, q, 4)."""
    return equation_of_motion.rates(stage_states, input_states[:, 4:5], stage_planes_mm)


def collect_and_sum(input_states, stage_rates, step_length_mm, weights):
    """The end state, shape (n, 4), from the rates at the stages, shape (n, q, 4).

    weights  the weights b of the tableau, a tensor of shape (q,)
    """
    return input_states[:, :4] + step_length_mm * torch.einsum(
        "j,njd->nd", weights, stage_rates)
