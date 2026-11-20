"""The stage residual of a path: the label-free loss of a whole crossing, with
the chaining inside the loss.

For a predicted track of N steps and q stages, the plane states are rebuilt
from the stages:

    s_0     = the input state of the first step
    F_ni    = f(S_ni, q/p, z_n + c_i h)
    s_{n+1} = s_n + h sum_i b_i F_ni

where S_ni is stage i of step n, f the rates of the equation of motion, z_n
the start plane of step n, c, b the nodes and weights of the tableau and h the
step length. The residual of stage i of step n, component d, is

    R_nid = S_nid - s_nd - h sum_j a_ij F_njd

with a the stage matrix. It is zero for every n, i and d exactly when the
stages solve the scheme chained from the start state. The loss of step n is

    L_n = mean over the rows, i and d of (R_nid / sigma_d)^2

with sigma_d the spread of component d over the first round's states, as in
the pooled loss (pooled.py). The loss is

    loss = sum_n w_n L_n / sum_n w_n

No label appears in it. The rates are read from the track when it keeps them
(the label-free path network does), and computed otherwise.

`gradient_through_plane_states`

    true    the gradient flows through s_n into every earlier stage
    false   s_n is detached: each step is fitted to the plane state the earlier
            steps give, and the earlier steps get no gradient from it. The value
            of the loss is the same

`weights`, w_n, never carry a gradient:

    uniform          w_n = 1
    causal           w_n = exp(-eps sum_{k<n} L_k), after Wang, Sankaran and
                     Perdikaris, arXiv 2203.07404. eps is measured once, from the
                     first round's states under the network as it is drawn from
                     the seed, so that the last step has the weight
                     `weight_of_the_last_step_at_round_one`:
                         eps = -ln(w_last) / sum_{k<N-1} L_k
                     It is recorded in the constants of the run
    sliding_window   w_n = 1 / (1 + exp((n - c) / tau)), n counted from 0, with
                     tau = `width_in_steps` and
                         c = block (1 + number of leading blocks of `block_of_steps`
                                    steps whose mean L_k is below `settled_below`)
                     It holds no state; c is found from the present residuals

The causal and sliding-window weights depend on the residuals. L-BFGS needs a
fixed objective within a restart, so the trainer calls `start_a_restart()`
before each restart; the weights are found at the first evaluation after it,
detached, and held until the next `start_a_restart()`. Before the first call,
or after `stop_holding()`, they are found afresh at every evaluation. Held
weights of another number of steps are found again.

A track of separate steps (S = 1, as in the check before training) has
s_0 = its input state, and the loss is the pooled loss with terms "stages".

Arms that leave a weight parameter unused give it as null.

Added 2026-11-20 for the experiment can_a_label_free_path_network_reach_the_exact_scheme
(approved by George the same day). No other loss changes.
Gates: tests/test_label_free_path.py.
"""
from __future__ import annotations

import math

import torch

from rkpinn.losses.pooled import spread_of_components
from rkpinn.registry import register

UNIFORM = "uniform"
SLIDING_WINDOW = "sliding_window"
CAUSAL = "causal"
WEIGHTS = (UNIFORM, SLIDING_WINDOW, CAUSAL)
PARAMETERS_OF_THE_WEIGHTS = {
    UNIFORM: (),
    SLIDING_WINDOW: ("block_of_steps", "width_in_steps", "settled_below"),
    CAUSAL: ("weight_of_the_last_step_at_round_one",),
}
ALL_PARAMETERS = ("block_of_steps", "width_in_steps", "settled_below",
                  "weight_of_the_last_step_at_round_one")


def residual_of_a_path(track, equation_of_motion, gradient_through_plane_states=True):
    """R, shape (n, S, q, 4), in the units of the state."""
    q = track.number_of_stages
    if q == 0:
        raise ValueError("a track with no stages has no stage residual")
    stages = torch.as_tensor(track.stage_states)
    inputs = torch.as_tensor(track.input_states)
    n, S = inputs.shape[:2]
    h = float(track.step_length_mm)
    A = torch.as_tensor(track.tableau.stage_matrix, dtype=torch.float64)
    b = torch.as_tensor(track.tableau.weights, dtype=torch.float64)
    if track.stage_rates is not None:
        F = torch.as_tensor(track.stage_rates)
    else:
        z = torch.as_tensor(track.stage_planes_mm(), dtype=torch.float64)      # (n, S, q)
        qop = inputs[:, :, 4:5].reshape(n * S, 1)
        F = equation_of_motion.rates(stages.reshape(n * S, q, 4), qop,
                                     z.reshape(n * S, q)).reshape(n, S, q, 4)
    first = inputs[:, :1, :4]
    if S > 1:
        increments = h * torch.einsum("i,bnid->bnd", b, F[:, :-1])
        planes = torch.cat([first, first + torch.cumsum(increments, dim=1)], dim=1)
    else:
        planes = first
    if not gradient_through_plane_states:
        planes = planes.detach()
    return stages - planes[:, :, None, :] - h * torch.einsum("ij,bnjd->bnid", A, F)


def loss_per_step(track, equation_of_motion, divisor, gradient_through_plane_states=True):
    """L_n, shape (S,)."""
    R = residual_of_a_path(track, equation_of_motion, gradient_through_plane_states)
    return ((R / torch.as_tensor(divisor, dtype=torch.float64)) ** 2).mean(dim=(0, 2, 3))


def causal_weights(per_step, eps):
    """w_n = exp(-eps sum_{k<n} L_k), detached."""
    L = per_step.detach()
    before = torch.cat([torch.zeros(1, dtype=L.dtype), torch.cumsum(L, 0)[:-1]])
    return torch.exp(-float(eps) * before)


def centre_of_the_window(per_step, block_of_steps, settled_below):
    """c = block (1 + the number of leading blocks whose mean L_k is below the threshold)."""
    L = per_step.detach()
    block = int(block_of_steps)
    settled = 0
    for start in range(0, len(L), block):
        if float(L[start:start + block].mean()) < float(settled_below):
            settled += 1
        else:
            break
    return block * (1 + settled)


def sliding_window_weights(per_step, block_of_steps, width_in_steps, settled_below):
    """w_n = 1 / (1 + exp((n - c) / tau)), detached."""
    c = centre_of_the_window(per_step, block_of_steps, settled_below)
    n = torch.arange(len(per_step), dtype=torch.float64)
    return torch.sigmoid(-(n - c) / float(width_in_steps))


@register("loss", "stage_residual_of_a_path")
class StageResidualOfAPath:
    name = "stage_residual_of_a_path"
    needs_labels = False
    settings_in_a_configuration = ("weights", "gradient_through_plane_states") + ALL_PARAMETERS

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(equation_of_motion=context.equation_of_motion, **settings)

    def __init__(self, *, equation_of_motion, weights, gradient_through_plane_states,
                 block_of_steps=None, width_in_steps=None, settled_below=None,
                 weight_of_the_last_step_at_round_one=None):
        if weights not in WEIGHTS:
            raise ValueError("the weights are %s, not %r" % (" or ".join(WEIGHTS), weights))
        if not isinstance(gradient_through_plane_states, bool):
            raise ValueError("gradient_through_plane_states is true or false")
        given = dict(block_of_steps=block_of_steps, width_in_steps=width_in_steps,
                     settled_below=settled_below,
                     weight_of_the_last_step_at_round_one=weight_of_the_last_step_at_round_one)
        used = PARAMETERS_OF_THE_WEIGHTS[weights]
        for name, value in given.items():
            if name in used and value is None:
                raise ValueError("the weights %r need %s" % (weights, name))
            if name not in used and value is not None:
                raise ValueError("the weights %r do not use %s; give it as null"
                                 % (weights, name))
        if weights == SLIDING_WINDOW:
            if int(block_of_steps) < 1 or float(width_in_steps) <= 0 or float(settled_below) <= 0:
                raise ValueError("the block, the width and the threshold are positive")
        if weights == CAUSAL and not 0.0 < float(weight_of_the_last_step_at_round_one) < 1.0:
            raise ValueError("the weight of the last step at round one is in (0, 1)")
        self.equation_of_motion = equation_of_motion
        self.weights = weights
        self.gradient_through_plane_states = gradient_through_plane_states
        self.parameters_of_the_weights = {name: given[name] for name in used}
        self._holding = False
        self._held = None

    # -- the constants ------------------------------------------------------------
    def constants(self, first_round_states) -> dict:
        if self.weights == CAUSAL:
            raise ValueError("the constants of the causal weights need the network: the "
                             "run measures them with constants_with_the_network")
        return {"name": self.name, "weights": self.weights,
                "gradient_through_plane_states": self.gradient_through_plane_states,
                "divisor_per_component":
                    spread_of_components(first_round_states)[:4].tolist(),
                **self.parameters_of_the_weights}

    def constants_with_the_network(self, first_round, network) -> dict:
        """The constants, and for the causal weights eps from the first round's
        states under the network as it is drawn from the seed."""
        if self.weights != CAUSAL:
            return self.constants(first_round.states)
        constants = {"name": self.name, "weights": self.weights,
                     "gradient_through_plane_states": self.gradient_through_plane_states,
                     "divisor_per_component":
                         spread_of_components(first_round.states)[:4].tolist(),
                     **self.parameters_of_the_weights}
        with torch.no_grad():
            track = network.predict(torch.as_tensor(first_round.states),
                                    torch.as_tensor(first_round.start_planes_mm))
            per_step = loss_per_step(track, self.equation_of_motion,
                                     constants["divisor_per_component"])
        before_the_last = float(per_step[:-1].sum())
        w_last = float(self.parameters_of_the_weights["weight_of_the_last_step_at_round_one"])
        constants.update(loss_per_step_summed_before_the_last_at_round_one=before_the_last,
                         causal_eps=-math.log(w_last) / before_the_last)
        return constants

    # -- the weights --------------------------------------------------------------
    def start_a_restart(self):
        """Hold the weights found at the next evaluation until the next call."""
        self._holding, self._held = True, None

    def stop_holding(self):
        self._holding, self._held = False, None

    def weights_of_the_steps(self, per_step, constants):
        """w_n, shape (S,), without a gradient."""
        if self.weights == UNIFORM:
            return torch.ones_like(per_step).detach()
        if self._holding and self._held is not None and len(self._held) == len(per_step):
            return self._held
        if self.weights == CAUSAL:
            w = causal_weights(per_step, constants["causal_eps"])
        else:
            p = self.parameters_of_the_weights
            w = sliding_window_weights(per_step, p["block_of_steps"], p["width_in_steps"],
                                       p["settled_below"])
        if self._holding:
            self._held = w
        return w

    # -- the loss -----------------------------------------------------------------
    def per_step(self, predicted_track, constants):
        return loss_per_step(predicted_track, self.equation_of_motion,
                             constants["divisor_per_component"],
                             constants["gradient_through_plane_states"])

    def squared_terms(self, predicted_track, target, constants):
        """The squared normalised residual of every row, step, stage and component."""
        R = residual_of_a_path(predicted_track, self.equation_of_motion,
                               constants["gradient_through_plane_states"])
        divisor = torch.as_tensor(constants["divisor_per_component"], dtype=torch.float64)
        return (R / divisor) ** 2

    def value(self, predicted_track, target, constants):
        L = self.per_step(predicted_track, constants)
        w = self.weights_of_the_steps(L, constants)
        return (w * L).sum() / w.sum()
