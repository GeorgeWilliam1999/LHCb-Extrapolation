#!/usr/bin/env python
"""The Block E network: one network for one step length, used on every step.

For a step count N and a stage count q the crossing z0 -> z1 is cut into N
steps of dz = L/N. ONE network serves all of them. It takes a track state and
the z of the plane the step starts on, and emits the q Gauss-Legendre stage
states and the end state of that step:

    (x, y, tx, ty, q/p) at z_start,  z_start   ->   (q + 1) x (x, y, tx, ty)

Applied N times - each end state, with q/p carried unchanged, is the next
step's input, and z_start advances by dz - it carries a track from z0 to z1
(`carry`).

## Inputs

The state is divided by its spread over the round-1 training states (a fixed
buffer, `in_scale`); z_start is mapped linearly onto [-1, 1] across the start
planes z0 .. z1 - dz. Knowing z_start is what lets one network serve every
step: the field a step crosses depends on where along the magnet it starts
(George 2026-09-16, decision 2 (a)).

## Outputs: the deviation from the straight line (Block D's form)

    output_j = straight_j + scale (x) raw_j,        j = 1 .. q, end
    straight_j = (x + tx (z_j - z_start), y + ty (z_j - z_start), tx, ty)

with per-track scales computed from the input and the field map alone, along
the straight line through the input (16-point midpoint rule):

    x, tx  (Block D, unchanged)
        base_x  = kappa |q/p| INT |B| dz
        slope_x = max(base_x, 1e-12)          pos_x = max(base_x |dz|/2, 1e-9 mm)

    y, ty  (new: a separate scale for the bending in y)
        I_y     = INT sqrt(1 + tx^2 + ty^2) ((1 + ty^2)|Bx| + |tx ty By| + |tx Bz|) dz
        base_y  = max(kappa |q/p| I_y,  Y_FLOOR_REL base_x)
        slope_y = max(base_y, 1e-12)          pos_y = max(base_y |dz|/2, 1e-9 mm)

I_y bounds the y-rate of the equation of motion,
dty/dz = kappa (q/p) N ((1 + ty^2) Bx - tx ty By - tx Bz), term by term, so it
does not vanish through cancellation. Block D used the x scale for y as well,
which set the y output about 500 times too coarse (D3_Error_anatomy).

## Loss

The network plugs into `_shared.model.physics_loss` unchanged: it declares
`n_extra = 1`, so the shared residual calls `model(S, z_start)`, and the stage
planes are passed per track, znodes = z_start + c dz.

Also here: numpy twins of the field integrals and scales (for the gate), and
`carry`, the N-fold application.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

import use_shared                                        # noqa: F401
from _shared.field_torch import FieldTorch
from _shared.reference import KAPPA

torch.set_default_dtype(torch.float64)

POS_FLOOR_MM = 1e-9
SLOPE_FLOOR = 1e-12
Y_FLOOR_REL = 1e-3
N_FIELD_SAMPLES = 16
WIDTH, DEPTH = 128, 2


# ------------------------------------------------------- numpy twins (gates) --
def field_integrals_numpy(S, z_start, dz, field, n_samples=N_FIELD_SAMPLES):
    """(I_B, I_y) along the straight line from each state; S (n, 5), z_start (n,)."""
    S = np.asarray(S, dtype=np.float64)
    z_start = np.broadcast_to(np.asarray(z_start, dtype=np.float64), (len(S),))
    u = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples
    s = u[None, :] * dz
    x = S[:, 0:1] + S[:, 2:3] * s
    y = S[:, 1:2] + S[:, 3:4] * s
    z = z_start[:, None] + s
    Bx, By, Bz = field(x.ravel(), y.ravel(), z.ravel())
    Bx, By, Bz = (np.asarray(v).reshape(x.shape) for v in (Bx, By, Bz))
    tx, ty = S[:, 2:3], S[:, 3:4]
    norm = np.sqrt(1.0 + tx * tx + ty * ty)
    I_B = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2).mean(axis=1) * abs(dz)
    I_y = (norm * ((1.0 + ty * ty) * np.abs(Bx) + np.abs(tx * ty * By)
                   + np.abs(tx * Bz))).mean(axis=1) * abs(dz)
    return I_B, I_y


def scales_numpy(S, dz, I_B, I_y, y_floor_rel=Y_FLOOR_REL):
    """(pos_x, pos_y, slope_x, slope_y), each (n,)."""
    qop = np.abs(np.asarray(S)[:, 4])
    base_x = KAPPA * qop * I_B
    base_y = np.maximum(KAPPA * qop * I_y, y_floor_rel * base_x)
    return (np.maximum(base_x * abs(dz) / 2.0, POS_FLOOR_MM),
            np.maximum(base_y * abs(dz) / 2.0, POS_FLOOR_MM),
            np.maximum(base_x, SLOPE_FLOOR),
            np.maximum(base_y, SLOPE_FLOOR))


def straight_numpy(S, dz):
    """(n, 4) the straight-line end state."""
    S = np.asarray(S)
    return np.stack([S[:, 0] + S[:, 2] * dz, S[:, 1] + S[:, 3] * dz, S[:, 2], S[:, 3]], axis=1)


# ---------------------------------------------------------------- the model --
class ChainNetwork(torch.nn.Module):
    """(state, z_start) -> q stage states + end state of one step of length dz."""

    def __init__(self, q, c, dz, z_first, z_last, in_scale, field, width=WIDTH,
                 depth=DEPTH, n_field_samples=N_FIELD_SAMPLES, y_floor_rel=Y_FLOOR_REL):
        super().__init__()
        self.q = int(q)
        self.n_extra = 1                       # the shared residual passes z_start
        cc = torch.as_tensor(np.asarray(c, dtype=np.float64)).reshape(-1)
        if len(cc) != self.q:
            raise ValueError("c has %d nodes but q = %d" % (len(cc), self.q))
        self.register_buffer("in_scale", torch.as_tensor(np.asarray(in_scale, dtype=np.float64)))
        self.register_buffer("c", cc)
        self.register_buffer("cout", torch.cat([cc, torch.ones(1, dtype=cc.dtype)]))
        self.register_buffer("dz", torch.tensor(float(dz)))
        self.register_buffer("z_mid", torch.tensor(0.5 * (float(z_first) + float(z_last))))
        self.register_buffer("z_half", torch.tensor(max(0.5 * (float(z_last) - float(z_first)), 1.0)))
        u = (torch.arange(int(n_field_samples), dtype=torch.float64) + 0.5) / int(n_field_samples)
        self.register_buffer("u", u)
        self.y_floor_rel = float(y_floor_rel)
        # held in a tuple ON PURPOSE: as a submodule the field twin's grid
        # buffers would be written into every checkpoint
        self._field_holder = (FieldTorch(field),)
        layers, n_in = [], 6
        for _ in range(int(depth)):
            layers += [torch.nn.Linear(n_in, int(width)), torch.nn.Tanh()]
            n_in = int(width)
        layers += [torch.nn.Linear(n_in, 4 * (self.q + 1))]
        self.net = torch.nn.Sequential(*layers)
        self._cache = None

    @property
    def field(self):
        return self._field_holder[0]

    def z_input(self, z_start):
        return (z_start - self.z_mid) / self.z_half

    def field_integrals(self, S, z_start):
        with torch.no_grad():
            s = self.u[None, :] * self.dz
            x = S[:, 0:1] + S[:, 2:3] * s
            y = S[:, 1:2] + S[:, 3:4] * s
            z = z_start[:, None] + s
            Bx, By, Bz = self.field(x, y, z)
            tx, ty = S[:, 2:3], S[:, 3:4]
            norm = torch.sqrt(1.0 + tx * tx + ty * ty)
            I_B = torch.sqrt(Bx * Bx + By * By + Bz * Bz).mean(dim=1) * self.dz.abs()
            I_y = (norm * ((1.0 + ty * ty) * Bx.abs() + (tx * ty * By).abs()
                           + (tx * Bz).abs())).mean(dim=1) * self.dz.abs()
        return I_B, I_y

    def residual_scale(self, S, z_start):
        """(n, q+1, 4) the per-track scale; cached on the identity of (S, z_start)."""
        cached = self._cache
        if cached is not None and cached[0] is S and cached[1] is z_start:
            return cached[2]
        with torch.no_grad():
            I_B, I_y = self.field_integrals(S, z_start)
            qop = S[:, 4].abs()
            base_x = KAPPA * qop * I_B
            base_y = torch.maximum(KAPPA * qop * I_y, self.y_floor_rel * base_x)
            half = self.dz.abs() / 2.0
            pos_x = (base_x * half).clamp_min(POS_FLOOR_MM)
            pos_y = (base_y * half).clamp_min(POS_FLOOR_MM)
            slope_x = base_x.clamp_min(SLOPE_FLOOR)
            slope_y = base_y.clamp_min(SLOPE_FLOOR)
            ones = torch.ones_like(self.cout)
            sc = torch.stack([pos_x[:, None] * ones, pos_y[:, None] * ones,
                              slope_x[:, None] * ones, slope_y[:, None] * ones], dim=-1)
        self._cache = (S, z_start, sc)
        return sc

    def straight(self, S):
        off = self.cout[None, :] * self.dz
        x = S[:, 0:1] + S[:, 2:3] * off
        y = S[:, 1:2] + S[:, 3:4] * off
        return torch.stack([x, y, S[:, 2:3].expand_as(x), S[:, 3:4].expand_as(x)], dim=-1)

    def raw(self, S, z_start):
        inp = torch.cat([S / self.in_scale, self.z_input(z_start)[:, None]], dim=1)
        return self.net(inp).reshape(-1, self.q + 1, 4)

    def forward(self, S, extra):
        # keep the caller's tensor object when it is already flat: the scale
        # cache is keyed on identity, and the loss calls this ~250 times a restart
        z_start = extra if extra.dim() == 1 else extra.reshape(-1)
        return self.straight(S) + self.residual_scale(S, z_start) * self.raw(S, z_start)


# ----------------------------------------------------------- building / using --
def znodes_for(model, z_start):
    """(n, q) the stage planes of each track's step."""
    return z_start[:, None] + model.c[None, :] * model.dz


def build(q, N, L, z0, in_scale, field, width=WIDTH, depth=DEPTH, seed=0,
          y_floor_rel=Y_FLOOR_REL):
    from _shared.reference import gauss_legendre
    c, _, _ = gauss_legendre(q)
    dz = L / N
    torch.manual_seed(seed)
    return ChainNetwork(q, c, dz, z0, z0 + (N - 1) * dz, in_scale, field,
                        width=width, depth=depth, y_floor_rel=y_floor_rel)


def load_network(run_dir, field):
    """Rebuild a trained network from its folder (scale.json + network.pt)."""
    with open(os.path.join(run_dir, "scale.json")) as f:
        sc = json.load(f)
    model = build(sc["q"], sc["N"], sc["L"], sc["z0"], sc["in_scale"], field,
                  width=sc["width"], depth=sc["depth"], seed=sc["seed"],
                  y_floor_rel=sc["y_floor_rel"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "network.pt"), weights_only=True))
    model.eval()
    return model


def step_outputs(model, S, z_start, batch=8192):
    """(n, q+1, 4) the network's outputs, numpy in and out, no gradient."""
    S = np.asarray(S, dtype=np.float64)
    z_start = np.broadcast_to(np.asarray(z_start, dtype=np.float64), (len(S),))
    out = np.empty((len(S), model.q + 1, 4))
    with torch.no_grad():
        for i in range(0, len(S), batch):
            St = torch.as_tensor(S[i:i + batch])
            zt = torch.as_tensor(np.ascontiguousarray(z_start[i:i + batch]))
            out[i:i + batch] = model(St, zt).numpy()
    return out


def carry(model, S0, z0, N):
    """(n, N+1, 5) the state on every plane: the network applied N times from z0.

    Each step's end state, with q/p carried through unchanged, is the next
    step's input; the step's start plane advances by dz."""
    S = np.asarray(S0, dtype=np.float64).copy()
    dz = float(model.dz)
    out = np.empty((len(S), N + 1, 5))
    out[:, 0] = S
    for k in range(N):
        end = step_outputs(model, S, z0 + k * dz)[:, -1, :]
        S = np.concatenate([end, S[:, 4:5]], axis=1)
        out[:, k + 1] = S
    return out
