#!/usr/bin/env python
"""The chaining rule, and the pieces `chain_one.py` and `tables.py` share.

**One step.** A trained grid network maps a start state and a leg to the state
at the end of that leg:

    (x, y, tx, ty, qop) at z0,  extra = normalised (z0, dz)
        ->  (x, y, tx, ty) at z0 + dz,   qop carried through unchanged

**The chain.** A column of this experiment fixes a *nominal* step length and
walks the whole crossing with it. Each track has its own crossing length
L = |dz|, so the step is fitted to the track rather than the track truncated to
the step:

    N    = round(L / dz_nominal)          (at least 1)
    step = dz / N                          (signed: the track's own direction)

so the N steps land exactly on the far plane, every step of one track is the
same length, and the step is within half a percent of nominal on a 5.2 m
crossing for every column but the longest. The network's own endpoint becomes
the next step's start state; `(z0, dz)` are recomputed for every step and fed
in as the extra inputs the grid networks take; `qop` is passed through, as the
scheme demands. Everything is fp64.

The full-crossing column is N = 1: one step across the magnet, which is the
single-step full-crossing cell of `../C3_Step_size_and_stage_grid`'s tables and is
kept in the tables as the reference the chain has to beat.

**Batching.** The tracks are sorted by N, descending, so at global step i the
tracks still to be advanced are exactly the first `n_active` of the sorted
array. The work is therefore sum(N) network evaluations and not
max(N) x n_tracks.

**Checkpoints.** Fifty evenly spaced fractions of the crossing, f_k = k/50 for
k = 1..50, are recorded along the way. For a track with N steps the state at
fraction f is taken at step round(f N) - the nearest step boundary - and is
compared with the fine reference's own path there, linearly interpolated
between the 10 mm dense states of `../C0_Magnet_tracks_dataset`. When N < 50
several checkpoints collapse onto the same step; the fraction actually reached
is stored alongside the nominal one.
"""
from __future__ import annotations

import numpy as np
import torch

# name, nominal |dz| in mm (None = the whole crossing in one step), and how
# many tracks of each direction the column is run on.
COLUMNS = (
    ("0.1 mm", 0.1, 250),
    ("1 mm", 1.0, 500),
    ("10 mm", 10.0, 500),
    ("100 mm", 100.0, 500),
    ("1000 mm", 1000.0, 500),
    ("full crossing", None, 500),
)
COLUMN_KEYS = {"0.1 mm": "dz0p1", "1 mm": "dz1", "10 mm": "dz10",
               "100 mm": "dz100", "1000 mm": "dz1000",
               "full crossing": "full"}
COMPONENTS = ("x", "y", "tx", "ty", "max_xy")
N_CHECKPOINTS = 50
REFERENCE_FLOOR_UM = 5e-5          # ../C1_Fine_reference C1.4, over a crossing


def column_tracks(direction, n_per_direction):
    """Indices of the tracks a column runs on: the first n of each direction."""
    direction = np.asarray(direction)
    fwd = np.flatnonzero(direction == 1)[:n_per_direction]
    bwd = np.flatnonzero(direction == -1)[:n_per_direction]
    return np.sort(np.concatenate([fwd, bwd]))


def step_counts(dz, nominal):
    """(N,) steps per track and (N,) the signed step length."""
    dz = np.asarray(dz, dtype=np.float64)
    if nominal is None:
        n = np.ones(len(dz), dtype=np.int64)
    else:
        n = np.maximum(np.round(np.abs(dz) / float(nominal)), 1).astype(np.int64)
    return n, dz / n


def checkpoint_steps(n_steps, n_checkpoints=N_CHECKPOINTS):
    """(N, K) the step index each checkpoint is taken at, 1-based."""
    n_steps = np.asarray(n_steps, dtype=np.int64)
    frac = (np.arange(n_checkpoints, dtype=np.float64) + 1.0) / n_checkpoints
    j = np.round(frac[None, :] * n_steps[:, None]).astype(np.int64)
    return np.clip(j, 1, n_steps[:, None])


def chain_column(model, S0, z0, dz, nominal, extra_mean, extra_scale,
                 n_checkpoints=N_CHECKPOINTS, progress=None):
    """Walk every track across the magnet with the network.

    Returns
        S_end       (N, 5)          the state at the far plane
        CP_S        (N, K, 4)       the state at each checkpoint
        CP_Z        (N, K)          the z of each checkpoint
        n_steps     (N,)            steps taken
        step        (N,)            the signed step length used
    """
    S0 = np.asarray(S0, dtype=np.float64)
    z0 = np.asarray(z0, dtype=np.float64)
    n_steps, step = step_counts(dz, nominal)
    n = len(S0)

    order = np.argsort(-n_steps, kind="stable")           # longest chains first
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    S = torch.as_tensor(S0[order].copy())
    z = torch.as_tensor(z0[order].copy())
    h = torch.as_tensor(step[order].copy())
    ns = n_steps[order]

    cp_at = checkpoint_steps(ns, n_checkpoints)           # (N, K), sorted rows
    # (step, track, checkpoint) triples, walked with a pointer as i advances
    trip = np.stack([cp_at.ravel(),
                     np.repeat(np.arange(n), n_checkpoints),
                     np.tile(np.arange(n_checkpoints), n)], axis=1)
    trip = trip[np.argsort(trip[:, 0], kind="stable")]
    ptr = 0
    CP_S = np.empty((n, n_checkpoints, 4))
    CP_Z = np.empty((n, n_checkpoints))

    em = torch.as_tensor(np.asarray(extra_mean, dtype=np.float64))
    es = torch.as_tensor(np.asarray(extra_scale, dtype=np.float64))
    max_steps = int(ns.max())
    Snp, znp = S.numpy(), z.numpy()          # views on the same storage
    with torch.no_grad():
        for i in range(1, max_steps + 1):
            m = int(np.searchsorted(-ns, -i, side="right"))     # active prefix
            Sa, za, ha = S[:m], z[:m], h[:m]
            extra = torch.stack([(za - em[0]) / es[0], (ha - em[1]) / es[1]],
                                dim=1)
            out = model(Sa, extra)[:, -1, :]
            S[:m, :4] = out                  # qop is carried through untouched
            z[:m] = za + ha
            while ptr < len(trip) and trip[ptr, 0] == i:
                t = trip[ptr, 1]
                CP_S[t, trip[ptr, 2]] = Snp[t, :4]
                CP_Z[t, trip[ptr, 2]] = znp[t]
                ptr += 1
            if progress is not None and (i % progress == 0 or i == max_steps):
                print("      step %d / %d, %d tracks active"
                      % (i, max_steps, m), flush=True)
    S_end = Snp.copy()[inv]
    return (S_end, CP_S[inv], CP_Z[inv], n_steps, step)


def dense_interp(DZ, DS, off, n_nodes, z_at):
    """(K, 4) the fine reference's own path at the K planes `z_at`.

    Linear interpolation between the stored 10 mm states of one track. The
    stored z run forwards for a forward track and backwards for a backward one,
    so they are flipped where needed before interpolating.
    """
    zz = DZ[off:off + n_nodes]
    ss = DS[off:off + n_nodes, :4]
    if zz[-1] < zz[0]:
        zz, ss = zz[::-1], ss[::-1]
    return np.stack([np.interp(z_at, zz, ss[:, j]) for j in range(4)], axis=1)


def component_errors(pred, truth):
    """(N, 4) dx, dy in mm and dtx, dty dimensionless, prediction minus truth."""
    return np.asarray(pred)[:, :4] - np.asarray(truth)[:, :4]


def component_stats(err, mask=None):
    """The five components' summary of one (N, 4) error block.

    dx, dy are reported in micrometres and dtx, dty in milliradians - the
    house units of this line - and `max_xy` is max(|dx|, |dy|) in micrometres,
    the measure every earlier folder quotes. `mean` is the signed bias and is
    the mean of |max_xy| for that one unsigned component.
    """
    err = np.asarray(err, dtype=np.float64)
    if mask is not None:
        err = err[mask]
    out = {}
    if not len(err):
        return {c: dict(median_abs=float("nan"), p95_abs=float("nan"),
                        mean=float("nan"), rms=float("nan"), n=0)
                for c in COMPONENTS}
    scaled = np.column_stack([err[:, 0] * 1e3, err[:, 1] * 1e3,
                              err[:, 2] * 1e3, err[:, 3] * 1e3])
    for j, c in enumerate(("x", "y", "tx", "ty")):
        v = scaled[:, j]
        out[c] = dict(median_abs=float(np.median(np.abs(v))),
                      p95_abs=float(np.quantile(np.abs(v), 0.95)),
                      mean=float(v.mean()), rms=float(np.sqrt((v ** 2).mean())),
                      n=int(len(v)))
    mx = np.abs(scaled[:, :2]).max(axis=1)
    out["max_xy"] = dict(median_abs=float(np.median(mx)),
                         p95_abs=float(np.quantile(mx, 0.95)),
                         mean=float(mx.mean()),
                         rms=float(np.sqrt((mx ** 2).mean())), n=int(len(mx)))
    return out


UNITS = {"x": "um", "y": "um", "tx": "mrad", "ty": "mrad", "max_xy": "um"}


def straight_endpoint(S0, dz):
    """(N, 5) the straight-line propagation - the null step, chained or not.

    Chaining a straight line reproduces the single straight step exactly: each
    sub-step keeps tx and ty and adds tx*step to x, and the steps sum to dz.
    """
    S0 = np.asarray(S0, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    return np.column_stack([S0[:, 0] + S0[:, 2] * dz,
                            S0[:, 1] + S0[:, 3] * dz, S0[:, 2], S0[:, 3],
                            S0[:, 4]])


def parse_tag(tag):
    """'w256_d8_q20_physics_s1' -> (width, depth, q, mode, seed)."""
    parts = tag.split("_")
    return (int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:]),
            parts[3], int(parts[4][1:]))
