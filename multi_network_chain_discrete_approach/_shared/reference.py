"""The reference card, generalised: constants, the ODE, the fp64 RK4 engine,
the metric, the tableau and the training set.

This is `S0_Baseline_data_exploration/reference_card.py` with two changes and no
others:

  * the field loader is the vendored copy in this package (`field_v8r1.py`),
    not a `sys.path` reach into the archive (see `vendoring_parity.py`);
  * `deriv` and `rk4_rows` take the field as an argument, so the MagDown and
    MagUp maps can be used side by side in one process. Called without a field
    they use the MagDown map, exactly as before.

Nothing here is new physics. The ODE, kappa, the q/p convention and RK4 mirror
`Data_generation_exploration/Data/make_training_set.py`; the scalar error rho
is the metric agreed for the van der Pol study (2026-07-13), applied to the
four dynamic state components.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

try:                                     # imported as a package, or run directly
    from .field_v8r1 import FieldV8R1, V8R1_DOWN
    from .irk import gauss_legendre, verify_tableau
except ImportError:                      # pragma: no cover
    from field_v8r1 import FieldV8R1, V8R1_DOWN
    from irk import gauss_legendre, verify_tableau

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# ---- the two field-map polarities ------------------------------------------
V8R1_UP = "/cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.up.bin"

# ---- constants (Allen conventions, validated in the vertex-fit line) --------
KAPPA = 1.0e-3            # with qop in Allen units
C_QP = 0.299792458        # qop = C_QP * q / p[GeV]
RK4_STEP = 5.0            # mm, the reference integrator's fixed step

# ---- the frozen leg of the baseline experiment (Task C, 2026-07-18): the
# modal plane pair of the 11,046 forward cross-magnet legs in the training set
FROZEN_LEG = dict(z0=2648.2, z1=7826.0)   # last UT plane -> first SciFi plane [mm]

# ---- the training set (v2: the official TestFileDB sample; George 2026-07-21
# ruled out self-generated events as a provenance source) --------------------
DATA_NPZ = os.environ.get("TRAINING_NPZ") or os.path.join(
    REPO, "Data_generation_exploration", "Official_xdigi",
    "training_v2", "train_official_v2.npz")

LEG_NAMES = ("A", "B", "C", "D")          # A vertex fetch, B cross-magnet,
                                          # C plane-to-plane, D downstream

_FIELDS: dict[str, FieldV8R1] = {}


def make_field(which: str = "down") -> FieldV8R1:
    """The canonical v8r1 map for one polarity, loaded once per process.

    which: 'down' (the default everywhere so far) or 'up'.
    """
    which = which.lower()
    if which not in ("down", "up"):
        raise ValueError("field polarity must be 'down' or 'up', not %r" % which)
    if which not in _FIELDS:
        _FIELDS[which] = FieldV8R1(V8R1_DOWN if which == "down" else V8R1_UP)
    return _FIELDS[which]


def field_path(which: str = "down") -> str:
    return V8R1_DOWN if which.lower() == "down" else V8R1_UP


def field_md5(which: str = "down", chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(field_path(which), "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def field_bounds(field: FieldV8R1):
    """(lo, hi) corners of the map in mm, as two 3-vectors."""
    lo = np.array([field.min[i] for i in range(3)], dtype=np.float64)
    hi = lo + (np.asarray(field.N, dtype=np.float64) - 1.0) / np.asarray(field.invD)
    return lo, hi


def deriv(S, z, field=None):
    """The LHCb equation of motion, dS/dz for S = (x, y, tx, ty, qop).

    Identical to Allen's ODE and to the training-set label engine
    (Data/make_training_set.py); z may be a scalar or a per-row array.
    """
    if field is None:
        field = make_field("down")
    x, y, tx, ty, qop = S.T
    if np.isscalar(z):
        z = np.full_like(x, z)
    Bx, By, Bz = field(x, y, z)
    k = KAPPA * qop
    N = np.sqrt(1 + tx * tx + ty * ty)
    return np.stack(
        [
            tx,
            ty,
            k * N * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz),
            k * N * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz),
            np.zeros_like(x),
        ],
        axis=1,
    )


def rk4_rows(S0, z0, z1, step=RK4_STEP, field=None):
    """The fp64 reference integrator: fixed-step RK4, per-row (z0, z1), masked.

    Mirrored from Data/make_training_set.py (single source of truth for the
    training labels; keep in sync).
    """
    if field is None:
        field = make_field("down")
    S = S0.astype(np.float64).copy()
    z = np.asarray(z0, dtype=np.float64).copy()
    z1 = np.asarray(z1, dtype=np.float64)
    sign = np.sign(z1 - z)
    sign[sign == 0] = 1.0
    remaining = (z1 - z) * sign
    while True:
        h = np.minimum(remaining, step) * sign
        active = remaining > 1e-12
        if not active.any():
            break
        ha = h[active][:, None]
        Sa, za = S[active], z[active]
        k1 = deriv(Sa, za, field)
        k2 = deriv(Sa + 0.5 * ha * k1, za + 0.5 * h[active], field)
        k3 = deriv(Sa + 0.5 * ha * k2, za + 0.5 * h[active], field)
        k4 = deriv(Sa + ha * k3, za + h[active], field)
        S[active] = Sa + (ha / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        z[active] += h[active]
        remaining = (z1 - z) * sign
    return S


# ---- the fine reference: Butcher's seven-stage explicit method of order six --
# Coefficients copied from /data/bfys/gscriven/Van_Der_Pole/RK_Truth/rk6.py
# (the van der Pol study's reference machinery, read-only there). They are not
# re-derived here; `C1_Fine_reference/check_tableau.py` compares these arrays with
# that file element by element, re-runs its identity battery, and re-measures
# the order both on its three exact-solution problems and on `rk6_rows` itself.
RK6_C = np.array([0.0, 1 / 3, 2 / 3, 1 / 3, 5 / 6, 1 / 6, 1.0])

RK6_A = np.zeros((7, 7))
RK6_A[1, 0] = 1 / 3
RK6_A[2, 0], RK6_A[2, 1] = 0.0, 2 / 3
RK6_A[3, 0], RK6_A[3, 1], RK6_A[3, 2] = 1 / 12, 1 / 3, -1 / 12
RK6_A[4, 0], RK6_A[4, 1], RK6_A[4, 2], RK6_A[4, 3] = 25 / 48, -55 / 24, 35 / 48, 15 / 8
RK6_A[5, 0], RK6_A[5, 1], RK6_A[5, 2], RK6_A[5, 3], RK6_A[5, 4] = \
    3 / 20, -11 / 24, -1 / 8, 1 / 2, 1 / 10
RK6_A[6, 0], RK6_A[6, 1], RK6_A[6, 2] = -261 / 260, 33 / 13, 43 / 156
RK6_A[6, 3], RK6_A[6, 4], RK6_A[6, 5] = -118 / 39, 32 / 195, 80 / 39

RK6_B = np.array([13 / 200, 0.0, 11 / 40, 11 / 40, 4 / 25, 4 / 25, 13 / 200])

RK6_ORDER = 6
RK6_STAGES = 7
RK6_STEP = 0.1           # mm, the fine reference's step (Block C, C1.4)


def rk6_rows(S0, z0, z1, step=RK6_STEP, field=None):
    """The fine fp64 reference: fixed-step RK6, per-row (z0, z1), masked.

    The same contract as `rk4_rows` - a state array `S0` of shape (N, 5), a
    per-row start plane `z0` and end plane `z1` (scalars broadcast), fp64
    throughout, the field defaulting to MagDown - with two differences: the
    tableau is Butcher's seven-stage method of order six instead of classical
    RK4, and the default step is the fine one.

    Every row marches in steps of `step` towards its own `z1`; the last step of
    each row is shortened so that the row lands exactly on `z1`, and rows that
    have arrived drop out of the active set. `z1 < z0` integrates backwards
    (the step is taken with the sign of `z1 - z0`), and a row with
    `z1 == z0` is returned unchanged.
    """
    if field is None:
        field = make_field("down")
    S = np.atleast_2d(np.asarray(S0, dtype=np.float64)).copy()
    z = np.broadcast_to(np.asarray(z0, dtype=np.float64),
                        (len(S),)).astype(np.float64).copy()
    z1 = np.broadcast_to(np.asarray(z1, dtype=np.float64), (len(S),)).astype(np.float64)
    step = abs(float(step))
    sign = np.sign(z1 - z)
    sign[sign == 0] = 1.0
    k = np.empty((RK6_STAGES, len(S), S.shape[1]), dtype=np.float64)
    remaining = (z1 - z) * sign
    # the march plus a small allowance for the sub-ulp residuals that a
    # shortened last step can leave behind; a hard cap so a row can never spin.
    max_sweeps = int(np.ceil(remaining.max() / step)) + 8 if len(S) else 0
    for _ in range(max_sweeps):
        active = remaining > 1e-12
        if not active.any():
            break
        h = np.minimum(remaining[active], step) * sign[active]
        ha = h[:, None]
        Sa, za = S[active], z[active]
        ka = k[:, :len(Sa)]
        ka[0] = deriv(Sa, za, field)
        for i in range(1, RK6_STAGES):
            Y = Sa + ha * np.tensordot(RK6_A[i, :i], ka[:i], axes=(0, 0))
            ka[i] = deriv(Y, za + RK6_C[i] * h, field)
        S[active] = Sa + ha * np.tensordot(RK6_B, ka, axes=(0, 0))
        z[active] = za + h
        remaining = (z1 - z) * sign
    return S


def rk6_dense_rows(S0, z0, z1, sample_mm=10.0, step=RK6_STEP, field=None):
    """`rk6_rows`, keeping the state every `sample_mm` of |z| along the way.

    Every row starts at its own `z0` and marches to its own `z1`; the sampling
    grid is the row's own, z0 + j * sample_mm * sign(z1 - z0), and the exact
    endpoint z1 is appended as the last node of every row. Rows differ in how
    many grid nodes they have (their |z1 - z0| differ), so the output is padded
    to the longest row and a boolean mask says which nodes are real.

    `sample_mm` must be a whole number of steps (it is asserted), so that the
    sampled nodes are step boundaries of the march and the stored states are
    exactly the states `rk6_rows` passes through - a state read out of this
    grid can be integrated onwards without any re-basing.

    Returns (Zg, Sg, valid) with shapes (N, K), (N, K, 5) and (N, K).
    Node 0 is the start state itself.
    """
    if field is None:
        field = make_field("down")
    step = abs(float(step))
    n_sub = int(round(sample_mm / step))
    assert abs(n_sub * step - sample_mm) < 1e-12 and n_sub >= 1, (
        "sample_mm=%r must be a whole number of steps of %r" % (sample_mm, step))
    S = np.atleast_2d(np.asarray(S0, dtype=np.float64)).copy()
    N = len(S)
    z0 = np.broadcast_to(np.asarray(z0, dtype=np.float64), (N,)).astype(np.float64)
    z1 = np.broadcast_to(np.asarray(z1, dtype=np.float64), (N,)).astype(np.float64)
    span = np.abs(z1 - z0)
    n_full = np.floor(span / sample_mm + 1e-9).astype(np.int64)   # whole nodes
    K = int(n_full.max()) + 2                                     # +start +endpoint
    Zg = np.full((N, K), np.nan)
    Sg = np.full((N, K, S.shape[1]), np.nan)
    valid = np.zeros((N, K), dtype=bool)
    sign = np.sign(z1 - z0)
    sign[sign == 0] = 1.0

    Zg[:, 0], Sg[:, 0], valid[:, 0] = z0, S, True
    cur = S.copy()
    zcur = z0.copy()
    for j in range(1, int(n_full.max()) + 1):
        live = n_full >= j
        ztarget = z0[live] + j * sample_mm * sign[live]
        cur[live] = rk6_rows(cur[live], zcur[live], ztarget, step=step, field=field)
        zcur[live] = ztarget
        Zg[live, j], Sg[live, j], valid[live, j] = ztarget, cur[live], True
    # the exact endpoint, as each row's last node (skipped where it coincides
    # with the last whole node to within a thousandth of a step)
    tail = np.abs(z1 - zcur) > 1e-3 * step
    idx = np.where(tail, n_full + 1, n_full)
    write = tail | (n_full > 0)          # never overwrite node 0 with itself
    endS = cur.copy()
    if tail.any():
        endS[tail] = rk6_rows(cur[tail], zcur[tail], z1[tail], step=step, field=field)
    rows = np.arange(N)[write]
    Zg[rows, idx[write]] = z1[write]
    Sg[rows, idx[write]] = endS[write]
    valid[rows, idx[write]] = True
    K_used = int(valid.any(axis=0).nonzero()[0].max()) + 1
    return Zg[:, :K_used], Sg[:, :K_used], valid[:, :K_used]


def rho(y_ref, y_pred):
    """The agreed scalar relative error (George, 2026-07-13, van der Pol study):
    rho = min(1, ||ref - pred||^2 / ||ref||^2), Euclidean over the four dynamic
    components (x, y, tx, ty), evaluated pointwise."""
    y_ref = np.atleast_2d(y_ref)[:, :4]
    y_pred = np.atleast_2d(y_pred)[:, :4]
    num = ((y_ref - y_pred) ** 2).sum(axis=1)
    den = (y_ref ** 2).sum(axis=1)
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0),
                   np.where(num > 0, np.inf, 0.0))
    return np.minimum(1.0, out)


# The columns `load_training` returns by default. The npz also carries ETA and
# ORIGIN_R; ask for them with `fields=` rather than changing this tuple, so that
# every caller written before 2026-09-06 gets exactly the dict it got then.
LOAD_FIELDS = ("X", "Y", "LEG", "P", "PID", "CORE", "SPLIT", "EVT", "MCKEY")


def load_training(split=None, leg=None, core_only=False, npz=None, fields=None):
    """The event-derived training set with its frozen splits.

    split:  None | 'train' | 'val' | 'test'   (by-particle, seed 20260718)
    leg:    None | 'A'..'D' | 0..3 | a sequence of either
    fields: None for `LOAD_FIELDS` (the default since this function was
            written), or an explicit tuple of column names - e.g.
            `LOAD_FIELDS + ("ETA", "ORIGIN_R")`.
    """
    d = np.load(os.path.abspath(npz or DATA_NPZ))
    m = np.ones(len(d["X"]), dtype=bool)
    if split is not None:
        m &= d["SPLIT"] == {"train": 0, "val": 1, "test": 2}[split]
    if leg is not None:
        m &= np.isin(d["LEG"], leg_indices(leg))
    if core_only:
        m &= d["CORE"] == 1
    return {k: d[k][m] for k in (LOAD_FIELDS if fields is None else tuple(fields))}


def leg_indices(leg):
    """'B' -> [1]; ('A','B','C') -> [0,1,2]; 1 -> [1]; already-integers pass."""
    if isinstance(leg, (str, int, np.integer)):
        leg = [leg]
    out = []
    for item in leg:
        if isinstance(item, str):
            out.append(LEG_NAMES.index(item.upper()))
        else:
            out.append(int(item))
    return out


def card(field_which="down"):
    """The printable reference card (what every experiment is standing on)."""
    return {
        "ode": "x'=tx, y'=ty, tx'=k qop N (tx ty Bx - (1+tx^2) By + ty Bz), "
               "ty'=k qop N ((1+ty^2) Bx - tx ty By - tx Bz), qop'=0;  "
               "N=sqrt(1+tx^2+ty^2)",
        "kappa": KAPPA,
        "qop_convention": "qop = 0.299792458 * q / p[GeV]  (Allen units)",
        "field": {"which": field_which, "file": field_path(field_which),
                  "md5": field_md5(field_which),
                  "loader": "vendored _shared/field_v8r1.py "
                            "(archive commit 1faa97e0, parity-gated)"},
        "reference_integrator": "fp64 fixed-step RK4, %.1f mm" % RK4_STEP,
        "metric": "rho = min(1, ||dY||^2/||Y||^2) over (x,y,tx,ty); "
                  "plus tier-1 um bars",
        "frozen_leg_mm": FROZEN_LEG,
        "training_set": os.path.abspath(DATA_NPZ),
        "splits": "by particle, 80/10/10, seed 20260718 (stored in the npz)",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(card(), indent=1))
    assert verify_tableau(8)["passes"]
    d = load_training(split="train")
    print("train rows:", len(d["X"]), " per leg:",
          {t: int((d["LEG"] == i).sum()) for i, t in enumerate(LEG_NAMES)})
    print("gauss_legendre(8) nodes:", np.round(gauss_legendre(8)[0], 6))
