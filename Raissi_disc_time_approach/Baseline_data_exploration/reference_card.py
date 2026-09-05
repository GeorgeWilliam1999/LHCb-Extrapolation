"""The reference card: every constant, convention and engine the discrete-time
port relies on, importable from one place. Step 0 of the port plan.

Nothing here is new physics. The ODE, kappa, q/p convention and RK4 are
mirrored from Data_generation_exploration/Data/make_training_set.py (which
itself mirrors the validated datagen engine); the field loader is the vendored
copy in ../_shared/ (parity-gated against the archive original); the scalar
error rho is the metric agreed for the van der Pol study (2026-07-13), applied
to the four dynamic state components.

Import:  from reference_card import deriv, rk4_rows, rho, FIELD, card, load_training
Run   :  /data/bfys/gscriven/conda/envs/TE/bin/python reference_card.py  (prints the card)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

# ---- canonical field (vendored into this repository, 2026-09-05) ------------
# Was a read-only sys.path import from the archive at
# Track_Extrapolation_work_archive/track-extrapolation-pinn/core (commit
# 1faa97e0). The module is now a byte-for-byte copy at ../_shared/field_v8r1.py,
# gated by ../_shared/vendoring_parity.py: field values on 200k random in-map
# points are bit-identical to the archive import (max abs diff exactly 0) and
# the map md5 is af284c6954d2273c637a5e766b82b58e.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared.field_v8r1 import FieldV8R1, V8R1_DOWN  # noqa: E402

V8R1_UP = "/cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.up.bin"  # exists, for the MagUp twin later

# ---- constants (Allen conventions, validated in the vertex-fit line) --------
KAPPA = 1.0e-3            # with qop in Allen units
C_QP = 0.299792458        # qop = C_QP * q / p[GeV]
RK4_STEP = 5.0            # mm, the reference integrator's fixed step

# ---- the frozen leg for step 2 (Task C, 2026-07-18): the modal plane pair of
# the 11,046 forward cross-magnet legs in the event-derived training set ------
FROZEN_LEG = dict(z0=2648.2, z1=7826.0)   # last UT plane -> first SciFi plane [mm]

# Overridable via TRAINING_NPZ (used by the One_step_network_v2 rerun on the
# official-sample training set); the default stays the v1 build.
DATA_NPZ = os.environ.get("TRAINING_NPZ") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "Data_generation_exploration", "Data", "training_v1", "train_mb100_v1.npz",
)

FIELD = FieldV8R1()


def deriv(S, z):
    """The LHCb equation of motion, dS/dz for S = (x, y, tx, ty, qop).

    Identical to Allen's ODE and to the training-set label engine
    (Data/make_training_set.py); z may be a scalar or a per-row array.
    """
    x, y, tx, ty, qop = S.T
    if np.isscalar(z):
        z = np.full_like(x, z)
    Bx, By, Bz = FIELD(x, y, z)
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


def rk4_rows(S0, z0, z1, step=RK4_STEP):
    """The fp64 reference integrator: fixed-step RK4, per-row (z0, z1), masked.

    Mirrored from Data/make_training_set.py (single source of truth for the
    training labels; keep in sync).
    """
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
        k1 = deriv(Sa, za)
        k2 = deriv(Sa + 0.5 * ha * k1, za + 0.5 * h[active])
        k3 = deriv(Sa + 0.5 * ha * k2, za + 0.5 * h[active])
        k4 = deriv(Sa + ha * k3, za + h[active])
        S[active] = Sa + (ha / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        z[active] += h[active]
        remaining = (z1 - z) * sign
    return S


def rho(y_ref, y_pred):
    """The agreed scalar relative error (George, 2026-07-13, van der Pol study):
    rho = min(1, ||ref - pred||^2 / ||ref||^2), Euclidean over the four dynamic
    components (x, y, tx, ty), evaluated pointwise."""
    y_ref = np.atleast_2d(y_ref)[:, :4]
    y_pred = np.atleast_2d(y_pred)[:, :4]
    num = ((y_ref - y_pred) ** 2).sum(axis=1)
    den = (y_ref ** 2).sum(axis=1)
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.where(num > 0, np.inf, 0.0))
    return np.minimum(1.0, out)


def load_training(split=None, leg=None, core_only=False):
    """The event-derived training set with its frozen splits.

    split: None | 'train' | 'val' | 'test'   (by-particle, seed 20260718)
    leg:   None | 0..3  (A vertex-fetch, B cross-magnet, C plane-to-plane, D downstream)
    """
    d = np.load(os.path.abspath(DATA_NPZ))
    m = np.ones(len(d["X"]), dtype=bool)
    if split is not None:
        m &= d["SPLIT"] == {"train": 0, "val": 1, "test": 2}[split]
    if leg is not None:
        m &= d["LEG"] == leg
    if core_only:
        m &= d["CORE"] == 1
    return {k: d[k][m] for k in ("X", "Y", "LEG", "P", "PID", "CORE", "SPLIT", "EVT", "MCKEY")}


def card():
    with open(V8R1_DOWN, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return {
        "ode": "x'=tx, y'=ty, tx'=k qop N (tx ty Bx - (1+tx^2) By + ty Bz), "
               "ty'=k qop N ((1+ty^2) Bx - tx ty By - tx Bz), qop'=0;  N=sqrt(1+tx^2+ty^2)",
        "kappa": KAPPA,
        "qop_convention": "qop = 0.299792458 * q / p[GeV]  (Allen units)",
        "field": {"down": V8R1_DOWN, "down_md5": md5, "up": V8R1_UP,
                  "up_exists": os.path.exists(V8R1_UP)},
        "reference_integrator": "fp64 fixed-step RK4, %.1f mm" % RK4_STEP,
        "metric": "rho = min(1, ||dY||^2/||Y||^2) over (x,y,tx,ty); plus tier-1 um bars",
        "frozen_leg_mm": FROZEN_LEG,
        "training_set": os.path.abspath(DATA_NPZ),
        "splits": "by particle, 80/10/10, seed 20260718 (stored in the npz)",
    }


if __name__ == "__main__":
    print(json.dumps(card(), indent=1))
    d = load_training(split="train")
    print("train rows:", len(d["X"]), " per leg:",
          {t: int((d["LEG"] == i).sum()) for i, t in enumerate("ABCD")})
