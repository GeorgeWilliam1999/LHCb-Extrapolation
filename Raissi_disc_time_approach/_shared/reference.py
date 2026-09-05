"""The reference card, generalised: constants, the ODE, the fp64 RK4 engine,
the metric, the tableau and the training set.

This is `Baseline_data_exploration/reference_card.py` with two changes and no
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


def load_training(split=None, leg=None, core_only=False, npz=None):
    """The event-derived training set with its frozen splits.

    split: None | 'train' | 'val' | 'test'   (by-particle, seed 20260718)
    leg:   None | 'A'..'D' | 0..3 | a sequence of either
    """
    d = np.load(os.path.abspath(npz or DATA_NPZ))
    m = np.ones(len(d["X"]), dtype=bool)
    if split is not None:
        m &= d["SPLIT"] == {"train": 0, "val": 1, "test": 2}[split]
    if leg is not None:
        m &= np.isin(d["LEG"], leg_indices(leg))
    if core_only:
        m &= d["CORE"] == 1
    return {k: d[k][m] for k in
            ("X", "Y", "LEG", "P", "PID", "CORE", "SPLIT", "EVT", "MCKEY")}


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
