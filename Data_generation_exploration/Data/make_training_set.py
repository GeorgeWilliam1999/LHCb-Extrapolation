#!/usr/bin/env python
"""Build the extrapolation training set from harvested event states.

Population : real track states harvested from simulated 2024 minbias crossings
             (results/states.npz, see harvest_states.py).
Labels     : fp64 RK4 (5 mm fixed step) through the canonical v8r1.down field
             map — the same ODE/engine convention validated against extrapUTT
             at the 15 um level in the track-extrapolation project. Field-only
             by design: material effects are the master extrapolator's job,
             not the surrogate's (project stance, vertex-fit line).

Leg types (all endpoints from the particle's OWN event geometry):
    A  first-VP-state -> backward to its primary vertex z      (vertex fetch)
    B  last pre-magnet state (z<2800) -> first T state (z>7000) + the reverse
                                                              (cross-magnet)
    C  consecutive plane crossings, forward                    (short in-tracker)
    D  first T state -> backward to its primary vertex z       (downstream leg)

Row format (matches the vertex-fit corpus contract):
    X[N,7] = (x, y, tx, ty, qop, z0, z1)   qop = 0.299792458 * q / p[GeV]
    Y[N,5] = (x', y', tx', ty', qop)       qop passthrough exact
plus LEG[N] (A=0,B=1,C=2,D=3), EVT[N], MCKEY[N], PID[N], P[N] and a
train/val/test split BY PARTICLE (no particle contributes to two splits).

Gates (results/gates.json):
    G1 re-propagation closure (forward then back, 2000 rows)
    G2 label vs reality: type-C predictions compared with the ACTUAL next MCHit
       (residual = multiple scattering + energy loss, i.e. exactly the material
       effects that are out of label scope — small at high p, growing at low p)
    G3 step convergence: 5 mm vs 1 mm step on 200 random rows
    G4 qop passthrough exact

Run:  /data/bfys/gscriven/conda/envs/TE/bin/python make_training_set.py
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_CORE = "/data/bfys/gscriven/Track_Extrapolation_work_archive/track-extrapolation-pinn/core"
sys.path.insert(0, ARCHIVE_CORE)  # read-only import of the canonical field loader
from field_v8r1 import FieldV8R1, V8R1_DOWN  # noqa: E402

# CLI (all optional, defaults = the v1 build): states.npz path, output dir, tag,
# and the key of the sample description recorded in the meta json (SAMPLES below).
STATES = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "results", "states.npz")
OUT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(HERE, "training_v1")
TAG = sys.argv[3] if len(sys.argv) > 3 else "train_mb100_v1"
SAMPLE_KEY = sys.argv[4] if len(sys.argv) > 4 else "gauss_mb100"

# Where the events came from. One entry per input sample; the meta json records
# the entry named by SAMPLE_KEY, so a rebuild on a different sample cannot
# inherit the previous sample's description (that mistake was made once, and
# corrected on 2026-09-05).
SAMPLES = {
    # v1: our own Gauss production (superseded as a provenance source by George's
    # 2026-07-21 directive to use centrally produced samples).
    "gauss_mb100": {
        "origin": "self-generated (Gauss run locally at Nikhef)",
        "generator": "Gauss v61r0p2 (Gauss-on-Gaussino), event type 30000000 minbias",
        "conditions": "2024 Block-7 beam, nu=7.6, geometry run3/2024-v00.02, "
                      "conditions sim10/2024, DD4hep",
        "sim_file": "First_Pass/run_output/GaussMB100-30000000-100ev-20260717.sim",
        "n_events": 100,
        "run_number": 1,
        "event_numbers": "1-100 (seeds reproducible)",
        "truth_dump": "First_Pass/dump_event.py -> Data/truth_mb100/",
    },
    # v2: the official TestFileDB sample, read from the local CVMFS mirror.
    "official_xdigi": {
        "origin": "official central production (LHCb TestFileDB), not self-generated",
        "testfiledb_entry": "expected_2024_minbias_xdigi (PRConfig)",
        "production": "00212966",
        "format": "XDIGI (digitised banks + packed MC truth pSim/...)",
        "conditions": "Simulation; DDDB dddb-20231017, CondDB sim-20231017-vc-mu100, DataType 2024",
        "input_files": [
            "/cvmfs/lhcbdev.cern.ch/testfiledb-mirror/lhcb/swtest/"
            "expected_2024_minbias_xdigi/00212966_000000%s_1.xdigi" % n
            for n in ("13", "18", "26", "31", "86")
        ],
        "access": "local CVMFS mirror /cvmfs/lhcbdev.cern.ch/testfiledb-mirror/ "
                  "(5 of the entry's 8 files, 19 GB) - no EOS, kerberos or DIRAC",
        "n_events": 200,
        "event_numbers": "0-199, the 0-based index in the concatenated file stream",
        "truth_dump": "Official_xdigi/dump_xdigi.py (bash run_dump.sh truth_official 200) "
                      "-> Official_xdigi/truth_official/",
    },
}
assert SAMPLE_KEY in SAMPLES, "unknown sample key %r (have %s)" % (SAMPLE_KEY, sorted(SAMPLES))
RES = os.path.dirname(STATES)
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)

KAPPA = 1.0e-3          # with qop in Allen units (0.299792458 * q/p[GeV])
C_QP = 0.299792458
STEP = 5.0              # mm, fixed-step RK4 (validated convention)
P_MIN, P_MAX = 1.0, 200.0   # GeV — the deployed extrapolator domain
MAX_C_LEGS = 3          # per particle
SPLIT_SEED = 20260718
SPLITS = (0.8, 0.1, 0.1)

FIELD = FieldV8R1()


def deriv(S, z):
    """Identical ODE to Allen / generate_data_v2 (per-row z array)."""
    x, y, tx, ty, qop = S.T
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


def rk4_rows(S0, z0, z1, step=STEP):
    """Vectorised fixed-step RK4 with PER-ROW (z0, z1): masked stepping."""
    S = S0.astype(np.float64).copy()
    z = z0.astype(np.float64).copy()
    sign = np.sign(z1 - z0)
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


# ---------------------------------------------------------------- load states
d = np.load(STATES, allow_pickle=False)
n = len(d["z"])
keys = d["evt"].astype(np.int64) * 10_000_000 + d["mc_key"]
order = np.lexsort((d["z"], keys))
K, Z = keys[order], d["z"][order]
X_, Y_, TX_, TY_ = d["x"][order], d["y"][order], d["tx"][order], d["ty"][order]
PID, Q, P = d["pid"][order], d["q"][order], d["p_GeV"][order]
PVZ = d["pv_z"][order]
ORX, ORY = d["origin_x"][order], d["origin_y"][order]
ETA = d["eta"][order]
EVT, MCK = d["evt"][order], d["mc_key"][order]

dom = (P >= P_MIN) & (P <= P_MAX)
uniq, starts = np.unique(K, return_index=True)
print("particles in domain contributing:", len(uniq))

rng = np.random.default_rng(SPLIT_SEED)
rows = []  # (state5, z0, z1, leg, evt, mckey, pid, p [, aux next-hit for C])
c_truth = []  # for G2: (row_index, true next-hit x, y, tx, ty)

for i0, i1 in zip(starts, np.append(starts[1:], len(K))):
    if not dom[i0]:
        continue
    idx = np.arange(i0, i1)
    if len(idx) < 2:
        continue
    qop = C_QP * Q[i0] / P[i0]
    ev, mk, pid, p = EVT[i0], MCK[i0], PID[i0], P[i0]
    orr = float(np.hypot(ORX[i0], ORY[i0]))
    eta0 = float(ETA[i0])

    def state(j):
        return [X_[j], Y_[j], TX_[j], TY_[j], qop]

    # A: first state backward to the primary vertex z
    j = idx[0]
    if np.isfinite(PVZ[j]) and Z[j] - PVZ[j] > 5.0:
        rows.append((state(j), Z[j], PVZ[j], 0, ev, mk, pid, p, -1, orr, eta0))
    # B: cross-magnet, both directions
    pre = idx[Z[idx] < 2800]
    post = idx[Z[idx] > 7000]
    if len(pre) and len(post):
        ja, jb = pre[-1], post[0]
        rows.append((state(ja), Z[ja], Z[jb], 1, ev, mk, pid, p, -1, orr, eta0))
        rows.append((state(jb), Z[jb], Z[ja], 1, ev, mk, pid, p, -1, orr, eta0))
    # C: consecutive crossings (forward), sampled; record the true next hit
    if len(idx) >= 2:
        pairs = np.arange(len(idx) - 1)
        take = rng.choice(pairs, size=min(MAX_C_LEGS, len(pairs)), replace=False)
        for t in take:
            ja, jb = idx[t], idx[t + 1]
            if Z[jb] - Z[ja] < 2.0:
                continue
            rows.append((state(ja), Z[ja], Z[jb], 2, ev, mk, pid, p, len(c_truth), orr, eta0))
            c_truth.append([X_[jb], Y_[jb], TX_[jb], TY_[jb]])
    # D: first T-station state backward to the vertex band
    if len(post) and np.isfinite(PVZ[i0]):
        j = post[0]
        rows.append((state(j), Z[j], PVZ[j], 3, ev, mk, pid, p, -1, orr, eta0))

print("legs built:", len(rows))
S0 = np.array([r[0] for r in rows], dtype=np.float64)
z0 = np.array([r[1] for r in rows])
z1 = np.array([r[2] for r in rows])
LEG = np.array([r[3] for r in rows], dtype=np.int8)
EVTr = np.array([r[4] for r in rows], dtype=np.int32)
MCKr = np.array([r[5] for r in rows], dtype=np.int64)
PIDr = np.array([r[6] for r in rows], dtype=np.int64)
Pr = np.array([r[7] for r in rows])
CIX = np.array([r[8] for r in rows], dtype=np.int64)
ORR = np.array([r[9] for r in rows])
ETAr = np.array([r[10] for r in rows])
CT = np.array(c_truth) if c_truth else np.empty((0, 4))

t0 = time.time()
S1 = rk4_rows(S0, z0, z1)
dt = time.time() - t0
print("RK4 done: %d rows in %.1f s (%.0f rows/s)" % (len(S0), dt, len(S0) / dt))

ok = np.isfinite(S1).all(axis=1) & (np.abs(S1[:, 0]) < 5000) & (np.abs(S1[:, 1]) < 5000)
print("finite+in-aperture rows: %d / %d" % (ok.sum(), len(ok)))

X = np.column_stack([S0, z0, z1])[ok].astype(np.float32)
Y = S1[ok].astype(np.float32)
LEG, EVTr, MCKr, PIDr, Pr, CIX, ORR, ETAr = (a[ok] for a in (LEG, EVTr, MCKr, PIDr, Pr, CIX, ORR, ETAr))
S0k, S1k, z0k, z1k = S0[ok], S1[ok], z0[ok], z1[ok]

# ------------------------------------------------------------------ gates
gates = {}
sub = np.random.default_rng(1).choice(len(S0k), size=min(2000, len(S0k)), replace=False)
back = rk4_rows(S1k[sub], z1k[sub], z0k[sub])
clos = np.abs(back[:, :2] - S0k[sub, :2]).max(axis=1)
gates["G1_reprop_closure_mm"] = {
    "median": float(np.median(clos)), "worst": float(clos.max()), "n": int(len(sub)),
}

cmask = (LEG == 2) & (CIX >= 0)
resid = np.abs(S1k[cmask][:, :2] - CT[CIX[cmask]][:, :2]).max(axis=1)
pC = Pr[cmask]
gates["G2_label_vs_next_hit_mm"] = {
    "median": float(np.median(resid)),
    "p95": float(np.quantile(resid, 0.95)),
    "median_p_gt_5GeV": float(np.median(resid[pC > 5])) if (pC > 5).any() else None,
    "median_p_lt_2GeV": float(np.median(resid[pC < 2])) if (pC < 2).any() else None,
    "n": int(cmask.sum()),
    "meaning": "residual = material effects (scattering/dE), out of label scope by design",
}
np.savez_compressed(
    os.path.join(RES, "g2_label_vs_next_hit.npz"),
    resid_mm=resid, p_GeV=pC, dz_mm=(z1k - z0k)[cmask],
    det_from=np.zeros(int(cmask.sum())),
)

s2 = np.random.default_rng(2).choice(len(S0k), size=min(200, len(S0k)), replace=False)
fine = rk4_rows(S0k[s2], z0k[s2], z1k[s2], step=1.0)
conv = np.abs(fine[:, :2] - S1k[s2, :2]).max(axis=1)
gates["G3_step_convergence_mm_5vs1"] = {
    "median": float(np.median(conv)), "worst": float(conv.max()), "n": int(len(s2)),
}
gates["G4_qop_passthrough_exact"] = bool(np.all(S1k[:, 4] == S0k[:, 4]))

# ------------------------------------------------------- split by particle
pkey = EVTr.astype(np.int64) * 10_000_000 + MCKr
up = np.unique(pkey)
perm = np.random.default_rng(SPLIT_SEED).permutation(len(up))
ntr = int(SPLITS[0] * len(up)); nva = int(SPLITS[1] * len(up))
lab = np.zeros(len(up), dtype=np.int8)
lab[perm[ntr:ntr + nva]] = 1
lab[perm[ntr + nva:]] = 2
SPLIT = lab[np.searchsorted(up, pkey)]

np.savez_compressed(
    os.path.join(OUT, TAG + ".npz"),
    X=X, Y=Y, LEG=LEG, EVT=EVTr, MCKEY=MCKr, PID=PIDr, P=Pr.astype(np.float32),
    SPLIT=SPLIT,
    ORIGIN_R=ORR.astype(np.float32), ETA=ETAr.astype(np.float32),
    CORE=((np.abs(PIDr) != 11) & (ORR < 10.0) & (ETAr > 1.8) & (ETAr < 5.2)).astype(np.int8),
)

def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

archive_commit = subprocess.run(
    ["git", "-C", ARCHIVE_CORE, "rev-parse", "HEAD"],
    capture_output=True, text=True,
).stdout.strip()

meta = {
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "sample": dict(SAMPLES[SAMPLE_KEY], key=SAMPLE_KEY),
    "population": "states harvested from tracker MCHits (see harvest_states.py + results/harvest_summary.json)",
    "labels": {
        "engine": "fp64 fixed-step RK4, step_mm=%.1f, ODE identical to Allen (deriv mirrored from track-extrapolation-pinn/datagen/generate_data_v2.py)" % STEP,
        "field": {"file": V8R1_DOWN, "md5": md5(V8R1_DOWN)},
        "kappa": KAPPA, "qop_convention": "qop = 0.299792458 * q / p[GeV] (Allen)",
        "material_effects": "excluded by design (master-extrapolator responsibility)",
        "archive_core_commit": archive_commit,
    },
    "leg_types": {"A": "first VP state -> its PV z (backward)",
                  "B": "cross-magnet, both directions",
                  "C": "consecutive plane crossings (forward), <=3/particle",
                  "D": "first T state -> its PV z (backward)"},
    "domain_cuts": {"p_GeV": [P_MIN, P_MAX], "slopes": "<1.0", "aperture_mm": 5000},
    "core_selection": "non-electron & origin radius < 10 mm & 1.8 < eta < 5.2 (flag CORE; tails kept, selectable)",
    "rows": {"total": int(len(X)),
             "per_leg": {t: int((LEG == i).sum()) for i, t in enumerate("ABCD")},
             "split": {"train": int((SPLIT == 0).sum()), "val": int((SPLIT == 1).sum()),
                        "test": int((SPLIT == 2).sum()), "by": "particle", "seed": SPLIT_SEED}},
    "gates": gates,
    "format": {"X": "(x,y,tx,ty,qop,z0,z1) fp32", "Y": "(x,y,tx,ty,qop) at z1, fp32",
               "note": "labels computed in fp64, stored fp32 (same as vertex-fit corpus)"},
}
with open(os.path.join(OUT, TAG + ".meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
with open(os.path.join(RES, "gates.json"), "w") as f:
    json.dump(gates, f, indent=1)
print(json.dumps(gates, indent=1))
print("training set:", os.path.join(OUT, TAG + ".npz"), "rows:", len(X))
