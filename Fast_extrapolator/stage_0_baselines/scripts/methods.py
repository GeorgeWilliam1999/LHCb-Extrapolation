"""The methods of stage 0: what is scored and timed.

One row a method:

    label       the name in the tables
    family      LHCb, classical or network
    method      the name for the compiled program (cpp/bench/fx.cpp)
    precision   double or float
    reader      how the field is read: linear, cubic (present package), rec (the
                read of LHCb's Detector project), allen (the read of Allen)
    tracks      linear: the test tracks of the linear track set, 2648.2 to 7826 mm
                cubic:  the test tracks of the cubic track set, the same planes
                linear_at_2665: the linear tracks carried to 2665 mm, the first
                plane of the parametrised step of HLT1
                linear_flat: the 1,000 linear tracks that are flat in momentum
                (2 to 42 GeV); only the adaptive integrators have a time that
                depends on the momentum, so only they are timed on it too
    tanh        for a network: library (tanh of the C library) or fast (from
                the vector exponential)
    calls       network calls a track
    weights     multiply-adds of one network call

The field map is v8r1, magnet up, for every row. A copy of an LHCb method runs
in the precision and with the field read of its original. The classical
methods and the networks run in double precision with the reader of the
present package, which is how they were trained and scored, and again in single
precision.
"""
from __future__ import annotations

Z_OURS, Z_UTT, Z_LAST = 2648.2, 2665.0, 7826.0

# The integrators of LHCb. Maarten van Veghel (LHCb, 2026-11-24): the reference
# for HLT1 and HLT2 is the adaptive Runge-Kutta-Nystrom integrator. In Rec that
# is TrackSTEPExtrapolator, in Allen AdaptiveRKNystromExtrapolator; LHCb checks
# the two against each other in a Moore test. The adaptive Cash-Karp integrator
# (TrackRungeKuttaExtrapolator) is what the Kalman fit of Rec v41r3 still
# reaches by default, so it stays in the table.
_ADAPTIVE = [
    ("HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um)", "rec_step", "double", "rec"),
    ("HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um)", "rec_cash_karp",
     "double", "rec"),
    ("HLT1 adaptive Nystrom (Allen, tolerance 5 um)", "allen_adaptive_nystrom", "float", "allen"),
]
LHCB = []
for label, method, precision, reader in _ADAPTIVE:
    LHCB.append(dict(label=label, method=method, precision=precision, reader=reader, tracks="linear"))
    if method.startswith("rec_"):
        LHCB.append(dict(label=label + ", with its transport matrix", method=method + ":jacobian=1",
                         precision=precision, reader=reader, tracks="linear"))
    LHCB.append(dict(label=label + ", flat sample only", method=method, precision=precision,
                     reader=reader, tracks="linear_flat"))
LHCB += [
    dict(label="HLT1 T-track integrator (fixed Nystrom, 500 mm, 1 read a step)",
         method="allen_fast_nystrom", precision="float", reader="allen", tracks="linear"),
    dict(label="HLT1 parametrised step (polynomials, no field read)", method="allen_utt",
         precision="float", reader="linear", tracks="linear_at_2665"),
]

CLASSICAL = []
for steps in (4, 8, 16, 32, 64):
    for of_map in ("linear", "cubic"):
        for precision in ("double", "float"):
            CLASSICAL.append(dict(label="RK4, %d steps, %s map" % (steps, of_map),
                                  method="rk4:%d" % steps, precision=precision, reader=of_map,
                                  tracks=of_map))
    CLASSICAL.append(dict(label="RK4, %d steps, linear map, field read of LHCb" % steps,
                          method="rk4:%d" % steps, precision="float", reader="rec",
                          tracks="linear"))

# The trained runs of the present store, one for each family and step count.
_RUNS = [
    ("Whole-crossing network, 2x256", "whole_crossing:dd9acf59ac88", "linear", 1, None),
    ("Chain, direct states, 64 steps, 2x128", "chain:29510f525130", "cubic", 64, None),
    ("Chain, straight line, global scale, 64 steps, 2x128", "chain:2382b5d314a5", "cubic", 64, None),
    ("Chain, straight line, per-track scale (16 samples), 64 steps, 2x128", "chain:400182ba3233",
     "cubic", 64, None),
    ("Chain, straight line, per-track scale (1 sample), 64 steps, 2x128",
     "chain:400182ba3233:samples=1", "cubic", 64, None),
    ("Physics layer (16 samples), 64 steps, 2x128", "chain:a8a4083032fd", "cubic", 64, None),
    ("Physics layer (1 sample), 64 steps, 2x128", "chain:a8a4083032fd:samples=1", "cubic", 64, None),
    ("Physics layer (1 sample), 32 steps, 2x128", "chain:2e8f364b79e2:samples=1", "cubic", 32, None),
    ("Chain, straight line, per-track scale (1 sample), 64 steps, 2x128", "chain:84306b16e99d:samples=1",
     "linear", 64, None),
    ("Chain, straight line, per-track scale (1 sample), 32 steps, 2x128", "chain:29cbba959300:samples=1",
     "linear", 32, None),
    ("Chain, straight line, per-track scale (1 sample), 16 steps, 2x128", "chain:6c4a3b022a2f:samples=1",
     "linear", 16, None),
    ("Chain, straight line, per-track scale (1 sample), 8 steps, 2x128", "chain:a66c06472d3d:samples=1",
     "linear", 8, None),
    ("Physics layer (1 sample), 64 steps, 2x128", "chain:ffc653fadc1e:samples=1", "linear", 64, None),
    ("Physics layer (1 sample), 64 steps, 2x32", "chain:a0f26d05a973:samples=1", "linear", 64, None),
    ("Physics layer (1 sample), 16 steps, 2x128", "chain:6af35e016a58:samples=1", "linear", 16, None),
]
# Each run three times: as trained and scored (double precision, tanh of the C
# library), in single precision, and in single precision with the fast tanh.
NETWORKS = []
for label, method, of_map, calls, _ in _RUNS:
    for precision, tanh in (("double", "library"), ("float", "library"), ("float", "fast")):
        NETWORKS.append(dict(label=label + ", %s map" % of_map,
                             method=method + (":fast_tanh=1" if tanh == "fast" else ""),
                             precision=precision, reader=of_map, tracks=of_map, calls=calls,
                             tanh=tanh))

METHODS = ([dict(m, family="LHCb") for m in LHCB]
           + [dict(m, family="classical") for m in CLASSICAL]
           + [dict(m, family="network") for m in NETWORKS])


def planes(tracks: str):
    return (Z_UTT, Z_LAST) if tracks == "linear_at_2665" else (Z_OURS, Z_LAST)
