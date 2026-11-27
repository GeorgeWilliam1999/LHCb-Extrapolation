"""The data of stage 3 (the corner of the magnet crossing), on a login node, once.

The corner: tracks whose straight line from the state on z0 = 2648.2 mm reaches |y| > 600 mm at
z = 5000 mm (the region where 12 fixed Nystrom steps fail, allen_nn/results/tail_diagnosis.txt).
The networks are applied above 700 mm (allen_nn corner rule); 600 mm gives a margin.

Writes <store>/data/<split>/{start_state,reference_end_state,start_nystrom_12,momentum_gev,abs_y_at_5m}.f64
for the splits
    training            all 108,915 training tracks of track set afc61a558211 (linear map v8r1 up)
    validation          all 13,684 validation tracks
    corner_training     the 9,791 corner training tracks + EXTRA new corner tracks (see below)
    corner_validation   the 1,277 corner validation tracks (real ones only)
The test tracks are not read.

Extra corner tracks: a start state of a real corner training track, picked at random, plus a
Gaussian change of each of its five components (sigma = JITTER times the spread of that component
in the corner training set); kept when it is still in the corner and inside the box of the
training tracks. Their reference end state is RK4 with 4,000 steps in double on the same map and
planes (an "reference:4000"; it equals the reference of the track set to 0.001 um at most, see
allen_nn/results/tail_diagnosis.txt). The start of every track (12 Nystrom steps, float) is that of
the tool (an "nystrom:12"), so that training and deployment use the same start.

    python stage_3_corner/python/make_corner_data.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "allen_nn", "python"))
import an_tools as an  # noqa: E402

STORE = "/data/bfys/gscriven/fast_extrapolator_store/stage_3_corner"
SPLITS = os.path.join(an.EXPORTS, "tracks_linear", "splits")
CORNER_Y_MM, CORNER_Z_MM = 600.0, 5000.0
EXTRA = 90000
JITTER = 0.05
SEED = 20261127
P_TIMES_QOP = 0.299792458  # p (GeV) * |state q/p| (q/p * c_light, 1/MeV)


def abs_y5(s):
    return np.abs(s[:, 1] + s[:, 3] * (CORNER_Z_MM - an.Z0))


def read(name):
    f = lambda a: os.path.join(SPLITS, name, a)  # noqa: E731
    return (np.fromfile(f("start_state.f64")).reshape(-1, 5), np.fromfile(f("reference_end_state.f64")).reshape(-1, 5),
            np.fromfile(f("momentum_gev.f64")))


def write(name, s, ref, p):
    folder = os.path.join(STORE, "data", name)
    os.makedirs(folder, exist_ok=True)
    start, _, _ = an.run("nystrom:12", s, an.Z0, an.Z1)
    for key, value in (("start_state", s), ("reference_end_state", ref), ("start_nystrom_12", start[:, :4]),
                       ("momentum_gev", p), ("abs_y_at_5m", abs_y5(s))):
        np.ascontiguousarray(value, dtype=np.float64).tofile(os.path.join(folder, key + ".f64"))
    return dict(tracks=int(len(s)), corner_above_600=int((abs_y5(s) > 600).sum()),
                corner_above_700=int((abs_y5(s) > 700).sum()))


def main():
    rng = np.random.default_rng(SEED)
    ts, tr, tp = read("training")
    vs, vr, vp = read("validation")
    box = np.abs(ts).max(axis=0)
    meta = dict(track_set="afc61a558211", map="linear v8r1 up", planes=[an.Z0, an.Z1], corner_y_mm=CORNER_Y_MM,
                corner_z_mm=CORNER_Z_MM, extra=EXTRA, jitter=JITTER, seed=SEED, box_of_training=box.tolist())
    meta["training"] = write("training", ts, tr, tp)
    meta["validation"] = write("validation", vs, vr, vp)
    c = abs_y5(ts) > CORNER_Y_MM
    cs, cr, cp = ts[c], tr[c], tp[c]
    spread = cs.std(axis=0)
    made = []
    while sum(len(m) for m in made) < EXTRA:
        pick = rng.integers(0, len(cs), 4 * EXTRA)
        new = cs[pick] + rng.normal(size=(len(pick), 5)) * spread * JITTER
        new[:, 4] = np.where(np.sign(new[:, 4]) == np.sign(cs[pick, 4]), new[:, 4], cs[pick, 4])  # keep the charge
        keep = (abs_y5(new) > CORNER_Y_MM) & np.all(np.abs(new) <= box, axis=1)
        made.append(new[keep])
    new = np.concatenate(made)[:EXTRA]
    new_ref, _, _ = an.run("reference:4000", new, an.Z0, an.Z1, precision="double")
    new_p = P_TIMES_QOP / np.abs(new[:, 4])
    meta["corner_training"] = write("corner_training", np.concatenate([cs, new]), np.concatenate([cr, new_ref]),
                                    np.concatenate([cp, new_p]))
    meta["corner_training"]["real"] = int(len(cs))
    meta["corner_training"]["extra"] = int(len(new))
    c = abs_y5(vs) > CORNER_Y_MM
    meta["corner_validation"] = write("corner_validation", vs[c], vr[c], vp[c])
    # a check of the reference of the extra tracks: the same RK4 on the real corner tracks
    check, _, _ = an.run("reference:4000", cs[:2000], an.Z0, an.Z1, precision="double")
    meta["check_reference_rk4_4000_vs_track_set_um"] = dict(
        median=float(np.median(np.hypot(*(check[:, :2] - cr[:2000, :2]).T) * 1e3)),
        largest=float(np.hypot(*(check[:, :2] - cr[:2000, :2]).T).max() * 1e3))
    json.dump(meta, open(os.path.join(STORE, "data", "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
