#!/usr/bin/env python
"""Harvest track states from the simulated-event truth dump.

Input : truth_mb100/{particles,vertices,hits}.csv  (from First_Pass/dump_event.py)
Output: results/states.npz — one row per (particle, detector-plane crossing):
        the true (x, y, tx, ty) read off the MCHit (midpoint position; slopes from
        the entry->exit displacement inside the sensor), plus per-particle truth
        (PDG id, charge, |p| at origin, primary-vertex position) and bookkeeping.

Why this is the training population: these are the actual states of charged
particles inside simulated 2024 LHCb bunch crossings (official Gauss minbias,
nu = 7.6 pile-up) — so momentum spectra, angles, origins and leg geometry are
inherited from real event structure rather than sampled from synthetic boxes.

Run:  /data/bfys/gscriven/conda/envs/TE/bin/python harvest_states.py [truth_dir]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "truth_mb100")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

MIN_SENSOR_DZ = 0.05  # mm; below this the entry->exit displacement carries no slope
TRACKER_DETS = ("VP", "UT", "FT")  # states harvested here (muon/calo: material-heavy)

print("loading truth CSVs from", TRUTH)
parts = pd.read_csv(os.path.join(TRUTH, "particles.csv"))
verts = pd.read_csv(os.path.join(TRUTH, "vertices.csv"))
hits = pd.read_csv(os.path.join(TRUTH, "hits.csv"))
print("  particles %d  vertices %d  hits %d" % (len(parts), len(verts), len(hits)))

# ---- per-particle truth ----------------------------------------------------
parts = parts[parts["charge3"].fillna(0) != 0].copy()  # charged only
parts["p_GeV"] = np.sqrt(parts.px**2 + parts.py**2 + parts.pz**2) / 1000.0
parts["q"] = parts["charge3"] / 3.0
# eta from the origin momentum direction
pt = np.hypot(parts.px, parts.py)
parts["eta"] = np.arcsinh(np.where(pt > 0, parts.pz / np.where(pt > 0, pt, 1), np.inf))

# primary-vertex position per particle (via pv_key -> vertices)
pv = verts[verts.is_primary == 1][["evt", "key", "x", "y", "z"]].rename(
    columns={"key": "pv_key", "x": "pv_x", "y": "pv_y", "z": "pv_z"}
)
parts = parts.merge(pv, on=["evt", "pv_key"], how="left")

# ---- states from tracker MCHits -------------------------------------------
h = hits[hits.det.isin(TRACKER_DETS)].copy()
h["dz"] = h.exit_z - h.entry_z
h = h[np.abs(h.dz) > MIN_SENSOR_DZ]
h["x"] = 0.5 * (h.entry_x + h.exit_x)
h["y"] = 0.5 * (h.entry_y + h.exit_y)
h["z"] = 0.5 * (h.entry_z + h.exit_z)
h["tx"] = (h.exit_x - h.entry_x) / h.dz
h["ty"] = (h.exit_y - h.entry_y) / h.dz
# a hit with |tx|,|ty| beyond 1.0 is a looper/backsplash - outside extrapolator domain
h = h[(np.abs(h.tx) < 1.0) & (np.abs(h.ty) < 1.0)]

h = h.drop(columns=["pid"])  # hits.csv carries its own pid column; use the particle one
st = h.merge(
    parts[
        ["evt", "key", "pid", "q", "p_GeV", "eta", "pv_x", "pv_y", "pv_z", "ox", "oy", "oz"]
    ].rename(columns={"key": "mc_key"}),
    on=["evt", "mc_key"],
    how="inner",
)
st = st.sort_values(["evt", "mc_key", "z"]).reset_index(drop=True)

n_per = st.groupby(["evt", "mc_key"]).size()
print(
    "harvested %d states from %d particles (of %d charged truth particles)"
    % (len(st), len(n_per), len(parts))
)

np.savez_compressed(
    os.path.join(OUT, "states.npz"),
    evt=st.evt.to_numpy(np.int32),
    mc_key=st.mc_key.to_numpy(np.int64),
    det=st.det.astype("category").cat.codes.to_numpy(np.int8),
    det_names=np.array(st.det.astype("category").cat.categories),
    x=st.x.to_numpy(np.float64),
    y=st.y.to_numpy(np.float64),
    z=st.z.to_numpy(np.float64),
    tx=st.tx.to_numpy(np.float64),
    ty=st.ty.to_numpy(np.float64),
    pid=st.pid.to_numpy(np.int64),
    q=st.q.to_numpy(np.float64),
    p_GeV=st.p_GeV.to_numpy(np.float64),
    eta=st.eta.to_numpy(np.float64),
    pv_x=st.pv_x.to_numpy(np.float64),
    pv_y=st.pv_y.to_numpy(np.float64),
    pv_z=st.pv_z.to_numpy(np.float64),
    origin_x=st.ox.to_numpy(np.float64),
    origin_y=st.oy.to_numpy(np.float64),
    origin_z=st.oz.to_numpy(np.float64),
)

summary = {
    "truth_dir": TRUTH,
    "n_hits_in": int(len(hits)),
    "n_states": int(len(st)),
    "n_particles_with_states": int(len(n_per)),
    "n_charged_truth": int(len(parts)),
    "states_per_det": {d: int((st.det == d).sum()) for d in TRACKER_DETS},
    "cuts": {
        "charged_only": True,
        "min_sensor_dz_mm": MIN_SENSOR_DZ,
        "max_abs_slope": 1.0,
        "dets": list(TRACKER_DETS),
    },
}
with open(os.path.join(OUT, "harvest_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
