"""Tables and figures of the experiment, from the arrays decompose.py cached.

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python summarise.py

Writes results/per_run.csv, results/per_band.csv and figures/*.png beside this file.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "/data/bfys/gscriven/rkpinn_analysis_cache/how_does_the_stage_residual_become_the_endpoint_error"
SCALE = np.array([1.0, 1.0, 1e3, 1e3])          # mm and mrad, as decompose.py
UM = 1e3
RUNS = [("straight line, per-track scale", "25dfa17ef168"),
        ("straight line, global scale", "9cef9c160b7d"),
        ("direct states", "49e5e2450eaa"),
        ("Broyden, straight line", "166cfc8cdfd6"),
        ("pilot, summed", "bddea42ea1ef"),
        ("pilot, predicted", "4cc888dcc710")]
STEP_LENGTH_RUNS = ["a66c06472d3d", "6c4a3b022a2f", "29cbba959300", "25dfa17ef168",
                    "15242aa17c65", "eeb809e9254d"]
BANDS = [("< 3 GeV", 0, 3), ("3-8 GeV", 3, 8), ("8-20 GeV", 8, 20), ("20-50 GeV", 20, 50),
         ("> 50 GeV", 50, np.inf)]


def norm(a):
    """Norm over the last axes after the step axis, in mm and mrad."""
    a = a * SCALE
    return np.linalg.norm(a.reshape(a.shape[0], a.shape[1], -1), axis=-1)


def load(key):
    return dict(np.load(os.path.join(CACHE, "%s.npz" % key)))


def coherence(c):
    return np.abs(c.sum(axis=1)) / np.sqrt((c ** 2).sum(axis=1))


def summary(name, key, d):
    end = d["states"][:, -1, :4]
    radial = lambda e: np.hypot(e[:, 0], e[:, 1]) * UM
    steps = d["delta"].shape[1]
    eps, r, eps1 = norm(d["eps"]), norm(d["r"]), norm(d["eps1"])
    delta = np.linalg.norm(d["delta"] * SCALE, axis=-1)
    delta1 = np.linalg.norm((d["delta"] - d["delta1"]) * SCALE, axis=-1)
    ec, lc = d["exact_carried"], d["linear_carried"][: len(d["exact_carried"])]
    rel_sum = [np.median(np.abs(lc[..., i].sum(1) - ec[..., i].sum(1)) / np.abs(ec[..., i].sum(1)))
               for i in range(2)]
    return {
        "run": name, "key": key, "steps": steps, "step_length_mm": float(d["step_length_mm"]),
        "end_state": str(d["end_state"]), "round": int(d["round"]),
        "endpoint_against_reference_um": float(np.median(radial(end - d["reference_end"]))),
        "endpoint_against_exact_chain_um": float(np.median(radial(end - d["exact_chain_end"]))),
        "exact_chain_against_reference_um": float(np.median(radial(d["exact_chain_end"] - d["reference_end"]))),
        "median_residual": float(np.median(r)), "median_stage_error": float(np.median(eps)),
        "median_stage_error_xy_um": float(np.median(np.linalg.norm(d["eps"][..., :2].reshape(len(eps), steps, -1), axis=-1)) * UM),
        "c1_eps_vs_first_order": float(np.median(norm(d["eps"] - d["eps1"]) / eps)),
        "c2_eps_vs_residual": float(np.median(norm(d["eps"] - d["r"]) / eps)),
        "c3_delta_vs_first_order": float(np.median(delta1 / delta)),
        "c6_delta_over_first_order": float(np.median(delta / np.linalg.norm(d["delta1"] * SCALE, axis=-1))),
        "median_defect": float(np.median(delta)),
        "median_defect_xy_um": float(np.median(np.linalg.norm(d["delta"][..., :2], axis=-1)) * UM),
        "c4_corr_x": float(np.corrcoef(lc[..., 0].ravel(), ec[..., 0].ravel())[0, 1]),
        "c4_corr_y": float(np.corrcoef(lc[..., 1].ravel(), ec[..., 1].ravel())[0, 1]),
        "c4_sum_rel_x": float(rel_sum[0]), "c4_sum_rel_y": float(rel_sum[1]),
        "c5_coherence_x": float(np.median(coherence(ec[..., 0]))),
        "c5_coherence_y": float(np.median(coherence(ec[..., 1]))),
        "c5_limit": float(np.sqrt(steps) / 2),
        "worst_exact_residual": float(d["worst_exact_residual"]),
    }


def per_band(name, key, d):
    rows = []
    p = d["momentum_gev"]
    end = d["states"][:, -1, :4]
    e = np.hypot(*((end - d["reference_end"])[:, :2].T)) * UM
    eps = norm(d["eps"])
    for label, lo, hi in BANDS:
        m = (p >= lo) & (p < hi)
        mc = m[: len(d["exact_carried"])]
        rows.append({"run": name, "key": key, "band": label, "tracks": int(m.sum()),
                     "endpoint_um": float(np.median(e[m])),
                     "median_stage_error": float(np.median(eps[m])),
                     "coherence_x": float(np.median(coherence(d["exact_carried"][mc][..., 0])))
                     if mc.sum() > 2 else float("nan")})
    return rows


def write(file, rows):
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", file), "w", newline="") as handle:
        w = csv.DictWriter(handle, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    have = {k for k in [k for _, k in RUNS] + STEP_LENGTH_RUNS
            if os.path.exists(os.path.join(CACHE, "%s.npz" % k))}
    rows, bands = [], []
    names = dict(RUNS)
    names = {k: n for n, k in RUNS}
    for key in STEP_LENGTH_RUNS:
        names.setdefault(key, "step length, straight line per-track")
    for key, name in names.items():
        if key not in have:
            continue
        d = load(key)
        rows.append(summary(name, key, d))
        bands += per_band(name, key, d)
    write("per_run.csv", rows)
    write("per_band.csv", bands)
    # figure: step length against stage error, defect and endpoint error
    sl = sorted([r for r in rows if r["key"] in STEP_LENGTH_RUNS], key=lambda r: r["step_length_mm"])
    if len(sl) >= 3:
        dz = np.array([r["step_length_mm"] for r in sl])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for label, k in [("stage error, x and y", "median_stage_error_xy_um"),
                         ("local defect, x and y", "median_defect_xy_um"),
                         ("endpoint against the exact chain", "endpoint_against_exact_chain_um")]:
            v = np.array([r[k] for r in sl])
            slope = np.polyfit(np.log(dz), np.log(v), 1)[0]
            ax[0].loglog(dz, v, "o-", label="%s (slope %.2f)" % (label, slope))
        ax[0].set_xlabel("step length (mm)")
        ax[0].set_ylabel("median (µm)")
        ax[0].legend(fontsize=8)
        ax[1].semilogx(dz, [r["c5_coherence_x"] for r in sl], "o-", label="measured, x")
        ax[1].semilogx(dz, [r["c5_coherence_y"] for r in sl], "s-", label="measured, y")
        ax[1].semilogx(dz, [np.sqrt(r["steps"]) for r in sl], "k--", label="√N: one sign, equal sizes")
        ax[1].semilogx(dz, [0.67] * len(sl), "k:", label="0.67: random signs (median)")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("step length (mm)")
        ax[1].set_ylabel("coherence ratio")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "figures", "step_length.png"), dpi=150)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
