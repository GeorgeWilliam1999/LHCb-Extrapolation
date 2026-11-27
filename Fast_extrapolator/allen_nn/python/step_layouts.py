"""Plain fixed Nystrom steps (no network) on the VALIDATION tracks: uniform K = 12, 16, 20, 24, 32 and
non-uniform layouts of K = 12 that put more planes where the field changes fastest.

A layout "equal integral of w(z)^a" puts the planes so that each step holds the same integral of
w^a between 2648.2 and 7826 mm; w is a field profile along z: |B| or |dB/dz| on the beam axis, or
their 95 % over the acceptance (x in [-1000, 1000] mm, y in [-1300, 1300] mm), or at y = 1000 mm.

    python allen_nn/python/step_layouts.py      writes results/step_layouts.csv, figures/step_layouts.png
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import an_tools as an  # noqa: E402
import fallback_validation as fv  # noqa: E402
from tail_diagnosis import Field, planes_spec, radial  # noqa: E402

RESULTS = os.path.join(an.FOLDER, "results")
FIGURES = os.path.join(an.FOLDER, "figures")


def profiles(field):
    z = np.linspace(an.Z0, an.Z1, 1036)
    xs, ys = np.meshgrid(np.linspace(-1000, 1000, 11), np.linspace(-1300, 1300, 27))
    xs, ys = xs.ravel(), ys.ravel()
    Bacc = np.stack([field(xs, ys, np.full_like(xs, zz)) for zz in z])  # (nz, npts, 3)
    absB = np.linalg.norm(Bacc, axis=-1)
    dB = np.linalg.norm(np.gradient(Bacc, z, axis=0), axis=-1) * 1e3
    d2B = np.linalg.norm(np.gradient(np.gradient(Bacc, z, axis=0), z, axis=0), axis=-1) * 1e6
    axis = np.argmin(xs ** 2 + ys ** 2)
    y1000 = np.argmin(xs ** 2 + (ys - 1000) ** 2)
    out = {
        "|B| axis": absB[:, axis], "|B| acc95": np.percentile(absB, 95, axis=1), "|B| y1000": absB[:, y1000],
        "|dB/dz| axis": dB[:, axis], "|dB/dz| acc95": np.percentile(dB, 95, axis=1), "|dB/dz| y1000": dB[:, y1000],
        "|d2B/dz2| acc95": np.percentile(d2B, 95, axis=1),
    }
    return z, out


def equal_integral(z, w, K):
    c = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(z))])
    c /= c[-1]
    p = np.interp(np.linspace(0, 1, K + 1), c, z)
    p[0], p[-1] = an.Z0, an.Z1
    return p


def main():
    s, ref, p = fv.validation_tracks()
    field = Field()
    z, prof = profiles(field)
    layouts = []
    for K in (12, 16, 20, 24, 32):
        layouts.append(("uniform K=%d" % K, an.Z0 + np.arange(K + 1) * (an.Z1 - an.Z0) / K))
    for name, w in prof.items():
        for a in (0.25, 0.5, 1.0):
            layouts.append(("K=12 equal int of (%s)^%g" % (name, a), equal_integral(z, np.maximum(w, 1e-6) ** a + 1e-9, 12)))
    # hand-picked: denser across the magnet entrance (3.5-5 m), where the tail is made
    layouts.append(("K=12 hand: 3 steps to 3.5 m, 6 in 3.5-5.2 m, 3 to end",
                    np.concatenate([np.linspace(an.Z0, 3500, 4), np.linspace(3500, 5200, 7)[1:], np.linspace(5200, an.Z1, 4)[1:]])))
    layouts.append(("K=12 hand: 2 / 7 in 3.4-5.6 m / 3",
                    np.concatenate([np.linspace(an.Z0, 3400, 3), np.linspace(3400, 5600, 8)[1:], np.linspace(5600, an.Z1, 4)[1:]])))
    layouts.append(("K=12 hand: 2 / 8 in 3.4-6.6 m / 2",
                    np.concatenate([np.linspace(an.Z0, 3400, 3), np.linspace(3400, 6600, 9)[1:], np.linspace(6600, an.Z1, 3)[1:]])))
    rows = []
    for name, pl in layouts:
        e, _, c = an.run(planes_spec(pl), s, an.Z0, an.Z1)
        r = radial(e, ref)
        row = dict(layout=name, K=len(pl) - 1, reads=float(c[:, 0].mean()), median=np.median(r), p95=np.percentile(r, 95),
                   p99=np.percentile(r, 99), p999=np.percentile(r, 99.9), max=r.max(), over_200um=int((r > 200).sum()),
                   over_1mm=int((r > 1000).sum()), planes=" ".join("%.0f" % v for v in pl))
        rows.append(row)
        print("%-58s reads %5.1f median %6.2f 95%% %6.1f 99%% %7.0f 99.9%% %7.0f max %7.0f >200 %4d >1mm %4d" % (
            name, row["reads"], row["median"], row["p95"], row["p99"], row["p999"], row["max"], row["over_200um"],
            row["over_1mm"]), flush=True)
    with open(os.path.join(RESULTS, "step_layouts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in ("|B| axis", "|B| acc95", "|dB/dz| acc95", "|dB/dz| y1000"):
        ax.plot(z, prof[name] / prof[name].max(), label=name)
    best = min((r for r in rows if r["K"] == 12), key=lambda r: r["p999"])
    for v in best["planes"].split():
        ax.axvline(float(v), color="k", lw=0.6)
    ax.set_title("field profiles (scaled to 1); black: planes of the K=12 layout with the lowest 99.9 %%:\n%s" % best["layout"], fontsize=8)
    ax.set_xlabel("z (mm)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "step_layouts.png"), dpi=130)


if __name__ == "__main__":
    main()
