"""Where the fixed steps of the crossing fail (no ARKN fallback), on the VALIDATION tracks only.

    1. error maps: radial end-point error of the tool (K=12 2x64, no fallback) and of plain 12 Nystrom
       steps against the start state; 2D maps (y0 vs ty0, x0 vs tx0) of the median and 99 %.
    2. where along z: the local error of each of the 12 steps (one Nystrom step started on the
       reference state at plane k-1, against the reference at plane k), the step that makes most of
       the error of the worst 100 tracks, its position and the field there (|B|, dBy/dz).
    3. the 1.1 mm track (18.8 GeV, y0 538 mm, ty0 0.21): the reference again with RK4 4,000 and
       16,000 steps, against the stored reference of the track set.

    python allen_nn/python/tail_diagnosis.py    writes results/tail_*.csv, results/tail_diagnosis.txt, figures/tail_*.png
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import an_tools as an  # noqa: E402
import fallback_validation as fv  # noqa: E402

RESULTS = os.path.join(an.FOLDER, "results")
FIGURES = os.path.join(an.FOLDER, "figures")
TOOL = "tool:" + os.path.join(an.WEIGHTS, "linear_nystrom_12_2x64.json") + ":fallback=1e30"
K = 12
PLANES = an.Z0 + np.arange(K + 1) * (an.Z1 - an.Z0) / K
REF = "reference:4000"


class Field:
    """The trilinear v8r1-up map in tesla (grid of the exports, x fastest), for diagnostics."""

    def __init__(self, name="v8r1_up"):
        folder = os.path.join(an.EXPORTS, "field_" + name)
        meta = dict(line.split()[:2] for line in open(os.path.join(folder, "meta")) if len(line.split()) >= 2)
        self.n = [int(meta["nx"]), int(meta["ny"]), int(meta["nz"])]
        self.lo = np.array([float(meta["lowest_corner_" + a]) for a in "xyz"])
        self.h = 1.0 / float(meta["inverse_cell_size_x"])
        g = np.fromfile(os.path.join(folder, "grid.f64")) * float(meta["scale_to_tesla"])
        self.B = g.reshape(3, self.n[2], self.n[1], self.n[0])

    def __call__(self, x, y, z):
        f = (np.stack([x, y, z], -1) - self.lo) / self.h
        i = np.clip(np.floor(f).astype(int), 0, np.array(self.n) - 2)
        t = f - i
        out = 0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (t[..., 0] if dx else 1 - t[..., 0]) * (t[..., 1] if dy else 1 - t[..., 1]) * (t[..., 2] if dz else 1 - t[..., 2])
                    out = out + w[..., None] * self.B[:, i[..., 2] + dz, i[..., 1] + dy, i[..., 0] + dx].T.reshape(w.shape + (3,))
        return out


def planes_spec(p):
    return "planes:" + ":".join("%r" % float(z) for z in p)


def radial(a, b):
    return np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]) * 1e3


def error_maps(s, err, label, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    pairs = [((1, 3), "y0 (mm)", "ty0"), ((0, 2), "x0 (mm)", "tx0")]
    for row, ((a, b), xl, yl) in enumerate(pairs):
        u, v = s[:, a], s[:, b]
        ue = np.linspace(u.min(), u.max(), 21)
        ve = np.linspace(v.min(), v.max(), 21)
        iu = np.clip(np.digitize(u, ue) - 1, 0, 19)
        iv = np.clip(np.digitize(v, ve) - 1, 0, 19)
        for col, (q, name) in enumerate(((50, "median"), (99, "99 %"))):
            m = np.full((20, 20), np.nan)
            for i in range(20):
                for j in range(20):
                    sel = (iu == i) & (iv == j)
                    if sel.sum() >= 5:
                        m[j, i] = np.percentile(err[sel], q)
            ax = axes[row, col]
            im = ax.pcolormesh(ue, ve, np.log10(m), cmap="viridis", vmin=-0.5, vmax=4)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title("%s: %s radial error (cells of >= 5 tracks)" % (label, name), fontsize=9)
            fig.colorbar(im, ax=ax, label="log10(um)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    s, ref, p = fv.validation_tracks()
    n = len(s)
    lines = ["Validation tracks: %d (track set afc61a558211, v8r1 up, 2648.2 -> 7826 mm). No fallback." % n]
    out = {}
    for label, spec in (("tool K=12 2x64", TOOL), ("Nystrom 12", "nystrom:12"), ("Nystrom 32", "nystrom:32"),
                        ("ARKN", "arkn")):
        e, _, c = an.run(spec, s, an.Z0, an.Z1)
        out[label] = radial(e, ref)
        r = out[label]
        lines.append("%-16s median %.2f  95%% %.1f  99%% %.0f  99.9%% %.0f  max %.0f um  >200um %d  >1mm %d  reads %.1f" % (
            label, np.median(r), np.percentile(r, 95), np.percentile(r, 99), np.percentile(r, 99.9), r.max(),
            (r > 200).sum(), (r > 1000).sum(), c[:, 0].mean()))
    tool, ny12 = out["tool K=12 2x64"], out["Nystrom 12"]
    lines.append("tracks > 200 um for the tool: %d, of which Nystrom 12 also > 200 um: %d" % (
        (tool > 200).sum(), ((tool > 200) & (ny12 > 200)).sum()))

    # 1. maps and 1D bands
    error_maps(s, tool, "tool K=12 2x64", os.path.join(FIGURES, "tail_map_tool.png"))
    error_maps(s, ny12, "Nystrom 12", os.path.join(FIGURES, "tail_map_nystrom12.png"))
    y5 = np.abs(s[:, 1] + s[:, 3] * (5000 - an.Z0))
    charge = np.sign(s[:, 4])
    lines.append("\nBy |y| of the straight line at z = 5000 mm (median / 99 % / max of the tool, um):")
    for lo, hi in ((0, 600), (600, 800), (800, 900), (900, 1000), (1000, 1100), (1100, 2000)):
        m = (y5 >= lo) & (y5 < hi)
        lines.append("  %4d-%4d mm: n %5d  tool %.1f / %.0f / %.0f   Nystrom 12 %.1f / %.0f   ARKN %.1f / %.0f   >200um %d" % (
            lo, hi, m.sum(), np.median(tool[m]), np.percentile(tool[m], 99), tool[m].max(), np.median(ny12[m]),
            np.percentile(ny12[m], 99), np.median(out["ARKN"][m]), np.percentile(out["ARKN"][m], 99), (tool[m] > 200).sum()))
    lines.append("By momentum (tool median / 99 %, tracks > 200 um):")
    for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 20), (20, 1e9)):
        m = (p >= lo) & (p < hi)
        lines.append("  p %4.0f-%-4.0f GeV: n %5d  %.1f / %.0f  >200um %d" % (lo, min(hi, 999), m.sum(), np.median(tool[m]),
                                                                       np.percentile(tool[m], 99), (tool[m] > 200).sum()))
    for q in (1, -1):
        m = charge == q
        lines.append("  charge %+d: n %5d  median %.1f 99%% %.0f  >200um %d" % (q, m.sum(), np.median(tool[m]),
                                                                          np.percentile(tool[m], 99), (tool[m] > 200).sum()))
    sign_y = np.sign(s[:, 1] + s[:, 3] * (5000 - an.Z0))
    for q in (1, -1):
        m = (sign_y == q) & (y5 > 800)
        lines.append("  y at 5 m %s, |y|>800: n %4d  >200um %d" % ("> 0" if q > 0 else "< 0", m.sum(), (tool[m] > 200).sum()))

    # 2. the reference at every plane, the error at every plane and the local error of every step
    ref_k = [s.copy()]
    for k in range(1, K + 1):
        e, _, _ = an.run(REF, ref_k[-1], PLANES[k - 1], PLANES[k])
        ref_k.append(e)
    cum = np.zeros((n, K + 1))
    local = np.zeros((n, K))
    for k in range(1, K + 1):
        e, _, _ = an.run(planes_spec(PLANES[:k + 1]), s, an.Z0, an.Z1)
        cum[:, k] = radial(e, ref_k[k])
        one, _, _ = an.run(planes_spec(PLANES[k - 1:k + 1]), ref_k[k - 1], an.Z0, an.Z1)
        local[:, k - 1] = radial(one, ref_k[k])
    lines.append("\nCheck: RK4 4000 per plane chained to z1 vs the stored reference, radial median %.4f max %.3f um" % (
        np.median(radial(ref_k[-1], ref)), radial(ref_k[-1], ref).max()))
    worst = np.argsort(tool)[::-1][:100]
    field = Field()
    zmid = 0.5 * (PLANES[1:] + PLANES[:-1])
    lines.append("\nLocal error of one Nystrom step started on the reference (um): median / 99 % over all tracks, and median over the worst 100:")
    lines.append("  step   z from-to (mm)     all median  all 99%   worst100 median  worst100 max  dominant in worst100")
    dominant = np.argmax(local[worst], axis=1)
    for k in range(K):
        lines.append("  %2d  %6.0f-%6.0f   %9.2f %9.1f %14.1f %13.0f %8d" % (
            k + 1, PLANES[k], PLANES[k + 1], np.median(local[:, k]), np.percentile(local[:, k], 99),
            np.median(local[worst, k]), local[worst, k].max(), (dominant == k).sum()))
    # where the dominant step sits and the field there (sampled along the reference chord)
    rows = []
    for t, k in zip(worst, dominant):
        za, zb = PLANES[k], PLANES[k + 1]
        zs = np.linspace(za, zb, 41)
        f = (zs - za) / (zb - za)
        xs = ref_k[k][t, 0] + f * (ref_k[k + 1][t, 0] - ref_k[k][t, 0])
        ys = ref_k[k][t, 1] + f * (ref_k[k + 1][t, 1] - ref_k[k][t, 1])
        B = field(xs, ys, zs)
        dBy = np.gradient(B[:, 1], zs)
        rows.append([t, p[t], s[t, 0], s[t, 1], s[t, 2], s[t, 3], tool[t], ny12[t], k + 1, za, zb, xs[20], ys[20],
                     np.abs(B).max(), np.linalg.norm(B, axis=1).max(), np.abs(dBy).max() * 1e3, local[t, k],
                     local[t].sum()])
    rows = np.array(rows)
    head = ("track,p_gev,x0,y0,tx0,ty0,tool_um,nystrom12_um,dominant_step,z_from,z_to,x_mid,y_mid,most_abs_Bcomp_T,"
            "most_B_T,most_abs_dBy_dz_T_per_m,local_um_of_dominant,sum_local_um")
    np.savetxt(os.path.join(RESULTS, "tail_worst100.csv"), rows, delimiter=",", header=head, comments="", fmt="%.6g")
    # the same field numbers on a typical (core) track of the same steps
    core = np.where(tool < np.median(tool))[0][:2000]
    Bc = []
    for k in range(K):
        zs = np.linspace(PLANES[k], PLANES[k + 1], 41)
        f = (zs - PLANES[k]) / (PLANES[k + 1] - PLANES[k])
        xs = ref_k[k][core, 0][:, None] + f * (ref_k[k + 1][core, 0] - ref_k[k][core, 0])[:, None]
        ys = ref_k[k][core, 1][:, None] + f * (ref_k[k + 1][core, 1] - ref_k[k][core, 1])[:, None]
        B = field(xs, ys, np.broadcast_to(zs, xs.shape))
        dBy = np.gradient(B[..., 1], zs, axis=1)
        Bc.append((np.median(np.linalg.norm(B, axis=-1).max(1)), np.median(np.abs(dBy).max(1)) * 1e3))
    lines.append("\nWorst 100 tracks (results/tail_worst100.csv): dominant step, its middle (x, y), the largest |B| and |dBy/dz| on it:")
    for k in sorted(set(dominant)):
        m = rows[:, 8] == k + 1
        lines.append("  step %2d (z %4.0f-%4.0f): %3d tracks; |y_mid| %s mm; |x_mid| %s mm; |B| max %s T; |dBy/dz| max %s T/m;"
                     " a core track on that step: |B| %.2f T, |dBy/dz| %.2f T/m" % (
                         k + 1, PLANES[k], PLANES[k + 1], m.sum(),
                         np.round(np.percentile(np.abs(rows[m, 12]), [0, 50, 100])).astype(int).tolist(),
                         np.round(np.percentile(np.abs(rows[m, 11]), [0, 50, 100])).astype(int).tolist(),
                         np.round(np.percentile(rows[m, 14], [0, 50, 100]), 2).tolist(),
                         np.round(np.percentile(rows[m, 15], [0, 50, 100]), 2).tolist(), Bc[k][0], Bc[k][1]))
    lines.append("  share of the worst-100 summed local error made by the dominant step: median %.2f" % np.median(rows[:, 16] / rows[:, 17]))

    # field along z at x = 0 and at the y of the tail, for the figure
    zz = np.linspace(2000, 8500, 651)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for yv in (0, 400, 800, 1000, 1200):
        B = field(np.zeros_like(zz), np.full_like(zz, yv), zz)
        ax[0].plot(zz, B[:, 1], label="y = %d mm" % yv)
        ax[1].plot(zz, np.gradient(B[:, 1], zz) * 1e3, label="y = %d mm" % yv)
    for a in ax:
        for zp in PLANES:
            a.axvline(zp, color="0.8", lw=0.6)
        a.legend(fontsize=7)
    ax[0].set_ylabel("By (T), x = 0")
    ax[1].set_ylabel("dBy/dz (T/m)")
    ax[1].set_xlabel("z (mm); grey: the 12 step planes")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "tail_field_profile.png"), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(np.arange(1, K + 1), np.median(local, 0), "o-", label="all tracks, median")
    ax.semilogy(np.arange(1, K + 1), np.percentile(local, 99, 0), "o-", label="all tracks, 99 %")
    ax.semilogy(np.arange(1, K + 1), np.median(local[worst], 0), "o-", label="worst 100, median")
    ax.set_xlabel("step (12 equal steps, 2648 -> 7826 mm)")
    ax.set_ylabel("local radial error of one step (um)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "tail_local_error_by_step.png"), dpi=130)
    plt.close(fig)

    # 3. the 1.1 mm track
    t = int(np.argmin(np.abs(p - 18.81) + np.abs(s[:, 1] - 537.6)))
    lines.append("\nTrack %d: p %.2f GeV, start %s" % (t, p[t], np.round(s[t], 4).tolist()))
    one = s[t:t + 1]
    for spec in ("reference:4000", "reference:16000", "reference:64000", "arkn", "nystrom:12", "nystrom:32", "nystrom:128", TOOL):
        e, _, _ = an.run(spec, one, an.Z0, an.Z1, precision="double" if spec.startswith("reference") else "float")
        lines.append("  %-28s end x %.4f y %.4f mm; radial vs stored reference %.2f um" % (
            spec.split("/")[-1], e[0, 0], e[0, 1], radial(e, ref[t:t + 1])[0]))
    lines.append("  local error by step (um): %s" % np.round(local[t], 1).tolist())
    lines.append("  cumulative error at each plane (um): %s" % np.round(cum[t], 1).tolist())

    text = "\n".join(lines)
    open(os.path.join(RESULTS, "tail_diagnosis.txt"), "w").write(text + "\n")
    np.save(os.path.join(an.SCRATCH, "tail_local.npy"), local)
    print(text)


if __name__ == "__main__":
    main()
