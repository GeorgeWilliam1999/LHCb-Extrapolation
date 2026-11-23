"""Where and how A1 fails: the figures and the tables, from the arrays of compute.py.

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python analyse.py

Reads CACHE and the constants of the runs. Writes figures/ and tables/ here.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

CACHE = os.environ.get("CACHE", "/data/bfys/gscriven/rkpinn_analysis_cache/failure_of_A1")
STORE = "/data/bfys/gscriven/rkpinn_store"
HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES, TABLES = os.path.join(HERE, "figures"), os.path.join(HERE, "tables")
KEYS = {"A1": "3ccd44d61cd8", "A2": "0b41ecdd2505"}
COLOURS = {"A1": "#1f6fb4", "A2": "#d9711c", "exact": "#555555", "P1": "#2a9d5c"}
COMPONENTS = [("x", 1e3, "µm"), ("y", 1e3, "µm"), ("tx", 1e6, "µrad"), ("ty", 1e6, "µrad")]
BANDS = [(0, 3), (3, 8), (8, 20), (20, 50), (50, np.inf)]
STEP_MM = 80.9

os.makedirs(FIGURES, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)
summary = {}


def load(part):
    with np.load(os.path.join(CACHE, part + ".npz")) as f:
        return {k: f[k] for k in f.files}


def sigma_of(key):
    with open(os.path.join(STORE, "runs", key, "constants.json")) as h:
        return np.asarray(json.load(h)["loss"]["divisor_per_component"])


def write_table(name, header, rows, fmt="%.4g"):
    def cell(v):
        return fmt % v if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    with open(os.path.join(TABLES, name + ".md"), "w") as h:
        h.write("\n".join(lines) + "\n")
    with open(os.path.join(TABLES, name + ".csv"), "w") as h:
        h.write(",".join(header) + "\n")
        h.writelines(",".join(cell(v) for v in row) + "\n" for row in rows)
    print("\n" + name + "\n" + "\n".join(lines))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, name + ".png"), dpi=130)
    plt.close(fig)


c = load("carried")
z = c["planes_mm"]
ref = c["reference"]
n_tracks = len(ref)
p = c["momentum_gev"]
qop = c["start"][:, 4]
summary["rounds"] = {k: int(c[k + "_round"]) for k in KEYS}
summary["tracks"] = int(n_tracks)
err = {k: (c[k][..., :4] - ref[..., :4]) for k in ("A1", "A2", "exact")}
Bmag = np.linalg.norm(c["B_on_reference"], axis=-1)
B_med = np.median(Bmag, axis=0)
By_med = np.median(c["B_on_reference"][..., 1], axis=0)

# -- 1 error along the track ----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for ax, (j, (cname, unit_scale, unit)) in zip(axes.flat, enumerate(COMPONENTS)):
    for k in ("A1", "A2", "exact"):
        a = np.abs(err[k][..., j]) * unit_scale
        ax.plot(z, np.median(a, 0), color=COLOURS[k], label="%s median" % k)
        ax.plot(z, np.percentile(a, 90, 0), color=COLOURS[k], ls="--", lw=0.9,
                label="%s p90" % k)
    ax.set_yscale("log")
    ax.set_ylabel("|d%s| (%s)" % (cname, unit))
    ax.grid(alpha=0.3)
    tw = ax.twinx()
    tw.fill_between(z, 0, B_med, color="k", alpha=0.08, lw=0)
    tw.set_ylim(0, 1.15 * B_med.max())
    tw.set_ylabel("|B| median on the reference path (T)", fontsize=7, color="0.4")
    tw.tick_params(labelsize=7, colors="0.4")
axes[0, 0].legend(fontsize=7, ncol=3)
for ax in axes[1]:
    ax.set_xlabel("z (mm)")
fig.suptitle("Error along the track against the reference, %d test tracks (grey: |B|)" % n_tracks)
save(fig, "1_error_along_the_track")

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for ax, (j, (cname, unit_scale, unit)) in zip(axes.flat, enumerate(COMPONENTS)):
    for k in ("A1", "A2"):
        e = err[k][..., j] * unit_scale
        ax.plot(z, np.median(e, 0), color=COLOURS[k], label="%s signed median" % k)
        ax.fill_between(z, np.percentile(e, 25, 0), np.percentile(e, 75, 0),
                        color=COLOURS[k], alpha=0.15, lw=0, label="%s 25-75 %%" % k)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("d%s (%s)" % (cname, unit))
    ax.grid(alpha=0.3)
    tw = ax.twinx()
    tw.plot(z, By_med, color="0.5", lw=0.8)
    tw.set_ylabel("By median (T)", fontsize=7, color="0.4")
    tw.tick_params(labelsize=7, colors="0.4")
axes[0, 0].legend(fontsize=7)
for ax in axes[1]:
    ax.set_xlabel("z (mm)")
fig.suptitle("Signed error along the track (grey line: By)")
save(fig, "1b_signed_error_along_the_track")

rows = []
for i in list(range(0, 65, 8)):
    row = [i, "%.0f" % z[i], "%.2f" % B_med[i]]
    for k in ("A1", "A2"):
        for j, (cname, s, unit) in enumerate(COMPONENTS):
            row.append(float(np.median(np.abs(err[k][:, i, j]) * s)))
    rows.append(row)
write_table("1_error_along_the_track_median",
            ["plane", "z mm", "|B| T"] + ["%s |d%s| %s" % (k, cn, u) for k in ("A1", "A2")
                                         for cn, _, u in COMPONENTS], rows, "%.3g")
# where the growth happens: share of the endpoint median |dx|, |dtx| reached by each quarter
for k in ("A1", "A2"):
    for j, (cname, s, unit) in enumerate(COMPONENTS):
        m = np.median(np.abs(err[k][..., j]) * s, 0)
        summary.setdefault("median_along", {}).setdefault(k, {})[cname] = \
            {"plane_%d" % i: float(m[i]) for i in (8, 16, 24, 32, 48, 64)}

# -- 2 who fails ------------------------------------------------------------------------
radial = {k: np.hypot(err[k][:, -1, 0], err[k][:, -1, 1]) * 1e3 for k in ("A1", "A2", "exact")}
old = c["old_subset"]
summary["endpoint"] = {k: dict(median=float(np.median(radial[k])),
                               p95=float(np.percentile(radial[k], 95)),
                               median_old_1452=float(np.median(radial[k][old])),
                               p95_old_1452=float(np.percentile(radial[k][old], 95)))
                       for k in radial}
start = c["start"]
groups = []
for lo, hi in BANDS:
    groups.append(("p %g-%g GeV" % (lo, hi) if np.isfinite(hi) else "p > %g GeV" % lo,
                   (p >= lo) & (p < hi)))
groups += [("charge +", qop > 0), ("charge -", qop < 0)]
for name, col, unit in (("|tx| start", 2, ""), ("|ty| start", 3, ""), ("|x| start", 0, "mm"),
                        ("|y| start", 1, "mm")):
    v = np.abs(start[:, col])
    edges = np.percentile(v, [0, 25, 50, 75, 100])
    for a, b in zip(edges[:-1], edges[1:]):
        groups.append(("%s %.3g-%.3g %s" % (name, a, b, unit), (v >= a) & (v <= b)))
rows = []
for name, m in groups:
    rows.append([name, int(m.sum()), float(np.median(radial["A1"][m])),
                 float(np.percentile(radial["A1"][m], 90)), float(np.median(radial["A2"][m])),
                 float(np.median(radial["exact"][m]))])
write_table("2_endpoint_by_group", ["group", "tracks", "A1 median µm", "A1 p90 µm",
                                     "A2 median µm", "exact median µm"], rows, "%.4g")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axes[0]
for k in ("A1", "A2"):
    ax.scatter(p, radial[k], s=1, alpha=0.15, color=COLOURS[k])
    edges = np.geomspace(1, p.max(), 25)
    mid = np.sqrt(edges[1:] * edges[:-1])
    med = [np.median(radial[k][(p >= a) & (p < b)]) if ((p >= a) & (p < b)).sum() > 20 else np.nan
           for a, b in zip(edges[:-1], edges[1:])]
    ax.plot(mid, med, color=COLOURS[k], lw=2, label=k + " median")
ax.plot(mid, 2 * np.nanmedian(radial["A1"]) * np.median(p) / mid, "k:", lw=1, label="∝ 1/p")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("p (GeV)"); ax.set_ylabel("endpoint radial error (µm)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fits = {}
for ax, j, cname in ((axes[1], 0, "x"), (axes[2], 1, "y")):
    for k in ("A1", "A2"):
        d = err[k][:, -1, j] * 1e3
        X = np.stack([qop, np.ones_like(qop)], 1)
        coef, *_ = np.linalg.lstsq(X, d, rcond=None)
        resid = d - X @ coef
        fits[(k, cname)] = dict(a_um_per_c_over_GeV=float(coef[0]), b_um=float(coef[1]),
                                r2=float(1 - resid.var() / d.var()),
                                median_abs_before=float(np.median(np.abs(d))),
                                median_abs_after=float(np.median(np.abs(resid))))
        ax.scatter(qop, d, s=1, alpha=0.15, color=COLOURS[k])
        xs = np.linspace(qop.min(), qop.max(), 50)
        ax.plot(xs, coef[0] * xs + coef[1], color=COLOURS[k], lw=2,
                label="%s fit, R² %.2f" % (k, fits[(k, cname)]["r2"]))
    ax.axhline(0, color="k", lw=0.5)
    lim = np.percentile(np.abs(err["A1"][:, -1, j] * 1e3), 99)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("q/p (1/GeV)"); ax.set_ylabel("signed d%s at the last plane (µm)" % cname)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("Who fails: endpoint error against momentum and q/p, %d test tracks" % n_tracks)
save(fig, "2_who_fails")
# the fit of the bend scale: does the bend of the reference explain the error?
bend_x = (ref[:, -1, 0] - (ref[:, 0, 0] + ref[:, 0, 2] * (z[-1] - z[0]))) * 1e3   # µm
rows = []
for k in ("A1", "A2"):
    for cname in ("x", "y"):
        f = fits[(k, cname)]
        rows.append([k, cname, f["a_um_per_c_over_GeV"], f["b_um"], f["r2"],
                     f["median_abs_before"], f["median_abs_after"]])
    d = err[k][:, -1, 0] * 1e3
    s, *_ = np.linalg.lstsq(bend_x[:, None], d, rcond=None)
    r = d - bend_x * s[0]
    summary.setdefault("bend_scale", {})[k] = dict(
        relative_scale=float(s[0]), r2=float(1 - r.var() / d.var()),
        median_bend_um=float(np.median(np.abs(bend_x))))
write_table("2_fit_against_q_over_p", ["run", "component", "a (µm GeV)", "b (µm)", "R²",
                                       "median |d| µm", "median |d - fit| µm"], rows, "%.4g")
summary["fit_against_q_over_p"] = {"%s_%s" % k: v for k, v in fits.items()}

# -- 3 position against slope ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
dz = np.diff(z)
for k in ("A1", "A2"):
    for j, cname in ((0, "x"), (1, "y")):
        dt = err[k][..., 2 + j]
        integral = np.concatenate([np.zeros((n_tracks, 1)),
                                   np.cumsum(0.5 * (dt[:, 1:] + dt[:, :-1]) * dz, 1)], 1) * 1e3
        d_end = err[k][:, -1, j] * 1e3
        rr = np.corrcoef(integral[:, -1], d_end)[0, 1]
        rel = np.median(np.abs(d_end - integral[:, -1])) / np.median(np.abs(d_end))
        summary.setdefault("position_from_slope", {})["%s_%s" % (k, cname)] = dict(
            correlation=float(rr), median_abs_difference_over_median_abs=float(rel))
        if j == 0:
            axes[0 if k == "A1" else 1].scatter(integral[:, -1], d_end, s=1, alpha=0.2,
                                                color=COLOURS[k])
            lim = np.percentile(np.abs(d_end), 99)
            a = axes[0 if k == "A1" else 1]
            a.plot([-lim, lim], [-lim, lim], "k:", lw=1)
            a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
            a.set_xlabel("∫ dtx dz from the first plane (µm)")
            a.set_ylabel("dx at the last plane (µm)")
            a.set_title("%s: correlation %.4f" % (k, rr))
            a.grid(alpha=0.3)
        # where along z the endpoint position error is built: the slope error at each
        # plane, weighted by what is left of the track, i.e. the integral's increments
        inc = 0.5 * (dt[:, 1:] + dt[:, :-1]) * dz * 1e3            # µm per step
        share = np.sum(inc * np.sign(d_end)[:, None], 0) / np.sum(np.abs(d_end))
        axes[2].plot(0.5 * (z[1:] + z[:-1]), np.cumsum(share), color=COLOURS[k],
                     ls="-" if j == 0 else "--", label="%s, %s" % (k, cname))
        summary["position_from_slope"]["%s_%s" % (k, cname)].update(
            {"share_by_plane_%d" % i: float(np.cumsum(share)[i - 1]) for i in (16, 32, 48, 64)})
axes[2].set_xlabel("z (mm)")
axes[2].set_ylabel("cumulative share of |d_end| from ∫ dt dz")
axes[2].set_title("Where the endpoint position error is built")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
tw = axes[2].twinx(); tw.fill_between(z, 0, B_med, color="k", alpha=0.08, lw=0)
tw.set_yticks([])
fig.suptitle("Position against carried slope error")
save(fig, "3_position_against_slope")

# -- 4 where the training lands ----------------------------------------------------------
t = load("training")
ch = load("chain")
steps = np.arange(64)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax = axes[0, 0]
sigA1 = sigma_of(KEYS["A1"])
for k in ("A1", "A2"):
    ax.plot(steps, t[k + "_L"], color=COLOURS[k], label="%s, training states" % k)
    r = ch[k + "_r"]
    Ltest = ((r / sigma_of(KEYS[k])[None, None, None, :]) ** 2).mean(axis=(0, 2, 3))
    ax.plot(steps, Ltest, color=COLOURS[k], ls=":", label="%s, %d test tracks" % (k, len(ch["rows"])))
    summary.setdefault("L_n", {})[k] = dict(
        training_mean=float(t[k + "_L"].mean()), test_mean=float(Ltest.mean()),
        training={str(i): float(t[k + "_L"][i]) for i in (0, 8, 16, 24, 32, 40, 48, 56, 63)})
LP1 = ((t["P1_R"] / sigA1[None, None, None, :]) ** 2).mean(axis=(0, 2, 3))
ax.plot(steps, LP1, color=COLOURS["P1"], label="P1 (linear map, own 1,452 tracks)")
summary["L_n"]["P1"] = dict(mean=float(LP1.mean()),
                            values={str(i): float(LP1[i]) for i in (0, 8, 16, 24, 32, 40, 48, 56, 63)})
ax.set_yscale("log"); ax.set_xlabel("step n"); ax.set_ylabel("L_n")
ax.set_title("Stage residual loss per step"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes[0, 1]
for k in ("A1", "A2"):
    L = t[k + "_L"]
    ax.plot(steps, L / L.sum(), color=COLOURS[k], label="%s loss share" % k)
    g = t[k + "_grad_outputs"] ** 2
    ax.plot(steps, g / g.sum(), color=COLOURS[k], ls="--", label="%s |∂loss/∂outputs_n|² share" % k)
    gp = t[k + "_grad_params_per_step"]
    ax.plot(steps, gp / gp.sum(), color=COLOURS[k], ls=":", label="%s |∇θ L_n| share" % k)
    # endpoint: the local defect of each step carried to the last plane
    cc = ch[k + "_carried"] * 1e3                               # µm, (n, N, 2)
    e2 = (cc ** 2).sum(-1).mean(0)
    ax.plot(steps, e2 / e2.sum(), color=COLOURS[k], lw=2.2, alpha=0.5,
            label="%s carried defect² share" % k)
    quarter = lambda v: [float(v[i:i + 16].sum() / v.sum()) for i in (0, 16, 32, 48)]
    summary.setdefault("shares_by_quarter", {})[k] = dict(
        loss=quarter(L), grad_outputs=quarter(g), grad_params=quarter(gp),
        carried_defect_squared=quarter(e2))
ax.set_xlabel("step n"); ax.set_ylabel("share of the sum over the steps")
ax.set_title("Where the loss and the gradient are, and where the endpoint error is made")
ax.legend(fontsize=6); ax.grid(alpha=0.3)

ax = axes[1, 0]
for k in ("A1", "A2"):
    ax.plot(steps, t[k + "_cos_with_total"], color=COLOURS[k], label=k)
ax.set_xlabel("step n"); ax.set_ylabel("cos(∇θ L_n, ∇θ loss)")
ax.set_title("Alignment of each step's gradient with the total"); ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1, 1]
for k in ("A1", "A2"):
    cc = ch[k + "_carried"][..., 0] * 1e3
    endpoint = cc.sum(1)
    signed = np.mean(cc * np.sign(endpoint)[:, None], 0)
    ax.plot(steps, np.cumsum(signed) / np.mean(np.abs(endpoint)), color=COLOURS[k], label=k)
    coherence = np.abs(cc.sum(1)) / np.sqrt((cc ** 2).sum(1))
    summary.setdefault("coherence_x", {})[k] = float(np.median(coherence))
ax.set_xlabel("step n"); ax.set_ylabel("cumulative signed share of |dx_end|")
ax.set_title("Carried local defects, x: where the endpoint is built"); ax.grid(alpha=0.3)
ax.legend(fontsize=8)
save(fig, "4_where_the_training_lands")

rows = []
for k in ("A1", "A2"):
    s = summary["shares_by_quarter"][k]
    for name in ("loss", "grad_outputs", "grad_params", "carried_defect_squared"):
        rows.append([k, name] + s[name])
write_table("4_shares_by_quarter", ["run", "quantity", "steps 0-15", "16-31", "32-47", "48-63"],
            rows, "%.3f")

# -- 5 residual against error --------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
rows = []
for k in ("A1", "A2"):
    r, e = ch[k + "_r"], ch[k + "_eps"]
    for j, (cname, s, unit) in enumerate(COMPONENTS):
        ar, ae = np.abs(r[..., j]) * s, np.abs(e[..., j]) * s
        ratio = ae / np.maximum(ar, 1e-300)
        rows.append([k, cname, float(np.median(ar)), float(np.median(ae)),
                     float(np.median(ratio)), float(np.percentile(ratio, 99)),
                     float(spearmanr(ar.reshape(-1)[::7], ae.reshape(-1)[::7])[0])])
        if j == 0:
            axes[0].plot(steps, np.median(ar.max(-1), 0), color=COLOURS[k],
                         label="%s |r_x| (µm)" % k)
            axes[0].plot(steps, np.median(ae.max(-1), 0), color=COLOURS[k], ls="--",
                         label="%s |ε_x| (µm)" % k)
        if j == 2:
            axes[1].plot(steps, np.median(ar.max(-1), 0), color=COLOURS[k],
                         label="%s |r_tx| (µrad)" % k)
            axes[1].plot(steps, np.median(ae.max(-1), 0), color=COLOURS[k], ls="--",
                         label="%s |ε_tx| (µrad)" % k)
    # per track: mean loss against endpoint error
    Ltrack = ((r / sigma_of(KEYS[k])) ** 2).mean(axis=(1, 2, 3))
    endpoint = radial[k][ch["rows"]]
    summary.setdefault("track_loss_vs_endpoint_spearman", {})[k] = float(
        spearmanr(Ltrack, endpoint)[0])
    axes[2].scatter(Ltrack, endpoint, s=3, alpha=0.3, color=COLOURS[k], label=k)
    summary.setdefault("delta_over_dz_J_eps", {})
for ax in axes[:2]:
    ax.set_yscale("log"); ax.set_xlabel("step n"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
axes[0].set_title("Stage residual and stage error, x")
axes[1].set_title("Stage residual and stage error, tx")
axes[2].set_xscale("log"); axes[2].set_yscale("log")
axes[2].set_xlabel("mean loss of the track"); axes[2].set_ylabel("endpoint radial error (µm)")
axes[2].set_title("Per track: loss against endpoint error"); axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)
save(fig, "5_residual_against_error")
write_table("5_residual_against_error",
            ["run", "component", "median |r|", "median |ε|", "median |ε|/|r|", "p99 |ε|/|r|",
             "Spearman |r|,|ε|"], rows, "%.4g")

# -- 6 one Newton sweep ----------------------------------------------------------------------
sw = load("sweep")
summary["sweep"] = {k: v.item() for k, v in sw.items() if k != "rows"}
summary["sample"] = int(len(sw["rows"]))
summary["worst_exact_residual"] = {k: float(ch[k + "_worst_exact_residual"]) for k in ("A1", "A2")}
with open(os.path.join(TABLES, "summary.json"), "w") as h:
    json.dump(summary, h, indent=1)
print(json.dumps(summary, indent=1))
