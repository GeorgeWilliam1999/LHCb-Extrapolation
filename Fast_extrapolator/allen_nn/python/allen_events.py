"""Paired Allen PrKF runs on the same events: ARKN (off) and the network on the UT -> T crossing (on).

Inputs: <run>/ntuple.root (TrackIPResolutionCheckerNT, one row per fitted long track, same order in both
runs) and <run>/histos.root (TrackResChecker: RICH2 entry/exit residuals against MC truth).

    python allen_nn/python/allen_events.py RUNS_DIR [off_name on_name]
writes allen_nn/results/events_*.csv, allen_nn/figures/events_*.png and prints the tables.
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import uproot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.dirname(HERE)
RESULTS = os.path.join(FOLDER, "results")
FIGURES = os.path.join(FOLDER, "figures")
BANDS = [(2, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 50), (50, 200)]
TAIL_Y_MM = 800.0  # |y| of the straight line at z = 5000 mm above which the fixed steps fail (validation)


def tracks(run):
    f = uproot.open(os.path.join(run, "ntuple.root"))
    key = [k for k in f.keys() if k.endswith("/tracks;1")][0]
    return f[key].arrays(library="np")


def rich2(run):
    f = uproot.open(os.path.join(run, "histos.root"))
    out = {}
    for k in f.keys(recursive=True):
        if "TrackResChecker" in k and "/Long/RICH2" in k and k.split("/")[-1].split(";")[0] in (
                "xres", "yres", "txres", "tyres", "xpull", "ypull", "txpull", "typull"):
            h = f[k]
            v, e = h.to_numpy()
            c = 0.5 * (e[1:] + e[:-1])
            n = v.sum()
            mean = (v * c).sum() / n if n else np.nan
            rms = np.sqrt((v * (c - mean) ** 2).sum() / n) if n else np.nan
            loc, name = k.split("/")[-2], k.split("/")[-1].split(";")[0]
            out[(loc, name)] = dict(entries=n, mean=mean, rms=rms, counts=v, edges=e)
    return out


def core_sigma(x):
    """Half the 15.9-84.1 % range (a Gaussian sigma that ignores the far tails)."""
    return 0.5 * (np.percentile(x, 84.1) - np.percentile(x, 15.9)) if len(x) > 10 else np.nan


def main():
    runs = sys.argv[1]
    off_name, on_name = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("off", "on_nofb")
    a, b = tracks(os.path.join(runs, off_name)), tracks(os.path.join(runs, on_name))
    lines = ["Runs: %s (off = ARKN) vs %s (on = network on the UT->T crossing of the second pass)" % (off_name, on_name)]
    n = len(a["qop"])
    paired = n == len(b["qop"]) and np.all(a["ntrack"] == b["ntrack"]) and np.all(a["itrack"] == b["itrack"])
    lines.append("fitted long tracks (rows): off %d, on %d; same events and order: %s" % (n, len(b["qop"]), paired))
    event = np.cumsum(a["itrack"] == 0) - 1
    if paired:
        bad_a = ~(np.isfinite(a["qop"]) & np.isfinite(a["chi2"]))
        bad_b = ~(np.isfinite(b["qop"]) & np.isfinite(b["chi2"]))
        lines.append("fits with a NaN q/p or chi2: off %d, on %d, both %d (left out below)" % (
            bad_a.sum(), bad_b.sum(), (bad_a & bad_b).sum()))
        keep = ~(bad_a | bad_b)
        a = {k: v[keep] for k, v in a.items()}
        b = {k: v[keep] for k, v in b.items()}
        event = event[keep]
    lines.append("events with tracks: %d" % (event.max() + 1))

    rows = []
    for name, t in (("off", a), ("on", b)):
        real = t["waslinked"].astype(bool)
        p_true = t["truemom"]
        p_rec = np.abs(1.0 / t["qop"]) / 1e3
        dpp = (p_rec - p_true) / p_true
        chi = t["chi2"] / np.maximum(t["ndof"], 1)
        r = dict(run=name, tracks=len(chi), real=int(real.sum()), ghosts=int((~real).sum()),
                 fitted=int(t["trackWasFitted"].sum()),
                 chi2ndf_real_median=np.median(chi[real]), chi2ndf_real_mean=np.mean(chi[real]),
                 chi2ndf_ghost_median=np.median(chi[~real]), chi2ndf_ghost_mean=np.mean(chi[~real]),
                 dpp_bias=np.median(dpp[real]), dpp_sigma=core_sigma(dpp[real]))
        for lo, hi in BANDS:
            m = real & (p_true >= lo) & (p_true < hi)
            r["dpp_bias_%d_%d" % (lo, hi)] = np.median(dpp[m]) if m.sum() > 10 else np.nan
            r["dpp_sigma_%d_%d" % (lo, hi)] = core_sigma(dpp[m])
            r["n_%d_%d" % (lo, hi)] = int(m.sum())
        rows.append(r)
    with open(os.path.join(RESULTS, "events_summary_%s.csv" % on_name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines.append("\n| run | tracks | real | ghosts | chi2/ndf real median / mean | chi2/ndf ghost median / mean | dp/p bias | dp/p sigma (core) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append("| %s | %d | %d | %d | %.4f / %.4f | %.4f / %.4f | %.2e | %.4f |" % (
            r["run"], r["tracks"], r["real"], r["ghosts"], r["chi2ndf_real_median"], r["chi2ndf_real_mean"],
            r["chi2ndf_ghost_median"], r["chi2ndf_ghost_mean"], r["dpp_bias"], r["dpp_sigma"]))
    lines.append("\nMomentum resolution of real tracks by true p (core sigma of dp/p, %; bias = median dp/p, %):")
    lines.append("| p (GeV) | tracks | off sigma | on sigma | off bias | on bias |")
    lines.append("|---|---|---|---|---|---|")
    for lo, hi in BANDS:
        k = "%d_%d" % (lo, hi)
        lines.append("| %d-%d | %d | %.3f | %.3f | %.4f | %.4f |" % (lo, hi, rows[0]["n_" + k], 100 * rows[0]["dpp_sigma_" + k],
                                                              100 * rows[1]["dpp_sigma_" + k], 100 * rows[0]["dpp_bias_" + k],
                                                              100 * rows[1]["dpp_bias_" + k]))

    if paired:
        real = a["waslinked"].astype(bool)
        dq = (b["qop"] - a["qop"]) / np.abs(a["qop"])
        ca, cb = a["chi2"] / np.maximum(a["ndof"], 1), b["chi2"] / np.maximum(b["ndof"], 1)
        changed = (dq != 0) | (cb != ca) | (a["ndof"] != b["ndof"])
        n = len(dq)
        lines.append("\nTrack by track (on - off):")
        lines.append("  tracks with any change in q/p, chi2 or ndof: %d of %d (%.2f %%); q/p changed: %d; ndof changed: %d" % (
            changed.sum(), n, 100 * changed.mean(), (dq != 0).sum(), (a["ndof"] != b["ndof"]).sum()))
        lines.append("  relative q/p change: 99 %% %.2e, largest %.2e" % (np.percentile(np.abs(dq), 99), np.abs(dq).max()))
        d = cb - ca
        for label, m in (("real", real), ("ghost", ~real)):
            lines.append("  chi2/ndf change, %s: mean %+.4f, median %+.4f, 1 %% %+.3f, 99 %% %+.3f; up %d, down %d" % (
                label, d[m].mean(), np.median(d[m]), np.percentile(d[m], 1), np.percentile(d[m], 99), (d[m] > 0).sum(),
                (d[m] < 0).sum()))
        # tracks that head to the tail region (straight line of the fitted first state at z = 5000 mm)
        y5 = np.abs(a["y"] + a["ty"] * (5000.0 - a["z"]))
        tail = y5 > TAIL_Y_MM
        per_event = np.bincount(event[tail], minlength=event.max() + 1)
        lines.append("  tracks heading to |y| > %.0f mm at z = 5 m: %d (%.2f %%), %.3f per event (events with >= 1: %.1f %%)" % (
            TAIL_Y_MM, tail.sum(), 100 * tail.mean(), per_event.mean(), 100 * (per_event > 0).mean()))
        for label, m in (("tail", tail), ("not tail", ~tail)):
            lines.append("    %-8s: changed %.2f %%, chi2/ndf change mean %+.4f, 0.1 %% %+.3f, 99.9 %% %+.3f, largest |change| %.2f;"
                         " ndof changed %d (real tracks %d)" % (
                             label, 100 * changed[m].mean(), d[m].mean(), np.percentile(d[m], 0.1), np.percentile(d[m], 99.9),
                             np.abs(d[m]).max(), (a["ndof"][m] != b["ndof"][m]).sum(), (m & real).sum()))
        np.savetxt(os.path.join(RESULTS, "events_tracks_%s.csv" % on_name),
                   np.column_stack([event, a["itrack"], real, a["truemom"], a["qop"], b["qop"], a["chi2"], b["chi2"], a["ndof"],
                                    b["ndof"], y5]), delimiter=",", fmt="%.8g",
                   header="event,itrack,real,true_p_gev,qop_off,qop_on,chi2_off,chi2_on,ndof_off,ndof_on,abs_y_at_5m_mm", comments="")

    # RICH2 states against MC truth (the T side, after the crossing)
    ra, rb = rich2(os.path.join(runs, off_name)), rich2(os.path.join(runs, on_name))
    lines.append("\nRICH2 states against MC truth (TrackResChecker, long tracks): mean and RMS of the residuals / pulls")
    lines.append("| state | quantity | off entries | off mean | on mean | off RMS | on RMS |")
    lines.append("|---|---|---|---|---|---|---|")
    with open(os.path.join(RESULTS, "events_rich2_%s.csv" % on_name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "quantity", "off_entries", "off_mean", "on_mean", "off_rms", "on_rms"])
        for key in sorted(ra):
            if key in rb:
                x, y = ra[key], rb[key]
                w.writerow([key[0], key[1], x["entries"], x["mean"], y["mean"], x["rms"], y["rms"]])
                lines.append("| %s | %s | %d | %.4g | %.4g | %.4g | %.4g |" % (key[0], key[1], x["entries"], x["mean"], y["mean"],
                                                                         x["rms"], y["rms"]))

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    mid = [np.sqrt(lo * hi) for lo, hi in BANDS]
    for r, mk in zip(rows, ("o-", "s--")):
        ax[0].plot(mid, [100 * r["dpp_sigma_%d_%d" % b_] for b_ in BANDS], mk, label=r["run"])
        ax[1].plot(mid, [100 * r["dpp_bias_%d_%d" % b_] for b_ in BANDS], mk, label=r["run"])
    for x_, lab in zip(ax, ("dp/p resolution, core sigma (%)", "dp/p bias, median (%)")):
        x_.set_xscale("log")
        x_.set_xlabel("true p (GeV)")
        x_.set_ylabel(lab)
        x_.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "events_momentum_%s.png" % on_name), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, (label, sel) in enumerate((("real", True), ("ghost", False))):
        for t, name, st in ((a, "off", "step"), (b, "on", "step")):
            m = t["waslinked"].astype(bool) == sel
            ax[j].hist(t["chi2"][m] / np.maximum(t["ndof"][m], 1), bins=100, range=(0, 10), histtype=st, label=name)
        ax[j].set_yscale("log")
        ax[j].set_xlabel("chi2/ndf, %s tracks" % label)
        ax[j].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "events_chi2_%s.png" % on_name), dpi=130)
    plt.close(fig)
    keys = [k for k in sorted(ra) if k[0] == "RICH2Beg" and k in rb]
    if keys:
        fig, axes = plt.subplots(2, 4, figsize=(15, 6))
        for x_, key in zip(axes.ravel(), keys):
            for res, name in ((ra, "off"), (rb, "on")):
                h = res[key]
                x_.stairs(h["counts"], h["edges"], label=name)
            x_.set_title("%s %s" % key, fontsize=9)
            x_.set_yscale("log")
            x_.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, "events_rich2_%s.png" % on_name), dpi=130)
        plt.close(fig)
    text = "\n".join(lines)
    open(os.path.join(RESULTS, "events_tables_%s.md" % on_name), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
