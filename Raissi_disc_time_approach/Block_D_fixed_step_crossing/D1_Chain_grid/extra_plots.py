#!/usr/bin/env python
"""D1 - the tables and figures George asked for on 2026-09-15, from the chain
records, their stored states and D0. Computes errors of stored predictions
against stored truths; trains nothing.

    results/single_track.json                     which particle the single-track
                                                  tables are for, and why
    results/table_single_track_<x,y,tx,ty>.csv     error(N, q) at z1 for that ONE
                                                  particle, signed, per component
    results/table_vs_rk6_test_<comp>_band_<band>.csv
                                                  the 5 x 20 tables per momentum
                                                  band, per component (medians)
    results/qop_check.csv                         max |dq/p| along every chain
    results/val_vs_test.csv                       the two splits' medians per chain
    results/cost_table.csv                        forward cost per crossing per N
    results/block_a_continuity.csv                the N = 1 column beside Block A's
                                                  frozen-leg numbers
    figures/err_vs_p_N<NNN>_q<qq>.png             signed error against momentum,
                                                  per component, one chain
    figures/err_vs_p_medians.png                  median |error| against momentum,
                                                  per component, one line per N
    figures/slope_to_position.png                 the x error a leg adds against
                                                  the tx error it inherited x dz
    figures/val_vs_test.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import csv
import glob
import json
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

import use_shared                                                 # noqa: E402,F401
from _shared.prepare import P_BANDS                               # noqa: E402
from metrics import COMPONENTS                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
D0 = os.path.join(HERE, "..", "D0_Crossing_dataset", "results", "crossing_particles.npz")
BLOCK_A = os.path.join(use_shared.SHARED_ROOT, "Block_A_technique_works")
N_VALUES = (1, 4, 16, 64, 128)
QS = tuple(range(1, 21))
CAT = {1: "#2a78d6", 4: "#eb6834", 16: "#1baf7a", 64: "#eda100", 128: "#e87ba4"}
INK, INK2, GRID = "#1a1a19", "#5f5e5a", "#e3e2dc"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
UNIT = {"x": "µm", "y": "µm", "tx": "mrad", "ty": "mrad"}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def pivot(values, fmt="%.6g"):
    rows = []
    for N in N_VALUES:
        r = {"N": N}
        for q in QS:
            v = values.get((N, q))
            r["q%02d" % q] = (fmt % v) if v is not None and np.isfinite(v) else ""
        rows.append(r)
    return rows


def chains():
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "N*_q*", "chain.json"))):
        m = re.search(r"N(\d+)_q(\d+)", p)
        with open(p) as f:
            out[(int(m.group(1)), int(m.group(2)))] = json.load(f)
    return out


def test_states(N, q):
    st = np.load(os.path.join(RESULTS, "N%03d_q%02d" % (N, q), "states.npz"))
    return st["test_states"]


def running_band(p, err, nbins=12):
    edges = np.geomspace(1.0, 200.0, nbins + 1)
    mid, med, lo, hi, absmed = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() < 8:
            continue
        mid.append(np.sqrt(a * b))
        med.append(np.median(err[m]))
        lo.append(np.quantile(err[m], 0.16))
        hi.append(np.quantile(err[m], 0.84))
        absmed.append(np.median(np.abs(err[m])))
    return np.array(mid), np.array(med), np.array(lo), np.array(hi), np.array(absmed)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    D = {k: v for k, v in np.load(D0, allow_pickle=False).items()}
    n_max = int(D["n_max"])
    truth_end = D["test_truth"][:, n_max]
    P = D["test_P"]
    ch = chains()
    if not ch:
        raise SystemExit("no chains yet")
    med = {k: v["test"]["vs_rk6_endpoint"]["pos_med_um"] for k, v in ch.items()}
    best = min(med, key=med.get)
    print("chains: %d; best by test median vs RK6: N = %d, q = %d (%.1f um)"
          % (len(ch), best[0], best[1], med[best]))

    # ---- 1. the single representative track ---------------------------------
    S0 = D["test_S0"]
    pm = np.median(P)
    r_xy = np.hypot(S0[:, 0], S0[:, 1])
    central = r_xy < np.quantile(r_xy, 0.5)
    cand = np.flatnonzero(central)
    i_star = int(cand[np.argmin(np.abs(P[cand] - pm))])
    info = {"index_in_test_split": i_star, "EVT": int(D["test_EVT"][i_star]),
            "MCKEY": int(D["test_MCKEY"][i_star]), "p_GeV": float(P[i_star]),
            "eta": float(D["test_ETA"][i_star]), "PID": int(D["test_PID"][i_star]),
            "S0_at_z0": S0[i_star].tolist(),
            "truth_at_z1": truth_end[i_star].tolist(),
            "why": "the test particle with momentum closest to the test median (%.2f GeV) "
                   "among those in the inner half of the transverse start positions" % pm}
    with open(os.path.join(RESULTS, "single_track.json"), "w") as f:
        json.dump(info, f, indent=1)
    single = {name: {} for name, _, _, _ in COMPONENTS}
    for (N, q) in ch:
        s = test_states(N, q)[i_star, N]
        for name, i, sc, u in COMPONENTS:
            single[name][(N, q)] = (s[i] - truth_end[i_star, i]) * sc
    for name in single:
        write_csv(os.path.join(RESULTS, "table_single_track_%s.csv" % name),
                  pivot(single[name], "%+.5g"))
    print("single track: EVT %d MCKEY %d, p = %.2f GeV" % (info["EVT"], info["MCKEY"], info["p_GeV"]))

    # ---- 2. per-band tables per component --------------------------------------
    for j, (lo, hi) in enumerate(P_BANDS):
        band = "%g-%g GeV" % (lo, hi)
        for name, _, _, u in COMPONENTS:
            vals = {}
            for k, v in ch.items():
                b = v["test"]["vs_rk6_endpoint"]["components"].get("by_p_band", {}).get(band)
                if b:
                    vals[k] = b["%s_med_%s" % (name, u)]
            write_csv(os.path.join(RESULTS, "table_vs_rk6_test_%s_band_%s.csv"
                                   % (name, band.replace(" ", "").replace("-", "to"))),
                      pivot(vals))
        vals = {k: v["test"]["vs_rk6_endpoint"]["by_p_band"][band]["pos_med_um"]
                for k, v in ch.items() if band in v["test"]["vs_rk6_endpoint"].get("by_p_band", {})}
        write_csv(os.path.join(RESULTS, "table_vs_rk6_test_pos_band_%s.csv"
                               % band.replace(" ", "").replace("-", "to")), pivot(vals))

    # ---- 3. q/p check, val vs test ------------------------------------------------
    write_csv(os.path.join(RESULTS, "qop_check.csv"),
              [{"N": N, "q": q, "qop_max_abs_change_test": v["test"]["qop_passthrough_max_abs_change"],
                "qop_max_abs_change_val": v["val"]["qop_passthrough_max_abs_change"]}
               for (N, q), v in sorted(ch.items())])
    vt = [{"N": N, "q": q, "val_med_um": v["val"]["vs_rk6_endpoint"]["pos_med_um"],
           "test_med_um": v["test"]["vs_rk6_endpoint"]["pos_med_um"]} for (N, q), v in sorted(ch.items())]
    write_csv(os.path.join(RESULTS, "val_vs_test.csv"), vt)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for N in N_VALUES:
        xs = [r["val_med_um"] for r in vt if r["N"] == N]
        ys = [r["test_med_um"] for r in vt if r["N"] == N]
        ax.plot(xs, ys, "o", ms=4, color=CAT[N], label="N = %d" % N)
    lim = [min(r["val_med_um"] for r in vt) * 0.7, max(r["val_med_um"] for r in vt) * 1.4]
    ax.plot(lim, lim, "-", lw=0.8, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("validation median  [µm]"); ax.set_ylabel("test median  [µm]")
    ax.legend(); ax.set_title("No selection on the test split: one seed, nothing chosen")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "val_vs_test.png"), dpi=140); plt.close(fig)

    # ---- 4. error against momentum ------------------------------------------------
    picks = [best] + [(N, 8) for N in N_VALUES if (N, 8) in ch and (N, 8) != best]
    for (N, q) in picks:
        s = test_states(N, q)[:, N]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, (name, i, sc, u) in zip(axes.ravel(), COMPONENTS):
            err = (s[:, i] - truth_end[:, i]) * sc
            lim = np.quantile(np.abs(err), 0.98)
            ax.hexbin(P, np.clip(err, -lim, lim), gridsize=(40, 30), xscale="log",
                      cmap="Blues", mincnt=1, linewidths=0.2)
            mid, m_, lo, hi, _ = running_band(P, err)
            ax.fill_between(mid, lo, hi, color="#eb6834", alpha=0.25, lw=0, label="16–84 %")
            ax.plot(mid, m_, "-", color="#eb6834", lw=1.8, label="running median")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_ylabel("Δ%s  [%s]  (network − RK6 at z1)" % (name, UNIT[name]))
            ax.set_title(name)
        for ax in axes[1]:
            ax.set_xlabel("truth momentum p  [GeV]")
        axes[0][0].legend(loc="upper right", fontsize=8)
        fig.suptitle("Signed error at z1 against momentum, test split: N = %d, q = %d "
                     "(clipped at the 98th percentile of |error|)" % (N, q))
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, "err_vs_p_N%03d_q%02d.png" % (N, q)), dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (name, i, sc, u) in zip(axes.ravel(), COMPONENTS):
        for N in N_VALUES:
            if (N, 8) not in ch:
                continue
            s = test_states(N, 8)[:, N]
            mid, _, _, _, am = running_band(P, (s[:, i] - truth_end[:, i]) * sc)
            ax.plot(mid, am, "-o", ms=3.5, lw=1.6, color=CAT[N], label="N = %d, q = 8" % N)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel("median |Δ%s|  [%s]" % (name, UNIT[name])); ax.set_title(name)
    for ax in axes[1]:
        ax.set_xlabel("truth momentum p  [GeV]")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Median |error| at z1 against momentum, per component, test split, q = 8")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "err_vs_p_medians.png"), dpi=140); plt.close(fig)

    # ---- 5. slope to position: what a leg adds against what it inherits ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, N in zip(axes, (16, 64, 128)):
        qs = [q for (n, q) in ch if n == N]
        if not qs:
            ax.set_visible(False); continue
        q = min(qs, key=lambda qq: med[(N, qq)])
        st = test_states(N, q)
        truth = D["test_truth"][:, ::n_max // N]
        dz = float(D["L"]) / N
        dx = (st[:, :, 0] - truth[:, :, 0]) * 1e3          # (n, N+1) um
        dtx = (st[:, :, 2] - truth[:, :, 2]) * 1e3         # mrad
        added = np.abs(dx[:, 1:] - dx[:, :-1]).ravel()
        inherited = np.abs(dtx[:, :-1] * dz).ravel()       # mrad * mm = um
        ax.hexbin(np.maximum(inherited, 1e-3), np.maximum(added, 1e-3), gridsize=45,
                  xscale="log", yscale="log", cmap="Blues", mincnt=1, linewidths=0.2)
        lim = [1e-2, max(added.max(), inherited.max())]
        ax.plot(lim, lim, "-", lw=0.9, color=INK2)
        ax.set_xlabel("|Δtx at plane k| × dz  [µm]  (the inherited slope error, carried over the leg)")
        ax.set_ylabel("|Δx at plane k+1 − Δx at plane k|  [µm]  (what the leg added)")
        ax.set_title("N = %d, q = %d, all legs, test split" % (N, q))
    fig.suptitle("Is the position error just the inherited slope error integrated? Points on the line say yes.")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "slope_to_position.png"), dpi=140); plt.close(fig)

    # ---- 6. cost ------------------------------------------------------------------
    from apply_chain import load_chain
    rows = []
    for N in N_VALUES:
        qs = [q for (n, q) in ch if n == N]
        if not qs:
            continue
        for q in sorted({8, min(qs, key=lambda qq: med[(N, qq)])} & set(qs)):
            c = load_chain(RESULTS, N, q)
            S = D["test_S0"]
            c.extrapolate(S[:50])                          # warm-up
            t0 = time.perf_counter(); c.extrapolate(S); dt = time.perf_counter() - t0
            npar = ch[(N, q)]["n_parameters_per_leg"]
            rows.append({"N": N, "q": q, "networks_per_crossing": N,
                         "parameters_per_leg": npar, "parameters_per_crossing": npar * N,
                         "forward_us_per_track_per_crossing_batch%d" % len(S): dt / len(S) * 1e6,
                         "test_med_um_vs_rk6": med[(N, q)],
                         "note": "one thread, fp64, PyTorch eager, includes the field integral for the scale"})
    write_csv(os.path.join(RESULTS, "cost_table.csv"), rows)

    # ---- 7. continuity with Block A -----------------------------------------------
    cont = []
    a1 = os.path.join(BLOCK_A, "A1_Stage_count_sweep", "results", "error_vs_stages.csv")
    if os.path.exists(a1):
        with open(a1) as f:
            for r in csv.DictReader(l for l in f if not l.startswith("#")):
                cont.append({"source": "Block A1 stage sweep, 4x50, magnet-DOWN map, median of 3 seeds",
                             "q": int(r["q"]), "N": 1, "physics_med_um": float(r["physics_endpoint_med_um"]),
                             "twin_med_um": float(r["data_endpoint_med_um"]),
                             "scheme_med_um": float(r["scheme_endpoint_med_um"]),
                             "straight_med_um": float(r["straight_med_um"])})
    a2 = os.path.join(BLOCK_A, "A2_Network_size_and_seed_study", "results", "by_architecture.csv")
    if os.path.exists(a2):
        with open(a2) as f:
            for r in csv.DictReader(f):
                if r["mode"] == "physics" and (r["width"], r["depth"]) in (("50", "4"), ("200", "4"), ("128", "2")):
                    cont.append({"source": "Block A2 size and seed, %sx%s, q = 8, magnet-DOWN map, median of 10 seeds"
                                           % (r["depth"], r["width"]),
                                 "q": 8, "N": 1, "physics_med_um": float(r["test_endpoint_med_um_median"]),
                                 "twin_med_um": "", "scheme_med_um": "", "straight_med_um": ""})
    for q in QS:
        if (1, q) in ch:
            v = ch[(1, q)]["test"]
            cont.append({"source": "Block D, 2x128, magnet-UP map, one seed", "q": q, "N": 1,
                         "physics_med_um": v["vs_rk6_endpoint"]["pos_med_um"], "twin_med_um": "",
                         "scheme_med_um": "", "straight_med_um": v["straight_line_vs_rk6_endpoint"]["pos_med_um"]})
    write_csv(os.path.join(RESULTS, "block_a_continuity.csv"), cont)
    print("extra tables and figures written")


if __name__ == "__main__":
    main()
