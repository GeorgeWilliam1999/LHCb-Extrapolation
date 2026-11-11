#!/usr/bin/env python
"""D3 - where does the error live? Track by track, for the best chain at each N.

For the best stage count at each step count (lowest test median against the
fine truth), every test particle's error at z1 against the RK6 truth is split
into its four components and set beside the particle's properties: momentum,
charge, species, pseudorapidity, position at both ends, slopes, and how far
the field bends it in x and in y. Three questions are answered from that:

  1. Is the median carried by a few problematic tracks? The error
     distribution's quantiles and tail share, the median after removing each
     kind of awkward track, and the median in slices of every property.
  2. Is it x and tx, or y and ty? The four component medians, which component
     sets the position error, and each component set against how much the
     field actually deflects the track in that plane.
  3. Are the same tracks bad for every N? The rank correlation of the
     per-track errors between chains.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python track_errors.py

Outputs:
    results/track_errors.csv        one row per (chain, test particle)
    results/error_distribution.csv  quantiles, tail share, medians with awkward tracks removed
    results/error_by_slice.csv      medians in slices of every property
    results/component_split.csv     x/tx against y/ty, per chain and momentum band
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BD = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "results")
PBANDS = [(1, 2), (2, 5), (5, 10), (10, 25), (25, 200), (1, 200)]


def best_chains():
    rows = list(csv.DictReader(open(os.path.join(BD, "D2_Comparators", "results",
                                                  "comparison_table.csv"))))
    best = {}
    for r in rows:
        N, q, med = int(r["N"]), int(r["q"]), float(r["network_vs_rk6_med_um"])
        if N not in best or med < best[N][1]:
            best[N] = (q, med)
    return [(N, best[N][0]) for N in sorted(best)]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    d = np.load(os.path.join(BD, "D0_Crossing_dataset", "results", "crossing_particles.npz"))
    Z0, Z1 = float(d["z0"]), float(d["z1"])
    S0, T = d["test_S0"], d["test_truth"][:, -1]
    P, ETA, PID, EVT, MCK = d["test_P"], d["test_ETA"], d["test_PID"], d["test_EVT"], d["test_MCKEY"]
    L = Z1 - Z0
    props = dict(
        evt=EVT, mckey=MCK, pid=PID, p_gev=P, charge=np.sign(S0[:, 4]), eta=ETA,
        x0_mm=S0[:, 0], y0_mm=S0[:, 1], tx0=S0[:, 2], ty0=S0[:, 3],
        x1_mm=T[:, 0], y1_mm=T[:, 1], tx1=T[:, 2], ty1=T[:, 3],
        r1_mm=np.hypot(T[:, 0], T[:, 1]),
        max_slope0=np.maximum(np.abs(S0[:, 2]), np.abs(S0[:, 3])),
        bend_x_mm=T[:, 0] - (S0[:, 0] + S0[:, 2] * L),
        bend_y_mm=T[:, 1] - (S0[:, 1] + S0[:, 3] * L),
        dtx_field_mrad=(T[:, 2] - S0[:, 2]) * 1e3,
        dty_field_mrad=(T[:, 3] - S0[:, 3]) * 1e3,
    )
    chains = best_chains()
    err = {}
    long_rows = []
    for N, q in chains:
        st = np.load(os.path.join(BD, "D1_Chain_grid", "results", "N%03d_q%02d" % (N, q),
                                  "states.npz"))["test_states"]
        assert np.allclose(st[:, 0], S0), "row alignment"
        e = st[:, -1] - T
        dx, dy, dtx, dty = e[:, 0] * 1e3, e[:, 1] * 1e3, e[:, 2] * 1e3, e[:, 3] * 1e3
        pos = np.maximum(np.abs(dx), np.abs(dy))
        err[(N, q)] = dict(dx=dx, dy=dy, dtx=dtx, dty=dty, pos=pos)
        for i in range(len(P)):
            row = dict(N=N, q=q)
            row.update({k: (float(v[i]) if np.issubdtype(np.asarray(v).dtype, np.floating) else int(v[i]))
                        for k, v in props.items()})
            row.update(dx_um=dx[i], dy_um=dy[i], dtx_mrad=dtx[i], dty_mrad=dty[i], pos_um=pos[i])
            long_rows.append(row)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "track_errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)

    # -- 1. the distribution and the awkward tracks ----------------------------
    r1, ms = props["r1_mm"], props["max_slope0"]
    removals = [
        ("all tracks", np.ones(len(P), bool)),
        ("without p < 5 GeV", P >= 5),
        ("without the beam-line region, r1 < 150 mm", r1 >= 150),
        ("without the outer region, r1 > 1500 mm", r1 <= 1500),
        ("without steep tracks, a start slope above 0.2", ms <= 0.2),
        ("without protons and kaons", np.abs(PID) == 211),
        ("clean core: p >= 5 GeV, 150 <= r1 <= 1500 mm, slopes <= 0.2", (P >= 5) & (r1 >= 150) & (r1 <= 1500) & (ms <= 0.2)),
    ]
    ref = err[chains[0]]["pos"]
    dist = []
    for (N, q), E in err.items():
        pos = E["pos"]
        srt = np.sort(pos)[::-1]
        row = dict(N=N, q=q, n=len(pos), mean_um=pos.mean(),
                   **{"p%02d_um" % int(k * 100): float(np.quantile(pos, k))
                      for k in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)},
                   max_um=float(pos.max()),
                   worst5pct_share_of_sum=float(srt[:int(0.05 * len(srt))].sum() / srt.sum()),
                   worst10pct_share_of_sum=float(srt[:int(0.10 * len(srt))].sum() / srt.sum()),
                   rank_corr_with_N1=spearman(pos, ref))
        for lab, m in removals:
            row["median | %s" % lab] = float(np.median(pos[m]))
            row["n | %s" % lab] = int(m.sum())
        dist.append(row)
    with open(os.path.join(OUT, "error_distribution.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist[0].keys()))
        w.writeheader()
        w.writerows(dist)

    # -- slices -------------------------------------------------------------------
    slices = [
        ("p_gev", P, [1, 2, 5, 10, 25, 200]),
        ("eta", ETA, [2, 2.5, 3, 3.5, 4, 4.5, 5]),
        ("r1_mm", r1, [0, 150, 300, 600, 1000, 1500, 4000]),
        ("abs_x1_mm", np.abs(props["x1_mm"]), [0, 150, 300, 600, 1000, 2000, 4000]),
        ("abs_y1_mm", np.abs(props["y1_mm"]), [0, 100, 300, 600, 1000, 3000]),
        ("max_slope0", ms, [0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]),
        ("charge", props["charge"], [-1.5, 0, 1.5]),
        ("abs_pid", np.abs(PID), [0, 212, 322, 2213]),
    ]
    by = []
    for (N, q), E in err.items():
        for name, v, edges in slices:
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (v >= lo) & (v < hi)
                if m.sum() == 0:
                    continue
                by.append(dict(N=N, q=q, variable=name, lo=lo, hi=hi, n=int(m.sum()),
                               pos_med_um=float(np.median(E["pos"][m])),
                               pos_p90_um=float(np.quantile(E["pos"][m], 0.9)),
                               dx_med_um=float(np.median(np.abs(E["dx"][m]))),
                               dy_med_um=float(np.median(np.abs(E["dy"][m]))),
                               dtx_med_mrad=float(np.median(np.abs(E["dtx"][m]))),
                               dty_med_mrad=float(np.median(np.abs(E["dty"][m])))))
    with open(os.path.join(OUT, "error_by_slice.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by[0].keys()))
        w.writeheader()
        w.writerows(by)

    # -- 2. x and tx against y and ty -------------------------------------------------
    comp = []
    for (N, q), E in err.items():
        for lo, hi in PBANDS:
            m = (P >= lo) & (P < hi)
            if m.sum() < 5:
                continue
            md = lambda a: float(np.median(np.abs(a[m])))
            comp.append(dict(
                N=N, q=q, p_lo=lo, p_hi=hi, n=int(m.sum()),
                dx_med_um=md(E["dx"]), dy_med_um=md(E["dy"]),
                dtx_med_mrad=md(E["dtx"]), dty_med_mrad=md(E["dty"]),
                x_sets_position_error=float(np.mean(np.abs(E["dx"][m]) >= np.abs(E["dy"][m]))),
                field_bend_x_med_mm=md(props["bend_x_mm"]), field_bend_y_med_mm=md(props["bend_y_mm"]),
                field_dtx_med_mrad=md(props["dtx_field_mrad"]), field_dty_med_mrad=md(props["dty_field_mrad"]),
                dx_over_bend_x=md(E["dx"]) / (md(props["bend_x_mm"]) * 1e3),
                dy_over_bend_y=md(E["dy"]) / (md(props["bend_y_mm"]) * 1e3),
                dtx_over_field_dtx=md(E["dtx"]) / md(props["dtx_field_mrad"]),
                dty_over_field_dty=md(E["dty"]) / md(props["dty_field_mrad"]),
            ))
    with open(os.path.join(OUT, "component_split.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0].keys()))
        w.writeheader()
        w.writerows(comp)
    print("chains:", chains)


if __name__ == "__main__":
    main()
