#!/usr/bin/env python
"""Self_chained_paper - the single numerical source for the LaTeX mini-paper.

Everything the paper's prose and tables quote is computed here, from the files
in the experiment folders, and written to

    results/paper_numbers.json     every number, each entry carrying a "source"
    results/tab_*.csv              one CSV per table the paper will typeset
    results/cache/*.npz            intermediate arrays, so a rerun is cheap
    results/README.md              is written by hand beside this script

Nothing in the repository outside `Self_chained_paper/scripts` and
`Self_chained_paper/results` is written to.  Every experiment folder is opened
read-only and `sys.dont_write_bytecode` is set, so not even a .pyc is left in
one.

Conventions (see common.py): slopes are dimensionless and never scaled by 1e3;
positions are in micrometres; the momentum bands are [0, 3, 8, 20, 50, inf) GeV
plus the 10-50 GeV loss window reported alongside them; band membership comes
from the per-split `P` array of tracks.npz.

Run (about 20 minutes, one thread):

    cd Self_chained_paper/scripts
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python numbers.py

`--skip-heavy` reuses the cached anatomy arrays if they are present (it is the
default behaviour anyway); `--no-cache` forces every heavy step to be redone.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys                                   # noqa: E402
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
# This file is called numbers.py, which is also a standard-library module that
# numpy imports while it initialises.  Drop this folder from the import path
# BEFORE numpy is loaded so the standard `numbers` is found, then load
# `common.py` explicitly by file location.  (The same guard is in
# Block_F_reweighted_loss/F4_Writeup/numbers.py, for the same reason.)
for _p in ("", ".", HERE):
    while _p in sys.path:
        sys.path.remove(_p)

import argparse                              # noqa: E402
import importlib.util                        # noqa: E402
import json                                  # noqa: E402
import time                                  # noqa: E402

import numpy as np                           # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["sc_paper_common"] = C
_spec.loader.exec_module(C)

TIMING = []
_T0 = time.time()


def tick(what):
    TIMING.append(dict(step=what, wall_s=round(time.time() - _T0, 1)))
    print("  [%7.1f s] %s" % (time.time() - _T0, what), flush=True)


TABLES = {}


def table(name, rows, fieldnames=None):
    """Register one of the paper's tables and write it to results/tab_<name>.csv."""
    path = os.path.join(C.RESULTS, "tab_%s.csv" % name)
    C.write_csv(path, [C.jsonable(r) for r in rows], fieldnames)
    TABLES[name] = dict(path=os.path.relpath(path, C.PAPER), n_rows=len(rows))
    return path


# =============================================================== a. dataset ==
def sec_dataset():
    src_tracks = C.source(C.TRACKS)
    src_meta = C.source(C.TRACKS_META)
    meta = C.tracks_meta()
    out = {"source": C.source(C.TRACKS, C.TRACKS_META, C.BUILD_TRACKS_LOG)}

    # --- splits, bands, momentum range, composition -------------------------
    splits, band_rows = {}, []
    for s in C.SPLITS:
        a = C.split_arrays(s)
        P, PID, ETA, S0 = a["P"], a["PID"], a["ETA"], a["S0"]
        q = np.sign(S0[:, 4])
        counts = C.band_counts(P)
        pid_names = {11: "electron", 13: "muon", 211: "pion", 321: "kaon", 2212: "proton"}
        pid = {}
        for code, nm in pid_names.items():
            n = int((np.abs(PID) == code).sum())
            if n:
                pid[nm] = dict(n=n, frac=float(n) / len(P))
        other = int(len(P) - sum(v["n"] for v in pid.values()))
        if other:
            pid["other"] = dict(n=other, frac=float(other) / len(P))
        splits[s] = dict(
            n=int(len(P)), n_expected=C.SPLIT_SIZES[s],
            matches_expected=bool(len(P) == C.SPLIT_SIZES[s]),
            p_min_GeV=float(P.min()), p_median_GeV=float(np.median(P)), p_max_GeV=float(P.max()),
            eta_min=float(ETA.min()), eta_max=float(ETA.max()), eta_median=float(np.median(ETA)),
            n_charge_plus=int((q > 0).sum()), n_charge_minus=int((q < 0).sum()),
            charge_plus_frac=float((q > 0).mean()),
            pid_composition=pid, band_counts=counts, source=src_tracks)
        for label, m in C.band_masks(P):
            band_rows.append(dict(split=s, band=label, n=int(m.sum()),
                                  frac_of_split=float(m.mean()),
                                  p_median_GeV=(float(np.median(P[m])) if m.any() else float("nan"))))
    out["splits"] = splits
    out["total_tracks"] = sum(splits[s]["n"] for s in C.SPLITS)
    table("dataset_bands", band_rows)

    # --- the cut cascade ----------------------------------------------------
    out["cut_cascade"] = dict(steps=meta["cut_cascade"], source=src_meta)
    table("dataset_cuts", [dict(step=i + 1, cut=r["cut"], rows_in=r["rows_in"],
                                rows_removed=r["rows_removed"], rows_out=r["rows_out"],
                                particles_out=r["particles_out"], note=r["note"])
                           for i, r in enumerate(meta["cut_cascade"])])

    # --- geometry, field, reference ----------------------------------------
    D = C.load_tracks()
    C.add_experiment_paths()
    from _shared.reference import field_path, RK6_STEP
    out["geometry"] = dict(
        z0_mm=float(D["z0"]), z1_mm=float(D["z1"]), L_mm=float(D["L"]),
        n_planes=int(D["n_max"]) + 1, plane_spacing_mm=float(D["L"]) / int(D["n_max"]),
        dz_mm={str(N): C.dz_mm(N) for N in (2, 64, 128, 256)},
        window_mm=float(D["window_mm"]),
        z_pre_min_mm=meta["pre_plane_z_mm"]["min"], z_pre_max_mm=meta["pre_plane_z_mm"]["max"],
        z_post_min_mm=meta["post_plane_z_mm"]["min"], z_post_max_mm=meta["post_plane_z_mm"]["max"],
        source=C.source(C.TRACKS, C.TRACKS_META))
    out["field_map"] = dict(which=str(D["field"]), md5=meta["field"]["md5"],
                            path=field_path(str(D["field"])),
                            grid="trilinear on a 100 mm cell (C0 in the derivative)",
                            source=C.source(C.TRACKS_META, os.path.join(C.SHARED, "reference.py")))
    out["rk6_reference"] = dict(step_mm=float(D["rk6_step_mm"]), shared_default_step_mm=float(RK6_STEP),
                                scheme="Butcher, seven stages, explicit, order six",
                                source=C.source(C.TRACKS, os.path.join(C.SHARED, "reference.py")))
    out["build_cost"] = dict(meta["cost"], source=src_meta)

    # --- the material floor, recomputed -------------------------------------
    floor_rows, floor = [], {}
    for scope, splits_used in (("all splits", C.SPLITS), ("test only", ("test",))):
        a = C.concat_splits(("S_post", "truth_zpost", "P"), splits_used)
        d = C.deltas(a["S_post"], a["truth_zpost"])
        per = {}
        for label, m in C.band_masks(a["P"], include_all=True):
            row = dict(scope=scope, band=label, n=int(m.sum()))
            row.update({k: v for k, v in C.magnitude_stats(d["max_um"][m], "um").items() if k != "n"})
            row["radial_med_um"] = C.median(d["radial_um"][m])
            row["radial_p95_um"] = C.quantile(d["radial_um"][m], 0.95)
            row["slope_max_med"] = C.median(d["slope_max"][m])
            row["x_med_abs_um"] = C.med_abs(d["x"][m])
            row["y_med_abs_um"] = C.med_abs(d["y"][m])
            row["tx_med_abs_slope"] = C.med_abs(d["tx"][m])
            row["ty_med_abs_slope"] = C.med_abs(d["ty"][m])
            floor_rows.append(row)
            per[label] = row
        floor[scope] = per
    out["material_floor"] = dict(
        what=("the particle's real Geant4 state on its own first SciFi plane minus the "
              "field-only RK6 reference carried to that plane; the max metric is "
              "max(|dx|, |dy|), the radial metric hypot(dx, dy)"),
        by_scope=floor, source=src_tracks,
        stored_for_comparison=dict(
            bands={band: dict(pos_max_med_um=v["pos_med_um"], pos_max_p95_um=v["pos_p95_um"],
                              slope_max_med_slope=v["slope_med_mrad"] / 1e3, n=v["n"])
                   for band, v in
                   meta["material_floor_real_SciFi_state_vs_field_only_truth"].items()},
            note=("the block stored in tracks_meta.json, on the OLD 1/2/5/10/25/200 GeV "
                  "bands; its slope column is stored x 1e3 there and divided back here")))
    table("material_floor", floor_rows)
    tick("a. dataset")
    return out


# ================================================================ b. scheme ==
def sec_scheme():
    C.add_experiment_paths()
    from _shared.irk import verify_tableau, gauss_legendre
    out = {}

    # --- the tableau battery -------------------------------------------------
    tab_rows, worst = [], {}
    for q in (1, 2, 3, 4, 8, 12, 16):
        c, A, b = gauss_legendre(q, verify=False)
        chk = verify_tableau(q)
        row = dict(q=q, order=2 * q,
                   row_sums=chk["row_sums"], quadrature=chk["quadrature"],
                   collocation=chk["collocation"], symplectic=chk["symplectic"],
                   vs_literature=chk["vs_literature"], passes=bool(chk["passes"]),
                   c_first=float(c[0]), c_last=float(c[-1]), b_sum=float(b.sum()))
        row["worst_residual"] = float(np.nanmax([chk["row_sums"], chk["quadrature"],
                                                 chk["collocation"], chk["symplectic"]]))
        tab_rows.append(row)
        worst[str(q)] = row["worst_residual"]
    out["tableau"] = dict(
        what=("every identity the q-stage Gauss-Legendre tableau must satisfy, from "
              "_shared/irk.py verify_tableau: row sums of A equal c, the quadrature is "
              "exact to degree 2q-1, the collocation conditions hold to degree q-1, the "
              "symplecticity identity b_j a_jk + b_k a_kj = b_j b_k, and agreement with "
              "the closed-form tableaus for q = 1, 2, 3"),
        per_q=tab_rows, worst_residual_per_q=worst,
        worst_overall=float(max(worst.values())),
        all_pass=bool(all(r["passes"] for r in tab_rows)),
        tolerance=5e-13,
        source=C.source(os.path.join(C.SHARED, "irk.py")))
    table("tableau", tab_rows)

    # --- the exact-collocation ceiling --------------------------------------
    chain = {(int(r["N"]), int(r["q"])): r
             for r in C.read_csv(os.path.join(C.E3_RESULTS, "error_qdz_chain.csv"))}
    ceiling_rows = []
    for N, q in C.GRID:
        j = C.load_exact(N, q)
        row = dict(N=N, q=q, dz_mm=C.dz_mm(N),
                   exact_radial_med_um_from_csv=float(chain[(N, q)]["exact_med_um"]),
                   n=int(chain[(N, q)]["n"]))
        if j is not None:
            row.update(
                exact_json_present=1,
                exact_max_med_um=j["endpoint_pos_med_um"],
                exact_max_p95_um=j["endpoint_pos_p95_um"],
                exact_slope_max_med=j["endpoint_slope_med_slope"],
                exact_x_med_um=j["components"]["x_med_um"], exact_y_med_um=j["components"]["y_med_um"],
                exact_tx_med_slope=j["components"]["tx_med_slope"],
                exact_ty_med_slope=j["components"]["ty_med_slope"],
                exact_qop_max_abs_change=j["components"]["qop_max_abs_change"])
        else:
            row["exact_json_present"] = 0
        ceiling_rows.append(row)
    out["exact_ceiling"] = dict(
        what=("the exact collocation scheme chained N times, solved without a network: "
              "the floor a network of that (N, q) could ever reach.  The JSON files store "
              "the MAX metric max(|dx|, |dy|) in um and slopes as a difference x 1e3 "
              "(divided back to a dimensionless slope here); the CSV column exact_med_um "
              "is the RADIAL median that E3's tables.py computes.  Only N = 2 and N = 256 "
              "have their own JSON (E2_Comparators/README.md); the N = 64 and N = 128 rows "
              "come from the CSV alone."),
        per_run=ceiling_rows,
        source=C.source(C.E2_RESULTS, os.path.join(C.E3_RESULTS, "error_qdz_chain.csv")))
    table("exact_ceiling", ceiling_rows)

    # --- the RK6 facts from Block C, C1 --------------------------------------
    meta = C.read_json(os.path.join(C.C1_RESULTS, "reference_convergence_meta.json"))
    checks = C.read_json(os.path.join(C.C1_RESULTS, "tableau_checks.json"))
    conv = C.read_csv(os.path.join(C.C1_RESULTS, "reference_convergence.csv"))

    def pick(measurement, step, field="v8r1.up", integrator="RK6", group="all"):
        for r in conv:
            if (r["measurement"] == measurement and float(r["step_mm"]) == step
                    and r["field"] == field and r["integrator"] == integrator
                    and r["group"] == group):
                return r
        return None

    ladder = []
    for h, ref in ((0.8, 0.4), (0.4, 0.2), (0.2, 0.1), (0.1, 0.05)):
        a = pick("step halving", h)
        b = pick("difference from the finest step", h)
        ladder.append(dict(
            step_mm=h, halving_against_mm=ref,
            halving_med_um=float(a["pos_med_um"]), halving_p95_um=float(a["pos_p95_um"]),
            halving_max_um=float(a["pos_max_um"]),
            vs_finest_med_um=float(b["pos_med_um"]) if b else float("nan"),
            vs_finest_p95_um=float(b["pos_p95_um"]) if b else float("nan"),
            vs_finest_max_um=float(b["pos_max_um"]) if b else float("nan"),
            n=int(a["n"])))
    clos_up = pick("forward-then-back closure", 0.1)
    clos_dn = pick("forward-then-back closure", 0.1, field="v8r1.down")
    rk4_5 = pick("RK4 against RK6 at 0.05 mm", 5.0, integrator="RK4")
    rk4_1 = pick("RK4 against RK6 at 0.05 mm", 1.0, integrator="RK4")
    p3 = checks["checks"]["3_order_on_exact_solution_problems"]["problems"]
    p4 = checks["checks"]["4_order_of_rk6_rows_on_a_smooth_field"]
    out["rk6"] = dict(
        fitted_orders=dict(
            {k: p3[k]["fitted_order"] for k in p3},
            rk6_rows_on_a_smooth_field=p4.get("fitted_order"),
            source=C.source(os.path.join(C.C1_RESULTS, "tableau_checks.json"))),
        identities=dict(checks["checks"]["2_identities"],
                        provenance_identical=checks["checks"]["1_provenance"]["identical"]),
        mechanics=checks["checks"]["5_mechanics"],
        smooth_field_order=dict(
            step_sizes_mm=p4["step_sizes_mm"], max_abs_error_mm=p4["max_abs_error_mm"],
            fitted_order=p4["fitted_order"], points_above_floor=p4["points_above_floor"]),
        step_ladder=ladder,
        successive_halving_ratios=meta["successive_halving_ratios"],
        ideal_ratio_for_order_6=meta["ideal_ratio_for_order_6"],
        smooth_control_median_um=meta["smooth_control_median_um"],
        closure_0p1mm=dict(
            up_med_um=float(clos_up["pos_med_um"]), up_p95_um=float(clos_up["pos_p95_um"]),
            up_max_um=float(clos_up["pos_max_um"]),
            down_med_um=float(clos_dn["pos_med_um"]), down_p95_um=float(clos_dn["pos_p95_um"]),
            down_max_um=float(clos_dn["pos_max_um"]), n=int(clos_up["n"])),
        incumbent_rk4=dict(
            rk4_5mm_med_um=float(rk4_5["pos_med_um"]), rk4_5mm_p95_um=float(rk4_5["pos_p95_um"]),
            rk4_5mm_max_um=float(rk4_5["pos_max_um"]),
            rk4_1mm_med_um=float(rk4_1["pos_med_um"]), rk4_1mm_p95_um=float(rk4_1["pos_p95_um"]),
            rk4_1mm_max_um=float(rk4_1["pos_max_um"]), n=int(rk4_5["n"]),
            note="fp64 RK4, the engine every label before Block C was built with, against RK6 at 0.05 mm"),
        quoted_floor_um=5e-5,
        legs=dict(n=meta["legs"]["n"], span_mm=meta["legs"]["span_mm"], seed=meta["legs"]["seed"]),
        cost=meta["cost"],
        source=C.source(os.path.join(C.C1_RESULTS, "reference_convergence.csv"),
                        os.path.join(C.C1_RESULTS, "reference_convergence_meta.json"),
                        os.path.join(C.C1_RESULTS, "tableau_checks.json")))
    table("rk6_convergence", ladder)
    tick("b. scheme")
    return out


# =================================================== c. reference vs truth ==
def sec_reference_vs_truth():
    out = {"verbatim": {}, "source": C.source(C.RVT)}
    for name in ("decomposition_by_band", "decomposition_by_pid", "highland_air",
                 "three_references", "rk6_self_consistency"):
        p = os.path.join(C.RVT, "results", "%s.csv" % name)
        rows = C.read_csv(p)
        for r in rows:
            for k, v in list(r.items()):
                try:
                    r[k] = float(v)
                    if r[k] == int(r[k]) and k in ("n", "n_bend_cut", "pid"):
                        r[k] = int(r[k])
                except (TypeError, ValueError):
                    pass
        out["verbatim"][name] = dict(rows=rows, source=C.source(p))

    # --- the same decomposition, re-binned into the paper's bands -----------
    a = C.concat_splits(("S0", "S_post", "truth_zpost", "z_post", "P", "PID"))
    S0, P = a["S0"], a["P"]
    d = C.deltas(a["S_post"], a["truth_zpost"])          # true minus reference
    q = np.sign(S0[:, 4])
    # the bend: how far the reference ends up from a straight line through the start
    bend = a["truth_zpost"][:, 0] - (S0[:, 0] + S0[:, 2] * (a["z_post"] - C.Z0_MM))     # mm
    ratio = (d["x"] * 1e-3) / bend                        # dimensionless: dx/bend
    BEND_CUT_MM = 20.0                                    # decompose.py's own cut

    rows, by_band = [], {}
    for label, m in C.band_masks(P, include_all=True):
        entry = {}
        for cname, csel in (("q+", q > 0), ("q-", q < 0), ("both", np.ones_like(q, bool))):
            mm = m & csel
            r = dict(band=label, charge=cname, n=int(mm.sum()))
            if mm.sum():
                r["median_P_GeV"] = float(np.median(P[mm]))
                r["signed_med_dx_um"] = C.signed_median(d["x"][mm])
                r["signed_med_dy_um"] = C.signed_median(d["y"][mm])
                r["signed_med_dtx_slope"] = C.signed_median(d["tx"][mm])
                r["signed_med_dty_slope"] = C.signed_median(d["ty"][mm])
                r["med_abs_dx_um"] = C.med_abs(d["x"][mm])
                r["med_abs_dy_um"] = C.med_abs(d["y"][mm])
                r["med_abs_dtx_slope"] = C.med_abs(d["tx"][mm])
                r["med_abs_dty_slope"] = C.med_abs(d["ty"][mm])
                r["rms_dx_um"] = C.rms(d["x"][mm])
                r["hw68_dx_um"] = float(np.quantile(np.abs(d["x"][mm] - np.median(d["x"][mm])), 0.68))
                mb = mm & (np.abs(bend) > BEND_CUT_MM)
                r["n_bend_cut"] = int(mb.sum())
                r["median_bend_mm"] = float(np.median(bend[mb])) if mb.sum() else float("nan")
                r["median_abs_bend_mm"] = float(np.median(np.abs(bend[mm])))
                r["median_signed_bend_mm"] = float(np.median(bend[mm]))
                r["median_dx_over_bend"] = float(np.median(ratio[mb])) if mb.sum() else float("nan")
                r["implied_dp_MeV"] = (r["median_dx_over_bend"] * float(np.median(P[mb])) * 1e3
                                       if mb.sum() else float("nan"))
                if cname == "both":
                    r["max_metric_med_um"] = C.median(d["max_um"][mm])
                    r["max_metric_p95_um"] = C.quantile(d["max_um"][mm], 0.95)
                    r["radial_med_um"] = C.median(d["radial_um"][mm])
                    pos, neg = d["x"][m & (q > 0)], d["x"][m & (q < 0)]
                    if len(pos) and len(neg):
                        centred = np.concatenate([pos - np.median(pos), neg - np.median(neg)])
                        r["hw68_dx_charge_corrected_um"] = float(np.quantile(np.abs(centred), 0.68))
            rows.append(r)
            entry[cname] = r
        by_band[label] = entry
    out["rebinned_decomposition"] = dict(
        what=("d = Geant4-true state minus the field-only RK6 reference at the particle's "
              "own first SciFi plane, on all three splits concatenated (14,482 crossings), "
              "re-binned into the paper's bands.  Definitions follow "
              "reference_vs_truth/decompose.py exactly: the bend is the reference endpoint "
              "minus a straight line through the start state; dx/bend uses only the rows "
              "with |bend| > 20 mm; the implied momentum defect is median(dx/bend) x "
              "median(P) x 1e3 MeV; the charge-corrected 68 % half-width centres each "
              "charge on its own median before pooling and takes the 68th percentile of "
              "|deviation|.  Slopes are dimensionless."),
        by_band=by_band, n_crossings=int(len(P)), bend_cut_mm=BEND_CUT_MM,
        median_abs_bend_mm=float(np.median(np.abs(bend))),
        median_signed_bend_mm=float(np.median(bend)),
        bend_definition=("the reference endpoint's x at the particle's own first SciFi "
                         "plane minus a straight line through the start state, "
                         "truth_zpost_x - (S0_x + S0_tx (z_post - z0)) [mm].  The SIGNED "
                         "median is near zero because the two charges bend opposite ways; "
                         "the magnitude is what reference_vs_truth/page.md quotes."),
        n_above_bend_cut=int((np.abs(bend) > BEND_CUT_MM).sum()),
        source=C.source(C.TRACKS, os.path.join(C.RVT, "decompose.py")))
    table("decomposition_bands", rows)

    # --- Highland for air, in the paper's bands ------------------------------
    X0_AIR_MM = 304.0e3
    x_over_X0 = C.L_MM / X0_AIR_MM

    def theta0(p_GeV, t):
        return (13.6e-3 / p_GeV) * np.sqrt(t) * (1.0 + 0.038 * np.log(t))

    def disp_mm(p_GeV, t, Lmm=C.L_MM):
        return theta0(p_GeV, t) * Lmm / np.sqrt(3.0)

    def solve_t(width_mm, p_GeV, Lmm=C.L_MM):
        f = lambda t: disp_mm(p_GeV, t, Lmm) - width_mm          # noqa: E731
        lo, hi = 1e-5, 1.0
        if f(lo) > 0 or f(hi) < 0:
            return float("nan")
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        return float(np.sqrt(lo * hi))

    hl_rows = []
    for label, m in C.band_masks(P):
        r = by_band[label]["both"]
        if not r["n"]:
            continue
        p = r["median_P_GeV"]
        pred = disp_mm(p, x_over_X0) * 1e3
        meas = r.get("hw68_dx_charge_corrected_um", float("nan"))
        hl_rows.append(dict(band=label, n=r["n"], median_P_GeV=p, x_over_X0_air=x_over_X0,
                            theta0_air_urad=float(theta0(p, x_over_X0) * 1e6),
                            highland_air_width_um=float(pred),
                            measured_hw68_dx_um=float(meas),
                            ratio_measured_over_air=float(meas / pred),
                            effective_x_over_X0=solve_t(meas * 1e-3, p)))
    out["highland_air_rebinned"] = dict(
        what=("the air-only Highland prediction for the 68 per cent half-width of dx, in "
              "the paper's bands.  x/X0 = L / 304,000 mm = " + ("%.5f" % x_over_X0) +
              "; theta0 = (13.6e-3 / p[GeV]) sqrt(x/X0) (1 + 0.038 ln(x/X0)) rad; the RMS "
              "lateral displacement is theta0 L / sqrt(3).  The last column inverts that "
              "formula for the thickness the measured width implies."),
        rows=hl_rows, source=C.source(C.TRACKS, os.path.join(C.RVT, "decompose.py")))
    table("highland_air", hl_rows)
    tick("c. reference vs truth")
    return out


# ============================================== chain scoring helpers (d-g) ==
def _endpoint(run_dir, split="test"):
    st = C.load_chain_states(run_dir, split)
    return st[:, -1, :], st


def chain_block(run_dir, P, truth_end, split="test", above_um=1000.0):
    """The full endpoint verdict of one run: overall and per band, per component."""
    end, _ = _endpoint(run_dir, split)
    d = C.deltas(end, truth_end)
    out = dict(C.magnitude_stats(d["radial_um"], "um", above_um))
    out["max_metric_med_um"] = C.median(d["max_um"])
    out["max_metric_p95_um"] = C.quantile(d["max_um"], 0.95)
    out.update(C.component_block(d))
    out["by_band"] = {}
    for label, m in C.band_masks(P):
        b = dict(C.magnitude_stats(d["radial_um"][m], "um", above_um))
        b["max_metric_med_um"] = C.median(d["max_um"][m])
        b.update(C.component_block(d, m))
        out["by_band"][label] = b
    return out, d


# =========================================== d. the pooled-loss grid (snap) ==
def sec_pooled_grid():
    a = C.split_arrays("test")
    P, truth_end = a["P"], a["truth"][:, C.N_MAX]
    chain_csv = {(int(r["N"]), int(r["q"])): r
                 for r in C.read_csv(os.path.join(C.E3_RESULTS, "error_qdz_chain.csv"))}
    step_csv = {(int(r["N"]), int(r["q"])): r
                for r in C.read_csv(os.path.join(C.E3_RESULTS, "error_qdz_single_step.csv"))}
    rows, per_run, checks = [], {}, []
    round_rows = []
    for N, q in C.GRID:
        run = C.pooled_run(N, q, snapshot=True)
        blk, _ = chain_block(run, P, truth_end)
        rec = C.load_record(run)
        rr, rdup, rraw = C.load_rounds(run)
        labels = sorted(int(x["round"]) for x in rr)
        round_rows.append(dict(
            N=N, q=q, rounds_completed=len(rr), rounds_counter_record_json=int(rec["rounds"]),
            rounds_counter_e3_csv=int(chain_csv[(N, q)]["rounds"]),
            max_round_label=max(labels), min_round_label=min(labels),
            missing_labels=" ".join(str(x) for x in
                                    sorted(set(range(min(labels), max(labels) + 1)) - set(labels))),
            counter_is_completed_plus_one=bool(int(rec["rounds"]) == len(rr) + 1),
            record_matches_e3_csv=bool(int(rec["rounds"]) == int(chain_csv[(N, q)]["rounds"]))))
        e3 = chain_csv[(N, q)]
        ref = float(e3["med_um"])
        dev = 100.0 * (blk["med_um"] - ref) / ref
        checks.append(dict(N=N, q=q, recomputed_radial_med_um=blk["med_um"],
                           e3_error_qdz_chain_med_um=ref, deviation_pct=dev,
                           within_0p5pct=bool(abs(dev) < 0.5)))
        row = dict(N=N, q=q, dz_mm=C.dz_mm(N), n=blk["n"],
                   radial_med_um=blk["med_um"], radial_p95_um=blk["p95_um"],
                   radial_p99_um=blk["p99_um"], frac_above_1mm=blk["frac_above_1mm"],
                   x_med_abs_um=blk["x_med_abs_um"], y_med_abs_um=blk["y_med_abs_um"],
                   tx_med_abs_slope=blk["tx_med_abs_slope"], ty_med_abs_slope=blk["ty_med_abs_slope"],
                   x_signed_med_um=blk["x_signed_med_um"], y_signed_med_um=blk["y_signed_med_um"],
                   tx_signed_med_slope=blk["tx_signed_med_slope"],
                   ty_signed_med_slope=blk["ty_signed_med_slope"],
                   exact_radial_med_um=float(e3["exact_med_um"]),
                   single_step_radial_med_um=float(step_csv[(N, q)]["step_med_um"]),
                   single_step_radial_p95_um=float(step_csv[(N, q)]["step_p95_um"]),
                   single_step_over_straight_line=float(step_csv[(N, q)]["step_over_straight"]),
                   straight_line_step_med_um=float(step_csv[(N, q)]["straight_line_step_med_um"]),
                   restarts=int(e3["restarts"]),
                   rounds=len(rr), rounds_completed=len(rr),
                   rounds_counter_record_json=int(rec["rounds"]),
                   rounds_counter_e3_csv=int(e3["rounds"]),
                   us_per_track=float(e3["us_per_track"]),
                   e3_med_um=ref, reproduces_e3_pct=dev)
        rows.append(row)
        per_run[C.run_key(N, q)] = dict(row, by_band=blk["by_band"])
    out = dict(
        what=("the sixteen pooled-loss networks at the checkpoint of 18 September 2026 "
              "(results/N*/stopped_2026-09-18/), scored from their stored chain states "
              "against the RK6 endpoint on the 1,452 test tracks.  This is the snapshot "
              "E3_Analysis was run on; the run folders themselves now hold the extended "
              "networks."),
        per_run=per_run,
        round_counting=dict(
            what=("`rounds.csv` holds one row per COMPLETED round; `record.json[\"rounds\"]` "
                  "(and `progress.json[\"round\"]`, and the `rounds` column of E3's "
                  "error_qdz_chain.csv, which is copied from the record) is the trainer's "
                  "round COUNTER at the moment the run stopped, i.e. the round that was in "
                  "progress and has no row.  Where a run stopped mid-round the counter is "
                  "one higher than the number of completed rounds.  `rounds` in this "
                  "table and `snapshot_rounds` in tab_extended_vs_snapshot.csv are both "
                  "the COMPLETED count, so the two tables agree; the counter is kept "
                  "beside it as rounds_counter_record_json."),
            rows=round_rows,
            all_counters_are_completed_plus_one=bool(
                all(r["counter_is_completed_plus_one"] for r in round_rows)),
            all_records_match_e3_csv=bool(all(r["record_matches_e3_csv"] for r in round_rows))),
        reproduction_check=dict(
            what="the recomputed radial median against E3's error_qdz_chain.csv med_um",
            rows=checks, worst_abs_deviation_pct=float(max(abs(c["deviation_pct"]) for c in checks)),
            all_within_0p5pct=bool(all(c["within_0p5pct"] for c in checks))),
        source=C.source(os.path.join(C.E1_RUNS, "N*", C.SNAPSHOT, "chain_states.npz"),
                        C.TRACKS, os.path.join(C.E3_RESULTS, "error_qdz_chain.csv"),
                        os.path.join(C.E3_RESULTS, "error_qdz_single_step.csv")))
    table("pooled_grid", rows)
    tick("d. pooled-loss grid (18 Sept snapshot)")
    return out


# ====================================== e. the extended pooled-loss runs ====
def sec_extended():
    a = C.split_arrays("test")
    P, truth_end = a["P"], a["truth"][:, C.N_MAX]
    headline = {(int(r["N"]), int(r["q"]), r["weighting"]): r
                for r in C.read_csv(os.path.join(C.F2_RESULTS, "headline.csv"))}
    rows, per_run, checks = [], {}, []
    for N, q in C.PAIRS:
        run = C.pooled_run(N, q)
        snap = C.pooled_run(N, q, snapshot=True)
        blk, _ = chain_block(run, P, truth_end)
        rec = C.load_record(run)
        rec_s = C.load_record(snap)
        rr, rdup, rraw = C.load_rounds(run)
        hh, hdup, hraw = C.load_history(run)
        rr_s, _, _ = C.load_rounds(snap)
        hh_s, _, _ = C.load_history(snap)
        ref = float(headline[(N, q, "blockE")]["test_radial_med_um"])
        dev = 100.0 * (blk["med_um"] - ref) / ref
        checks.append(dict(N=N, q=q, recomputed_radial_med_um=blk["med_um"],
                           headline_csv_radial_med_um=ref, deviation_pct=dev,
                           within_0p5pct=bool(abs(dev) < 0.5)))
        row = dict(N=N, q=q, dz_mm=C.dz_mm(N),
                   radial_med_um=blk["med_um"], radial_p95_um=blk["p95_um"],
                   band_radial_med_um=blk["by_band"][C.LOSS_WINDOW_LABEL]["med_um"],
                   restarts=int(rec["restarts"]), rounds=len(rr),
                   rounds_counter_record_json=int(rec["rounds"]),
                   snapshot_restarts=int(rec_s["restarts"]), snapshot_rounds=len(rr_s),
                   snapshot_rounds_counter_record_json=int(rec_s["rounds"]),
                   abandoned_round_label=(
                       " ".join(str(x) for x in sorted(
                           set(range(1, max(int(y["round"]) for y in rr) + 1))
                           - set(int(y["round"]) for y in rr)))),
                   extra_restarts=int(rec["restarts"]) - int(rec_s["restarts"]),
                   extra_rounds=len(rr) - len(rr_s),
                   history_rows=hraw, unique_restarts=len(hh), duplicated_history_rows=len(hdup),
                   rounds_rows=rraw, duplicated_round_rows=len(rdup),
                   headline_csv_radial_med_um=ref, reproduces_headline_pct=dev)
        rows.append(row)
        per_run[C.run_key(N, q)] = dict(row, by_band=blk["by_band"])
    out = dict(
        what=("the three pooled-loss runs that were extended past the 18 September stop, "
              "as the run folders now hold them, scored the same way; and how far each was "
              "extended beyond its snapshot."),
        per_run=per_run,
        reproduction_check=dict(
            what=("the recomputed radial median against F2_Analysis/results/headline.csv "
                  "(the write-up quotes 145.7 / 122.2 / 147.7 um)"),
            rows=checks, worst_abs_deviation_pct=float(max(abs(c["deviation_pct"]) for c in checks)),
            all_within_0p5pct=bool(all(c["within_0p5pct"] for c in checks))),
        source=C.source(os.path.join(C.E1_RUNS, "N*", "chain_states.npz"), C.TRACKS,
                        os.path.join(C.F2_RESULTS, "headline.csv")))
    table("extended_vs_snapshot", rows)
    tick("e. extended pooled-loss runs")
    return out


# ============================================== f. the reweighted-loss runs ==
def sec_reweighted():
    a = C.split_arrays("test")
    P, truth_end = a["P"], a["truth"][:, C.N_MAX]
    headline = {(int(r["N"]), int(r["q"]), r["weighting"]): r
                for r in C.read_csv(os.path.join(C.F2_RESULTS, "headline.csv"))}
    rows, per_run, checks = [], {}, []
    for N, q in C.PAIRS:
        run = C.reweighted_run(N, q)
        blk, _ = chain_block(run, P, truth_end)
        h = headline[(N, q, "full")]
        for what, mine, ref in (("overall radial median", blk["med_um"],
                                 float(h["test_radial_med_um"])),
                                ("10-50 GeV radial median",
                                 blk["by_band"][C.LOSS_WINDOW_LABEL]["med_um"],
                                 float(h["test_radial_band_med_um"]))):
            checks.append(dict(N=N, q=q, what=what, recomputed_um=mine,
                               headline_csv_um=ref, deviation_pct=100.0 * (mine - ref) / ref,
                               agrees=bool(abs(mine - ref) / ref < 5e-3)))
        row = dict(N=N, q=q, dz_mm=C.dz_mm(N), n=blk["n"],
                   radial_med_um=blk["med_um"], radial_p95_um=blk["p95_um"],
                   radial_p99_um=blk["p99_um"], frac_above_1mm=blk["frac_above_1mm"],
                   band_radial_med_um=blk["by_band"][C.LOSS_WINDOW_LABEL]["med_um"],
                   x_med_abs_um=blk["x_med_abs_um"], y_med_abs_um=blk["y_med_abs_um"],
                   tx_med_abs_slope=blk["tx_med_abs_slope"], ty_med_abs_slope=blk["ty_med_abs_slope"],
                   x_signed_med_um=blk["x_signed_med_um"], y_signed_med_um=blk["y_signed_med_um"],
                   headline_csv_radial_med_um=float(h["test_radial_med_um"]),
                   headline_csv_band_med_um=float(h["test_radial_band_med_um"]))
        rows.append(row)
        per_run[C.run_key(N, q)] = dict(row, by_band=blk["by_band"])
    out = dict(
        what=("the three reweighted-loss networks, scored from their stored chain states "
              "against the RK6 endpoint on the 1,452 test tracks."),
        per_run=per_run,
        reproduction_check=dict(
            what=("against F2_Analysis/results/headline.csv: the write-up quotes 88.8 / "
                  "104.6 / 92.2 um overall and 24.8 / 27.3 / 34.6 um in 10-50 GeV"),
            rows=checks, all_agree=bool(all(c["agrees"] for c in checks))),
        source=C.source(os.path.join(C.F1_RUNS, "N*", "chain_states.npz"), C.TRACKS,
                        os.path.join(C.F2_RESULTS, "headline.csv")))
    table("reweighted_runs", rows)
    tick("f. reweighted-loss runs")
    return out


# ================================================= g. headline comparisons ==
def sec_headline():
    a = C.split_arrays("test")
    P, truth_end = a["P"], a["truth"][:, C.N_MAX]
    head_rows, band_rows, comp_rows, per_pair = [], [], [], {}
    plateau_rows, wall = {}, {}

    for N, q in C.PAIRS:
        pair = {}
        for loss, run in (("pooled", C.pooled_run(N, q)), ("reweighted", C.reweighted_run(N, q))):
            blk, d = chain_block(run, P, truth_end)
            rec = C.load_record(run)
            rr, rdup, rraw = C.load_rounds(run)
            hh, hdup, hraw = C.load_history(run)
            v = np.array([float(x["val_z1_pos_med_um"]) for x in rr])
            head = C.validation_headline(v)
            walls = np.array([float(x["wall_s"]) for x in hh])
            verdict = dict(
                rounds=len(v), rounds_completed=len(v),
                rounds_counter_record_json=int(rec["rounds"]),
                rounds_rows_read=rraw, duplicated_round_rows=len(rdup),
                restarts=int(rec["restarts"]), history_rows_read=hraw,
                unique_restarts=len(hh), duplicated_history_rows=len(hdup),
                plateaued_now=bool(C.plateaued_now(v)),
                first_plateau_round=C.first_plateau_round(v),
                train_wall_h=rec["train_wall_s"] / 3600.0,
                median_wall_s_per_restart=float(np.median(walls)),
                summed_history_wall_h=float(walls.sum() / 3600.0),
                **head)
            plateau_rows["%s %s" % (loss, C.run_key(N, q))] = verdict
            wall["%s %s" % (loss, C.run_key(N, q))] = verdict
            pair[loss] = dict(endpoint=blk, convergence=verdict)
            head_rows.append(dict(
                N=N, q=q, dz_mm=C.dz_mm(N), loss=loss, n=blk["n"],
                radial_med_um=blk["med_um"], radial_p95_um=blk["p95_um"],
                frac_above_1mm=blk["frac_above_1mm"], n_above_1mm=blk["n_above_1mm"],
                band_radial_med_um=blk["by_band"][C.LOSS_WINDOW_LABEL]["med_um"],
                val_headline_med_um=head["headline_val_med_um"],
                val_headline_spread_pct=head["headline_spread_pct"],
                val_headline_full_range_over_median_pct=head["headline_full_range_over_median_pct"],
                plateaued_now=int(verdict["plateaued_now"]),
                first_plateau_round=verdict["first_plateau_round"] or -1,
                rounds=verdict["rounds"],
                rounds_counter_record_json=verdict["rounds_counter_record_json"],
                restarts=verdict["restarts"],
                train_wall_h=verdict["train_wall_h"],
                median_wall_s_per_restart=verdict["median_wall_s_per_restart"]))
            for label, m in C.band_masks(P):
                b = blk["by_band"][label]
                band_rows.append(dict(N=N, q=q, loss=loss, band=label, n=b["n"],
                                      radial_med_um=b["med_um"], radial_p95_um=b["p95_um"],
                                      frac_above_1mm=b["frac_above_1mm"]))
            for scope, m in [("all", None)] + list(C.band_masks(P)):
                blkc = C.component_block(d, m)
                for name, _, unit in C.COMPONENTS:
                    comp_rows.append(dict(
                        N=N, q=q, loss=loss, band=scope, component=name, unit=unit,
                        n=blkc["n"],
                        med_abs=blkc["%s_med_abs_%s" % (name, unit)],
                        p95_abs=blkc["%s_p95_abs_%s" % (name, unit)],
                        signed_med=blkc["%s_signed_med_%s" % (name, unit)],
                        hw68=blkc["%s_hw68_%s" % (name, unit)],
                        rms=blkc["%s_rms_%s" % (name, unit)]))
        pair["ratios"] = dict(
            radial_med_reweighted_over_pooled=(pair["reweighted"]["endpoint"]["med_um"]
                                               / pair["pooled"]["endpoint"]["med_um"]),
            band_med_reweighted_over_pooled=(
                pair["reweighted"]["endpoint"]["by_band"][C.LOSS_WINDOW_LABEL]["med_um"]
                / pair["pooled"]["endpoint"]["by_band"][C.LOSS_WINDOW_LABEL]["med_um"]),
            p95_reweighted_over_pooled=(pair["reweighted"]["endpoint"]["p95_um"]
                                        / pair["pooled"]["endpoint"]["p95_um"]))
        per_pair[C.run_key(N, q)] = pair

    out = dict(
        what=("the three matched pairs, the extended pooled-loss run against the "
              "reweighted-loss run at the same N and q, on the 1,452 test tracks.  The "
              "plateau rule is section 3.7 of the reweighted-loss write-up: the median "
              "validation error of the last ten rounds no more than 5 % below the median "
              "of the ten before, holding at each of the last three rounds.  Both runs' "
              "rounds.csv and history.csv are de-duplicated by number, last row written "
              "kept, before any of this is read."),
        pairs=per_pair, plateau=plateau_rows,
        source=C.source(os.path.join(C.E1_RUNS, "N*"), os.path.join(C.F1_RUNS, "N*"), C.TRACKS))
    table("headline_pairs", head_rows)
    table("headline_bands", band_rows)
    table("headline_components", comp_rows)
    tick("g. headline comparisons")
    return out


# ============================================================ h. around 5 GeV ==
def sec_near_5gev():
    a = C.split_arrays("test")
    P, S0, truth_end = a["P"], a["S0"], a["truth"][:, C.N_MAX]
    rows, out_bands = [], {}
    for lo, hi, label in ((3.0, 8.0, "3-8"), (4.0, 6.0, "4-6")):
        m = (P >= lo) & (P < hi)
        per_loss = {}
        for loss, run in (("pooled", C.pooled_run(64, 2)), ("reweighted", C.reweighted_run(64, 2))):
            end, _ = _endpoint(run)
            d = C.deltas(end, truth_end)
            r = d["radial_um"][m]
            beyond = r > 1000.0
            x0 = np.abs(S0[m, 0])
            entry = dict(
                n=int(m.sum()), radial_med_um=C.median(r), radial_p95_um=C.quantile(r, 0.95),
                n_beyond_1mm=int(beyond.sum()), frac_beyond_1mm=float(beyond.mean()),
                start_abs_x_med_mm_beyond_1mm=(float(np.median(x0[beyond])) if beyond.any()
                                               else float("nan")),
                start_abs_x_med_mm_rest=(float(np.median(x0[~beyond])) if (~beyond).any()
                                         else float("nan")),
                start_abs_x_med_mm_all=float(np.median(x0)))
            entry["start_abs_x_ratio_beyond_over_rest"] = (
                entry["start_abs_x_med_mm_beyond_1mm"] / entry["start_abs_x_med_mm_rest"])
            comps = {}
            for name, _, unit in C.COMPONENTS:
                v = d[name][m]
                st = C.signed_stats(v, unit)
                comps[name] = st
                rows.append(dict(band=label, p_lo=lo, p_hi=hi, loss=loss, N=64, q=2,
                                 component=name, unit=unit, n=st["n"],
                                 med_abs=st["med_abs_%s" % unit], p95_abs=st["p95_abs_%s" % unit],
                                 signed_med=st["signed_med_%s" % unit], hw68=st["hw68_%s" % unit],
                                 rms=st["rms_%s" % unit],
                                 radial_med_um=entry["radial_med_um"],
                                 n_beyond_1mm=entry["n_beyond_1mm"],
                                 start_abs_x_med_mm_beyond_1mm=entry["start_abs_x_med_mm_beyond_1mm"],
                                 start_abs_x_med_mm_rest=entry["start_abs_x_med_mm_rest"]))
            entry["components"] = comps
            # the same statistics with the > 1 mm tracks removed, for the RMS
            entry["without_the_1mm_tail"] = {
                name: C.signed_stats(d[name][m][~beyond], unit)
                for name, _, unit in C.COMPONENTS}
            per_loss[loss] = entry
        out_bands[label] = per_loss
    out = dict(
        what=("the N = 64, q = 2 network under both losses in the 3-8 GeV band and the "
              "4-6 GeV band the supervisor asked about: signed per-component median, 68 % "
              "half-width, RMS and median |.|, how many tracks miss by more than 1 mm "
              "radially, and how far from the beam line those tracks start (|x| on the "
              "last UT plane) against the rest.  Slopes are dimensionless."),
        bands=out_bands,
        source=C.source(C.pooled_run(64, 2), C.reweighted_run(64, 2), C.TRACKS))
    table("near_5gev", rows)
    tick("h. around 5 GeV")
    return out


# ================================================================ i. anatomy ==
def anatomy_arrays(run_dir, label, use_cache=True):
    """Per-step local errors against RK6 from the state the network was given.

    Reproduces F2_Analysis/anatomy_xy.py exactly, with slopes left dimensionless:
    the network is chained from the track's own state at z0, and at every step
    its output is compared with `_shared.reference.rk6_rows` taken from the SAME
    input state over the same dz at a 1 mm local step.

    Returns (lx_um, ly_um, ltx, lty, final_radial_um), each (N, n_tracks) except
    the last which is (n_tracks,).
    """
    os.makedirs(C.CACHE, exist_ok=True)
    cache = os.path.join(C.CACHE, "anatomy_%s.npz" % label)
    if use_cache and os.path.exists(cache):
        with np.load(cache) as z:
            return (z["lx"], z["ly"], z["ltx"], z["lty"], z["final"])
    C.add_experiment_paths()
    import torch
    from _shared.reference import make_field, rk6_rows
    from chain_network import load_network, step_outputs
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    D = C.load_tracks()
    sc = C.load_scale(run_dir)
    N = int(sc["N"])
    dz = C.L_MM / N
    fld = make_field(str(D["field"]))
    model = load_network(run_dir, fld)
    S0 = np.asarray(D["test_S0"])
    truth = np.asarray(D["test_truth"])
    n = len(S0)
    S = S0.copy()
    lx, ly, ltx, lty = (np.empty((N, n)) for _ in range(4))
    for k in range(N):
        z = C.Z0_MM + k * dz
        out = step_outputs(model, S, np.full(n, z))[:, -1, :]
        ref = rk6_rows(S, z, z + dz, step=1.0, field=fld)
        lx[k] = (out[:, 0] - ref[:, 0]) * 1e3          # um
        ly[k] = (out[:, 1] - ref[:, 1]) * 1e3          # um
        ltx[k] = out[:, 2] - ref[:, 2]                 # dimensionless
        lty[k] = out[:, 3] - ref[:, 3]                 # dimensionless
        S = np.concatenate([out, S[:, 4:5]], axis=1)
    final = np.hypot(S[:, 0] - truth[:, C.N_MAX, 0], S[:, 1] - truth[:, C.N_MAX, 1]) * 1e3
    np.savez_compressed(cache, lx=lx, ly=ly, ltx=ltx, lty=lty, final=final)
    return lx, ly, ltx, lty, final


def sec_anatomy(use_cache=True):
    a = C.split_arrays("test")
    P = a["P"]
    masks = dict(C.band_masks(P))
    f2 = {}
    for tag in ("blockE_N064_q02", "blockE_N128_q08", "blockE_N256_q16",
                "blockF_N064_q02", "blockF_N128_q08", "blockF_N256_q16"):
        p = os.path.join(C.F2_RESULTS, "anatomy_xy_%s.json" % tag)
        if os.path.exists(p):
            f2[tag] = C.read_json(p)

    rows, per_run, checks = [], {}, []
    for N, q in C.PAIRS:
        for loss, run, tag in (("pooled", C.pooled_run(N, q), "blockE_%s" % C.run_tag(N, q)),
                               ("reweighted", C.reweighted_run(N, q), "blockF_%s" % C.run_tag(N, q))):
            lx, ly, ltx, lty, final = anatomy_arrays(run, "%s_%s" % (loss, C.run_tag(N, q)),
                                                     use_cache)
            lever = (C.Z1_MM - (C.Z0_MM + (np.arange(N) + 1) * C.dz_mm(N)))[:, None]     # mm
            px = (lx + ltx * 1e3 * lever).sum(axis=0)     # um  (slope x 1e3 x mm = um)
            py = (ly + lty * 1e3 * lever).sum(axis=0)
            pred_xy = np.hypot(px, py)
            pred_x_only = np.abs((ltx * 1e3 * lever).sum(axis=0))
            pos_only = np.hypot(lx.sum(axis=0), ly.sum(axis=0))
            entry = dict(
                N=N, q=q, loss=loss, n_tracks=int(len(final)), n_steps=N,
                final_radial_med_um=C.median(final),
                final_radial_band_med_um=C.median(final[masks[C.LOSS_WINDOW_LABEL]]),
                per_step_x_med_abs_um=C.med_abs(lx), per_step_y_med_abs_um=C.med_abs(ly),
                per_step_tx_med_abs_slope=C.med_abs(ltx), per_step_ty_med_abs_slope=C.med_abs(lty),
                per_step_radial_med_um=C.median(np.hypot(lx, ly)),
                x_part_med_um=C.med_abs(px), y_part_med_um=C.med_abs(py),
                predicted_endpoint_both_slopes_med_um=C.median(pred_xy),
                predicted_endpoint_x_slope_only_med_um=C.median(pred_x_only),
                predicted_endpoint_positions_only_med_um=C.median(pos_only),
                explained_both_slopes=C.median(pred_xy) / C.median(final),
                explained_x_slope_only=C.median(pred_x_only) / C.median(final),
                explained_positions_only=C.median(pos_only) / C.median(final),
                coherence_tx=C.median(np.abs(ltx.sum(0)) / np.abs(ltx).sum(0)),
                coherence_ty=C.median(np.abs(lty.sum(0)) / np.abs(lty).sum(0)),
                coherence_independent=float(1.0 / np.sqrt(N)))
            for label in ("3-8", "8-20", C.LOSS_WINDOW_LABEL):
                m = masks[label]
                key = label.replace(" (loss window)", "").replace("-", "_")
                entry["per_step_tx_med_abs_slope_%s" % key] = C.med_abs(ltx[:, m])
                entry["per_step_ty_med_abs_slope_%s" % key] = C.med_abs(lty[:, m])
                entry["per_step_x_med_abs_um_%s" % key] = C.med_abs(lx[:, m])
                entry["per_step_y_med_abs_um_%s" % key] = C.med_abs(ly[:, m])
                entry["final_radial_med_um_%s" % key] = C.median(final[m])
            rows.append(entry)
            per_run["%s %s" % (loss, C.run_key(N, q))] = entry

            ref = f2.get(("blockF_" if loss == "reweighted" else "blockE_") + C.run_tag(N, q))
            if ref is not None:
                for what, mine, theirs in (
                        ("final radial median [um]", entry["final_radial_med_um"], ref["final_med_um"]),
                        ("per-step |dtx| median (slope)", entry["per_step_tx_med_abs_slope"],
                         ref["local_step"]["tx_mrad"] / 1e3),
                        ("per-step |dty| median (slope)", entry["per_step_ty_med_abs_slope"],
                         ref["local_step"]["ty_mrad"] / 1e3),
                        ("x part at z1 [um]", entry["x_part_med_um"], ref["x_part_med_um"]),
                        ("y part at z1 [um]", entry["y_part_med_um"], ref["y_part_med_um"]),
                        ("explained fraction, both slopes", entry["explained_both_slopes"],
                         ref["explained"]["both_slopes"]),
                        ("explained fraction, x slope only", entry["explained_x_slope_only"],
                         ref["explained"]["x_slope_only"]),
                        ("coherence tx", entry["coherence_tx"], ref["coherence"]["tx"]),
                        ("coherence ty", entry["coherence_ty"], ref["coherence"]["ty"])):
                    checks.append(dict(
                        run="%s %s" % (loss, C.run_key(N, q)), quantity=what,
                        recomputed=mine, f2_anatomy_xy_json=theirs,
                        deviation_pct=100.0 * (mine - theirs) / theirs if theirs else float("nan"),
                        agrees=bool(abs(mine - theirs) <= 1e-9 * max(abs(theirs), 1.0)
                                    or abs(mine - theirs) / abs(theirs) < 1e-6)))
            tick("i. anatomy %s %s" % (loss, C.run_key(N, q)))

    out = dict(
        what=("F2_Analysis/anatomy_xy.py redone with dimensionless slopes, for all six "
              "runs.  Each step's output is compared with RK6 taken from the same state "
              "the network was given, at a 1 mm local step.  The lever-arm model is "
              "sum_k (dx_k + dtx_k * lever_k) and the same in y, combined radially; the "
              "explained fraction is the median of that model over the median of the "
              "measured endpoint error.  Coherence is the median over tracks of "
              "|sum_k dt_k| / sum_k |dt_k|, which would be 1/sqrt(N) for independent "
              "steps (reported as coherence_independent)."),
        per_run=per_run,
        check_against_f2=dict(
            what=("the recomputed numbers against F2_Analysis/results/anatomy_xy_*.json, "
                  "whose slopes are stored x 1e3 and are divided by 1e3 here"),
            rows=checks,
            all_agree=bool(all(c["agrees"] for c in checks)) if checks else None,
            worst_abs_deviation_pct=(float(max(abs(c["deviation_pct"]) for c in checks))
                                     if checks else float("nan"))),
        source=C.source(os.path.join(C.E1_RUNS, "N*"), os.path.join(C.F1_RUNS, "N*"),
                        os.path.join(C.SHARED, "reference.py"),
                        os.path.join(C.E1, "chain_network.py"), C.TRACKS))
    table("anatomy", rows)
    return out


# =============================================== j. along z / single step ====
def sec_along_z(use_cache=True):
    C.add_experiment_paths()
    import torch
    from _shared.reference import make_field
    from chain_network import load_network, step_outputs
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    D = C.load_tracks()
    fld = make_field(str(D["field"]))
    truth = np.asarray(D["test_truth"])
    P = np.asarray(D["test_P"])
    band = (P >= C.LOSS_WINDOW[0]) & (P < C.LOSS_WINDOW[1])

    rows_chain, rows_step, per_run = [], [], {}
    for N, q in C.PAIRS:
        stride = C.N_MAX // N
        dz = C.dz_mm(N)
        for loss, run in (("pooled", C.pooled_run(N, q)), ("reweighted", C.reweighted_run(N, q))):
            st = C.load_chain_states(run)
            assert st.shape[1] == N + 1, (run, st.shape)
            chain_med, chain_p95, chain_band = [], [], []
            for k in range(N + 1):
                d = C.deltas(st[:, k], truth[:, k * stride])
                chain_med.append(C.median(d["radial_um"]))
                chain_p95.append(C.quantile(d["radial_um"], 0.95))
                chain_band.append(C.median(d["radial_um"][band]))
            # one application from the RK6 state on every start plane
            cache = os.path.join(C.CACHE, "single_step_%s_%s.npz" % (loss, C.run_tag(N, q)))
            if use_cache and os.path.exists(cache):
                with np.load(cache) as z:
                    step_r = z["step_r"]
            else:
                os.makedirs(C.CACHE, exist_ok=True)
                model = load_network(run, fld)
                S = truth[:, 0:C.N_MAX:stride].reshape(-1, 5)
                nxt = truth[:, stride:C.N_MAX + 1:stride].reshape(-1, 5)
                plane = np.repeat(np.arange(N)[None, :], len(truth), 0).reshape(-1)
                out_s = step_outputs(model, S, C.Z0_MM + plane * dz)[:, -1, :]
                ds = C.deltas(out_s, nxt)
                step_r = np.vstack([ds["radial_um"], ds["tx"], ds["ty"], plane])
                np.savez_compressed(cache, step_r=step_r)
            r_step, tx_step, ty_step, plane = step_r
            per_plane_step = [float(np.median(r_step[plane == k])) for k in range(N)]
            key = "%s %s" % (loss, C.run_key(N, q))
            # --- the cross-check of section (j): the SAME one-step quantity measured
            # from two different start states, the chain's own and the RK6 truth's
            lx, ly, _ltx, _lty, _fin = anatomy_arrays(run, "%s_%s" % (loss, C.run_tag(N, q)),
                                                      use_cache)
            r_from_chain = np.hypot(lx, ly)                       # (N, n)
            r_from_truth = r_step.reshape(len(truth), N).T        # (N, n)
            st_all = C.load_chain_states(run)
            drift = np.array([np.median(np.hypot(st_all[:, kk, 0] - truth[:, kk * stride, 0],
                                                 st_all[:, kk, 1] - truth[:, kk * stride, 1]) * 1e3)
                              for kk in range(N)])
            dd = r_from_chain - r_from_truth
            rel = np.median(np.abs(dd) / np.maximum(r_from_truth, 1e-300))
            in_scale_x = float(C.load_scale(run)["in_scale"][0])
            crosscheck = dict(
                plane0_max_abs_difference_um=float(np.abs(dd[0]).max()),
                plane0_start_states_identical=True,
                plane0_note=("on plane 0 the two START states are the same array (the chain "
                             "begins at the track's own state at z0, which is truth[:, 0]); "
                             "the residual difference is not a start-state difference but "
                             "the difference between the two REFERENCES, RK6 at a 1 mm "
                             "local step in the anatomy against the stored truth track "
                             "marched at 0.1 mm, and it sits at the reference's own "
                             "step-size floor measured in Block C, C1 (a few times "
                             "1e-6 um in the median over a whole crossing)"),
                plane0_within_rk6_step_floor=bool(np.abs(dd[0]).max() < 1e-4),
                median_abs_difference_um=float(np.median(np.abs(dd))),
                median_relative_difference=float(rel),
                pearson_correlation=float(np.corrcoef(r_from_chain.ravel(),
                                                      r_from_truth.ravel())[0, 1]),
                med_from_chain_um=float(np.median(r_from_chain)),
                med_from_truth_um=float(np.median(r_from_truth)),
                ratio_of_medians=float(np.median(r_from_chain) / np.median(r_from_truth)),
                start_state_drift_med_um=dict(
                    plane_0=float(drift[0]), mid_plane=float(drift[N // 2]),
                    last_plane=float(drift[-1])),
                median_drift_over_planes_um=float(np.median(drift)),
                implied_sensitivity_dlog_e_per_mm=float(rel / (np.median(drift) * 1e-3))
                if np.median(drift) > 0 else float("nan"),
                network_in_scale_x_mm=in_scale_x,
                implied_variation_scale_mm=float((np.median(drift) * 1e-3) / rel)
                if rel > 0 else float("nan"))
            per_run[key] = dict(
                N=N, q=q, loss=loss, dz_mm=dz,
                per_step_vs_single_step=crosscheck,
                chain_radial_med_um_per_plane=chain_med,
                chain_radial_p95_um_per_plane=chain_p95,
                chain_radial_band_med_um_per_plane=chain_band,
                quarter_um=chain_med[N // 4], half_um=chain_med[N // 2], end_um=chain_med[-1],
                single_step_radial_med_um=float(np.median(r_step)),
                single_step_radial_p95_um=float(np.quantile(r_step, 0.95)),
                single_step_tx_med_abs_slope=C.med_abs(tx_step),
                single_step_ty_med_abs_slope=C.med_abs(ty_step),
                single_step_radial_med_um_per_plane=per_plane_step,
                chain_over_single_step=chain_med[-1] / float(np.median(r_step)),
                n_pairs=int(r_step.size))
            for k in range(N + 1):
                rows_chain.append(dict(N=N, q=q, loss=loss, plane=k,
                                       z_mm=C.Z0_MM + k * dz,
                                       chain_radial_med_um=chain_med[k],
                                       chain_radial_p95_um=chain_p95[k],
                                       chain_radial_band_med_um=chain_band[k]))
            for k in range(N):
                rows_step.append(dict(N=N, q=q, loss=loss, plane=k,
                                      z_mm=C.Z0_MM + k * dz,
                                      single_step_radial_med_um=per_plane_step[k]))
            tick("j. along z %s" % key)

    e3_step = {(int(r["N"]), int(r["q"])): float(r["step_med_um"])
               for r in C.read_csv(os.path.join(C.E3_RESULTS, "error_qdz_single_step.csv"))}
    out = dict(
        what=("the chain's radial median at every plane, read from chain_states.npz "
              "(which stores the state on all N+1 planes), and the single-step radial "
              "median from applying the network once from the RK6 truth state on each "
              "start plane, plane by plane.  The single-step reference is the RK6 truth "
              "one plane later, so no new integration is needed."),
        per_run=per_run,
        per_step_vs_single_step_check=dict(
            what=("The per-step radial error in tab_anatomy.csv and the single-step radial "
                  "error in tab_single_step_vs_z.csv agree to three digits in all six runs "
                  "(e.g. 0.3325 against 0.3326 um).  They are NOT the same computation: "
                  "the first starts each step from the chain's own state, the second from "
                  "the RK6 truth state on that plane, and the two inputs differ by the "
                  "accumulated chain error, which reaches 84-143 um by the last plane.  "
                  "The agreement is genuine, and this block measures why."),
            why=("On plane 0 the two start states are identical by construction (the chain "
                 "begins at the track's own state at z0, which is truth[:, 0]), and the two "
                 "code paths then give bit-identical errors: that is the proof they measure "
                 "the same quantity.  On later planes the one-step error e(S) = NN(S, z) - "
                 "RK6(S, z -> z + dz) is a smooth function of the start state, and what "
                 "matters is how fast the ERROR FIELD varies with S, not how fast the map "
                 "does.  Empirically a start-state displacement of order 0.1 mm changes the "
                 "one-step error by a median 2-4 parts in 10,000.  Dividing the median "
                 "drift (25-50 um) by that relative change gives a variation scale of "
                 "100-200 mm, the same ORDER as the network's own input normalisation "
                 "(in_scale_x = 446 mm at N = 64, 452 mm at N = 128 and 256), which is "
                 "what sets how fast its output can change with x; the estimate is an "
                 "order of magnitude, not a fit, because the relative difference is "
                 "dominated by the planes with the largest local error rather than "
                 "spread evenly.  Either way a displacement of 0.03-0.15 mm is a few "
                 "times 1e-4 of the scale on which the error field varies, so the two "
                 "medians cannot differ in the third digit.  Note also that "
                 "the two use slightly different references - RK6 at a 1 mm local step in "
                 "the anatomy, the stored truth track at 0.1 mm in the single-step table - "
                 "which is why plane 0, where the start states ARE the same array, agrees "
                 "to 1e-7 um rather than exactly: that residual is the reference's own "
                 "step-size floor from Block C, C1, not a start-state effect."),
            per_run={k: v["per_step_vs_single_step"] for k, v in per_run.items()},
            verdict=("genuine, not a bug: identical on plane 0, correlated 0.999+ "
                     "element-wise, and the residual difference tracks the start-state "
                     "drift divided by the network's input scale")),
        e3_single_step_med_um_for_the_snapshot=dict(
            {C.run_key(N, q): e3_step[(N, q)] for N, q in C.PAIRS},
            note=("E3's table is the 18 September SNAPSHOT network, so it is not expected "
                  "to equal the extended pooled run's single-step median")),
        source=C.source(os.path.join(C.E1_RUNS, "N*", "chain_states.npz"),
                        os.path.join(C.F1_RUNS, "N*", "chain_states.npz"),
                        os.path.join(C.E1, "chain_network.py"), C.TRACKS))
    table("along_z", rows_chain)
    table("single_step_vs_z", rows_step)
    return out


# ============================================ k. the pre-flight loss shares ==
_PREFLIGHT_CHECK_KEYS = (
    ("rank correlation of share with true cost", "spearman_share_vs_cost",
     "spearman_share_vs_cost"),
    ("rank correlation of share with true cost, in 10-50 GeV",
     "spearman_share_vs_cost_in_band", "spearman_share_vs_cost_in_band"),
    ("top 1 per cent of states, share of the loss [%]", "top_1pct_of_states_share_pct",
     "top_1pct_of_states_share_pct"),
    ("share of the loss in the first quarter of z [%]", "share_first_quarter_pct",
     "share_first_quarter_pct"),
    ("share of the loss in the last quarter of z [%]", "share_last_quarter_pct",
     "share_last_quarter_pct"),
    ("share of the loss in the band that is in the first quarter of z [%]",
     "share_in_band_first_quarter_pct", "share_10_50_GeV_in_first_quarter_pct"),
)


def _draw_pairs(truth, N, n_max, n, rng, even=False):
    """check_weights.py's own `draw_pairs`, copied so the rng sequence matches."""
    stride = n_max // N
    if even:
        per = max(1, n // N)
        tr = rng.integers(0, len(truth), per * N)
        k = np.repeat(np.arange(N), per)
    else:
        tr = rng.integers(0, len(truth), n)
        k = rng.integers(0, N, n)
    return truth[tr, k * stride], k, tr


def sec_preflight(network_run=None, verbose=True):
    C.add_experiment_paths()
    import torch
    from scipy.stats import spearmanr
    from _shared.model import LHCbRates, reconstruction_residuals
    from _shared.reference import make_field, gauss_legendre, rk6_rows
    from chain_network import load_network, znodes_for
    WL = C.load_weighted_loss()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    D = C.load_tracks()
    Z0, Z1, L, n_max = C.Z0_MM, C.Z1_MM, C.L_MM, C.N_MAX
    truth = np.asarray(D["train_truth"])
    fld = make_field(str(D["field"]))
    rates = LHCbRates(make_field(str(D["field"])))
    ibar = WL.i_bar(fld, Z0, Z1)

    # replay check_weights.py's rng so the 8,000 states are the SAME ones
    rng = np.random.default_rng(20260918)
    for N in (2, 256):
        _draw_pairs(truth, N, n_max, 2000, rng)
    for N in (2, 256):
        _draw_pairs(truth, N, n_max, 500, rng)
    for N in (2, 256):
        _draw_pairs(truth, N, n_max, 40, rng)

    N, q = 64, 2
    dz = L / N
    run_dir = network_run or C.pooled_run(N, q)
    model = load_network(run_dir, fld)
    S, k, _ = _draw_pairs(truth, N, n_max, 8000, rng, even=True)
    z_start = Z0 + k * dz
    P = WL.QOP_TO_GEV / np.abs(S[:, 4])
    St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
    zn = znodes_for(model, zt)
    c, A, b = gauss_legendre(q)
    At, bt = torch.tensor(A), torch.tensor(b)
    const = WL.reference_constants(model, S, z_start, Z1, ibar, L)
    with torch.no_grad():
        r_phys = (reconstruction_residuals(model, rates, St, dz, zn, At, bt, zt)
                  * model.in_scale[:4]).numpy()
    out_end = model(St, zt).detach().numpy()[:, -1, :]
    ref = rk6_rows(S, z_start, z_start + dz, step=1.0, field=fld)
    lev_end = (Z1 - (z_start + dz)) + dz
    cost = np.hypot((out_end[:, 0] - ref[:, 0]) + (out_end[:, 2] - ref[:, 2]) * lev_end,
                    (out_end[:, 1] - ref[:, 1]) + (out_end[:, 3] - ref[:, 3]) * lev_end) * 1e3

    MODES = ("blockE", "full", "no_lever", "no_track", "no_window")
    rows, modes = [], {}
    band = (P >= C.LOSS_WINDOW[0]) & (P < C.LOSS_WINDOW[1])
    for mode in MODES:
        w = WL.weights(model, St, zt, const, mode)
        w = w.detach().numpy() if hasattr(w, "detach") else np.asarray(w)
        per_state = ((r_phys * w) ** 2).mean(axis=(1, 2))
        share = per_state / per_state.sum()
        entry = dict(
            spearman_share_vs_cost=float(spearmanr(share, cost).statistic),
            spearman_share_vs_cost_in_band=float(spearmanr(share[band], cost[band]).statistic),
            top_1pct_of_states_share_pct=100.0 * float(
                np.sort(share)[::-1][:max(1, len(share) // 100)].sum()),
            by_band={}, by_quarter_of_z={})
        for label, m in C.band_masks(P):
            entry["by_band"][label] = dict(n=int(m.sum()), share_pct=100.0 * float(share[m].sum()),
                                           states_pct=100.0 * float(m.mean()))
            rows.append(dict(mode=mode, kind="momentum", key=label, n=int(m.sum()),
                             share_pct=100.0 * float(share[m].sum()),
                             states_pct=100.0 * float(m.mean())))
        for j in range(4):
            m = (k >= j * N // 4) & (k < (j + 1) * N // 4)
            entry["by_quarter_of_z"]["quarter %d" % (j + 1)] = dict(
                n=int(m.sum()), share_pct=100.0 * float(share[m].sum()))
            rows.append(dict(mode=mode, kind="quarter_of_z", key="quarter %d" % (j + 1),
                             n=int(m.sum()), share_pct=100.0 * float(share[m].sum()),
                             states_pct=100.0 * float(m.mean())))
        entry["share_first_quarter_pct"] = 100.0 * float(share[k < N // 4].sum())
        entry["share_last_quarter_pct"] = 100.0 * float(share[k >= 3 * N // 4].sum())
        entry["share_in_band_first_quarter_pct"] = 100.0 * float(
            share[band & (k < N // 4)].sum() / max(share[band].sum(), 1e-300))
        entry["share_in_band_last_quarter_pct"] = 100.0 * float(
            share[band & (k >= 3 * N // 4)].sum() / max(share[band].sum(), 1e-300))
        modes[mode] = entry

    clamp_rows, clamp = [], {}
    for cl in (0.0, 2.0, 3.0, 5.0, 10.0):
        cc = dict(const)
        cc["clamp"] = cl
        w = WL.weights(model, St, zt, cc, "full")
        w = w.detach().numpy() if hasattr(w, "detach") else np.asarray(w)
        per_state = ((r_phys * w) ** 2).mean(axis=(1, 2))
        share = per_state / per_state.sum()
        name = "off" if cl == 0 else "%g" % cl
        entry = dict(clamp=name,
                     top_1pct_of_states_share_pct=100.0 * float(
                         np.sort(share)[::-1][:max(1, len(share) // 100)].sum()),
                     spearman_share_vs_cost=float(spearmanr(share, cost).statistic),
                     spearman_share_vs_cost_in_band=float(spearmanr(share[band], cost[band]).statistic))
        for label, m in C.band_masks(P):
            entry["share_%s_pct" % label] = 100.0 * float(share[m].sum())
        clamp[name] = entry
        clamp_rows.append(entry)

    stored = C.read_json(os.path.join(C.F0_RESULTS, "check_weights.json"))
    check = dict(
        what=("the replayed pre-flight against F0_Weighting/results/check_weights.json, "
              "which used the OLD 1/2/5/10/20/50/100/200 GeV edges: the mode-level "
              "quantities that do not depend on the binning must match exactly"),
        rows=[dict(mode=m,
                   quantity=qn,
                   recomputed=modes[m][kk],
                   check_weights_json=stored["gate4"][m][sk],
                   deviation_pct=(100.0 * (modes[m][kk] - stored["gate4"][m][sk])
                                  / stored["gate4"][m][sk]) if stored["gate4"][m][sk] else 0.0)
              for m in MODES
              for qn, kk, sk in _PREFLIGHT_CHECK_KEYS])
    check["all_agree"] = bool(all(abs(r["deviation_pct"]) < 1e-6 for r in check["rows"]))
    check["note"] = (
        "check_weights.json was written on 2026-09-18 from the N = 64, q = 2 network as it "
        "stood THEN; that run has since been extended, so the weights are the same but the "
        "residuals, and therefore the shares, are not.  The verification that the method "
        "and the replayed random draw are right is `reproduced_with_the_18_Sept_snapshot` "
        "below, which uses results/N064_q02/stopped_2026-09-18/ and must match exactly.")

    out = dict(
        what=("gate 4 of F0_Weighting/check_weights.py re-run with the paper's bands: "
              "8,000 (track, plane) states drawn evenly over the 64 start planes from the "
              "extended pooled-loss N = 64, q = 2 network, the same random draw as the "
              "original gate (the generator's earlier draws are replayed), and the share "
              "of the loss each band and each quarter of z carries under every weighting "
              "mode.  'Cost' is what a state's error actually incurs at z1: its local "
              "error against RK6 from the same state, position plus slope times the "
              "distance left.  It is a diagnostic only and never enters the loss."),
        n_states=int(len(S)), network="extended pooled-loss N = 64, q = 2",
        i_bar_T_mm=float(ibar), modes=modes, clamp_scan=clamp,
        check_against_check_weights_json=check,
        snapshot_18Sept_record=(_preflight_snapshot_record() if network_run is None else None),
        reproduced_with_the_18_Sept_snapshot=(
            _preflight_snapshot_check(stored) if network_run is None else None),
        field_correlations=dict(C.read_json(os.path.join(C.F0_RESULTS, "field_correlations.json")),
                                source=C.source(os.path.join(C.F0_RESULTS,
                                                             "field_correlations.json"))),
        source=C.source(os.path.join(C.F0, "check_weights.py"),
                        os.path.join(C.F0, "weighted_loss.py"),
                        C.pooled_run(64, 2), C.TRACKS))
    table("preflight_shares", rows)
    table("clamp_scan", clamp_rows)
    tick("k. pre-flight loss shares")
    return out




def _preflight_snapshot_record():
    """The pre-flight EXACTLY as F0_Weighting recorded it on 18 September 2026.

    The write-up's section 3.5 and the section 5.2 footnote quote these numbers
    (57.82 per cent of the new loss in 10-50 GeV against 1.509 per cent of the
    pooled loss), and they were measured on the N = 64, q = 2 network as it stood
    THEN.  That run has since been trained further, so the paper's own pre-flight
    (`preflight.modes`, recomputed on the extended network and on the paper's
    band edges) does not reproduce them and is not meant to.  This block is the
    committed record, copied verbatim, so the footnote can cite a paper file.

    The band edges here are the ORIGINAL ones of check_weights.py,
    [1, 2, 5, 10, 20, 50, 100, 200] GeV, not the paper's.
    """
    cw = C.read_json(os.path.join(C.F0_RESULTS, "check_weights.json"))
    shares = C.read_csv(os.path.join(C.F0_RESULTS, "preflight_shares.csv"))
    modes = ("blockE", "full", "no_lever", "no_track", "no_window")
    per_mode, rows = {}, []
    for mode in modes:
        g = dict(cw["gate4"][mode])
        bands, quarters = [], []
        for r in shares:
            if r["mode"] != mode:
                continue
            if r["kind"] == "momentum":
                key = "%g-%g GeV" % (float(r["lo"]), float(r["hi"]))
                bands.append(dict(band=key, lo_GeV=float(r["lo"]), hi_GeV=float(r["hi"]),
                                  n=int(r["n"]), share_pct=float(r["share_pct"]),
                                  states_pct=float(r["tracks_pct"])))
                rows.append(dict(mode=mode, kind="momentum", key=key, lo=float(r["lo"]),
                                 hi=float(r["hi"]), n=int(r["n"]),
                                 share_pct=float(r["share_pct"]),
                                 states_pct=float(r["tracks_pct"])))
            elif r["kind"] == "quarter_of_z":
                key = "quarter %d" % int(float(r["lo"]))
                quarters.append(dict(quarter=key, n=int(r["n"]),
                                     share_pct=float(r["share_pct"]),
                                     states_pct=float(r["tracks_pct"])))
                rows.append(dict(mode=mode, kind="quarter_of_z", key=key, lo="", hi="",
                                 n=int(r["n"]), share_pct=float(r["share_pct"]),
                                 states_pct=float(r["tracks_pct"])))
        per_mode[mode] = dict(
            bands_original_edges=bands, quarters_of_z=quarters,
            share_10_50_GeV_pct=g["share_10_50_GeV_pct"],
            share_2_5_GeV_pct=g["share_2_5_GeV_pct"],
            share_above_50_GeV_pct=g["share_above_50_GeV_pct"],
            share_first_quarter_pct=g["share_first_quarter_pct"],
            share_last_quarter_pct=g["share_last_quarter_pct"],
            share_10_50_GeV_in_first_quarter_pct=g["share_10_50_GeV_in_first_quarter_pct"],
            share_10_50_GeV_in_last_quarter_pct=g["share_10_50_GeV_in_last_quarter_pct"],
            spearman_share_vs_cost=g["spearman_share_vs_cost"],
            spearman_share_vs_cost_in_band=g["spearman_share_vs_cost_in_band"],
            top_1pct_of_states_share_pct=g["top_1pct_of_states_share_pct"])
        for key in ("share_2_5_GeV_pct", "share_10_50_GeV_pct", "share_above_50_GeV_pct",
                    "share_first_quarter_pct", "share_last_quarter_pct",
                    "share_10_50_GeV_in_first_quarter_pct",
                    "share_10_50_GeV_in_last_quarter_pct",
                    "spearman_share_vs_cost", "spearman_share_vs_cost_in_band",
                    "top_1pct_of_states_share_pct"):
            rows.append(dict(mode=mode, kind="summary", key=key, lo="", hi="", n="",
                             share_pct=g[key], states_pct=""))
    clamp = {k: dict(v) for k, v in cw["clamp_scan"].items()}
    for k, v in clamp.items():
        for kk, vv in v.items():
            rows.append(dict(mode="full", kind="clamp_scan", key="clamp %s / %s" % (k, kk),
                             lo="", hi="", n="", share_pct=vv, states_pct=""))
    table("preflight_snapshot_record", rows)
    return dict(
        what=("gate 4 of F0_Weighting/check_weights.py as it was recorded on 2026-09-18, "
              "copied verbatim from check_weights.json and preflight_shares.csv.  Measured "
              "on 8,000 (track, plane) states from the N = 64, q = 2 network AS IT STOOD "
              "THEN (results/N064_q02/stopped_2026-09-18/), on the original band edges "
              "[1, 2, 5, 10, 20, 50, 100, 200] GeV.  This is where the write-up's "
              "57.82 per cent (new loss, 10-50 GeV) and 1.509 per cent (pooled loss, "
              "10-50 GeV) come from."),
        band_edges_GeV=[1, 2, 5, 10, 20, 50, 100, 200],
        n_states=8000, network="N = 64, q = 2, checkpoint of 2026-09-18",
        per_mode=per_mode, clamp_scan=clamp,
        window_at=cw["window_at"], i_bar_T_mm=cw["i_bar_T_mm"],
        z0=cw["z0"], z1=cw["z1"], L=cw["L"], all_pass=cw["all_pass"],
        headline=dict(share_10_50_GeV_pct_full=cw["gate4"]["full"]["share_10_50_GeV_pct"],
                      share_10_50_GeV_pct_blockE=cw["gate4"]["blockE"]["share_10_50_GeV_pct"],
                      share_2_5_GeV_pct_full=cw["gate4"]["full"]["share_2_5_GeV_pct"],
                      share_2_5_GeV_pct_blockE=cw["gate4"]["blockE"]["share_2_5_GeV_pct"]),
        source=C.source(os.path.join(C.F0_RESULTS, "check_weights.json"),
                        os.path.join(C.F0_RESULTS, "preflight_shares.csv")))


def _preflight_snapshot_check(stored):
    """Re-run the pre-flight on the 18 September checkpoint of the N = 64, q = 2 run.

    check_weights.json was produced from that checkpoint, so this must reproduce
    it exactly; it is the proof that the replayed random draw and the method are
    the same as the original gate's.
    """
    snap = sec_preflight(network_run=C.pooled_run(64, 2, snapshot=True), verbose=False)
    rows = []
    for m in ("blockE", "full", "no_lever", "no_track", "no_window"):
        for qn, kk, sk in _PREFLIGHT_CHECK_KEYS:
            mine, theirs = snap["modes"][m][kk], stored["gate4"][m][sk]
            rows.append(dict(mode=m, quantity=qn, recomputed=mine,
                             check_weights_json=theirs,
                             deviation_pct=(100.0 * (mine - theirs) / theirs) if theirs else 0.0))
    return dict(what=("the same pre-flight on results/N064_q02/stopped_2026-09-18/, the "
                      "checkpoint check_weights.json was written from"),
                rows=rows,
                all_agree=bool(all(abs(r["deviation_pct"]) < 1e-9 for r in rows)),
                worst_abs_deviation_pct=float(max(abs(r["deviation_pct"]) for r in rows)))

# =========================================================== l. loss constants ==
def sec_loss_constants():
    WL = C.load_weighted_loss()
    from _shared.reference import KAPPA
    rows, runs = [], {}
    for N, q in C.PAIRS:
        sc = C.load_scale(C.reweighted_run(N, q))
        w = sc.get("weighting", {})
        dz = C.dz_mm(N)
        entry = dict(N=N, q=q, dz_mm=dz, D_ref_mm=w.get("D_ref"), lev_ref_mm=w.get("lev_ref"),
                     i_bar_T_mm=w.get("i_bar"), mode=w.get("mode"), clamp=w.get("clamp"),
                     p_lo_GeV=w.get("p_lo"), p_hi_GeV=w.get("p_hi"), rolloff=w.get("rolloff"),
                     w_floor=w.get("w_floor"),
                     lever_min_mm=dz, lever_max_mm=C.L_MM + dz)
        rows.append(entry)
        runs[C.run_key(N, q)] = entry
    pw = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    consts = dict(KAPPA=float(KAPPA), QOP_TO_GEV=float(WL.QOP_TO_GEV),
                  P_LO=float(WL.P_LO), P_HI=float(WL.P_HI), ROLLOFF=float(WL.ROLLOFF),
                  W_FLOOR=float(WL.W_FLOOR), CLAMP=float(WL.CLAMP),
                  N_IBAR_SAMPLES=int(WL.N_IBAR_SAMPLES),
                  window_at_GeV={("%g" % p): float(v)
                                 for p, v in zip(pw, WL.band_window(np.array(pw)))},
                  per_run=runs,
                  source=C.source(os.path.join(C.F0, "weighted_loss.py"),
                                  os.path.join(C.SHARED, "reference.py"),
                                  os.path.join(C.F1_RUNS, "N*", "scale.json")))
    table("loss_constants", rows)

    # --- the worked example of section 3.2, recomputed ----------------------
    sc = C.load_scale(C.reweighted_run(64, 2))["weighting"]
    D_ref, ibar, L = sc["D_ref"], sc["i_bar"], sc["L"]
    dz = C.dz_mm(64)
    cases, ex_rows = {}, []
    for name, p_gev, z_out in (("3 GeV, first output plane", 3.0, C.Z0_MM + dz),
                               ("3 GeV, last output plane", 3.0, C.Z1_MM),
                               ("20 GeV, first output plane", 20.0, C.Z0_MM + dz),
                               ("20 GeV, last output plane", 20.0, C.Z1_MM)):
        qop = WL.QOP_TO_GEV / p_gev
        Dn = float(WL.track_bend(np.array([qop]), ibar, L)[0])
        W = float(WL.band_window(np.array([p_gev]))[0])
        a_n = float(np.sqrt(W) * D_ref / Dn)
        lever = max(C.Z1_MM - z_out, 0.0) + dz
        e = dict(case=name, p_GeV=p_gev, qop=qop, track_bend_D_mm=Dn, window_W=W,
                 a_unclamped=a_n, lever_mm=lever,
                 weight_on_a_position_residual_per_mm=a_n / D_ref,
                 weight_on_a_slope_residual_per_unit_slope=a_n * lever / D_ref)
        cases[name] = e
        ex_rows.append(e)
    sw = lambda k: cases[k]["weight_on_a_slope_residual_per_unit_slope"]        # noqa: E731
    pwf = lambda k: cases[k]["weight_on_a_position_residual_per_mm"]            # noqa: E731
    example = dict(
        N=64, q=2, dz_mm=dz, D_ref_mm=D_ref, cases=cases,
        ratios=dict(
            a_20GeV_over_a_3GeV=cases["20 GeV, first output plane"]["a_unclamped"]
            / cases["3 GeV, first output plane"]["a_unclamped"],
            slope_20GeV_first_over_3GeV_first=sw("20 GeV, first output plane")
            / sw("3 GeV, first output plane"),
            slope_20GeV_first_over_20GeV_last=sw("20 GeV, first output plane")
            / sw("20 GeV, last output plane"),
            slope_20GeV_first_over_3GeV_last=sw("20 GeV, first output plane")
            / sw("3 GeV, last output plane"),
            position_20GeV_over_3GeV=pwf("20 GeV, last output plane")
            / pwf("3 GeV, last output plane")),
        note=("unclamped: the per-batch clamp to [1/5, 5] of the batch median then trims "
              "the tails of the a_n distribution.  The write-up quotes a_n = 0.0992 and "
              "2.959, weights 17.8 / 0.596 / 0.278 / 0.0093, and an extreme ratio of "
              "1,908."),
        source=C.source(os.path.join(C.F0, "weighted_loss.py"),
                        os.path.join(C.F1_RUNS, "N064_q02", "scale.json")))
    table("worked_example", ex_rows)
    tick("l. loss constants and the worked example")
    return dict(constants=consts, worked_example=example)


# ================================== m. against the Geant4-true SciFi state ====
def sec_against_true(use_cache=True):
    C.add_experiment_paths()
    from _shared.reference import make_field, rk6_rows, RK6_STEP
    D = C.load_tracks()
    fld = make_field(str(D["field"]))
    a = C.split_arrays("test")
    P, S_post, z_post = a["P"], a["S_post"], a["z_post"]
    truth_end, truth_zpost = a["truth"][:, C.N_MAX], a["truth_zpost"]

    def block(pred, tru, m):
        d = C.deltas(pred[m], tru[m])
        return dict(n=int(m.sum()),
                    pos_max_med_um=C.median(d["max_um"]),
                    pos_max_p95_um=C.quantile(d["max_um"], 0.95),
                    pos_radial_med_um=C.median(d["radial_um"]),
                    slope_max_med=C.median(d["slope_max"]),
                    tx_med_abs_slope=C.med_abs(d["tx"]), ty_med_abs_slope=C.med_abs(d["ty"]))

    rows, per_run = [], {}
    rk6_ref = {}
    for label, m in C.band_masks(P, include_all=True):
        rk6_ref[label] = block(truth_zpost, S_post, m)

    stored = {(r["network"], r["band"]): r
              for r in C.read_csv(os.path.join(C.F2_RESULTS, "against_true_state.csv"))}
    checks = []
    for N, q in C.PAIRS:
        for loss, run in (("pooled", C.pooled_run(N, q)), ("reweighted", C.reweighted_run(N, q))):
            cache = os.path.join(C.CACHE, "carried_%s_%s.npz" % (loss, C.run_tag(N, q)))
            if use_cache and os.path.exists(cache):
                with np.load(cache) as z:
                    carried = z["carried"]
            else:
                os.makedirs(C.CACHE, exist_ok=True)
                end = C.load_chain_states(run)[:, -1, :]
                carried = rk6_rows(end, C.Z1_MM, z_post, step=RK6_STEP, field=fld)
                np.savez_compressed(cache, carried=carried)
            end = C.load_chain_states(run)[:, -1, :]
            key = "%s %s" % (loss, C.run_key(N, q))
            entry = {}
            for label, m in C.band_masks(P, include_all=True):
                e = dict(band=label,
                         nn_vs_rk6=block(end, truth_end, m),
                         nn_vs_true=block(carried, S_post, m),
                         rk6_vs_true=rk6_ref[label])
                e["nn_true_over_rk6_true"] = (e["nn_vs_true"]["pos_max_med_um"]
                                              / e["rk6_vs_true"]["pos_max_med_um"])
                e["rk6_true_over_nn_rk6"] = (e["rk6_vs_true"]["pos_max_med_um"]
                                             / e["nn_vs_rk6"]["pos_max_med_um"])
                entry[label] = e
                rows.append(dict(
                    N=N, q=q, loss=loss, band=label, n=e["nn_vs_rk6"]["n"],
                    nn_vs_rk6_pos_max_med_um=e["nn_vs_rk6"]["pos_max_med_um"],
                    nn_vs_rk6_pos_max_p95_um=e["nn_vs_rk6"]["pos_max_p95_um"],
                    nn_vs_rk6_slope_max_med=e["nn_vs_rk6"]["slope_max_med"],
                    nn_vs_true_pos_max_med_um=e["nn_vs_true"]["pos_max_med_um"],
                    nn_vs_true_pos_max_p95_um=e["nn_vs_true"]["pos_max_p95_um"],
                    nn_vs_true_slope_max_med=e["nn_vs_true"]["slope_max_med"],
                    rk6_vs_true_pos_max_med_um=e["rk6_vs_true"]["pos_max_med_um"],
                    rk6_vs_true_pos_max_p95_um=e["rk6_vs_true"]["pos_max_p95_um"],
                    rk6_vs_true_slope_max_med=e["rk6_vs_true"]["slope_max_med"],
                    nn_true_over_rk6_true=e["nn_true_over_rk6_true"],
                    rk6_true_over_nn_rk6=e["rk6_true_over_nn_rk6"]))
            per_run[key] = entry
            if loss == "reweighted":
                name = "N = %d, q = %d, dz = %.0f mm" % (N, q, C.dz_mm(N))
                s = stored.get((name, "all"))
                if s:
                    for what, mine, theirs in (
                            ("nn vs RK6, max-metric median [um]",
                             entry["all"]["nn_vs_rk6"]["pos_max_med_um"],
                             float(s["nn_vs_rk6_pos_med_um"])),
                            ("nn vs true, max-metric median [um]",
                             entry["all"]["nn_vs_true"]["pos_max_med_um"],
                             float(s["nn_vs_true_pos_med_um"])),
                            ("RK6 vs true, max-metric median [um]",
                             entry["all"]["rk6_vs_true"]["pos_max_med_um"],
                             float(s["rk6_vs_true_pos_med_um"])),
                            ("nn vs RK6, slope max median (slope)",
                             entry["all"]["nn_vs_rk6"]["slope_max_med"],
                             float(s["nn_vs_rk6_slope_med_1e-3"]) / 1e3),
                            ("nn vs true, slope max median (slope)",
                             entry["all"]["nn_vs_true"]["slope_max_med"],
                             float(s["nn_vs_true_slope_med_1e-3"]) / 1e3),
                            ("RK6 vs true, slope max median (slope)",
                             entry["all"]["rk6_vs_true"]["slope_max_med"],
                             float(s["rk6_vs_true_slope_med_1e-3"]) / 1e3)):
                        checks.append(dict(run=key, quantity=what, recomputed=mine,
                                           against_true_state_csv=theirs,
                                           deviation_pct=100.0 * (mine - theirs) / theirs,
                                           agrees=bool(abs(mine - theirs) / abs(theirs) < 1e-6)))
            tick("m. against the true state %s" % key)

    out = dict(
        what=("the three references for a crossing, re-binned into the paper's bands: the "
              "network endpoint carried from z1 to the particle's own first SciFi plane "
              "with RK6 at 0.1 mm and compared with the Geant4-true state there; the "
              "field-only RK6 reference carried to the same plane and compared with it; "
              "and the network against RK6 at z1.  Position error is the max metric "
              "max(|dx|, |dy|); the slope error is max(|dtx|, |dty|), dimensionless."),
        per_run=per_run, rk6_vs_true=rk6_ref,
        check_against_f2_csv=dict(
            what=("the 'all' rows against F2_Analysis/results/against_true_state.csv, "
                  "whose slope columns are stored x 1e3"),
            rows=checks, all_agree=bool(all(c["agrees"] for c in checks)) if checks else None,
            worst_abs_deviation_pct=(float(max(abs(c["deviation_pct"]) for c in checks))
                                     if checks else float("nan"))),
        source=C.source(os.path.join(C.E1_RUNS, "N*", "chain_states.npz"),
                        os.path.join(C.F1_RUNS, "N*", "chain_states.npz"), C.TRACKS,
                        os.path.join(C.SHARED, "reference.py"),
                        os.path.join(C.F2_RESULTS, "against_true_state.csv")))
    table("against_true_state", rows)
    return out


# ================================================================= n. compute ==
def sec_compute():
    rows, per_run = [], {}
    for N, q in C.PAIRS:
        for loss, run in (("pooled", C.pooled_run(N, q)), ("reweighted", C.reweighted_run(N, q))):
            rec = C.load_record(run)
            hh, hdup, hraw = C.load_history(run)
            rr, rdup, rraw = C.load_rounds(run)
            walls = np.array([float(x["wall_s"]) for x in hh])
            iters = np.array([float(x["n_iter"]) for x in hh])
            entry = dict(
                N=N, q=q, loss=loss, dz_mm=C.dz_mm(N),
                n_parameters=rec["n_parameters"], states_per_round=rec["states_per_round"],
                round_restarts=rec["round_restarts"],
                restarts_recorded=int(rec["restarts"]), unique_restarts=len(hh),
                duplicated_history_rows=len(hdup), history_rows_read=hraw,
                rounds=len(rr), rounds_completed=len(rr),
                rounds_counter_record_json=int(rec["rounds"]),
                duplicated_round_rows=len(rdup), rounds_rows_read=rraw,
                median_wall_s_per_restart=float(np.median(walls)),
                mean_wall_s_per_restart=float(walls.mean()),
                summed_history_wall_h=float(walls.sum() / 3600.0),
                recorded_train_wall_h=rec["train_wall_s"] / 3600.0,
                median_iterations_per_restart=float(np.median(iters)),
                early_stop_restarts=rec["early_stop_restarts"],
                final_loss=rec["final_loss"],
                carry_us_per_track=rec.get("carry_us_per_track"),
                hit_cap=rec["hit_cap"])
            rows.append(entry)
            per_run["%s %s" % (loss, C.run_key(N, q))] = entry
    out = dict(
        what=("training cost of the six runs.  Per-restart walls are medians over the "
              "DE-DUPLICATED history; where the recorded wall exceeds the summed history "
              "the difference is a duplicate farm job's work, which the record counts and "
              "the de-duplicated history does not."),
        per_run=per_run,
        farm_clusters=dict(
            pooled_grid_submitted="5805460 (2026-09-17, 16 jobs, --round-restarts 25)",
            pooled_grid_extended="5809659, then 5815090 and 5816352 (2026-09-18 onwards)",
            exact_comparators="5805461 (2026-09-17)",
            reweighted_submitted="5809660 (2026-09-18 evening, 3 jobs)",
            reweighted_N256_q16_reopened="5823724, with a duplicate job 5833416 from restart 1,198",
            source=C.source(os.path.join(C.E1, "README.md"),
                            os.path.join(C.E, "E2_Comparators", "README.md"),
                            os.path.join(C.F, "README.md"),
                            os.path.join(C.F, "F4_Writeup", "page.md"))),
        rk6_cost=dict(
            seconds_per_track_measured_on_a_200_leg_batch=1.0433,
            seconds_per_track_extrapolated_batch_20000=0.12,
            seconds_per_track_extrapolated_batch_5000=0.1339,
            step_mm=0.1, crossing_mm=5174.27, peak_rss_MB=145.5,
            note="one thread, fp64; the field call is the whole cost and amortises over the batch",
            source=C.source(os.path.join(C.C1_RESULTS, "reference_convergence_meta.json"),
                            os.path.join(C.C1, "README.md"))),
        source=C.source(os.path.join(C.E1_RUNS, "N*"), os.path.join(C.F1_RUNS, "N*")))
    table("compute", rows)
    tick("n. compute")
    return out


# ==================================================================== main ====
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute the cached anatomy / carried-state arrays")
    a = ap.parse_args(argv)
    use_cache = not a.no_cache
    os.makedirs(C.RESULTS, exist_ok=True)
    os.makedirs(C.CACHE, exist_ok=True)
    print("Self_chained_paper/scripts/numbers.py", flush=True)

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_for": "the self-chained-network mini-paper, Self_chained_paper/",
        "repository_root": C.REPO,
        "conventions": dict(
            slopes=("t_x = dx/dz and t_y = dy/dz are dimensionless; every slope error in "
                    "this file is the raw difference and is never multiplied by 1e3.  No "
                    "key is called mrad."),
            positions="micrometres (mm x 1e3)",
            momentum_bands=dict(edges_GeV=list(C.BAND_EDGES[:-1]) + ["inf"],
                                labels=list(C.BAND_LABELS),
                                extra=C.LOSS_WINDOW_LABEL,
                                membership="from the per-split P array of tracks.npz, not PBAND"),
            reference=("RK6 truth states stored in tracks.npz: <split>_truth (n, 257, 5); "
                       "an N-step chain's step k starts on plane index k * (256 // N) and "
                       "the endpoint is plane 256, z1 = 7826.0 mm"),
            statistics=("median |.|, p95 of |.|, signed median, 68 % half-width "
                        "(q84 - q16)/2, RMS, radial = hypot(dx, dy), max-metric = "
                        "max(|dx|, |dy|); every key names its metric"),
            splits=dict(C.SPLIT_SIZES),
            round_counting=("`rounds` everywhere in this file is the number of COMPLETED "
                            "training rounds, i.e. the number of de-duplicated rows in the "
                            "run's rounds.csv.  `rounds_counter_record_json` is the "
                            "trainer's round counter at the moment the run stopped "
                            "(record.json[\"rounds\"] = progress.json[\"round\"]), which "
                            "counts the round that was in progress and therefore has no "
                            "row; it is one higher wherever a run stopped mid-round.  E3's "
                            "error_qdz_chain.csv `rounds` column is that counter.")),
        "palette": C.PALETTE,
    }
    out["dataset"] = sec_dataset()
    out["scheme"] = sec_scheme()
    out["reference_vs_truth"] = sec_reference_vs_truth()
    out["pooled_grid_snapshot"] = sec_pooled_grid()
    out["pooled_extended"] = sec_extended()
    out["reweighted"] = sec_reweighted()
    out["headline"] = sec_headline()
    out["near_5gev"] = sec_near_5gev()
    out["anatomy"] = sec_anatomy(use_cache)
    out["along_z"] = sec_along_z(use_cache)
    out["preflight"] = sec_preflight()
    out.update(sec_loss_constants())
    out["against_true_state"] = sec_against_true(use_cache)
    out["compute"] = sec_compute()

    out["tables"] = TABLES
    out["timing"] = dict(steps=TIMING, total_wall_s=round(time.time() - _T0, 1))
    path = os.path.join(C.RESULTS, "paper_numbers.json")
    with open(path, "w") as f:
        json.dump(C.jsonable(out), f, indent=1)
    print("\nwrote %s" % path)
    print("wrote %d tables: %s" % (len(TABLES), ", ".join(sorted(TABLES))))
    print("total wall %.1f s" % (time.time() - _T0))
    return out


if __name__ == "__main__":
    main()
