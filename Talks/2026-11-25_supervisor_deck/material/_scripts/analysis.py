"""Items 6, 7, 8, 9: best run per type, momentum bands, three references, steps and stages.
Reads _work/pred (my test-split scoring, nothing written to the store), the stored reports, the
classical caches. Writes CSV and PNG into the deck-material folder."""
import os, glob, json, csv
import numpy as np, pandas as pd
from style import *
P = os.path.join(WORK, "pred")
TS = {k: np.load(os.path.join(WORK, f"test_{k}.npz")) for k in ("5558be31abeb", "afc61a558211", "12a8d35c3165")}
CUBIC_CACHE = "/data/bfys/gscriven/rkpinn_analysis_cache/what_are_the_classical_references_on_the_cubic_map"
LIN_CACHE = os.path.join(WORK, "classical_linear")
INV = pd.read_csv(os.path.join(WORK, "inventory.csv"))
MAPNAME = {"5558be31abeb": "cubic", "afc61a558211": "linear", "12a8d35c3165": "linear"}
_idx = {tuple(np.round(r, 9)): i for i, r in enumerate(TS["afc61a558211"]["start_state"])}
SUB12 = np.array([_idx.get(tuple(np.round(r, 9)), -1) for r in TS["12a8d35c3165"]["start_state"]])


def errs(end, T):
    ref = T["reference_states_on_planes"][:, -1]
    d = end[:, :4] - ref[:, :4]
    return dict(x=d[:, 0] * 1e3, y=d[:, 1] * 1e3, tx=d[:, 2], ty=d[:, 3], radial=np.hypot(d[:, 0], d[:, 1]) * 1e3)

def load_pred(key, tracks, s=0):
    f = os.path.join(P, f"{key}__{tracks}__s{s}.npz")
    return np.load(f) if os.path.exists(f) else None

def exact_end(tracks, N, q):
    if tracks == "5558be31abeb":
        if q == 2:
            z = np.load(f"{STORE}/tracks/5558be31abeb/exact_scheme/test/{N}_steps_2_stages.npz")
            return z[z.files[0]][:, -1] if z[z.files[0]].ndim == 3 else None
        f = os.path.join(CUBIC_CACHE, "exact_%03d_steps_%02d_stages.npz" % (N, q))
        return np.load(f)["final"] if os.path.exists(f) else None
    f = os.path.join(LIN_CACHE, "exact_%03d_%02d.npz" % (N, q))
    return np.load(f)["final"] if os.path.exists(f) else None

def exact_median(tracks, N, q):
    T = TS[tracks]
    if tracks == "5558be31abeb":
        f = os.path.join(CUBIC_CACHE, "exact_%03d_steps_%02d_stages.npz" % (N, q))
        if os.path.exists(f): return float(np.median(np.load(f)["endpoint"]))
    if tracks == "12a8d35c3165":
        e = exact_end("afc61a558211", N, q)
        if e is None or (SUB12 < 0).any(): return np.nan
        return float(np.median(errs(e[SUB12], T)["radial"]))
    e = exact_end(tracks, N, q)
    return float(np.median(errs(e, T)["radial"])) if e is not None else np.nan

# ---------------------------------------------------------------- item 6: candidates and types
TYPES = [  # type, plain name, candidate keys, tracks to score on, supervised?, sweeps
    ("direct_states", "direct states", ["29510f525130", "3a9b83527a38", "d362eee5da3e", "0036186e5958", "e090a6a7f865"], "5558be31abeb", "label-free", 0),
    ("straight_line_per_track", "straight line, per-track scale", ["400182ba3233", "570b4813d0f9", "204ebe10b50d", "9730fcd6c859", "471efb0f5454",
        "4a2ce9cd7133", "3ab59d0cd0ad", "e0d045abcb63", "3a96648cde9e", "713f38afa7a3"], "5558be31abeb", "label-free", 0),
    ("straight_line_global", "straight line, global scale", ["2382b5d314a5", "9cb1b261cf92", "68e0a47804fc", "fd66c823715a", "67bae3fce343"], "5558be31abeb", "label-free", 0),
    ("physics_layer", "physics layer", ["a8a4083032fd", "3a0f806cb6b2", "ad4a7be64235", "4504dcb05fb9", "cdd676e19320"], "5558be31abeb", "label-free", 0),
    ("supervised_path", "supervised path network", ["25934ed22604"], "12a8d35c3165", "supervised", 2),
    ("label_free_path", "label-free path network", ["3ccd44d61cd8", "0b41ecdd2505"], "5558be31abeb", "label-free", 2),
    ("whole_crossing", "whole-crossing network", ["dd9acf59ac88"], "afc61a558211", "supervised", 0),
    ("learned_gain", "learned gain on a frozen network", ["c6a7d5d15caa"], "afc61a558211", "label-free", 0),
    ("error_conditioned", "error-conditioned corrector", ["1eaf33293506"], "afc61a558211", "label-free", 0),
    ("direct_predicted_end", "direct states, predicted end state", ["4cc888dcc710"], "afc61a558211", "label-free", 0),
]
import yaml
def conf(k): return yaml.safe_load(open(f"{STORE}/runs/{k}/configuration.yaml"))
def state(k): return json.load(open(f"{STORE}/runs/{k}/state.json"))
def stored_test(k):
    r = INV[(INV.key == k) & (INV.split == "test")]
    return (float(r.iloc[-1]["median"]), r.iloc[-1]["report"]) if len(r) else (np.nan, "")

def score_of(k, tracks, s=0):
    """median, p95 on the given test split; from my scoring when present, else the stored report."""
    d = load_pred(k, tracks, s)
    if d is not None:
        e = errs(d["end"], TS[tracks])["radial"]
        return float(np.median(e)), float(np.percentile(e, 95)), "scored here (end states of the last round)", float(d["seconds_of_carrying"]) / len(e) * 1e3, int(d["round"])
    m, rep = stored_test(k)
    return m, np.nan, "stored report " + rep, np.nan, None

def _sw(k):
    r = sorted(glob.glob(f"{STORE}/runs/{k}/reports/*/summary.json"))
    return json.load(open(r[-1])).get("sweeps") or []
def sweep_ms(k, s):
    for row in _sw(k):
        if row["sweeps"] == s: return 1e3 * row["seconds_of_the_sweeps"] / row["against_the_reference"]["number_of_tracks"]
    return np.nan
def sweep_n(k):
    return _sw(k)[0]["against_the_reference"]["number_of_tracks"] if _sw(k) else "?"
rows, best = [], {}
all_cand = []
for typ, plain, keys, tracks, sup, sweeps in TYPES:
    cands = []
    for k in keys:
        st = state(k)
        if st["state"] == "training": continue
        m, p95, src, ms, rnd = score_of(k, tracks)
        cands.append((m, k, p95, src, ms, rnd)); all_cand.append(dict(type=typ, run_key=k, test_median_um=m, source=src))
    m, k, p95, src, ms, rnd = sorted(cands, key=lambda c: (np.isnan(c[0]), c[0]))[0]
    c = conf(k); st = state(k); N, q = c["steps"]["number_of_steps"], c["steps"]["number_of_stages"]
    ex = exact_median(tracks, N, q) if q else np.nan
    best[typ] = (k, tracks, plain, sweeps)
    for s in range(sweeps + 1):
        if s:
            d = load_pred(k, tracks, s); e = errs(d["end"], TS[tracks])["radial"]; m, p95 = float(np.median(e)), float(np.percentile(e, 95))
        rows.append(dict(type=typ + ("" if not s else "_plus_%d_sweep%s" % (s, "s" if s > 1 else "")),
            plain_name=plain + ("" if not s else " + %d sweep%s" % (s, "s" if s > 1 else "")),
            map=MAPNAME[tracks] + (" map" if tracks != "5558be31abeb" else ""), test_tracks=len(TS[tracks]["momentum_gev"]),
            track_set=tracks, steps=N, stages=q, optimiser=c["training"]["optimiser"]["name"], loss=c["loss"]["name"],
            supervision=sup, run_key=k, round=st["rounds_done"], state=st["state"] if st["state"] != "at its cap" else "finished at its cap (no plateau)",
            test_median_um=m, test_p95_um=p95, exact_scheme_median_um_same_layout=ex, ratio_to_exact=m / ex if ex == ex else np.nan,
            python_ms_per_track=ms if not s else ms + sweep_ms(k, s), source=src + ("" if not s else "; sweep time from the stored report's sweeps (%s tracks)" % sweep_n(k)),
            note={"afc61a558211": "trained on the smaller linear set 12a8d35c3165; scored here on the 13,447 linear test tracks of afc61a558211 (each 12a8 split is a subset with identical references)", "12a8d35c3165": "linear map, 1,452 test tracks of 12a8d35c3165 (the run cannot be rebuilt on afc61a558211: it needs stored exact states with stages there)"}.get(tracks, "")))
rows.append(dict(type="analytic_gain", plain_name="analytic gain (one Newton step from the stage residual)", map="linear map", test_tracks=1452, track_set="12a8d35c3165",
                 steps=64, stages=2, optimiser="none (no training)", loss="none", supervision="none", run_key="(evaluation, no run)", state="finished",
                 test_median_um=0.147, exact_scheme_median_um_same_layout=0.147, ratio_to_exact=1.0,
                 source="experiments/does_a_kalman_like_gain_remove_the_bias/README.md (Part B)", note="equals the exact scheme by construction; quoted, not re-scored"))
B = pd.DataFrame(rows); B.to_csv(os.path.join(OUT, "best_per_type.csv"), index=False)
pd.DataFrame(all_cand).to_csv(os.path.join(WORK, "candidates.csv"), index=False)
print(B[["plain_name", "map", "run_key", "state", "test_median_um", "test_p95_um", "exact_scheme_median_um_same_layout", "ratio_to_exact", "python_ms_per_track"]].to_string())

# all_test_scores.csv: the stored table + live state + last validation + my scores
ts = pd.read_csv("/data/bfys/gscriven/worktrees/cubic_map_standard/experiments/do_the_headline_results_hold_on_the_cubic_map/analysis/results/test_scores.csv")
live = []
for _, r in ts.iterrows():
    st = state(r.run_key); rr = list(csv.DictReader(open(f"{STORE}/runs/{r.run_key}/rounds.csv")))
    mine = load_pred(r.run_key, "5558be31abeb")
    e = errs(mine["end"], TS["5558be31abeb"])["radial"] if mine is not None else None
    live.append(dict(state_now=st["state"], rounds_done_now=st["rounds_done"],
                     last_validation_larger_of_x_and_y_median_um=float(rr[-1]["validation_endpoint_error_larger_of_x_and_y_median_micrometres"]),
                     test_median_um_scored_here=float(np.median(e)) if e is not None else np.nan,
                     test_p95_um_scored_here=float(np.percentile(e, 95)) if e is not None else np.nan,
                     round_scored_here=int(mine["round"]) if mine is not None else np.nan))
A = pd.concat([ts, pd.DataFrame(live)], axis=1); A.to_csv(os.path.join(OUT, "all_test_scores.csv"), index=False)

# ---------------------------------------------------------------- item 7: momentum bands
MODELS = []   # (label, key, tracks, sweeps)
for typ, plain, keys, tracks, sup, sweeps in TYPES:
    k = best[typ][0]
    MODELS.append((plain, k, tracks, 0))
    if sweeps: MODELS.append((plain + " + 2 sweeps", k, tracks, 2))
rows = []
for lab, k, tracks, s in MODELS:
    d = load_pred(k, tracks, s)
    if d is None: continue
    T = TS[tracks]; e = errs(d["end"], T); p = T["momentum_gev"]
    for b, m in [("all momenta", np.ones_like(p, bool))] + band_masks(p):
        row = dict(model=lab, map=MAPNAME[tracks], run_key=k, sweeps=s, band=b, number_of_tracks=int(m.sum()),
                   radial_median_um=np.median(e["radial"][m]), radial_p95_um=np.percentile(e["radial"][m], 95))
        for c, u in (("x", "um"), ("y", "um"), ("tx", ""), ("ty", "")):
            row[f"{c}_median_abs{('_' + u) if u else ''}"] = np.median(np.abs(e[c][m]))
            row[f"{c}_signed_median{('_' + u) if u else ''}"] = np.median(e[c][m])
        rows.append(row)
BM = pd.DataFrame(rows)
# classical references on the cubic map, same columns where available
cub = TS["5558be31abeb"]; pc = cub["momentum_gev"]
refrows = []
for lab, f, endkey in [("RK4, 16 steps", "rk4_016_steps.npz", None), ("RK4, 32 steps", "rk4_032_steps.npz", None), ("RK4, 64 steps", "rk4_064_steps.npz", None),
                       ("exact scheme, 64 steps x 2 stages", "exact_064_steps_02_stages.npz", "final")]:
    z = np.load(os.path.join(CUBIC_CACHE, f)); er = z["endpoint"]
    for b, m in [("all momenta", np.ones_like(pc, bool))] + band_masks(pc):
        row = dict(model=lab, map="cubic", run_key="classical", sweeps=0, band=b, number_of_tracks=int(m.sum()), radial_median_um=np.median(er[m]), radial_p95_um=np.percentile(er[m], 95))
        if endkey:
            e = errs(z["final"], cub)
            for c, u in (("x", "um"), ("y", "um"), ("tx", ""), ("ty", "")):
                row[f"{c}_median_abs{('_' + u) if u else ''}"] = np.median(np.abs(e[c][m])); row[f"{c}_signed_median{('_' + u) if u else ''}"] = np.median(e[c][m])
        refrows.append(row)
BM = pd.concat([BM, pd.DataFrame(refrows)]); BM.to_csv(os.path.join(OUT, "by_momentum.csv"), index=False)

x = np.arange(5)
fig, axs = plt.subplots(1, 2, figsize=(14, 6.8), sharey=True, layout="constrained")
linex = exact_end("afc61a558211", 64, 2)
if linex is not None:
    TL = TS["afc61a558211"]; el = errs(linex, TL)["radial"]
    for b, m in [("all momenta", np.ones(len(el), bool))] + band_masks(TL["momentum_gev"]):
        BM = pd.concat([BM, pd.DataFrame([dict(model="exact scheme, 64 steps x 2 stages (linear map)", map="linear", run_key="classical", sweeps=0, band=b,
                         number_of_tracks=int(m.sum()), radial_median_um=np.median(el[m]), radial_p95_um=np.percentile(el[m], 95))])])
    BM.to_csv(os.path.join(OUT, "by_momentum.csv"), index=False)
for ax, cubic in ((axs[0], True), (axs[1], False)):
    ms = [m for m in MODELS if (m[2] == "5558be31abeb") == cubic and load_pred(m[1], m[2], m[3]) is not None]
    for (lab, k, tracks, s), c in zip(ms, COLOURS):
        t = BM[(BM.model == lab)].set_index("band").loc[list(BANDS)]
        ax.plot(x, t.radial_median_um, marker="o", lw=1.8, color=c, label=lab + ("" if cubic else (" (1,452 tracks)" if tracks == "12a8d35c3165" else "")))
    refs = (("RK4, 16 steps", 1.0, "-"), ("RK4, 32 steps", 1.4, "-"), ("RK4, 64 steps", 1.8, "-"), ("exact scheme, 64 steps x 2 stages", 2.4, ":")) if cubic else \
           (("exact scheme, 64 steps x 2 stages (linear map)", 2.4, ":"),)
    for lab, lw, ls in refs:
        t = BM[BM.model == lab]
        if not len(t): continue
        t = t.set_index("band").loc[list(BANDS)]
        ax.plot(x, t.radial_median_um, color=GREY, lw=lw, ls=ls, alpha=0.85)
        ax.text(4.1, t.radial_median_um.iloc[-1] * (0.6 if "exact" in lab and cubic else 1), lab.replace(" (linear map)", ""), color=GREY, fontsize=8.5, va="center")
    ax.set_yscale("log"); ax.set_xticks(x, BAND_LABELS); ax.set_xlim(-0.2, 5.6)
    ax.set_xlabel("momentum band (GeV)"); ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.text(0.02, 0.02, "cubic map, 13,447 test tracks" if cubic else "linear map, 13,447 test tracks unless marked", transform=ax.transAxes, fontsize=10, color=GREY)
axs[0].set_ylabel("endpoint radial median (µm)")
save(fig, "by_momentum")

# resolution per model: |error| of x, y, tx, ty against p, binned median and 16-84 band
pb = np.logspace(0, np.log10(200), 16); pm = np.sqrt(pb[1:] * pb[:-1])
resrows = []
for lab, k, tracks, s in MODELS:
    d = load_pred(k, tracks, s)
    if d is None: continue
    T = TS[tracks]; e = errs(d["end"], T); p = T["momentum_gev"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for a, (c, lab2) in zip(axs.ravel(), (("x", "|Δx| (µm)"), ("y", "|Δy| (µm)"), ("tx", "|Δtx|"), ("ty", "|Δty|"))):
        v = np.abs(e[c]); med, lo, hi = [], [], []
        for i in range(len(pm)):
            m = (p >= pb[i]) & (p < pb[i + 1])
            if m.sum() < 10: med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            med.append(np.median(v[m])); lo.append(np.percentile(v[m], 16)); hi.append(np.percentile(v[m], 84))
            resrows.append(dict(model=lab, run_key=k, map=MAPNAME[tracks], component=c, p_low_gev=pb[i], p_high_gev=pb[i + 1], tracks=int(m.sum()), median_abs=med[-1], p16_abs=lo[-1], p84_abs=hi[-1]))
        a.fill_between(pm, lo, hi, color=COLOURS[0], alpha=0.2, lw=0); a.plot(pm, med, color=COLOURS[0], lw=2, marker="o", ms=3)
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylabel(lab2)
    for a in axs[1]: a.set_xlabel("p (GeV)")
    fig.tight_layout(); save(fig, "resolution_" + lab.replace(",", "").replace(" + ", "_plus_").replace(" ", "_").replace("-", "_"))
pd.DataFrame(resrows).to_csv(os.path.join(OUT, "resolution_all_models.csv"), index=False)

# ---------------------------------------------------------------- item 8: three references
def sgn(a, b):
    d = a[:, :4] - b[:, :4]
    return dict(x=d[:, 0] * 1e3, y=d[:, 1] * 1e3, tx=d[:, 2], ty=d[:, 3], radial=np.hypot(d[:, 0], d[:, 1]) * 1e3)
def stats(dd, m):
    out = {}
    for c in ("x", "y", "tx", "ty"):
        v = dd[c][m]; out[c + "_signed_median"] = np.median(v); out[c + "_half_width_68"] = (np.percentile(v, 84) - np.percentile(v, 16)) / 2
        out[c + "_median_abs"] = np.median(np.abs(v))
    out["radial_median_um"] = np.median(dd["radial"][m]); return out
rows = []
done_bench = set()
for lab, k, tracks, s in MODELS:
    d = load_pred(k, tracks, s)
    if d is None: continue
    T = TS[tracks]; p = T["momentum_gev"]; true = T["true_state_on_own_fibre_plane"]; bench = T["reference_state_on_own_fibre_plane"]
    pairs = [("prediction against true state", sgn(d["own"], true)), ("prediction against benchmark", sgn(d["own"], bench))]
    if tracks not in done_bench:
        pairs.append(("benchmark against true state (floor)", sgn(bench, true))); done_bench.add(tracks)
    for name, dd in pairs:
        for b, m in [("all momenta", np.ones_like(p, bool))] + band_masks(p):
            rows.append(dict(model=lab if "floor" not in name else "benchmark (" + MAPNAME[tracks] + " map)", run_key=k if "floor" not in name else "", map=MAPNAME[tracks],
                             comparison=name, band=b, number_of_tracks=int(m.sum()), **stats(dd, m)))
TR = pd.DataFrame(rows); TR.to_csv(os.path.join(OUT, "against_true_state.csv"), index=False)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
for a, c, ylab in ((a1, "x", "68 % half-width of Δx (µm)"), (a2, "tx", "68 % half-width of Δtx")):
    f = TR[(TR.comparison.str.contains("floor")) & (TR["map"] == "cubic")].set_index("band").loc[list(BANDS)]
    a.plot(x, f[c + "_half_width_68"], color=GREY, lw=3, label="benchmark against true state (floor)")
    for (lab, k, tracks, s), col in zip([m for m in MODELS if m[2] == "5558be31abeb"], COLOURS * 2):
        t = TR[(TR.model == lab) & (TR.comparison == "prediction against benchmark")].set_index("band").loc[list(BANDS)]
        a.plot(x, t[c + "_half_width_68"], marker="o", color=col, lw=1.6, label=lab + ": prediction against benchmark")
        t2 = TR[(TR.model == lab) & (TR.comparison == "prediction against true state")].set_index("band").loc[list(BANDS)]
        a.plot(x, t2[c + "_half_width_68"], ls="none", marker="x", color=col, ms=8)
    a.set_yscale("log"); a.set_xticks(x, BAND_LABELS); a.set_xlabel("momentum band (GeV)"); a.set_ylabel(ylab)
a1.legend(frameon=False, fontsize=8.5, loc="lower left")
a2.text(0.98, 0.98, "×: prediction against true state", transform=a2.transAxes, ha="right", va="top", fontsize=10, color=GREY)
save(fig, "against_true_state")

# benchmark against true state against momentum (cubic set)
T = TS["5558be31abeb"]; p = T["momentum_gev"]; dd = sgn(T["reference_state_on_own_fibre_plane"], T["true_state_on_own_fibre_plane"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
rows = []
for a, c, ylab, unit in ((a1, "x", "Δx, benchmark − true (µm)", 1), (a2, "tx", "Δtx, benchmark − true", 1)):
    v = dd[c]; a.scatter(p, np.abs(v), s=1.5, color=COLOURS[0], alpha=0.15, lw=0, rasterized=True)
    med, lo, hi = [], [], []
    for i in range(len(pm)):
        m = (p >= pb[i]) & (p < pb[i + 1]); w = np.abs(v[m])
        if m.sum() < 10: med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        med.append(np.median(w)); lo.append(np.percentile(w, 16)); hi.append(np.percentile(w, 84))
        rows.append(dict(component=c, p_low_gev=pb[i], p_high_gev=pb[i + 1], tracks=int(m.sum()), median_abs=med[-1], p16_abs=lo[-1], p84_abs=hi[-1],
                         signed_median=np.median(v[m]), half_width_68=(np.percentile(v[m], 84) - np.percentile(v[m], 16)) / 2))
    a.fill_between(pm, lo, hi, color=COLOURS[1], alpha=0.25, lw=0); a.plot(pm, med, color=COLOURS[1], lw=2.2, marker="o", ms=4, label="median |Δ| per bin, 16–84 % band")
    i10 = np.argmin(np.abs(pm - 10)); g = med[i10] * 10 / pm
    a.plot(pm, g, color=GREY, ls="--", lw=1.6, label="1/p guide (through the 10 GeV bin)")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("p (GeV)"); a.set_ylabel("|" + ylab + "|")
a1.legend(frameon=False, fontsize=10, loc="lower left")
save(fig, "benchmark_against_true_state"); pd.DataFrame(rows).to_csv(os.path.join(OUT, "benchmark_against_true_state.csv"), index=False)

# ---------------------------------------------------------------- item 9: steps and stages
SS = [  # class, map/tracks, key ; tracks for scoring
    ("direct states", "linear", "49e5e2450eaa"), ("direct states", "linear", "75200e48408c"), ("direct states", "linear", "f38ec3e29819"),
    ("straight line, per-track scale", "linear", "a66c06472d3d"), ("straight line, per-track scale", "linear", "6c4a3b022a2f"), ("straight line, per-track scale", "linear", "29cbba959300"),
    ("straight line, per-track scale", "linear", "25dfa17ef168"), ("straight line, per-track scale", "linear", "15242aa17c65"), ("straight line, per-track scale", "linear", "eeb809e9254d"),
    ("straight line, per-track scale", "linear", "4543836de1ce"), ("straight line, per-track scale", "linear", "9aed5ce39ff8"), ("straight line, per-track scale", "linear", "06a45b4dd0d3"),
    ("straight line, per-track scale (self-scaled Broyden)", "cubic", "c08a9da58999"), ("straight line, per-track scale (self-scaled Broyden)", "cubic", "1978be93a5c3"),
    ("straight line, per-track scale (self-scaled Broyden)", "cubic", best["straight_line_per_track"][0]), ("straight line, per-track scale (self-scaled Broyden)", "cubic", "29a416b77582"),
    ("direct states", "cubic", best["direct_states"][0]),
    ("physics layer", "cubic", "a524012c8d41"), ("physics layer", "cubic", "2e8f364b79e2"), ("physics layer", "cubic", best["physics_layer"][0]),
]
# 1,452-track linear subset of the afc6 test split, to put the exact linear scheme beside the old reports
lin = TS["afc61a558211"]
old = None
rows = []
for cls, mp, k in SS:
    c = conf(k); N, q = c["steps"]["number_of_steps"], c["steps"]["number_of_stages"]; st = state(k)
    if mp == "cubic":
        m, p95, src, ms, rnd = score_of(k, "5558be31abeb"); ntr = 13447; ex = exact_median("5558be31abeb", N, q)
    else:
        m, rep = stored_test(k); src = "stored report " + rep; ntr = 1452; ex = np.nan
    rows.append(dict(model_class=cls, map=mp, run_key=k, steps=N, stages=q, optimiser=c["training"]["optimiser"]["name"], state=st["state"], test_tracks=ntr,
                     test_median_um=m, exact_scheme_median_um=ex, source=src))
SD = pd.DataFrame(rows)
# exact linear on the 1,452 old test tracks: subset of afc6 by matching start states
z12 = np.load(f"{STORE}/tracks/12a8d35c3165/tracks.npz")
s12 = z12["test.start_state"]; ref12 = z12["test.reference_states_on_planes"][:, -1]
idx = {tuple(np.round(r, 9)): i for i, r in enumerate(lin["start_state"])}
sub = np.array([idx.get(tuple(np.round(r, 9)), -1) for r in s12])
for i, r in SD.iterrows():
    if r["map"] == "linear" and (sub >= 0).all():
        e = exact_end("afc61a558211", int(r.steps), int(r.stages))
        if e is not None: SD.loc[i, "exact_scheme_median_um"] = float(np.median(np.hypot(*((e[sub, :2] - ref12[:, :2]) * 1e3).T)))
SD.to_csv(os.path.join(OUT, "steps_and_stages.csv"), index=False)
print("subset matched", (sub >= 0).sum(), "of", len(sub))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
styles = {"direct states": COLOURS[1], "straight line, per-track scale": COLOURS[0], "straight line, per-track scale (self-scaled Broyden)": COLOURS[0], "physics layer": COLOURS[2]}
for (cls, mp), g in SD[SD.stages == 2].groupby(["model_class", "map"]):
    g = g.sort_values("steps")
    a1.plot(g.steps, g.test_median_um, ("-" if mp == "cubic" else "--"), marker="o", color=styles[cls], lw=2, label="%s, %s map" % (cls, mp))
for mp, col in (("cubic", "#2B2B2B"), ("linear", GREY)):
    g = SD[(SD.stages == 2) & (SD["map"] == mp)].drop_duplicates("steps").sort_values("steps")
    a1.plot(g.steps, g.exact_scheme_median_um, ":", color=col, lw=2.2, label="exact scheme (floor), %s map" % mp)
a1.set_xscale("log", base=2); a1.set_yscale("log"); a1.set_xlabel("steps (2 stages)"); a1.set_ylabel("endpoint radial median (µm)")
a1.legend(frameon=False, fontsize=8.5)
g = SD[(SD.steps == 64)]
for (cls, mp), gg in g.groupby(["model_class", "map"]):
    gg = gg.sort_values("stages"); a2.plot(gg.stages, gg.test_median_um, ("-" if mp == "cubic" else "--"), marker="o", color=styles[cls], lw=2)
ge = g[g["map"] == "linear"].drop_duplicates("stages").sort_values("stages"); a2.plot(ge.stages, ge.exact_scheme_median_um, ":", color=GREY, lw=2.2)
gc = g[g["map"] == "cubic"].drop_duplicates("stages").sort_values("stages"); a2.plot(gc.stages, gc.exact_scheme_median_um, ":", color="#2B2B2B", lw=2.2)
a2.set_xscale("log", base=2); a2.set_yscale("log"); a2.set_xlabel("stages (64 steps)"); a2.set_ylabel("endpoint radial median (µm)")
save(fig, "steps_and_stages")
print(SD.to_string())
