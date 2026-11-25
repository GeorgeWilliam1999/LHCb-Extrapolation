"""Item 8 with the LHCb adaptive Nyström added as a reference.

The LHCb adaptive Runge-Kutta-Nyström integrator (Rec TrackSTEPExtrapolator,
tolerance 5 um) is the compiled copy in Fast_extrapolator (`fx`, method
`rec_step`, reader `rec`: trilinear reads of the linear v8r1 magnet-up map, as
LHCb reads it). It carries the 13,447 test tracks from 2648.2 mm to 7826.0 mm.
The end states are then carried to each particle's own fibre plane with the
benchmark integrator of the set they are compared with, as item 8 does:
cubic map for the cubic set, linear map for the linear set.

Writes nystrom_three_references.csv, benchmark_against_true_state_with_nystrom.png
and against_true_state_with_nystrom.png. Reads the deck's _work cache and
against_true_state.csv; writes nothing else.
"""
import os, sys, time
import numpy as np, pandas as pd
from style import *

sys.path.insert(0, "/data/bfys/gscriven/LHCb_Extrapolation_Project/Fast_extrapolator/python")
from fastex.compiled import run as fx_run
from rkpinn import registry
from rkpinn.evaluation.three_references import carry_to_own_planes
from rkpinn.integrators.runge_kutta_sixth_order import REFERENCE_STEP_LENGTH_MM

WORK = "/data/bfys/gscriven/LHCb_Extrapolation_Project_deck_material/_work"
SETS = {"cubic": ("5558be31abeb", "v8r1_up_cubic"), "linear": ("afc61a558211", "v8r1_up")}
TS = {m: np.load(os.path.join(WORK, f"test_{k}.npz")) for m, (k, _) in SETS.items()}
LABEL = "LHCb adaptive Nyström"

# The two sets hold the same tracks; their start states differ by up to 0.13 um
# (each set was carried to the first plane with its own map), so the Nyström
# starts from the start state of the set it is compared with.
for c in ("true_state_on_own_fibre_plane", "own_fibre_plane_mm", "momentum_gev"):
    assert np.array_equal(TS["cubic"][c], TS["linear"][c]), c
END = {}
for m in SETS:
    t0 = time.time()
    END[m], counts = fx_run("rec_step", TS[m]["start_state"], FIRST_PLANE, LAST_PLANE, map_name="v8r1_up",
                            reader="rec", precision="double")
    print("Nyström from the %s start states: %d tracks, %.1f s, median accepted steps %d"
          % (m, len(END[m]), time.time() - t0, np.median(counts[:, 1])))

def sgn(a, b):
    d = a[:, :4] - b[:, :4]
    return dict(x=d[:, 0] * 1e3, y=d[:, 1] * 1e3, tx=d[:, 2], ty=d[:, 3], radial=np.hypot(d[:, 0], d[:, 1]) * 1e3)
def stats(dd, m):
    out = {}
    for c in ("x", "y", "tx", "ty"):
        v = dd[c][m]; out[c + "_signed_median"] = np.median(v); out[c + "_half_width_68"] = (np.percentile(v, 84) - np.percentile(v, 16)) / 2
        out[c + "_median_abs"] = np.median(np.abs(v))
    out["radial_median_um"] = np.median(dd["radial"][m]); return out

rows, own = [], {}
p = TS["cubic"]["momentum_gev"]; true = TS["cubic"]["true_state_on_own_fibre_plane"]
for m, (k, fmap) in SETS.items():
    T = TS[m]
    eq = registry.component("equation_of_motion", "lhcb")(registry.component("field_map", fmap)())
    own[m] = np.asarray(carry_to_own_planes(eq, END[m], LAST_PLANE, T["own_fibre_plane_mm"],
                                            step_length_of_the_integrator_mm=REFERENCE_STEP_LENGTH_MM))
    pairs = [("Nyström against benchmark, last plane", sgn(END[m], T["reference_states_on_planes"][:, -1])),
             ("Nyström against benchmark, own plane", sgn(own[m], T["reference_state_on_own_fibre_plane"])),
             ("Nyström against true state", sgn(own[m], true))]
    for name, dd in pairs:
        for b, mk in [("all momenta", np.ones_like(p, bool))] + band_masks(p):
            rows.append(dict(model=LABEL, method="fx rec_step, linear map read as LHCb reads it", benchmark_set=m,
                             carried_with=fmap, comparison=name, band=b, number_of_tracks=int(mk.sum()), **stats(dd, mk)))
NR = pd.DataFrame(rows); NR.to_csv(os.path.join(OUT, "nystrom_three_references.csv"), index=False)

# ---- against_true_state_with_nystrom.png: the item 8 figure, Nyström line added
TR = pd.read_csv(os.path.join(OUT, "against_true_state.csv"))
cubic_models = [lab for lab in TR[(TR["map"] == "cubic") & (TR.comparison == "prediction against benchmark")].model.unique()]
NYC = "#B5179E"
x = np.arange(len(BANDS))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
for a, c, ylab in ((a1, "x", "68 % half-width of Δx (µm)"), (a2, "tx", "68 % half-width of Δtx")):
    f = TR[(TR.comparison.str.contains("floor")) & (TR["map"] == "cubic")].set_index("band").loc[list(BANDS)]
    a.plot(x, f[c + "_half_width_68"], color=GREY, lw=3, label="benchmark against true state (floor)")
    for lab, col in zip(cubic_models, COLOURS * 2):
        t = TR[(TR.model == lab) & (TR.comparison == "prediction against benchmark")].set_index("band").loc[list(BANDS)]
        a.plot(x, t[c + "_half_width_68"], marker="o", color=col, lw=1.6, label=lab + ": prediction against benchmark")
        t2 = TR[(TR.model == lab) & (TR.comparison == "prediction against true state")].set_index("band").loc[list(BANDS)]
        a.plot(x, t2[c + "_half_width_68"], ls="none", marker="x", color=col, ms=8)
    n = NR[(NR.benchmark_set == "cubic") & (NR.comparison == "Nyström against benchmark, own plane")].set_index("band").loc[list(BANDS)]
    a.plot(x, n[c + "_half_width_68"], marker="s", color=NYC, lw=2.2, ls="--", label=LABEL + " (linear map): against cubic benchmark")
    n2 = NR[(NR.benchmark_set == "cubic") & (NR.comparison == "Nyström against true state")].set_index("band").loc[list(BANDS)]
    a.plot(x, n2[c + "_half_width_68"], ls="none", marker="x", color=NYC, ms=9)
    a.set_yscale("log"); a.set_xticks(x, BAND_LABELS); a.set_xlabel("momentum band (GeV)"); a.set_ylabel(ylab)
a1.legend(frameon=False, fontsize=8.5, loc="lower left")
a2.text(0.98, 0.98, "×: prediction against true state", transform=a2.transAxes, ha="right", va="top", fontsize=10, color=GREY)
save(fig, "against_true_state_with_nystrom")

# ---- benchmark_against_true_state_with_nystrom.png: the floor against p, Nyström against true state added
pb = np.logspace(0, np.log10(200), 16); pm = np.sqrt(pb[1:] * pb[:-1])  # as analysis.py
B = sgn(TS["cubic"]["reference_state_on_own_fibre_plane"], true); N = sgn(own["cubic"], true)
NB = sgn(own["cubic"], TS["cubic"]["reference_state_on_own_fibre_plane"])
def binned(v):
    med, lo, hi = [], [], []
    for i in range(len(pm)):
        m = (p >= pb[i]) & (p < pb[i + 1]); w = np.abs(v[m])
        if m.sum() < 10: med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        med.append(np.median(w)); lo.append(np.percentile(w, 16)); hi.append(np.percentile(w, 84))
    return np.array(med), np.array(lo), np.array(hi)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
for a, c, ylab in ((a1, "x", "|Δx| (µm)"), (a2, "tx", "|Δtx|")):
    a.scatter(p, np.abs(B[c]), s=1.5, color=COLOURS[0], alpha=0.15, lw=0, rasterized=True)
    med, lo, hi = binned(B[c])
    a.fill_between(pm, lo, hi, color=COLOURS[1], alpha=0.25, lw=0); a.plot(pm, med, color=COLOURS[1], lw=2.2, marker="o", ms=4, label="benchmark − true: median |Δ| per bin, 16–84 % band")
    med2, _, _ = binned(N[c]); a.plot(pm, med2, color=NYC, lw=1.8, ls="--", marker="s", ms=3, label=LABEL + " − true: median |Δ| per bin")
    med3, lo3, hi3 = binned(NB[c])
    a.fill_between(pm, lo3, hi3, color=NYC, alpha=0.15, lw=0); a.plot(pm, med3, color=NYC, lw=2.2, marker="^", ms=4, label=LABEL + " − benchmark (linear vs cubic map)")
    i10 = np.argmin(np.abs(pm - 10)); a.plot(pm, med[i10] * 10 / pm, color=GREY, ls=":", lw=1.6, label="1/p guide (through the 10 GeV bin)")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("p (GeV)"); a.set_ylabel(ylab)
a1.legend(frameon=False, fontsize=9, loc="lower left")
save(fig, "benchmark_against_true_state_with_nystrom")
print(NR[NR.band == "all momenta"][["benchmark_set", "comparison", "x_half_width_68", "radial_median_um", "tx_half_width_68"]].to_string())
