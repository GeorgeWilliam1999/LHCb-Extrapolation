"""Item 5: classical methods on the cubic map, test split, from the per-track cache of
what_are_the_classical_references_on_the_cubic_map (no new integration)."""
import numpy as np, pandas as pd, os
from style import *
C = "/data/bfys/gscriven/rkpinn_analysis_cache/what_are_the_classical_references_on_the_cubic_map"
T = np.load(os.path.join(WORK, "test_5558be31abeb.npz")); p = T["momentum_gev"]; ref = T["reference_states_on_planes"][:, -1]
rows = []
methods = [("RK4", "rk4_%03d_steps.npz", None)] + [("exact Gauss-Legendre, %d stages" % q, "exact_%%03d_steps_%02d_stages.npz" % q, q) for q in (2, 4, 8, 16)]
for name, pat, q in methods:
    for N in (16, 32, 64, 128, 256):
        f = os.path.join(C, pat % N)
        if not os.path.exists(f): continue
        e = np.load(f)["endpoint"]
        for b, m in [("all momenta", np.ones_like(p, bool))] + band_masks(p):
            rows.append(dict(method=name, steps=N, stages=4 if q is None else q, field_lookups_per_track=N * (4 if q is None else q),
                             band=b, number_of_tracks=int(m.sum()), median_um=np.median(e[m]), p95_um=np.percentile(e[m], 95)))
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "gauss_legendre_accuracy.csv"), index=False)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
a = d[d.band == "all momenta"]
for (name, _, q), c in zip(methods, [GREY] + COLOURS):
    s = a[a.method == name]
    ls = "--" if q is None else "-"
    a1.plot(s.steps, s.median_um, ls, marker="o", color=c, lw=2, label=name.replace("exact Gauss-Legendre", "exact GL"))
    a1.plot(s.steps, s.p95_um, ":", color=c, lw=1.2)
a1.axhline(3e-6, color=LIGHT, lw=1); a1.text(17, 4e-6, "error of the reference itself", color=GREY, fontsize=10)
a1.set_xscale("log", base=2); a1.set_yscale("log"); a1.set_xlabel("steps across the crossing"); a1.set_ylabel("endpoint radial error (µm)")
a1.legend(frameon=False, fontsize=10); a1.text(0.98, 0.98, "solid/dashed: median   dotted: 95 %", transform=a1.transAxes, ha="right", va="top", fontsize=10, color=GREY)
b = d[(d.steps == 64) & (d.band != "all momenta")]
x = np.arange(5)
for (name, _, q), c in zip(methods, [GREY] + COLOURS):
    s = b[b.method == name].set_index("band").loc[list(BANDS)]
    a2.plot(x, s.median_um, ("--" if q is None else "-"), marker="o", color=c, lw=2)
a2.set_xticks(x, BAND_LABELS); a2.set_yscale("log"); a2.set_xlabel("momentum band (GeV), 64 steps"); a2.set_ylabel("endpoint radial median (µm)")
save(fig, "gauss_legendre_accuracy")
print(a.pivot_table(index="steps", columns="method", values="median_um").to_string())
