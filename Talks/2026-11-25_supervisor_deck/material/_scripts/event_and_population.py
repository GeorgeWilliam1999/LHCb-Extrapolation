"""Items 3 and 4 (and the track overlay of the field-map slice): example event, single tracks, population."""
import numpy as np, pandas as pd, os, collections
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from style import *
from rkpinn.equation_of_motion.field_map import load_field_map

T = np.load(os.path.join(WORK, "test_5558be31abeb.npz"))
p = T["momentum_gev"]; s = T["start_state"]; ref = T["reference_states_on_planes"]; z = T["planes_mm"]
ev = T["event"]; n = len(p)
charge = np.sign(s[:, 4])
pdg = T["particle_type"]
pi = np.abs(pdg) == 211
agree = np.mean(np.sign(pdg[pi]) == charge[pi])

# ---- item 4: population
rows = []
for b, m in band_masks(p):
    rows.append(dict(band=b, number_of_tracks=int(m.sum()), fraction=m.mean()))
rows.append(dict(band="all momenta", number_of_tracks=n, fraction=1.0))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "population_counts_by_band.csv"), index=False)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
bins = np.logspace(0, np.log10(200), 50)
axs[0].hist(p, bins=bins, color=COLOURS[0]); axs[0].set_xscale("log")
for e in EDGES[1:-1]: axs[0].axvline(e, color=GREY, lw=1, ls="--")
axs[0].set_xlabel("momentum p (GeV)"); axs[0].set_ylabel("tracks")
tb = np.linspace(-0.35, 0.35, 71)
axs[1].hist(s[:, 2], bins=tb, color=COLOURS[0], histtype="step", lw=1.8, label="$t_x$ = dx/dz")
axs[1].hist(s[:, 3], bins=tb, color=COLOURS[1], histtype="step", lw=1.8, label="$t_y$ = dy/dz")
axs[1].set_xlabel("slope at the first plane"); axs[1].set_ylabel("tracks"); axs[1].legend(frameon=False)
cnt = [np.sum(charge < 0), np.sum(charge > 0)]
axs[2].bar(["negative", "positive"], cnt, color=[COLOURS[3], COLOURS[2]], width=0.6)
axs[2].set_ylabel("tracks"); axs[2].set_xlabel("charge"); axs[2].grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "population")
h, _ = np.histogram(p, bins); ht, _ = np.histogram(s[:, 2], tb); hy, _ = np.histogram(s[:, 3], tb)
pd.DataFrame({"momentum_bin_low_gev": bins[:-1], "momentum_bin_high_gev": bins[1:], "tracks": h}).to_csv(os.path.join(OUT, "population_momentum.csv"), index=False)
pd.DataFrame({"slope_bin_low": tb[:-1], "slope_bin_high": tb[1:], "tracks_tx": ht, "tracks_ty": hy}).to_csv(os.path.join(OUT, "population_slopes.csv"), index=False)
pd.DataFrame({"charge": ["negative", "positive"], "tracks": cnt}).to_csv(os.path.join(OUT, "population_charge.csv"), index=False)
species = collections.Counter(pdg)
with open(os.path.join(WORK, "population_notes.txt"), "w") as f:
    f.write("charge from sign of q/p; agrees with PDG sign for pions in %.4f of cases\n" % agree)
    f.write("species %s\n" % species.most_common(12))
    f.write("p median %.2f, mean %.2f GeV; |tx|<0.3 frac %.3f\n" % (np.median(p), p.mean(), np.mean(np.abs(s[:, 2]) < 0.3)))

# ---- item 3: one event, 15 to 40 tracks
cnts = collections.Counter(ev)
cand = sorted([e for e, c in cnts.items() if 20 <= c <= 30])
e0 = cand[len(cand) // 2]
sel = np.where(ev == e0)[0]
norm = LogNorm(1, 200); cmap = plt.get_cmap("viridis")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
for i in sel[np.argsort(-p[sel])]:
    c = cmap(norm(p[i]))
    a1.plot(z, ref[i, :, 0], color=c, lw=1.3); a2.plot(z, ref[i, :, 1], color=c, lw=1.3)
for a, lab in ((a1, "x (mm)"), (a2, "y (mm)")):
    a.set_xlabel("z (mm)"); a.set_ylabel(lab); a.set_xlim(FIRST_PLANE, LAST_PLANE)
cb = fig.colorbar(ScalarMappable(norm, cmap), ax=[a1, a2], shrink=0.9, pad=0.02); cb.set_label("p (GeV)", fontsize=14)
save(fig, "example_event_paths")
recs = []
for i in sel:
    for k in range(0, len(z), 4):
        recs.append(dict(event=int(e0), particle_in_event=int(T["particle_in_event"][i]), pdg=int(pdg[i]), momentum_gev=p[i],
                         charge=int(charge[i]), z_mm=z[k], x_mm=ref[i, k, 0], y_mm=ref[i, k, 1], tx=ref[i, k, 2], ty=ref[i, k, 3]))
pd.DataFrame(recs).to_csv(os.path.join(OUT, "example_event_paths.csv"), index=False)

# three single tracks near 2, 8, 40 GeV (central slopes, from the same test split)
picks = []
for target in (2.0, 8.0, 40.0):
    ok = np.where((np.abs(s[:, 2]) < 0.1) & (np.abs(s[:, 3]) < 0.05))[0]
    picks.append(ok[np.argmin(np.abs(p[ok] - target))])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
recs = []
for i, c in zip(picks, COLOURS):
    lab = "%.1f GeV, %s" % (p[i], "positive" if charge[i] > 0 else "negative")
    a1.plot(z, ref[i, :, 0] - ref[i, 0, 0] - ref[i, 0, 2] * (z - z[0]), color=c, lw=2, label=lab)
    a2.plot(z, ref[i, :, 0], color=c, lw=2, label=lab)
    for k in range(len(z)):
        recs.append(dict(event=int(ev[i]), particle_in_event=int(T["particle_in_event"][i]), momentum_gev=p[i], charge=int(charge[i]), z_mm=z[k], x_mm=ref[i, k, 0], y_mm=ref[i, k, 1], tx=ref[i, k, 2], ty=ref[i, k, 3],
                         x_minus_straight_line_mm=ref[i, k, 0] - ref[i, 0, 0] - ref[i, 0, 2] * (z[k] - z[0])))
a2.set_ylabel("x (mm)"); a1.set_ylabel("x minus the straight line (mm)")
for a in (a1, a2): a.set_xlabel("z (mm)"); a.set_xlim(FIRST_PLANE, LAST_PLANE)
a1.legend(frameon=False)
fig.tight_layout(); save(fig, "example_event_single_tracks")
pd.DataFrame(recs).to_csv(os.path.join(OUT, "example_event_single_tracks.csv"), index=False)
with open(os.path.join(WORK, "event_notes.txt"), "w") as f:
    f.write("event %d, %d tracks; singles %s p %s\n" % (e0, len(sel), list(map(int, picks)), p[picks]))

# ---- item 2b again: x-y slice with the tracks at that z
lin = load_field_map("v8r1_up")
zc = 4720.0; k = int(np.argmin(np.abs(z - zc)))
xs = np.arange(-3000, 3001, 20.0); ys = np.arange(-1600, 1601, 20.0)
X, Y = np.meshgrid(xs, ys); B = lin(X, Y, np.full_like(X, z[k]))[1]
fig, ax = plt.subplots(figsize=(10, 5.6))
m = ax.pcolormesh(X, Y, B, cmap="cividis", vmin=0, vmax=1.6, shading="auto", rasterized=True)
rng = np.random.default_rng(1); sub = rng.choice(n, 3000, replace=False)
ax.scatter(ref[sub, k, 0], ref[sub, k, 1], s=2, color="white", alpha=0.6, lw=0, label="test tracks at this z")
ax.grid(False); ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
cb = fig.colorbar(m, ax=ax, shrink=0.85, extend="both"); cb.set_label("$B_y$ (T)", fontsize=14)
ax.text(0.01, 0.97, "z = %.0f mm" % z[k], transform=ax.transAxes, va="top", fontsize=12, color="white")
save(fig, "field_map_By_xy_slice")
pd.DataFrame({"x_mm": X.ravel(), "y_mm": Y.ravel(), "By_T": B.ravel()}).iloc[::2].to_csv(os.path.join(OUT, "field_map_By_xy_slice.csv"), index=False)
print(open(os.path.join(WORK, "event_notes.txt")).read(), open(os.path.join(WORK, "population_notes.txt")).read())
