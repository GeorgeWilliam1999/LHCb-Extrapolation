"""Item 2: the field map. By along z, a slice in x-y, linear against cubic reading."""
import numpy as np, pandas as pd, os
from style import *
from rkpinn.equation_of_motion.field_map import load_field_map

lin = load_field_map("v8r1_up"); cub = load_field_map("v8r1_up_cubic")
cell = 1.0 / lin.inverse_cell_size
lo = lin.lowest_corner; hi = lo + (lin.number_of_points - 1) * cell
with open(os.path.join(WORK, "field_map_grid.txt"), "w") as f:
    f.write(lin.description() + "\n")
print(lin.description())

# (a) By against z on the beam axis and on one off-axis line
z = np.arange(lo[2], hi[2], 5.0)
lines = [("beam axis (x = 0, y = 0)", 0.0, 0.0), ("off axis (x = 1000 mm, y = 500 mm)", 1000.0, 500.0)]
rows = {"z_mm": z}
fig, ax = plt.subplots()
ax.axvspan(FIRST_PLANE, LAST_PLANE, color="#E36414", alpha=0.10, lw=0, label="the crossing (2648.2 to 7826 mm)")
for (lab, x0, y0), c in zip(lines, COLOURS):
    by = cub(np.full_like(z, x0), np.full_like(z, y0), z)[1]
    rows["By_T_" + lab.split(" (")[0].replace(" ", "_")] = by
    ax.plot(z, by, color=c, lw=1.8, label=lab)
ax.set_xlabel("z (mm)"); ax.set_ylabel("$B_y$ (T)"); ax.legend(frameon=False, loc="lower right")
save(fig, "field_map_By_along_z"); pd.DataFrame(rows).to_csv(os.path.join(OUT, "field_map_By_along_z.csv"), index=False)

# (b) By on an x-y slice near the magnet centre (z of the largest |By| on axis)
zc = z[np.argmax(np.abs(rows["By_T_beam_axis"]))]
xs = np.arange(-3000, 3001, 20.0); ys = np.arange(-2500, 2501, 20.0)
X, Y = np.meshgrid(xs, ys)
B = lin(X, Y, np.full_like(X, zc))[1]
fig, ax = plt.subplots(figsize=(9, 6))
vmax = np.nanmax(np.abs(B))
m = ax.pcolormesh(X, Y, B, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
ax.grid(False); ax.set_aspect("equal")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
cb = fig.colorbar(m, ax=ax, shrink=0.85); cb.set_label("$B_y$ (T)", fontsize=14)
ax.text(0.02, 0.97, "z = %.0f mm" % zc, transform=ax.transAxes, va="top", fontsize=12)
save(fig, "field_map_By_xy_slice")
pd.DataFrame({"x_mm": X.ravel(), "y_mm": Y.ravel(), "By_T": B.ravel()}).iloc[::4].to_csv(
    os.path.join(OUT, "field_map_By_xy_slice.csv"), index=False)  # every 4th point, 40 mm in x

# (c) linear against cubic reading over four grid cells along z, on one line
x0, y0 = 1000.0, 500.0
k0 = int(round((zc - lo[2]) / cell[2]))
za, zb = lo[2] + (k0 - 2) * cell[2], lo[2] + (k0 + 2) * cell[2]
zz = np.linspace(za, zb, 2001)
bl = lin(np.full_like(zz, x0), np.full_like(zz, y0), zz)[1]
bc = cub(np.full_like(zz, x0), np.full_like(zz, y0), zz)[1]
dl = np.gradient(bl, zz) * 1000; dc = np.gradient(bc, zz) * 1000   # T per metre
faces = lo[2] + np.arange(k0 - 2, k0 + 3) * cell[2]
fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6.5))
for a in (a1, a2):
    for f in faces: a.axvline(f, color=LIGHT, lw=1, ls="--", zorder=0)
a1.plot(zz, bl, color=COLOURS[1], lw=1.8, label="linear reading (LHCb)")
a1.plot(zz, bc, color=COLOURS[0], lw=1.8, label="cubic reading")
a1.set_ylabel("$B_y$ (T)"); a1.legend(frameon=False)
# drop the one-sample spike of np.gradient at the faces for the linear curve: draw it piecewise
for i in range(len(faces) - 1):
    s = (zz > faces[i] + 0.5) & (zz < faces[i + 1] - 0.5)
    a2.plot(zz[s], dl[s], color=COLOURS[1], lw=1.8, label="linear reading (LHCb)" if i == 0 else None)
a2.plot(zz, dc, color=COLOURS[0], lw=1.8, label="cubic reading")
a2.set_ylabel("d$B_y$/dz (T/m)"); a2.set_xlabel("z (mm)  —  dashed lines: grid cell faces")
a2.legend(frameon=False)
save(fig, "field_map_linear_against_cubic")
pd.DataFrame({"z_mm": zz, "By_linear_T": bl, "By_cubic_T": bc, "dBy_dz_linear_T_per_m": dl,
              "dBy_dz_cubic_T_per_m": dc}).iloc[::4].to_csv(os.path.join(OUT, "field_map_linear_against_cubic.csv"), index=False)
with open(os.path.join(WORK, "field_map_grid.txt"), "a") as f:
    f.write("slice z %.1f; panel c line x=%g y=%g, z %.1f to %.1f, faces %s\n" % (zc, x0, y0, za, zb, faces))
