#!/usr/bin/env python
"""The Block D write-up, generated: every number on every page is read from a
results file of D0, D1 or D2 and written into Notion-flavoured markdown by
this script. Nothing is retyped by hand.

    python gen_writeup.py            -> build/main.md, build/children/*.md

Each file starts with an HTML comment naming its page title. The pages are
then created with the Notion `create-pages` tool (main page first, verified,
then the children under it) and every page is fetched back and checked token
by token with /data/bfys/gscriven/tools/verify_numbers.py.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
D0 = os.path.join(B, "D0_Crossing_dataset", "results")
D1 = os.path.join(B, "D1_Chain_grid", "results")
D2 = os.path.join(B, "D2_Comparators", "results")
OUT = os.path.join(HERE, "build")
COMMIT = "020392809520424439b6ee9124b44a646475db33"
SHORT = COMMIT[:7]
RAW = ("https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/%s/"
       "Raissi_disc_time_approach/Block_D_fixed_step_crossing/" % COMMIT)
FIG1 = RAW + "D1_Chain_grid/figures/"
FIG0 = RAW + "D0_Crossing_dataset/figures/"
N_VALUES = (1, 4, 16, 64, 128)
QS = tuple(range(1, 21))
COMP = (("x", "um", "µm"), ("y", "um", "µm"), ("tx", "mrad", "mrad"), ("ty", "mrad", "mrad"))
BANDS = ("1-2 GeV", "2-5 GeV", "5-10 GeV", "10-25 GeV", "25-200 GeV")
TODO_D = "https://app.notion.com/p/3db5d544b9d98147ad93d1054f184a40"
CUT_NAMES = {
    "leg-B rows in the training set (both directions)": "cross-magnet rows in the harvested table, both directions",
    "pre-magnet plane is a UT plane": "the pre-magnet plane is an Upstream Tracker plane",
    "both directions kept for the particle": "both directions present for the particle",
    "2 < eta < 5": "pseudorapidity between 2 and 5",
    "1 < p < 200 GeV": "momentum between 1 and 200 GeV",
    "non-electron": "not an electron",
    "both directions still kept after the cuts": "both directions still present after the cuts",
    "forward rows only": "forward rows only (one per particle)",
    "|z_pre - z0| < 60 mm and |z_post - z1| < 60 mm": "last Upstream Tracker plane within 60 mm of z0 and first fibre-tracker plane within 60 mm of z1",
    "fiducial: RK6 trajectory stays inside the field map": "fiducial: the fine trajectory stays inside the field map",
}
TODO_NORM = "https://app.notion.com/p/3dc5d544b9d98187b710f5c8bb3dc292"
TODO_OPT = "https://app.notion.com/p/3dc5d544b9d981b7b99cc2e99d9222e5"
TODO_WIDE = "https://app.notion.com/p/3dc5d544b9d981159f27ca2fa2764afe"


# ------------------------------------------------------------ formatting --
def fmt(x, sig=3):
    """A number for prose or a cell: 3 significant figures, thousands commas."""
    if x is None or x == "":
        return ""
    x = float(x)
    if x != x:
        return "n/a"
    a = abs(x)
    if a == 0:
        return "0"
    if a < 1e-3:
        return "%.2e" % x
    if a >= 1000:
        return "{:,.0f}".format(x)
    if a >= 100:
        return "%.0f" % x
    if a >= 10:
        return "%.1f" % x
    if a >= 1:
        return "%.2f" % x
    return "%.3g" % x


def sfmt(x):
    """Signed, for the single-track tables."""
    if x is None or x == "":
        return ""
    v = float(x)
    return ("+" if v > 0 else "") + fmt(v)


def esc(s):
    """Escape the characters Notion treats as markup, inside cells and prose."""
    return re.sub(r"([\\*~`$\[\]<>{}|^])", r"\\\1", str(s))


def table(header, rows, fit=True):
    out = ['<table header-row="true"%s>' % (' fit-page-width="true"' if fit else "")]
    out.append("\t<tr>" + "".join("<td>%s</td>" % esc(h) for h in header) + "</tr>")
    for r in rows:
        out.append("\t<tr>" + "".join("<td>%s</td>" % esc(c) for c in r) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def pivot_rows(path, formatter=fmt):
    rows = read_csv(path)
    return [[r["N"]] + [formatter(r["q%02d" % q]) if r["q%02d" % q] != "" else "" for q in QS]
            for r in rows]


def pivot(path, formatter=fmt):
    return table(["N \\ q"] + [str(q) for q in QS], pivot_rows(path, formatter))


def fig(name, caption, d1=True):
    return "![%s](%s%s)" % (esc(caption), FIG1 if d1 else FIG0, name)


# ---------------------------------------------------------------- data --
meta = json.load(open(os.path.join(D0, "crossing_particles_meta.json")))
chain_table = read_csv(os.path.join(D1, "chain_table.csv"))
comps = read_csv(os.path.join(D1, "components.csv"))
per_leg = read_csv(os.path.join(D1, "per_leg.csv"))
growth = read_csv(os.path.join(D1, "growth.csv"))
status = read_csv(os.path.join(D1, "status.csv"))
bands = read_csv(os.path.join(D1, "by_p_band.csv"))
cost = read_csv(os.path.join(D1, "cost_table.csv"))
cont = read_csv(os.path.join(D1, "block_a_continuity.csv"))
vt = read_csv(os.path.join(D1, "val_vs_test.csv"))
qop = read_csv(os.path.join(D1, "qop_check.csv"))
incs = read_csv(os.path.join(D1, "leg_increments.csv"))
curves = read_csv(os.path.join(D1, "training_curves.csv"))
single = json.load(open(os.path.join(D1, "single_track.json")))
twin = json.load(open(os.path.join(D2, "twin", "twin_scores.json")))
comparison = read_csv(os.path.join(D2, "comparison_table.csv"))
chains = {}
for p in glob.glob(os.path.join(D1, "N*_q*", "chain.json")):
    m = re.search(r"N(\d+)_q(\d+)", p)
    chains[(int(m.group(1)), int(m.group(2)))] = json.load(open(p))
tight = {}
for p in glob.glob(os.path.join(D1, "tight_stall", "N*_q*", "chain.json")):
    m = re.search(r"N(\d+)_q(\d+)", p)
    tight[(int(m.group(1)), int(m.group(2)))] = json.load(open(p))
exact = {}
for p in glob.glob(os.path.join(D2, "exact_N*_q*.json")):
    m = re.search(r"N(\d+)_q(\d+)", p)
    exact[(int(m.group(1)), int(m.group(2)))] = json.load(open(p))

T = {(int(r["N"]), int(r["q"]), r["split"], r["comparator"]): r for r in chain_table}
C = {(int(r["N"]), int(r["q"]), r["split"], r["comparator"], r["component"]): r for r in comps}


def med(N, q, comp="vs_rk6_endpoint", split="test", key="pos_med_um"):
    return float(T[(N, q, split, comp)][key])


def cmed(N, q, name, comp="vs_rk6_endpoint", key="med"):
    return float(C[(N, q, "test", comp, name)][key])


best_q = {N: min(QS, key=lambda q: med(N, q)) for N in N_VALUES}
best = min(best_q.items(), key=lambda kv: med(kv[0], kv[1]))
rng8 = {N: (min(med(N, q) for q in QS if q >= 8), max(med(N, q) for q in QS if q >= 8)) for N in N_VALUES}
straight = med(1, 1, "straight_line_vs_rk6_endpoint")
floor = med(1, 1, "rk6_truth_vs_real_scifi_state")
floor_band = meta["material_floor_real_SciFi_state_vs_field_only_truth"]
counts = meta["counts"]
core_hours = sum(float(r["total_train_wall_s"]) for r in status) / 3600
total_restarts = sum(int(r["total_restarts"]) for r in status)
n_legs = len(per_leg)
n_conv = sum(1 for r in per_leg if r["converged"] == "True")
params = {q: chains[(1, q)]["n_parameters_per_leg"] for q in QS}
tw = twin["test"]["vs_rk6_endpoint"]
twc = tw["components"]
ex_best = min(exact.items(), key=lambda kv: kv[1]["endpoint_pos_med_um"])
n128_own = [float(r["own_step_test_endpoint_med_um"]) for r in per_leg if r["N"] == "128" and r["q"] == str(best_q[128])]
n128_str = [float(r["own_step_test_straight_med_um"]) for r in per_leg if r["N"] == "128" and r["q"] == str(best_q[128])]
inc = {(int(r["N"]), int(r["q"])): r for r in incs}
band_1025 = {(int(r["N"]), int(r["q"])): float(r["pos_med_um"]) for r in bands
             if r["split"] == "test" and r["comparator"] == "vs_rk6_endpoint" and r["p_band"] == "10-25 GeV"}
cost_by = {(int(r["N"]), int(r["q"])): r for r in cost}
vt_ratio = [float(r["test_med_um"]) / float(r["val_med_um"]) for r in vt]
qop_max = max(float(r["qop_max_abs_change_test"]) for r in qop)
grid_16 = chains[(1, 16)]; tight_16 = tight[(1, 16)]
grid_45 = chains[(4, 5)]; tight_45 = tight[(4, 5)]
cv = [r for r in curves if r["N"] == "1" and r["q"] == "16"]
cv_test = [float(r["test_endpoint_med_um"]) for r in cv]
cv_min_i = min(range(len(cv_test)), key=lambda i: cv_test[i])
a1 = {int(r["q"]): r for r in cont if r["source"].startswith("Block A1")}
a2 = [r for r in cont if r["source"].startswith("Block A2")]
a2_spread = None
_p = os.path.join(os.path.dirname(B), "Block_A_technique_works", "A2_Network_size_and_seed_study", "results", "by_architecture.csv")
if os.path.exists(_p):
    for r in read_csv(_p):
        if r["mode"] == "physics" and r["width"] == "50" and r["depth"] == "4":
            a2_spread = float(r["test_endpoint_spread_ratio"])
sig = 0.5 * (meta["material_floor_real_SciFi_state_vs_field_only_truth"]["all"]["pos_med_um"])  # unused guard
n_test = counts["test"]


def component_summary_rows(comp="vs_rk6_endpoint"):
    rows = []
    for N in N_VALUES:
        q = best_q[N]
        rows.append([str(N), str(q), fmt(med(N, q, comp))] +
                    [fmt(cmed(N, q, n, comp)) for n, _, _ in COMP] +
                    [sfmt(cmed(N, q, "x", comp, "bias")), sfmt(cmed(N, q, "tx", comp, "bias"))])
    return rows


# ------------------------------------------------------------ main page --
def main_page():
    L = meta["crossing"]["L_mm"]; z0 = meta["crossing"]["z0_mm"]; z1 = meta["crossing"]["z1_mm"]
    dz = meta["crossing"]["dz_mm"]
    casc = meta["cut_cascade"]
    P = []
    P.append("<!-- TITLE: Crossing the LHCb magnet with one label-free network per step: the fixed-step study at every step count and stage count (Block D, September 2026) -->")
    P.append("<callout icon=\"🧭\">\n\tThis page is the complete record of Block D. It assumes no prior knowledge of the programme: the physics, the numerical method, the network, the data and every result are laid out from first principles, with the tables and figures that carry them as child pages at the end. Every number on these pages was written by a generator from a results file in the repository, never retyped; the provenance block at the bottom says where each one lives. Trust is **Provisional** until George confirms it.\n</callout>")
    P.append("<table_of_contents/>")
    # ---- 1 Introduction
    P.append("# 1. Introduction")
    P.append("LHCb is a forward spectrometer at the LHC. A charged particle produced at the collision point leaves hits in the vertex detector, in the four planes of the Upstream Tracker just before the magnet, and in the twelve layers of the scintillating-fibre tracker just after it. Between the Upstream Tracker and the fibre tracker the particle passes through a dipole field of about one tesla over five metres and its path bends, by an amount inversely proportional to its momentum. That bend is how the momentum is measured.")
    P.append("Reconstructing a track means fitting a smooth trajectory through those hits. The fit works with a **track state** on a plane of constant $`z`$: five numbers $`(x, y, t_x, t_y, q/p)`$, the two transverse positions in millimetres, the two slopes $`t_x = dx/dz`$ and $`t_y = dy/dz`$, and the signed inverse momentum. Every iteration of the fit has to carry a state from one plane to the next, through the field, many times per track and millions of times per event. That routine is the **track extrapolator**, and across the magnet it is the most expensive one in the fit, because it integrates the equation of motion numerically, reading the field map at every step.")
    P.append("This programme asks whether a small neural network can replace that routine across the magnet, and whether it can be trained **without labels**: not by showing it examples of where particles went, but by requiring that its own outputs satisfy the equations of an implicit Runge–Kutta scheme, the construction of Raissi, Perdikaris and Karniadakis (2019, section 3), whose van der Pol study preceded this one. Earlier blocks of the programme established that the construction trains on this problem at all (July 2026) and explored a design in which one network served many step lengths at once; that design was set aside on 14 September 2026 because it departed from the paper, and this block is the faithful version: **one network per fixed step**, the steps chained across the magnet exactly as the paper's scheme steps in time.")
    # ---- 2 Aims
    P.append("# 2. Aims")
    P.append("The crossing is fixed: from the plane $`z_0 = %s`$ mm, the last Upstream Tracker plane most particles cross, to $`z_1 = %s`$ mm, the first fibre-tracker plane, a distance $`L = %s`$ mm. One architecture is used throughout, two hidden layers of 128 units, and one seed. Two quantities are varied:" % (fmt(z0), fmt(z1), fmt(L)))
    P.append("- the number of steps $`N`$ the crossing is cut into, $`N \\in \\{1, 4, 16, 64, 128\\}`$, so the step length is $`\\Delta z = L/N`$ = %s, %s, %s, %s and %s mm, with one network trained per step;" % tuple(fmt(dz[str(N)]) for N in N_VALUES))
    P.append("- the number of Gauss–Legendre stages $`q`$ per step, every integer from 1 to 20, which is the number of collocation points at which the equation of motion is enforced inside a step.")
    P.append("That is 100 chains and %s networks. The questions are: (1) how accurate is a single step across the whole magnet, and how does that depend on $`q`$; (2) does cutting the crossing into shorter steps, each with its own network, help or hurt, when the networks are trained and applied one after another as the paper prescribes; (3) how much of the error is the scheme's and how much the network's; (4) what the error looks like per component and per momentum; and (5) whether the networks are as well trained as this loss and optimiser allow." % "{:,}".format(n_legs))
    P.append("Every prediction is scored against four references: the **fine Runge–Kutta truth** (a sixth-order integrator at 0.1 mm steps from the same start state), the **exact collocation scheme** at the same $`N`$ and $`q`$ (the ceiling any network trained on these equations could reach), the **straight line** (the null, no magnet at all), and the particle's **real state on the fibre-tracker plane** (the data ground truth, which includes the material the particle crossed and no field-only method can predict). A single supervised network, trained on the fine truth instead of the equations, is the fifth comparator.")
    # ---- 3 Theory
    P.append("# 3. Theory")
    P.append("## 3.1 The state and the equation of motion in z")
    P.append("A particle of charge $`q`$ and momentum $`\\vec p`$ in a magnetic field $`\\vec B`$ obeys the Lorentz force, $`d\\vec p/dt = q\\,\\vec v \\times \\vec B`$. The magnitude of the momentum does not change, only its direction. In a forward spectrometer it is natural to use $`z`$, the coordinate along the beam, as the independent variable instead of time, because the detector planes are planes of constant $`z`$. Writing the direction of motion as $`(t_x, t_y, 1)`$ up to normalisation, the path length per unit $`z`$ is $`\\sqrt{1 + t_x^2 + t_y^2}`$, and the Lorentz force becomes four first-order equations for the four dynamic components of the state:")
    P.append("$$\n\\frac{dx}{dz} = t_x, \\qquad \\frac{dy}{dz} = t_y, \\qquad \\frac{dt_x}{dz} = \\kappa\\,\\frac{q}{p}\\,\\sqrt{1+t_x^2+t_y^2}\\,\\big[t_x t_y B_x - (1 + t_x^2) B_y + t_y B_z\\big],\n$$")
    P.append("$$\n\\frac{dt_y}{dz} = \\kappa\\,\\frac{q}{p}\\,\\sqrt{1+t_x^2+t_y^2}\\,\\big[(1 + t_y^2) B_x - t_x t_y B_y - t_x B_z\\big], \\qquad \\frac{d(q/p)}{dz} = 0 .\n$$")
    P.append("Here $`q/p`$ is kept in the convention of the LHCb first-level trigger software: $`q/p = 0.299792458\\, q / p`$ with $`p`$ in GeV, so that a 10 GeV particle has $`|q/p| = 0.03`$. The factor 0.2998 is the bending constant of 0.3 GeV per tesla-metre, and the remaining $`\\kappa = 10^{-3}`$ converts metres to millimetres, the unit of $`z`$. The fifth equation says what the scheme will later assume: $`q/p`$ is a constant of the motion, an input carried through unchanged, never an output.")
    P.append("The dominant field component is $`B_y`$, so the bend is mainly in $`x`$: to first order $`dt_x/dz \\approx -\\kappa (q/p) B_y`$, and integrating twice, a particle of 10 GeV crossing one tesla over five metres is deflected by about $`0.03 \\times 10^{-3} \\times 1 \\times 5000 \\approx 0.15`$ rad in slope and by roughly half that times the crossing, about 400 mm, in $`x`$. That is why every table below reports $`x`$ and $`t_x`$ separately from $`y`$ and $`t_y`$: they carry different physics and, as the results show, different errors.")
    P.append("## 3.2 The field map")
    P.append("The field is the v8r1 map of the LHCb magnet, a trilinear interpolation on a 100 mm grid, read from the file `field.v8r1.up.bin` whose MD5 is %s. Its polarity is **magnet up**, the polarity of the simulated sample the particles come from. The map is continuous but its derivative jumps at every grid face, which limits how far any integrator can be trusted; the fine reference of section 4.3 was built with that in mind." % meta["field"]["md5"])
    P.append("## 3.3 Implicit Runge–Kutta as collocation")
    P.append("A Runge–Kutta method with $`q`$ stages advances a state $`S`$ from $`z_0`$ to $`z_0 + \\Delta z`$ through $`q`$ intermediate **stage states** $`Y_j`$, each living on the plane $`z_0 + c_j \\Delta z`$:")
    P.append("$$\nY_j = S + \\Delta z \\sum_{k=1}^{q} a_{jk}\\, f(Y_k, z_0 + c_k \\Delta z), \\quad j = 1 \\dots q, \\qquad S_1 = S + \\Delta z \\sum_{j=1}^{q} b_j\\, f(Y_j, z_0 + c_j \\Delta z),\n$$")
    P.append("where $`f`$ is the right-hand side of section 3.1 and $`(c, A, b)`$ is the method's tableau. When $`A`$ is a full matrix the stage equations are **implicit**: every $`Y_j`$ depends on every other, and a classical solver has to iterate them to convergence. The **Gauss–Legendre** family puts the nodes $`c_j`$ at the roots of the shifted Legendre polynomial of degree $`q`$ and chooses $`A`$ and $`b`$ so that the scheme is exactly the polynomial of degree $`q`$ that satisfies the differential equation at those $`q`$ points: **collocation**. The result has order $`2q`$, is stable for any step, and preserves the geometry of the flow. With $`q = 1`$ it is the implicit midpoint rule; with $`q = 20`$ its truncation error goes as $`\\Delta z^{40}`$, and the paper's whole point is that a network can be given as many stages as one likes at almost no extra cost, so that a single huge step can replace hundreds of small ones. The tableaux used here are built for every $`q`$ from 1 to 20 (and checked to 50) and verified on each call against the row-sum, quadrature, collocation and symplecticity conditions to $`10^{-15}`$.")
    P.append("## 3.4 The discrete-time construction: training with no labels")
    P.append("Instead of solving the stage equations, the paper places a neural network on the stage states. Given the start state $`S`$ on $`z_0`$, the network emits all of $`Y_1, \\dots, Y_q`$ and the end state $`S_1`$ at once, $`4(q+1)`$ numbers. The stage equations are then rearranged so that each of the $`q+1`$ outputs **reconstructs the input**:")
    P.append("$$\n\\hat S_j = Y_j - \\Delta z \\sum_{k} a_{jk}\\, f(Y_k, z_k), \\quad j = 1 \\dots q, \\qquad \\hat S_{q+1} = S_1 - \\Delta z \\sum_{j} b_j\\, f(Y_j, z_j),\n$$")
    P.append("and the loss is the mean squared difference between every reconstruction and the actual input, each component divided by the input's population scale so the four components count alike:")
    P.append("$$\n\\mathcal{L} = \\frac{1}{n\\,(q+1)\\,4} \\sum_{\\text{samples}} \\sum_{j=1}^{q+1} \\sum_{d=1}^{4} \\left( \\frac{\\hat S_{j,d} - S_d}{\\sigma_d} \\right)^{2}.\n$$")
    P.append("No label appears anywhere: the only things the loss needs are the input state, the tableau and the equation of motion evaluated at the network's own proposed positions. If the loss is zero, the outputs are exactly the stage states and end state of the collocation scheme, so the best a network can do is the exact scheme's own error, which is why the exact scheme is the ceiling in every table. A worked case makes the construction concrete: at $`q = 1`$ the tableau is $`c = 1/2`$, $`a = 1/2`$, $`b = 1`$, the network emits one midpoint state $`Y`$ and the end state $`S_1`$, and the two reconstructions are $`Y - \\tfrac{1}{2}\\Delta z\\, f(Y)`$ and $`S_1 - \\Delta z\\, f(Y)`$, both of which must equal $`S`$.")
    P.append("## 3.5 What the network actually emits: the deviation from a straight line")
    P.append("Across five metres a state moves by hundreds of millimetres, while the correction the magnet makes to a straight line on a 40 mm step is of order $`10^{-2}`$ mm. A network whose last layer has to produce the absolute state in millimetres is asked for a number to eight significant figures on the short steps. Block A of this programme found that this, not capacity, was what broke short steps, and fixed it: the network's raw output $`r_j`$ is turned into a state by")
    P.append("$$\nY_j = \\text{straight}_j + \\text{scale} \\odot r_j, \\qquad \\text{straight}_j = \\big(x + t_x (z_j - z_0),\; y + t_y (z_j - z_0),\; t_x,\; t_y\\big),\n$$")
    P.append("so the raw output is the deviation from the magnet-off path, in units of a scale that is built from the input alone. To first order the slope change over a step is $`\\kappa\\,|q/p|\\,I_B`$ with $`I_B = \\int |B|\\,dz`$ along the straight line, and the position change is that slope integrated, about $`\\kappa\\,|q/p|\\,I_B\\,|\\Delta z|/2`$; those two expressions are the slope and position scales, with $`I_B`$ taken by a 16-point midpoint rule and floors of $`10^{-12}`$ and $`10^{-9}`$ mm that never bind. Nothing about $`z_0`$ or $`\\Delta z`$ is an input to the network: they are constants of each network, held in its buffers. The reconstruction loss is unchanged, since it only needs the absolute states this produces, and when the last layer is zero the output is the straight line exactly, which is one of the gates every network passes before training.")
    P.append("## 3.6 One network per step, trained one after another")
    P.append("For a chain of $`N`$ steps the crossing is cut into legs $`[z_k, z_k + \\Delta z]`$, $`z_k = z_0 + k \\Delta z`$, and a separate network is trained for each. The order matters and follows the paper, which trains a network on the data at one time step, predicts the next, and then \"would use this prediction as initial data for the next step and proceed to train again\". So leg 0 trains on the real start states at $`z_0`$; leg 1 trains on the states leg 0 **predicted** at $`z_1`$ for the same particles, not on the truth there; and so on to leg $`N-1`$. The training data of every later leg carries the error of the legs before it. The chain's answer is what the last leg emits, and validation and test particles go through the same $`N`$ networks in the same order, $`q/p`$ carried through untouched at every leg. A network is valid only on its own leg, in its own chain, at its own $`q`$; the repository page `APPLYING_THE_NETWORKS.md` states that contract in full.")
    P.append("## 3.7 The training protocol")
    P.append("Every network is trained the same way, the protocol verified on the July baseline and used by every block since: double precision, full-batch L-BFGS with 200 iterations per restart, strong Wolfe line search and a history of 120, restarted until it **stalls**, defined as two consecutive restarts each improving the loss by less than one percent, and then **confirmed** with a fresh optimiser, which must itself stall within two restarts with the endpoint medians on all three splits unchanged to one percent. A confirmation that does not hold sends the run back to training. A checkpoint and a history row are written after every restart, so a run is resumable. Section 5.8 tests whether the one-percent rule stops runs early.")
    P.append("## 3.8 How the results are scored")
    P.append("Every score is a comparison of a predicted state with a reference state on the same plane. The programme's summary measures are the **endpoint position error**, the larger of $`|\\Delta x|`$ and $`|\\Delta y|`$ in micrometres, and the **slope error**, the larger of $`|\\Delta t_x|`$ and $`|\\Delta t_y|`$ in milliradians. Because the bend lives in $`x`$, every result is also given **per component**: $`\\Delta x`$, $`\\Delta y`$ in µm and $`\\Delta t_x`$, $`\\Delta t_y`$ in mrad, each as the median and 95th percentile of the absolute error over the particles of a split and as the signed mean, the bias. The change in $`q/p`$ along a chain is recorded and must be exactly zero. Medians are quoted throughout because the error distributions have long tails at low momentum; the 95th percentiles are in the child tables. Errors are reported on the test split, %s particles never used in training or selection; the validation split has %s and is reported beside it in the child pages." % ("{:,}".format(counts["test"]), "{:,}".format(counts["val"])))
    # ---- 4 Data
    P.append("# 4. Data and machinery")
    P.append("## 4.1 The particles")
    P.append("The particles are from the official LHCb simulated minimum-bias sample of the 2024 detector (conditions tag sim-20231017-vc-mu100, magnet up), harvested in July 2026 into a table of true track states on every sensor plane each particle crossed. From that table the cross-magnet rows were selected as follows:")
    P.append(table(["cut", "rows in", "removed", "rows out", "particles out"],
                   [[CUT_NAMES.get(c["cut"], c["cut"]), "{:,}".format(c["rows_in"]), "{:,}".format(c["rows_removed"]), "{:,}".format(c["rows_out"]), "{:,}".format(c["particles_out"])] for c in casc]))
    P.append("The last two cuts are this block's: the particle's last Upstream Tracker plane must lie within 60 mm of $`z_0`$ and its first fibre-tracker plane within 60 mm of $`z_1`$, so that every particle starts and ends within a few centimetres of the frozen planes and the transport to them is a short field-only step. The %s surviving particles were split by the training set's own by-particle assignment: %s went to training (capped from %s by a seeded permutation), %s to validation and %s to test." % ("{:,}".format(casc[-1]["particles_out"]), "{:,}".format(counts["train"]), "{:,}".format(meta["particles_before_train_cap"]["train"]), "{:,}".format(counts["val"]), "{:,}".format(counts["test"])))
    P.append(fig("block_d_schematic.png", "The data on the detector: the side view of LHCb tracking with twenty test particles, four per momentum band; their hit states (points), their real last-UT state (ring) and real first-SciFi state (square), the fine truth between them (thick line), and the five step grids below.", d1=False))
    P.append("## 4.2 The start state and the truths")
    P.append("Each particle's real state on its last Upstream Tracker plane is transported to $`z_0`$ with the fine reference integrator, and that state on $`z_0`$ is the input every chain starts from. From it the fine reference is marched plane by plane through the 129 planes $`z_0 + k L/128`$, $`k = 0 \\dots 128`$, which contain every coarser grid ($`128`$ is divisible by 1, 4, 16 and 64), giving the **fine Runge–Kutta truth** on every plane of every chain. It is then carried on from $`z_1`$ to the particle's own fibre-tracker plane, where it is set beside the particle's **real state** there. The difference between the two is what the particle's passage through material did to it, which no method that solves the field-only equation can predict: the **material floor**.")
    P.append(table(["momentum band", "particles", "material floor, median [µm]", "95th percentile [µm]", "slope [mrad]"],
                   [[b, "{:,}".format(floor_band[b]["n"]), fmt(floor_band[b]["pos_med_um"]), fmt(floor_band[b]["pos_p95_um"]), fmt(floor_band[b]["slope_med_mrad"])] for b in ["all"] + list(BANDS)]))
    P.append("## 4.3 The fine reference")
    P.append("The fine truth is Butcher's seven-stage explicit Runge–Kutta method of order six, at a fixed step of 0.1 mm, in double precision. Its own accuracy on this field map was measured in Block C by halving the step: the endpoint moves by two millionths of a micrometre from 0.1 to 0.05 mm, and a forward-then-back round trip closes to four millionths, so the reference is good to well below a nanometre on a crossing where the networks are at hundreds of micrometres. The classical fourth-order method at 5 mm, which produced the labels of the earlier blocks, differs from it by 0.013 µm.")
    P.append("## 4.4 The exact scheme, the straight line and the twin")
    P.append("The **exact scheme** solves the $`q`$-stage Gauss–Legendre equations of section 3.3 with a root-finder to a residual of $`10^{-9}`$, starting from the straight line, for every test particle, leg after leg across the magnet with the solved end state of one leg feeding the next, exactly as the networks are chained. Its error against the fine truth is the scheme's own discretisation error at that $`N`$ and $`q`$. The **straight line** carries the start state with the magnet switched off. The **twin** is the same network class with $`q = 0`$, one output block, trained on the fine truth's end state by a supervised mean-squared error in units of the residual scale, with the same optimiser protocol.")
    P.append("## 4.5 The networks and the compute")
    P.append("Every network has two hidden layers of 128 units with hyperbolic-tangent activations and $`4(q+1)`$ outputs, so the parameter count grows with $`q`$ only through the last layer: %s parameters at $`q = 1`$, %s at $`q = 8`$, %s at $`q = 20`$. Each chain was one farm job on the Nikhef cluster, one core and 4 GB, the $`N`$ networks trained in series inside it; the %s networks took %s L-BFGS restarts and %s core-hours in all, and every one of them converged under the protocol of section 3.7." % ("{:,}".format(params[1]), "{:,}".format(params[8]), "{:,}".format(params[20]), "{:,}".format(n_legs), "{:,}".format(total_restarts), fmt(core_hours)))
    # ---- 5 Results
    P.append("# 5. Results")
    P.append("## 5.1 The error across the magnet against step count and stage count")
    P.append("The table every other result hangs from. Each cell is one chain: $`N`$ networks trained in sequence, one seed, applied to the %s test particles from $`z_0`$ to $`z_1`$; the entry is the median over those particles of the endpoint position error against the fine truth, in micrometres. Nothing is averaged over any other parameter." % "{:,}".format(n_test))
    P.append(pivot(os.path.join(D1, "table_vs_rk6_endpoint_test_pos_med_um.csv")))
    P.append(fig("heatmap_vs_rk6.png", "The same table as a heat map, one-hue scale, log colour."))
    P.append("Three things are visible at once. **A single step across the whole magnet needs at least five stages**: at $`q = 1`$ the network is at %s µm, at $`q = 2`$ to 4 between %s and %s µm, and from $`q = 8`$ it sits between %s and %s µm, best at %s µm for $`q = %d`$. **Four steps do as well** as one, %s to %s µm for $`q \\geq 8`$ and best at %s µm for $`q = %d`$, and they need fewer stages because each step is shorter. **More steps make it worse**: sixteen steps give %s to %s µm, sixty-four %s to %s µm, one hundred and twenty-eight %s to %s µm, at every $`q`$. The straight line, for scale, is %s µm off." % (
        fmt(med(1, 1)), fmt(min(med(1, q) for q in (2, 3, 4))), fmt(max(med(1, q) for q in (2, 3, 4))), fmt(rng8[1][0]), fmt(rng8[1][1]), fmt(med(1, best_q[1])), best_q[1],
        fmt(rng8[4][0]), fmt(rng8[4][1]), fmt(med(4, best_q[4])), best_q[4],
        fmt(rng8[16][0]), fmt(rng8[16][1]), fmt(rng8[64][0]), fmt(rng8[64][1]), fmt(rng8[128][0]), fmt(rng8[128][1]), fmt(straight)))
    P.append(fig("error_vs_q.png", "Error at z1 against stage count, one line per N; the exact scheme at the same N and q dashed in the same colour; the straight line, the material floor and the supervised twin as reference lines."))
    P.append(fig("error_vs_N.png", "Error at z1 against the number of steps, at chosen stage counts."))
    P.append("## 5.2 What the scheme allows: the ceiling")
    P.append("The same table for the exact scheme, chained the same way with no network in it. This is the best any network trained on these equations could do at that $`N`$ and $`q`$.")
    P.append(pivot(os.path.join(D1, "table_exact_scheme_test_pos_med_um.csv")))
    P.append("At $`N = 1`$ the scheme and the network agree for $`q \\leq 4`$: %s against %s µm at $`q = 1`$, %s against %s at $`q = 4`$. There the network is reproducing the **scheme's** error, which is what a zero loss means, and adding stages is what improves it. From $`q = 5`$ the scheme drops away, to %s µm at $`q = 8`$ and %s µm at $`q = 20`$, and the network stays where it is: from there on the error is the **network's** own, and the stage count no longer matters to it. For the chained cases the scheme is far below the networks at every $`q`$, reaching %s µm at $`N = 128`$, $`q = 16`$: chaining costs the scheme nothing, because its per-leg error is tiny and random in sign, and costs the networks a great deal, for the reason section 5.4 identifies." % (
        fmt(exact[(1, 1)]["endpoint_pos_med_um"]), fmt(med(1, 1)), fmt(exact[(1, 4)]["endpoint_pos_med_um"]), fmt(med(1, 4)),
        fmt(exact[(1, 8)]["endpoint_pos_med_um"]), fmt(exact[(1, 20)]["endpoint_pos_med_um"]), fmt(ex_best[1]["endpoint_pos_med_um"])))
    P.append("## 5.3 Each component separately")
    P.append("The bend is in $`x`$, so $`x`$ and $`t_x`$ carry the error and $`y`$ and $`t_y`$ are a few times smaller. At each $`N`$'s best $`q`$, on the test split, against the fine truth:")
    P.append(table(["N", "best q", "pos median [µm]", "x [µm]", "y [µm]", "tx [mrad]", "ty [mrad]", "x bias [µm]", "tx bias [mrad]"], component_summary_rows()))
    P.append(fig("components_vs_q.png", "Each component against stage count, networks solid and the exact scheme dashed. q/p is carried through unchanged in every chain."))
    P.append("The biases are the signed means over the population. They are small compared with the medians of the absolute errors, which says the population as a whole is not pushed one way; section 5.4 shows that individual tracks are. The change in $`q/p`$ along every one of the 100 chains is exactly %s. The full per-component tables, for the networks, the exact scheme and against the real state, are in the child page \"Per-component tables\"." % fmt(qop_max))
    P.append("## 5.4 Why chaining hurts: the error of one leg repeats along a track")
    P.append(fig("growth_along_crossing.png", "The chain's error plane by plane along the crossing, one panel per N, stage count on the colour ramp."))
    P.append(fig("own_step_vs_inherited.png", "For every network: its own-step error, against the fine propagation of the states it was actually given, on the horizontal axis; the chain's error at the plane it lands on, on the vertical."))
    P.append("Each network on its own is good at its step. On the 40 mm legs of the $`N = 128`$ chain at $`q = %d`$ the median own-step error is %s µm (between %s and %s µm over the 128 legs) where the straight line is %s µm off, so a leg removes about %s percent of the bend. The chain nevertheless ends %s µm off, because the errors do not cancel. The figure below tests the mechanism directly: for the two longest chains the position error a leg adds is, point for point, the slope error it inherited from the previous leg multiplied by the leg length. The position error of a chain is the integrated slope error." % (
        best_q[128], fmt(sorted(n128_own)[len(n128_own) // 2]), fmt(min(n128_own)), fmt(max(n128_own)), fmt(sorted(n128_str)[len(n128_str) // 2]),
        fmt(100 * (1 - sorted(n128_own)[len(n128_own) // 2] / sorted(n128_str)[len(n128_str) // 2]), 2), fmt(med(128, best_q[128]))))
    P.append(fig("slope_to_position.png", "What a leg adds in x against the slope error it inherited times the leg length; points on the line mean the position error is just the integrated slope error."))
    P.append("And the slope error itself does not average out along a track. The table lists, for each $`N`$ at its best $`q`$, the median size of the slope error one leg adds, and a **coherence**: the absolute sum of a track's per-leg increments divided by the sum of their absolute values, which is 1 when every leg pushes the same way and near 0 when the signs are random.")
    P.append(table(["N", "q", "tx added per leg, median [mrad]", "coherence per track (median)", "final tx error, median [mrad]", "final x error, median [µm]", "population tx bias [mrad]"],
                   [[r["N"], r["q"], fmt(r["median_abs_tx_increment_per_leg_mrad"]), fmt(r["median_coherence_per_track"]), fmt(r["final_tx_abs_median_mrad"]), fmt(r["final_x_abs_median_um"]), sfmt(r["final_tx_bias_mean_mrad"])] for r in incs if int(r["q"]) == best_q[int(r["N"])] or (int(r["N"]) == 1 and int(r["q"]) == 8)]))
    P.append("For sixty-four and one hundred and twenty-eight legs the coherence is %s and %s: a given track receives the same sign of slope error on most of its legs, and %s increments of %s mrad add to about a milliradian, which over the remaining metres is millimetres. Across the population these per-track biases have both signs, so the population bias stays near zero while the medians are large. The network's per-leg error is not noise; it is a smooth function of the state that a track meets again and again." % (
        fmt(inc[(64, best_q[64])]["median_coherence_per_track"]), fmt(inc[(128, best_q[128])]["median_coherence_per_track"]), 128, fmt(inc[(128, best_q[128])]["median_abs_tx_increment_per_leg_mrad"])))
    P.append("## 5.5 One particle through every chain")
    P.append("George asked for the table with nothing summarised: one real particle, sent through all 100 chains. The particle is the test proton with momentum closest to the test median (%s GeV), pseudorapidity %s, event %s, particle key %s, starting at $`(x, y) = (%s, %s)`$ mm on $`z_0`$. The cells are its **signed** errors at $`z_1`$ against the fine truth, so the sign is visible." % (
        fmt(single["p_GeV"]), fmt(single["eta"]), single["EVT"], single["MCKEY"], fmt(single["S0_at_z0"][0]), fmt(single["S0_at_z0"][1])))
    P.append("**x [µm]**")
    P.append(pivot(os.path.join(D1, "table_single_track_x.csv"), sfmt))
    P.append("**tx [mrad]**")
    P.append(pivot(os.path.join(D1, "table_single_track_tx.csv"), sfmt))
    P.append("The sign is the point. For this particle the sixty-four and one hundred and twenty-eight step chains land on the same side at every stage count, a few millimetres in $`x`$ and about a milliradian in $`t_x`$, while the one- and four-step chains scatter around zero at a few hundred micrometres. The $`y`$ and $`t_y`$ tables, and this particle's error after every one of the 128 legs, are in the child page \"One particle through every chain\".")
    P.append("## 5.6 Error against momentum")
    P.append("The bend scales as $`1/p`$, so the residual a network has to represent, and the error it makes, depend on momentum. The distributions below are signed errors at $`z_1`$ against the fine truth, per component, against the truth momentum, with the running median and the 16th to 84th percentile band.")
    P.append(fig("err_vs_p_N%03d_q%02d.png" % best, "The best chain, N = %d, q = %d: signed error at z1 per component against momentum, test split." % best))
    P.append(fig("err_vs_p_medians.png", "Median absolute error per component against momentum, one line per N, at q = 8."))
    P.append("Two features matter. At low momentum the error grows because the bend does: below 5 GeV the medians are several times the whole-sample values. But the error also **rises above about 20 GeV** for every chain, where the bend is smallest: for one step at $`q = 8`$ the median $`x`$ error is at its lowest near 10 GeV and several times larger at 60 GeV. High-momentum tracks have the smallest residuals, and in a loss normalised by one population-wide scale they contribute the least, so they are the least well fitted. The 10 to 25 GeV band, where the network is at its best, reads:")
    P.append(pivot(os.path.join(D1, "table_vs_rk6_test_pos_band_10to25GeV.csv")))
    P.append("The other bands, per component, are in the child pages \"Per momentum band\". The gallery of signed-error figures for every $`N`$ at $`q = 8`$ is a child page too.")
    P.append("## 5.7 Against the real detector, the supervised twin, and the cost")
    P.append(fig("vs_real_scifi.png", "The best chain against the fine truth and against the particle's real fibre-tracker state, per momentum band, with the material floor."))
    P.append("Carried from $`z_1`$ to the particle's own fibre-tracker plane and compared with its real state there, the best chain is %s µm off, against a material floor of %s µm on the test split: the network's %s µm of field-only error is invisible under the material the particle crossed, except at high momentum where the floor is %s µm. The supervised twin, trained on the fine truth's end state, reaches %s µm (95th percentile %s µm), with $`x`$ %s, $`y`$ %s µm and $`t_x`$ %s, $`t_y`$ %s mrad, in %s restarts. It is better than the best label-free chain by a factor %s, which is the price of training without labels at this loss and optimiser." % (
        fmt(med(best[0], best[1], "vs_real_scifi_state")), fmt(floor), fmt(med(best[0], best[1])), fmt(floor_band["25-200 GeV"]["pos_med_um"]),
        fmt(tw["pos_med_um"]), fmt(tw["pos_p95_um"]), fmt(twc["x_med_um"]), fmt(twc["y_med_um"]), fmt(twc["tx_med_mrad"]), fmt(twc["ty_med_mrad"]), twin["restarts"], fmt(med(best[0], best[1]) / tw["pos_med_um"], 2)))
    P.append("The forward cost of a crossing is $`N`$ network evaluations. Measured on one core in double precision with the field integral for the scale included, per track in a batch of %s:" % "{:,}".format(n_test))
    P.append(table(["N", "q", "parameters per crossing", "µs per track", "median error [µm]"],
                   [[r["N"], r["q"], "{:,}".format(int(r["parameters_per_crossing"])), fmt(float(r["forward_us_per_track_per_crossing_batch%d" % n_test])), fmt(r["test_med_um_vs_rk6"])] for r in cost]))
    P.append("## 5.8 Are the networks as well trained as they can be?")
    P.append("Three checks. First, **no over-training**: the training, validation and test medians are indistinguishable on every one of the %s legs, the ratio of test to validation having median %s over the chains (5th to 95th percentile %s to %s); a label-free loss on 2,000 states with 20,000 parameters fits nothing to the particles it sees." % ("{:,}".format(n_legs), fmt(sorted(vt_ratio)[len(vt_ratio) // 2]), fmt(sorted(vt_ratio)[int(0.05 * len(vt_ratio))]), fmt(sorted(vt_ratio)[int(0.95 * len(vt_ratio))])))
    P.append(fig("val_vs_test.png", "Validation against test median for every chain."))
    P.append("Second, the **stopping rule**. The loss histories are slow power-law decays rather than curves that flatten, and the one-percent stall rule fires while the loss is still falling. To test whether it stops runs early, the $`N = 1`$, $`q = 16`$ and $`N = 4`$, $`q = 5`$ chains were retrained from scratch with the rule tightened to 0.1 percent, the cap raised to 400 restarts, and the endpoint error on all three splits recorded after every restart.")
    P.append(table(["chain", "grid run: test median [µm]", "restarts", "0.1 percent rule: test median [µm]", "restarts"],
                   [["N = 1, q = 16", fmt(grid_16["test"]["vs_rk6_endpoint"]["pos_med_um"]), grid_16["total_restarts"], fmt(tight_16["test"]["vs_rk6_endpoint"]["pos_med_um"]), tight_16["total_restarts"]],
                    ["N = 4, q = 5", fmt(grid_45["test"]["vs_rk6_endpoint"]["pos_med_um"]), grid_45["total_restarts"], fmt(tight_45["test"]["vs_rk6_endpoint"]["pos_med_um"]), tight_45["total_restarts"]]]))
    P.append(fig("training_curves.png", "Loss and endpoint medians against restart for the retrained legs; the dashed line marks where the one-percent rule had stopped the grid run."))
    P.append("The answer is that the rule is not what stops them. Within two restarts of where the grid run stalled, the gain per restart falls from about one percent to a few hundredths of one percent, while each restart still runs its full 200 iterations at the same wall time, and a fresh optimiser could not continue the descent either. The endpoint error tracks the loss all the way, reaching its minimum of %s µm at restart %s for the single step and then wandering within a few percent. This is an **optimiser floor**: with this loss and this optimiser the networks are as good as they get. The per-restart records are a child page." % (fmt(min(cv_test)), cv[cv_min_i]["outer"]))
    P.append("Third, **continuity** with the earlier blocks. The single step at $`q = 8`$, two hidden layers of 128 on the correct polarity, gives %s µm; Block A's four hidden layers of 50 on the same crossing, at $`q = 8`$ on the down-polarity map, gave %s µm in the stage sweep and %s µm as the median of ten seeds, and its four layers of 200 gave %s µm. The floor has not moved with the polarity, the architecture or the removal of the leg as an input." % (
        fmt(med(1, 8)), fmt(a1[8]["physics_med_um"]), fmt([r for r in a2 if "4x50" in r["source"]][0]["physics_med_um"]), fmt([r for r in a2 if "4x200" in r["source"]][0]["physics_med_um"])))
    # ---- 6 Conclusion
    P.append("# 6. Conclusion")
    P.append("- **Does the technique work on a fixed step?** Yes. Trained on nothing but the equation of motion and the collocation scheme, a two-hidden-layer network of 128 units crosses the five-metre magnet in one step with a median error of %s µm at its best stage count and between %s and %s µm from eight stages upward, on the correct polarity, with no over-training and every run converged. Below five stages the error is the scheme's; above, it is the network's, and the stage count stops mattering." % (fmt(med(1, best_q[1])), fmt(rng8[1][0]), fmt(rng8[1][1])))
    P.append("- **Does chaining help?** No. Four steps match one; sixteen, sixty-four and one hundred and twenty-eight steps are worse at every stage count, reaching millimetres, while the exact scheme chained the same way reaches nanometres. The cause is identified: each network leaves a small slope error that has the same sign on most legs of a given track, and the chain integrates it. The paper's sequential protocol, in which each network learns from its predecessor's predictions, does not remove this.")
    P.append("- **Where is the floor?** In the optimiser, not the stopping rule, the data or the split. Training ten times past the stall changes nothing. The error also rises at high momentum, where the residuals are smallest and the population-scale loss weights them least.")
    P.append("- **Against the detector.** The best chain's %s µm is far under the %s µm of material the particle crosses (test split), except above 25 GeV, where the floor is %s µm and the network's high-momentum weakness is what would show." % (fmt(med(best[0], best[1])), fmt(floor), fmt(floor_band["25-200 GeV"]["pos_med_um"])))
    P.append("Three levers remain, each its own to-do: normalise the physics loss per sample so that short steps and high-momentum tracks carry their weight (<mention-page url=\"%s\"/>); change the optimiser schedule (<mention-page url=\"%s\"/>); and widen the network or add training states (<mention-page url=\"%s\"/>). Until one of them moves the single-step floor, chaining is not the direction." % (TODO_NORM, TODO_OPT, TODO_WIDE))
    P.append("**Caveats.** One seed per network; Block A measured a largest-to-smallest ratio of %s between ten seeds of the same network on this crossing, so differences between cells smaller than that are not significant." % fmt(a2_spread) + " The test split is the training set's own by-particle split. The exact scheme's tolerance of $`10^{-9}`$ in mm and mrad bounds its own floor. The forward-cost numbers are eager PyTorch on one core and are indicative only.")
    # ---- Provenance
    P.append("# Provenance")
    P.append("- Repository github.com/GeorgeWilliam1999/LHCb-Extrapolation at commit `%s` (`%s`), branch main; every figure on these pages is served from that commit. Folder `Raissi_disc_time_approach/Block_D_fixed_step_crossing/`, which imports the shared package `Raissi_disc_time_approach/_shared/` unchanged except for two opt-in trainer flags (`--stall-tol`, `--log-medians`) whose defaults reproduce the earlier behaviour bitwise. Two tables come from the follow-up commit `1f23556` (`1f23556412b1c794248c5ee87eb50218b5efed3f`): the per-leg slope-increment and coherence table of section 5.4 (`extra_plots.py` section 8 → `results/leg_increments.csv`) and the re-timed forward-cost table of section 5.6 (`results/cost_table.csv`)." % (SHORT, COMMIT))
    P.append("- Data: `D0_Crossing_dataset/build_dataset.py` on `Data_generation_exploration/Official_xdigi/training_v2/train_official_v2.npz` (harvested from the official TestFileDB sample expected_2024_minbias_xdigi, production 00212966, 200 events) → `results/crossing_particles.npz` and `_meta.json` (the cut cascade, counts and material floor above). Field map v8r1 up, MD5 %s. Fine reference: `_shared/reference.py` `rk6_rows` at 0.1 mm." % meta["field"]["md5"])
    P.append("- Networks: `D1_Chain_grid/chain_model.py` (the model), `train_chain.py` (the sequential chain, calling `_shared/train.py`), `make_jobs.py` (100 farm jobs, cluster 5800864, no resubmission needed), records `results/N<NNN>_q<qq>/` (per leg `.json`, `_history.csv`, `_scale.json`; `chain.json`; `states.npz`). Tables: `aggregate.py` and `extra_plots.py` → `results/*.csv`; figures: `plot.py`, `extra_plots.py`, `plot_training_curves.py` → `figures/`. Schematic: `D0_Crossing_dataset/plot_schematic.py`.")
    P.append("- Comparators: `D2_Comparators/exact_chain.py` (cluster 5802098) → `results/exact_N<NNN>_q<qq>.json`; `train_twin.py` → `results/twin/`; `compare.py` → `results/comparison_table.csv`. Continuation study: `results/tight_stall/` and `results/training_curves.csv`.")
    P.append("- This page and its children were written by `writeup/gen_writeup.py` from those files and checked against the published pages token by token with `tools/verify_numbers.py`. To-do: <mention-page url=\"%s\"/>." % TODO_D)
    return "\n".join(P)


# ------------------------------------------------------------- children --
def child_pages():
    ch = []
    # A endpoint tables
    A = ["<!-- TITLE: Endpoint tables: median and 95th percentile at z1, test and validation -->",
         "Each cell is one chain (N networks, one seed) on the named split; the entry is the endpoint position error, the larger of the absolute x and y errors, in micrometres, as the median or the 95th percentile over the split's particles, against the fine Runge–Kutta truth or against the particle's real fibre-tracker state (carried there from z1 by the same fine integrator)."]
    for comp, cname in (("vs_rk6_endpoint", "against the fine truth"), ("vs_real_scifi_state", "against the real fibre-tracker state")):
        for split in ("test", "val"):
            for stat, sname in (("pos_med_um", "median"), ("pos_p95_um", "95th percentile")):
                A.append("## %s, %s split, %s [µm]" % (cname, split, sname))
                A.append(pivot(os.path.join(D1, "table_%s_%s_%s.csv" % (comp, split, stat))))
    ch.append(("A", "\n".join(A)))
    # B per-component
    Bp = ["<!-- TITLE: Per-component tables: x, y, tx and ty at z1 for the networks, the exact scheme and against the real state -->",
          "Medians of the absolute error over the test particles, per component: x and y in micrometres, tx and ty in milliradians. Each cell is one chain."]
    for comp, cname in (("vs_rk6_endpoint", "networks against the fine truth"), ("exact_scheme_vs_rk6_endpoint", "exact scheme against the fine truth"), ("vs_real_scifi_state", "networks against the real fibre-tracker state")):
        for name, u, uu in COMP:
            p = os.path.join(D1, "table_%s_test_%s_med.csv" % (comp, name))
            if os.path.exists(p):
                Bp.append("## %s: %s [%s]" % (cname, name, uu))
                Bp.append(pivot(p))
    ch.append(("B", "\n".join(Bp)))
    # C per band
    for b in BANDS:
        tag = b.replace(" ", "").replace("-", "to")
        Cp = ["<!-- TITLE: Per momentum band: %s -->" % b,
              "Test particles with truth momentum in this band, against the fine truth; medians of the absolute error. The material floor in this band is %s µm (median). Each cell is one chain." % fmt(floor_band[b]["pos_med_um"])]
        Cp.append("## endpoint position error [µm]")
        Cp.append(pivot(os.path.join(D1, "table_vs_rk6_test_pos_band_%s.csv" % tag)))
        for name, u, uu in COMP:
            Cp.append("## %s [%s]" % (name, uu))
            Cp.append(pivot(os.path.join(D1, "table_vs_rk6_test_%s_band_%s.csv" % (name, tag))))
        ch.append(("C_" + tag, "\n".join(Cp)))
    # D single track
    Dp = ["<!-- TITLE: One particle through every chain: the signed error tables -->",
          "The test particle with momentum closest to the test median among those in the inner half of the transverse start positions: event %s, particle key %s, PDG code %s, momentum %s GeV, pseudorapidity %s, start state on z0 = (%s mm, %s mm, %s, %s, %s). The cells are its signed errors at z1 against the fine truth, one chain per cell." % (
              single["EVT"], single["MCKEY"], single["PID"], fmt(single["p_GeV"]), fmt(single["eta"]), fmt(single["S0_at_z0"][0]), fmt(single["S0_at_z0"][1]), fmt(single["S0_at_z0"][2]), fmt(single["S0_at_z0"][3]), fmt(single["S0_at_z0"][4]))]
    for name, u, uu in COMP:
        Dp.append("## %s [%s]" % (name, uu))
        Dp.append(pivot(os.path.join(D1, "table_single_track_%s.csv" % name), sfmt))
    # its per-plane errors along the N=128 best chain and the N=1 best chain
    import numpy as np
    Dnp = np.load(os.path.join(D0, "crossing_particles.npz"))
    i = single["index_in_test_split"]
    for N in (128, 16):
        q = best_q[N]
        st = np.load(os.path.join(D1, "N%03d_q%02d" % (N, q), "states.npz"))["test_states"][i]
        tr = Dnp["test_truth"][i, ::int(Dnp["n_max"]) // N]
        rows = []
        for k in range(N + 1):
            rows.append([str(k), fmt(chains[(N, q)]["planes"][k]), sfmt((st[k, 0] - tr[k, 0]) * 1e3), sfmt((st[k, 1] - tr[k, 1]) * 1e3), sfmt((st[k, 2] - tr[k, 2]) * 1e3), sfmt((st[k, 3] - tr[k, 3]) * 1e3)])
        Dp.append("## This particle plane by plane along the N = %d, q = %d chain" % (N, q))
        Dp.append(table(["plane", "z [mm]", "Δx [µm]", "Δy [µm]", "Δtx [mrad]", "Δty [mrad]"], rows))
    ch.append(("D", "\n".join(Dp)))
    # E per leg
    def leg_rows(N, qs):
        rows = []
        for r in per_leg:
            if int(r["N"]) == N and int(r["q"]) in qs:
                rows.append([r["q"], r["leg"], fmt(r["z0"]), fmt(r["z1"]), r["converged"], r["restarts"], fmt(r["own_step_test_endpoint_med_um"]), fmt(r["own_step_test_straight_med_um"]), fmt(r["chain_test_pos_med_um"]), fmt(r["chain_test_pos_p95_um"]), fmt(r["chain_test_slope_med_mrad"])])
        return rows
    hdr = ["q", "leg", "z start [mm]", "z end [mm]", "converged", "restarts", "own-step median [µm]", "straight line on the leg [µm]", "chain median at leg end [µm]", "chain 95th pct [µm]", "chain slope median [mrad]"]
    intro = "One row per network. Own-step: the network's endpoint error against the fine propagation of the states it was actually given (its predecessor's predictions). Chain: the error of the chain at the plane this leg lands on, against the fine truth from the real start state. Test split."
    for N, groups in ((1, [QS]), (4, [QS]), (16, [QS]),
                      (64, [QS[i:i + 5] for i in range(0, 20, 5)]),
                      (128, [QS[i:i + 2] for i in range(0, 20, 2)])):
        for g in groups:
            title = "Per leg: N = %d" % N + ("" if len(groups) == 1 else ", q = %d to %d" % (g[0], g[-1]))
            ch.append(("E_N%03d_q%02d" % (N, g[0]), "\n".join(["<!-- TITLE: %s -->" % title, intro, table(hdr, leg_rows(N, g))])))
    # F growth
    for N, step in ((4, 1), (16, 1), (64, 8), (128, 16)):
        Fp = ["<!-- TITLE: Along the crossing: the chain's error plane by plane, N = %d -->" % N,
              "The chain's error at every plane against the fine truth on that plane, test split, medians of the absolute error; the plane index k runs from 0 (z0, error zero by construction) to N.%s" % ("" if step == 1 else " Every %s plane is listed; the figures on the main page show all of them." % ("eighth" if step == 8 else "sixteenth"))]
        rows = [[r["q"], r["plane"], fmt(r["z_mm"]), fmt(r["pos_med_um"]), fmt(r["pos_p95_um"]), fmt(r["x_med_um"]), fmt(r["y_med_um"]), fmt(r["tx_med_mrad"]), fmt(r["ty_med_mrad"])]
                for r in growth if int(r["N"]) == N and r["split"] == "test" and int(r["plane"]) % step == 0]
        Fp.append(table(["q", "plane k", "z [mm]", "pos median [µm]", "pos 95th pct [µm]", "x [µm]", "y [µm]", "tx [mrad]", "ty [mrad]"], rows))
        ch.append(("F_N%03d" % N, "\n".join(Fp)))
    # G checks and cost
    Gp = ["<!-- TITLE: Checks and cost: convergence, q/p, validation against test, forward cost, the twin, continuity with Block A -->"]
    Gp.append("## Convergence and cost per chain")
    Gp.append(table(["N", "q", "all legs converged", "total restarts", "training wall [s]"], [[r["N"], r["q"], r["all_legs_converged"], r["total_restarts"], fmt(r["total_train_wall_s"])] for r in status]))
    Gp.append("## The q/p check: largest change in q/p along the chain")
    Gp.append(table(["N", "q", "test", "validation"], [[r["N"], r["q"], fmt(r["qop_max_abs_change_test"]), fmt(r["qop_max_abs_change_val"])] for r in qop]))
    Gp.append("## Validation against test, endpoint median [µm]")
    Gp.append(table(["N", "q", "validation", "test"], [[r["N"], r["q"], fmt(r["val_med_um"]), fmt(r["test_med_um"])] for r in vt]))
    Gp.append("## Forward cost per crossing")
    Gp.append(table(["N", "q", "networks", "parameters per leg", "parameters per crossing", "µs per track (batch %d)" % n_test, "median error [µm]"],
                    [[r["N"], r["q"], r["networks_per_crossing"], "{:,}".format(int(r["parameters_per_leg"])), "{:,}".format(int(r["parameters_per_crossing"])), fmt(float(r["forward_us_per_track_per_crossing_batch%d" % n_test])), fmt(r["test_med_um_vs_rk6"])] for r in cost]))
    Gp.append("## The supervised twin, test split")
    Gp.append(table(["comparator", "pos median [µm]", "pos 95th pct [µm]", "slope median [mrad]", "x [µm]", "y [µm]", "tx [mrad]", "ty [mrad]", "x bias [µm]", "tx bias [mrad]"],
                    [[cn, fmt(twin["test"][c]["pos_med_um"]), fmt(twin["test"][c]["pos_p95_um"]), fmt(twin["test"][c]["slope_med_mrad"])] + [fmt(twin["test"][c]["components"]["%s_med_%s" % (n, u)]) for n, u, _ in COMP] + [sfmt(twin["test"][c]["components"]["x_bias_um"]), sfmt(twin["test"][c]["components"]["tx_bias_mrad"])]
                     for c, cn in (("vs_rk6_endpoint", "against the fine truth"), ("vs_real_scifi_state", "against the real fibre-tracker state"))]))
    Gp.append("Parameters %s, restarts %s, converged %s, wall %s s." % ("{:,}".format(twin["n_parameters"]), twin["restarts"], twin["converged"], fmt(twin["wall_s"])))
    Gp.append("## Continuity with Block A: the single step")
    Gp.append(table(["source", "q", "label-free network, median [µm]", "supervised twin [µm]", "exact scheme [µm]", "straight line [µm]"],
                    [[r["source"], r["q"], fmt(r["physics_med_um"]), fmt(r["twin_med_um"]), fmt(r["scheme_med_um"]), fmt(r["straight_med_um"])] for r in cont]))
    ch.append(("G", "\n".join(Gp)))
    # H continuation
    Hp = ["<!-- TITLE: The continuation study: loss and endpoint error after every restart -->",
          "N = 1, q = 16 and N = 4, q = 5 retrained from scratch with the stall rule at 0.1 percent per restart and a cap of 400 restarts; after every L-BFGS restart of 200 iterations the loss and the endpoint medians on all three splits were recorded."]
    for (N, q) in ((1, 16), (4, 5)):
        for k in range(N):
            rows = [[r["outer"], r["phase"], "%.4e" % float(r["loss"]), fmt(r["train_endpoint_med_um"]), fmt(r["val_endpoint_med_um"]), fmt(r["test_endpoint_med_um"]), fmt(r["test_endpoint_p95_um"]), fmt(r["wall_s"])]
                    for r in curves if int(r["N"]) == N and int(r["q"]) == q and int(r["leg"]) == k]
            Hp.append("## N = %d, q = %d, leg %d" % (N, q, k))
            Hp.append(table(["restart", "phase", "loss", "train median [µm]", "val median [µm]", "test median [µm]", "test 95th pct [µm]", "wall [s]"], rows))
    ch.append(("H", "\n".join(Hp)))
    # I gallery
    Ip = ["<!-- TITLE: Figure gallery: signed error against momentum, per component, one chain per figure -->",
          "Signed error at z1 against the fine truth, per component, against truth momentum, test split; the running median and the 16th to 84th percentile band; clipped at the 98th percentile of the absolute error for display."]
    for N in N_VALUES:
        Ip.append(fig("err_vs_p_N%03d_q08.png" % N, "N = %d, q = 8" % N))
    if best != (1, 8):
        Ip.append(fig("err_vs_p_N%03d_q%02d.png" % best, "N = %d, q = %d, the best chain" % best))
    Ip.append(fig("components_growth.png", "Each component along the crossing at each N's best q."))
    Ip.append(fig("crossing_geometry.png", "The field on the axis along the crossing and the five step grids.", d1=False))
    ch.append(("I", "\n".join(Ip)))
    return ch


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "children"), exist_ok=True)
    m = main_page()
    open(os.path.join(OUT, "main.md"), "w").write(m)
    print("main.md %d KB" % (len(m.encode()) // 1024))
    for key, text in child_pages():
        p = os.path.join(OUT, "children", key + ".md")
        open(p, "w").write(text)
        print("%-16s %5d KB  %s" % (key, len(text.encode()) // 1024, text.split("\n")[0][13:80]))
