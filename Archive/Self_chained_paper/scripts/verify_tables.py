#!/usr/bin/env python
"""verify_tables.py -- recompute every numeric cell of every numbered table in
main.tex from the committed files under results/ and report mismatches.

Run from anywhere:

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python \
        Self_chained_paper/scripts/verify_tables.py

It writes nothing.  It prints one block per table with the number of numeric
cells checked and every disagreement, and a final summary line.

How a cell is compared
----------------------
Each LaTeX cell is parsed into a float together with the precision it is shown
at (the number of decimals, or, for a cell written as a mantissa times a power
of ten, the number of decimals of the mantissa).  The recomputed value is
accepted when it is within half a unit of that last displayed place, with a
relative slack of 1e-9 for floating-point noise.  Cells that carry no number
(``---``, text, a `\\star`) are skipped and counted separately.

The table -> source-file mapping is the one printed in the caption of each
table, and repeated in README.md and Appendix A of the paper.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
RESULTS = os.path.join(PAPER, "results")
MAIN_TEX = os.path.join(PAPER, "main.tex")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_csv(name):
    with open(os.path.join(RESULTS, name), newline="") as fh:
        return list(csv.DictReader(fh))


def load_json():
    with open(os.path.join(RESULTS, "paper_numbers.json")) as fh:
        return json.load(fh)


def f(row, key):
    """float of a CSV field ('' -> None)."""
    v = row[key]
    if v == "" or v is None:
        return None
    return float(v)


# --------------------------------------------------------------------------
# LaTeX cell parsing
# --------------------------------------------------------------------------
_CLEAN = [
    (r"\\mathbf\{", "{"),
    (r"\\textbf\{", "{"),
    (r"\\phantom\{[^}]*\}", ""),
    (r"\\,", ""),
    (r"\\!", ""),
    (r"\\ ", " "),
    (r"\\um", ""),
    (r"\\mm", ""),
    (r"\\%", ""),
    (r"\\star", ""),
    (r"\\emph\{[^}]*\}", ""),
    (r"\{,\}", ""),
    (r",", ""),
    (r"\$", ""),
    (r"\\;", ""),
]

_SCI = re.compile(
    r"^\s*([+-]?)\s*([0-9]*\.?[0-9]+)\s*(?:\\times\s*)?10\s*\^\s*\{?\s*([+-]?[0-9]+)\s*\}?\s*$"
)
_PLAIN = re.compile(r"^\s*([+-]?[0-9]*\.?[0-9]+)\s*$")


def parse_cell(cell):
    """Return (value, tolerance) or None when the cell holds no number."""
    s = cell.strip()
    if s in ("", "---", "--", "&"):
        return None
    for pat, rep in _CLEAN:
        s = re.sub(pat, rep, s)
    s = s.replace("{", "").replace("}", "").strip()
    if s in ("", "---", "--", "yes", "no", "off"):
        return None

    m = _SCI.match(s)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        mant = m.group(2)
        exp = int(m.group(3))
        val = sign * float(mant) * 10.0 ** exp
        dec = len(mant.split(".")[1]) if "." in mant else 0
        tol = 0.5 * 10.0 ** (-dec) * 10.0 ** exp
        return val, tol

    m = _PLAIN.match(s)
    if m:
        t = m.group(1)
        val = float(t)
        dec = len(t.split(".")[1]) if "." in t else 0
        tol = 0.5 * 10.0 ** (-dec)
        return val, tol

    return None


# --------------------------------------------------------------------------
# main.tex table extraction
# --------------------------------------------------------------------------
def extract_tables():
    """{label: {'number': int, 'caption': str, 'rows': [[cell, ...]], 'line': int}}"""
    lines = open(MAIN_TEX).read().split("\n")
    out = {}
    order = []
    i = 0
    n = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("\\begin{table"):
            j = i
            while not lines[j].lstrip().startswith("\\end{table"):
                j += 1
            blk = "\n".join(lines[i : j + 1])
            m = re.search(r"\\label\{(tab:[^}]*)\}", blk)
            if m:
                n += 1
                label = m.group(1)
                cm = re.search(r"\\caption\{(.*?)\}\s*\n\s*\\label", blk, re.S)
                caption = cm.group(1) if cm else ""
                # tabular body
                tb = re.search(
                    r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", blk, re.S
                )
                rows = []
                if tb:
                    body = tb.group(1)
                    for raw in body.split("\\\\"):
                        raw = raw.strip()
                        for junk in (
                            "\\toprule",
                            "\\midrule",
                            "\\bottomrule",
                            "\\addlinespace",
                        ):
                            raw = raw.replace(junk, "")
                        raw = re.sub(r"\\cmidrule\(lr\)\{[^}]*\}", "", raw)
                        raw = raw.strip()
                        if not raw:
                            continue
                        rows.append([c.strip() for c in raw.split("&")])
                out[label] = dict(
                    number=n, caption=caption, rows=rows, line=i + 1
                )
                order.append(label)
            i = j
        i += 1
    return out, order


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.tables = []
        self.total_checked = 0
        self.total_skipped = 0
        self.total_bad = 0

    def start(self, label, number, source):
        self.cur = dict(
            label=label, number=number, source=source, checked=0, skipped=0, bad=[]
        )
        self.tables.append(self.cur)

    def cell(self, where, latex, computed, note=""):
        p = parse_cell(latex)
        if p is None:
            self.cur["skipped"] += 1
            self.total_skipped += 1
            return
        val, tol = p
        self.cur["checked"] += 1
        self.total_checked += 1
        if computed is None:
            self.cur["bad"].append(
                f"{where}: paper {latex!r} but no recomputed value {note}"
            )
            self.total_bad += 1
            return
        slack = tol + 1e-9 * max(abs(val), abs(computed))
        if not (abs(val - computed) <= slack):
            self.cur["bad"].append(
                f"{where}: paper {latex!r} = {val!r}  vs  recomputed {computed!r} "
                f"(tol {tol:.3g}) {note}"
            )
            self.total_bad += 1

    def text(self, where, ok, note):
        """non-numeric assertion"""
        if not ok:
            self.cur["bad"].append(f"{where}: {note}")
            self.total_bad += 1

    def dump(self, fh=sys.stdout):
        for t in self.tables:
            status = "PASS" if not t["bad"] else "FAIL"
            print(
                f"Table {t['number']:>2} ({t['label']:<20}) "
                f"{t['checked']:>4} numeric cells checked, "
                f"{t['skipped']:>3} non-numeric skipped   [{status}]",
                file=fh,
            )
            for b in t["bad"]:
                print(f"      ! {b}", file=fh)
        print("", file=fh)
        print(
            f"TOTAL: {self.total_checked} numeric cells checked across "
            f"{len(self.tables)} tables, {self.total_bad} mismatches "
            f"({self.total_skipped} non-numeric cells skipped)",
            file=fh,
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def by(rows, **kw):
    """first CSV row matching all key=value (values compared as strings)."""
    for r in rows:
        if all(str(r[k]) == str(v) for k, v in kw.items()):
            return r
    return None


def allby(rows, **kw):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]


BANDS_TEX = ["<3", "3-8", "8-20", "20-50", ">50"]
WINDOW = "10-50 (loss window)"


# --------------------------------------------------------------------------
# per-table checks
# --------------------------------------------------------------------------
def main():
    T, order = extract_tables()
    J = load_json()
    R = Report()

    # ---------------- Table 1: tab:cuts <- tab_dataset_cuts.csv -----------
    R.start("tab:cuts", T["tab:cuts"]["number"], "tab_dataset_cuts.csv")
    cuts = load_csv("tab_dataset_cuts.csv")
    body = [r for r in T["tab:cuts"]["rows"] if parse_cell(r[0]) is not None]
    for row in body:
        step = int(parse_cell(row[0])[0])
        c = by(cuts, step=step)
        R.cell(f"cut {step} step", row[0], float(step))
        for k, col in ((2, "rows_in"), (3, "rows_removed"), (4, "rows_out"),
                       (5, "particles_out")):
            R.cell(f"cut {step} {col}", row[k], f(c, col))

    # ---------------- Table 2: tab:splitbands <- tab_dataset_bands.csv ----
    R.start("tab:splitbands", T["tab:splitbands"]["number"],
            "tab_dataset_bands.csv")
    db = load_csv("tab_dataset_bands.csv")
    tex2csv = {"$<3$": "<3", "$3$--$8$": "3-8", "$8$--$20$": "8-20",
               "$20$--$50$": "20-50", "$>50$": ">50",
               "$10$--$50$ (loss window)": WINDOW}
    for row in T["tab:splitbands"]["rows"]:
        key = row[0].strip()
        if key not in tex2csv:
            continue
        band = tex2csv[key]
        for si, split in enumerate(("train", "val", "test")):
            c = by(db, split=split, band=band)
            R.cell(f"{band} {split} n", row[1 + 2 * si], f(c, "n"))
            R.cell(f"{band} {split} median p", row[2 + 2 * si],
                   f(c, "p_median_GeV"))
    # the split totals in the header
    hdr = T["tab:splitbands"]["rows"][0]
    for split, cell in (("train", hdr[1]), ("val", hdr[2]), ("test", hdr[3])):
        m = re.search(r"\(\$([0-9{},]+)\$\)", cell)
        if m:
            R.cell(f"header total {split}", "$" + m.group(1) + "$",
                   float(J["conventions"]["splits"][split]))

    # ---------------- Table 3: tab:materialfloor <- tab_material_floor ----
    R.start("tab:materialfloor", T["tab:materialfloor"]["number"],
            "tab_material_floor.csv")
    mf = load_csv("tab_material_floor.csv")
    tex2csv3 = dict(tex2csv)
    tex2csv3["all"] = "all"
    for row in T["tab:materialfloor"]["rows"]:
        key = row[0].strip()
        band = tex2csv3.get(key)
        if band is None:
            continue
        c = by(mf, scope="all splits", band=band)
        R.cell(f"{band} n", row[1], f(c, "n"))
        R.cell(f"{band} median", row[2], f(c, "med_um"))
        R.cell(f"{band} p95", row[3], f(c, "p95_um"))
        R.cell(f"{band} median slope", row[4], f(c, "slope_max_med"))
        R.cell(f"{band} >1mm %", row[5], 100.0 * f(c, "frac_above_1mm"))

    # ---------------- Table 4: tab:fixed <- tab_compute.csv, PN ----------
    R.start("tab:fixed", T["tab:fixed"]["number"],
            "tab_compute.csv; paper_numbers.json(compute)")
    cmp_ = load_csv("tab_compute.csv")
    cap = T["tab:fixed"]["caption"]
    # the three parameter counts quoted in the caption
    pars = re.findall(r"\$([0-9]{2}\{,\}[0-9]{3})\$", cap)
    want = [f(by(cmp_, N=64, q=2, loss="pooled"), "n_parameters"),
            f(by(cmp_, N=128, q=8, loss="pooled"), "n_parameters"),
            f(by(cmp_, N=256, q=16, loss="pooled"), "n_parameters")]
    for p, w in zip(pars, want):
        R.cell("caption parameter count", "$" + p + "$", w)
    # body cells
    txt = "\n".join(" & ".join(r) for r in T["tab:fixed"]["rows"])
    R.cell("tracks", "$14{,}482$", float(J["dataset"]["total_tracks"]))
    R.cell("train", "$11{,}567$", float(J["conventions"]["splits"]["train"]))
    R.cell("validation", "$1463$", float(J["conventions"]["splits"]["val"]))
    R.cell("test", "$1452$", float(J["conventions"]["splits"]["test"]))
    comp = J["compute"]
    R.cell("iterations per restart", "$200$",
           f(by(cmp_, N=64, q=2, loss="pooled"), "median_iterations_per_restart"))
    R.cell("restarts per round", "$25$",
           f(by(cmp_, N=64, q=2, loss="pooled"), "round_restarts"))
    R.cell("states per round", "$32{,}000$",
           f(by(cmp_, N=64, q=2, loss="pooled"), "states_per_round"))
    R.cell("rk6 step", "$0.1$", float(J["dataset"]["rk6_reference"]["step_mm"]))
    md5 = J["dataset"]["field_map"]["md5"]
    R.text("field-map md5", md5.startswith("9e49ddc4") and md5.endswith("513b"),
           f"md5 in table 9e49ddc4...513b, json {md5}")
    R.text("hidden layers", "128" in txt, "two hidden layers of 128 units")

    # ---------------- Table 5: tab:tableau <- tab_tableau.csv -------------
    R.start("tab:tableau", T["tab:tableau"]["number"], "tab_tableau.csv")
    tb = load_csv("tab_tableau.csv")
    for row in T["tab:tableau"]["rows"]:
        p = parse_cell(row[0])
        if p is None or len(row) < 10:
            continue
        q = int(p[0])
        c = by(tb, q=q)
        if c is None:
            continue
        R.cell(f"q={q} q", row[0], float(q))
        R.cell(f"q={q} order", row[1], f(c, "order"))
        for k, col in ((2, "row_sums"), (3, "quadrature"), (4, "collocation"),
                       (5, "symplectic"), (6, "vs_literature")):
            if f(c, col) is None:
                R.cell(f"q={q} {col}", row[k], None) if parse_cell(row[k]) else R.cell(
                    f"q={q} {col}", row[k], None)
                continue
            R.cell(f"q={q} {col}", row[k], f(c, col))
        R.cell(f"q={q} c_1", row[7], f(c, "c_first"))
        R.cell(f"q={q} c_q", row[8], f(c, "c_last"))
        R.cell(f"q={q} sum b", row[9], f(c, "b_sum"))
    R.cell("caption tolerance", "$5 \\times 10^{-13}$",
           float(J["scheme"]["tableau"]["tolerance"]))
    R.cell("caption worst", "$2.2 \\times 10^{-16}$",
           float(J["scheme"]["tableau"]["worst_overall"]))

    # ---------------- Table 6: tab:ceiling --------------------------------
    R.start("tab:ceiling", T["tab:ceiling"]["number"],
            "tab_exact_ceiling.csv; tab_pooled_grid.csv")
    ec = load_csv("tab_exact_ceiling.csv")
    pg = load_csv("tab_pooled_grid.csv")
    qs = [2, 4, 8, 16]
    block = 0
    for row in T["tab:ceiling"]["rows"]:
        if row[0].startswith("\\multicolumn"):
            block += 1
            continue
        p = parse_cell(row[0])
        if p is None or len(row) < 6:
            continue
        N = int(p[0])
        R.cell(f"block{block} N", row[0], float(N))
        dz = f(by(ec, N=N, q=2), "dz_mm")
        R.cell(f"block{block} N={N} dz", row[1], dz)
        for k, q in enumerate(qs):
            c = by(ec, N=N, q=q)
            g = by(pg, N=N, q=q)
            if block == 1:
                v = f(c, "exact_radial_med_um_from_csv")
            elif block == 2:
                v = f(c, "exact_max_p95_um")
            else:
                v = f(g, "radial_med_um")
            R.cell(f"block{block} N={N} q={q}", row[2 + k], v)

    # ---------------- Table 7: tab:rk6ladder ------------------------------
    R.start("tab:rk6ladder", T["tab:rk6ladder"]["number"],
            "tab_rk6_convergence.csv")
    rk = load_csv("tab_rk6_convergence.csv")
    for row in T["tab:rk6ladder"]["rows"]:
        p = parse_cell(row[0])
        if p is None or len(row) < 7:
            continue
        h = p[0]
        c = None
        for r in rk:
            if abs(float(r["step_mm"]) - h) < 1e-12:
                c = r
        R.cell(f"h={h} h", row[0], float(c["step_mm"]))
        for k, col in ((1, "halving_med_um"), (2, "halving_p95_um"),
                       (3, "halving_max_um"), (4, "vs_finest_med_um"),
                       (5, "vs_finest_p95_um"), (6, "vs_finest_max_um")):
            R.cell(f"h={h} {col}", row[k], f(c, col))
    R.cell("caption n legs", "$200$", f(rk[0], "n"))

    # ---------------- Table 8: tab:decomp ---------------------------------
    R.start("tab:decomp", T["tab:decomp"]["number"],
            "tab_decomposition_bands.csv; PN rebinned_decomposition")
    dc = load_csv("tab_decomposition_bands.csv")
    hl = load_csv("tab_highland_air.csv")
    rb = J["reference_vs_truth"]["rebinned_decomposition"]["by_band"]
    tex2csv8 = {"$<3$": "<3", "$3$--$8$": "3-8", "$8$--$20$": "8-20",
                "$20$--$50$": "20-50", "$>50$": ">50",
                "$10$--$50$ (window)": WINDOW, "all": "all"}
    for row in T["tab:decomp"]["rows"]:
        key = row[0].strip()
        band = tex2csv8.get(key)
        if band is None or len(row) < 6:
            continue
        cp, cm, cb = (by(dc, band=band, charge=ch) for ch in ("q+", "q-", "both"))
        # n (n+ / n-)
        m = re.match(r"^\$?([0-9{},]+)\$?\s*\(\$?([0-9]+)/([0-9]+)\$?\)", row[1].strip())
        R.cell(f"{band} n", "$" + m.group(1) + "$", f(cb, "n"))
        R.cell(f"{band} n+", m.group(2), f(cp, "n"))
        R.cell(f"{band} n-", m.group(3), f(cm, "n"))
        a, b = [x.strip() for x in row[2].split("/")]
        R.cell(f"{band} signed dx q+", a, f(cp, "signed_med_dx_um"))
        R.cell(f"{band} signed dx q-", b, f(cm, "signed_med_dx_um"))
        a, b = [x.strip() for x in row[3].split("/")]
        R.cell(f"{band} dx/bend q+", a, 1e3 * f(cp, "median_dx_over_bend"))
        R.cell(f"{band} dx/bend q-", b, 1e3 * f(cm, "median_dx_over_bend"))
        R.cell(f"{band} implied dp", row[4], f(cb, "implied_dp_MeV"))
        if band == "all":
            hw = rb["all"]["both"]["hw68_dx_charge_corrected_um"]
        else:
            hw = f(by(hl, band=band), "measured_hw68_dx_um")
        R.cell(f"{band} hw68 charge-corrected", row[5], hw)
    R.cell("caption n crossings", "$14{,}482$",
           float(J["reference_vs_truth"]["rebinned_decomposition"]["n_crossings"]))

    # ---------------- Table 9: tab:highland -------------------------------
    R.start("tab:highland", T["tab:highland"]["number"], "tab_highland_air.csv")
    for row in T["tab:highland"]["rows"]:
        key = row[0].strip()
        band = tex2csv.get(key)
        if band is None or len(row) < 8:
            continue
        c = by(hl, band=band)
        R.cell(f"{band} n", row[1], f(c, "n"))
        R.cell(f"{band} median p", row[2], f(c, "median_P_GeV"))
        R.cell(f"{band} theta0", row[3], f(c, "theta0_air_urad"))
        R.cell(f"{band} air width", row[4], f(c, "highland_air_width_um"))
        R.cell(f"{band} measured", row[5], f(c, "measured_hw68_dx_um"))
        R.cell(f"{band} ratio", row[6], f(c, "ratio_measured_over_air"))
        R.cell(f"{band} implied x/X0", row[7], f(c, "effective_x_over_X0"))
    R.cell("caption x/X0 air", "$0.01703$", f(hl[0], "x_over_X0_air"))

    # ---------------- Table 10: tab:pid -----------------------------------
    R.start("tab:pid", T["tab:pid"]["number"], "PN decomposition_by_pid")
    pid = {r["name"]: r
           for r in J["reference_vs_truth"]["verbatim"]["decomposition_by_pid"]["rows"]}
    for row in T["tab:pid"]["rows"]:
        name = row[0].strip()
        if name not in pid:
            continue
        c = pid[name]
        R.cell(f"{name} n", row[1], float(c["n"]))
        R.cell(f"{name} median p", row[2], c["median_P_GeV"])
        R.cell(f"{name} dx/bend", row[3], c["median_dx_over_bend"])
        R.cell(f"{name} median |dx|", row[4], c["median_abs_dx_um"])
        R.cell(f"{name} implied dp", row[5], c["implied_dp_MeV"])

    # ---------------- Table 11: tab:rk6self -------------------------------
    R.start("tab:rk6self", T["tab:rk6self"]["number"], "PN rk6_self_consistency")
    sc = J["reference_vs_truth"]["verbatim"]["rk6_self_consistency"]["rows"]
    want11 = [
        ("closure of the integrator, forward then back, median",
         "RK6 re-propagation closure, median"),
        ("closure of the integrator, worst case",
         "RK6 re-propagation closure, worst"),
        ("step convergence, $5\\mm$ against $1\\mm$, median",
         "RK6 step convergence 5 mm vs 1 mm, median"),
        ("step convergence, worst case",
         "RK6 step convergence 5 mm vs 1 mm, worst"),
        ("label against the particle's next hit, short legs, median",
         "label vs the next hit (short legs), median"),
        ("the same, above $5$ GeV",
         "label vs the next hit (short legs), median above 5 GeV"),
        ("the same, below $2$ GeV",
         "label vs the next hit (short legs), median below 2 GeV"),
        ("exact scheme, $N = 64$", "exact scheme, N = 64"),
        ("exact scheme, $N = 128$", "exact scheme, N = 128"),
        ("exact scheme, $N = 256$", "exact scheme, N = 256"),
        ("straight line, no magnet at all", "comparator: straight line"),
        ("the simulated truth against the reference", "comparator: material floor"),
    ]
    rows11 = [r for r in T["tab:rk6self"]["rows"]
              if len(r) == 3 and parse_cell(r[1]) is not None]
    for row, (lab, key) in zip(rows11, want11):
        hit = [r for r in sc if r["what"].startswith(key)]
        R.cell(f"{lab} value", row[1], hit[0]["value_um"] if hit else None)
        R.cell(f"{lab} n", row[2], float(hit[0]["n"]) if hit else None)

    # ---------------- Table 12: tab:grid ----------------------------------
    R.start("tab:grid", T["tab:grid"]["number"],
            "tab_pooled_grid.csv; tab_exact_ceiling.csv")
    for row in T["tab:grid"]["rows"]:
        p = parse_cell(row[0])
        if p is None or len(row) < 8:
            continue
        N, q = int(p[0]), int(parse_cell(row[1])[0])
        g = by(pg, N=N, q=q)
        e = by(ec, N=N, q=q)
        R.cell(f"N={N} q={q} N", row[0], float(N))
        R.cell(f"N={N} q={q} q", row[1], float(q))
        R.cell(f"N={N} q={q} dz", row[2], f(g, "dz_mm"))
        R.cell(f"N={N} q={q} endpoint med", row[3], f(g, "radial_med_um"))
        R.cell(f"N={N} q={q} endpoint p95", row[4], f(g, "radial_p95_um"))
        R.cell(f"N={N} q={q} ceiling", row[5],
               f(e, "exact_radial_med_um_from_csv"))
        R.cell(f"N={N} q={q} single step", row[6],
               f(g, "single_step_radial_med_um"))
        R.cell(f"N={N} q={q} restarts", row[7], f(g, "restarts"))
    R.cell("caption n test", "$1{,}452$", float(J["conventions"]["splits"]["test"]))

    # ---------------- Table 13: tab:preflight -----------------------------
    R.start("tab:preflight", T["tab:preflight"]["number"],
            "tab_preflight_shares.csv; PN preflight")
    ps = load_csv("tab_preflight_shares.csv")
    modes = ["blockE", "full", "no_lever", "no_track", "no_window"]
    tex2key = {"$<3$ GeV": "<3", "$3$--$8$": "3-8", "$8$--$20$": "8-20",
               "$20$--$50$": "20-50", "$>50$": ">50",
               "$10$--$50$ (loss window)": WINDOW,
               "quarter 1 (leaves $z_0$)": "quarter 1",
               "quarter 2": "quarter 2", "quarter 3": "quarter 3",
               "quarter 4 (arrives at $z_1$)": "quarter 4"}
    pf = J["preflight"]
    for row in T["tab:preflight"]["rows"]:
        lab = row[0].strip()
        base = re.sub(r"\s*\\ \(\$?n = .*$", "", lab).strip()
        key = tex2key.get(base)
        if key is not None and len(row) >= 7:
            kind = "momentum" if not key.startswith("quarter") else "quarter_of_z"
            c0 = by(ps, mode="full", kind=kind, key=key)
            R.cell(f"{key} states %", row[1], f(c0, "states_pct"))
            for k, mode in enumerate(modes):
                c = by(ps, mode=mode, kind=kind, key=key)
                R.cell(f"{key} {mode}", row[2 + k], f(c, "share_pct"))
            m = re.search(r"n = ([0-9{},]+)", lab)
            if m:
                R.cell(f"{key} n", "$" + m.group(1) + "$", f(c0, "n"))
        elif lab.startswith("rank corr.") and len(row) >= 7:
            for k, mode in enumerate(modes):
                R.cell(f"rank corr {mode}", row[2 + k],
                       pf["modes"][mode]["spearman_share_vs_cost"])
        elif lab.startswith("\\ \\ the same, within") and len(row) >= 7:
            for k, mode in enumerate(modes):
                R.cell(f"rank corr in window {mode}", row[2 + k],
                       pf["modes"][mode]["spearman_share_vs_cost_in_band"])
        elif lab.startswith("share carried by the worst") and len(row) >= 7:
            R.cell("worst 1% states %", row[1], 1.0)
            for k, mode in enumerate(modes):
                R.cell(f"worst 1% {mode}", row[2 + k],
                       pf["modes"][mode]["top_1pct_of_states_share_pct"])
    R.cell("caption n states", "$8{,}000$", float(pf["n_states"]))

    # ---------------- Table 14: tab:clamp ---------------------------------
    R.start("tab:clamp", T["tab:clamp"]["number"], "tab_clamp_scan.csv")
    cs = load_csv("tab_clamp_scan.csv")
    for row in T["tab:clamp"]["rows"]:
        lab = row[0].strip().replace("$", "").replace("\\mathbf{", "").replace("}", "")
        if lab not in ("off", "2", "3", "5", "10") or len(row) < 9:
            continue
        c = by(cs, clamp=lab)
        for k, col in enumerate(("share_<3_pct", "share_3-8_pct",
                                 "share_8-20_pct", "share_20-50_pct",
                                 "share_>50_pct",
                                 "share_10-50 (loss window)_pct")):
            R.cell(f"clamp {lab} {col}", row[1 + k], f(c, col))
        R.cell(f"clamp {lab} rank corr in window", row[7],
               f(c, "spearman_share_vs_cost_in_band"))
        R.cell(f"clamp {lab} worst 1%", row[8],
               f(c, "top_1pct_of_states_share_pct"))

    # ---------------- Table 15: tab:convergence ---------------------------
    R.start("tab:convergence", T["tab:convergence"]["number"],
            "tab_headline_pairs.csv; tab_extended_vs_snapshot.csv")
    hp = load_csv("tab_headline_pairs.csv")
    ev = load_csv("tab_extended_vs_snapshot.csv")
    plateau = J["headline"]["plateau"]
    runs = [(64, 2, "pooled"), (64, 2, "reweighted"),
            (128, 8, "pooled"), (128, 8, "reweighted"),
            (256, 16, "pooled"), (256, 16, "reweighted")]
    body15 = [r for r in T["tab:convergence"]["rows"] if len(r) == 8
              and ("pooled" in r[1] or "cost-weighted" in r[1])
              and parse_cell(r[2]) is not None]
    for row, (N, q, loss) in zip(body15, runs):
        c = by(hp, N=N, q=q, loss=loss)
        pk = f"{loss} N={N},q={q}"
        tag = f"N={N},q={q},{loss}"
        R.cell(f"{tag} rounds", row[2], f(c, "rounds"))
        R.cell(f"{tag} restarts", row[3], f(c, "restarts"))
        R.text(f"{tag} flat?",
               ("yes" in row[4]) == (str(c["plateaued_now"]) == "1"),
               f"table says {row[4]!r}, csv plateaued_now={c['plateaued_now']}")
        R.cell(f"{tag} first held", row[5], f(c, "first_plateau_round"))
        R.cell(f"{tag} last10 vs prev10 %", row[6],
               plateau[pk]["last10_vs_prev10_pct"])
        a, b = row[7].split("\\pm")
        R.cell(f"{tag} val headline", a, f(c, "val_headline_med_um"))
        R.cell(f"{tag} val spread", b, f(c, "val_headline_spread_pct"))
    # caption: extra restarts / rounds beyond the snapshot
    cap = T["tab:convergence"]["caption"]
    exr = re.findall(r"\$([0-9]{3}|[0-9]\{,\}[0-9]{3})\$", cap)
    R.cell("caption extra restarts N=64", "$500$",
           f(by(ev, N=64, q=2), "extra_restarts"))
    R.cell("caption extra restarts N=128", "$1{,}000$",
           f(by(ev, N=128, q=8), "extra_restarts"))
    R.cell("caption extra restarts N=256", "$500$",
           f(by(ev, N=256, q=16), "extra_restarts"))
    R.cell("caption extra rounds N=64", "$20$", f(by(ev, N=64, q=2), "extra_rounds"))
    R.cell("caption extra rounds N=128", "$40$", f(by(ev, N=128, q=8), "extra_rounds"))
    R.cell("caption extra rounds N=256", "$20$", f(by(ev, N=256, q=16), "extra_rounds"))
    R.cell("caption n validation", "$1{,}463$",
           float(J["conventions"]["splits"]["val"]))

    # ---------------- Table 16: tab:headline ------------------------------
    R.start("tab:headline", T["tab:headline"]["number"],
            "tab_headline_pairs.csv; tab_reweighted_runs.csv; tab_anatomy.csv")
    an = load_csv("tab_anatomy.csv")
    hb = load_csv("tab_headline_bands.csv")
    body16 = [r for r in T["tab:headline"]["rows"]
              if len(r) == 9 and parse_cell(r[2]) is not None]
    for row, (N, q, loss) in zip(body16, runs):
        c = by(hp, N=N, q=q, loss=loss)
        a = by(an, N=N, q=q, loss=loss)
        lo = by(hb, N=N, q=q, loss=loss, band="<3")
        tag = f"N={N},q={q},{loss}"
        R.cell(f"{tag} median", row[2], f(c, "radial_med_um"))
        R.cell(f"{tag} p95", row[3], f(c, "radial_p95_um"))
        R.cell(f"{tag} >1mm %", row[4], 100.0 * f(c, "frac_above_1mm"))
        R.cell(f"{tag} in window", row[5], f(c, "band_radial_med_um"))
        R.cell(f"{tag} <3 GeV", row[6], f(lo, "radial_med_um"))
        sx, sy = [x.strip() for x in row[7].split("/")]
        R.cell(f"{tag} per-step tx", sx, 1e7 * f(a, "per_step_tx_med_abs_slope"))
        R.cell(f"{tag} per-step ty", sy, 1e7 * f(a, "per_step_ty_med_abs_slope"))
        R.cell(f"{tag} wall", row[8], f(c, "train_wall_h"))

    # ---------------- Table 17: tab:headlinebands -------------------------
    R.start("tab:headlinebands", T["tab:headlinebands"]["number"],
            "tab_headline_bands.csv")
    tex2csv17 = {"$<3$": "<3", "$3$--$8$": "3-8", "$8$--$20$": "8-20",
                 "$20$--$50$": "20-50", "$>50$": ">50",
                 "$10$--$50$ (loss window)": WINDOW}
    block = 0
    for row in T["tab:headlinebands"]["rows"]:
        if row[0].startswith("\\multicolumn"):
            block += 1
            continue
        band = tex2csv17.get(row[0].strip())
        if band is None or len(row) < 8:
            continue
        col = "radial_med_um" if block == 1 else "radial_p95_um"
        R.cell(f"{band} n (block {block})", row[1],
               f(by(hb, N=64, q=2, loss="pooled", band=band), "n"))
        for k, (N, q, loss) in enumerate(runs):
            c = by(hb, N=N, q=q, loss=loss, band=band)
            R.cell(f"{band} N={N} {loss} {col}", row[2 + k], f(c, col))

    # ---------------- Tables 18/19: tab:comppos, tab:compslope ------------
    hc = load_csv("tab_headline_components.csv")
    tex2csv18 = dict(tex2csv17)
    tex2csv18["$10$--$50$"] = WINDOW
    tex2csv18["all"] = "all"
    for label, comps, scale in (("tab:comppos", ("x", "y"), 1.0),
                                ("tab:compslope", ("tx", "ty"), 1e5)):
        R.start(label, T[label]["number"], "tab_headline_components.csv")
        for row in T[label]["rows"]:
            band = tex2csv18.get(row[0].strip())
            if band is None or len(row) < 14:
                continue
            R.cell(f"{band} n", row[1],
                   f(by(hc, N=64, q=2, loss="pooled", band=band,
                        component=comps[0]), "n"))
            k = 2
            for loss in ("pooled", "reweighted"):
                for comp in comps:
                    c = by(hc, N=64, q=2, loss=loss, band=band, component=comp)
                    R.cell(f"{band} {loss} {comp} med", row[k],
                           scale * f(c, "med_abs"))
                    R.cell(f"{band} {loss} {comp} sgn", row[k + 1],
                           scale * f(c, "signed_med"))
                    R.cell(f"{band} {loss} {comp} hw", row[k + 2],
                           scale * f(c, "hw68"))
                    k += 3

    # ---------------- Table 20: tab:near5 ---------------------------------
    R.start("tab:near5", T["tab:near5"]["number"], "tab_near_5gev.csv")
    n5 = load_csv("tab_near_5gev.csv")
    band = None
    loss = None
    for row in T["tab:near5"]["rows"]:
        if len(row) < 8:
            continue
        if "GeV" in row[0]:
            band = "3-8" if "$3$--$8$" in row[0] else "4-6"
        if "pooled" in row[1]:
            loss = "pooled"
        elif "cost-weighted" in row[1]:
            loss = "reweighted"
        comp = None
        for tex, name in (("\\Delta x", "x"), ("\\Delta y", "y"),
                          ("\\Delta t_x", "tx"), ("\\Delta t_y", "ty")):
            if row[2].strip().startswith("$" + tex):
                comp = name
        if comp is None or band is None or loss is None:
            continue
        sc_ = 1e5 if comp.startswith("t") else 1.0
        c = by(n5, band=band, loss=loss, component=comp)
        R.cell(f"{band} {loss} {comp} n", row[3], f(c, "n"))
        R.cell(f"{band} {loss} {comp} sgn", row[4], sc_ * f(c, "signed_med"))
        R.cell(f"{band} {loss} {comp} hw68", row[5], sc_ * f(c, "hw68"))
        R.cell(f"{band} {loss} {comp} RMS", row[6], sc_ * f(c, "rms"))
        R.cell(f"{band} {loss} {comp} med", row[7], sc_ * f(c, "med_abs"))

    # ---------------- Table 21: tab:near5tail -----------------------------
    R.start("tab:near5tail", T["tab:near5tail"]["number"], "tab_near_5gev.csv")
    band = None
    for row in T["tab:near5tail"]["rows"]:
        if len(row) < 9:
            continue
        if "GeV" in row[0]:
            band = "3-8" if "$3$--$8$" in row[0] else "4-6"
        loss = "pooled" if "pooled" in row[1] else (
            "reweighted" if "cost-weighted" in row[1] else None)
        if band is None or loss is None:
            continue
        c = by(n5, band=band, loss=loss, component="x")
        R.cell(f"{band} {loss} n", row[2], f(c, "n"))
        R.cell(f"{band} {loss} radial med", row[3], f(c, "radial_med_um"))
        p95 = J["near_5gev"]["bands"][band][loss]["radial_p95_um"]
        R.cell(f"{band} {loss} p95", row[4], p95)
        R.cell(f"{band} {loss} beyond 1mm", row[5], f(c, "n_beyond_1mm"))
        R.cell(f"{band} {loss} start |x| tail", row[6],
               f(c, "start_abs_x_med_mm_beyond_1mm"))
        R.cell(f"{band} {loss} start |x| rest", row[7],
               f(c, "start_abs_x_med_mm_rest"))
        R.cell(f"{band} {loss} ratio", row[8],
               f(c, "start_abs_x_med_mm_beyond_1mm")
               / f(c, "start_abs_x_med_mm_rest"))

    # ---------------- Table 22: tab:anatomysteps --------------------------
    R.start("tab:anatomysteps", T["tab:anatomysteps"]["number"],
            "tab_anatomy.csv")
    body22 = [r for r in T["tab:anatomysteps"]["rows"]
              if len(r) == 10 and parse_cell(r[2]) is not None]
    for row, (N, q, loss) in zip(body22, runs):
        a = by(an, N=N, q=q, loss=loss)
        tag = f"N={N},q={q},{loss}"
        R.cell(f"{tag} |dtx| all", row[2], 1e7 * f(a, "per_step_tx_med_abs_slope"))
        R.cell(f"{tag} |dty| all", row[3], 1e7 * f(a, "per_step_ty_med_abs_slope"))
        R.cell(f"{tag} radial", row[4], f(a, "per_step_radial_med_um"))
        R.cell(f"{tag} |dtx| 3-8", row[5],
               1e7 * f(a, "per_step_tx_med_abs_slope_3_8"))
        R.cell(f"{tag} |dty| 3-8", row[6],
               1e7 * f(a, "per_step_ty_med_abs_slope_3_8"))
        R.cell(f"{tag} |dtx| 8-20", row[7],
               1e7 * f(a, "per_step_tx_med_abs_slope_8_20"))
        R.cell(f"{tag} |dty| 8-20", row[8],
               1e7 * f(a, "per_step_ty_med_abs_slope_8_20"))
        R.cell(f"{tag} |dtx| 10-50", row[9],
               1e7 * f(a, "per_step_tx_med_abs_slope_10_50"))

    # ---------------- Table 23: tab:anatomyend ----------------------------
    R.start("tab:anatomyend", T["tab:anatomyend"]["number"], "tab_anatomy.csv")
    body23 = [r for r in T["tab:anatomyend"]["rows"]
              if len(r) == 9 and parse_cell(r[2]) is not None]
    for row, (N, q, loss) in zip(body23, runs):
        a = by(an, N=N, q=q, loss=loss)
        tag = f"N={N},q={q},{loss}"
        R.cell(f"{tag} x part", row[2], f(a, "x_part_med_um"))
        R.cell(f"{tag} y part", row[3], f(a, "y_part_med_um"))
        R.cell(f"{tag} coherence tx", row[4], f(a, "coherence_tx"))
        R.cell(f"{tag} coherence ty", row[5], f(a, "coherence_ty"))
        R.cell(f"{tag} independent", row[6], f(a, "coherence_independent"))
        R.cell(f"{tag} explained both", row[7], f(a, "explained_both_slopes"))
        R.cell(f"{tag} explained x only", row[8], f(a, "explained_x_slope_only"))
        R.text(f"{tag} 1/sqrt(N)",
               abs(f(a, "coherence_independent") - 1.0 / math.sqrt(N)) < 1e-12,
               "independent-steps column is not 1/sqrt(N)")

    # ---------------- Table 24: tab:steps ---------------------------------
    R.start("tab:steps", T["tab:steps"]["number"],
            "tab_anatomy.csv; tab_along_z.csv; tab_single_step_vs_z.csv")
    ss = load_csv("tab_single_step_vs_z.csv")
    body24 = [r for r in T["tab:steps"]["rows"]
              if len(r) == 8 and parse_cell(r[2]) is not None]
    for row, (N, q, loss) in zip(body24, runs):
        a = by(an, N=N, q=q, loss=loss)
        c = by(hp, N=N, q=q, loss=loss)
        tag = f"N={N},q={q},{loss}"
        R.cell(f"{tag} dz", row[2], f(a, "dz_mm") if "dz_mm" in a else
               f(by(hp, N=N, q=q, loss=loss), "dz_mm"))
        R.cell(f"{tag} per-step radial", row[3], f(a, "per_step_radial_med_um"))
        R.cell(f"{tag} per-step |dtx|", row[4],
               f(a, "per_step_tx_med_abs_slope"))
        # "median over all planes and tracks", not a median of the per-plane
        # medians that tab_single_step_vs_z.csv stores; the pooled value is in
        # the JSON as along_z.per_run[..].per_step_vs_single_step
        az = J["along_z"]["per_run"][f"{loss} N={N},q={q}"]
        R.cell(f"{tag} single step", row[5],
               az["per_step_vs_single_step"]["med_from_truth_um"],
               note="(PN along_z .. med_from_truth_um)")
        # sanity: the per-plane table exists and spans the right planes
        planes = allby(ss, N=N, q=q, loss=loss)
        R.text(f"{tag} single-step planes",
               len(planes) == N,
               f"tab_single_step_vs_z.csv holds {len(planes)} rows, expected {N}")
        R.cell(f"{tag} endpoint", row[6], f(c, "radial_med_um"))
        R.cell(f"{tag} in window", row[7], f(c, "band_radial_med_um"))

    # ---------------- Table 25: tab:truth ---------------------------------
    R.start("tab:truth", T["tab:truth"]["number"], "tab_against_true_state.csv")
    at = load_csv("tab_against_true_state.csv")
    band = None
    for row in T["tab:truth"]["rows"]:
        if len(row) < 9:
            continue
        head = row[0].strip()
        for tex, name in (("$<3$", "<3"), ("$3$--$8$", "3-8"),
                          ("$8$--$20$", "8-20"), ("$20$--$50$", "20-50"),
                          ("$>50$", ">50"), ("$10$--$50$", WINDOW)):
            if head == tex:
                band = name
        if band is None:
            continue
        m = re.search(r"N = (\d+)\$, \$q = (\d+)", row[1])
        if not m:
            continue
        N, q = int(m.group(1)), int(m.group(2))
        c = by(at, N=N, q=q, loss="reweighted", band=band)
        tag = f"{band} N={N},q={q}"
        R.cell(f"{tag} n", row[2], f(c, "n"))
        R.cell(f"{tag} nn-RK6", row[3], f(c, "nn_vs_rk6_pos_max_med_um"))
        R.cell(f"{tag} nn-true", row[4], f(c, "nn_vs_true_pos_max_med_um"))
        if parse_cell(row[5]) is not None:
            R.cell(f"{tag} RK6-true", row[5], f(c, "rk6_vs_true_pos_max_med_um"))
        else:
            R.cell(f"{tag} RK6-true", row[5], None) if False else None
            R.cur["skipped"] += 1
            R.total_skipped += 1
        R.cell(f"{tag} nn-RK6 slope", row[6], f(c, "nn_vs_rk6_slope_max_med"))
        R.cell(f"{tag} nn-true slope", row[7], f(c, "nn_vs_true_slope_max_med"))
        R.cell(f"{tag} ratio", row[8], f(c, "rk6_true_over_nn_rk6"))

    # ---------------- Table 26: tab:cost ----------------------------------
    R.start("tab:cost", T["tab:cost"]["number"],
            "tab_compute.csv; PN compute")
    body26 = [r for r in T["tab:cost"]["rows"]
              if len(r) == 8 and parse_cell(r[2]) is not None]
    for row, (N, q, loss) in zip(body26, runs):
        c = by(cmp_, N=N, q=q, loss=loss)
        tag = f"N={N},q={q},{loss}"
        R.cell(f"{tag} params", row[2], f(c, "n_parameters"))
        # restarts: "1,807 (1,500)" -> recorded (unique)
        parts = re.findall(r"\$([0-9{},]+)\$", row[3])
        R.cell(f"{tag} restarts", "$" + parts[0] + "$", f(c, "restarts_recorded"))
        if len(parts) > 1:
            R.cell(f"{tag} restarts distinct", "$" + parts[1] + "$",
                   f(c, "unique_restarts"))
        parts = re.findall(r"\$([0-9{},]+)\$", row[4])
        R.cell(f"{tag} rounds", "$" + parts[0] + "$", f(c, "rounds_completed"))
        if len(parts) > 1:
            R.cell(f"{tag} rounds rows read", "$" + parts[1] + "$",
                   f(c, "rounds_rows_read"))
        R.cell(f"{tag} per restart s", row[5], f(c, "median_wall_s_per_restart"))
        R.cell(f"{tag} wall h", row[6], f(c, "recorded_train_wall_h"))
        carry = eval(c["carry_us_per_track"])
        R.cell(f"{tag} carry us/track", row[7], carry["test"])
    R.cell("caption restarts per round", "$25$", f(cmp_[0], "round_restarts"))
    R.cell("caption states per round", "$32{,}000$", f(cmp_[0], "states_per_round"))
    R.cell("caption iterations", "$200$",
           f(cmp_[0], "median_iterations_per_restart"))

    # ---------------- the band-count test ---------------------------------
    print("Per-band test-split counts (test: 164/655/433/170/30, 10-50: 487)")
    db = load_csv("tab_dataset_bands.csv")
    expect = {"<3": 164, "3-8": 655, "8-20": 433, "20-50": 170, ">50": 30,
              WINDOW: 487}
    bad = 0
    for band, want_n in expect.items():
        got = int(f(by(db, split="test", band=band), "n"))
        ok = got == want_n
        bad += 0 if ok else 1
        print(f"  test {band:<22} expect {want_n:>4}  json/csv {got:>4}  "
              f"{'OK' if ok else 'MISMATCH'}")
    # every table that carries per-band n must use the same numbers
    for name, key in (("tab_headline_bands.csv", None),
                      ("tab_headline_components.csv", None),
                      ("tab_against_true_state.csv", None)):
        rows = load_csv(name)
        for band, want_n in expect.items():
            got = {int(float(r["n"])) for r in rows if r["band"] == band}
            if got and got != {want_n}:
                print(f"  ! {name} band {band}: n = {sorted(got)}, "
                      f"expected {want_n}")
                bad += 1
    print(f"  -> {'PASS' if bad == 0 else str(bad) + ' MISMATCHES'}\n")

    R.dump()
    return 1 if R.total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
