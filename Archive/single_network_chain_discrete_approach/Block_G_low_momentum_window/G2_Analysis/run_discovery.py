#!/usr/bin/env python
"""G2/G3 - finding the trained runs and naming them the way an output should.

Every analysis script in G2_Analysis and G3_Analysis takes `--runs <dir>` and
asks this module which run folders in it are analysable. A folder counts when
it holds BOTH a `record.json` (the trainer's summary) and a `chain_states.npz`
(the stored states of every test/validation track on every plane of the chain):
those two are what the analysis reads. A run that is still training may have
written a record when it hit its restart cap and then been reopened with
`--extend`; `progress.json`'s `phase` is the only honest way to tell, so the
checkpoint flag is read from there and never from the presence of a record.

The loss window each run was trained with is read from `scale.json`
("weighting" -> p_lo, p_hi), so a figure or table can say which window it is
describing without the caller having to know.

NAMING (George, 2026-09-21): an output a human reads names a network by what it
is - N steps, q stages, dz mm a step - never by the folder it lives in and never
by a block letter. `label`, `tick_label`, `window_text` and `study_text` below
are the only place that wording is written down, so every script says it the
same way. Block letters are fine here in the docstrings.

This module imports nothing but the standard library, so importing it cannot
reorder anything.
"""
from __future__ import annotations

import glob
import json
import os

L_MM = 5177.8                     # the magnet crossing, last UT plane to first SciFi plane
STUDY = "one network per step length, chained"


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def window_of(run):
    """(p_lo, p_hi) in GeV from the run's scale.json, or None if not recorded."""
    w = _load(os.path.join(run, "scale.json")).get("weighting")
    if not isinstance(w, dict):
        return None
    lo, hi = w.get("p_lo"), w.get("p_hi")
    if lo is None or hi is None:
        return None
    return float(lo), float(hi)


def find_runs(runs_dir, only=None, require_states=True):
    """The analysable run folders under `runs_dir`, in N then q order.

    `only` is an optional comma-separated list of folder names (N064_q02,...).
    `require_states` False keeps the folders that have a record but no stored
    chain states (the convergence check only needs rounds.csv).
    """
    runs_dir = os.path.abspath(runs_dir)
    keep = set(only.split(",")) if only else None
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        tag = os.path.basename(d)
        if keep is not None and tag not in keep:
            continue
        rec = _load(os.path.join(d, "record.json"))
        has_states = os.path.exists(os.path.join(d, "chain_states.npz"))
        if not rec or (require_states and not has_states):
            continue
        prog = _load(os.path.join(d, "progress.json"))
        out.append(dict(
            path=d, tag=tag, N=int(rec["N"]), q=int(rec["q"]), dz_mm=L_MM / int(rec["N"]),
            restarts=rec.get("restarts"), rounds=rec.get("rounds"),
            checkpoint=prog.get("phase") != "done",
            at_restart=prog.get("restart"), at_round=prog.get("round"),
            has_states=has_states, window=window_of(d)))
    out.sort(key=lambda r: (r["N"], r["q"]))
    return out


def label(r, checkpoint=True):
    """"N = 64, q = 2, dz = 81 mm" (+ the checkpoint warning) - for a table row."""
    s = "N = %d, q = %d, dz = %.0f mm" % (r["N"], r["q"], r["dz_mm"])
    if checkpoint and r["checkpoint"]:
        s += "  [checkpoint at restart %s; the run is at %s and still training]" % (
            r["restarts"], r["at_restart"])
    return s


def short_label(r):
    """"N = 64, q = 2, dz = 81 mm (checkpoint)" - for a legend entry."""
    return "N = %d, q = %d, dz = %.0f mm%s" % (
        r["N"], r["q"], r["dz_mm"], " (checkpoint)" if r["checkpoint"] else "")


def tick_label(r):
    """The same name over three lines, for an axis tick."""
    return "N = %d\nq = %d\ndz = %.0f mm%s" % (
        r["N"], r["q"], r["dz_mm"], "\n(checkpoint)" if r["checkpoint"] else "")


def window_text(runs):
    """"3–8 GeV" - the loss window the runs were trained with."""
    ws = sorted({r["window"] for r in runs})
    if ws == [None]:
        return "not recorded"
    parts = ["%g–%g GeV" % w for w in ws if w is not None]
    if None in ws:
        parts.append("one run without a recorded window")
    return parts[0] if len(parts) == 1 else "mixed: " + ", ".join(parts)


def study_text(runs):
    """The one-line description of the study, for a figure title."""
    return "%s; loss window %s" % (STUDY, window_text(runs))


def describe(runs, runs_dir):
    """The header every script prints, so a log says what it read."""
    lines = ["runs: %s" % os.path.abspath(runs_dir),
             "study: %s" % study_text(runs)]
    for r in runs:
        lines.append("  %-70s restarts %-6s rounds %-4s %s" % (
            label(r), r["restarts"], r["rounds"],
            "CHECKPOINT (still training)" if r["checkpoint"] else "training finished"))
    return "\n".join(lines)


def require(runs, runs_dir):
    if not runs:
        raise SystemExit(
            "no analysable run folders under %s\n"
            "(a run counts once it has both record.json and chain_states.npz)"
            % os.path.abspath(runs_dir))
    return runs
