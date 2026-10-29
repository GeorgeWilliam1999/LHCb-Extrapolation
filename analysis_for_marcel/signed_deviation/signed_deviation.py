"""Signed deviation, reference (RK6) minus predicted, per track, split by charge and momentum.

For Marcel, 2026-10-29. Read-only on the store and the package; writes only to the
output folder given as the first argument.

Network loading, whole-track carry and single steps are exactly those of
rkpinn.run_record.report_of_a_run (the code of standard output 11): the last
round's snapshot, the test split, run.network.whole_track from the start state,
the reference on the planes the steps start on plus the last plane, and
_separate_steps for the single steps. The package's sign is predicted - reference;
here it is negated to reference - predicted.

    python signed_deviation.py <out> 9cef9c160b7d 25dfa17ef168 49e5e2450eaa
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rkpinn.evaluation import conventions, resolution
from rkpinn.equation_of_motion.lhcb import CHARGE_OVER_MOMENTUM_PER_INVERSE_GEV
from rkpinn.run_record.configuration import read_configuration
from rkpinn.run_record.report_of_a_run import _separate_steps
from rkpinn.run_record.store import Store
from rkpinn.training.checkpoints import RunFolder
from rkpinn.training.round_trainer import build_run

STORE = "/data/bfys/gscriven/rkpinn_store"
TITLES = {
    "25dfa17ef168": "straight line, per-track scale, 64 steps",
    "9cef9c160b7d": "straight line, global scale, 64 steps",
    "49e5e2450eaa": "direct stage states, 64 steps",
}
STORED_MEDIANS = {"9cef9c160b7d": (1.40, 1.00), "49e5e2450eaa": (3.72, 2.81),
                  "25dfa17ef168": (1.26, 0.86)}
COMPONENTS = (("dx_um", "x", "x [um]"), ("dy_um", "y", "y [um]"),
              ("dtx", "slope in x", "tx"), ("dty", "slope in y", "ty"))
P_EDGES = (2.0, 4.0, 8.0, 16.0, 32.0)          # last bin is [32, max]
BINS_IN_A_PANEL = 50
CHARGES = ((+1, "q > 0", "tab:red"), (-1, "q < 0", "tab:blue"))


def momentum_bins(p):
    """[(label, mask)]: [2,4), [4,8), [8,16), [16,32), [32, max]."""
    out = []
    for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
        out.append(("[%g, %g) GeV" % (lo, hi), (p >= lo) & (p < hi)))
    out.append(("[%g, %.0f] GeV" % (P_EDGES[-1], p.max()), p >= P_EDGES[-1]))
    return out


def deviations(predicted, reference):
    """reference - predicted, x and y in micrometres, slopes with no unit."""
    e = conventions.errors(predicted, reference)          # predicted - reference
    return {key: -np.asarray(e[name], dtype=float) for key, name, _ in COMPONENTS}


def stats(v):
    n = v.size
    if n == 0:
        return dict(N=0, mean=np.nan, median=np.nan, std=np.nan, half_width_68=np.nan,
                    mean_over_standard_error=np.nan)
    lo, hi = np.quantile(v, [0.16, 0.84])
    std = float(v.std(ddof=1)) if n > 1 else np.nan
    se = std / np.sqrt(n) if n > 1 else np.nan
    return dict(N=int(n), mean=float(v.mean()), median=float(np.median(v)), std=std,
                half_width_68=float((hi - lo) / 2), mean_over_standard_error=float(v.mean() / se)
                if se and se > 0 else np.nan)


def fmt(x, key):
    if key in ("dx_um", "dy_um"):
        return "%.3g" % x
    return "%.2e" % x


def page(key, kind, dev, p, q, note):
    bins = momentum_bins(p)
    fig, axes = plt.subplots(len(bins), 4, figsize=(20, 4.0 * len(bins)))
    for row, (label, mask) in zip(axes, bins):
        for ax, (ck, _, cl) in zip(row, COMPONENTS):
            v_all = dev[ck][mask]
            if v_all.size == 0:
                ax.set_axis_off()
                continue
            lo, hi = np.quantile(v_all, [0.01, 0.99])
            if not hi > lo:
                lo, hi = lo - 1e-12, hi + 1e-12
            edges = np.linspace(lo, hi, BINS_IN_A_PANEL + 1)
            for charge, name, colour in CHARGES:
                v = dev[ck][mask & (q == charge)]
                s = stats(v)
                under, over = int((v < lo).sum()), int((v > hi).sum())
                ax.hist(np.clip(v, lo, hi)[(v >= lo) & (v <= hi)], bins=edges, histtype="step",
                        color=colour, lw=1.3,
                        label="%s: N=%d, mean=%s, std=%s,\n  w68=%s, median=%s, outside=%d+%d"
                        % (name, s["N"], fmt(s["mean"], ck), fmt(s["std"], ck),
                           fmt(s["half_width_68"], ck), fmt(s["median"], ck), under, over))
            ax.axvline(0.0, color="k", lw=0.8)
            ax.set_xlim(lo, hi)
            ax.set_xlabel("reference - predicted, %s" % cl)
            ax.set_title("%s, %s" % (label, cl), fontsize=10)
            ax.legend(fontsize=7, loc="upper left", frameon=False)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
        row[0].set_ylabel("entries")
    fig.suptitle("%s: %s, signed deviation reference (RK6) - predicted, test split, by charge\n"
                 "run %s; x-range = 1st-99th percentile of both charges; outside = entries "
                 "below+above the range (counted in N, mean, std, w68); "
                 "w68 = half the 16th-84th percentile range\n%s"
                 % (TITLES[key], kind, key, note), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def main():
    out = sys.argv[1]
    keys = sys.argv[2:]
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    store = Store(STORE)
    summary = {"endpoint": [], "single_step": []}
    pages = {"endpoint": [], "single_step": []}
    facts = {}
    for key in keys:
        folder = RunFolder(store, key)
        state = folder.state()
        rnd = state["rounds_done"]
        configuration = read_configuration(folder.path("configuration.yaml"))
        run = build_run(configuration, store, folder.constants())
        run.network.load_state_dict(folder.weights_of_round(rnd))
        tracks = run.tracks.split("test")
        layout = run.layout
        steps = layout.number_of_steps
        whole = run.network.whole_track(tracks.start_state, layout)
        on_planes = whole.states_on_planes()
        on = np.append(run.tracks.planes_a_step_starts_on(steps),
                       run.tracks.number_of_steps_between_planes)
        reference = tracks.reference_states_on_planes[:, on]
        single = _separate_steps(run, reference)

        p = np.asarray(tracks.momentum_gev, dtype=float)
        qop = np.asarray(tracks.start_state[:, 4], dtype=float)
        q = np.sign(qop).astype(int)
        p_from_qop = CHARGE_OVER_MOMENTUM_PER_INVERSE_GEV / np.abs(qop)
        end = deviations(on_planes[:, -1], reference[:, -1])
        one = deviations(single["end_states"], single["reference_after_the_step"])
        p_one = resolution.momentum_of_each_single_step(p, len(single["end_states"]))
        q_one = np.repeat(q, steps)

        med = (float(np.median(np.abs(end["dx_um"]))), float(np.median(np.abs(end["dy_um"]))))
        facts[key] = dict(
            round=rnd, n_tracks=int(len(p)), n_single_steps=int(len(p_one)),
            planes_mm_first=float(run.tracks.planes_mm[0]),
            planes_mm_last=float(run.tracks.planes_mm[-1]),
            number_of_stored_planes=int(len(run.tracks.planes_mm)),
            step_length_mm=float(layout.step_length_mm),
            start_planes_first_last=[float(layout.start_planes_mm[0]),
                                     float(layout.start_planes_mm[-1])],
            reference_plane_indices_first_last=[int(on[0]), int(on[-1])], number_of_steps=int(steps),
            endpoint_median_abs_x_um=med[0], endpoint_median_abs_y_um=med[1],
            stored_median_x_y=STORED_MEDIANS.get(key),
            n_positive=int((q > 0).sum()), n_negative=int((q < 0).sum()), n_zero=int((q == 0).sum()),
            max_rel_diff_p_vs_qop=float(np.max(np.abs(p_from_qop / p - 1))),
            qop_conserved_on_last_plane=float(np.max(np.abs(reference[:, -1, 4] - qop))),
            p_min=float(p.min()), p_max=float(p.max()), n_below_2_gev=int((p < 2).sum()))
        print(key, json.dumps(facts[key]), flush=True)

        with open(os.path.join(out, "per_track_endpoint_%s.csv" % key), "w", newline="") as h:
            w = csv.writer(h)
            w.writerow(["track_index", "p_GeV", "charge", "dx_um", "dy_um", "dtx", "dty"])
            for i in range(len(p)):
                w.writerow([i, "%.6g" % p[i], int(q[i])] +
                           ["%.9g" % end[ck][i] for ck, _, _ in COMPONENTS])

        for kind, dev, pp, qq, note in (
                ("endpoint", end, p, q,
                 "endpoint: whole track carried by the network from the first plane, compared "
                 "on the last plane; one entry per track"),
                ("single_step", one, p_one, q_one,
                 "single step: one network step from the RK6 state on each of the %d start planes, "
                 "compared one step later; %d entries per track (not independent)" % (steps, steps))):
            fig = page(key, kind.replace("_", " "), dev, pp, qq, note)
            fig.savefig(os.path.join(out, "figures", "%s_%s.png" % (kind, key)), dpi=110)
            pages[kind].append(fig)
            for label, mask in momentum_bins(pp):
                for ck, _, _ in COMPONENTS:
                    for charge, name, _ in CHARGES:
                        summary[kind].append(dict(run=key, title=TITLES[key], round=rnd,
                                                  momentum_bin=label, component=ck, charge=charge,
                                                  **stats(dev[ck][mask & (qq == charge)])))

    for kind in pages:
        with PdfPages(os.path.join(out, "figures", "%s_by_charge_and_momentum.pdf" % kind)) as pdf:
            for fig in pages[kind]:
                pdf.savefig(fig)
                plt.close(fig)
        with open(os.path.join(out, "summary_%s.csv" % kind), "w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(summary[kind][0]))
            w.writeheader()
            w.writerows(summary[kind])
    with open(os.path.join(out, "facts.json"), "w") as h:
        json.dump(facts, h, indent=1)


if __name__ == "__main__":
    main()
