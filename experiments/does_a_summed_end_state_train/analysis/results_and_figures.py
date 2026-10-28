"""The results tables and figures of the experiment, read from the standard reports in the store.

    results_and_figures.py

Reads, for every run in RUNS, the standard report of the round named there (the last round
when None) and the run's own rounds.csv. Computes nothing new: every number is copied from
those files. Writes nothing to the store.

Writes
    results/runs.csv                         one row per run: state, rounds, the three medians,
                                             the exact scheme's endpoint (the ceiling), the cost
    results/endpoint_by_momentum_band.csv    radial endpoint error per momentum band
    results/error_along_the_track.csv        radial median at every plane of the chain
    results/validation_per_round.csv         the validation error and the loss of every round
    results/references.csv                   the numbers the criteria are read against, with source
    figures/validation_per_round.png
    figures/endpoint_by_momentum_band.png
    figures/error_along_the_track.png
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STORE = "/data/bfys/gscriven/rkpinn_store"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS, FIGURES = os.path.join(HERE, "results"), os.path.join(HERE, "figures")

# panel of the figures, label, run key, round of the report (None: the last round of the run)
RUNS = [
    ("64 steps, 2 stages, pooled loss", "end state summed from the stages (run A)", "bddea42ea1ef", None),
    ("64 steps, 2 stages, pooled loss", "end state predicted (run B)", "4cc888dcc710", None),
]
# label, endpoint radial median (micrometres), single-step radial median (micrometres), source
REFERENCES = [
    ("frozen network, pooled loss, 64 steps 2 stages, end state predicted", 145.7, 0.333,
     "../README.md, criteria"),
    ("pass line of criteria 1 and 2: ten times the frozen single step", "", 3.33,
     "../README.md, criteria"),
]

# panel, label, value: drawn as a cross at "all momenta" in the figure of the momentum bands
REFERENCE_MARKS = [
    ("64 steps, 2 stages, pooled loss", "frozen network, end state predicted", 145.7),
]

BANDS = ["below 3 GeV", "3 to 8 GeV", "8 to 20 GeV", "20 to 50 GeV", "above 50 GeV", "all momenta"]
# the reference categorical order of the dataviz palette, as in the step-length experiment
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a8985"]
INK, INK_2 = "#0b0b0b", "#52514e"


def read_csv(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_folder(key, round_asked):
    state = json.load(open(os.path.join(STORE, "runs", key, "state.json")))
    wanted = state["rounds_done"] if round_asked is None else round_asked
    found = sorted(glob.glob(os.path.join(STORE, "runs", key, "reports", "round_%04d_at_*" % wanted)))
    if not found:
        raise SystemExit("no standard report of round %d of run %s" % (wanted, key))
    return state, wanted, found[-1]


def collect():
    os.makedirs(RESULTS, exist_ok=True)
    runs, bands, along, rounds = [], [], [], []
    for panel, label, key, round_asked in RUNS:
        state, used, folder = report_folder(key, round_asked)
        summary = json.load(open(os.path.join(folder, "summary.json")))
        exact = [r for r in read_csv(os.path.join(folder, "three_references.csv"))
                 if r["band"] == "all momenta" and r["comparison"] == "exact scheme against reference"][0]
        cost = read_csv(os.path.join(folder, "cost.csv"))[0]
        radial = {r["band"]: r for r in read_csv(os.path.join(folder, "endpoint_error_magnitudes.csv"))
                  if r["magnitude"] == "radial"}
        runs.append({
            "panel": panel, "label": label, "run_key": key,
            "state_of_the_run": state["state"], "rounds_done": state["rounds_done"],
            "rounds_at_most": state["rounds_at_most"], "round_of_the_report": used,
            "report": os.path.relpath(folder, STORE),
            "endpoint_radial_median_micrometres": summary["endpoint_radial_median_micrometres"],
            "endpoint_radial_percentile_95_micrometres": radial["all momenta"]["percentile_95_micrometres"],
            "endpoint_larger_of_x_and_y_median_micrometres":
                summary["endpoint_larger_of_x_and_y_median_micrometres"],
            "single_step_radial_median_micrometres": summary["single_step_radial_median_micrometres"],
            "exact_scheme_endpoint_radial_median_micrometres": exact["position_radial_median_micrometres"],
            "hours_of_training": cost["hours_of_training"],
            "microseconds_to_carry_one_track": cost["microseconds_to_carry_one_track"],
        })
        for band in BANDS:
            bands.append({"label": label, "run_key": key, "round_of_the_report": used, "band": band,
                          "number_of_tracks": radial[band]["number_of_tracks"],
                          "radial_median_micrometres": radial[band]["median_micrometres"],
                          "radial_percentile_95_micrometres": radial[band]["percentile_95_micrometres"]})
        for row in read_csv(os.path.join(folder, "error_along_the_track.csv")):
            along.append({"label": label, "run_key": key, "round_of_the_report": used,
                          "plane": row["plane"], "z_mm": row["z_mm"],
                          "radial_median_micrometres": row["radial_median_micrometres"]})
        for row in read_csv(os.path.join(STORE, "runs", key, "rounds.csv")):
            rounds.append({"label": label, "run_key": key, "round": row["round"],
                           "loss_last": row["loss_last"],
                           "validation_endpoint_error_larger_of_x_and_y_median_micrometres":
                               row["validation_endpoint_error_larger_of_x_and_y_median_micrometres"]})
    write_csv("runs.csv", runs)
    write_csv("endpoint_by_momentum_band.csv", bands)
    write_csv("error_along_the_track.csv", along)
    write_csv("validation_per_round.csv", rounds)
    write_csv("references.csv", [{"label": a, "endpoint_radial_median_micrometres": b,
                                  "single_step_radial_median_micrometres": c, "source": d}
                                 for a, b, c, d in REFERENCES])


def style(ax, title, xlabel, ylabel):
    ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw():
    os.makedirs(FIGURES, exist_ok=True)
    runs = read_csv(os.path.join(RESULTS, "runs.csv"))
    panels = list(dict.fromkeys(r["panel"] for r in runs))
    labels = list(dict.fromkeys(r["label"] for r in runs))
    colour = dict(zip(labels, COLOURS))

    def rows_of(name, run):
        return [r for r in read_csv(os.path.join(RESULTS, name))
                if r["label"] == run["label"] and r["run_key"] == run["run_key"]
                and r.get("round_of_the_report", run["round_of_the_report"]) == run["round_of_the_report"]]

    def figure(file, title, xlabel, plot_one, marks=()):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels) + 2.6, 4.4),
                                 sharey=True, squeeze=False)
        for ax, panel in zip(axes[0], panels):
            for run in runs:
                if run["panel"] == panel:
                    plot_one(ax, run)
            for mark_panel, mark_label, value in marks:
                if mark_panel == panel:
                    ax.plot([len(BANDS) - 1], [value], marker="x", markersize=8, linestyle="none",
                            color=INK_2)
                    ax.annotate(mark_label, (len(BANDS) - 1, value), fontsize=6, color=INK_2,
                                xytext=(-6, 3), textcoords="offset points", ha="right")
            style(ax, panel, xlabel, "micrometres" if ax is axes[0][0] else "")
            ax.legend(frameon=False, fontsize=7)
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, file), dpi=150)
        plt.close(fig)

    def hollow(run):
        return run["state_of_the_run"] != "plateaued" or run["round_of_the_report"] != run["rounds_done"]

    def validation(ax, run):
        rows = rows_of("validation_per_round.csv", run)
        if hollow(run):
            return
        ax.plot([float(r["round"]) for r in rows],
                [float(r["validation_endpoint_error_larger_of_x_and_y_median_micrometres"]) for r in rows],
                color=colour[run["label"]], linewidth=1.2, label=run["label"])

    def by_band(ax, run):
        rows = {r["band"]: r for r in rows_of("endpoint_by_momentum_band.csv", run)}
        name = run["label"] + ("" if not hollow(run) else ", round %s (not plateaued)" % run["round_of_the_report"])
        ax.plot(range(len(BANDS)), [float(rows[b]["radial_median_micrometres"]) for b in BANDS],
                marker="o", markerfacecolor="white" if hollow(run) else colour[run["label"]],
                linestyle="--" if hollow(run) else "-", color=colour[run["label"]], linewidth=1.2,
                label=name)
        ax.set_xticks(range(len(BANDS)), [b.replace(" GeV", "") for b in BANDS], fontsize=7, rotation=30)

    def along(ax, run):
        if hollow(run):
            return
        rows = rows_of("error_along_the_track.csv", run)[1:]
        ax.plot([float(r["z_mm"]) for r in rows], [float(r["radial_median_micrometres"]) for r in rows],
                color=colour[run["label"]], linewidth=1.2, label=run["label"])

    figure("validation_per_round.png",
           "Validation endpoint error, larger of x and y, median, every round",
           "round", validation)
    figure("endpoint_by_momentum_band.png",
           "Endpoint radial error, test tracks, median per momentum band (GeV)", "", by_band,
           REFERENCE_MARKS)
    figure("error_along_the_track.png",
           "Radial error along the chain, median over the test tracks", "z (mm)", along)


if __name__ == "__main__":
    collect()
    draw()
