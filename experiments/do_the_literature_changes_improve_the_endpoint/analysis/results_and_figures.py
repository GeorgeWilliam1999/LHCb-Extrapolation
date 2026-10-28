"""The results tables and figure of the literature tests, read from the store.

    results_and_figures.py

The tests were stopped after 3 to 6 rounds (state "interrupted" in the store), so no arm has a
standard report. Each arm is compared with its baseline at the same round, on the validation
error that every round records (rounds.csv). Round 1 of the fixed-draw arms is round 1 of the
baseline, which checks the comparison. Computes nothing new; writes nothing to the store.

Writes
    results/arms_against_baselines.csv      one row per arm: its last round against the baseline
    results/validation_per_round.csv        the validation error and the loss of every round
    results/baselines_on_the_test_split.csv the standard reports of the baselines that exist
    results/best_round_and_snapshot_average.csv
                                            change 2, copied from the tranche-1 analysis
    figures/validation_per_round.png
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
TRANCHE_1 = os.path.join(HERE, "..", "..", "..", "analysis", "tranche1_2026-10-28", "results",
                         "a_best_round_and_snapshot_average.csv")
VALIDATION = "validation_endpoint_error_larger_of_x_and_y_median_micrometres"

BASELINES = {"direct states": "49e5e2450eaa", "straight line, one global scale": "9cef9c160b7d"}
# change, form, arm, run key
ARMS = [
    (1, "direct states", "fixed draw, own predictions after round 1", "23f3bc73c61b"),
    (1, "direct states", "fixed draw, reference states of round 1", "c594bdddd98f"),
    (4, "direct states", "self-scaled Broyden", "f4c86e05eafe"),
    (1, "straight line, one global scale", "fixed draw, own predictions after round 1", "1212e5c801c2"),
    (1, "straight line, one global scale", "fixed draw, reference states of round 1", "68a9f0f91e9e"),
    (4, "straight line, one global scale", "self-scaled Broyden", "9a3df286d5fb"),
]
# the reference categorical order of the dataviz palette, as in the step-length experiment
COLOURS = {"fixed draw, own predictions after round 1": "#2a78d6",
           "fixed draw, reference states of round 1": "#1baf7a",
           "self-scaled Broyden": "#eb6834"}
INK, INK_2, GRAY = "#0b0b0b", "#52514e", "#8a8985"


def read_csv(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rounds_of(key):
    return {int(r["round"]): r for r in read_csv(os.path.join(STORE, "runs", key, "rounds.csv"))}


def collect():
    os.makedirs(RESULTS, exist_ok=True)
    arms, per_round = [], []
    for form, key in BASELINES.items():
        for number, row in sorted(rounds_of(key).items()):
            per_round.append({"form": form, "arm": "baseline", "run_key": key, "round": number,
                              "loss_last": row["loss_last"], VALIDATION: row[VALIDATION]})
    for change, form, arm, key in ARMS:
        state = json.load(open(os.path.join(STORE, "runs", key, "state.json")))
        mine, base = rounds_of(key), rounds_of(BASELINES[form])
        last = max(mine)
        for number, row in sorted(mine.items()):
            per_round.append({"form": form, "arm": arm, "run_key": key, "round": number,
                              "loss_last": row["loss_last"], VALIDATION: row[VALIDATION]})
        arm_value, base_value = float(mine[last][VALIDATION]), float(base[last][VALIDATION])
        arms.append({
            "change": change, "form": form, "arm": arm, "run_key": key,
            "state_of_the_run": state["state"], "rounds_done": state["rounds_done"],
            "baseline_run_key": BASELINES[form],
            "round_one_equals_the_baseline": mine[1][VALIDATION] == base[1][VALIDATION],
            "validation_at_the_last_round_micrometres": arm_value,
            "baseline_validation_at_the_same_round_micrometres": base_value,
            "arm_over_baseline": arm_value / base_value,
            "baseline_validation_at_its_last_round_micrometres": float(base[max(base)][VALIDATION]),
            "baseline_rounds_done": max(base),
            "loss_last_at_the_last_round": mine[last]["loss_last"],
            "baseline_loss_last_at_the_same_round": base[last]["loss_last"],
        })
    write_csv("arms_against_baselines.csv", arms)
    write_csv("validation_per_round.csv", per_round)

    reports = []
    for form, key in BASELINES.items():
        for folder in sorted(glob.glob(os.path.join(STORE, "runs", key, "reports", "round_*"))):
            summary = json.load(open(os.path.join(folder, "summary.json")))
            reports.append({
                "form": form, "run_key": key, "round": summary["round"],
                "round_asked_for": summary.get("round_asked_for", "last"),
                "report": os.path.relpath(folder, STORE),
                "endpoint_radial_median_micrometres": summary["endpoint_radial_median_micrometres"],
                "single_step_radial_median_micrometres": summary["single_step_radial_median_micrometres"],
            })
    write_csv("baselines_on_the_test_split.csv", reports)

    rows = read_csv(TRANCHE_1)
    for row in rows:
        row["source"] = "analysis/tranche1_2026-10-28/results/a_best_round_and_snapshot_average.csv"
    write_csv("best_round_and_snapshot_average.csv", rows)


def draw():
    os.makedirs(FIGURES, exist_ok=True)
    rows = read_csv(os.path.join(RESULTS, "validation_per_round.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), squeeze=False)
    for ax, form in zip(axes[0], BASELINES):
        mine = [r for r in rows if r["form"] == form]
        base = [r for r in mine if r["arm"] == "baseline" and int(r["round"]) <= 12]
        ax.plot([int(r["round"]) for r in base], [float(r[VALIDATION]) for r in base], color=GRAY,
                linestyle="--", linewidth=1.4, label="baseline %s (L-BFGS, fresh draw)" % BASELINES[form])
        for arm in COLOURS:
            these = [r for r in mine if r["arm"] == arm]
            ax.plot([int(r["round"]) for r in these], [float(r[VALIDATION]) for r in these],
                    marker="o", markersize=3, color=COLOURS[arm], linewidth=1.4, label=arm)
        ax.set_yscale("log")
        ax.set_title(form, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("round", color=INK_2)
        ax.set_ylabel("micrometres" if ax is axes[0][0] else "", color=INK_2)
        ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Validation endpoint error, larger of x and y, median over the validation tracks; "
                 "the tests stopped after 3 to 6 rounds", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "validation_per_round.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    collect()
    draw()
