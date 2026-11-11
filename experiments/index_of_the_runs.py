"""Write RUNS.md: one row per run in the store, with the experiments that name it.

Reads the store only; writes nothing to it. From this folder:

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python index_of_the_runs.py \
        /data/bfys/gscriven/rkpinn_store > RUNS.md
"""
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SUFFIXES = (".md", ".txt", ".csv", ".sh", ".sub", ".yaml")


def text_of_each_experiment():
    texts = {}
    for name in sorted(os.listdir(HERE)):
        folder = os.path.join(HERE, name)
        if not os.path.isdir(folder):
            continue
        parts = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(SUFFIXES):
                    with open(os.path.join(root, f), errors="ignore") as handle:
                        parts.append(handle.read())
        texts[name] = "\n".join(parts)
    return texts


def latest_report(run_folder):
    reports = os.path.join(run_folder, "reports")
    if not os.path.isdir(reports):
        return 0, ""
    names = sorted(os.listdir(reports))
    for name in reversed(names):
        summary = os.path.join(reports, name, "summary.json")
        if os.path.exists(summary):
            with open(summary) as handle:
                median = json.load(handle)["endpoint_radial_median_micrometres"]
            round_number = int(name.split("_")[1])
            return len(names), f"{median:.3g} (round {round_number})"
    return len(names), ""


def main(store):
    runs = os.path.join(store, "runs")
    texts = text_of_each_experiment()
    rows = []
    for key in sorted(os.listdir(runs)):
        folder = os.path.join(runs, key)
        with open(os.path.join(folder, "configuration.yaml")) as handle:
            configuration = yaml.safe_load(handle)
        with open(os.path.join(folder, "state.json")) as handle:
            state = json.load(handle)
        with open(os.path.join(folder, "provenance.json")) as handle:
            provenance = json.load(handle)
        network = configuration.get("network", {})
        steps = configuration["steps"]
        number_of_reports, latest = latest_report(folder)
        experiments = [name for name, text in texts.items() if key in text]
        rows.append([
            f"`{key}`",
            provenance.get("created", "")[:10],
            ", ".join(experiments) or "none names it",
            network.get("kind", ""),
            network.get("output_form", "") or "",
            str(network.get("width", "")),
            configuration.get("loss", {}).get("name", ""),
            configuration["training"]["optimiser"]["name"],
            f"{steps['number_of_steps']}/{steps['number_of_stages']}",
            str(configuration.get("seed", "")),
            f"{state.get('state', '')}, {state.get('rounds_done', '')} rounds",
            str(number_of_reports),
            latest,
        ])
    rows.sort(key=lambda row: (row[2], row[1], row[0]))
    header = ["Key", "Created", "Experiment", "Network", "Output form", "Width", "Loss",
              "Optimiser", "Steps/stages", "Seed", "State", "Reports", "Test endpoint, radial median (um), latest report"]
    print("# Runs in the store\n")
    print(f"Written by `index_of_the_runs.py` from `{store}`. {len(rows)} runs. "
          "Do not edit by hand; run the script again.\n")
    print("A run that no experiment folder names is listed with \"none names it\". "
          "An empty last column means the run has no standard report; some experiments score their runs with their own analysis scripts.\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
