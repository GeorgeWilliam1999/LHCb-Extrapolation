"""The ARKN fallback threshold, chosen on the VALIDATION tracks and scored once on the test tracks.

Rule: the smallest fallback fraction such that no validation track has a radial end-point error
above 2x the largest radial error of the ARKN on the validation tracks.

    python allen_nn/python/fallback_validation.py      writes allen_nn/results/fallback_validation.csv
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import an_tools as an  # noqa: E402

SPLITS = os.path.join(an.EXPORTS, "tracks_linear", "splits")
NETWORKS = ["linear_nystrom_12_2x64", "linear_nystrom_8_2x64"]
THRESHOLDS = [0.1, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.28, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
              1.8, 1.9, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]


def validation_tracks():
    f = lambda name: os.path.join(SPLITS, "validation", name)  # noqa: E731
    return (np.fromfile(f("start_state.f64")).reshape(-1, 5), np.fromfile(f("reference_end_state.f64")).reshape(-1, 5),
            np.fromfile(f("momentum_gev.f64")))


def radial(end, ref):
    return np.hypot(end[:, 0] - ref[:, 0], end[:, 1] - ref[:, 1]) * 1e3


def score(spec, s, ref):
    end, _, counts = an.run(spec, s, an.Z0, an.Z1)
    r = radial(end, ref)
    return dict(median=np.median(r), p95=np.percentile(r, 95), p999=np.percentile(r, 99.9), max=r.max(),
                fell_back=float((counts[:, 3] == 1).mean()), failed=int((counts[:, 3] == -1).sum()),
                reads=float(counts[:, 0].mean()))


def main():
    sv, rv, _ = validation_tracks()
    st, rt, _ = an.test_tracks()
    rows = []
    arkn_v, arkn_t = score("arkn", sv, rv), score("arkn", st, rt)
    bound = 2 * arkn_v["max"]
    rows.append(dict(method="ARKN", threshold_mm="", split="validation", **arkn_v))
    rows.append(dict(method="ARKN", threshold_mm="", split="test", **arkn_t))
    print("ARKN validation max %.1f um -> bound %.1f um; test max %.1f um" % (arkn_v["max"], bound, arkn_t["max"]))
    for net in NETWORKS:
        tool = "tool:" + os.path.join(an.WEIGHTS, net + ".json")
        scan = []
        for e in THRESHOLDS:
            v = score(tool + ":fallback=%g" % e, sv, rv)
            scan.append((e, v))
            rows.append(dict(method=net, threshold_mm=e, split="validation", **v))
            print("%s E=%-4g val: fell back %.2f%% median %.2f 95%% %.1f 99.9%% %.0f max %.0f %s" % (
                net, e, 100 * v["fell_back"], v["median"], v["p95"], v["p999"], v["max"],
                "ok" if v["max"] <= bound else ""), flush=True)
        ok = [(e, v) for e, v in scan if v["max"] <= bound]
        if not ok:
            print("%s: no threshold meets the rule" % net)
            continue
        e, v = min(ok, key=lambda ev: ev[1]["fell_back"])
        t = score(tool + ":fallback=%g" % e, st, rt)
        rows.append(dict(method=net, threshold_mm=e, split="test (chosen)", **t))
        print("%s CHOSEN E=%g: test fell back %.2f%% median %.2f 95%% %.1f 99.9%% %.0f max %.0f" % (
            net, e, 100 * t["fell_back"], t["median"], t["p95"], t["p999"], t["max"]), flush=True)
    path = os.path.join(an.FOLDER, "results", "fallback_validation.csv")
    with open(path, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
