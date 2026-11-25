"""Items 10 and 11: state of the compiled timing test (fast extrapolator) and every open run. Reads only."""
import json, os, glob, subprocess, csv, datetime
import numpy as np, pandas as pd, yaml
from style import OUT, STORE
FX = "/data/bfys/gscriven/LHCb_Extrapolation_Project/Fast_extrapolator/stage_2_training"
FXS = "/data/bfys/gscriven/fast_extrapolator_store/runs"
now = datetime.datetime.utcnow()
q = subprocess.run(["condor_q", "gscriven", "-af:j", "JobStatus", "Args"], capture_output=True, text=True, timeout=120).stdout
queued = {}
for line in q.splitlines():
    parts = line.split(None, 2)
    if len(parts) == 3:
        for tok in parts[2].split():
            if tok.endswith(".yaml"): queued[os.path.basename(tok)[:-5]] = {"1": "idle", "2": "running", "5": "held"}.get(parts[1], parts[1])

# ---- item 10
measured = pd.read_csv(f"{FX}/results/runs.csv").set_index("label")
rows = []
for d in sorted(glob.glob(FXS + "/*")):
    name = os.path.basename(d)
    if name.startswith("gate_"): continue
    s = json.load(open(d + "/state.json")); c = yaml.safe_load(open(d + "/configuration.yaml"))
    v = s.get("validation_medians_um", [])
    ended = s.get("ended")
    st = ("finished (" + ended + ")") if ended and name not in queued else ("training" if name in queued else "stopped, not in queue")
    if ended and name in queued: st = "training (extended after " + ended + ")"
    m = measured.loc[name] if name in measured.index else None
    rows.append(dict(section="stage-2 network", name=name, reference_map=name.split("_")[0], state=st, in_queue=queued.get(name, ""),
                     rounds_done=s.get("rounds_done"), last_validation_median_um=v[-1] if v else None, best_validation_median_um=min(v) if v else None,
                     test_round_measured=(int(m["rounds"]) if m is not None else None),
                     test_median_um=(m["radial_median_um"] if m is not None else None), test_p95_um=(m["radial_95_um"] if m is not None else None),
                     worst_band_median_um=(m["worst_band_median_um"] if m is not None else None),
                     us_per_track=(m["ns_median"] / 1000 if m is not None else None), host=(m["host"] if m is not None else ""),
                     final="no: test score at round %s, run %s" % (int(m["rounds"]), "still training" if "training" in st else "ended later") if m is not None else "no: not scored yet"))
pb = pd.read_csv(f"{FX}/results/polynomial_baseline.csv")
for _, r in pb.iterrows():
    sec = "LHCb method" if r["family"] == "LHCb" else "no network"
    rows.append(dict(section=sec, name=r["label"], reference_map=r["reference"], state="finished", rounds_done=None,
                     test_median_um=r["radial_median_um"], test_p95_um=r["radial_95_um"], worst_band_median_um=r["worst_band_median_um"],
                     us_per_track=r["ns_median"] / 1000, host=r["host"], final="yes (timing on stbc-i2, shared login node)"))
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "timing_test_state.csv"), index=False)
def fmt(x, f="%.3g"):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f % x
with open(os.path.join(OUT, "timing_test_state.md"), "w") as f:
    f.write("# Compiled timing test (fast extrapolator), state at %s UTC\n\n" % now.strftime("%Y-%m-%d %H:%M"))
    f.write("Test tracks: 13,447 of the cubic or the linear track set, compiled program `fx`, single precision for the networks, one core of stbc-i2. "
            "Errors are the endpoint radial error against the reference of the named map (µm). Times are the median ns per track over 15 repeats, shown in µs. "
            "Stage 0/1 numbers are final; every stage-2 network number is NOT final (runs still training or extended; test scores are from the round given, measured 2026-11-25 08:35 UTC).\n\n")
    for sec in ("LHCb method", "no network", "stage-2 network"):
        s = d[d.section == sec]
        f.write("## %s\n\n| name | map | state | rounds | last val. median | test median | test 95 %% | worst band median | µs/track |\n|---|---|---|---|---|---|---|---|---|\n" % sec)
        for _, r in s.iterrows():
            f.write("| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" % (r["name"], r["reference_map"], r["state"], fmt(r["rounds_done"], "%d"),
                    fmt(r["last_validation_median_um"]), fmt(r["test_median_um"]) + ("" if sec != "stage-2 network" or not r["test_median_um"] == r["test_median_um"] else " (round %d, provisional)" % r["test_round_measured"]),
                    fmt(r["test_p95_um"]), fmt(r["worst_band_median_um"]), fmt(r["us_per_track"])))
        f.write("\n")
    f.write("Source: Fast_extrapolator/stage_2_training/results/runs.csv and polynomial_baseline.csv, /data/bfys/gscriven/fast_extrapolator_store/runs/*/state.json, condor_q.\n")

# ---- item 11
TS = pd.read_csv("/data/bfys/gscriven/worktrees/cubic_map_standard/experiments/do_the_headline_results_hold_on_the_cubic_map/analysis/results/test_scores.csv").set_index("run_key")["configuration"].to_dict()
open_rows = []
for p in sorted(glob.glob(STORE + "/runs/*/state.json")):
    k = p.split("/")[-2]; s = json.load(open(p))
    if s["state"] != "training": continue
    c = yaml.safe_load(open(p.replace("state.json", "configuration.yaml")))
    rr = list(csv.DictReader(open(p.replace("state.json", "rounds.csv"))))
    secs = [float(r["seconds_of_training"]) + float(r.get("seconds_of_validation") or 0) for r in rr[-5:]]
    hpr = np.mean(secs) / 3600 if secs else None
    cap = s.get("rounds_at_most")
    exp = None
    for e in glob.glob("/data/bfys/gscriven/rkpinn_code/*/experiments/*/configurations/*.yaml"):
        pass
    n = c["network"]; nm = "%s, %s, %s, %d steps x %d stages, seed %s" % (n["kind"], n.get("output_form") or "", c["training"]["optimiser"]["name"], c["steps"]["number_of_steps"], c["steps"]["number_of_stages"], c.get("seed"))
    open_rows.append(dict(system="rkpinn store", experiment=("do_the_headline_results_hold_on_the_cubic_map" if k in TS else "can_a_label_free_path_network_reach_the_exact_scheme"), configuration=TS.get(k, ""), run_key=k, name=nm.replace("_", " "), map=c["equation_of_motion"]["field_map"], rounds_done=s["rounds_done"], cap=cap,
                          last_validation_larger_xy_um=rr[-1]["validation_endpoint_error_larger_of_x_and_y_median_micrometres"] if rr else "",
                          hours_per_round=hpr, latest_end_at_cap_utc=(now + datetime.timedelta(hours=hpr * (cap - s["rounds_done"]))).strftime("%Y-%m-%d") if hpr and cap else "",
                          note="ends earlier at a validation plateau (rule: 10-round window, 5 %, 3 rounds held)"))
for _, r in d[d.section == "stage-2 network"].iterrows():
    if "training" in r["state"] or r["in_queue"]:
        open_rows.append(dict(system="fast extrapolator stage 2", experiment="Fast_extrapolator stage 2", configuration=r["name"], run_key=r["name"], name=r["name"].replace("_", " "), map=r["reference_map"], rounds_done=r["rounds_done"], cap=1000,
                              last_validation_larger_xy_um=r["last_validation_median_um"], hours_per_round=None, latest_end_at_cap_utc="",
                              note="radial validation median; stop rule 30 rounds, 1 %, cap 1,000; queue: " + str(r["in_queue"])))
o = pd.DataFrame(open_rows); o.to_csv(os.path.join(OUT, "open_runs.csv"), index=False)
print(d[["section","name","state","rounds_done","test_median_um","us_per_track"]].to_string()); print(o[["system","run_key","rounds_done","cap","latest_end_at_cap_utc"]].to_string())
