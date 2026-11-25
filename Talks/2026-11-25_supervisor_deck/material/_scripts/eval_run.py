"""Score one finished run on a test split, writing only to _work/pred/.

    eval_run.py <key> <tracks_key or 'own'> <sweeps 0|1|2>

Uses the package code of the run's own commit (PYTHONPATH set by the caller).
Builds the run from its stored configuration and constants, loads the weights of
the last round, carries the test split, optionally applies Newton sweeps, carries
the end states to each particle's own fibre plane with the reference integrator,
and saves the states. Nothing is written to the store.
"""
import sys, os, time, json, copy
import numpy as np, torch
from rkpinn.run_record.configuration import read_configuration
from rkpinn.run_record.store import Store
from rkpinn.training.checkpoints import RunFolder
from rkpinn.training.round_trainer import build_run
from rkpinn.evaluation.three_references import carry_to_own_planes
from rkpinn.integrators.runge_kutta_sixth_order import REFERENCE_STEP_LENGTH_MM

key, tracks_key, max_sweeps = sys.argv[1], sys.argv[2], int(sys.argv[3])
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_work", "pred")
os.makedirs(OUT, exist_ok=True)
store = Store("/data/bfys/gscriven/rkpinn_store")
folder = RunFolder(store, key)
state = folder.state(); rnd = int(state["rounds_done"])
conf = read_configuration(folder.path("configuration.yaml"))
if tracks_key != "own":
    conf = copy.deepcopy(conf); conf["tracks"]["key"] = tracks_key
run = build_run(conf, store, folder.constants())
run.network.load_state_dict(folder.weights_of_round(rnd))
torch.set_num_threads(1)
tracks = run.tracks.split("test"); layout = run.layout
t0 = time.time()
with torch.no_grad():
    whole = run.network.whole_track(tracks.start_state, layout)
seconds = time.time() - t0
variants = [(0, whole)]
if max_sweeps:
    from rkpinn.evaluation.newton_sweep import one_sweep
    cur = whole
    for k in range(1, max_sweeps + 1):
        cur = one_sweep(cur, run.differentiable_equation).track; variants.append((k, cur))
for k, tr in variants:
    end = np.asarray(tr.final_state()) if hasattr(tr, "final_state") else np.asarray(tr.states_on_planes())[:, -1]
    if end.shape[1] == 4:
        end = np.concatenate([end, tracks.start_state[:, 4:5]], axis=1)
    own = carry_to_own_planes(run.equation_of_motion, end, layout.last_plane_mm, tracks.own_fibre_plane_mm,
                              step_length_of_the_integrator_mm=REFERENCE_STEP_LENGTH_MM)
    np.savez(os.path.join(OUT, f"{key}__{run.tracks.key}__s{k}.npz"), end=end, own=np.asarray(own),
             round=rnd, seconds_of_carrying=seconds, tracks=len(end), threads=1, map=conf["equation_of_motion"]["field_map"],
             commit=state["invocations"][-1]["commit"])
print(key, run.tracks.key, "round", rnd, "us/track %.1f" % (1e6 * seconds / len(tracks.start_state)), "done")
