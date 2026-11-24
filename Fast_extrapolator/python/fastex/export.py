"""Flat binary files for the compiled program.

The compiled program reads no Python format. Every array is one file of raw
little-endian numbers (`.f64` float64, `.i64` int64) and every folder has a
`meta` file of lines `key value`. The numbers come from the classes of the
present package, so the compiled program starts from the same numbers.

    field maps   <exports>/field_<name>/   meta, grid.f64 [, cubic.f64]
                 grid.f64  : Bx, By, Bz in the unit of the map file, each
                             nz * ny * nx values, x fastest
                 cubic.f64 : the B-spline coefficients in tesla, same layout
    tracks       <exports>/tracks_<map>/   meta, start_state.f64 (n x 5),
                 reference_end_state.f64 (n x 5), momentum_gev.f64 (n),
                 flat_sample.i64 (1,000 row numbers, flat in momentum);
                 training_*.f64 (2,000 training tracks) and field_profile.f64
                 (rows of z, mean |B|, mean |By| on those tracks);
                 splits/<split>/ : every track of the training and of the
                 validation split, the same three arrays
    networks     <exports>/networks/<run key>/   meta, W<k>.f64, b<k>.f64,
                 scale_of_inputs.f64, [scale_of_outputs.f64], nodes.f64,
                 weights.f64, stage_matrix.f64

    Allen step   <exports>/allen_utt/   meta (the path of the .tab file),
                 UTT_META.f64 (19), UTTF.f64 (20)

    python -m fastex.export fields | tracks | splits | allen_utt | networks <run key> ...
"""
from __future__ import annotations

import os
import sys

import numpy as np

from fastex import FOLDER_OF_EXPORTS, STORE_OF_THE_PRESENT_RUNS, TRACK_SETS

# Field maps that the present package does not register. The magnet-down map
# of the simulation is the one the reference output of the Rec test was made
# with (conditions database, Magnet.yml, first interval of validity). The v6r1
# files are not in the list: the unit check of the present package refuses
# them (peak |By| in the file 0.071).
FOLDER_OF_MAP_FILES = "/cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/"
OTHER_MAP_FILES = {name: FOLDER_OF_MAP_FILES + "field.%s.%s.bin" % tuple(name.split("_"))
                   for name in ("v5r0_down", "v5r0_up", "v8r1_down")}

# The sample that is flat in momentum, as in the timing study of 2026-11-23.
SEED_OF_THE_FLAT_SAMPLE = 20261123
BIN_EDGES_GEV = np.arange(2.0, 42.0 + 1e-9, 4.0)
TRACKS_PER_BIN = 100


def flat_in_momentum(momentum_gev) -> np.ndarray:
    """Row numbers of 1,000 tracks: 100 in each 4 GeV bin from 2 to 42 GeV,
    drawn without replacement with a fixed seed, sorted."""
    rng = np.random.default_rng(SEED_OF_THE_FLAT_SAMPLE)
    p = np.asarray(momentum_gev)
    chosen = []
    for low, high in zip(BIN_EDGES_GEV[:-1], BIN_EDGES_GEV[1:]):
        inside = np.flatnonzero((p >= low) & (p < high))
        chosen.append(rng.choice(inside, TRACKS_PER_BIN, replace=False))
    return np.sort(np.concatenate(chosen))


def write_array(path, array, dtype=np.float64):
    np.ascontiguousarray(array, dtype=dtype).tofile(path)


def write_meta(path, pairs):
    with open(path, "w") as handle:
        for key, value in pairs:
            if isinstance(value, float):
                value = repr(float(value))
            handle.write("%s %s\n" % (key, value))


def folder(*parts) -> str:
    path = os.path.join(FOLDER_OF_EXPORTS, *parts)
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------ field maps

def export_field_map(name, field_map, cubic=None):
    """`field_map` a FieldMap of the present package; `cubic` the CubicFieldMap
    of the same file, or None."""
    out = folder("field_" + name)
    nx, ny, nz = (int(n) for n in field_map.number_of_points)
    write_array(os.path.join(out, "grid.f64"),
                np.stack([field_map.field_x, field_map.field_y, field_map.field_z]))
    pairs = [("name", name), ("file", field_map.path), ("nx", nx), ("ny", ny), ("nz", nz)]
    for axis, letter in enumerate("xyz"):
        pairs.append(("inverse_cell_size_" + letter, float(field_map.inverse_cell_size[axis])))
    for axis, letter in enumerate("xyz"):
        pairs.append(("lowest_corner_" + letter, float(field_map.lowest_corner[axis])))
    pairs.append(("scale_to_tesla", float(field_map.scale_to_tesla)))
    pairs.append(("has_cubic", int(cubic is not None)))
    if cubic is not None:
        write_array(os.path.join(out, "cubic.f64"), np.stack(cubic.coefficients))
    write_meta(os.path.join(out, "meta"), pairs)
    print("field map %s: %d x %d x %d points, %s" % (name, nx, ny, nz, field_map.description()),
          flush=True)


def export_fields():
    from rkpinn.equation_of_motion.field_map import FieldMap, load_field_map
    export_field_map("v8r1_up", load_field_map("v8r1_up"), load_field_map("v8r1_up_cubic"))
    for name, path in OTHER_MAP_FILES.items():
        export_field_map(name, FieldMap(path))


# ---------------------------------------------------------------------- tracks

def export_tracks():
    from rkpinn.run_record.store import Store
    from rkpinn.track_data.load_tracks import load_tracks
    store = Store(STORE_OF_THE_PRESENT_RUNS)
    sample = None
    for of_map, key in TRACK_SETS.items():
        tracks = load_tracks(store, key)
        test = tracks.test
        if sample is None:
            sample = flat_in_momentum(test.momentum_gev)
        out = folder("tracks_" + of_map)
        write_array(os.path.join(out, "start_state.f64"), test.start_state)
        write_array(os.path.join(out, "reference_end_state.f64"), test.reference_end_state)
        write_array(os.path.join(out, "momentum_gev.f64"), test.momentum_gev)
        write_array(os.path.join(out, "flat_sample.i64"), sample, np.int64)
        write_meta(os.path.join(out, "meta"), [
            ("track_set", key), ("split", "test"), ("map", of_map),
            ("field_map", tracks.description["settings"]["field_map"]),
            ("number_of_tracks", test.number_of_tracks),
            ("first_plane_mm", float(tracks.first_plane_mm)),
            ("last_plane_mm", float(tracks.last_plane_mm)),
            ("flat_sample", len(sample))])
        print("tracks %s (%s): %d test tracks, planes %r to %r mm"
              % (of_map, key, test.number_of_tracks, tracks.first_plane_mm,
                 tracks.last_plane_mm), flush=True)


def export_training_subset(number=2000):
    """2,000 training tracks of each track set, evenly spaced in the split, for
    choices that must not see the test tracks (a step plan, for example); and
    the profile of the field along z on them: for each of the planes of the
    track set, the mean of |B| and of |By| over the reference states."""
    from rkpinn import registry
    from rkpinn.run_record.store import Store
    from rkpinn.track_data.load_tracks import load_tracks
    store = Store(STORE_OF_THE_PRESENT_RUNS)
    for of_map, key in TRACK_SETS.items():
        tracks = load_tracks(store, key)
        training = tracks.training
        rows = np.linspace(0, training.number_of_tracks - 1, number).astype(np.int64)
        out = folder("tracks_" + of_map)
        write_array(os.path.join(out, "training_start_state.f64"), training.start_state[rows])
        write_array(os.path.join(out, "training_reference_end_state.f64"),
                    training.reference_end_state[rows])
        write_array(os.path.join(out, "training_momentum_gev.f64"), training.momentum_gev[rows])
        field_map = registry.component("field_map", tracks.description["settings"]["field_map"])()
        states = training.reference_states_on_planes[rows]            # (number, planes, 5)
        profile = np.empty((len(tracks.planes_mm), 3))
        for j, z in enumerate(tracks.planes_mm):
            bx, by, bz = field_map(states[:, j, 0], states[:, j, 1], np.full(len(rows), z))
            profile[j] = z, np.sqrt(bx * bx + by * by + bz * bz).mean(), np.abs(by).mean()
        write_array(os.path.join(out, "field_profile.f64"), profile)
        print("training subset %s: %d tracks; field profile on %d planes, peak mean |B| %.3f T at "
              "%.0f mm" % (of_map, len(rows), len(profile), profile[:, 1].max(),
                           profile[profile[:, 1].argmax(), 0]), flush=True)


def export_splits(splits=("training", "validation")):
    """Every track of the training and validation splits of each track set:
    <exports>/tracks_<map>/splits/<split>/start_state.f64, reference_end_state.f64
    and momentum_gev.f64. They are for the choices and fits of stage 2, which
    must not see the test tracks."""
    from rkpinn.run_record.store import Store
    from rkpinn.track_data.load_tracks import load_tracks
    store = Store(STORE_OF_THE_PRESENT_RUNS)
    for of_map, key in TRACK_SETS.items():
        tracks = load_tracks(store, key)
        for name in splits:
            split = tracks.split(name)
            out = folder("tracks_" + of_map, "splits", name)
            write_array(os.path.join(out, "start_state.f64"), split.start_state)
            write_array(os.path.join(out, "reference_end_state.f64"), split.reference_end_state)
            write_array(os.path.join(out, "momentum_gev.f64"), split.momentum_gev)
            write_meta(os.path.join(out, "meta"), [
                ("track_set", key), ("split", name), ("map", of_map),
                ("number_of_tracks", split.number_of_tracks)])
            print("split %s of the %s track set (%s): %d tracks"
                  % (name, of_map, key, split.number_of_tracks), flush=True)


# -------------------------------------------------------------------- networks

def network_and_layout(store, key):
    """The network of a run with the weights of its last round, and its layout."""
    from rkpinn.run_record.configuration import read_configuration
    from rkpinn.training.checkpoints import RunFolder
    from rkpinn.training.round_trainer import build_run
    run_folder = RunFolder(store, key)
    configuration = read_configuration(run_folder.path("configuration.yaml"))
    rounds = run_folder.state()["rounds_done"]
    run = build_run(configuration, store, run_folder.constants())
    run.network.load_state_dict(run_folder.weights_of_round(int(rounds)))
    run.network.eval()
    return run.network, run.layout, configuration, int(rounds)


def export_network(store, key):
    import torch
    network, layout, configuration, rounds = network_and_layout(store, key)
    out = folder("networks", key)
    kind = network.kind
    form = network.output_form
    of_map = "cubic" if configuration["equation_of_motion"]["field_map"].endswith("_cubic") \
        else "linear"
    pairs = [("run", key), ("round", rounds), ("kind", kind), ("map", of_map),
             ("field_map", configuration["equation_of_motion"]["field_map"]),
             ("tracks", configuration["tracks"]["key"]),
             ("width", network.width), ("depth", network.depth),
             ("activation", network.activation),
             ("output_form", form.name),
             ("first_plane_mm", float(layout.first_plane_mm)),
             ("last_plane_mm", float(layout.last_plane_mm)),
             ("number_of_steps", int(layout.number_of_steps)),
             ("number_of_stages", int(layout.number_of_stages)),
             ("step_length_mm", float(layout.step_length_mm))]
    per_track = bool(getattr(form, "scale_is_per_track", False))
    pairs.append(("scale", "per_track" if per_track else "fixed"))
    if not per_track:
        write_array(os.path.join(out, "scale_of_outputs.f64"), form.scale_of_outputs.numpy())
    write_array(os.path.join(out, "scale_of_inputs.f64"), network.scale_of_inputs.numpy())
    if kind == "whole_crossing_network":
        pairs += [("end_state", "predicted"), ("sweeps_inside", 0), ("states_emitted", 1)]
    else:
        pairs += [("end_state", network.end_state),
                  ("sweeps_inside", int(getattr(network, "sweeps_inside", 0))),
                  ("states_emitted", int(network.number_of_outputs)),
                  ("middle_of_start_planes_mm", float(network.middle_of_start_planes_mm)),
                  ("half_range_of_start_planes_mm", float(network.half_range_of_start_planes_mm))]
        tableau = network.tableau
        write_array(os.path.join(out, "nodes.f64"), tableau.nodes)
        write_array(os.path.join(out, "weights.f64"), tableau.weights)
        write_array(os.path.join(out, "stage_matrix.f64"), tableau.stage_matrix)
    layers = [m for m in network.body if isinstance(m, torch.nn.Linear)]
    pairs.append(("layers", len(layers)))
    for k, layer in enumerate(layers):
        pairs.append(("layer_%d" % k, "%d %d" % (layer.out_features, layer.in_features)))
        write_array(os.path.join(out, "W%d.f64" % k), layer.weight.detach().numpy())
        write_array(os.path.join(out, "b%d.f64" % k), layer.bias.detach().numpy())
    write_meta(os.path.join(out, "meta"), pairs)
    print("network %s: %s, %s, %d steps x %d stages, %d x %d, %s map, round %d"
          % (key, kind, form.name, layout.number_of_steps, layout.number_of_stages,
             network.depth, network.width, of_map, rounds), flush=True)
    return network, layout


def export_networks(keys):
    from rkpinn.run_record.store import Store
    store = Store(STORE_OF_THE_PRESENT_RUNS)
    for key in keys:
        export_network(store, key)


# ------------------------------------------------------ Gauss-Legendre tableaux

def export_tableaux(stages=(1, 2, 3, 4, 6, 8, 16)):
    """The nodes, weights and stage matrix of the Gauss-Legendre scheme, from
    the present package, for the compiled sweeps of the stage map."""
    from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
    for q in stages:
        tableau = gauss_legendre_tableau(q)
        out = folder("tableau_%d" % q)
        write_array(os.path.join(out, "nodes.f64"), tableau.nodes)
        write_array(os.path.join(out, "weights.f64"), tableau.weights)
        write_array(os.path.join(out, "stage_matrix.f64"), tableau.stage_matrix)
        write_meta(os.path.join(out, "meta"), [("number_of_stages", q)])
    print("tableaux of %s stages" % (stages,))


# ------------------------------------------------- the parametrised step of Allen

FOLDER_OF_KALMAN_PARAMETERS = ("/cvmfs/lhcb.cern.ch/lib/lhcb/PARAM/ParamFiles/v8r74/data/"
                               "ParametrizedKalmanFit")


def export_allen_utt():
    """The parameters that Allen v9r3 loads for its step from the UT to the
    SciFi: the tabulated polynomials of 25v0 (read by the compiled program from
    CVMFS) and UTT_META and UTTFParams of 25v1/params.json."""
    import json
    with open(os.path.join(FOLDER_OF_KALMAN_PARAMETERS, "25v1", "params.json")) as handle:
        parameters = json.load(handle)
    out = folder("allen_utt")
    meta = np.asarray(parameters["UTT_META"], dtype=np.float64).reshape(-1)
    uttf = np.asarray(parameters["UTTFParams"], dtype=np.float64).reshape(-1)
    assert meta.shape == (19,) and uttf.shape == (20,)
    write_array(os.path.join(out, "UTT_META.f64"), meta)
    write_array(os.path.join(out, "UTTF.f64"), uttf)
    write_meta(os.path.join(out, "meta"), [
        ("tab", os.path.join(FOLDER_OF_KALMAN_PARAMETERS, "25v0", "params_UTT_v0.tab")),
        ("json", os.path.join(FOLDER_OF_KALMAN_PARAMETERS, "25v1", "params.json")),
        ("first_plane_mm", float(meta[0])), ("last_plane_mm", float(meta[1])),
        ("lowest_momentum_mev", float(meta[2]))])
    print("parametrised step of Allen: %r mm to %r mm, PMIN %r MeV" % (meta[0], meta[1], meta[2]))


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "fields":
        export_fields()
    elif command == "allen_utt":
        export_allen_utt()
    elif command == "tableaux":
        export_tableaux()
    elif command == "training_subset":
        export_training_subset()
    elif command == "tracks":
        export_tracks()
    elif command == "splits":
        export_splits()
    elif command == "networks":
        export_networks(sys.argv[2:])
    else:
        raise SystemExit("fields, tracks, allen_utt or networks")
