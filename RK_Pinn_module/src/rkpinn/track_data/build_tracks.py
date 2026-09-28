"""Build a track set: real particles carried across the crossing with the
reference integrator, and written to the store.

What is done, in order:

 1. The rows of the sample that cross the magnet are read. The sample holds
    each crossing twice, once in each direction.
 2. The cuts are applied and counted: the plane before the magnet is a plane
    of the Upstream Tracker; both directions are present; the pseudorapidity
    and momentum ranges; electrons excluded; both directions still present.
 3. One row per particle is kept, the forward one, paired with its backward
    row, whose start is the particle's true state on the fibre tracker.
 4. Particles whose own planes lie within a window of the first and last
    plane of the crossing are kept.
 5. Each particle's true state on its own plane of the Upstream Tracker is
    carried to the first plane. This is the start state of the track.
 6. The start state is carried across the crossing, plane to plane, and the
    state is kept on every plane. It is then carried on to the particle's own
    plane of the fibre tracker.
 7. Tracks that leave the field map, or are not finite, are removed.
 8. The tracks are divided into training, validation and test by the split
    the sample already carries, which is by particle.

Every setting is an argument. None has a default.

Ported, with the arithmetic and the order of the rows unchanged, from

    E0_Track_dataset/build_tracks.py          (main, march_planes)
    _shared/prepare.py                        (magnet_leg_rows, _inside_map, _particle_key)
    _shared/reference.py                      (load_training)

Gates: tests/test_tracks_match_the_frozen_file.py.

To build the track set of the project, from `RK_Pinn_module`:

    PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python \\
        -m rkpinn.track_data.build_tracks \\
        --store /data/bfys/gscriven/rkpinn_store \\
        --sample ../Data_generation_exploration/Official_xdigi/training_v2/train_official_v2.npz \\
        --field-map v8r1_up --first-plane-mm 2648.2 --last-plane-mm 7826.0 \\
        --number-of-steps-between-planes 256 --window-mm 60 \\
        --pseudorapidity-range 2 5 --momentum-range-gev 1 200 --exclude-electrons yes \\
        --lowest-upstream-tracker-plane-mm 1500 --step-length-mm 0.1 --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from multiprocessing import Pool

import numpy as np

from rkpinn.equation_of_motion.field_map import (
    corners_of_field_map, hash_of_field_map_file, load_field_map)
from rkpinn.equation_of_motion.lhcb import LhcbEquationOfMotion
from rkpinn.integrators.runge_kutta_sixth_order import integrate_with_sixth_order
from rkpinn.run_record import manifest
from rkpinn.run_record.provenance import folder_of_the_module, provenance
from rkpinn.run_record.store import Store, hash_of_file, key_of_content

SPLITS = ("training", "validation", "test")
CODE_OF_SPLIT_IN_THE_SAMPLE = {"training": 0, "validation": 1, "test": 2}

# The sample names its kinds of leg by number. The leg across the magnet, from
# the last tracker plane before it to the first after it, is number 1.
LEG_ACROSS_THE_MAGNET = 1

# Electrons, by the particle numbering scheme.
PARTICLE_TYPE_OF_THE_ELECTRON = 11

COLUMNS_OF_THE_SAMPLE = ("X", "LEG", "P", "PID", "SPLIT", "EVT", "MCKEY", "ETA")

# What each array of a split holds. The names are the names in the file.
ARRAYS_OF_A_SPLIT = {
    "start_state": "the true state on the particle's own plane of the Upstream "
                   "Tracker, carried to the first plane; shape (n, 5)",
    "reference_states_on_planes": "the start state carried across the crossing "
                                  "with the reference integrator, on every plane; "
                                  "shape (n, number of planes, 5)",
    "reference_state_on_own_fibre_plane": "the reference track carried on from the "
                                          "last plane to the particle's own plane of "
                                          "the fibre tracker; shape (n, 5)",
    "true_state_on_own_upstream_plane": "the true state on the particle's own plane "
                                        "of the Upstream Tracker; shape (n, 5)",
    "true_state_on_own_fibre_plane": "the true state on the particle's own plane of "
                                     "the fibre tracker; shape (n, 5)",
    "own_upstream_plane_mm": "z of the particle's own plane of the Upstream Tracker",
    "own_fibre_plane_mm": "z of the particle's own plane of the fibre tracker",
    "momentum_gev": "the true momentum",
    "pseudorapidity": "the true pseudorapidity at the particle's origin",
    "particle_type": "the number of the particle type",
    "event": "the event the particle is in",
    "particle_in_event": "the number of the particle within its event",
}


@dataclass(frozen=True)
class SettingsOfTheTracks:
    """Every setting of a track set. None has a default."""

    field_map: str
    first_plane_mm: float
    last_plane_mm: float
    number_of_steps_between_planes: int
    window_mm: float
    pseudorapidity_range: tuple
    momentum_range_gev: tuple
    exclude_electrons: bool
    lowest_upstream_tracker_plane_mm: float
    step_length_mm: float

    @property
    def length_mm(self) -> float:
        return self.last_plane_mm - self.first_plane_mm

    @property
    def planes_mm(self) -> np.ndarray:
        n = self.number_of_steps_between_planes
        return self.first_plane_mm + np.arange(n + 1) * (self.length_mm / n)


def key_of_particle(event, particle_in_event):
    """One number per particle, from its event and its number in the event."""
    return event.astype(np.int64) * 10_000_000 + particle_in_event.astype(np.int64)


def read_rows_across_the_magnet(sample_file: str) -> dict:
    """The rows of the sample whose leg crosses the magnet, in both directions."""
    sample = np.load(os.path.abspath(sample_file))
    across = np.isin(sample["LEG"], [LEG_ACROSS_THE_MAGNET])
    return {column: sample[column][across] for column in COLUMNS_OF_THE_SAMPLE}


def select_rows(rows: dict, settings: SettingsOfTheTracks, report=print) -> dict:
    """Apply the cuts, in order, and count what each removes."""
    X = rows["X"].astype(np.float64)
    keep = np.ones(len(X), dtype=bool)
    cascade = []
    key = key_of_particle(rows["EVT"], rows["MCKEY"])

    def cut(name, mask, note=""):
        nonlocal keep
        before = int(keep.sum())
        keep = keep & mask
        after = int(keep.sum())
        cascade.append({"cut": name, "rows_in": before, "rows_removed": before - after,
                        "rows_out": after,
                        "particles_out": int(len(np.unique(key[keep]))), "note": note})
        report("  %-46s %7d -> %7d  (-%d)" % (name, before, after, before - after))

    z0, z1 = X[:, 5], X[:, 6]
    forward = z1 > z0
    plane_before = np.where(forward, z0, z1)

    cascade.append({"cut": "rows across the magnet in the sample, both directions",
                    "rows_in": len(X), "rows_removed": 0, "rows_out": len(X),
                    "particles_out": int(len(np.unique(key))), "note": ""})
    report("  %-46s %7d" % ("rows across the magnet", len(X)))

    def both_directions():
        kept = set(np.unique(key[keep & forward])) & set(np.unique(key[keep & ~forward]))
        return np.isin(key, np.fromiter(kept, dtype=np.int64, count=len(kept))
                       if kept else np.empty(0, dtype=np.int64))

    cut("the plane before the magnet is in the Upstream Tracker",
        plane_before > settings.lowest_upstream_tracker_plane_mm,
        "its z is above %g mm; below it the particle left no hit in the Upstream "
        "Tracker" % settings.lowest_upstream_tracker_plane_mm)
    cut("both directions present", both_directions(),
        "the forward and backward rows of a particle are a pair")
    low, high = settings.pseudorapidity_range
    cut("pseudorapidity between %g and %g" % (low, high),
        (rows["ETA"] > low) & (rows["ETA"] < high),
        "the true pseudorapidity at the particle's origin")
    low, high = settings.momentum_range_gev
    cut("momentum between %g and %g GeV" % (low, high),
        (rows["P"] > low) & (rows["P"] < high), "the true momentum")
    if settings.exclude_electrons:
        cut("electrons excluded",
            np.abs(rows["PID"]) != PARTICLE_TYPE_OF_THE_ELECTRON, "")
    cut("both directions still present", both_directions(), "")

    return {
        "state": X[keep, :5], "start_plane_mm": z0[keep], "end_plane_mm": z1[keep],
        "forward": forward[keep],
        "momentum_gev": rows["P"][keep].astype(np.float64),
        "pseudorapidity": rows["ETA"][keep].astype(np.float64),
        "particle_type": rows["PID"][keep], "event": rows["EVT"][keep],
        "particle_in_event": rows["MCKEY"][keep],
        "split_in_the_sample": rows["SPLIT"][keep],
        "cascade": cascade,
    }


def _carry_over_the_planes(arguments):
    """The reference integrator through every plane, for one share of the
    start states. Run in a worker."""
    start_states, settings = arguments
    equation = LhcbEquationOfMotion(load_field_map(settings.field_map))
    planes = settings.planes_mm
    on_planes = np.empty((len(start_states), len(planes), 5))
    on_planes[:, 0] = start_states
    current = start_states.copy()
    for k in range(1, len(planes)):
        current = integrate_with_sixth_order(
            equation, current, planes[k - 1], planes[k],
            step_length_mm=settings.step_length_mm)
        on_planes[:, k] = current
    return on_planes


def carry_over_the_planes(start_states, settings: SettingsOfTheTracks, workers: int):
    shares = np.array_split(
        np.arange(len(start_states)),
        max(1, min(workers, len(start_states) // 200 or 1)))
    if workers <= 1 or len(shares) == 1:
        return _carry_over_the_planes((start_states, settings))
    with Pool(len(shares)) as pool:
        parts = pool.map(_carry_over_the_planes,
                         [(start_states[share], settings) for share in shares])
    return np.concatenate(parts, axis=0)


def stays_inside_the_field_map(states_on_planes, field_map):
    """True for the tracks whose x and y stay inside the map on every plane."""
    lowest, highest = corners_of_field_map(field_map)
    outside = ((states_on_planes[:, :, 0] < lowest[0])
               | (states_on_planes[:, :, 0] > highest[0])
               | (states_on_planes[:, :, 1] < lowest[1])
               | (states_on_planes[:, :, 1] > highest[1])).any(axis=1)
    return ~outside


def build_arrays(sample_file: str, settings: SettingsOfTheTracks, workers: int,
                 report=print, most_particles=None):
    """The arrays of a track set and the count of every cut.

    most_particles  keep only this many particles after the window cut; for
                    the gates, which rebuild a small part. None keeps all.
    Returns (arrays, cascade, counts).
    """
    field_map = load_field_map(settings.field_map)
    equation = LhcbEquationOfMotion(field_map)
    selected = select_rows(read_rows_across_the_magnet(sample_file), settings, report)
    cascade = list(selected["cascade"])

    key = key_of_particle(selected["event"], selected["particle_in_event"])
    forward = selected["forward"]
    backward_row_of = {k: i for i, k in zip(np.flatnonzero(~forward), key[~forward])}
    fi = np.flatnonzero(forward)
    bi = np.array([backward_row_of[k] for k in key[fi]])
    own_upstream = selected["start_plane_mm"][fi]
    own_fibre = selected["end_plane_mm"][fi]
    assert (np.allclose(selected["start_plane_mm"][bi], own_fibre)
            and np.allclose(selected["end_plane_mm"][bi], own_upstream)), \
        "the backward row of a particle must run from its fibre plane to its upstream plane"
    cascade.append({"cut": "forward rows only", "rows_in": int(len(key)),
                    "rows_removed": int(len(key) - len(fi)), "rows_out": int(len(fi)),
                    "particles_out": int(len(fi)), "note": "one row per particle"})

    Z0, Z1, window = settings.first_plane_mm, settings.last_plane_mm, settings.window_mm
    near = (np.abs(own_upstream - Z0) < window) & (np.abs(own_fibre - Z1) < window)
    cascade.append({"cut": "own planes within %g mm of the first and last plane" % window,
                    "rows_in": int(len(fi)), "rows_removed": int((~near).sum()),
                    "rows_out": int(near.sum()), "particles_out": int(near.sum()),
                    "note": "first plane %.1f mm, last plane %.1f mm" % (Z0, Z1)})
    report("  %-46s %7d -> %7d  (-%d)" % ("own planes near the crossing", len(fi),
                                          near.sum(), (~near).sum()))
    fi, bi = fi[near], bi[near]
    own_upstream, own_fibre = own_upstream[near], own_fibre[near]
    if most_particles is not None:
        fi, bi = fi[:most_particles], bi[:most_particles]
        own_upstream, own_fibre = own_upstream[:most_particles], own_fibre[:most_particles]
    true_upstream = selected["state"][fi]
    true_fibre = selected["state"][bi]
    n = len(fi)

    step = settings.step_length_mm
    start_state = integrate_with_sixth_order(
        equation, true_upstream, own_upstream, Z0, step_length_mm=step)
    on_planes = carry_over_the_planes(start_state, settings, workers)
    on_own_fibre = integrate_with_sixth_order(
        equation, on_planes[:, -1], Z1, own_fibre, step_length_mm=step)

    finite = np.isfinite(on_planes).all(axis=(1, 2)) & np.isfinite(on_own_fibre).all(axis=1)
    keep = finite & stays_inside_the_field_map(on_planes, field_map)
    cascade.append({"cut": "the reference track stays inside the field map",
                    "rows_in": int(n), "rows_removed": int((~keep).sum()),
                    "rows_out": int(keep.sum()), "particles_out": int(keep.sum()),
                    "note": "x and y inside the map on every plane, all finite"})
    report("  %-46s %7d -> %7d  (-%d)" % ("inside the field map", n, keep.sum(),
                                          (~keep).sum()))

    columns = {
        "start_state": start_state,
        "reference_states_on_planes": on_planes,
        "reference_state_on_own_fibre_plane": on_own_fibre,
        "true_state_on_own_upstream_plane": true_upstream,
        "true_state_on_own_fibre_plane": true_fibre,
        "own_upstream_plane_mm": own_upstream,
        "own_fibre_plane_mm": own_fibre,
        "momentum_gev": selected["momentum_gev"][fi],
        "pseudorapidity": selected["pseudorapidity"][fi],
        "particle_type": selected["particle_type"][fi],
        "event": selected["event"][fi],
        "particle_in_event": selected["particle_in_event"][fi],
    }
    assert set(columns) == set(ARRAYS_OF_A_SPLIT)
    columns = {name: np.asarray(values)[keep] for name, values in columns.items()}
    split = np.asarray(selected["split_in_the_sample"][fi])[keep]

    arrays = {"planes_mm": settings.planes_mm}
    counts = {}
    for name in SPLITS:
        members = np.flatnonzero(split == CODE_OF_SPLIT_IN_THE_SAMPLE[name])
        counts[name] = int(len(members))
        for column, values in columns.items():
            arrays["%s.%s" % (name, column)] = values[members]
    return arrays, cascade, counts


def name_of_sample_in_the_record(sample_file: str) -> str:
    """The sample's path from the project folder; never an absolute path."""
    project = os.path.dirname(folder_of_the_module())
    path = os.path.abspath(sample_file)
    if path.startswith(project + os.sep):
        return os.path.relpath(path, project)
    return os.path.basename(path)


def build_tracks(store: Store, sample_file: str, settings: SettingsOfTheTracks,
                 workers: int, allow_uncommitted_changes: bool = False,
                 report=print, most_particles=None) -> str:
    """Build a track set and write it to the store. Returns its key.

    most_particles  None builds the whole track set. A number builds a part,
                    for the gates, and the record says so.
    """
    record = provenance(allow_uncommitted_changes)      # refuses before any work
    store.create()
    arrays, cascade, counts = build_arrays(sample_file, settings, workers, report,
                                           most_particles)
    key = key_of_content(arrays)
    store.refuse_if_present(store.folder_of_track_set(key))
    os.makedirs(store.folder_of_track_set(key))
    np.savez_compressed(store.file_of_tracks(key), **arrays)

    settings_in_the_record = asdict(settings)
    settings_in_the_record["pseudorapidity_range"] = list(settings.pseudorapidity_range)
    settings_in_the_record["momentum_range_gev"] = list(settings.momentum_range_gev)
    description = {
        "what": "real particles carried across the crossing with the reference "
                "integrator, on every plane, with their true states on their own planes",
        "key": key,
        "key_is": "the first %d characters of the sha256 hash of the arrays" % len(key),
        "settings": settings_in_the_record,
        "whole_track_set": most_particles is None,
        "most_particles": most_particles,
        "length_mm": settings.length_mm,
        "number_of_planes": int(len(settings.planes_mm)),
        "field_map": {"name": settings.field_map,
                      "hash_of_file": hash_of_field_map_file(settings.field_map)},
        "sample": {"file": name_of_sample_in_the_record(sample_file),
                   "hash_of_file": hash_of_file(sample_file)},
        "reference": "Butcher's explicit method with seven stages and order six, at a "
                     "fixed step of %g mm, carried plane to plane" % settings.step_length_mm,
        "split": "by particle, as the sample carries it",
        "counts": counts,
        "cuts": cascade,
        "arrays": {"planes_mm": "z of every plane",
                   **{"<split>." + name: meaning
                      for name, meaning in ARRAYS_OF_A_SPLIT.items()}},
        "state": "(x in mm, y in mm, slope in x, slope in y, charge over momentum)",
        "provenance": record,
    }
    with open(store.description_of_tracks(key), "w") as handle:
        json.dump(description, handle, indent=1)

    manifest.add_track_set(store, {
        "key": key, "created": record["created"],
        "number_of_training_tracks": counts["training"],
        "number_of_validation_tracks": counts["validation"],
        "number_of_test_tracks": counts["test"],
        "number_of_planes": int(len(settings.planes_mm)),
        "first_plane_mm": settings.first_plane_mm,
        "last_plane_mm": settings.last_plane_mm,
        "field_map": settings.field_map,
        "hash_of_field_map_file": description["field_map"]["hash_of_file"],
        "hash_of_sample_file": description["sample"]["hash_of_file"],
        "reference_step_length_mm": settings.step_length_mm,
        "package_version": record["package_version"], "commit": record["commit"],
        "traceable_to_the_commit": record["traceable_to_the_commit"],
        "machine": record["machine"],
    })
    report("wrote the track set %s: %r" % (key, counts))
    return key


def _yes_or_no(text: str) -> bool:
    if text not in ("yes", "no"):
        raise argparse.ArgumentTypeError("say yes or no")
    return text == "yes"


def main(arguments=None):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description="Build a track set into the store.")
    parser.add_argument("--store", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--field-map", required=True)
    parser.add_argument("--first-plane-mm", type=float, required=True)
    parser.add_argument("--last-plane-mm", type=float, required=True)
    parser.add_argument("--number-of-steps-between-planes", type=int, required=True)
    parser.add_argument("--window-mm", type=float, required=True)
    parser.add_argument("--pseudorapidity-range", type=float, nargs=2, required=True)
    parser.add_argument("--momentum-range-gev", type=float, nargs=2, required=True)
    parser.add_argument("--exclude-electrons", type=_yes_or_no, required=True)
    parser.add_argument("--lowest-upstream-tracker-plane-mm", type=float, required=True)
    parser.add_argument("--step-length-mm", type=float, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--allow-uncommitted-changes", action="store_true",
                        help="build although the package has uncommitted changes; "
                             "the record then says it cannot be traced")
    a = parser.parse_args(arguments)
    settings = SettingsOfTheTracks(
        field_map=a.field_map, first_plane_mm=a.first_plane_mm,
        last_plane_mm=a.last_plane_mm,
        number_of_steps_between_planes=a.number_of_steps_between_planes,
        window_mm=a.window_mm, pseudorapidity_range=tuple(a.pseudorapidity_range),
        momentum_range_gev=tuple(a.momentum_range_gev),
        exclude_electrons=a.exclude_electrons,
        lowest_upstream_tracker_plane_mm=a.lowest_upstream_tracker_plane_mm,
        step_length_mm=a.step_length_mm)
    return build_tracks(Store(a.store), a.sample, settings, a.workers,
                        a.allow_uncommitted_changes)


if __name__ == "__main__":
    main()
