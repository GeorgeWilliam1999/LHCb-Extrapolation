"""Where the parity gates find the frozen code and the frozen files.

The package itself never inserts into the import path (WORKFLOW.md, section
3). The parity gates have to, because the frozen code was written to be found
that way. The insertion is confined to this file, and nothing here is changed
in the frozen folders: they are only read.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

# Importing the frozen code must not write into the frozen folders.
sys.dont_write_bytecode = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SINGLE_NETWORK_FOLDER = os.path.join(PROJECT_ROOT, "single_network_chain_discrete_approach")
EXACT_SOLVER_FOLDER = os.path.join(
    PROJECT_ROOT, "multi_network_chain_discrete_approach",
    "Block_C_step_size_and_stages", "C2_Exact_scheme_table")

TRACKS_FILE = os.path.join(
    SINGLE_NETWORK_FOLDER, "Block_E_single_network_chain", "E0_Track_dataset",
    "results", "tracks.npz")
STORED_EXACT_STATES_FOLDER = os.path.join(
    SINGLE_NETWORK_FOLDER, "Block_E_single_network_chain", "E2_Comparators", "results")


def frozen_module(name: str):
    """A module of the frozen `_shared` folder, for example `reference`."""
    if SINGLE_NETWORK_FOLDER not in sys.path:
        sys.path.insert(0, SINGLE_NETWORK_FOLDER)
    return importlib.import_module("_shared." + name)


def frozen_exact_solver():
    """The frozen `exact_solver.py` of the exact scheme table."""
    frozen_module("reference")              # so `_shared` is the frozen single copy
    if EXACT_SOLVER_FOLDER not in sys.path:
        sys.path.insert(0, EXACT_SOLVER_FOLDER)
    return importlib.import_module("exact_solver")


def needed_file(path: str) -> str:
    """The path, or a failure that names the missing file. A parity gate that
    cannot find its frozen file has not passed, so it fails; it does not skip."""
    if not os.path.exists(path):
        pytest.fail("the frozen file %s is missing, so this gate cannot be run" % path)
    return path


def frozen_tracks():
    """The frozen tracks on 257 planes."""
    return np.load(needed_file(TRACKS_FILE))


def stored_exact_states(number_of_steps: int, number_of_stages: int):
    """The exact scheme's states stored by the frozen chain, shape (n, N + 1, 5)."""
    name = "exact_N%03d_q%02d_states.npz" % (number_of_steps, number_of_stages)
    return np.load(needed_file(os.path.join(STORED_EXACT_STATES_FOLDER, name)))["states"]


def real_states(number: int, seed: int = 20260928):
    """`number` real states and the planes they lie on, drawn from the frozen
    validation tracks at planes all along the crossing."""
    tracks = frozen_tracks()
    states_on_planes = tracks["val_truth"]
    planes = tracks["planes"]
    generator = np.random.default_rng(seed)
    which_track = generator.integers(0, states_on_planes.shape[0], number)
    which_plane = generator.integers(0, states_on_planes.shape[1], number)
    return states_on_planes[which_track, which_plane].copy(), planes[which_plane].copy()


def identical(a, b) -> bool:
    """True when two arrays have the same shape and the same bits."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


SAMPLE_FILE = os.path.join(
    PROJECT_ROOT, "Data_generation_exploration", "Official_xdigi", "training_v2",
    "train_official_v2.npz")
ROUND_TRAINER_FOLDER = os.path.join(
    SINGLE_NETWORK_FOLDER, "Block_E_single_network_chain", "E1_Network_grid")

# The names of the frozen tracks file, and the names the package gives them.
FROZEN_NAME_OF_SPLIT = {"training": "train", "validation": "val", "test": "test"}
FROZEN_NAME_OF_ARRAY = {
    "start_state": "S0",
    "reference_states_on_planes": "truth",
    "reference_state_on_own_fibre_plane": "truth_zpost",
    "true_state_on_own_upstream_plane": "S_pre",
    "true_state_on_own_fibre_plane": "S_post",
    "own_upstream_plane_mm": "z_pre",
    "own_fibre_plane_mm": "z_post",
    "momentum_gev": "P",
    "pseudorapidity": "ETA",
    "particle_type": "PID",
    "event": "EVT",
    "particle_in_event": "MCKEY",
}


def frozen_round_trainer():
    """The frozen `train_network.py` of the self-chained networks."""
    frozen_module("reference")
    if ROUND_TRAINER_FOLDER not in sys.path:
        sys.path.insert(0, ROUND_TRAINER_FOLDER)
    return importlib.import_module("train_network")
