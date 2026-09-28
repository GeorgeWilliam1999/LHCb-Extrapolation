"""Where the gates find the store, and what the project's track set is.

Not a gate. The package never holds the location of the store; the gates
need one, so it is here. Set the environment variable RKPINN_STORE to point
the gates at another store.
"""
from __future__ import annotations

import os

import pytest

from rkpinn.run_record.store import Store

AGREED_LOCATION_OF_THE_STORE = "/data/bfys/gscriven/rkpinn_store"

# The settings of the project's track set: the tracks on 257 planes.
SETTINGS_OF_THE_TRACKS = dict(
    field_map="v8r1_up", first_plane_mm=2648.2, last_plane_mm=7826.0,
    number_of_steps_between_planes=256, window_mm=60.0,
    pseudorapidity_range=(2.0, 5.0), momentum_range_gev=(1.0, 200.0),
    exclude_electrons=True, lowest_upstream_tracker_plane_mm=1500.0,
    step_length_mm=0.1)

# The key of the project's track set. It is computed from the content, so a
# rebuild that gives other content gives another key, and the gates fail.
KEY_OF_THE_TRACKS = "not built yet"

NUMBERS_OF_STEPS = (2, 64, 128, 256)
NUMBERS_OF_STAGES = (2, 4, 8, 16)


def the_store() -> Store:
    location = os.environ.get("RKPINN_STORE", AGREED_LOCATION_OF_THE_STORE)
    store = Store(location)
    if not store.holds_tracks(KEY_OF_THE_TRACKS):
        pytest.fail("the store at %s does not hold the track set %s, so this gate "
                    "cannot be run" % (location, KEY_OF_THE_TRACKS))
    return store
