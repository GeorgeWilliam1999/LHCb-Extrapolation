"""fastex: the Python side of the fast-extrapolator experiment.

It holds new code only. It imports the present package (rkpinn) to read field
maps, track sets and trained runs, and it never edits it.

    export             flat binary files for the compiled program
    lhcb_integrators   Python copies of the Rec and Allen integrators
"""

STORE_OF_THE_PRESENT_RUNS = "/data/bfys/gscriven/rkpinn_store"
STORE_OF_THIS_EXPERIMENT = "/data/bfys/gscriven/fast_extrapolator_store"
FOLDER_OF_EXPORTS = STORE_OF_THIS_EXPERIMENT + "/exports"

# The track sets of the present store: the same particles carried on each map.
TRACK_SETS = {"cubic": "5558be31abeb", "linear": "afc61a558211"}
