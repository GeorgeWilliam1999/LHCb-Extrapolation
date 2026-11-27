"""Allen (CPU, in Gaudi) PrKF reconstruction of MR !2407 on MC, with the MC checkers of Rec.

The fit of the sequence hlt1_pp_forward_then_matching_and_downstream_with_fullkf_1300kHz_rich
(PrKF, make_kalman_long version 2), reconstruction only (no lines). The magnet crossing of the
second fit pass uses the network when the environment variable MAGNET_NN_WEIGHTS names a JSON
weights file, the ARKN when it is empty or unset.

    lbexec prkf_physics.py:main options.yaml

Outputs (names from the yaml): histo_file (TrackResChecker, StateDumper) and ntuple_file
(TrackIPResolutionCheckerNT, one row per fitted long track: true p, fitted q/p, chi2, ndof, MC link).
"""

import functools
import os

from Allen.config import AllenTestOptions, run_allen
from AllenConf.enum_types import TrackingType
import AllenConf.hlt1_reconstruction as h1r
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.scifi_reconstruction import make_seeding_tracks
import AllenConf.secondary_vertex_reconstruction as svr
from AllenConf.secondary_vertex_reconstruction import ParKF_cuts
from PyConf.Algorithms import (
    ConvertAllenLongToV3Tracks_Rich,
    GaudiAllenPVsToPrimaryVertexContainer,
    fromV3TrackV1Track,
)
from PyConf.control_flow import CompositeNode, NodeLogic
from PyConf.reading import get_mc_track_info
from RecoConf.core_algorithms import make_unique_id_generator
from RecoConf.decoders import default_VeloCluster_source
from RecoConf.legacy_rec_hlt1_tracking import get_global_clusters_on_track_tool
from RecoConf.mc_checking import check_track_resolution, dump_states, monitor_IPresolution


def make_reconstruction():
    weights = os.environ.get("MAGNET_NN_WEIGHTS", "")
    # make_kalman_long and make_full_kalman_long are plain functions (not configurable here): bind the
    # fit version (2 = PrKF) and the weights in the modules that look them up at call time.
    svr.make_full_kalman_long = functools.partial(svr.make_full_kalman_long, magnet_nn_weights=weights)
    h1r.make_kalman_long = functools.partial(svr.make_kalman_long, version=2)
    with (
        make_velo_scifi_matches.bind(ghost_killer_threshold=0.8),
        make_seeding_tracks.bind(version=2),
    ):
        reco = h1r.hlt1_reconstruction(
            tracking_type=TrackingType.FORWARD_THEN_MATCHING,
            with_ut=True,
            enableDownstream=True,
            with_fullKF=True,
            with_downstream_KF=True,
            with_ttracks=True,
            with_rich=True,
            track_max_chi2ndof=ParKF_cuts.SV_track_max_chi2ndof,
        )
    kalman = reco["kalman_long_track"]
    v3 = ConvertAllenLongToV3Tracks_Rich(
        dev_particle_container=reco["long_track_particles"]["dev_multi_event_basic_particles"],
        dev_rich_R1F_data=kalman["dev_kalman_R1_F_view"],
        dev_rich_R1B_data=kalman["dev_kalman_R1_B_view"],
        dev_rich_R2F_data=kalman["dev_kalman_R2_F_view"],
        dev_rich_R2B_data=kalman["dev_kalman_R2_B_view"],
        dev_ctb_data=kalman["dev_kalman_ctb_view"],
        InputUniqueIDGenerator=make_unique_id_generator(),
    ).OutputTracks
    v1keyed = fromV3TrackV1Track(
        InputTracks=v3, TrackAddClusterTool=get_global_clusters_on_track_tool()
    ).OutputTracks
    tracks = {"v3": v3, "v1keyed": v1keyed}

    pvs = GaudiAllenPVsToPrimaryVertexContainer(
        number_of_multivertex=reco["pvs"]["dev_number_of_multi_final_vertices"],
        reconstructed_multi_pvs=reco["pvs"]["dev_multi_final_vertices"],
    ).OutputPVs

    resolution = check_track_resolution(InputTracks=tracks, per_hit_resolutions=True, split_per_type=True,
                                        track_type="v1keyed")
    states = dump_states(InputTracks=tracks, parkf=True, track_type="v1keyed")
    per_track = monitor_IPresolution(v1keyed, pvs, None)

    return CompositeNode(
        "prkf_physics",
        combine_logic=NodeLogic.NONLAZY_AND,
        children=[get_mc_track_info(), resolution, states, per_track],
        force_order=True,
    )


def main(options: AllenTestOptions):
    with default_VeloCluster_source.bind(bank_type="VPRetinaCluster"):
        return run_allen(options, sequence=make_reconstruction)
