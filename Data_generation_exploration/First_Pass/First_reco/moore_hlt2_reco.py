# Moore options: run the HLT2 light track reconstruction (+ Kalman fit + MC-truth
# checking) on the locally digitised GaussDev5.digi.
#
# Run with (see run_moore.sh):
#   lb-run Moore/v59r4 gaudirun.py moore_hlt2_reco.py
#
# 'Light reco' = the full HLT2 tracking chain (VELO clusters -> VELO tracks ->
# forward/seed/match -> long tracks -> PrKalmanFilter) without RICH/CALO PID —
# exactly right here, since our .sim carried no RICH hits. MC checking runs
# PrTrackChecker against the truth-link tables Boole wrote into the .digi
# (efficiencies, ghost rates, hit purities -> ntuple + histograms).
#
# Geometry/conditions are pinned to what Gauss simulated and Boole digitised.
import os

from RecoConf.config import run_reconstruction
from RecoConf.options import options
from RecoConf.standalone import standalone_hlt2_light_reco

HERE = os.path.dirname(os.path.abspath(__file__))

# NB: must be an 'Extended'-DigiType file (carries pSim MCHits) — the MC
# checkers segfault on a Default .digi whose truth links dangle (2026-07-17).
options.input_files = [
    os.environ.get(
        "MOORE_INPUT_DIGI",
        os.path.join(HERE, "reco_output", "GaussDev5-Extended.digi"),
    )
]
options.ntuple_file = os.environ.get("MOORE_NTUPLE", "moore_hlt2_mccheck.root")
options.histo_file = os.environ.get("MOORE_HISTOS", "moore_hlt2_histos.root")
options.input_type = "ROOT"
options.simulation = True
options.geometry_version = "run3/2024-v00.02"
options.conditions_version = "sim10/2024"
options.evt_max = -1

with standalone_hlt2_light_reco.bind(
    do_mc_checking=True, do_data_monitoring=True, use_pr_kf=True
):
    run_reconstruction(options, standalone_hlt2_light_reco)
