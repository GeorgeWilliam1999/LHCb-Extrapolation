# Shared input options for running ANY Moore/RecoConf option file on our local
# Extended .digi. Stack it FIRST:
#   lb-run Moore/v59r4 gaudirun.py moore_input_local.py <recoconf_option_file.py>
# e.g. for HLT1 (Allen forward tracking + efficiency vs truth):
#   ... moore_input_local.py $MOORE/Hlt/RecoConf/options/allen_gaudi_forward.py
#
# Input/tuple names overridable via MOORE_INPUT_DIGI / MOORE_NTUPLE / MOORE_HISTOS.
import os

from RecoConf.options import options

HERE = os.path.dirname(os.path.abspath(__file__))

options.input_files = [
    os.environ.get(
        "MOORE_INPUT_DIGI",
        os.path.join(HERE, "reco_output", "GaussDev5-Extended.digi"),
    )
]
options.input_type = "ROOT"
options.simulation = True
options.geometry_version = "run3/2024-v00.02"
options.conditions_version = "sim10/2024"
options.evt_max = -1
if os.environ.get("MOORE_NTUPLE"):
    options.ntuple_file = os.environ["MOORE_NTUPLE"]
if os.environ.get("MOORE_HISTOS"):
    options.histo_file = os.environ["MOORE_HISTOS"]
