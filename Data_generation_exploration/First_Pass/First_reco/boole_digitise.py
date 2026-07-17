# Boole options: digitise the Gauss .sim (5 minbias events, 2024 DD4hep) into a .digi.
#
# Run with (see run_boole.sh):
#   lb-run Boole/v48r0 gaudirun.py boole_digitise.py
#
# Boole v48r0 is built against the SAME stack as Gauss v61r0p2 (Gaudi 40.3,
# Detector 3.10, LHCb 59.0), so the event-model versions match exactly.
#
# Notes:
# - Geometry/conditions are pinned to the tags Gauss simulated with
#   (run3/2024-v00.02 + sim10/2024), not Boole's 'sim10/run3-ideal' default:
#   digitisation must see the same detector that made the hits.
# - Rich is dropped from every phase: our .sim contains no pSim/Rich hits
#   (Gauss v61 DD4hep default writes VP/UT/FT/Muon/Ecal/Hcal only).
# - Output: <DatasetName>-<n>ev-<date>.digi with MC truth kept (Default DigiType),
#   i.e. raw banks for the trigger + the links back to MCParticles/MCHits.
import os

from Gaudi.Configuration import EventSelector
from Configurables import Boole

HERE = os.path.dirname(os.path.abspath(__file__))
# Overridable via environment so pipeline scripts can chain different samples:
SIM = os.environ.get(
    "BOOLE_INPUT_SIM",
    os.path.join(HERE, "..", "run_output", "GaussDev5-30000000-5ev-20260716.sim"),
)
DATASET = os.environ.get("BOOLE_DATASET", "GaussDev5")

Boole().EvtMax = -1  # all events in the file
Boole().DataType = "2024"
# Extended => the six pSim/<det>/Hits containers are KEPT in the .digi.
# This is required for Moore's MC checkers (root-caused 2026-07-17): a
# 'Default' .digi stores the Link/Raw/*2MCHits link tables but NOT the MCHits
# they point to, and PrChecker/TrackResChecker segfault dereferencing the
# dangling links on the very first event. Output name gains '-Extended'.
Boole().DigiType = "Extended"
Boole().GeometryVersion = "run3/2024-v00.02"
Boole().ConditionsVersion = "sim10/2024"
Boole().DetectorDigi = ["VP", "UT", "FT", "Calo", "Muon"]
Boole().DetectorLink = ["VP", "UT", "FT", "Tr", "Calo", "Muon"]
Boole().DetectorMoni = ["VP", "UT", "FT", "Calo", "Muon", "MC"]
Boole().DatasetName = DATASET

EventSelector().Input = [
    "DATAFILE='PFN:%s' SVC='Gaudi::RootEvtSelector' OPT='READ'" % os.path.normpath(SIM)
]
