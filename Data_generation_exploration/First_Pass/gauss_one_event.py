# Gauss (Gauss-on-Gaussino) options: ONE full simulated Run 3 minimum-bias event.
#
# Run with (see run_gauss.sh):
#   lb-run --use "AppConfig v4r8" --use "Gen/DecFiles v33r3" Gauss/v61r0p2 \
#   (NB: AppConfig v4rX = Gaussino-era option files; the v3rX series is written
#    for classic Gauss and fails on Gauss v61 with 'no B1Particle property')
#       gaudirun.py gauss_one_event.py
#
# What each import does:
#   Geometry/DD4hep.py   -> use the DD4hep geometry backend (Run 3 default;
#                           GaussGeometry().Legacy = False)
#   General/2024.py      -> 2024 "Block 7" beam conditions (pp at 6.8 TeV per beam,
#                           nu = 7.6 pileup, crossing angles, IP position/smearing)
#                           + DD4hep DB tags: geometry run3/2024-v00.02,
#                           conditions sim10/2024 + DataType "2024"
#   30000000.py          -> DecFiles "event type" 30000000 = inclusive minimum bias:
#                           Pythia8 (multi-threaded production tool) generates the
#                           pp collisions, EvtGen handles the hadron decays
#
# Output: <DatasetName>-30000000-1ev-<date>.sim in the working directory
# (a Gaudi/ROOT file with the packed MC truth: MCParticles, MCVertices, MCHits).

from Gaudi.Configuration import importOptions
from Configurables import Gauss

Gauss().EvtMax = 1            # one event = one LHC bunch crossing
Gauss().RunNumber = 1         # seeds derive from (run, event) numbers -> reproducible
Gauss().FirstEventNumber = 1
Gauss().DatasetName = "GaussDev"

importOptions("$GAUSSOPTS/Geometry/DD4hep.py")
importOptions("$GAUSSOPTS/General/2024.py")
importOptions("$DECFILESROOT/options/30000000.py")
