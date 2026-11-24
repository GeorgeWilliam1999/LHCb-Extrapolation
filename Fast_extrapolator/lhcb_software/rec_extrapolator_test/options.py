###############################################################################
# The extrapolator test of Rec, run by us.
#
# This is tests/options/test_extrapolators.py of Rec v41r3
# (Tr/TrackExtrapolators), cut down to one tool with its default properties:
# one of the two integrators that our compiled program copies. The algorithm ExtrapolatorTester carries its fixed grid of
# 1,331 states from z = 3000 mm to z = 7000 mm in every event.
#
# The environment variables choose what the job does:
#   FX_EVENTS         number of events (default 1)
#   FX_PRINT          1: print the final states, as the Rec test does (default 1)
#   FX_EXTRAPOLATOR   rk: TrackRungeKuttaExtrapolator, the adaptive Cash-Karp
#                     integrator (default); step: TrackSTEPExtrapolator, the
#                     adaptive Runge-Kutta-Nystrom integrator; linear: a
#                     straight line (its time is the time of the test
#                     algorithm itself)
###############################################################################
import os

from Configurables import (
    TrackLinearExtrapolator,
    TrackRungeKuttaExtrapolator,
    TrackSTEPExtrapolator,
)
from DDDB.CheckDD4Hep import UseDD4Hep
from PyConf.Algorithms import ExtrapolatorTester
from PyConf.application import ApplicationOptions, configure, configure_input
from PyConf.control_flow import CompositeNode

options = ApplicationOptions(_enabled=False)
options.set_input_and_conds_from_testfiledb("MiniBrunel_2018_MinBias_FTv4_DIGI")
options.evt_max = int(os.environ.get("FX_EVENTS", "1"))
config = configure_input(options)

if UseDD4Hep:
    dd4hepSvc = config["LHCb::Det::LbDD4hep::DD4hepSvc/LHCb::Det::LbDD4hep::DD4hepSvc"]
    dd4hepSvc.DetectorList = ["/world", "Magnet"]

# As in the Rec test, the algorithm is made with an empty list of tools and
# the tools are added to that list afterwards: PyConf cannot store a tool that
# is made with Configurables in the properties of an algorithm.
extrapolators = []
ex = ExtrapolatorTester(
    name="ExtrapolatorTester",
    Extrapolators=extrapolators,
    EnablePrints=os.environ.get("FX_PRINT", "1") == "1",
)
tool = {
    "rk": TrackRungeKuttaExtrapolator,
    "step": TrackSTEPExtrapolator,
    "linear": TrackLinearExtrapolator,
}[os.environ.get("FX_EXTRAPOLATOR", "rk")]
extrapolators += [tool("Reference")]

config.update(configure(options, CompositeNode("TopSeq", [ex])))
