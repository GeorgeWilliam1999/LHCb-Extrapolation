# Stack this AFTER gauss_one_event.py for the Phase-3 bulk sample:
# 100 minimum-bias bunch crossings on 8 threads.
#   gaudirun.py gauss_one_event.py override_bulk_events.py
# RunNumber stays 1 -> events 1-100; the first 5 are bit-identical to the
# GaussDev5 sample (seeds derive from run/event numbers).
from Gaudi.Configuration import importOptions
from Configurables import Gauss

Gauss().EvtMax = 100
Gauss().DatasetName = "GaussMB100"
importOptions("$GAUSSOPTS/General/Threads-8.py")
