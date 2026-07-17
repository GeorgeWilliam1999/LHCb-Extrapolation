# Stack this AFTER gauss_one_event.py to get 5 events instead of 1, processed
# on 8 threads (Gaussino parallelises across events, seeds fixed per event
# number -> event 1 is bit-identical to the single-event run):
#   gaudirun.py gauss_one_event.py override_five_events.py
from Gaudi.Configuration import importOptions
from Configurables import Gauss

Gauss().EvtMax = 5
Gauss().DatasetName = "GaussDev5"
importOptions("$GAUSSOPTS/General/Threads-8.py")
