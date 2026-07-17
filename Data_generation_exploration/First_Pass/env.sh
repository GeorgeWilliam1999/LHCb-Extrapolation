# Source me to get the LHCb software environment (LbEnv) on this Nikhef node:
#     source /data/bfys/gscriven/LHCb_Extrapolation_Project/Data_generation_exploration/First_Pass/env.sh
#
# After sourcing, `lb-run` is available, e.g.:
#     lb-run Gauss/v61r0p2 gaudirun.py <options...>
#
# NOTE: an active conda env leaks PYTHONPATH/PYTHONHOME into Gaudi jobs and
# breaks them (same caveat as /data/bfys/gscriven/grid-env.sh) -> clean it first.
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME
source /cvmfs/lhcb.cern.ch/lib/LbEnv-stable.sh
