#!/bin/bash
# Digitise the First_Pass .sim into a .digi with Boole v48r0.
# Usage:  bash run_boole.sh          (writes into reco_output/, log to reco_output/boole.log)
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clean LbEnv environment (conda clashes with Gaudi; LbEnv eats positional args)
unset PYTHONPATH PYTHONHOME
ARGS=("$@")
set --
source /cvmfs/lhcb.cern.ch/lib/LbEnv-stable.sh
set -- "${ARGS[@]}"

mkdir -p "$HERE/reco_output"
cd "$HERE/reco_output"

exec lb-run Boole/v48r0 gaudirun.py "$HERE/boole_digitise.py" 2>&1 | tee boole.log
