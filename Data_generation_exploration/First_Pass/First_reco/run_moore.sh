#!/bin/bash
# Run Moore HLT2 light reconstruction on the local .digi.
# Usage:  bash run_moore.sh          (log to reco_output/moore.log)
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

unset PYTHONPATH PYTHONHOME
ARGS=("$@")
set --
source /cvmfs/lhcb.cern.ch/lib/LbEnv-stable.sh
set -- "${ARGS[@]}"

mkdir -p "$HERE/reco_output"
cd "$HERE/reco_output"

exec lb-run Moore/v59r4 gaudirun.py "$HERE/moore_hlt2_reco.py" 2>&1 | tee moore.log
