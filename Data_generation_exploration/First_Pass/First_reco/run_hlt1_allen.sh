#!/bin/bash
# Phase 2: run Allen (HLT1) forward tracking + truth-matched efficiency
# checking on the local Extended .digi, inside Gaudi (CPU backend).
# Usage:  bash run_hlt1_allen.sh [input.digi]
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOORE_OPTS=/cvmfs/lhcb.cern.ch/lib/lhcb/MOORE/MOORE_v59r4/Hlt/RecoConf/options

unset PYTHONPATH PYTHONHOME
ARGS=("$@")
set --
source /cvmfs/lhcb.cern.ch/lib/LbEnv-stable.sh
set -- "${ARGS[@]}"

mkdir -p "$HERE/reco_output"
cd "$HERE/reco_output"

MOORE_INPUT_DIGI="${1:-$HERE/reco_output/GaussDev5-Extended.digi}" \
    lb-run Moore/v59r4 gaudirun.py \
    "$HERE/moore_input_local.py" \
    "$MOORE_OPTS/allen_gaudi_forward.py" 2>&1 | tee hlt1_allen_forward.log
