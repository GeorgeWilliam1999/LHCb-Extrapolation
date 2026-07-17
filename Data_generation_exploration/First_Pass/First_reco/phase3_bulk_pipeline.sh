#!/bin/bash
# Phase 3: full local pipeline on the 100-event minbias bulk sample.
#   Gauss (100 ev, 8 threads) -> Boole (Extended .digi) -> Moore (HLT2 reco + MC checking)
# Usage:  bash phase3_bulk_pipeline.sh     (all output under ../run_output and ./reco_output)
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FP="$(dirname "$HERE")"   # First_Pass

echo "=== [1/3] Gauss: 100 minbias events ==="
bash "$FP/run_gauss.sh" override_bulk_events.py

SIM=$(ls -t "$FP"/run_output/GaussMB100-30000000-100ev-*.sim | head -1)
echo "=== [2/3] Boole: digitising $SIM ==="
unset PYTHONPATH PYTHONHOME
set --
source /cvmfs/lhcb.cern.ch/lib/LbEnv-stable.sh

mkdir -p "$HERE/reco_output"
cd "$HERE/reco_output"
BOOLE_INPUT_SIM="$SIM" BOOLE_DATASET="GaussMB100" \
    lb-run Boole/v48r0 gaudirun.py "$HERE/boole_digitise.py" 2>&1 | tee boole_mb100.log

echo "=== [3/3] Moore: HLT2 light reco + MC checking ==="
MOORE_INPUT_DIGI="$HERE/reco_output/GaussMB100-Extended.digi" \
MOORE_NTUPLE="moore_hlt2_mccheck_mb100.root" \
MOORE_HISTOS="moore_hlt2_histos_mb100.root" \
    lb-run Moore/v59r4 gaudirun.py "$HERE/moore_hlt2_reco.py" 2>&1 | tee moore_mb100.log

echo "=== phase 3 pipeline complete ==="
