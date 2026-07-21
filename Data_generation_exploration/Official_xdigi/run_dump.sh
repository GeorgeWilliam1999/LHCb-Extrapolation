#!/bin/bash
# Dump MC truth from the official expected_2024_minbias_xdigi sample (CVMFS mirror).
# Usage: bash run_dump.sh [outdir] [n_events]
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset PYTHONPATH PYTHONHOME
ARGS=("$@")
set --
source /cvmfs/lhcb.cern.ch/lib/LbEnv-stable.sh
set -- "${ARGS[@]}"
cd "$HERE"
exec lb-run Gauss/v61r0p2 python dump_xdigi.py "${1:-truth_official}" "${2:-200}"
