#!/bin/bash
# Run the extrapolator test of Rec v41r3 (options.py) in the LHCb software.
# Usage:  bash run.sh <name of the log> [core]
#   FX_EVENTS, FX_PRINT and FX_EXTRAPOLATOR are read by options.py.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${1:-rec_test}"
CORE="${2:-}"

unset PYTHONPATH PYTHONHOME
ARGS=("$@")
set --
source /cvmfs/lhcb.cern.ch/lib/LbEnv-stable.sh
set -- "${ARGS[@]}"

mkdir -p "$HERE/output"
cd "$HERE/output"
PIN=""
if [ -n "$CORE" ]; then PIN="taskset -c $CORE"; fi
$PIN lb-run -c x86_64_v3-el9-gcc15-opt Rec/v41r3 gaudirun.py "$HERE/options.py" > "$LOG.log" 2>&1
echo "exit code $?" >> "$LOG.log"
