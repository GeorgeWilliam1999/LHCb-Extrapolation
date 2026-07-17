#!/bin/bash
# Generate full simulated LHCb events (Run 3, 2024 conditions, minimum bias).
# Usage:  bash run_gauss.sh                            -> 1 event (gauss_one_event.py)
#         bash run_gauss.sh override_five_events.py    -> 5 events, 8 threads
#
# Data packages not pulled in automatically by Gauss's manifest, so we add them:
#   AppConfig    -> beam-condition option files ($APPCONFIGOPTS)
#   Gen/DecFiles -> event-type option files + decay tables ($DECFILESROOT)
# (no `set -u`: LbEnv-stable.sh reads unset vars and would abort the script)
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clean LbEnv environment (conda clashes with Gaudi: PYTHONPATH/PYTHONHOME leak).
# LbEnv-stable.sh parses the shell's positional args -> stash and restore them.
unset PYTHONPATH PYTHONHOME
ARGS=("$@")
set --
source /cvmfs/lhcb.cern.ch/lib/LbEnv-stable.sh
set -- "${ARGS[@]}"

mkdir -p "$HERE/run_output"
cd "$HERE/run_output"

EXTRA_OPTS=()
LOG=gauss.log
for f in "$@"; do
    EXTRA_OPTS+=("$HERE/$f")
    LOG="gauss_$(basename "$f" .py).log"
done

exec lb-run \
    --use "AppConfig v4r8" \
    --use "Gen/DecFiles v33r3" \
    Gauss/v61r0p2 \
    gaudirun.py "$HERE/gauss_one_event.py" "${EXTRA_OPTS[@]}" 2>&1 | tee "$LOG"
