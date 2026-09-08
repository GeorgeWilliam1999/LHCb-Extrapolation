#!/bin/bash
# HTCondor job wrapper for the chained-crossing study.
#
# The shared wrapper (../../_shared/condor/wrapper.sh) hands its arguments to
# _shared/train.py; this experiment needs Chained_crossing/chain_one.py, which
# chains one already-trained grid network across the magnet. Everything else -
# the single-thread pinning, the absolute interpreter, the "first argument is
# the experiment folder" convention - is the shared wrapper's, unchanged.
#
# Nothing here trains, and nothing here writes into
# ../Step_size_and_stage_grid: the checkpoints and the datasets are opened
# read-only.
#
# Usage (from a jobs_chain.txt line):
#     wrapper_chain.sh <experiment folder> <chain_one.py arguments...>
set -eu

CONDOR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$CONDOR")")"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EXPERIMENT="$1"; shift
case "$EXPERIMENT" in
    /*) WORKDIR="$EXPERIMENT" ;;
    *)  WORKDIR="$ROOT/$EXPERIMENT" ;;
esac
cd "$WORKDIR"

echo "host      : $(hostname)"
echo "workdir   : $WORKDIR"
echo "arguments : $*"

exec /data/bfys/gscriven/conda/envs/TE/bin/python "$WORKDIR/chain_one.py" "$@"
