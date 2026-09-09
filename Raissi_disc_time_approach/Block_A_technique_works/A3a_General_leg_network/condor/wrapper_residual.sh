#!/bin/bash
# HTCondor job wrapper for the straight-line-residual arm.
#
# The shared wrapper (../../../_shared/condor/wrapper.sh) hands its arguments to
# _shared/train.py; this arm needs A3a_General_leg_network/train_residual.py, which
# installs the residual model into that same trainer. Everything else - the
# single-thread pinning, the absolute interpreter, the "first argument is the
# experiment folder" convention - is the shared wrapper's, unchanged.
#
# Usage (from a jobs_residual.txt line):
#     wrapper_residual.sh <experiment folder> <train_residual.py arguments...>
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

exec /data/bfys/gscriven/conda/envs/TE/bin/python "$WORKDIR/train_residual.py" "$@"
