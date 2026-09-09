#!/bin/bash
# HTCondor job wrapper for the width x depth x q grid.
#
# The shared wrapper (../../../_shared/condor/wrapper.sh) hands its arguments to
# _shared/train.py; this experiment needs C3_Step_size_and_stage_grid/train_grid.py,
# which installs the grid model and the per-stratum scoring into that same
# trainer. Everything else - the single-thread pinning, the absolute
# interpreter, the "first argument is the experiment folder" convention - is
# the shared wrapper's, unchanged.
#
# Usage (from a jobs_grid.txt line):
#     wrapper_grid.sh <experiment folder> <train_grid.py arguments...>
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

exec /data/bfys/gscriven/conda/envs/TE/bin/python "$WORKDIR/train_grid.py" "$@"
