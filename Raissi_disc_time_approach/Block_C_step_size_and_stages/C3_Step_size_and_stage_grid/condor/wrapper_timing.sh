#!/bin/bash
# HTCondor job wrapper for the C3.3 row ladder.
#
# One job = one rung of the ladder = one L-BFGS restart of the heaviest grid
# point on N training rows, measured on the same kind of slot the 720 grid jobs
# will run on. Same conventions as condor/wrapper_grid.sh: first argument is
# the experiment folder, single-thread pinning, absolute interpreter.
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

exec /data/bfys/gscriven/conda/envs/TE/bin/python "$WORKDIR/measure_timing.py" "$@"
