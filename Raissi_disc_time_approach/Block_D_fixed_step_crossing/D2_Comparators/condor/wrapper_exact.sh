#!/bin/bash
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$HERE"
echo "host      : $(hostname)"
echo "arguments : $*"
exec /data/bfys/gscriven/conda/envs/TE/bin/python "$HERE/exact_chain.py" "$@"
