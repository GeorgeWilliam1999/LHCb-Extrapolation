#!/bin/bash
# Build one per-q training set for every q in the grid, one q at a time.
#
# Each q costs one RK6 crossing per row at 0.1 mm; the full 60,000 rows come to
# roughly half an hour of CPU, so the builds are run sequentially with a small
# worker pool rather than all at once - this is a shared interactive node.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
WORKERS="${1:-8}"
shift || true
QS="${*:-20 2 4 6 8 10 12 14 16 18}"
cd "$HERE"
for q in $QS; do
    out="results/grid_q$(printf %02d "$q").npz"
    if [ -f "$out" ]; then
        echo "== q=$q already built, skipping"
        continue
    fi
    echo "== q=$q  $(date +%H:%M:%S)"
    $PY prepare_nodes.py --q "$q" --workers "$WORKERS"
done
echo "== all done $(date +%H:%M:%S)"
