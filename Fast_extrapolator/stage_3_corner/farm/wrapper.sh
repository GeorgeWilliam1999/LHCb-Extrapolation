#!/bin/bash
# One farm job of stage 3: train the run that a configuration names (4 cores). A job that is started
# again goes on from the last round; a call stops after HOURS_AT_MOST hours.
#   wrapper.sh <configuration file>
set -eu
CONFIGURATION="$1"
HERE=/data/bfys/gscriven/worktrees/allen_nn/Fast_extrapolator
PYTHON=/data/bfys/gscriven/conda/envs/TE/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HERE/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HOURS_AT_MOST=22
echo "machine        : $(hostname)"
echo "configuration  : $CONFIGURATION"
echo "commit         : $(git -C $HERE rev-parse HEAD)"
echo "started        : $(date '+%F %T')"
$PYTHON -c "import torch; torch.set_num_threads(4); import runpy, sys; sys.argv=['train_corner.py', '$CONFIGURATION']; runpy.run_path('$HERE/stage_3_corner/python/train_corner.py', run_name='__main__')" &
TRAINER=$!
trap 'kill -TERM $TRAINER 2>/dev/null' TERM INT
set +e
wait $TRAINER
CODE=$?
echo "ended          : $(date '+%F %T'), exit $CODE"
exit $CODE
