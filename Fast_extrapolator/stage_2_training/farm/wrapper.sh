#!/bin/bash
# The job of one run of stage 2 on the farm: train the run that a configuration
# names. A job that the farm stops and starts again goes on from the last round.
#
#   wrapper.sh <configuration file>
set -eu
CONFIGURATION="$1"
FOLDER=/data/bfys/gscriven/LHCb_Extrapolation_Project
PYTHON=/data/bfys/gscriven/conda/envs/TE/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH="$FOLDER/RK_Pinn_module/src:$FOLDER/Fast_extrapolator/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "machine        : $(hostname)"
echo "configuration  : $CONFIGURATION"
echo "started        : $(date '+%F %T')"
$PYTHON -m fastex.train "$CONFIGURATION" &
TRAINER=$!
trap 'kill -TERM $TRAINER 2>/dev/null' TERM INT
set +e
wait $TRAINER
CODE_OF_EXIT=$?
echo "ended          : $(date '+%F %T'), exit $CODE_OF_EXIT"
exit $CODE_OF_EXIT
