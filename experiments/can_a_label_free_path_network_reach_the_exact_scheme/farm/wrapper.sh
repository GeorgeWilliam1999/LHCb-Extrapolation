#!/bin/bash
# The job of one run on the farm: train the run a configuration names, on one
# thread, from a copy of the code that is fixed at one commit.
#
#   wrapper.sh <code> <store> <configuration file>
#
#   code            the folder of the project as checked out at one commit
#                   (made by make_code.sh); the package is in <code>/RK_Pinn_module
#   store           the store the run is written to
#   configuration   the configuration file of the run
#
# The farm stops a job at its time limit and starts it again, so the job makes
# the run if the store does not hold it, and resumes it if it does. A lock that
# was not touched for 30 minutes was left behind by a job that was killed; no
# restart takes that long.
set -eu
CODE="$1"; STORE="$2"; CONFIGURATION="$3"
PYTHON=/data/bfys/gscriven/conda/envs/TE/bin/python
A_LOCK_IS_LEFT_BEHIND_AFTER_MINUTES=30

export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE/RK_Pinn_module/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$CODE/RK_Pinn_module"

KEY=$($PYTHON -m rkpinn.run_record.configuration "$CONFIGURATION")
if [ -f "$STORE/runs/$KEY/state.json" ]; then ASKED=resume; else ASKED=refuse; fi

echo "machine        : $(hostname)"
echo "processor      : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "code           : $CODE at $(git -C "$CODE" rev-parse HEAD)"
echo "store          : $STORE"
echo "configuration  : $CONFIGURATION"
echo "run            : $KEY, asked to $ASKED"
echo "started        : $(date '+%F %T')"

# the trainer is a child, so that the signal of the farm can be handed to it
$PYTHON -m rkpinn.training.round_trainer --store "$STORE" \
    --configuration "$CONFIGURATION" --if-the-run-exists "$ASKED" \
    --a-lock-is-left-behind-after-minutes "$A_LOCK_IS_LEFT_BEHIND_AFTER_MINUTES" &
TRAINER=$!
trap 'kill -TERM $TRAINER 2>/dev/null' TERM INT
set +e
wait $TRAINER
CODE_OF_EXIT=$?
# a signal interrupts `wait`; wait again for the trainer to release its lock
if kill -0 $TRAINER 2>/dev/null; then wait $TRAINER; CODE_OF_EXIT=$?; fi
echo "ended          : $(date '+%F %T'), exit $CODE_OF_EXIT"
exit $CODE_OF_EXIT
