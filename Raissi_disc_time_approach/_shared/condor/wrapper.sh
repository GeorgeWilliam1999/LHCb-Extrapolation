#!/bin/bash
# HTCondor job wrapper: one training run of the shared driver.
#
# Usage (from a jobs.txt line):
#     wrapper.sh <experiment folder> <train.py arguments...>
#
# The first argument is the experiment folder the job runs in (absolute, or
# relative to Raissi_disc_time_approach/); everything after it is passed to
# _shared/train.py unchanged, so a jobs.txt line reads exactly like the command
# you would type by hand.
#
# Example line:
#     One_step_network_v3 --data results/frozen_leg_data.npz --mode physics \
#         --seed 0 --out results --tag physics_seed0
set -eu

SHARED="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "$SHARED")"

# The training node's torch spin-waits on its worker threads (measured 68x
# slower); one thread per job, and the farm slot is requested as one CPU.
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

exec /data/bfys/gscriven/conda/envs/TE/bin/python "$SHARED/train.py" "$@"
