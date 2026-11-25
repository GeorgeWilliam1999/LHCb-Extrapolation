#!/bin/bash
# run_eval.sh <key> <tracks or own> <sweeps>: pick the code of the run's commit
K=$1; C=$(python3 -c "import json;print(json.load(open('/data/bfys/gscriven/rkpinn_store/runs/$K/state.json'))['invocations'][-1]['commit'][:8])")
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/data/bfys/gscriven/rkpinn_code/$C/RK_Pinn_module/src
nice -n 10 /data/bfys/gscriven/conda/envs/TE/bin/python $(dirname $0)/eval_run.py "$@" 2>&1 | tail -3
