#!/bin/bash
# run_one_step.sh <script> <args...>: runs one analysis step on this node from this
# folder, and writes what it prints to logs/<script>_<args>.txt
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs
export PYTHONNOUSERSITE=1 PYTHONPATH=../../../RK_Pinn_module/src OMP_NUM_THREADS=${THREADS:-4} THREADS=${THREADS:-4} PYTHONDONTWRITEBYTECODE=1
args=$(echo ${@:2} | tr " " "_"); log=logs/$(basename $1 .py)${args:+_$args}.txt
/data/bfys/gscriven/conda/envs/TE/bin/python "$@" > $log 2>&1
echo "exit $? : $*" >> $log
