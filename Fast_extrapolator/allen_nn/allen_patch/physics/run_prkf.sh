#!/bin/bash
# run_prkf.sh OUTDIR EVTMAX FIRSTEVT NTHREADS [WEIGHTS]   (WEIGHTS empty = ARKN)
set -e
out=$1; mkdir -p "$out"; cd "$out"
D=/data/bfys/gscriven/allen_physics
sed -e "s/EVTMAX/$2/" -e "s/FIRSTEVT/$3/" -e "s/NTHREADS/$4/" -e "s#HISTO#$out/histos.root#" -e "s#NTUPLE#$out/ntuple.root#" \
    $D/scripts/options_template.yaml > options.yaml
# MR !2407 reads ParametrizedKalmanFit/ParametrisedScatters.json, only on the ParamFiles branch pehoffma_HLT1_KF_ARKN (cf69c93)
export PARAMFILESROOT=/data/bfys/gscriven/allen_physics/ParamFiles
export GITCONDDBPATH=/cvmfs/lhcb.cern.ch/lib/lhcb/git-conddb
w=${5:-}; [ "$w" = none ] && w=""
export MAGNET_NN_WEIGHTS=$w
M=$D/Moore/Hlt
export PYTHONPATH=$M/Hlt1Conf/python:$M/Moore/python:$PYTHONPATH
echo "host $(hostname) weights '${MAGNET_NN_WEIGHTS}' start $(date)"
/usr/bin/time -v lbexec $D/scripts/prkf_physics.py:main options.yaml > run.log 2>&1
echo "done $(date)"
