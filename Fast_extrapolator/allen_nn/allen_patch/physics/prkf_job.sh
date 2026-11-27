#!/bin/bash
# One farm job: run_prkf.sh in the Allen (nightly 3237) environment.  prkf_job.sh OUTDIR EVTMAX FIRSTEVT NTHREADS [WEIGHTS]
source /cvmfs/lhcb.cern.ch/lib/LbEnv >/dev/null 2>&1
exec /data/bfys/gscriven/allen_physics/Allen/build.x86_64_v3-el9-gcc15-opt/run /data/bfys/gscriven/allen_physics/scripts/run_prkf.sh "$@"
