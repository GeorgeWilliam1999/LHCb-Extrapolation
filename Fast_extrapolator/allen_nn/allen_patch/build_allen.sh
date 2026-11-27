#!/bin/bash
# CPU build of the local Allen branch allen_nn_magnet_crossing (MR !2407 + the magnet-crossing switch)
# against the lhcb-master nightly on CVMFS. Log: /tmp/gscriven/allen_build/build.log
set -x
source /cvmfs/lhcb.cern.ch/lib/LbEnv
cd /tmp/gscriven/allen_build/Allen
export BINARY_TAG=x86_64_v3-el9-gcc15-opt CMTCONFIG=x86_64_v3-el9-gcc15-opt
export LCG_VERSION=108c
export CMAKE_PREFIX_PATH=/cvmfs/lhcbdev.cern.ch/nightlies/nightly/lhcb-master/3236:$CMAKE_PREFIX_PATH
type lb-project-init
lb-project-init . || exit 1
make configure || exit 2
nice -n 10 make -j6 install || exit 3
echo BUILD_DONE
