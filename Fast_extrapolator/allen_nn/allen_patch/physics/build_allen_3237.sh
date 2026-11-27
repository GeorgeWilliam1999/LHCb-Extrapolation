#!/bin/bash
# CPU build of the local Allen branch allen_nn_magnet_crossing (MR !2407 + the magnet-crossing switch)
# against ONE nightly slot, lhcb-master/3237 (Gaudi, Detector, LHCb, Lbcom, Rec all from it), on /data
# so that farm jobs can run it. Compiles with ninja -j6 (login node: 8 cores or fewer).
set -x
source /cvmfs/lhcb.cern.ch/lib/LbEnv
D=/data/bfys/gscriven/allen_physics
cd $D
[ -d Allen ] || git clone -b allen_nn_magnet_crossing /tmp/gscriven/lhcb_read/Allen_mr2407 Allen || exit 1
cd Allen
git log --oneline -3
export BINARY_TAG=x86_64_v3-el9-gcc15-opt CMTCONFIG=x86_64_v3-el9-gcc15-opt
export LCG_VERSION=108c
export CMAKE_PREFIX_PATH=/cvmfs/lhcbdev.cern.ch/nightlies/nightly/lhcb-master/3237:$CMAKE_PREFIX_PATH
lb-project-init . || exit 2
make configure || exit 3
grep -E "^(Rec|LHCb|Gaudi|Detector|Lbcom)_DIR" build.$BINARY_TAG/CMakeCache.txt
cd build.$BINARY_TAG && nice -n 10 ninja -j6 && nice -n 10 ninja -j6 install || exit 4
echo BUILD_DONE
