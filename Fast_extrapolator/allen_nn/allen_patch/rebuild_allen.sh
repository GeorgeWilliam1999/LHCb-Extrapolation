#!/bin/bash
# Re-run cmake (to pick up the new unit test) and build with 6 jobs, then run the magnet_nn unit tests.

source /cvmfs/lhcb.cern.ch/lib/LbEnv
cd /tmp/gscriven/allen_build/Allen/build.x86_64_v3-el9-gcc15-opt
export BINARY_TAG=x86_64_v3-el9-gcc15-opt CMTCONFIG=x86_64_v3-el9-gcc15-opt LCG_VERSION=108c
export CMAKE_PREFIX_PATH=/cvmfs/lhcbdev.cern.ch/nightlies/nightly/lhcb-master/3236:$CMAKE_PREFIX_PATH
cmake . > /dev/null || exit 1
nice -n 19 ninja -j6 install || exit 2
echo BUILD_DONE
./run ./unit_tests "[magnet_nn]" ; echo UNIT_TESTS_EXIT=$?
