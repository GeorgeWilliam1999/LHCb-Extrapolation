#!/bin/bash
# Build the standalone program of the magnet-crossing tool on this node:
#   allen_nn/build/<node>/an      (-O3 -march=native, as cpp/CMakeLists.txt; no -ffast-math)
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=$here/build/$(hostname -s)
mkdir -p "$out"
g++ -std=c++20 -O3 -march=${AN_ARCH:-native} -fno-math-errno -ffp-contract=${AN_CONTRACT:-fast} -Wall -Wextra \
    -o "$out/an" "$here/src/an.cpp" -lm
echo "built $out/an"
