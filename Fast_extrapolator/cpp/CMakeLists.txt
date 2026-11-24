# The compiled program of the fast-extrapolator experiment.
#
#   cmake -S . -B build/$(hostname -s) && cmake --build build/$(hostname -s) -j 4
#
# The build folder is named after the node, because the default target is the
# processor of the node that compiles (-march=native). For a binary that runs on
# every node with AVX2, pass -DFX_ARCH=x86-64-v3 (the target of the LHCb builds).
# -fno-math-errno lets the compiler use the square-root instruction with no
# check of errno; it does not change any result. No other change of the
# floating-point rules is made (no -ffast-math).
cmake_minimum_required(VERSION 3.16)
project(fx CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(FX_ARCH "native" CACHE STRING "the value of -march")
# With FX_CONTRACT=off the compiler does not fuse a multiplication and an
# addition into one instruction. The program then does the same operations as
# Python and numpy, one by one. That build is for the gate only; the timed build
# keeps the default of the compiler (fused), as the LHCb builds do.
set(FX_CONTRACT "fast" CACHE STRING "the value of -ffp-contract")

add_compile_options(-O3 -march=${FX_ARCH} -fno-math-errno -ffp-contract=${FX_CONTRACT} -Wall -Wextra)
add_executable(fx bench/fx.cpp)
# libmvec holds the vector exponential that the fast tanh uses
target_link_libraries(fx mvec m)
