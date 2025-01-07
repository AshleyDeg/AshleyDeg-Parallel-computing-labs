# Problem 1 - Assignment 1
Authors: Ashley Degl'Innocenti, Dorian Briodeau
Approximate π using the Bailey-Borwein-Plouffe formula.

### Prerequisites
- CMake (CMAKE_CXX_STANDARD 17)
- C++ compiler (supporting C++17)
- OpenMP 

### Instructions to run
Open terminal, navigate to project directory, and run CMake to generate build files.
cmake -S . -B build 
mpiexec -n 8 ./cmake-build-debug/Problem2