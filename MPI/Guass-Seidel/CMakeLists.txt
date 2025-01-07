cmake_minimum_required(VERSION 3.25)
project(Problem_3)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES main.cpp)
add_executable(Problem_3 ${SOURCE_FILES})

# Add MPI Package to Project
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

# Link MPI library
target_link_libraries(Problem_3 PRIVATE MPI::MPI_CXX)
