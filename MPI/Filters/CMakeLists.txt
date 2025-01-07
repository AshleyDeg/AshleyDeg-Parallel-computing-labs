cmake_minimum_required(VERSION 3.25)
project(Problem2)
set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES main.cpp)
add_executable(Problem2 ${SOURCE_FILES})

# Add MPI Package to Project
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

# Link MPI library
target_link_libraries(Problem2 PRIVATE MPI::MPI_CXX)

# Link OpenCV Library
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Problem2 PRIVATE ${OpenCV_LIBS})