cmake_minimum_required(VERSION 3.25)
project(Problem_2)
set(CMAKE_CXX_STANDARD 17)

# try also to compile and execute without: -fopenmp
# in Linux just use -fopenmp, without -lomp and -Xpreprocessor
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -lomp")
# the following 2 lines are needed on macOS. Check the position of headers in libraries (e.g. Macports vs. Homebrew)
include_directories(/opt/homebrew/opt/libomp/include)
link_directories(/opt/homebrew/opt/libomp/lib)

set(SOURCE_FILES main.cpp)
add_executable(Problem_2 ${SOURCE_FILES})

#find_library(OpenCV)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Problem_2 ${OpenCV_LIBS})