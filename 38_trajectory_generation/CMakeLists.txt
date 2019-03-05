project(TrajectoryGeneration)
cmake_minimum_required(VERSION 3.5)

add_definitions(-std=c++11)
set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

include_directories(src/)
add_executable(hybrid_breadth_first src/main.cpp src/hybrid_breadth_first.cpp)
