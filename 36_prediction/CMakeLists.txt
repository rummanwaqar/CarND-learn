project(Prediction)
cmake_minimum_required(VERSION 3.5)

add_definitions(-std=c++11)
set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

include_directories(src/)
add_executable(naive_bayes src/main.cpp src/classifier.cpp)
