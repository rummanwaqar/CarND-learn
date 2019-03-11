project(VehicleModels)

cmake_minimum_required (VERSION 3.5)

# -g allows for gdb debugging
# turn on -03 for best performance
add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

include_directories(/usr/local/include)
include_directories(src/Eigen-3.3)
link_directories(/usr/local/lib)

add_executable(global_kinematic_model src/kinematic_model.cpp)
target_link_libraries(global_kinematic_model)

add_executable(polyfit src/polyfit.cpp)
target_link_libraries(polyfit)
