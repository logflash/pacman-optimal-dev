cmake_minimum_required(VERSION 3.10)
project(PacmanDP)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

# Value iteration executable - computes and saves value tables
add_executable(value_iteration
    value_iteration.cpp
)
target_link_libraries(value_iteration PRIVATE OpenMP::OpenMP_CXX)

# Simulation executable - loads value tables and runs simulations
add_executable(simulation
    simulation.cpp
)
