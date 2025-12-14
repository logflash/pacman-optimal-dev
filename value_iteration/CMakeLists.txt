cmake_minimum_required(VERSION 3.10)
project(PacmanDP)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

# === Original (no super pellet) ===

# Value iteration executable - computes and saves value tables
add_executable(value_iteration
    value_iteration.cpp
)
target_link_libraries(value_iteration PRIVATE OpenMP::OpenMP_CXX)

# Simulation executable - loads value tables and runs simulations
add_executable(simulation
    simulation.cpp
)

# === Super pellet version ===

# Value iteration with super pellet + ghost permadeath
add_executable(value_iteration_super
    value_iteration_super.cpp
)
target_link_libraries(value_iteration_super PRIVATE OpenMP::OpenMP_CXX)

# Simulation with super pellet
add_executable(simulation_super
    simulation_super.cpp
)
