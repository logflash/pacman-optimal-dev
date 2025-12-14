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

# === Multi-pellet version ===

# Value iteration with multiple super pellets (1-4)
# add_executable(value_iteration_multi
#     value_iteration_multi.cpp
# )
# target_link_libraries(value_iteration_multi PRIVATE OpenMP::OpenMP_CXX)

# === Optimized multi-pellet version ===

find_package(ZLIB REQUIRED)

# Optimized value iteration with compact encoding + compression
add_executable(value_iteration_opt
    value_iteration_multi_opt.cpp
)
target_link_libraries(value_iteration_opt PRIVATE OpenMP::OpenMP_CXX ZLIB::ZLIB)

# Simulation for optimized multi-pellet
add_executable(simulation_opt
    simulation_multi_opt.cpp
)
target_link_libraries(simulation_opt PRIVATE ZLIB::ZLIB)

# === Analysis tools ===

# Value analysis tool - generates statistics and exports CSVs
# add_executable(analyze_values
#     analyze_values.cpp
# )
# target_link_libraries(analyze_values PRIVATE ZLIB::ZLIB)
