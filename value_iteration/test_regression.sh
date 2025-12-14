#!/bin/zsh
# Regression test: Compare refactored code vs original main branch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "REGRESSION TEST: Refactored vs Original"
echo "=============================================="
echo

# Create test directories
TEST_DIR="$SCRIPT_DIR/test_output"
ORIGINAL_DIR="$TEST_DIR/original"
REFACTORED_DIR="$TEST_DIR/refactored"

rm -rf "$TEST_DIR"
mkdir -p "$ORIGINAL_DIR" "$REFACTORED_DIR"

echo "=== Step 1: Build original (main branch) ==="
# Extract main.cpp from main branch
git show main:value_iteration/main.cpp > "$ORIGINAL_DIR/main.cpp"

# Create CMakeLists for original
cat > "$ORIGINAL_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(PacmanDP_Original)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)
add_executable(value_iteration_original main.cpp)
target_link_libraries(value_iteration_original PRIVATE OpenMP::OpenMP_CXX)
EOF

# Build original
mkdir -p "$ORIGINAL_DIR/build"
cd "$ORIGINAL_DIR/build"
cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
make -j$(nproc) > /dev/null 2>&1
echo "Original build: OK"

echo
echo "=== Step 2: Build refactored version ==="
mkdir -p "$REFACTORED_DIR/build"
cd "$REFACTORED_DIR/build"
cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
make -j$(nproc) > /dev/null 2>&1
echo "Refactored build: OK"

echo
echo "=== Step 3: Run original value iteration ==="
cd "$ORIGINAL_DIR"
./build/value_iteration_original > original_output.txt 2>&1
echo "Original run: OK"

echo
echo "=== Step 4: Run refactored value iteration ==="
cd "$REFACTORED_DIR"
./build/value_iteration -s safety.bin -t ttr.bin > value_iteration_output.txt 2>&1
echo "Refactored value iteration: OK"

echo
echo "=== Step 5: Run refactored simulation ==="
./build/simulation -s safety.bin -t ttr.bin -p 40 -1 22 -2 58 -m 255 > simulation_output.txt 2>&1
echo "Refactored simulation: OK"

echo
echo "=== Step 6: Compare simulation outputs ==="

# Extract the simulation part from original output (after "OPTIMAL PLAY SIMULATION")
sed -n '/OPTIMAL PLAY SIMULATION/,$p' "$ORIGINAL_DIR/original_output.txt" > "$ORIGINAL_DIR/simulation_only.txt"

# Extract simulation from refactored
sed -n '/OPTIMAL PLAY SIMULATION/,$p' "$REFACTORED_DIR/simulation_output.txt" > "$REFACTORED_DIR/simulation_only.txt"

# Compare
if diff -q "$ORIGINAL_DIR/simulation_only.txt" "$REFACTORED_DIR/simulation_only.txt" > /dev/null 2>&1; then
    echo "✓ Simulation outputs MATCH!"
else
    echo "✗ Simulation outputs DIFFER!"
    echo
    echo "--- Differences ---"
    diff "$ORIGINAL_DIR/simulation_only.txt" "$REFACTORED_DIR/simulation_only.txt" || true
    echo
    FAILED=1
fi

echo
echo "=== Step 7: Test multiple starting positions ==="

# Test a few different starting configurations
TEST_CONFIGS=(
    "40 22 58"   # Original test case
    "13 25 85"   # Pacman top-left area
    "130 14 106" # Pacman bottom area
    "52 16 88"   # Pacman center, ghosts at edges
)

for config in "${TEST_CONFIGS[@]}"; do
    read -r p g1 g2 <<< "$config"
    
    # Run original with this config (modify main.cpp temporarily)
    # Since original doesn't take args, we'll just verify refactored runs without crash
    cd "$REFACTORED_DIR"
    if ./build/simulation -s safety.bin -t ttr.bin -p $p -1 $g1 -2 $g2 -m 50 > /dev/null 2>&1; then
        echo "✓ Config (P=$p, G1=$g1, G2=$g2): OK"
    else
        echo "✗ Config (P=$p, G1=$g1, G2=$g2): FAILED"
        FAILED=1
    fi
done

echo
echo "=============================================="
if [ "${FAILED:-0}" -eq 1 ]; then
    echo "REGRESSION TEST: FAILED"
    echo "=============================================="
    exit 1
else
    echo "REGRESSION TEST: PASSED"
    echo "=============================================="
    exit 0
fi

