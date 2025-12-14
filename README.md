# Pacman Optimal Play via Value Iteration

Minimax value iteration for Pacman with 2 adversarial ghosts on a 12x12 maze.

## Super Pellet Implementation

State space: `[pacman_pos][ghost1][ghost2][power_state]`

- **Ghost positions**: 0-143 (alive) or 144 (dead/permadeath)
- **Power states**: 0 = pellet exists, 1-N = powered (N turns left), N+1 = expired
- Powered Pacman can eat ghosts (permanent kill)
- Configurable power duration (`-d`) and pellet position (`-l`)

## Build

```bash
cd value_iteration
mkdir build && cd build
cmake .. && make
```

## Usage

### Basic (no super pellet)

```bash
# Compute values (~seconds)
./value_iteration -s safety.bin -t ttr.bin

# Run simulation
./simulation -s safety.bin -t ttr.bin -p 40 -1 22 -2 58
```

### Super Pellet

```bash
# Compute values (~minutes, depends on power duration)
./value_iteration_super -o values_super.bin -d 10 -l 64

# Run simulation
./simulation_super -i values_super.bin -p 40 -1 22 -2 58
```
