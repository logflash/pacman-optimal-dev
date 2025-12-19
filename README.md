# Pacman Optimal Play via Value Iteration

Minimax value iteration for Pacman with 2 adversarial ghosts on a 12x12 maze.

## Ghost Behavior

Ghosts are **fully coordinated adversaries**. The value iteration considers all possible joint ghost moves and assumes they choose the worst-case combination for Pacman:

- Both ghosts move simultaneously each turn
- They jointly minimize Pacman's safety value (and TTR when tied)
- This models a "team" of ghosts with perfect coordination
- Not independent random or greedy agents—optimal adversarial play

## Super Pellet Implementation

State space: `[pacman_pos][ghost1][ghost2][power_state]`

- **Ghost positions**: 0-143 (alive) or 144 (dead/permadeath)
- **Power states**: 0 = pellet exists, 1-N = powered (N turns left), N+1 = expired
- Powered Pacman can eat ghosts (permanent kill)
- Configurable power duration (`-d`) and pellet position (`-l`)

## Build

```bash
cd value_iteration
mkdir -p build && cd build
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

### Super Pellet (with speed boost when powered)

```bash
# Compute values (~minutes, depends on power duration)
# -o output file, -d power duration, -l pellet position
./value_iteration_super -o values.bin -d 6 -l 64

# Run simulation
# -v values file, -p pacman, -1/-2 ghosts, -m max steps
./simulation_super -v values.bin -p 52 -1 22 -2 58 -m 50
```

When powered, Pacman moves twice per turn (double move).

### Multi-Pellet (Optimized)

```bash
# Compute values with 2 pellets (compact encoding + gzip compression)
./value_iteration_opt -n 2 -d 6 -o values_2p.bin.gz

# Compute values with 4 pellets
./value_iteration_opt -n 4 -d 6 -o values_4p.bin.gz

# Custom pellet positions
./value_iteration_opt -n 2 -d 10 -p 25,118 -o custom.bin.gz

# Run simulation
./simulation_opt -v values_2p.bin.gz -p 64 -1 52 -2 76 -m 50
```

**Size comparison (4 pellets, D=6):**
| Format | File Size |
|--------|-----------|
| Unoptimized | 4.4 GB |
| Optimized | **5 MB** |

Optimizations:
- **Compact power state encoding**: Only valid states enumerated (512 → 106 states)
- **gzip compression**: ~100-200× compression on value tables

### A* Evaluation Framework

Evaluate A* pathfinding strategies against optimal and greedy ghost play:

```bash
# Run full evaluation (all strategies vs all ghost types)
./eval_astar -v values_4p_corners.bin.gz -n 1000 -t 100 --all -o eval_results.csv

# Generate all visualizations
python3 ../visualize.py eval_results.csv --analysis-prefix analysis_4p --output-dir .
```

**Pac-Man strategies evaluated:**
- `Optimal`: Value iteration optimal policy (100% survival from safe states)
- `A* + Safety Heuristic`: A* with penalty for unsafe positions
- `A* + FRS Filter`: A* avoiding forward-reachable ghost positions
- `A* + Safety Filter`: A* allowing only provably safe moves (uses value table)

**Ghost strategies:**
- `Optimal Ghost`: Minimax optimal play from value iteration
- `Greedy BFS Ghost`: Each ghost independently BFS toward Pac-Man

**Output plots:**
- `survival_comparison.png`: Survival/win rates by strategy
- `intervention_rates.png`: Filter intervention frequency
- `survival_by_distance.png`: Survival vs initial ghost distance
- `death_heatmap.png`: Death location heatmaps (3x2 grid)
- `win_time_distribution.png`: Win time statistics
- `safety_heatmap.png`: Position-wise safety rates
- `critical_positions.png`: Choke points and trapped positions

### Value Analysis Tools

```bash
# Run all analyses and export CSVs
./analyze_values -v values.bin --all --export-all analysis

# For multi-pellet configs (compressed)
./analyze_values -v values_4p_corners.bin.gz --export-all analysis_4p
```

**Available analyses:**
- `--basic`: Safe/unsafe state counts, average TTR
- `--position`: Per-cell safety rates (heatmap)
- `--ghost`: Ghost configuration danger analysis
- `--critical`: Choke point identification
- `--threshold`: Safety filter threshold sensitivity
