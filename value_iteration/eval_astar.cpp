/* A* Evaluation Framework for Pac-Man
 *
 * Evaluates A* pathfinding strategies against optimal and greedy ghost play.
 * Outputs CSV with survival rates, intervention statistics, and detailed metrics.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <zlib.h>

#include "forward_reachable.h"
#include "safety_filter.h"

using namespace std;

// ============================================================================
// Maze Configuration
// ============================================================================

#define MAZE_ROWS 12
#define MAZE_COLS 12
#define MAZE_CELLS (MAZE_ROWS * MAZE_COLS)

#define DEAD MAZE_CELLS
#define GHOST_STATES (MAZE_CELLS + 1)

#define MAX_PELLETS 4
#define MAX_POWER_DURATION 30

uint32_t maze[MAZE_ROWS] = {0};

int num_pellets = 2;
int power_duration = 6;
int pellet_positions[MAX_PELLETS] = {0};
int num_pellet_masks = 4;
int total_power_states = 0;

uint8_t* safety_value = nullptr;
uint8_t* ttr_g_value = nullptr;
uint8_t* ttr_p_value = nullptr;
size_t table_size = 0;

std::mt19937 rng;

// ============================================================================
// Statistics Tracking
// ============================================================================

struct DetailedStats {
    // Per-run aggregates
    int total_ticks = 0;
    int frs_interventions = 0;      // Times FRS blocked A*'s preferred move
    int safety_interventions = 0;   // Times safety filter blocked A*'s preferred move
    int safety_fallbacks = 0;       // Times safety filter had NO safe moves
    int frs_recomputations = 0;     // Times FRS was recomputed
    
    // Goal tracking
    int ticks_targeting_pellet = 0;
    int ticks_targeting_ghost = 0;
    int ticks_no_target = 0;
    
    int moves_blocked_by_frs = 0;
    int moves_blocked_by_safety = 0;
    
    int ticks_in_safe_state = 0;
    int ticks_in_unsafe_state = 0;
    int safe_to_unsafe_transitions = 0;
    int first_pellet_tick = -1;
    
    void reset() {
        total_ticks = 0;
        frs_interventions = 0;
        safety_interventions = 0;
        safety_fallbacks = 0;
        frs_recomputations = 0;
        ticks_targeting_pellet = 0;
        ticks_targeting_ghost = 0;
        ticks_no_target = 0;
        moves_blocked_by_frs = 0;
        moves_blocked_by_safety = 0;
        ticks_in_safe_state = 0;
        ticks_in_unsafe_state = 0;
        safe_to_unsafe_transitions = 0;
        first_pellet_tick = -1;
    }
};

// Global stats for current simulation
DetailedStats current_stats;

// ============================================================================
// Basic Utilities
// ============================================================================

inline int row(int idx) { return idx / MAZE_COLS; }
inline int col(int idx) { return idx % MAZE_COLS; }

inline bool is_free(int r, int c) {
    return r >= 0 && r < MAZE_ROWS && c >= 0 && c < MAZE_COLS && ((maze[r] >> c) & 1);
}

inline bool is_valid_pos(int idx) {
    return idx >= 0 && idx < MAZE_CELLS && is_free(row(idx), col(idx));
}

inline void get_neighbors(int idx, int* neighbors, int &count) {
    int r = row(idx), c = col(idx);
    count = 0;
    int dr[4] = {-1,1,0,0}, dc[4] = {0,0,-1,1};
    for (int k = 0; k < 4; k++) {
        int nr = r + dr[k], nc = c + dc[k];
        if (nr >= 0 && nr < MAZE_ROWS && nc >= 0 && nc < MAZE_COLS && is_free(nr, nc)) {
            neighbors[count++] = nr * MAZE_COLS + nc;
        }
    }
}

inline int manhattan_dist(int a, int b) {
    return abs(row(a) - row(b)) + abs(col(a) - col(b));
}

// ============================================================================
// Power State Encoding
// ============================================================================

inline int encode_power_state(int pellet_mask, int power_timer) {
    if (power_timer == 0) {
        return pellet_mask;
    } else {
        return num_pellet_masks + pellet_mask * power_duration + (power_timer - 1);
    }
}

inline void decode_power_state(int state, int& pellet_mask, int& power_timer) {
    if (state < num_pellet_masks) {
        pellet_mask = state;
        power_timer = 0;
    } else {
        int powered_idx = state - num_pellet_masks;
        pellet_mask = powered_idx / power_duration;
        power_timer = (powered_idx % power_duration) + 1;
    }
}

inline bool is_powered(int power_timer) {
    return power_timer >= 1 && power_timer <= power_duration;
}

inline bool pellet_exists(int pellet_mask, int pellet_idx) {
    return (pellet_mask >> pellet_idx) & 1;
}

inline int remove_pellet(int pellet_mask, int pellet_idx) {
    return pellet_mask & ~(1 << pellet_idx);
}

inline int get_pellet_at(int pos, int pellet_mask) {
    for (int i = 0; i < num_pellets; i++) {
        if (pellet_exists(pellet_mask, i) && pos == pellet_positions[i]) {
            return i;
        }
    }
    return -1;
}

inline size_t value_index(int p, int g1, int g2, int pw) {
    return ((size_t)p * GHOST_STATES * GHOST_STATES +
            (size_t)g1 * GHOST_STATES +
            (size_t)g2) * total_power_states + pw;
}

// ============================================================================
// File Loading
// ============================================================================

struct ValueFileHeaderOpt {
    char magic[4];
    uint8_t version;
    uint8_t value_type;
    uint16_t maze_rows;
    uint16_t maze_cols;
    uint8_t num_ghosts;
    uint8_t num_pellets;
    uint8_t power_duration;
    uint8_t compression;
    uint16_t pellet_positions[MAX_PELLETS];
    uint32_t total_power_states;
    uint32_t uncompressed_size;
};

bool load_values(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Could not open " << filename << endl;
        return false;
    }

    ValueFileHeaderOpt header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (memcmp(header.magic, "PVAO", 4) != 0) {
        cerr << "Error: Invalid file format" << endl;
        return false;
    }

    num_pellets = header.num_pellets;
    power_duration = header.power_duration;
    num_pellet_masks = 1 << num_pellets;
    total_power_states = header.total_power_states;

    for (int i = 0; i < MAX_PELLETS; i++) {
        pellet_positions[i] = header.pellet_positions[i];
    }

    table_size = (size_t)MAZE_CELLS * GHOST_STATES * GHOST_STATES * total_power_states;

    file.read(reinterpret_cast<char*>(maze), sizeof(maze));

    try {
        safety_value = new uint8_t[table_size];
        ttr_g_value = new uint8_t[table_size];
        ttr_p_value = new uint8_t[table_size];
    } catch (const bad_alloc& e) {
        cerr << "Error: Failed to allocate memory" << endl;
        return false;
    }

    if (header.compression == 1) {
        uint32_t compressed_size;
        file.read(reinterpret_cast<char*>(&compressed_size), sizeof(compressed_size));

        vector<uint8_t> compressed(compressed_size);
        file.read(reinterpret_cast<char*>(compressed.data()), compressed_size);

        uLongf uncompressed_size = header.uncompressed_size;
        vector<uint8_t> uncompressed(uncompressed_size);

        int result = uncompress(uncompressed.data(), &uncompressed_size,
                               compressed.data(), compressed_size);

        if (result != Z_OK) {
            cerr << "Error: Decompression failed" << endl;
            return false;
        }

        memcpy(safety_value, uncompressed.data(), table_size);
        memcpy(ttr_g_value, uncompressed.data() + table_size, table_size);
        memcpy(ttr_p_value, uncompressed.data() + 2 * table_size, table_size);
    } else {
        file.read(reinterpret_cast<char*>(safety_value), table_size);
        file.read(reinterpret_cast<char*>(ttr_g_value), table_size);
        file.read(reinterpret_cast<char*>(ttr_p_value), table_size);
    }

    file.close();
    return true;
}

void free_tables() {
    delete[] safety_value;
    delete[] ttr_g_value;
    delete[] ttr_p_value;
    safety_value = ttr_g_value = ttr_p_value = nullptr;
}

// ============================================================================
// Ghost Strategies
// ============================================================================

enum class GhostStrategy {
    OPTIMAL,
    GREEDY_BFS
};

string ghost_strategy_name(GhostStrategy s) {
    switch (s) {
        case GhostStrategy::OPTIMAL: return "Optimal";
        case GhostStrategy::GREEDY_BFS: return "Greedy_BFS";
    }
    return "Unknown";
}

int get_greedy_bfs_ghost_move(int ghost_pos, int pacman_pos) {
    if (ghost_pos == DEAD) return DEAD;
    if (ghost_pos == pacman_pos) return ghost_pos;
    
    const int MAX_NODES = MAZE_CELLS;
    int parent[MAX_NODES];
    bool visited[MAX_NODES];
    
    for (int i = 0; i < MAX_NODES; i++) {
        parent[i] = -1;
        visited[i] = false;
    }
    
    int queue[MAX_NODES];
    int front = 0, back = 0;
    
    queue[back++] = ghost_pos;
    visited[ghost_pos] = true;
    
    bool found = false;
    
    while (front < back && !found) {
        int current = queue[front++];
        
        int neighbors[4], count;
        get_neighbors(current, neighbors, count);
        
        for (int i = 0; i < count; i++) {
            int neighbor = neighbors[i];
            
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                parent[neighbor] = current;
                queue[back++] = neighbor;
                
                if (neighbor == pacman_pos) {
                    found = true;
                    break;
                }
            }
        }
    }
    
    if (!found) {
        int neighbors[4], count;
        get_neighbors(ghost_pos, neighbors, count);
        return (count > 0) ? neighbors[0] : ghost_pos;
    }
    
    int node = pacman_pos;
    while (parent[node] != ghost_pos && parent[node] != -1) {
        node = parent[node];
    }
    
    return (parent[node] == ghost_pos) ? node : ghost_pos;
}

void get_optimal_ghost_moves(int old_p, int new_p, int g1, int g2, 
                             int pellet_mask, int power_timer,
                             int& best_g1, int& best_g2) {
    bool g1_alive = (g1 != DEAD);
    bool g2_alive = (g2 != DEAD);
    bool powered = is_powered(power_timer);
    
    int g1_neighbors[4], g1_n = 0;
    int g2_neighbors[4], g2_n = 0;
    
    if (g1_alive) get_neighbors(g1, g1_neighbors, g1_n);
    if (g2_alive) get_neighbors(g2, g2_neighbors, g2_n);
    
    best_g1 = g1_alive ? (g1_n > 0 ? g1_neighbors[0] : g1) : DEAD;
    best_g2 = g2_alive ? (g2_n > 0 ? g2_neighbors[0] : g2) : DEAD;
    
    if (!g1_alive && !g2_alive) return;
    
    uint8_t worst_safety = 1;
    uint8_t worst_ttr_g = 255;
    
    int g1_moves = g1_alive ? g1_n : 1;
    int g2_moves = g2_alive ? g2_n : 1;
    
    for (int gi1 = 0; gi1 < g1_moves; gi1++) {
        for (int gi2 = 0; gi2 < g2_moves; gi2++) {
            int ng1 = g1_alive ? g1_neighbors[gi1] : DEAD;
            int ng2 = g2_alive ? g2_neighbors[gi2] : DEAD;
            
            if (!powered) {
                bool clip1 = g1_alive && (old_p == ng1 && g1 == new_p);
                bool clip2 = g2_alive && (old_p == ng2 && g2 == new_p);
                bool catches = (ng1 == new_p || ng2 == new_p || clip1 || clip2);
                
                if (catches) {
                    best_g1 = ng1;
                    best_g2 = ng2;
                    return;
                }
            }
            
            int pw = encode_power_state(pellet_mask, power_timer);
            size_t idx = value_index(new_p, ng1, ng2, pw);
            uint8_t s = safety_value[idx];
            uint8_t tg = ttr_g_value[idx];
            
            if (s < worst_safety || (s == worst_safety && tg < worst_ttr_g)) {
                worst_safety = s;
                worst_ttr_g = tg;
                best_g1 = ng1;
                best_g2 = ng2;
            }
        }
    }
}

void get_ghost_moves(GhostStrategy strategy, int old_p, int new_p, 
                     int g1, int g2, int pellet_mask, int power_timer,
                     int& ng1, int& ng2) {
    switch (strategy) {
        case GhostStrategy::OPTIMAL:
            get_optimal_ghost_moves(old_p, new_p, g1, g2, pellet_mask, power_timer, ng1, ng2);
            break;
        case GhostStrategy::GREEDY_BFS:
            ng1 = get_greedy_bfs_ghost_move(g1, new_p);
            ng2 = get_greedy_bfs_ghost_move(g2, new_p);
            break;
    }
}

// ============================================================================
// Pac-Man Strategies with Detailed Tracking
// ============================================================================

enum class PacmanStrategy {
    OPTIMAL,
    ASTAR_SAFETY,
    ASTAR_FRS,
    ASTAR_FILTER
};

string pacman_strategy_name(PacmanStrategy s) {
    switch (s) {
        case PacmanStrategy::OPTIMAL: return "Optimal";
        case PacmanStrategy::ASTAR_SAFETY: return "A*_Safety_Heuristic";
        case PacmanStrategy::ASTAR_FRS: return "A*_FRS_Filter";
        case PacmanStrategy::ASTAR_FILTER: return "A*_Safety_Filter";
    }
    return "Unknown";
}

int find_nearest_pellet(int pos, int pellet_mask) {
    int best_pellet = -1;
    int best_dist = 999;
    
    for (int i = 0; i < num_pellets; i++) {
        if (pellet_exists(pellet_mask, i)) {
            int dist = manhattan_dist(pos, pellet_positions[i]);
            if (dist < best_dist) {
                best_dist = dist;
                best_pellet = i;
            }
        }
    }
    
    return best_pellet >= 0 ? pellet_positions[best_pellet] : -1;
}

int find_nearest_ghost(int pos, int g1, int g2) {
    int best = -1;
    int best_dist = 999;
    
    if (g1 != DEAD) {
        int d = manhattan_dist(pos, g1);
        if (d < best_dist) { best_dist = d; best = g1; }
    }
    if (g2 != DEAD) {
        int d = manhattan_dist(pos, g2);
        if (d < best_dist) { best_dist = d; best = g2; }
    }
    
    return best;
}

// Basic A* - returns preferred move
int astar_get_preferred_move(int start, int goal) {
    if (start == goal) return start;
    
    const int MAX_NODES = MAZE_CELLS;
    int g_score[MAX_NODES];
    int f_score[MAX_NODES];
    int parent[MAX_NODES];
    bool closed[MAX_NODES];
    
    for (int i = 0; i < MAX_NODES; i++) {
        g_score[i] = 999999;
        f_score[i] = 999999;
        parent[i] = -1;
        closed[i] = false;
    }
    
    int open_set[MAX_NODES];
    int open_size = 0;
    
    g_score[start] = 0;
    f_score[start] = manhattan_dist(start, goal);
    open_set[open_size++] = start;
    
    while (open_size > 0) {
        int min_idx = 0;
        for (int i = 1; i < open_size; i++) {
            if (f_score[open_set[i]] < f_score[open_set[min_idx]]) {
                min_idx = i;
            }
        }
        
        int current = open_set[min_idx];
        open_set[min_idx] = open_set[--open_size];
        
        if (current == goal) break;
        
        closed[current] = true;
        
        int neighbors[4], count;
        get_neighbors(current, neighbors, count);
        
        for (int i = 0; i < count; i++) {
            int neighbor = neighbors[i];
            if (closed[neighbor]) continue;
            
            int tentative_g = g_score[current] + 1;
            
            if (tentative_g < g_score[neighbor]) {
                parent[neighbor] = current;
                g_score[neighbor] = tentative_g;
                f_score[neighbor] = tentative_g + manhattan_dist(neighbor, goal);
                
                bool in_open = false;
                for (int j = 0; j < open_size; j++) {
                    if (open_set[j] == neighbor) { in_open = true; break; }
                }
                if (!in_open && open_size < MAX_NODES) {
                    open_set[open_size++] = neighbor;
                }
            }
        }
    }
    
    int node = goal;
    while (parent[node] != start && parent[node] != -1) {
        node = parent[node];
    }
    
    return (parent[node] == start) ? node : start;
}

// A* with safety heuristic
int astar_safety(int start, int goal, int g1, int g2, int pellet_mask, int power_timer) {
    if (start == goal) return start;
    
    const int MAX_NODES = MAZE_CELLS;
    int g_score[MAX_NODES];
    int f_score[MAX_NODES];
    int parent[MAX_NODES];
    bool closed[MAX_NODES];
    
    for (int i = 0; i < MAX_NODES; i++) {
        g_score[i] = 999999;
        f_score[i] = 999999;
        parent[i] = -1;
        closed[i] = false;
    }
    
    int open_set[MAX_NODES];
    int open_size = 0;
    
    auto safety_penalty = [&](int pos) -> int {
        int pw = encode_power_state(pellet_mask, power_timer);
        size_t idx = value_index(pos, g1, g2, pw);
        uint8_t s = safety_value[idx];
        return (s == 0) ? 100 : 0;
    };
    
    g_score[start] = 0;
    f_score[start] = manhattan_dist(start, goal) + safety_penalty(start);
    open_set[open_size++] = start;
    
    while (open_size > 0) {
        int min_idx = 0;
        for (int i = 1; i < open_size; i++) {
            if (f_score[open_set[i]] < f_score[open_set[min_idx]]) {
                min_idx = i;
            }
        }
        
        int current = open_set[min_idx];
        open_set[min_idx] = open_set[--open_size];
        
        if (current == goal) break;
        
        closed[current] = true;
        
        int neighbors[4], count;
        get_neighbors(current, neighbors, count);
        
        for (int i = 0; i < count; i++) {
            int neighbor = neighbors[i];
            if (closed[neighbor]) continue;
            
            int tentative_g = g_score[current] + 1;
            
            if (tentative_g < g_score[neighbor]) {
                parent[neighbor] = current;
                g_score[neighbor] = tentative_g;
                f_score[neighbor] = tentative_g + manhattan_dist(neighbor, goal) + safety_penalty(neighbor);
                
                bool in_open = false;
                for (int j = 0; j < open_size; j++) {
                    if (open_set[j] == neighbor) { in_open = true; break; }
                }
                if (!in_open && open_size < MAX_NODES) {
                    open_set[open_size++] = neighbor;
                }
            }
        }
    }
    
    int node = goal;
    while (parent[node] != start && parent[node] != -1) {
        node = parent[node];
    }
    
    return (parent[node] == start) ? node : start;
}

// A* with FRS filtering - tracks interventions
int astar_frs_tracked(int start, int goal, int g1, int g2, int pellet_mask, int power_timer,
                      const ForwardReachableSet& frs1, const ForwardReachableSet& frs2,
                      bool& intervention_occurred) {
    intervention_occurred = false;
    
    if (start == goal) return start;
    
    // First, get what basic A* would prefer
    int preferred = astar_get_preferred_move(start, goal);
    
    // Check if preferred move is blocked by FRS
    bool preferred_blocked = false;
    if (!is_powered(power_timer)) {
        if (is_frs_dangerous(frs1, frs2, preferred, 1)) {
            preferred_blocked = true;
            current_stats.moves_blocked_by_frs++;
        }
    }
    
    // Now run A* with FRS filtering
    const int MAX_NODES = MAZE_CELLS;
    int g_score[MAX_NODES];
    int f_score[MAX_NODES];
    int parent[MAX_NODES];
    bool closed[MAX_NODES];
    
    for (int i = 0; i < MAX_NODES; i++) {
        g_score[i] = 999999;
        f_score[i] = 999999;
        parent[i] = -1;
        closed[i] = false;
    }
    
    int open_set[MAX_NODES];
    int open_size = 0;
    
    g_score[start] = 0;
    f_score[start] = manhattan_dist(start, goal);
    open_set[open_size++] = start;
    
    while (open_size > 0) {
        int min_idx = 0;
        for (int i = 1; i < open_size; i++) {
            if (f_score[open_set[i]] < f_score[open_set[min_idx]]) {
                min_idx = i;
            }
        }
        
        int current = open_set[min_idx];
        open_set[min_idx] = open_set[--open_size];
        
        if (current == goal) break;
        
        closed[current] = true;
        
        int neighbors[4], count;
        get_neighbors(current, neighbors, count);
        
        int time_step = g_score[current] + 1;
        
        for (int i = 0; i < count; i++) {
            int neighbor = neighbors[i];
            if (closed[neighbor]) continue;
            
            // FRS filter
            if (!is_powered(power_timer)) {
                if (is_frs_dangerous(frs1, frs2, neighbor, time_step)) {
                    continue;
                }
            }
            
            int tentative_g = g_score[current] + 1;
            
            if (tentative_g < g_score[neighbor]) {
                parent[neighbor] = current;
                g_score[neighbor] = tentative_g;
                f_score[neighbor] = tentative_g + manhattan_dist(neighbor, goal);
                
                bool in_open = false;
                for (int j = 0; j < open_size; j++) {
                    if (open_set[j] == neighbor) { in_open = true; break; }
                }
                if (!in_open && open_size < MAX_NODES) {
                    open_set[open_size++] = neighbor;
                }
            }
        }
    }
    
    int node = goal;
    while (parent[node] != start && parent[node] != -1) {
        node = parent[node];
    }
    
    int result = (parent[node] == start) ? node : start;
    
    // Check if intervention occurred
    if (preferred_blocked && result != preferred) {
        intervention_occurred = true;
    }
    
    return result;
}

// Check if a first move (np1) is safe against all ghost responses
// This must match the value iteration logic exactly
bool is_move_safe(int p, int np1, int g1, int g2, int pellet_mask, int power_timer) {
    bool g1_alive = (g1 != DEAD);
    bool g2_alive = (g2 != DEAD);
    
    // Pre-compute ghost neighbors
    int g1_neighbors[4], g1_n = 0;
    int g2_neighbors[4], g2_n = 0;
    
    if (g1_alive) get_neighbors(g1, g1_neighbors, g1_n);
    if (g2_alive) get_neighbors(g2, g2_neighbors, g2_n);
    
    int g1_moves = g1_alive ? g1_n : 1;
    int g2_moves = g2_alive ? g2_n : 1;
    
    // Compute power state after first move
    int new_pellet_mask = pellet_mask;
    int new_power_timer = power_timer;
    
    int pellet_eaten = get_pellet_at(np1, pellet_mask);
    if (pellet_eaten >= 0) {
        new_pellet_mask = remove_pellet(pellet_mask, pellet_eaten);
        new_power_timer = power_duration;
    }
    
    // KEY: will_be_powered if already powered OR just ate a pellet
    bool will_be_powered = is_powered(new_power_timer);
    
    // Get second move options (for powered mode, we need to check all np2)
    int neighbors2[5], count2;
    if (will_be_powered) {
        get_neighbors(np1, neighbors2, count2);
        neighbors2[count2++] = np1;  // Can stay in place on move 2
    } else {
        count2 = 1;
        neighbors2[0] = np1;  // np2 == np1 for unpowered
    }
    
    // For a move to be safe, there must exist at least one np2 that is safe
    // against ALL ghost responses (this matches value iteration's max over np2)
    for (int i2 = 0; i2 < count2; i2++) {
        int np2 = neighbors2[i2];
        
        // Check pellet on second move
        int final_pellet_mask = new_pellet_mask;
        int final_power_timer = new_power_timer;
        
        if (will_be_powered && np2 != np1) {
            int pellet_eaten2 = get_pellet_at(np2, new_pellet_mask);
            if (pellet_eaten2 >= 0) {
                final_pellet_mask = remove_pellet(new_pellet_mask, pellet_eaten2);
                final_power_timer = power_duration;
            }
        }
        
        bool np2_safe_against_all_ghosts = true;
        
        // For powered mode: check if ghosts are caught on first move
        bool g1_caught_move1 = will_be_powered && g1_alive && (np1 == g1);
        bool g2_caught_move1 = will_be_powered && g2_alive && (np1 == g2);
        
        int g1_iter = g1_caught_move1 ? 1 : g1_moves;
        int g2_iter = g2_caught_move1 ? 1 : g2_moves;
        
        for (int gi1 = 0; gi1 < g1_iter && np2_safe_against_all_ghosts; gi1++) {
            for (int gi2 = 0; gi2 < g2_iter && np2_safe_against_all_ghosts; gi2++) {
                int ng1 = g1_caught_move1 ? DEAD : (g1_alive ? g1_neighbors[gi1] : DEAD);
                int ng2 = g2_caught_move1 ? DEAD : (g2_alive ? g2_neighbors[gi2] : DEAD);
                
                int result_g1 = ng1;
                int result_g2 = ng2;
                bool pacman_dies = false;
                
                if (will_be_powered) {
                    // Powered mode: check catches/clips on second move
                    bool g1_alive_after_move1 = g1_alive && !g1_caught_move1;
                    bool g2_alive_after_move1 = g2_alive && !g2_caught_move1;
                    
                    bool catch1_m2 = g1_alive_after_move1 && (np2 == ng1);
                    bool catch2_m2 = g2_alive_after_move1 && (np2 == ng2);
                    
                    bool clip1 = g1_alive_after_move1 && (np1 == ng1 && g1 == np2);
                    bool clip2 = g2_alive_after_move1 && (np1 == ng2 && g2 == np2);
                    
                    if (g1_caught_move1 || catch1_m2 || clip1) result_g1 = DEAD;
                    if (g2_caught_move1 || catch2_m2 || clip2) result_g2 = DEAD;
                } else {
                    // Unpowered mode: check collision/clip
                    bool clip1 = g1_alive && (p == ng1 && g1 == np1);
                    bool clip2 = g2_alive && (p == ng2 && g2 == np1);
                    bool collide1 = g1_alive && (np1 == ng1 || clip1);
                    bool collide2 = g2_alive && (np1 == ng2 || clip2);
                    
                    if (collide1 || collide2) pacman_dies = true;
                }
                
                if (pacman_dies) {
                    np2_safe_against_all_ghosts = false;
                } else if (result_g1 == DEAD && result_g2 == DEAD) {
                    // Both ghosts dead = always safe, continue checking
                } else {
                    // Decrement power timer for lookup
                    int result_power_timer = final_power_timer;
                    if (is_powered(result_power_timer)) {
                        result_power_timer--;
                    }
                    
                    int pw = encode_power_state(final_pellet_mask, result_power_timer);
                    size_t idx = value_index(np2, result_g1, result_g2, pw);
                    if (safety_value[idx] == 0) {
                        np2_safe_against_all_ghosts = false;
                    }
                }
            }
        }
        
        // If this np2 is safe against all ghost responses, the move np1 is safe
        if (np2_safe_against_all_ghosts) {
            return true;
        }
    }
    
    // No np2 was safe against all ghost responses
    return false;
}

// Find the best safe np2 for a given np1 (for powered mode)
// Returns the np2 that is safe and closest to the goal, or -1 if none
int find_safe_np2(int p, int np1, int g1, int g2, int pellet_mask, int power_timer, int goal) {
    bool g1_alive = (g1 != DEAD);
    bool g2_alive = (g2 != DEAD);
    
    // Pre-compute ghost neighbors
    int g1_neighbors[4], g1_n = 0;
    int g2_neighbors[4], g2_n = 0;
    
    if (g1_alive) get_neighbors(g1, g1_neighbors, g1_n);
    if (g2_alive) get_neighbors(g2, g2_neighbors, g2_n);
    
    int g1_moves = g1_alive ? g1_n : 1;
    int g2_moves = g2_alive ? g2_n : 1;
    
    // Compute power state after first move
    int new_pellet_mask = pellet_mask;
    int new_power_timer = power_timer;
    
    int pellet_eaten = get_pellet_at(np1, pellet_mask);
    if (pellet_eaten >= 0) {
        new_pellet_mask = remove_pellet(pellet_mask, pellet_eaten);
        new_power_timer = power_duration;
    }
    
    bool will_be_powered = is_powered(new_power_timer);
    if (!will_be_powered) {
        return np1;  // Not powered, np2 == np1
    }
    
    // For powered mode: check if ghosts are caught on first move
    bool g1_caught_move1 = g1_alive && (np1 == g1);
    bool g2_caught_move1 = g2_alive && (np1 == g2);
    
    int g1_iter = g1_caught_move1 ? 1 : g1_moves;
    int g2_iter = g2_caught_move1 ? 1 : g2_moves;
    
    // Get second move options
    int neighbors2[5], count2;
    get_neighbors(np1, neighbors2, count2);
    neighbors2[count2++] = np1;  // Can stay in place
    
    int best_np2 = -1;
    int best_dist = 999999;
    
    for (int i2 = 0; i2 < count2; i2++) {
        int np2 = neighbors2[i2];
        
        // Check pellet on second move
        int final_pellet_mask = new_pellet_mask;
        int final_power_timer = new_power_timer;
        
        if (np2 != np1) {
            int pellet_eaten2 = get_pellet_at(np2, new_pellet_mask);
            if (pellet_eaten2 >= 0) {
                final_pellet_mask = remove_pellet(new_pellet_mask, pellet_eaten2);
                final_power_timer = power_duration;
            }
        }
        
        bool np2_safe_against_all_ghosts = true;
        
        for (int gi1 = 0; gi1 < g1_iter && np2_safe_against_all_ghosts; gi1++) {
            for (int gi2 = 0; gi2 < g2_iter && np2_safe_against_all_ghosts; gi2++) {
                int ng1 = g1_caught_move1 ? DEAD : (g1_alive ? g1_neighbors[gi1] : DEAD);
                int ng2 = g2_caught_move1 ? DEAD : (g2_alive ? g2_neighbors[gi2] : DEAD);
                
                int result_g1 = ng1;
                int result_g2 = ng2;
                
                // Powered mode: check catches/clips on second move
                bool g1_alive_after_move1 = g1_alive && !g1_caught_move1;
                bool g2_alive_after_move1 = g2_alive && !g2_caught_move1;
                
                bool catch1_m2 = g1_alive_after_move1 && (np2 == ng1);
                bool catch2_m2 = g2_alive_after_move1 && (np2 == ng2);
                
                bool clip1 = g1_alive_after_move1 && (np1 == ng1 && g1 == np2);
                bool clip2 = g2_alive_after_move1 && (np1 == ng2 && g2 == np2);
                
                if (g1_caught_move1 || catch1_m2 || clip1) result_g1 = DEAD;
                if (g2_caught_move1 || catch2_m2 || clip2) result_g2 = DEAD;
                
                if (result_g1 == DEAD && result_g2 == DEAD) {
                    // Both ghosts dead = always safe
                } else {
                    // Decrement power timer for lookup
                    int result_power_timer = final_power_timer;
                    if (is_powered(result_power_timer)) {
                        result_power_timer--;
                    }
                    
                    int pw = encode_power_state(final_pellet_mask, result_power_timer);
                    size_t idx = value_index(np2, result_g1, result_g2, pw);
                    if (safety_value[idx] == 0) {
                        np2_safe_against_all_ghosts = false;
                    }
                }
            }
        }
        
        if (np2_safe_against_all_ghosts) {
            int dist = manhattan_dist(np2, goal);
            if (best_np2 < 0 || dist < best_dist) {
                best_np2 = np2;
                best_dist = dist;
            }
        }
    }
    
    return best_np2;
}

// A* with safety filter - tracks interventions
// Also returns safe_np2 for powered mode
int astar_filter_tracked(int start, int goal, int g1, int g2, int pellet_mask, int power_timer,
                         bool& intervention_occurred, bool& fallback_occurred, int* out_safe_np2 = nullptr) {
    intervention_occurred = false;
    fallback_occurred = false;
    if (out_safe_np2) *out_safe_np2 = start;
    
    if (start == goal) return start;
    
    // First, get what basic A* would prefer
    int preferred = astar_get_preferred_move(start, goal);
    
    // Get all valid moves
    int neighbors[4], count;
    get_neighbors(start, neighbors, count);
    
    if (count == 0) return start;
    
    // Filter to safe moves
    vector<pair<int, int>> safe_moves;  // (np1, best_safe_np2)
    bool preferred_is_safe = false;
    int preferred_np2 = start;
    
    for (int i = 0; i < count; i++) {
        int np = neighbors[i];
        
        if (is_move_safe(start, np, g1, g2, pellet_mask, power_timer)) {
            int safe_np2 = find_safe_np2(start, np, g1, g2, pellet_mask, power_timer, goal);
            if (safe_np2 >= 0) {
                safe_moves.push_back({np, safe_np2});
                if (np == preferred) {
                    preferred_is_safe = true;
                    preferred_np2 = safe_np2;
                }
            }
        } else {
            if (np == preferred) current_stats.moves_blocked_by_safety++;
        }
    }
    
    // If preferred move is safe, use it
    if (preferred_is_safe) {
        if (out_safe_np2) *out_safe_np2 = preferred_np2;
        return preferred;
    }
    
    // Intervention occurred - preferred was blocked
    intervention_occurred = true;
    
    // If no safe moves, fall back to basic A*
    if (safe_moves.empty()) {
        fallback_occurred = true;
        return preferred;  // Use A*'s preference as emergency
    }
    
    // Among safe moves, pick closest to goal
    int best_move = safe_moves[0].first;
    int best_np2 = safe_moves[0].second;
    int best_dist = manhattan_dist(safe_moves[0].first, goal);
    
    for (size_t i = 1; i < safe_moves.size(); i++) {
        int dist = manhattan_dist(safe_moves[i].first, goal);
        if (dist < best_dist) {
            best_dist = dist;
            best_move = safe_moves[i].first;
            best_np2 = safe_moves[i].second;
        }
    }
    
    if (out_safe_np2) *out_safe_np2 = best_np2;
    return best_move;
}

// Unified optimal Pac-Man move function
// Handles both powered and unpowered modes, including the case where
// eating a pellet on the first move triggers powered mode
// Returns the first move (np1). For powered mode, also sets best_np2.
int get_optimal_pacman_move(int p, int g1, int g2, int pellet_mask, int power_timer,
                            int* out_np2 = nullptr) {
    int neighbors1[4], count1;
    get_neighbors(p, neighbors1, count1);
    
    if (count1 == 0) {
        if (out_np2) *out_np2 = p;
        return p;
    }
    
    bool g1_alive = (g1 != DEAD);
    bool g2_alive = (g2 != DEAD);
    
    // Pre-compute ghost neighbors
    int g1_neighbors[4], g1_n = 0;
    int g2_neighbors[4], g2_n = 0;
    
    if (g1_alive) get_neighbors(g1, g1_neighbors, g1_n);
    if (g2_alive) get_neighbors(g2, g2_neighbors, g2_n);
    
    int g1_moves = g1_alive ? g1_n : 1;
    int g2_moves = g2_alive ? g2_n : 1;
    
    int best_np1 = neighbors1[0];
    int best_np2 = neighbors1[0];
    uint8_t best_safety = 0;
    uint8_t best_ttr_g = 0;
    
    for (int i1 = 0; i1 < count1; i1++) {
        int np1 = neighbors1[i1];
        
        // Compute power state after first move
        int new_pellet_mask = pellet_mask;
        int new_power_timer = power_timer;
        
        int pellet_eaten = get_pellet_at(np1, pellet_mask);
        if (pellet_eaten >= 0) {
            new_pellet_mask = remove_pellet(pellet_mask, pellet_eaten);
            new_power_timer = power_duration;
        }
        
        // KEY: will_be_powered if already powered OR just ate a pellet
        bool will_be_powered = is_powered(new_power_timer);
        
        // Get second move options
        int neighbors2[5], count2;
        if (will_be_powered) {
            get_neighbors(np1, neighbors2, count2);
            neighbors2[count2++] = np1;  // Can stay in place on move 2
        } else {
            count2 = 1;
            neighbors2[0] = np1;  // np2 == np1 for unpowered
        }
        
        for (int i2 = 0; i2 < count2; i2++) {
            int np2 = neighbors2[i2];
            
            // Check pellet on second move
            int final_pellet_mask = new_pellet_mask;
            int final_power_timer = new_power_timer;
            
            if (will_be_powered && np2 != np1) {
                int pellet_eaten2 = get_pellet_at(np2, new_pellet_mask);
                if (pellet_eaten2 >= 0) {
                    final_pellet_mask = remove_pellet(new_pellet_mask, pellet_eaten2);
                    final_power_timer = power_duration;
                }
            }
            
            uint8_t worst_safety = 1;
            uint8_t worst_ttr_g = 255;
            
            // For powered mode: check if ghosts are caught on first move
            bool g1_caught_move1 = will_be_powered && g1_alive && (np1 == g1);
            bool g2_caught_move1 = will_be_powered && g2_alive && (np1 == g2);
            
            int g1_iter = g1_caught_move1 ? 1 : g1_moves;
            int g2_iter = g2_caught_move1 ? 1 : g2_moves;
            
            for (int gi1 = 0; gi1 < g1_iter; gi1++) {
                for (int gi2 = 0; gi2 < g2_iter; gi2++) {
                    int ng1 = g1_caught_move1 ? DEAD : (g1_alive ? g1_neighbors[gi1] : DEAD);
                    int ng2 = g2_caught_move1 ? DEAD : (g2_alive ? g2_neighbors[gi2] : DEAD);
                    
                    int result_g1 = ng1;
                    int result_g2 = ng2;
                    bool pacman_dies = false;
                    
                    if (will_be_powered) {
                        // Powered mode: check catches/clips on second move
                        bool g1_alive_after_move1 = g1_alive && !g1_caught_move1;
                        bool g2_alive_after_move1 = g2_alive && !g2_caught_move1;
                        
                        bool catch1_m2 = g1_alive_after_move1 && (np2 == ng1);
                        bool catch2_m2 = g2_alive_after_move1 && (np2 == ng2);
                        
                        bool clip1 = g1_alive_after_move1 && (np1 == ng1 && g1 == np2);
                        bool clip2 = g2_alive_after_move1 && (np1 == ng2 && g2 == np2);
                        
                        if (g1_caught_move1 || catch1_m2 || clip1) result_g1 = DEAD;
                        if (g2_caught_move1 || catch2_m2 || clip2) result_g2 = DEAD;
                    } else {
                        // Unpowered mode: check collision/clip
                        bool clip1 = g1_alive && (p == ng1 && g1 == np1);
                        bool clip2 = g2_alive && (p == ng2 && g2 == np1);
                        bool collide1 = g1_alive && (np1 == ng1 || clip1);
                        bool collide2 = g2_alive && (np1 == ng2 || clip2);
                        
                        if (collide1 || collide2) pacman_dies = true;
                    }
                    
                    // Decrement power timer for lookup
                    int result_power_timer = final_power_timer;
                    if (is_powered(result_power_timer)) {
                        result_power_timer--;
                    }
                    
                    uint8_t s, tg;
                    if (pacman_dies) {
                        s = 0;
                        tg = 0;
                    } else if (result_g1 == DEAD && result_g2 == DEAD) {
                        s = 1;
                        tg = 255;
                    } else {
                        int pw = encode_power_state(final_pellet_mask, result_power_timer);
                        size_t idx = value_index(np2, result_g1, result_g2, pw);
                        s = safety_value[idx];
                        tg = ttr_g_value[idx];
                    }
                    
                    worst_safety = min(worst_safety, s);
                    worst_ttr_g = min(worst_ttr_g, tg);
                }
            }
            
            if (worst_safety > best_safety || 
                (worst_safety == best_safety && worst_ttr_g > best_ttr_g)) {
                best_safety = worst_safety;
                best_ttr_g = worst_ttr_g;
                best_np1 = np1;
                best_np2 = np2;
            }
        }
    }
    
    if (out_np2) *out_np2 = best_np2;
    return best_np1;
}

// Check if current state is safe (Pac-Man can survive against optimal ghost play)
bool is_state_safe(int p, int g1, int g2, int pellet_mask, int power_timer) {
    int pw = encode_power_state(pellet_mask, power_timer);
    size_t idx = value_index(p, g1, g2, pw);
    return safety_value[idx] == 1;
}

// Check if any ghost is within N steps (Manhattan distance)
bool ghost_nearby(int p, int g1, int g2, int threshold) {
    if (g1 != DEAD && manhattan_dist(p, g1) <= threshold) return true;
    if (g2 != DEAD && manhattan_dist(p, g2) <= threshold) return true;
    return false;
}

// ============================================================================
// Simulation with Detailed Tracking
// ============================================================================

struct SimulationResult {
    bool survived;
    bool won;
    int survival_time;
    DetailedStats stats;
    int initial_min_ghost_dist;
    int death_position;
    int win_time;
    int death_tick;           // When death occurred
    int pellets_eaten_at_death;  // How many pellets eaten when died (0-4)
};

SimulationResult run_simulation_detailed(PacmanStrategy pac_strategy, GhostStrategy ghost_strategy,
                                         int start_p, int start_g1, int start_g2, int max_ticks) {
    SimulationResult result;
    result.survived = false;
    result.won = false;
    result.survival_time = 0;
    result.stats.reset();
    result.death_position = -1;
    result.win_time = -1;
    result.death_tick = -1;
    result.pellets_eaten_at_death = -1;
    
    int d1 = (start_g1 != DEAD) ? manhattan_dist(start_p, start_g1) : 999;
    int d2 = (start_g2 != DEAD) ? manhattan_dist(start_p, start_g2) : 999;
    result.initial_min_ghost_dist = min(d1, d2);
    
    int p = start_p;
    int g1 = start_g1;
    int g2 = start_g2;
    int pellet_mask = num_pellet_masks - 1;
    int power_timer = 0;
    bool prev_safe = true;
    
    ForwardReachableSet frs1, frs2;
    
    for (int tick = 0; tick < max_ticks; tick++) {
        result.stats.total_ticks++;
        
        bool current_safe = is_state_safe(p, g1, g2, pellet_mask, power_timer);
        if (current_safe) {
            result.stats.ticks_in_safe_state++;
        } else {
            result.stats.ticks_in_unsafe_state++;
        }
        if (prev_safe && !current_safe) {
            result.stats.safe_to_unsafe_transitions++;
        }
        prev_safe = current_safe;
        
        // Win condition: both ghosts dead
        if (g1 == DEAD && g2 == DEAD) {
            result.won = true;
            result.survived = true;
            result.survival_time = tick;
            result.win_time = tick;
            return result;
        }
        
        // Collision check at start of tick (shouldn't happen if simulation is correct)
        bool caught = !is_powered(power_timer) && 
                      ((g1 != DEAD && p == g1) || (g2 != DEAD && p == g2));
        if (caught) {
            result.survival_time = tick;
            result.death_position = p;
            result.death_tick = tick;
            result.pellets_eaten_at_death = num_pellets - __builtin_popcount(pellet_mask);
            return result;
        }
        
        // Determine goal for A* strategies
        int goal;
        if (is_powered(power_timer)) {
            goal = find_nearest_ghost(p, g1, g2);
            if (goal < 0) {
                goal = p;
                result.stats.ticks_no_target++;
            } else {
                result.stats.ticks_targeting_ghost++;
            }
        } else {
            bool unsafe = !is_state_safe(p, g1, g2, pellet_mask, power_timer);
            bool ghost_close = ghost_nearby(p, g1, g2, power_duration + 2);
            
            if (unsafe || ghost_close) {
                goal = find_nearest_pellet(p, pellet_mask);
                if (goal < 0) {
                    goal = p;
                    result.stats.ticks_no_target++;
                } else {
                    result.stats.ticks_targeting_pellet++;
                }
            } else {
                goal = p;
                result.stats.ticks_no_target++;
            }
        }
        
        if (pac_strategy == PacmanStrategy::ASTAR_FRS) {
            // Use power_duration + 2 as horizon: ghosts beyond this range don't matter
            // because Pac-Man can reach a pellet and become powered
            int horizon = min(max_ticks - tick, power_duration + 2);
            if (g1 != DEAD) {
                frs1.compute(maze, MAZE_ROWS, MAZE_COLS, g1, horizon);
                result.stats.frs_recomputations++;
            }
            if (g2 != DEAD) {
                frs2.compute(maze, MAZE_ROWS, MAZE_COLS, g2, horizon);
                result.stats.frs_recomputations++;
            }
        }
        
        int old_p = p;
        int safe_np2_from_filter = p;  // Will be set by safety filter if used
        
        // ================================================================
        // Get Pac-Man's first move
        // ================================================================
        int np1;
        
        if (pac_strategy == PacmanStrategy::OPTIMAL) {
            // For optimal, we need to decide based on whether we WILL be powered
            // Check all first moves to see if any eats a pellet
            np1 = get_optimal_pacman_move(p, g1, g2, pellet_mask, power_timer);
        } else {
            switch (pac_strategy) {
                case PacmanStrategy::ASTAR_SAFETY:
                    np1 = astar_safety(p, goal, g1, g2, pellet_mask, power_timer);
                    break;
                case PacmanStrategy::ASTAR_FRS: {
                    bool intervention;
                    np1 = astar_frs_tracked(p, goal, g1, g2, pellet_mask, power_timer, frs1, frs2, intervention);
                    if (intervention) result.stats.frs_interventions++;
                    break;
                }
                case PacmanStrategy::ASTAR_FILTER: {
                    bool intervention, fallback;
                    np1 = astar_filter_tracked(p, goal, g1, g2, pellet_mask, power_timer, intervention, fallback, &safe_np2_from_filter);
                    if (intervention) result.stats.safety_interventions++;
                    if (fallback) result.stats.safety_fallbacks++;
                    break;
                }
                default:
                    np1 = astar_get_preferred_move(p, goal);
                    break;
            }
        }
        
        // Check if pellet eaten on first move
        int pellet_eaten1 = get_pellet_at(np1, pellet_mask);
        int new_pellet_mask = pellet_mask;
        int new_power_timer = power_timer;
        
        if (pellet_eaten1 >= 0) {
            if (result.stats.first_pellet_tick < 0) {
                result.stats.first_pellet_tick = tick;
            }
            new_pellet_mask = remove_pellet(pellet_mask, pellet_eaten1);
            new_power_timer = power_duration;
        }
        
        // KEY: will_be_powered is true if already powered OR just ate a pellet
        bool will_be_powered = is_powered(new_power_timer);
        
        // ================================================================
        // POWERED MODE: Pac-Man moves twice, ghosts move once
        // Triggered if already powered OR if eating pellet on first move
        // ================================================================
        if (will_be_powered) {
            // First move: Pac-Man moves to np1, ghosts stay
            // Check if Pac-Man catches ghost on first move
            bool g1_caught_move1 = (g1 != DEAD) && (np1 == g1);
            bool g2_caught_move1 = (g2 != DEAD) && (np1 == g2);
            
            int temp_g1 = g1_caught_move1 ? DEAD : g1;
            int temp_g2 = g2_caught_move1 ? DEAD : g2;
            
            // Win check after first move
            if (temp_g1 == DEAD && temp_g2 == DEAD) {
                result.won = true;
                result.survived = true;
                result.survival_time = tick;
                result.win_time = tick;
                return result;
            }
            
            // Get second move
            int np2;
            if (pac_strategy == PacmanStrategy::OPTIMAL) {
                // For optimal play, use the unified move function which returns np2
                get_optimal_pacman_move(p, g1, g2, pellet_mask, power_timer, &np2);
            } else if (pac_strategy == PacmanStrategy::ASTAR_FILTER) {
                // For safety filter, use the pre-computed safe np2
                np2 = safe_np2_from_filter;
            } else {
                // For other A* strategies, do second A* move toward goal
                int goal2 = find_nearest_ghost(np1, temp_g1, temp_g2);
                if (goal2 < 0) goal2 = np1;
                np2 = astar_get_preferred_move(np1, goal2);
            }
            
            // Check pellet on second move
            int final_pellet_mask = new_pellet_mask;
            int final_power_timer = new_power_timer;
            if (np2 != np1) {
                int pellet_eaten2 = get_pellet_at(np2, new_pellet_mask);
                if (pellet_eaten2 >= 0) {
                    if (result.stats.first_pellet_tick < 0) {
                        result.stats.first_pellet_tick = tick;
                    }
                    final_pellet_mask = remove_pellet(new_pellet_mask, pellet_eaten2);
                    final_power_timer = power_duration;
                }
            }
            
            // Second move: Pac-Man moves np1->np2, ghosts move g1->ng1, g2->ng2 (simultaneous)
            // Note: ghosts caught on move 1 don't get to move
            int ng1, ng2;
            
            // Get ghost moves together (they coordinate in optimal play)
            get_ghost_moves(ghost_strategy, np1, np2, temp_g1, temp_g2, final_pellet_mask, final_power_timer, ng1, ng2);
            
            // Override with DEAD for ghosts caught on move 1
            if (g1_caught_move1) ng1 = DEAD;
            if (g2_caught_move1) ng2 = DEAD;
            
            // Check catches on second move (simultaneous collision)
            // Only check for ghosts that weren't caught on move 1
            bool g1_alive_after_move1 = (g1 != DEAD) && !g1_caught_move1;
            bool g2_alive_after_move1 = (g2 != DEAD) && !g2_caught_move1;
            
            bool catch1_m2 = g1_alive_after_move1 && (np2 == ng1);
            bool catch2_m2 = g2_alive_after_move1 && (np2 == ng2);
            
            // Check clipping on second move (Pac-Man at np1 -> np2, ghost at g1 -> ng1)
            bool clip1 = g1_alive_after_move1 && (np1 == ng1 && g1 == np2);
            bool clip2 = g2_alive_after_move1 && (np1 == ng2 && g2 == np2);
            
            if (catch1_m2 || clip1) ng1 = DEAD;
            if (catch2_m2 || clip2) ng2 = DEAD;
            
            p = np2;
            g1 = ng1;
            g2 = ng2;
            pellet_mask = final_pellet_mask;
            power_timer = final_power_timer;
            
            if (power_timer > 0) power_timer--;
        }
        // ================================================================
        // NORMAL MODE: Pac-Man and ghosts move once simultaneously
        // ================================================================
        else {
            // Get ghost moves (simultaneous with Pac-Man's move)
            int ng1, ng2;
            get_ghost_moves(ghost_strategy, old_p, np1, g1, g2, new_pellet_mask, new_power_timer, ng1, ng2);
            
            // Check collision and clipping
            bool clip1 = (g1 != DEAD) && (old_p == ng1 && g1 == np1);
            bool clip2 = (g2 != DEAD) && (old_p == ng2 && g2 == np1);
            if (np1 == ng1 || np1 == ng2 || clip1 || clip2) {
                result.survival_time = tick + 1;
                result.death_position = np1;
                result.death_tick = tick + 1;
                result.pellets_eaten_at_death = num_pellets - __builtin_popcount(new_pellet_mask);
                return result;
            }
            
            p = np1;
            g1 = ng1;
            g2 = ng2;
            pellet_mask = new_pellet_mask;
            power_timer = new_power_timer;
            
            if (power_timer > 0) power_timer--;
        }
    }
    
    result.survived = true;
    result.survival_time = max_ticks;
    return result;
}

// ============================================================================
// Aggregated Evaluation
// ============================================================================

struct AggregatedStats {
    int total_runs = 0;
    int survived = 0;
    int won = 0;
    long long total_survival_time = 0;
    
    // Aggregated detailed stats
    long long total_ticks = 0;
    long long total_frs_interventions = 0;
    long long total_safety_interventions = 0;
    long long total_safety_fallbacks = 0;
    long long total_frs_recomputations = 0;
    long long total_moves_blocked_frs = 0;
    long long total_moves_blocked_safety = 0;
    long long ticks_targeting_pellet = 0;
    long long ticks_targeting_ghost = 0;
    long long ticks_no_target = 0;
    
    long long ticks_in_safe_state = 0;
    long long ticks_in_unsafe_state = 0;
    long long safe_to_unsafe_transitions = 0;
    
    long long total_first_pellet_tick = 0;
    int runs_with_pellet = 0;
    
    int survival_by_distance[4] = {0, 0, 0, 0};
    int runs_by_distance[4] = {0, 0, 0, 0};
    
    int death_counts[MAZE_CELLS] = {0};
    
    long long total_win_time = 0;
    vector<int> win_times;
    
    // Failure mode analysis
    int deaths_before_any_pellet = 0;     // Died with 0 pellets eaten
    int deaths_with_some_pellets = 0;     // Died with 1-3 pellets eaten
    int deaths_after_all_pellets = 0;     // Died with all 4 pellets eaten (resources exhausted)
    vector<int> death_ticks;              // When deaths occurred
};

AggregatedStats evaluate_detailed(PacmanStrategy pac_strategy, GhostStrategy ghost_strategy,
                                  int num_runs, int max_ticks, bool verbose = false) {
    AggregatedStats agg;
    agg.total_runs = num_runs;
    
    vector<int> valid_positions;
    for (int i = 0; i < MAZE_CELLS; i++) {
        if (is_valid_pos(i)) {
            valid_positions.push_back(i);
        }
    }
    
    std::uniform_int_distribution<int> pos_dist(0, valid_positions.size() - 1);
    
    int run = 0;
    int attempts = 0;
    const int max_attempts = num_runs * 100;  // Prevent infinite loop
    
    while (run < num_runs && attempts < max_attempts) {
        attempts++;
        
        int p_idx = pos_dist(rng);
        int g1_idx = pos_dist(rng);
        int g2_idx = pos_dist(rng);
        
        while (g1_idx == p_idx) g1_idx = pos_dist(rng);
        while (g2_idx == p_idx || g2_idx == g1_idx) g2_idx = pos_dist(rng);
        
        int start_p = valid_positions[p_idx];
        int start_g1 = valid_positions[g1_idx];
        int start_g2 = valid_positions[g2_idx];
        
        // Skip configurations that are not guaranteed safe starting states
        int initial_pellet_mask = num_pellet_masks - 1;  // All pellets present
        int initial_power_timer = 0;  // Not powered
        if (!is_state_safe(start_p, start_g1, start_g2, initial_pellet_mask, initial_power_timer)) {
            continue;  // Skip this configuration
        }
        
        run++;  // Only count runs with safe starting states
        
        current_stats.reset();
        SimulationResult sim = run_simulation_detailed(pac_strategy, ghost_strategy,
                                                       start_p, start_g1, start_g2, max_ticks);
        
        if (sim.survived) agg.survived++;
        if (sim.won) agg.won++;
        agg.total_survival_time += sim.survival_time;
        
        agg.total_ticks += sim.stats.total_ticks;
        agg.total_frs_interventions += sim.stats.frs_interventions;
        agg.total_safety_interventions += sim.stats.safety_interventions;
        agg.total_safety_fallbacks += sim.stats.safety_fallbacks;
        agg.total_frs_recomputations += sim.stats.frs_recomputations;
        agg.total_moves_blocked_frs += sim.stats.moves_blocked_by_frs;
        agg.total_moves_blocked_safety += sim.stats.moves_blocked_by_safety;
        agg.ticks_targeting_pellet += sim.stats.ticks_targeting_pellet;
        agg.ticks_targeting_ghost += sim.stats.ticks_targeting_ghost;
        agg.ticks_no_target += sim.stats.ticks_no_target;
        
        agg.ticks_in_safe_state += sim.stats.ticks_in_safe_state;
        agg.ticks_in_unsafe_state += sim.stats.ticks_in_unsafe_state;
        agg.safe_to_unsafe_transitions += sim.stats.safe_to_unsafe_transitions;
        
        if (sim.stats.first_pellet_tick >= 0) {
            agg.total_first_pellet_tick += sim.stats.first_pellet_tick;
            agg.runs_with_pellet++;
        }
        
        int dist_bucket = min(sim.initial_min_ghost_dist / 3, 3);
        agg.runs_by_distance[dist_bucket]++;
        if (sim.survived) agg.survival_by_distance[dist_bucket]++;
        
        if (sim.death_position >= 0 && sim.death_position < MAZE_CELLS) {
            agg.death_counts[sim.death_position]++;
            agg.death_ticks.push_back(sim.death_tick);
            
            // Failure mode analysis
            if (sim.pellets_eaten_at_death == 0) {
                agg.deaths_before_any_pellet++;
            } else if (sim.pellets_eaten_at_death >= num_pellets) {
                agg.deaths_after_all_pellets++;
            } else {
                agg.deaths_with_some_pellets++;
            }
        }
        
        if (sim.won) {
            agg.total_win_time += sim.win_time;
            agg.win_times.push_back(sim.win_time);
        }
        
        if (verbose && run % 100 == 0) {
            cout << "\r  Progress: " << run << "/" << num_runs << flush;
        }
    }
    
    if (verbose) {
        cout << "\r  Progress: " << run << "/" << num_runs;
        if (attempts > run) {
            cout << " (skipped " << (attempts - run) << " unsafe starting states)";
        }
        cout << endl;
    }
    
    return agg;
}

// ============================================================================
// CSV Output
// ============================================================================

void write_csv_header(ofstream& csv) {
    csv << "pacman_strategy,ghost_strategy,num_runs,max_ticks,"
        << "survival_rate,win_rate,avg_survival_time,"
        << "total_ticks,avg_ticks_per_run,"
        << "frs_interventions,frs_intervention_rate,"
        << "safety_interventions,safety_intervention_rate,"
        << "safety_fallbacks,safety_fallback_rate,"
        << "frs_recomputations,avg_frs_recomps_per_run,"
        << "moves_blocked_frs,moves_blocked_safety,"
        << "pct_targeting_pellet,pct_targeting_ghost,pct_no_target,"
        << "pct_in_safe_state,pct_in_unsafe_state,avg_safe_to_unsafe_transitions,"
        << "avg_first_pellet_tick,"
        << "survival_dist_0_2,survival_dist_3_5,survival_dist_6_8,survival_dist_9plus,"
        << "avg_win_time,"
        << "deaths_before_pellet,deaths_with_some_pellets,deaths_after_all_pellets"
        << endl;
}

void write_csv_row(ofstream& csv, const string& pac_name, const string& ghost_name,
                   int num_runs, int max_ticks, const AggregatedStats& agg) {
    double survival_rate = (double)agg.survived / agg.total_runs;
    double win_rate = (double)agg.won / agg.total_runs;
    double avg_survival = (double)agg.total_survival_time / agg.total_runs;
    double avg_ticks = (double)agg.total_ticks / agg.total_runs;
    
    double frs_intervention_rate = agg.total_ticks > 0 ? 
        (double)agg.total_frs_interventions / agg.total_ticks : 0;
    double safety_intervention_rate = agg.total_ticks > 0 ? 
        (double)agg.total_safety_interventions / agg.total_ticks : 0;
    double safety_fallback_rate = agg.total_ticks > 0 ? 
        (double)agg.total_safety_fallbacks / agg.total_ticks : 0;
    double avg_frs_recomps = (double)agg.total_frs_recomputations / agg.total_runs;
    
    double pct_pellet = agg.total_ticks > 0 ? 
        100.0 * agg.ticks_targeting_pellet / agg.total_ticks : 0;
    double pct_ghost = agg.total_ticks > 0 ? 
        100.0 * agg.ticks_targeting_ghost / agg.total_ticks : 0;
    double pct_none = agg.total_ticks > 0 ? 
        100.0 * agg.ticks_no_target / agg.total_ticks : 0;
    
    double pct_safe = agg.total_ticks > 0 ? 
        100.0 * agg.ticks_in_safe_state / agg.total_ticks : 0;
    double pct_unsafe = agg.total_ticks > 0 ? 
        100.0 * agg.ticks_in_unsafe_state / agg.total_ticks : 0;
    double avg_transitions = (double)agg.safe_to_unsafe_transitions / agg.total_runs;
    double avg_first_pellet = agg.runs_with_pellet > 0 ? 
        (double)agg.total_first_pellet_tick / agg.runs_with_pellet : -1;
    
    // Survival by distance
    double surv_0_2 = agg.runs_by_distance[0] > 0 ? 
        100.0 * agg.survival_by_distance[0] / agg.runs_by_distance[0] : 0;
    double surv_3_5 = agg.runs_by_distance[1] > 0 ? 
        100.0 * agg.survival_by_distance[1] / agg.runs_by_distance[1] : 0;
    double surv_6_8 = agg.runs_by_distance[2] > 0 ? 
        100.0 * agg.survival_by_distance[2] / agg.runs_by_distance[2] : 0;
    double surv_9plus = agg.runs_by_distance[3] > 0 ? 
        100.0 * agg.survival_by_distance[3] / agg.runs_by_distance[3] : 0;
    
    double avg_win_time = agg.won > 0 ? (double)agg.total_win_time / agg.won : -1;
    
    csv << pac_name << "," << ghost_name << "," << num_runs << "," << max_ticks << ","
        << fixed << setprecision(4) << survival_rate << ","
        << fixed << setprecision(4) << win_rate << ","
        << fixed << setprecision(2) << avg_survival << ","
        << agg.total_ticks << ","
        << fixed << setprecision(2) << avg_ticks << ","
        << agg.total_frs_interventions << ","
        << fixed << setprecision(4) << frs_intervention_rate << ","
        << agg.total_safety_interventions << ","
        << fixed << setprecision(4) << safety_intervention_rate << ","
        << agg.total_safety_fallbacks << ","
        << fixed << setprecision(4) << safety_fallback_rate << ","
        << agg.total_frs_recomputations << ","
        << fixed << setprecision(2) << avg_frs_recomps << ","
        << agg.total_moves_blocked_frs << "," << agg.total_moves_blocked_safety << ","
        << fixed << setprecision(2) << pct_pellet << ","
        << fixed << setprecision(2) << pct_ghost << ","
        << fixed << setprecision(2) << pct_none << ","
        // NEW fields
        << fixed << setprecision(2) << pct_safe << ","
        << fixed << setprecision(2) << pct_unsafe << ","
        << fixed << setprecision(2) << avg_transitions << ","
        << fixed << setprecision(2) << avg_first_pellet << ","
        << fixed << setprecision(2) << surv_0_2 << ","
        << fixed << setprecision(2) << surv_3_5 << ","
        << fixed << setprecision(2) << surv_6_8 << ","
        << fixed << setprecision(2) << surv_9plus << ","
        << fixed << setprecision(2) << avg_win_time << ","
        << agg.deaths_before_any_pellet << ","
        << agg.deaths_with_some_pellets << ","
        << agg.deaths_after_all_pellets
        << endl;
}

// ============================================================================
// Additional CSV Writers
// ============================================================================

void write_death_locations_csv(const string& filename, 
                               const vector<pair<string, AggregatedStats>>& all_results) {
    ofstream csv(filename);
    if (!csv) return;
    
    csv << "row,col";
    for (const auto& result : all_results) {
        csv << "," << result.first;
    }
    csv << endl;
    
    for (int r = 0; r < MAZE_ROWS; r++) {
        for (int c = 0; c < MAZE_COLS; c++) {
            int idx = r * MAZE_COLS + c;
            if (!is_valid_pos(idx)) continue;
            
            csv << r << "," << c;
            for (const auto& result : all_results) {
                csv << "," << result.second.death_counts[idx];
            }
            csv << endl;
        }
    }
    csv.close();
}

void write_win_times_csv(const string& filename,
                         const vector<pair<string, AggregatedStats>>& all_results) {
    ofstream csv(filename);
    if (!csv) return;
    
    size_t max_wins = 0;
    for (const auto& result : all_results) {
        max_wins = max(max_wins, result.second.win_times.size());
    }
    
    csv << "index";
    for (const auto& result : all_results) {
        csv << "," << result.first;
    }
    csv << endl;
    
    for (size_t i = 0; i < max_wins; i++) {
        csv << i;
        for (const auto& result : all_results) {
            if (i < result.second.win_times.size()) {
                csv << "," << result.second.win_times[i];
            } else {
                csv << ",";
            }
        }
        csv << endl;
    }
    csv.close();
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* program) {
    cout << "Usage: " << program << " [options]" << endl;
    cout << endl;
    cout << "Required:" << endl;
    cout << "  -v, --values FILE   Value table file (.bin.gz)" << endl;
    cout << endl;
    cout << "Optional:" << endl;
    cout << "  -n, --num-runs N    Number of simulation runs (default: 1000)" << endl;
    cout << "  -t, --ticks N       Max ticks per simulation (default: 100)" << endl;
    cout << "  -s, --seed N        Random seed (default: time-based)" << endl;
    cout << "  -o, --output FILE   Output CSV file (default: eval_results.csv)" << endl;
    cout << "  --all               Run all strategy combinations" << endl;
    cout << "  -h, --help          Show this help message" << endl;
}

void print_detailed_results(const string& pac_name, const string& ghost_name,
                           const AggregatedStats& agg, int max_ticks) {
    double survival_rate = 100.0 * agg.survived / agg.total_runs;
    double win_rate = 100.0 * agg.won / agg.total_runs;
    double avg_survival = (double)agg.total_survival_time / agg.total_runs;
    double avg_ticks = (double)agg.total_ticks / agg.total_runs;
    
    cout << "\n--- " << pac_name << " vs " << ghost_name << " ---" << endl;
    cout << "  Survival rate: " << fixed << setprecision(1) << survival_rate << "%" << endl;
    cout << "  Win rate: " << fixed << setprecision(1) << win_rate << "%" << endl;
    cout << "  Avg survival time: " << fixed << setprecision(1) << avg_survival << " ticks" << endl;
    cout << "  Avg ticks per run: " << fixed << setprecision(1) << avg_ticks << endl;
    
    if (agg.total_frs_interventions > 0 || agg.total_frs_recomputations > 0) {
        cout << "  FRS recomputations: " << agg.total_frs_recomputations 
             << " (" << fixed << setprecision(1) 
             << (double)agg.total_frs_recomputations / agg.total_runs << " per run)" << endl;
        cout << "  FRS interventions: " << agg.total_frs_interventions 
             << " (" << fixed << setprecision(2) 
             << 100.0 * agg.total_frs_interventions / agg.total_ticks << "% of ticks)" << endl;
    }
    
    if (agg.total_safety_interventions > 0 || agg.total_safety_fallbacks > 0) {
        cout << "  Safety interventions: " << agg.total_safety_interventions 
             << " (" << fixed << setprecision(2) 
             << 100.0 * agg.total_safety_interventions / agg.total_ticks << "% of ticks)" << endl;
        cout << "  Safety fallbacks (no safe move): " << agg.total_safety_fallbacks 
             << " (" << fixed << setprecision(2) 
             << 100.0 * agg.total_safety_fallbacks / agg.total_ticks << "% of ticks)" << endl;
    }
    
    double pct_pellet = 100.0 * agg.ticks_targeting_pellet / agg.total_ticks;
    double pct_ghost = 100.0 * agg.ticks_targeting_ghost / agg.total_ticks;
    double pct_none = 100.0 * agg.ticks_no_target / agg.total_ticks;
    cout << "  Targeting: pellet=" << fixed << setprecision(1) << pct_pellet << "%, "
         << "ghost=" << pct_ghost << "%, none=" << pct_none << "%" << endl;
}

int main(int argc, char* argv[]) {
    string values_file = "";
    string output_file = "eval_results.csv";
    int num_runs = 1000;
    int max_ticks = 100;
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    bool run_all = false;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "-v" || arg == "--values") && i + 1 < argc) {
            values_file = argv[++i];
        } else if ((arg == "-n" || arg == "--num-runs") && i + 1 < argc) {
            num_runs = stoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--ticks") && i + 1 < argc) {
            max_ticks = stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
            seed = stoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--all") {
            run_all = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (values_file.empty()) {
        cerr << "Error: Values file is required" << endl;
        print_usage(argv[0]);
        return 1;
    }
    
    rng.seed(seed);
    
    cout << "=== Detailed A* Evaluation Framework ===" << endl;
    cout << "Loading values from " << values_file << "..." << endl;
    
    if (!load_values(values_file)) {
        return 1;
    }
    
    cout << "Configuration:" << endl;
    cout << "  Maze: " << MAZE_ROWS << "x" << MAZE_COLS << endl;
    cout << "  Pellets: " << num_pellets << " at positions: ";
    for (int i = 0; i < num_pellets; i++) {
        cout << pellet_positions[i];
        if (i < num_pellets - 1) cout << ", ";
    }
    cout << endl;
    cout << "  Power duration: " << power_duration << " ticks" << endl;
    cout << "  Runs per evaluation: " << num_runs << endl;
    cout << "  Max ticks: " << max_ticks << endl;
    cout << "  Random seed: " << seed << endl;
    cout << "  Output CSV: " << output_file << endl;
    
    // Open CSV file
    ofstream csv(output_file);
    if (!csv) {
        cerr << "Error: Could not open " << output_file << " for writing" << endl;
        return 1;
    }
    write_csv_header(csv);
    
    // Collect all results for additional CSV outputs
    vector<pair<string, AggregatedStats>> all_results;
    
    if (run_all) {
        vector<PacmanStrategy> pac_strategies = {
            PacmanStrategy::OPTIMAL,
            PacmanStrategy::ASTAR_SAFETY,
            PacmanStrategy::ASTAR_FRS,
            PacmanStrategy::ASTAR_FILTER
        };
        
        vector<GhostStrategy> ghost_strategies = {
            GhostStrategy::OPTIMAL,
            GhostStrategy::GREEDY_BFS
        };
        
        cout << "\n=== Running All Evaluations ===" << endl;
        
        for (auto gs : ghost_strategies) {
            for (auto ps : pac_strategies) {
                string pac_name = pacman_strategy_name(ps);
                string ghost_name = ghost_strategy_name(gs);
                string combo_name = pac_name + "_vs_" + ghost_name;
                
                cout << "\nEvaluating " << pac_name << " vs " << ghost_name << "..." << endl;
                
                AggregatedStats agg = evaluate_detailed(ps, gs, num_runs, max_ticks, true);
                
                print_detailed_results(pac_name, ghost_name, agg, max_ticks);
                write_csv_row(csv, pac_name, ghost_name, num_runs, max_ticks, agg);
                
                all_results.push_back({combo_name, agg});
            }
        }
        
        // Write additional analysis CSVs
        string base_name = output_file.substr(0, output_file.find_last_of('.'));
        write_death_locations_csv(base_name + "_deaths.csv", all_results);
        write_win_times_csv(base_name + "_wins.csv", all_results);
        cout << "  Additional CSVs: " << base_name << "_deaths.csv, " << base_name << "_wins.csv" << endl;
    }
    
    csv.close();
    cout << "\n=== Results saved to " << output_file << " ===" << endl;
    
    free_tables();
    return 0;
}
