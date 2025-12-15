/* Simulation for optimized multi-pellet Pacman
 * Loads compressed value tables and runs optimal play simulation
 *
 * Features:
 *   - 1-4 super pellets
 *   - Compact power state encoding
 *   - gzip decompression
 *   - Ghost permadeath when eaten
 *   - Pacman moves TWICE per turn when powered
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <zlib.h>
using namespace std;

#define MAZE_ROWS 12
#define MAZE_COLS 12
#define MAZE_CELLS (MAZE_ROWS * MAZE_COLS)

#define DEAD MAZE_CELLS
#define GHOST_STATES (MAZE_CELLS + 1)

#define MAX_PELLETS 4
#define MAX_POWER_DURATION 30

uint32_t maze[MAZE_ROWS] = {0};

// Configuration (loaded from file)
int num_pellets = 2;
int power_duration = 6;
int pellet_positions[MAX_PELLETS] = {0};
int num_pellet_masks = 4;
int total_power_states = 0;

// Value tables
uint8_t* safety_value = nullptr;
uint8_t* ttr_g_value = nullptr;
uint8_t* ttr_p_value = nullptr;
size_t table_size = 0;

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

// ============================================================================
// Compact Power State Encoding
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

inline int manhattan_dist(int a, int b) {
    return abs(row(a) - row(b)) + abs(col(a) - col(b));
}

inline size_t value_index(int p, int g1, int g2, int pw) {
    return ((size_t)p * GHOST_STATES * GHOST_STATES +
            (size_t)g1 * GHOST_STATES +
            (size_t)g2) * total_power_states + pw;
}

// ============================================================================
// File I/O
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
        cerr << "Error: Invalid file format (expected PVAO)" << endl;
        return false;
    }

    if (header.version != 1) {
        cerr << "Error: Unsupported version " << (int)header.version << endl;
        return false;
    }

    if (header.maze_rows != MAZE_ROWS || header.maze_cols != MAZE_COLS) {
        cerr << "Error: Maze size mismatch" << endl;
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

    cout << "Loading optimized values..." << endl;
    cout << "  Pellets: " << num_pellets << endl;
    cout << "  Power duration: " << power_duration << endl;
    cout << "  Power states: " << total_power_states << endl;

    // Read maze
    file.read(reinterpret_cast<char*>(maze), sizeof(maze));

    // Allocate tables
    try {
        safety_value = new uint8_t[table_size];
        ttr_g_value = new uint8_t[table_size];
        ttr_p_value = new uint8_t[table_size];
    } catch (const bad_alloc& e) {
        cerr << "Error: Failed to allocate memory" << endl;
        return false;
    }

    if (header.compression == 1) {
        // Read compressed data
        uint32_t compressed_size;
        file.read(reinterpret_cast<char*>(&compressed_size), sizeof(compressed_size));

        vector<uint8_t> compressed(compressed_size);
        file.read(reinterpret_cast<char*>(compressed.data()), compressed_size);

        cout << "  Decompressing " << (compressed_size / 1024) << " KB..." << endl;

        // Decompress
        uLongf uncompressed_size = header.uncompressed_size;
        vector<uint8_t> uncompressed(uncompressed_size);

        int result = uncompress(uncompressed.data(), &uncompressed_size,
                               compressed.data(), compressed_size);

        if (result != Z_OK) {
            cerr << "Error: Decompression failed with code " << result << endl;
            return false;
        }

        // Copy to tables
        memcpy(safety_value, uncompressed.data(), table_size);
        memcpy(ttr_g_value, uncompressed.data() + table_size, table_size);
        memcpy(ttr_p_value, uncompressed.data() + 2 * table_size, table_size);
    } else {
        // Read uncompressed
        file.read(reinterpret_cast<char*>(safety_value), table_size);
        file.read(reinterpret_cast<char*>(ttr_g_value), table_size);
        file.read(reinterpret_cast<char*>(ttr_p_value), table_size);
    }

    if (!file) {
        cerr << "Error: Failed to read value data" << endl;
        return false;
    }

    file.close();
    cout << "  Loaded successfully." << endl;
    return true;
}

void free_tables() {
    delete[] safety_value;
    delete[] ttr_g_value;
    delete[] ttr_p_value;
    safety_value = ttr_g_value = ttr_p_value = nullptr;
}

// ============================================================================
// Game State Display
// ============================================================================

void print_game_state(int p, int g1, int g2, int pellet_mask, int power_timer, int step) {
    cout << "Step " << step << ":";

    // Show power status
    if (is_powered(power_timer)) {
        cout << " [POWERED: " << power_timer << " turns left]";
    }

    // Show pellet count
    int pellet_count = __builtin_popcount(pellet_mask);
    cout << " [Pellets: " << pellet_count << "/" << num_pellets << "]";
    cout << endl;

    // Column headers
    cout << "    ";
    for (int c = 0; c < MAZE_COLS; c++) cout << c << " ";
    cout << endl;

    bool g1_alive = (g1 != DEAD);
    bool g2_alive = (g2 != DEAD);
    bool pacman_caught = (!is_powered(power_timer)) &&
                         ((g1_alive && p == g1) || (g2_alive && p == g2));
    bool pacman_wins = !g1_alive && !g2_alive;

    for (int r = 0; r < MAZE_ROWS; r++) {
        cout << setw(3) << setfill(' ') << r << " ";
        for (int c = 0; c < MAZE_COLS; c++) {
            int idx = r * MAZE_COLS + c;

            if (pacman_caught && idx == p) {
                cout << "X ";  // Caught
            } else if (idx == p) {
                if (is_powered(power_timer)) cout << "@ ";  // Powered Pacman
                else cout << "P ";
            } else if (g1_alive && idx == g1 && g2_alive && idx == g2) {
                cout << "B ";  // Both ghosts
            } else if (g1_alive && idx == g1) {
                cout << "1 ";
            } else if (g2_alive && idx == g2) {
                cout << "2 ";
            } else if (get_pellet_at(idx, pellet_mask) >= 0) {
                cout << "O ";  // Pellet
            } else if (is_free(r, c)) {
                cout << ". ";
            } else {
                cout << "  ";
            }
        }
        cout << endl;
    }

    // Position info
    cout << "Pacman=" << p << " [" << row(p) << "," << col(p) << "]";
    if (g1_alive) cout << ", Ghost1=" << g1 << " [" << row(g1) << "," << col(g1) << "]";
    else cout << ", Ghost1=DEAD";
    if (g2_alive) cout << ", Ghost2=" << g2 << " [" << row(g2) << "," << col(g2) << "]";
    else cout << ", Ghost2=DEAD";

    if (pacman_caught) cout << " (CAUGHT!)";
    if (pacman_wins) cout << " (PACMAN WINS!)";
    cout << endl << endl;
}

// ============================================================================
// Optimal Move Computation
// ============================================================================

struct MoveResult {
    int pos1;           // First position
    int pos2;           // Second position (same as pos1 if not powered)
    int new_g1;         // Ghost 1 after move
    int new_g2;         // Ghost 2 after move
    int new_pellet_mask;
    int new_power_timer;
};

void get_optimal_ghost_moves(int p, int np1, int np2, int g1, int g2,
                             int pellet_mask, int power_timer,
                             int& best_g1, int& best_g2);

MoveResult get_optimal_pacman_move(int p, int g1, int g2, int pellet_mask, int power_timer) {
    MoveResult result;
    result.pos1 = p;
    result.pos2 = p;
    result.new_g1 = g1;
    result.new_g2 = g2;
    result.new_pellet_mask = pellet_mask;
    result.new_power_timer = power_timer;

    int pac_neighbors[4], pac_n;
    get_neighbors(p, pac_neighbors, pac_n);

    if (pac_n == 0) {
        cerr << "ERROR: Pacman has no valid moves!" << endl;
        return result;
    }

    bool g1_alive = (g1 != DEAD);
    bool g2_alive = (g2 != DEAD);

    int g1_neighbors[4], g1_n = 0;
    int g2_neighbors[4], g2_n = 0;

    if (g1_alive) get_neighbors(g1, g1_neighbors, g1_n);
    if (g2_alive) get_neighbors(g2, g2_neighbors, g2_n);

    int g1_moves = g1_alive ? g1_n : 1;
    int g2_moves = g2_alive ? g2_n : 1;

    uint8_t best_safety = 0;
    uint8_t best_ttr_g = 0;
    uint8_t best_ttr_p = 255;
    int best_p1 = pac_neighbors[0];
    int best_p2 = pac_neighbors[0];
    int best_pellet_mask = pellet_mask;
    int best_power_timer = power_timer;

    for (int pi = 0; pi < pac_n; pi++) {
        int np1 = pac_neighbors[pi];

        // Power state after first move
        int new_pellet_mask = pellet_mask;
        int new_power_timer = power_timer;

        int pellet_eaten = get_pellet_at(np1, pellet_mask);
        if (pellet_eaten >= 0) {
            new_pellet_mask = remove_pellet(pellet_mask, pellet_eaten);
            new_power_timer = power_duration;
        }

        bool will_be_powered = is_powered(new_power_timer) || pellet_eaten >= 0;

        // Second move options
        int pac_neighbors2[5], pac_n2;
        if (will_be_powered) {
            get_neighbors(np1, pac_neighbors2, pac_n2);
            pac_neighbors2[pac_n2++] = np1;  // Can stay
        } else {
            pac_n2 = 1;
            pac_neighbors2[0] = np1;
        }

        for (int pi2 = 0; pi2 < pac_n2; pi2++) {
            int np2 = pac_neighbors2[pi2];

            // Skip moves into ghosts when not powered
            if (!will_be_powered) {
                if ((g1_alive && np1 == g1) || (g2_alive && np1 == g2)) {
                    continue;
                }
            }

            // Check second move pellet
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
            uint8_t worst_ttr_p = 0;

            // For powered mode: check if ghosts are caught on Pacman's first move
            // If so, they don't get to move on the second (simultaneous) move
            bool g1_caught_move1 = will_be_powered && g1_alive && (np1 == g1);
            bool g2_caught_move1 = will_be_powered && g2_alive && (np1 == g2);

            // Adjust iteration: ghosts caught on move 1 have no valid moves
            int g1_iter_count = g1_caught_move1 ? 1 : g1_moves;
            int g2_iter_count = g2_caught_move1 ? 1 : g2_moves;

            for (int gi1 = 0; gi1 < g1_iter_count; gi1++) {
                for (int gi2 = 0; gi2 < g2_iter_count; gi2++) {
                    // Ghosts caught on move 1 are already DEAD for move 2
                    int ng1 = g1_caught_move1 ? DEAD : (g1_alive ? g1_neighbors[gi1] : DEAD);
                    int ng2 = g2_caught_move1 ? DEAD : (g2_alive ? g2_neighbors[gi2] : DEAD);

                    int result_g1 = ng1;
                    int result_g2 = ng2;
                    int result_pellet_mask = final_pellet_mask;
                    int result_power_timer = final_power_timer;
                    bool pacman_dies = false;

                    if (will_be_powered) {
                        // Powered: Pacman moves twice
                        // First move: Pacman moves to np1, ghosts stay at g1/g2
                        // Second move: Pacman moves to np2 SIMULTANEOUSLY with ghosts moving to ng1/ng2

                        // Ghosts caught on first move are already handled above (ng1/ng2 = DEAD)
                        bool g1_alive_after_move1 = g1_alive && !g1_caught_move1;
                        bool g2_alive_after_move1 = g2_alive && !g2_caught_move1;

                        // Check if Pacman catches ghost on second move (simultaneous collision)
                        bool catch1_m2 = g1_alive_after_move1 && (np2 == ng1);
                        bool catch2_m2 = g2_alive_after_move1 && (np2 == ng2);

                        // Check for clipping on second move (Pacman and ghost swap positions)
                        bool clip1 = g1_alive_after_move1 && (np1 == ng1 && g1 == np2);
                        bool clip2 = g2_alive_after_move1 && (np1 == ng2 && g2 == np2);

                        // Determine final ghost states
                        if (g1_caught_move1 || catch1_m2 || clip1) result_g1 = DEAD;
                        if (g2_caught_move1 || catch2_m2 || clip2) result_g2 = DEAD;
                    } else {
                        // Check Pacman capture
                        bool clip1 = g1_alive && (p == ng1 && g1 == np1);
                        bool clip2 = g2_alive && (p == ng2 && g2 == np1);
                        bool collide1 = g1_alive && (np1 == ng1 || clip1);
                        bool collide2 = g2_alive && (np1 == ng2 || clip2);

                        if (collide1 || collide2) pacman_dies = true;
                    }

                    // Decrement power timer
                    if (is_powered(result_power_timer)) {
                        result_power_timer--;
                    }

                    uint8_t s, tg, tp;
                    if (pacman_dies) {
                        s = 0; tg = 0; tp = 255;
                    } else if (result_g1 == DEAD && result_g2 == DEAD) {
                        s = 1; tg = 255; tp = 0;
                    } else {
                        int next_pw = encode_power_state(result_pellet_mask, result_power_timer);
                        size_t next_idx = value_index(np2, result_g1, result_g2, next_pw);
                        s = safety_value[next_idx];
                        tg = ttr_g_value[next_idx];
                        tp = ttr_p_value[next_idx];
                    }

                    worst_safety = min(worst_safety, s);
                    worst_ttr_g = min(worst_ttr_g, tg);
                    worst_ttr_p = max(worst_ttr_p, tp);
                }
            }

            // Pacman maximizes safety, then TTR_G, then minimizes TTR_P
            bool better = false;
            if (worst_safety > best_safety) {
                better = true;
            } else if (worst_safety == best_safety) {
                if (worst_ttr_g > best_ttr_g) {
                    better = true;
                } else if (worst_ttr_g == best_ttr_g && worst_ttr_p < best_ttr_p) {
                    better = true;
                }
            }

            if (better) {
                best_safety = worst_safety;
                best_ttr_g = worst_ttr_g;
                best_ttr_p = worst_ttr_p;
                best_p1 = np1;
                best_p2 = np2;
                best_pellet_mask = final_pellet_mask;
                best_power_timer = final_power_timer;
            }
        }
    }

    result.pos1 = best_p1;
    result.pos2 = best_p2;
    result.new_pellet_mask = best_pellet_mask;
    result.new_power_timer = best_power_timer;

    // Compute ghost response
    get_optimal_ghost_moves(p, best_p1, best_p2, g1, g2,
                           best_pellet_mask, best_power_timer,
                           result.new_g1, result.new_g2);

    return result;
}

void get_optimal_ghost_moves(int p, int np1, int np2, int g1, int g2,
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
    uint8_t worst_ttr_p = 0;
    bool found_catch = false;
    int best_catch_dist = 999;

    int g1_moves = g1_alive ? g1_n : 1;
    int g2_moves = g2_alive ? g2_n : 1;

    // For powered mode: check if ghosts are caught on Pacman's first move
    // If so, they don't get to move on the second (simultaneous) move
    bool g1_caught_move1 = powered && g1_alive && (np1 == g1);
    bool g2_caught_move1 = powered && g2_alive && (np1 == g2);

    // Adjust iteration: ghosts caught on move 1 have no valid moves
    int g1_iter_count = g1_caught_move1 ? 1 : g1_moves;
    int g2_iter_count = g2_caught_move1 ? 1 : g2_moves;

    for (int gi1 = 0; gi1 < g1_iter_count; gi1++) {
        for (int gi2 = 0; gi2 < g2_iter_count; gi2++) {
            // Ghosts caught on move 1 are already DEAD for move 2
            int ng1 = g1_caught_move1 ? DEAD : (g1_alive ? g1_neighbors[gi1] : DEAD);
            int ng2 = g2_caught_move1 ? DEAD : (g2_alive ? g2_neighbors[gi2] : DEAD);

            int result_g1 = ng1;
            int result_g2 = ng2;
            int result_power_timer = power_timer;

            if (powered) {
                // Ghosts want to avoid being eaten
                // Ghosts caught on first move are already handled above (ng1/ng2 = DEAD)
                bool g1_alive_after_move1 = g1_alive && !g1_caught_move1;
                bool g2_alive_after_move1 = g2_alive && !g2_caught_move1;

                // Check if Pacman catches ghost on second move (simultaneous collision)
                bool catch1_m2 = g1_alive_after_move1 && (np2 == ng1);
                bool catch2_m2 = g2_alive_after_move1 && (np2 == ng2);

                // Check for clipping on second move (Pacman and ghost swap positions)
                bool clip1 = g1_alive_after_move1 && (np1 == ng1 && g1 == np2);
                bool clip2 = g2_alive_after_move1 && (np1 == ng2 && g2 == np2);

                // Determine final ghost states
                if (g1_caught_move1 || catch1_m2 || clip1) result_g1 = DEAD;
                if (g2_caught_move1 || catch2_m2 || clip2) result_g2 = DEAD;
            } else {
                // Ghosts want to catch Pacman
                bool clip1 = g1_alive && (p == ng1 && g1 == np1);
                bool clip2 = g2_alive && (p == ng2 && g2 == np1);
                bool collide1 = g1_alive && (np1 == ng1 || clip1);
                bool collide2 = g2_alive && (np1 == ng2 || clip2);

                if (collide1 || collide2) {
                    int dist1 = g1_alive ? manhattan_dist(ng1, np1) : 999;
                    int dist2 = g2_alive ? manhattan_dist(ng2, np1) : 999;
                    int total_dist = dist1 + dist2;

                    if (!found_catch || total_dist < best_catch_dist) {
                        best_g1 = ng1;
                        best_g2 = ng2;
                        best_catch_dist = total_dist;
                        found_catch = true;
                    }
                    continue;
                }

                if (found_catch) continue;
            }

            if (is_powered(result_power_timer)) {
                result_power_timer--;
            }

            uint8_t s, tg;
            if (result_g1 == DEAD && result_g2 == DEAD) {
                s = 1; tg = 255;
            } else {
                int next_pw = encode_power_state(pellet_mask, result_power_timer);
                size_t next_idx = value_index(np2, result_g1, result_g2, next_pw);
                s = safety_value[next_idx];
                tg = ttr_g_value[next_idx];
            }

            // Ghosts minimize Pacman's values
            if (s < worst_safety || (s == worst_safety && tg < worst_ttr_g)) {
                worst_safety = s;
                worst_ttr_g = tg;
                best_g1 = ng1;
                best_g2 = ng2;
            }
        }
    }
}

// ============================================================================
// Simulation
// ============================================================================

void simulate_optimal_play(int start_p, int start_g1, int start_g2, int max_steps) {
    int p = start_p;
    int g1 = start_g1;
    int g2 = start_g2;
    int pellet_mask = num_pellet_masks - 1;  // All pellets exist
    int power_timer = 0;  // Not powered

    cout << "\n========================================" << endl;
    cout << "OPTIMAL PLAY SIMULATION" << endl;
    cout << "  " << num_pellets << " pellets, power duration " << power_duration << endl;
    cout << "  Pacman moves 2x when powered" << endl;
    cout << "========================================\n" << endl;

    cout << "Pellet positions: ";
    for (int i = 0; i < num_pellets; i++) {
        cout << pellet_positions[i] << " [" << row(pellet_positions[i]) << ","
             << col(pellet_positions[i]) << "]";
        if (i < num_pellets - 1) cout << ", ";
    }
    cout << endl << endl;

    // Initial state info
    int pw = encode_power_state(pellet_mask, power_timer);
    size_t idx = value_index(p, g1, g2, pw);
    cout << "Initial state:" << endl;
    cout << "  Safety: " << (int)safety_value[idx] << " (0=lose, 1=win)" << endl;
    cout << "  TTR_G: " << (int)ttr_g_value[idx] << " (survival time)" << endl;
    cout << "  TTR_P: " << (int)ttr_p_value[idx] << " (time to catch ghosts)" << endl << endl;

    print_game_state(p, g1, g2, pellet_mask, power_timer, 0);

    for (int step = 1; step <= max_steps; step++) {
        bool g1_alive = (g1 != DEAD);
        bool g2_alive = (g2 != DEAD);

        // Win condition
        if (!g1_alive && !g2_alive) {
            cout << "PACMAN WINS! Both ghosts eliminated!" << endl;
            break;
        }

        // Lose condition
        bool caught = (!is_powered(power_timer)) &&
                      ((g1_alive && p == g1) || (g2_alive && p == g2));
        if (caught) {
            cout << "Game over - Pacman caught at step " << (step-1) << "!" << endl;
            break;
        }

        // Get optimal move
        MoveResult move = get_optimal_pacman_move(p, g1, g2, pellet_mask, power_timer);

        // Check pellet eating
        int pellet_eaten = get_pellet_at(move.pos1, pellet_mask);
        if (pellet_eaten >= 0) {
            cout << ">>> Pacman eats pellet " << pellet_eaten << " at position "
                 << pellet_positions[pellet_eaten] << "! <<<" << endl;
        }

        bool will_be_powered = is_powered(move.new_power_timer) || pellet_eaten >= 0;

        // Display move
        if (will_be_powered && move.pos1 != move.pos2) {
            cout << "Pacman moves: " << p << " -> " << move.pos1 << " -> " << move.pos2
                 << " (double move!)" << endl;
        } else {
            cout << "Pacman moves: " << p << " -> " << move.pos1 << endl;
        }

        // Check ghost eating on first move
        if (will_be_powered) {
            if (g1_alive && move.pos1 == g1) {
                cout << ">>> Pacman eats Ghost 1! <<<" << endl;
                move.new_g1 = DEAD;
            }
            if (g2_alive && move.pos1 == g2) {
                cout << ">>> Pacman eats Ghost 2! <<<" << endl;
                move.new_g2 = DEAD;
            }
        }

        // Check ghost eating on second move
        g1_alive = (move.new_g1 != DEAD);
        g2_alive = (move.new_g2 != DEAD);

        if (will_be_powered) {
            if (g1_alive && move.pos2 == move.new_g1) {
                cout << ">>> Pacman eats Ghost 1 on second move! <<<" << endl;
                move.new_g1 = DEAD;
            }
            if (g2_alive && move.pos2 == move.new_g2) {
                cout << ">>> Pacman eats Ghost 2 on second move! <<<" << endl;
                move.new_g2 = DEAD;
            }
        }

        // Check collision when not powered
        bool collision = false;
        if (!will_be_powered) {
            g1_alive = (move.new_g1 != DEAD);
            g2_alive = (move.new_g2 != DEAD);
            bool clip1 = g1_alive && (p == move.new_g1 && g1 == move.pos1);
            bool clip2 = g2_alive && (p == move.new_g2 && g2 == move.pos1);
            bool collide1 = g1_alive && (move.pos1 == move.new_g1 || clip1);
            bool collide2 = g2_alive && (move.pos1 == move.new_g2 || clip2);
            collision = collide1 || collide2;
        }

        // Decrement power timer
        int new_power_timer = move.new_power_timer;
        if (is_powered(new_power_timer)) {
            new_power_timer--;
        }

        // Update state
        p = move.pos2;
        g1 = move.new_g1;
        g2 = move.new_g2;
        pellet_mask = move.new_pellet_mask;
        power_timer = new_power_timer;

        print_game_state(p, g1, g2, pellet_mask, power_timer, step);

        // Check win/lose after move
        g1_alive = (g1 != DEAD);
        g2_alive = (g2 != DEAD);

        if (!g1_alive && !g2_alive) {
            cout << "PACMAN WINS! Both ghosts eliminated!" << endl;
            break;
        }

        if (collision) {
            cout << "COLLISION: Pacman caught by ghost!" << endl;
            cout << "Game over - Pacman caught at step " << step << "!" << endl;
            break;
        }

        // Show state value
        pw = encode_power_state(pellet_mask, power_timer);
        idx = value_index(p, g1, g2, pw);
        cout << "State: Safety=" << (int)safety_value[idx]
             << ", TTR_G=" << (int)ttr_g_value[idx]
             << ", TTR_P=" << (int)ttr_p_value[idx] << endl;
        cout << "----------------------------------------" << endl << endl;
    }
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* program) {
    cout << "Usage: " << program << " [options]" << endl;
    cout << endl;
    cout << "Required:" << endl;
    cout << "  -v, --values FILE   Optimized value table file (.bin.gz)" << endl;
    cout << endl;
    cout << "Optional:" << endl;
    cout << "  -p, --pacman POS    Initial Pacman position (default: 64)" << endl;
    cout << "  -1, --ghost1 POS    Initial Ghost1 position (default: 22)" << endl;
    cout << "  -2, --ghost2 POS    Initial Ghost2 position (default: 130)" << endl;
    cout << "  -m, --max-steps N   Maximum simulation steps (default: 100)" << endl;
    cout << "  -h, --help          Show this help message" << endl;
}

int main(int argc, char* argv[]) {
    string values_file = "";
    int start_p = 64;
    int start_g1 = 22;
    int start_g2 = 130;
    int max_steps = 100;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "-v" || arg == "--values") && i + 1 < argc) {
            values_file = argv[++i];
        } else if ((arg == "-p" || arg == "--pacman") && i + 1 < argc) {
            start_p = stoi(argv[++i]);
        } else if ((arg == "-1" || arg == "--ghost1") && i + 1 < argc) {
            start_g1 = stoi(argv[++i]);
        } else if ((arg == "-2" || arg == "--ghost2") && i + 1 < argc) {
            start_g2 = stoi(argv[++i]);
        } else if ((arg == "-m" || arg == "--max-steps") && i + 1 < argc) {
            max_steps = stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            cerr << "Unknown argument: " << arg << endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (values_file.empty()) {
        cerr << "Error: Values file is required" << endl << endl;
        print_usage(argv[0]);
        return 1;
    }

    cout << "=== Pacman Simulation (Multi-Pellet Optimized) ===" << endl;

    if (!load_values(values_file)) {
        return 1;
    }

    cout << "Maze: " << MAZE_ROWS << "x" << MAZE_COLS << endl;

    // Validate positions
    if (!is_valid_pos(start_p)) {
        cerr << "Error: Invalid Pacman position " << start_p << endl;
        free_tables();
        return 1;
    }
    if (!is_valid_pos(start_g1)) {
        cerr << "Error: Invalid Ghost1 position " << start_g1 << endl;
        free_tables();
        return 1;
    }
    if (!is_valid_pos(start_g2)) {
        cerr << "Error: Invalid Ghost2 position " << start_g2 << endl;
        free_tables();
        return 1;
    }

    simulate_optimal_play(start_p, start_g1, start_g2, max_steps);

    free_tables();
    return 0;
}

