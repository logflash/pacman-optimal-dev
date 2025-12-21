/* Optimized value iteration with multiple super pellets
 *
 * Optimizations:
 *   1. Compact power state encoding - only valid states enumerated
 *   2. gzip compression on output file
 *
 * State encoding:
 *   Power states are compactly encoded:
 *   - States 0 to 2^N - 1: Not powered, pellet_mask = state
 *   - States 2^N onwards: Powered, encoding (mask, timer)
 *
 *   Total states = 2^N + (2^N - 1) * D
 *   where N = num_pellets, D = power_duration
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <zlib.h>
using namespace std;
using namespace std::chrono;

#define MAZE_ROWS 12
#define MAZE_COLS 12
#define MAZE_CELLS (MAZE_ROWS * MAZE_COLS)

#define DEAD MAZE_CELLS
#define GHOST_STATES (MAZE_CELLS + 1)

#define MAX_PELLETS 4
#define MAX_POWER_DURATION 30

// Runtime configuration
int num_pellets = 2;
int power_duration = 6;
int pellet_positions[MAX_PELLETS] = {52, 76, 0, 0};

// Derived constants
int num_pellet_masks;      // 2^num_pellets
int total_power_states;    // Compact: 2^N + (2^N - 1) * D

// Maze layout
uint32_t maze[MAZE_ROWS] = {
    0b000000000000,
    0b011111111110,
    0b010010010010,
    0b010010010010,
    0b011111111110,
    0b010010010010,
    0b010010010010,
    0b011111111110,
    0b010010010010,
    0b010010010010,
    0b011111111110,
    0b000000000000,
};

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
//
// Layout:
//   [0, 2^N - 1]: Not powered states, index = pellet_mask
//   [2^N, ...]:   Powered states, index = 2^N + mask * D + (timer - 1)
//
// Valid powered states: mask can be 0 to 2^N - 2 (at least one pellet eaten)
// Timer: 1 to D (power_duration)

inline int encode_power_state(int pellet_mask, int power_timer) {
    if (power_timer == 0) {
        // Not powered
        return pellet_mask;
    } else {
        // Powered: base + mask * duration + (timer - 1)
        return num_pellet_masks + pellet_mask * power_duration + (power_timer - 1);
    }
}

inline void decode_power_state(int state, int& pellet_mask, int& power_timer) {
    if (state < num_pellet_masks) {
        // Not powered
        pellet_mask = state;
        power_timer = 0;
    } else {
        // Powered
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

// Check if a power state is valid
inline bool is_valid_power_state(int state) {
    int pellet_mask, power_timer;
    decode_power_state(state, pellet_mask, power_timer);

    if (power_timer == 0) {
        // Not powered - all masks valid
        return true;
    } else {
        // Powered - must have eaten at least one pellet
        // i.e., mask < all_pellets_mask
        return pellet_mask < (num_pellet_masks - 1);
    }
}

// ============================================================================
// Value Tables
// ============================================================================

uint8_t* safety_value = nullptr;
uint8_t* ttr_g_value = nullptr;
uint8_t* ttr_p_value = nullptr;
size_t table_size = 0;

inline size_t value_index(int p, int g1, int g2, int pw) {
    return ((size_t)p * GHOST_STATES * GHOST_STATES +
            (size_t)g1 * GHOST_STATES +
            (size_t)g2) * total_power_states + pw;
}

bool allocate_tables() {
    table_size = (size_t)MAZE_CELLS * GHOST_STATES * GHOST_STATES * total_power_states;

    cout << "Allocating value tables..." << endl;
    cout << "  Power states (compact): " << total_power_states << endl;
    cout << "  Table entries: " << table_size << endl;
    cout << "  Memory per table: " << (table_size / 1024 / 1024) << " MB" << endl;
    cout << "  Total memory: " << (3 * table_size / 1024 / 1024) << " MB" << endl;

    try {
        safety_value = new uint8_t[table_size];
        ttr_g_value = new uint8_t[table_size];
        ttr_p_value = new uint8_t[table_size];
    } catch (const bad_alloc& e) {
        cerr << "Error: Failed to allocate memory" << endl;
        return false;
    }

    return true;
}

void free_tables() {
    delete[] safety_value;
    delete[] ttr_g_value;
    delete[] ttr_p_value;
    safety_value = ttr_g_value = ttr_p_value = nullptr;
}

// ============================================================================
// Value Iteration
// ============================================================================

void print_progress(int iter, int max_iters, double elapsed_sec, bool converged) {
    double progress = (double)(iter + 1) / max_iters;
    int bar_width = 40;
    int filled = (int)(progress * bar_width);

    double time_per_iter = elapsed_sec / (iter + 1);
    double remaining_sec = time_per_iter * (max_iters - iter - 1);

    cout << "\r[";
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) cout << "=";
        else if (i == filled) cout << ">";
        else cout << " ";
    }
    cout << "] ";
    cout << setw(4) << (iter + 1) << "/" << max_iters << " ";
    cout << "Elapsed: " << fixed << setprecision(1) << elapsed_sec << "s ";

    if (!converged) {
        cout << "ETA: " << fixed << setprecision(1) << remaining_sec << "s ";
    } else {
        cout << "CONVERGED!        ";
    }
    cout << flush;
}

void run_value_iteration() {
    cout << "Initializing value tables..." << endl;

    int all_pellets_mask = num_pellet_masks - 1;

    // Initialize all states
    #pragma omp parallel for collapse(3)
    for (int p = 0; p < MAZE_CELLS; p++) {
        for (int g1 = 0; g1 < GHOST_STATES; g1++) {
            for (int g2 = 0; g2 < GHOST_STATES; g2++) {
                for (int pw = 0; pw < total_power_states; pw++) {
                    if (!is_valid_power_state(pw)) continue;

                    size_t idx = value_index(p, g1, g2, pw);

                    int pellet_mask, power_timer;
                    decode_power_state(pw, pellet_mask, power_timer);

                    // Terminal win: both ghosts dead
                    if (g1 == DEAD && g2 == DEAD) {
                        safety_value[idx] = 1;
                        ttr_g_value[idx] = 255;
                        ttr_p_value[idx] = 0;  // Already caught at least one ghost
                        continue;
                    }

                    // At least one ghost caught
                    if (g1 == DEAD || g2 == DEAD) {
                        ttr_p_value[idx] = 0;  // Already caught at least one ghost
                    }

                    bool g1_alive = (g1 != DEAD && is_valid_pos(g1));
                    bool g2_alive = (g2 != DEAD && is_valid_pos(g2));
                    bool collision = (g1_alive && p == g1) || (g2_alive && p == g2);

                    if (collision && !is_powered(power_timer)) {
                        safety_value[idx] = 0;
                        ttr_g_value[idx] = 0;
                        if (g1 != DEAD && g2 != DEAD) {
                            ttr_p_value[idx] = 255;  // Pacman can't catch ghosts (dead)
                        }
                    } else {
                        safety_value[idx] = 1;
                        ttr_g_value[idx] = 255;
                        if (g1 != DEAD && g2 != DEAD) {
                            ttr_p_value[idx] = 255;  // Will be computed
                        }
                    }
                }
            }
        }
    }

    cout << "Running value iteration..." << endl;

    const int MAX_ITERS = 1000;
    bool converged;
    auto start_time = high_resolution_clock::now();

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        converged = true;

        #pragma omp parallel for collapse(2) schedule(dynamic) reduction(&&:converged)
        for (int p = 0; p < MAZE_CELLS; p++) {
            for (int g1 = 0; g1 < GHOST_STATES; g1++) {
                if (!is_valid_pos(p)) continue;
                if (g1 != DEAD && !is_valid_pos(g1)) continue;

                for (int g2 = 0; g2 < GHOST_STATES; g2++) {
                    if (g2 != DEAD && !is_valid_pos(g2)) continue;

                    for (int pw = 0; pw < total_power_states; pw++) {
                        if (!is_valid_power_state(pw)) continue;

                        size_t idx = value_index(p, g1, g2, pw);

                        if (g1 == DEAD && g2 == DEAD) continue;

                        int pellet_mask, power_timer;
                        decode_power_state(pw, pellet_mask, power_timer);

                        bool g1_alive = (g1 != DEAD);
                        bool g2_alive = (g2 != DEAD);
                        bool powered = is_powered(power_timer);

                        bool collision = (g1_alive && p == g1) || (g2_alive && p == g2);
                        if (collision && !powered) continue;

                        int pac_neighbors[4], pac_n;
                        get_neighbors(p, pac_neighbors, pac_n);
                        if (pac_n == 0) continue;

                        uint8_t best_safety = 0;
                        uint8_t best_ttr_g = 0;
                        uint8_t best_ttr_p = 255;

                        int g1_neighbors[4], g1_n = 0;
                        int g2_neighbors[4], g2_n = 0;

                        if (g1_alive) {
                            get_neighbors(g1, g1_neighbors, g1_n);
                            if (g1_n == 0) continue;
                        }
                        if (g2_alive) {
                            get_neighbors(g2, g2_neighbors, g2_n);
                            if (g2_n == 0) continue;
                        }

                        if (!g1_alive && !g2_alive) continue;

                        int g1_moves = g1_alive ? g1_n : 1;
                        int g2_moves = g2_alive ? g2_n : 1;

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

                            int pac_neighbors2[5], pac_n2;
                            if (will_be_powered) {
                                get_neighbors(np1, pac_neighbors2, pac_n2);
                                pac_neighbors2[pac_n2++] = np1;
                            } else {
                                pac_n2 = 1;
                                pac_neighbors2[0] = np1;
                            }

                            for (int pi2 = 0; pi2 < pac_n2; pi2++) {
                                int np2 = pac_neighbors2[pi2];

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

                                            // Second move is simultaneous: Pacman goes np1->np2, ghosts go g1->ng1, g2->ng2
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
                                        } else if (result_g1 == DEAD || result_g2 == DEAD) {
                                            // Pacman caught at least one ghost!
                                            tp = 0;  // Already caught at least one ghost
                                            if (result_g1 == DEAD && result_g2 == DEAD) {
                                                // Both ghosts dead
                                                s = 1;
                                                tg = 255;  // Ghosts can never catch Pacman
                                            } else {
                                                // One ghost dead, one still alive
                                                int next_pw = encode_power_state(result_pellet_mask, result_power_timer);
                                                size_t next_idx = value_index(np2, result_g1, result_g2, next_pw);
                                                s = safety_value[next_idx];
                                                tg = ttr_g_value[next_idx];
                                                // Ensure ttr_p is 0 for any state with a dead ghost
                                                tp = 0;
                                            }
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

                                best_safety = max(best_safety, worst_safety);
                                uint8_t new_ttr_g = (worst_ttr_g >= 255) ? 255 : (worst_ttr_g + 1);
                                best_ttr_g = max(best_ttr_g, new_ttr_g);
                                uint8_t new_ttr_p = (worst_ttr_p >= 255) ? 255 : (worst_ttr_p + 1);
                                best_ttr_p = min(best_ttr_p, new_ttr_p);
                            }
                        }

                        // Enforce ttr_p = 0 for any state with a dead ghost
                        if (g1 == DEAD || g2 == DEAD) {
                            best_ttr_p = 0;
                        }

                        if (best_safety != safety_value[idx] ||
                            best_ttr_g != ttr_g_value[idx] ||
                            best_ttr_p != ttr_p_value[idx]) {
                            converged = false;
                            safety_value[idx] = best_safety;
                            ttr_g_value[idx] = best_ttr_g;
                            ttr_p_value[idx] = best_ttr_p;
                        }
                    }
                }
            }
        }

        auto now = high_resolution_clock::now();
        double elapsed = duration_cast<milliseconds>(now - start_time).count() / 1000.0;
        print_progress(iter, MAX_ITERS, elapsed, converged);

        if (converged) {
            cout << endl;
            break;
        }
    }
}

// ============================================================================
// File I/O with gzip compression
// ============================================================================

struct ValueFileHeaderOpt {
    char magic[4];              // "PVAO" for Pacman Value Optimized
    uint8_t version;            // 1
    uint8_t value_type;         // 2 = combined
    uint16_t maze_rows;
    uint16_t maze_cols;
    uint8_t num_ghosts;
    uint8_t num_pellets;
    uint8_t power_duration;
    uint8_t compression;        // 0 = none, 1 = gzip
    uint16_t pellet_positions[MAX_PELLETS];
    uint32_t total_power_states;
    uint32_t uncompressed_size;  // Size of value data before compression
};

bool save_values_compressed(const string& filename) {
    // First, write to a memory buffer
    size_t data_size = 3 * table_size;  // safety + ttr_g + ttr_p

    cout << "Compressing data (" << (data_size / 1024 / 1024) << " MB)..." << endl;

    // Compress using zlib
    uLongf compressed_size = compressBound(data_size);
    vector<uint8_t> compressed(compressed_size);

    // Combine all three tables into one buffer for compression
    vector<uint8_t> combined(data_size);
    memcpy(combined.data(), safety_value, table_size);
    memcpy(combined.data() + table_size, ttr_g_value, table_size);
    memcpy(combined.data() + 2 * table_size, ttr_p_value, table_size);

    int result = compress2(compressed.data(), &compressed_size,
                          combined.data(), data_size, Z_BEST_COMPRESSION);

    if (result != Z_OK) {
        cerr << "Error: Compression failed with code " << result << endl;
        return false;
    }

    double ratio = (double)data_size / compressed_size;
    cout << "  Compressed: " << (compressed_size / 1024 / 1024) << " MB "
         << "(ratio: " << fixed << setprecision(1) << ratio << "x)" << endl;

    // Write to file
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Could not open " << filename << " for writing" << endl;
        return false;
    }

    // Write header
    ValueFileHeaderOpt header;
    memcpy(header.magic, "PVAO", 4);
    header.version = 1;
    header.value_type = 2;
    header.maze_rows = MAZE_ROWS;
    header.maze_cols = MAZE_COLS;
    header.num_ghosts = 2;
    header.num_pellets = num_pellets;
    header.power_duration = power_duration;
    header.compression = 1;  // gzip
    for (int i = 0; i < MAX_PELLETS; i++) {
        header.pellet_positions[i] = (i < num_pellets) ? pellet_positions[i] : 0;
    }
    header.total_power_states = total_power_states;
    header.uncompressed_size = data_size;

    file.write(reinterpret_cast<char*>(&header), sizeof(header));

    // Write maze
    file.write(reinterpret_cast<char*>(maze), sizeof(maze));

    // Write compressed size and data
    uint32_t comp_size_32 = compressed_size;
    file.write(reinterpret_cast<char*>(&comp_size_32), sizeof(comp_size_32));
    file.write(reinterpret_cast<char*>(compressed.data()), compressed_size);

    file.close();
    return true;
}

// ============================================================================
// Statistics
// ============================================================================

void print_statistics() {
    int all_pellets_mask = num_pellet_masks - 1;
    int initial_state = encode_power_state(all_pellets_mask, 0);

    long long safe_count = 0;
    long long unsafe_count = 0;
    long long total_valid = 0;

    for (int p = 0; p < MAZE_CELLS; p++) {
        if (!is_valid_pos(p)) continue;
        for (int g1 = 0; g1 < MAZE_CELLS; g1++) {
            if (!is_valid_pos(g1)) continue;
            for (int g2 = 0; g2 < MAZE_CELLS; g2++) {
                if (!is_valid_pos(g2)) continue;
                if (p == g1 || p == g2) continue;

                total_valid++;
                size_t idx = value_index(p, g1, g2, initial_state);
                if (safety_value[idx] == 1) {
                    safe_count++;
                } else {
                    unsafe_count++;
                }
            }
        }
    }

    cout << "\n=== Statistics (all pellets exist, not powered) ===" << endl;
    cout << "Total valid states: " << total_valid << endl;
    cout << "Safe (Pacman can win): " << safe_count
         << " (" << fixed << setprecision(2) << (100.0 * safe_count / total_valid) << "%)" << endl;
    cout << "Unsafe (ghosts win): " << unsafe_count
         << " (" << fixed << setprecision(2) << (100.0 * unsafe_count / total_valid) << "%)" << endl;

    // Per pellet count
    cout << "\n=== Safety by pellets remaining ===" << endl;
    for (int pellet_count = 0; pellet_count <= num_pellets; pellet_count++) {
        long long safe = 0, total = 0;

        for (int mask = 0; mask < num_pellet_masks; mask++) {
            if (__builtin_popcount(mask) != pellet_count) continue;

            int pw = encode_power_state(mask, 0);

            for (int p = 0; p < MAZE_CELLS; p++) {
                if (!is_valid_pos(p)) continue;
                for (int g1 = 0; g1 < MAZE_CELLS; g1++) {
                    if (!is_valid_pos(g1)) continue;
                    for (int g2 = 0; g2 < MAZE_CELLS; g2++) {
                        if (!is_valid_pos(g2)) continue;
                        if (p == g1 || p == g2) continue;

                        total++;
                        size_t idx = value_index(p, g1, g2, pw);
                        if (safety_value[idx] == 1) safe++;
                    }
                }
            }
        }

        if (total > 0) {
            cout << "  " << pellet_count << " pellets: "
                 << safe << "/" << total << " safe ("
                 << fixed << setprecision(1) << (100.0 * safe / total) << "%)" << endl;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* program) {
    cout << "Usage: " << program << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  -o, --output FILE     Output file (default: values_opt.bin.gz)" << endl;
    cout << "  -n, --num-pellets N   Number of pellets, 1-4 (default: 2)" << endl;
    cout << "  -d, --duration N      Power-up duration in turns (default: 10, max: 30)" << endl;
    cout << "  -p, --pellets P1,...  Pellet positions (comma-separated)" << endl;
    cout << "  -h, --help            Show this help message" << endl;
}

bool parse_pellet_positions(const string& arg) {
    vector<int> positions;
    size_t pos = 0;
    string s = arg;

    while ((pos = s.find(',')) != string::npos) {
        positions.push_back(stoi(s.substr(0, pos)));
        s.erase(0, pos + 1);
    }
    if (!s.empty()) {
        positions.push_back(stoi(s));
    }

    if (positions.size() > MAX_PELLETS) {
        cerr << "Error: Maximum " << MAX_PELLETS << " pellets supported" << endl;
        return false;
    }

    num_pellets = positions.size();
    for (size_t i = 0; i < positions.size(); i++) {
        if (!is_valid_pos(positions[i])) {
            cerr << "Error: Invalid pellet position " << positions[i] << endl;
            return false;
        }
        pellet_positions[i] = positions[i];
    }

    return true;
}

void set_default_pellet_positions() {
    switch (num_pellets) {
        case 1:
            pellet_positions[0] = 64;
            break;
        case 2:
            pellet_positions[0] = 52;
            pellet_positions[1] = 76;
            break;
        case 3:
            pellet_positions[0] = 25;
            pellet_positions[1] = 64;
            pellet_positions[2] = 118;
            break;
        case 4:
            pellet_positions[0] = 25;
            pellet_positions[1] = 34;
            pellet_positions[2] = 109;
            pellet_positions[3] = 118;
            break;
    }
}

int main(int argc, char* argv[]) {
    string output_file = "values_opt.bin.gz";
    bool custom_pellets = false;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((arg == "-n" || arg == "--num-pellets") && i + 1 < argc) {
            num_pellets = stoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--duration") && i + 1 < argc) {
            power_duration = stoi(argv[++i]);
        } else if ((arg == "-p" || arg == "--pellets") && i + 1 < argc) {
            if (!parse_pellet_positions(argv[++i])) return 1;
            custom_pellets = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (num_pellets < 1 || num_pellets > MAX_PELLETS) {
        cerr << "Error: Number of pellets must be 1-" << MAX_PELLETS << endl;
        return 1;
    }

    if (power_duration < 1 || power_duration > MAX_POWER_DURATION) {
        cerr << "Error: Power duration must be 1-" << MAX_POWER_DURATION << endl;
        return 1;
    }

    if (!custom_pellets) {
        set_default_pellet_positions();
    }

    // Compute derived constants
    num_pellet_masks = 1 << num_pellets;
    // Compact encoding: 2^N not-powered + (2^N - 1) * D powered states
    total_power_states = num_pellet_masks + (num_pellet_masks - 1) * power_duration;

    cout << "=== Pacman Value Iteration (Optimized) ===" << endl;
    cout << "Maze: " << MAZE_ROWS << "x" << MAZE_COLS << endl;
    cout << "Ghosts: 2 (with permadeath)" << endl;
    cout << "Pellets: " << num_pellets << endl;
    cout << "  Positions: ";
    for (int i = 0; i < num_pellets; i++) {
        cout << pellet_positions[i] << " [" << row(pellet_positions[i]) << ","
             << col(pellet_positions[i]) << "]";
        if (i < num_pellets - 1) cout << ", ";
    }
    cout << endl;
    cout << "Power duration: " << power_duration << " turns" << endl;
    cout << "Power states (compact): " << total_power_states;

    // Show comparison with naive encoding
    int naive_states = num_pellet_masks * (power_duration + 2);
    cout << " (vs " << naive_states << " naive, "
         << fixed << setprecision(1) << ((double)naive_states / total_power_states) << "x reduction)" << endl;

    cout << "Output: " << output_file << endl;
    cout << endl;

    if (!allocate_tables()) {
        return 1;
    }

    auto start = high_resolution_clock::now();
    run_value_iteration();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "\nTotal computation time: " << (duration.count() / 1000.0) << " s" << endl;

    print_statistics();

    cout << "\nSaving values to " << output_file << "..." << endl;
    if (!save_values_compressed(output_file)) {
        free_tables();
        return 1;
    }

    // Report file size
    ifstream check(output_file, ios::binary | ios::ate);
    if (check) {
        size_t file_size = check.tellg();
        cout << "Final file size: " << (file_size / 1024 / 1024) << " MB" << endl;
    }

    cout << "Done!" << endl;

    free_tables();
    return 0;
}

