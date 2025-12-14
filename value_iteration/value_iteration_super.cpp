/* Value iteration code with two ghosts, Pacman, and super pellet
 * Features:
 *   - 1 super pellet (no respawn)
 *   - 10-turn power-up duration
 *   - Ghost permadeath when eaten by powered Pacman
 * Outputs combined value table to binary file
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <chrono>
#include <string>
using namespace std;
using namespace std::chrono;

#define MAZE_ROWS 12
#define MAZE_COLS 12
#define MAZE_CELLS (MAZE_ROWS * MAZE_COLS)

// Ghost position encoding: 0 to MAZE_CELLS-1 = alive at position
//                          DEAD = ghost has been eaten (permadeath)
#define DEAD MAZE_CELLS
#define GHOST_STATES (MAZE_CELLS + 1)  // 145 states per ghost

// Power-pellet state encoding (12 states total):
// 0 = pellet exists on map, Pacman not powered
// 1-10 = pellet eaten, Pacman powered with N turns remaining
// 11 = pellet eaten, power expired (Pacman no longer powered)
// Power duration (configurable, but arrays sized for max 30)
#define MAX_POWER_DURATION 30
#define PELLET_EXISTS 0
// POWER_EXPIRED = power_duration + 1
// POWER_STATES = power_duration + 2
#define MAX_POWER_STATES (MAX_POWER_DURATION + 2)

int power_duration = 10;  // Default, can be set via command line

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

// Super pellet location (configurable)
int pellet_pos = 64;  // Default: center-ish position (row 5, col 4)

inline int row(int idx) { return idx / MAZE_COLS; }
inline int col(int idx) { return idx % MAZE_COLS; }

inline bool is_free(int r, int c) {
    return (maze[r] >> c) & 1;
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

// Value tables: [pacman_pos][ghost1][ghost2][power_state]
// Size: 144 * 145 * 145 * 32 * 2 bytes = ~192 MB total (max)
uint8_t safety_value[MAZE_CELLS][GHOST_STATES][GHOST_STATES][MAX_POWER_STATES];
uint8_t ttr_value[MAZE_CELLS][GHOST_STATES][GHOST_STATES][MAX_POWER_STATES];

// Computed constants based on power_duration
inline int power_expired() { return power_duration + 1; }
inline int power_states() { return power_duration + 2; }

// Helper to check if Pacman is powered
inline bool is_powered(int power_state) {
    return power_state >= 1 && power_state <= power_duration;
}

// Get next power state after a turn
inline int next_power_state(int current, int pacman_pos) {
    if (current == PELLET_EXISTS) {
        // Pellet still exists - check if Pacman eats it
        if (pacman_pos == pellet_pos) {
            return power_duration;  // Start powered!
        }
        return PELLET_EXISTS;  // Pellet still there
    } else if (current >= 1 && current <= power_duration) {
        // Powered - decrement timer
        return current - 1;
    }
    // Power expired or invalid - stays expired
    return power_expired();
}

// Correct next_power_state: timer decrements, 1->power_expired()
inline int decrement_power(int current) {
    if (current >= 2 && current <= power_duration) {
        return current - 1;
    } else if (current == 1) {
        return power_expired();  // Power just ran out
    }
    return current;  // PELLET_EXISTS or power_expired() stay same
}

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
    cout << "DEBUG: power_duration=" << power_duration 
         << ", power_states()=" << power_states() 
         << ", power_expired()=" << power_expired() << endl;
    
    // Initialize all states
    #pragma omp parallel for collapse(3)
    for (int p = 0; p < MAZE_CELLS; p++) {
        for (int g1 = 0; g1 < GHOST_STATES; g1++) {
            for (int g2 = 0; g2 < GHOST_STATES; g2++) {
                for (int pw = 0; pw < power_states(); pw++) {
                    // Terminal win: both ghosts dead
                    if (g1 == DEAD && g2 == DEAD) {
                        safety_value[p][g1][g2][pw] = 1;
                        ttr_value[p][g1][g2][pw] = 0;
                        continue;
                    }
                    
                    // Check for collision (Pacman caught by ghost)
                    bool g1_alive = (g1 != DEAD && is_valid_pos(g1));
                    bool g2_alive = (g2 != DEAD && is_valid_pos(g2));
                    bool collision = (g1_alive && p == g1) || (g2_alive && p == g2);
                    
                    if (collision && !is_powered(pw)) {
                        // Pacman caught while not powered = lose
                        safety_value[p][g1][g2][pw] = 0;
                        ttr_value[p][g1][g2][pw] = 0;
                    } else {
                        // Default: assume safe, will be updated by iteration
                        safety_value[p][g1][g2][pw] = 1;
                        ttr_value[p][g1][g2][pw] = 255;
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
                    
                    for (int pw = 0; pw < power_states(); pw++) {
                        // Skip terminal states
                        if (g1 == DEAD && g2 == DEAD) continue;
                        
                        bool g1_alive = (g1 != DEAD);
                        bool g2_alive = (g2 != DEAD);
                        bool powered = is_powered(pw);
                        
                        // Check current collision
                        bool collision = (g1_alive && p == g1) || (g2_alive && p == g2);
                        if (collision && !powered) {
                            // Already set to lose
                            continue;
                        }
                        
                        // Get Pacman's possible moves
                        int pac_neighbors[4], pac_n;
                        get_neighbors(p, pac_neighbors, pac_n);
                        if (pac_n == 0) continue;
                        
                        uint8_t best_safety = 0;
                        uint8_t best_ttr = 0;
                        
                        // Pacman chooses move to maximize value
                        for (int pi = 0; pi < pac_n; pi++) {
                            int np = pac_neighbors[pi];
                            
                            // Compute power state after Pacman moves
                            int next_pw = pw;
                            if (pw == PELLET_EXISTS && np == pellet_pos) {
                                next_pw = power_duration;  // Pacman eats pellet
                            }
                            bool will_be_powered = is_powered(next_pw);
                            
                            // Get ghost moves (only for alive ghosts)
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
                            
                            // If no ghosts alive, this shouldn't happen (terminal)
                            if (!g1_alive && !g2_alive) continue;
                            
                            uint8_t worst_safety = 1;
                            uint8_t worst_ttr = 255;
                            
                            // Iterate over ghost move combinations
                            int g1_moves = g1_alive ? g1_n : 1;
                            int g2_moves = g2_alive ? g2_n : 1;
                            
                            for (int gi1 = 0; gi1 < g1_moves; gi1++) {
                                for (int gi2 = 0; gi2 < g2_moves; gi2++) {
                                    int ng1 = g1_alive ? g1_neighbors[gi1] : DEAD;
                                    int ng2 = g2_alive ? g2_neighbors[gi2] : DEAD;
                                    
                                    // Check for clipping (position swap)
                                    bool clip1 = g1_alive && (p == ng1 && g1 == np);
                                    bool clip2 = g2_alive && (p == ng2 && g2 == np);
                                    
                                    // Determine outcome of this move combination
                                    int final_g1 = ng1;
                                    int final_g2 = ng2;
                                    int final_pw = next_pw;
                                    bool pacman_dies = false;
                                    
                                    // Check collisions after movement
                                    bool collide1 = g1_alive && (np == ng1 || clip1);
                                    bool collide2 = g2_alive && (np == ng2 || clip2);
                                    
                                    if (will_be_powered) {
                                        // Pacman eats ghosts on collision
                                        if (collide1) final_g1 = DEAD;
                                        if (collide2) final_g2 = DEAD;
                                    } else {
                                        // Ghosts eat Pacman on collision
                                        if (collide1 || collide2) {
                                            pacman_dies = true;
                                        }
                                    }
                                    
                                    // Decrement power timer for next state
                                    if (is_powered(final_pw)) {
                                        final_pw = decrement_power(final_pw);
                                    }
                                    
                                    uint8_t s, t;
                                    if (pacman_dies) {
                                        s = 0;
                                        t = 0;
                                    } else if (final_g1 == DEAD && final_g2 == DEAD) {
                                        // Pacman wins!
                                        s = 1;
                                        t = 0;
                                    } else {
                                        s = safety_value[np][final_g1][final_g2][final_pw];
                                        t = ttr_value[np][final_g1][final_g2][final_pw];
                                    }
                                    
                                    worst_safety = min(worst_safety, s);
                                    worst_ttr = min(worst_ttr, t);
                                }
                            }
                            
                            best_safety = max(best_safety, worst_safety);
                            uint8_t new_ttr = (worst_ttr >= 255) ? 255 : (worst_ttr + 1);
                            best_ttr = max(best_ttr, new_ttr);
                        }
                        
                        if (best_safety != safety_value[p][g1][g2][pw] ||
                            best_ttr != ttr_value[p][g1][g2][pw]) {
                            converged = false;
                            safety_value[p][g1][g2][pw] = best_safety;
                            ttr_value[p][g1][g2][pw] = best_ttr;
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

// Extended header for super pellet version
struct ValueFileHeaderSuper {
    char magic[4];          // "PVAS" for Pacman Value with Super pellet
    uint8_t version;        // File format version (2 for super pellet)
    uint8_t value_type;     // 0 = safety, 1 = TTR, 2 = combined
    uint16_t maze_rows;
    uint16_t maze_cols;
    uint8_t num_ghosts;
    uint8_t power_duration;
    uint16_t pellet_pos;
    uint8_t reserved[2];    // Padding for alignment
};

bool save_values(const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Could not open " << filename << " for writing" << endl;
        return false;
    }
    
    // Write header
    ValueFileHeaderSuper header;
    memcpy(header.magic, "PVAS", 4);
    header.version = 2;
    header.value_type = 2;  // Combined
    header.maze_rows = MAZE_ROWS;
    header.maze_cols = MAZE_COLS;
    header.num_ghosts = 2;
    header.power_duration = power_duration;
    header.pellet_pos = pellet_pos;
    memset(header.reserved, 0, sizeof(header.reserved));
    
    file.write(reinterpret_cast<char*>(&header), sizeof(header));
    
    // Write maze data
    file.write(reinterpret_cast<char*>(maze), sizeof(maze));
    
    // Write value data (interleaved safety and TTR)
    file.write(reinterpret_cast<char*>(safety_value), sizeof(safety_value));
    file.write(reinterpret_cast<char*>(ttr_value), sizeof(ttr_value));
    
    file.close();
    return true;
}

void print_statistics() {
    int safe_count = 0;
    int unsafe_count = 0;
    int total_valid = 0;
    
    // Also check one-ghost-dead states
    int safe_one_dead = 0;
    int total_one_dead = 0;
    
    // Only count states where pellet exists (initial game state)
    for (int p = 0; p < MAZE_CELLS; p++) {
        if (!is_valid_pos(p)) continue;
        for (int g1 = 0; g1 < MAZE_CELLS; g1++) {
            if (!is_valid_pos(g1)) continue;
            for (int g2 = 0; g2 < MAZE_CELLS; g2++) {
                if (!is_valid_pos(g2)) continue;
                if (p == g1 || p == g2) continue;
                
                total_valid++;
                if (safety_value[p][g1][g2][PELLET_EXISTS] == 1) {
                    safe_count++;
                } else {
                    unsafe_count++;
                }
            }
        }
    }
    
    // Check one-ghost-dead states with max power
    for (int p = 0; p < MAZE_CELLS; p++) {
        if (!is_valid_pos(p)) continue;
        for (int g = 0; g < MAZE_CELLS; g++) {
            if (!is_valid_pos(g)) continue;
            if (p == g) continue;
            
            // Ghost1 dead, Ghost2 alive, powered
            total_one_dead++;
            if (safety_value[p][DEAD][g][power_duration] == 1) {
                safe_one_dead++;
            }
            // Ghost2 dead, Ghost1 alive, powered
            total_one_dead++;
            if (safety_value[p][g][DEAD][power_duration] == 1) {
                safe_one_dead++;
            }
        }
    }
    
    cout << "\nDEBUG: One ghost dead, powered states: " << safe_one_dead << "/" << total_one_dead << " safe" << endl;
    
    cout << "\n=== Statistics (pellet exists, both ghosts alive) ===" << endl;
    cout << "Total valid states: " << total_valid << endl;
    cout << "Safe (Pacman can win): " << safe_count 
         << " (" << fixed << setprecision(2) << (100.0 * safe_count / total_valid) << "%)" << endl;
    cout << "Unsafe (ghosts win): " << unsafe_count 
         << " (" << fixed << setprecision(2) << (100.0 * unsafe_count / total_valid) << "%)" << endl;
}

int main(int argc, char* argv[]) {
    string output_file = "values_super_12x12.bin";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((arg == "-p" || arg == "--pellet") && i + 1 < argc) {
            pellet_pos = stoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--duration") && i + 1 < argc) {
            power_duration = stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  -o, --output FILE   Output file (default: values_super_12x12.bin)" << endl;
            cout << "  -p, --pellet POS    Super pellet position (default: 64)" << endl;
            cout << "  -d, --duration N    Power-up duration in turns (default: 10, max: 30)" << endl;
            cout << "  -h, --help          Show this help message" << endl;
            return 0;
        }
    }
    
    // Validate pellet position
    if (!is_valid_pos(pellet_pos)) {
        cerr << "Error: Invalid pellet position " << pellet_pos << endl;
        return 1;
    }
    
    // Validate power duration
    if (power_duration < 1 || power_duration > MAX_POWER_DURATION) {
        cerr << "Error: Power duration must be between 1 and " << MAX_POWER_DURATION << endl;
        return 1;
    }
    
    cout << "=== Pacman Value Iteration (Super Pellet) ===" << endl;
    cout << "Maze: " << MAZE_ROWS << "x" << MAZE_COLS << endl;
    cout << "Ghosts: 2 (with permadeath)" << endl;
    cout << "Super pellet: position " << pellet_pos 
         << " [" << row(pellet_pos) << "," << col(pellet_pos) << "]" << endl;
    cout << "Power duration: " << power_duration << " turns" << endl;
    cout << "Output: " << output_file << endl;
    cout << endl;
    
    size_t table_size = sizeof(safety_value) + sizeof(ttr_value);
    cout << "Value table size: " << (table_size / 1024 / 1024) << " MB" << endl;
    cout << endl;
    
    auto start = high_resolution_clock::now();
    run_value_iteration();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    cout << "\nTotal computation time: " << (duration.count() / 1000.0) << " s" << endl;
    
    print_statistics();
    
    cout << "\nSaving values to " << output_file << "..." << endl;
    if (!save_values(output_file)) {
        return 1;
    }
    
    cout << "Done!" << endl;
    return 0;
}

