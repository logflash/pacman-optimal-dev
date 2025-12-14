/* Simulation/Evaluation code for Pacman with super pellet
 * Reads precomputed values from binary file
 * Features:
 *   - 1 super pellet (no respawn)
 *   - Configurable power-up duration
 *   - Ghost permadeath when eaten
 *   - Pacman moves TWICE per turn when powered (faster than ghosts)
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>
using namespace std;

#define MAZE_ROWS 12
#define MAZE_COLS 12
#define MAZE_CELLS (MAZE_ROWS * MAZE_COLS)

#define DEAD MAZE_CELLS
#define GHOST_STATES (MAZE_CELLS + 1)

#define MAX_POWER_DURATION 30
#define PELLET_EXISTS 0
#define MAX_POWER_STATES (MAX_POWER_DURATION + 2)

uint32_t maze[MAZE_ROWS] = {0};
int pellet_pos = 64;
int power_duration = 10;  // Will be loaded from file

uint8_t safety_value[MAZE_CELLS][GHOST_STATES][GHOST_STATES][MAX_POWER_STATES];
uint8_t ttr_g_value[MAZE_CELLS][GHOST_STATES][GHOST_STATES][MAX_POWER_STATES];  // Time for ghosts to catch Pacman
uint8_t ttr_p_value[MAZE_CELLS][GHOST_STATES][GHOST_STATES][MAX_POWER_STATES];  // Time for Pacman to catch ghosts

inline int power_expired() { return power_duration + 1; }
inline int power_states() { return power_duration + 2; }

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

inline bool is_powered(int power_state) {
    return power_state >= 1 && power_state <= power_duration;
}

inline int decrement_power(int current) {
    if (current >= 2 && current <= power_duration) {
        return current - 1;
    } else if (current == 1) {
        return power_expired();
    }
    return current;
}

inline int manhattan_dist(int a, int b) {
    return abs(row(a) - row(b)) + abs(col(a) - col(b));
}

// Forward declarations
void get_optimal_ghost_moves_internal(int p, int np1, int np2, int g1, int g2, int power_state,
                                      int& best_g1, int& best_g2);

// Result structure for Pacman's optimal move (supports double-move when powered)
struct PacmanMoveResult {
    int pos1;       // First position (always used)
    int pos2;       // Second position (same as pos1 if not powered)
    int new_g1;     // Ghost 1 position after move
    int new_g2;     // Ghost 2 position after move
    int new_power;  // Power state after move
};

// Extended header for super pellet version
struct ValueFileHeaderSuper {
    char magic[4];
    uint8_t version;
    uint8_t value_type;
    uint16_t maze_rows;
    uint16_t maze_cols;
    uint8_t num_ghosts;
    uint8_t power_duration;
    uint16_t pellet_pos;
    uint8_t reserved[2];
};

bool load_values(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Could not open " << filename << endl;
        return false;
    }
    
    ValueFileHeaderSuper header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (memcmp(header.magic, "PVAS", 4) != 0) {
        cerr << "Error: Invalid file format (expected PVAS)" << endl;
        return false;
    }
    
    if (header.version != 3) {
        cerr << "Error: Unsupported version " << (int)header.version << " (expected 3)" << endl;
        return false;
    }
    
    if (header.maze_rows != MAZE_ROWS || header.maze_cols != MAZE_COLS) {
        cerr << "Error: Maze size mismatch" << endl;
        return false;
    }
    
    pellet_pos = header.pellet_pos;
    power_duration = header.power_duration;
    
    if (power_duration > MAX_POWER_DURATION) {
        cerr << "Error: Power duration " << power_duration << " exceeds max " << MAX_POWER_DURATION << endl;
        return false;
    }
    
    file.read(reinterpret_cast<char*>(maze), sizeof(maze));
    file.read(reinterpret_cast<char*>(safety_value), sizeof(safety_value));
    file.read(reinterpret_cast<char*>(ttr_g_value), sizeof(ttr_g_value));
    file.read(reinterpret_cast<char*>(ttr_p_value), sizeof(ttr_p_value));
    
    if (!file) {
        cerr << "Error: Failed to read value data" << endl;
        return false;
    }
    
    file.close();
    return true;
}

void print_game_state(int p, int g1, int g2, int power_state, int step) {
    cout << "Step " << step << ":";
    if (is_powered(power_state)) {
        cout << " [POWERED: " << power_state << " turns left]";
    } else if (power_state == PELLET_EXISTS) {
        cout << " [Pellet available]";
    } else {
        cout << " [Power expired]";
    }
    cout << endl;
    
    cout << "  ";
    for (int c = 0; c < MAZE_COLS; c++) cout << c << " ";
    cout << endl;
    
    bool g1_alive = (g1 != DEAD);
    bool g2_alive = (g2 != DEAD);
    bool pacman_caught = (!is_powered(power_state)) && 
                         ((g1_alive && p == g1) || (g2_alive && p == g2));
    bool pacman_wins = !g1_alive && !g2_alive;
    
    for (int r = 0; r < MAZE_ROWS; r++) {
        cout << setw(3) << setfill(' ') << r << " ";
        for (int c = 0; c < MAZE_COLS; c++) {
            int idx = r * MAZE_COLS + c;
            
            if (pacman_caught && idx == p) {
                // Pacman caught
                if (g1_alive && g1 == p && g2_alive && g2 == p) cout << "X ";
                else if (g1_alive && g1 == p) cout << "X ";
                else if (g2_alive && g2 == p) cout << "X ";
            } else if (idx == p) {
                if (is_powered(power_state)) cout << "@ ";  // Powered Pacman
                else cout << "P ";
            } else if (g1_alive && idx == g1 && g2_alive && idx == g2) {
                cout << "B ";  // Both ghosts
            } else if (g1_alive && idx == g1) {
                cout << "1 ";
            } else if (g2_alive && idx == g2) {
                cout << "2 ";
            } else if (power_state == PELLET_EXISTS && idx == pellet_pos) {
                cout << "O ";  // Super pellet
            } else if (is_free(r, c)) {
                cout << ". ";
            } else {
                cout << "  ";
            }
        }
        cout << endl;
    }
    
    cout << "Pacman=" << p << " [" << row(p) << "," << col(p) << "]";
    if (g1_alive) cout << ", Ghost1=" << g1 << " [" << row(g1) << "," << col(g1) << "]";
    else cout << ", Ghost1=DEAD";
    if (g2_alive) cout << ", Ghost2=" << g2 << " [" << row(g2) << "," << col(g2) << "]";
    else cout << ", Ghost2=DEAD";
    
    if (pacman_caught) cout << " (CAUGHT!)";
    if (pacman_wins) cout << " (PACMAN WINS!)";
    cout << endl << endl;
}

// Get optimal Pacman move (returns two positions when powered)
PacmanMoveResult get_optimal_pacman_move(int p, int g1, int g2, int power_state) {
    PacmanMoveResult result;
    result.pos1 = p;
    result.pos2 = p;
    result.new_g1 = g1;
    result.new_g2 = g2;
    result.new_power = power_state;
    
    int pac_neighbors[4], pac_n;
    get_neighbors(p, pac_neighbors, pac_n);
    
    if (pac_n == 0) {
        cout << "ERROR: Pacman has no valid moves!" << endl;
        return result;
    }
    
    bool g1_alive = (g1 != DEAD);
    bool g2_alive = (g2 != DEAD);
    
    // Get ghost moves (only for alive ghosts)
    int g1_neighbors[4], g1_n = 0;
    int g2_neighbors[4], g2_n = 0;
    
    if (g1_alive) get_neighbors(g1, g1_neighbors, g1_n);
    if (g2_alive) get_neighbors(g2, g2_neighbors, g2_n);
    
    int g1_moves = g1_alive ? g1_n : 1;
    int g2_moves = g2_alive ? g2_n : 1;
    
    uint8_t best_safety = 0;
    uint8_t best_ttr_g = 0;    // Max time until ghosts catch Pacman (Pacman maximizes)
    uint8_t best_ttr_p = 255;  // Min time for Pacman to catch all ghosts (Pacman minimizes)
    int best_p1 = pac_neighbors[0];
    int best_p2 = pac_neighbors[0];
    int best_pw = power_state;
    
    // Pacman chooses move to maximize value
    for (int pi = 0; pi < pac_n; pi++) {
        int np1 = pac_neighbors[pi];
        
        // Compute power state after Pacman's first move
        int pw_after_move1 = power_state;
        if (power_state == PELLET_EXISTS && np1 == pellet_pos) {
            pw_after_move1 = power_duration;  // Pacman eats pellet
        }
        bool will_be_powered = is_powered(pw_after_move1);
        
        // If powered: Pacman moves twice. Otherwise: single move.
        int pac_neighbors2[5], pac_n2;
        if (will_be_powered) {
            get_neighbors(np1, pac_neighbors2, pac_n2);
            pac_neighbors2[pac_n2++] = np1;  // Allow staying in place on move 2
        } else {
            pac_n2 = 1;
            pac_neighbors2[0] = np1;  // Only one move, np2 == np1
        }
        
        for (int pi2 = 0; pi2 < pac_n2; pi2++) {
            int np2 = pac_neighbors2[pi2];
            
            // Skip moves into ghost positions (unless powered)
            if (!will_be_powered) {
                if ((g1_alive && np1 == g1) || (g2_alive && np1 == g2)) {
                    continue;
                }
            }
            
            uint8_t worst_safety = 1;
            uint8_t worst_ttr_g = 255;  // Ghosts minimize (catch Pacman faster)
            uint8_t worst_ttr_p = 0;    // Ghosts maximize (delay being caught)
            
            // Iterate over ghost move combinations
            for (int gi1 = 0; gi1 < g1_moves; gi1++) {
                for (int gi2 = 0; gi2 < g2_moves; gi2++) {
                    int ng1 = g1_alive ? g1_neighbors[gi1] : DEAD;
                    int ng2 = g2_alive ? g2_neighbors[gi2] : DEAD;
                    
                    // Determine outcome of this move combination
                    int final_g1 = ng1;
                    int final_g2 = ng2;
                    int final_pw = pw_after_move1;
                    bool pacman_dies = false;
                    
                    if (will_be_powered) {
                        // Powered: Pacman moves twice, can eat ghosts
                        // Check if Pacman catches ghost on first move
                        bool catch1_move1 = g1_alive && (np1 == g1);
                        bool catch2_move1 = g2_alive && (np1 == g2);
                        
                        // Check if Pacman catches ghost on second move (after ghosts move)
                        bool catch1_move2 = g1_alive && (np2 == ng1);
                        bool catch2_move2 = g2_alive && (np2 == ng2);
                        
                        // Check for clipping on second move (Pacman and ghost swap)
                        bool clip1 = g1_alive && (np1 == ng1 && g1 == np2);
                        bool clip2 = g2_alive && (np1 == ng2 && g2 == np2);
                        
                        // Pacman eats ghosts
                        if (catch1_move1 || catch1_move2 || clip1) final_g1 = DEAD;
                        if (catch2_move1 || catch2_move2 || clip2) final_g2 = DEAD;
                    } else {
                        // Not powered: single move, ghosts can catch Pacman
                        // Check for clipping (position swap)
                        bool clip1 = g1_alive && (p == ng1 && g1 == np1);
                        bool clip2 = g2_alive && (p == ng2 && g2 == np1);
                        
                        // Check collisions after movement
                        bool collide1 = g1_alive && (np1 == ng1 || clip1);
                        bool collide2 = g2_alive && (np1 == ng2 || clip2);
                        
                        if (collide1 || collide2) {
                            pacman_dies = true;
                        }
                    }
                    
                    // Decrement power timer for next state
                    if (is_powered(final_pw)) {
                        final_pw = decrement_power(final_pw);
                    }
                    
                    uint8_t s, tg, tp;
                    if (pacman_dies) {
                        s = 0;
                        tg = 0;    // Ghosts caught Pacman now
                        tp = 255;  // Pacman can't catch ghosts (dead)
                    } else if (final_g1 == DEAD && final_g2 == DEAD) {
                        // Pacman wins!
                        s = 1;
                        tg = 255;  // Ghosts can never catch Pacman
                        tp = 0;    // Pacman caught all ghosts now
                    } else {
                        s = safety_value[np2][final_g1][final_g2][final_pw];
                        tg = ttr_g_value[np2][final_g1][final_g2][final_pw];
                        tp = ttr_p_value[np2][final_g1][final_g2][final_pw];
                    }
                    
                    // Ghosts minimize Pacman's values
                    worst_safety = min(worst_safety, s);
                    worst_ttr_g = min(worst_ttr_g, tg);  // Ghosts minimize survival time
                    worst_ttr_p = max(worst_ttr_p, tp);  // Ghosts maximize time until caught
                }
            }
            
            // Update best move: Pacman maximizes safety and TTR_G, minimizes TTR_P
            bool dominated_by_new = false;
            
            if (worst_safety > best_safety) {
                // New move is safer - always better
                dominated_by_new = true;
            } else if (worst_safety == best_safety) {
                // Same safety - use TTR_G (survival time) as tiebreaker
                // Pacman wants to maximize survival time
                if (worst_ttr_g > best_ttr_g) {
                    dominated_by_new = true;
                } else if (worst_ttr_g == best_ttr_g && worst_ttr_p < best_ttr_p) {
                    // Same survival, but can catch ghosts faster
                    dominated_by_new = true;
                }
            }
            
            if (dominated_by_new) {
                best_safety = worst_safety;
                best_ttr_g = worst_ttr_g;
                best_ttr_p = worst_ttr_p;
                best_p1 = np1;
                best_p2 = np2;
                best_pw = pw_after_move1;
            }
        }
    }
    
    result.pos1 = best_p1;
    result.pos2 = best_p2;
    result.new_power = best_pw;
    
    // Now compute actual ghost response to best moves
    get_optimal_ghost_moves_internal(p, best_p1, best_p2, g1, g2, best_pw, result.new_g1, result.new_g2);
    
    return result;
}

// Internal function to get ghost moves given Pacman's move(s)
// p = Pacman's CURRENT position (before move)
// np1 = Pacman's position after first move
// np2 = Pacman's position after second move (same as np1 if not powered)
// power_state = power state AFTER Pacman's move (accounts for pellet eating)
void get_optimal_ghost_moves_internal(int p, int np1, int np2, int g1, int g2, int power_state,
                                      int& best_g1, int& best_g2) {
    bool g1_alive = (g1 != DEAD);
    bool g2_alive = (g2 != DEAD);
    bool powered = is_powered(power_state);
    
    int g1_neighbors[4], g1_n = 0;
    int g2_neighbors[4], g2_n = 0;
    
    if (g1_alive) get_neighbors(g1, g1_neighbors, g1_n);
    if (g2_alive) get_neighbors(g2, g2_neighbors, g2_n);
    
    best_g1 = g1_alive ? (g1_n > 0 ? g1_neighbors[0] : g1) : DEAD;
    best_g2 = g2_alive ? (g2_n > 0 ? g2_neighbors[0] : g2) : DEAD;
    
    if (!g1_alive && !g2_alive) return;
    
    uint8_t worst_safety = 1;
    uint8_t worst_ttr_g = 255;  // Ghosts want to minimize this (catch Pacman faster)
    bool found_catch = false;
    int best_catch_distance = 999;
    
    int g1_moves = g1_alive ? g1_n : 1;
    int g2_moves = g2_alive ? g2_n : 1;
    
    for (int gi1 = 0; gi1 < g1_moves; gi1++) {
        for (int gi2 = 0; gi2 < g2_moves; gi2++) {
            int ng1 = g1_alive ? g1_neighbors[gi1] : DEAD;
            int ng2 = g2_alive ? g2_neighbors[gi2] : DEAD;
            
            int final_g1 = ng1;
            int final_g2 = ng2;
            int final_pw = power_state;
            
            if (powered) {
                // Powered: Pacman moves twice, can eat ghosts
                // Check if Pacman catches ghost on first move
                bool catch1_move1 = g1_alive && (np1 == g1);
                bool catch2_move1 = g2_alive && (np1 == g2);
                
                // Check if Pacman catches ghost on second move (after ghosts move)
                bool catch1_move2 = g1_alive && (np2 == ng1);
                bool catch2_move2 = g2_alive && (np2 == ng2);
                
                // Check for clipping on second move (Pacman and ghost swap)
                bool clip1 = g1_alive && (np1 == ng1 && g1 == np2);
                bool clip2 = g2_alive && (np1 == ng2 && g2 == np2);
                
                // Ghosts want to AVOID being eaten
                if (catch1_move1 || catch1_move2 || clip1) final_g1 = DEAD;
                if (catch2_move1 || catch2_move2 || clip2) final_g2 = DEAD;
            } else {
                // Not powered: single move (np1 == np2)
                // Check for clipping (position swap)
                bool clip1 = g1_alive && (p == ng1 && g1 == np1);
                bool clip2 = g2_alive && (p == ng2 && g2 == np1);
                
                // Check collisions after movement (includes clipping)
                bool collide1 = g1_alive && (np1 == ng1 || clip1);
                bool collide2 = g2_alive && (np1 == ng2 || clip2);
                
                // Ghosts want to CATCH Pacman
                bool catches = collide1 || collide2;
                
                if (catches) {
                    int dist1 = g1_alive ? manhattan_dist(ng1, np1) : 999;
                    int dist2 = g2_alive ? manhattan_dist(ng2, np1) : 999;
                    int total_dist = dist1 + dist2;
                    
                    if (!found_catch || total_dist < best_catch_distance) {
                        best_g1 = ng1;
                        best_g2 = ng2;
                        best_catch_distance = total_dist;
                        found_catch = true;
                    }
                    continue;
                }
                
                if (found_catch) continue;
            }
            
            if (is_powered(final_pw)) {
                final_pw = decrement_power(final_pw);
            }
            
            uint8_t s, tg;
            if (final_g1 == DEAD && final_g2 == DEAD) {
                s = 1; tg = 255;  // Pacman wins - bad for ghosts
            } else {
                s = safety_value[np2][final_g1][final_g2][final_pw];
                tg = ttr_g_value[np2][final_g1][final_g2][final_pw];
            }
            
            // Ghosts minimize Pacman's survival time
            if (s < worst_safety || (s == worst_safety && tg < worst_ttr_g)) {
                worst_safety = s;
                worst_ttr_g = tg;
                best_g1 = ng1;
                best_g2 = ng2;
            }
        }
    }
}

void simulate_optimal_play(int start_p, int start_g1, int start_g2, int max_steps) {
    int p = start_p;
    int g1 = start_g1;
    int g2 = start_g2;
    int power_state = PELLET_EXISTS;
    
    cout << "========================================" << endl;
    cout << "OPTIMAL PLAY SIMULATION (Super Pellet)" << endl;
    cout << "  Pacman moves 2x when powered!" << endl;
    cout << "========================================" << endl << endl;
    
    cout << "Super pellet at position " << pellet_pos 
         << " [" << row(pellet_pos) << "," << col(pellet_pos) << "]" << endl;
    cout << "Power duration: " << power_duration << " turns" << endl << endl;
    
    cout << "Initial state:" << endl;
    cout << "  Safety: " << (int)safety_value[p][g1][g2][power_state]
         << " (0=ghosts win, 1=Pacman survives)" << endl;
    cout << "  TTR_G: " << (int)ttr_g_value[p][g1][g2][power_state] 
         << " (survival time - higher is better)" << endl;
    cout << "  TTR_P: " << (int)ttr_p_value[p][g1][g2][power_state] 
         << " (time to catch ghosts - lower is better)" << endl << endl;
    
    print_game_state(p, g1, g2, power_state, 0);
    
    for (int step = 1; step <= max_steps; step++) {
        bool g1_alive = (g1 != DEAD);
        bool g2_alive = (g2 != DEAD);
        
        // Check win condition
        if (!g1_alive && !g2_alive) {
            cout << "PACMAN WINS! Both ghosts eliminated at step " << step-1 << "!" << endl;
            break;
        }
        
        // Check lose condition
        bool caught = (!is_powered(power_state)) && 
                      ((g1_alive && p == g1) || (g2_alive && p == g2));
        if (caught) {
            cout << "Game over - Pacman caught at step " << step-1 << "!" << endl;
            break;
        }
        
        // Get optimal moves
        PacmanMoveResult move = get_optimal_pacman_move(p, g1, g2, power_state);
        
        // Check for eating pellet
        if (power_state == PELLET_EXISTS && move.pos1 == pellet_pos) {
            cout << ">>> Pacman eats the super pellet! <<<" << endl;
            move.new_power = power_duration;
        }
        
        bool will_be_powered = is_powered(move.new_power);
        
        // Display Pacman's move(s)
        if (will_be_powered && move.pos1 != move.pos2) {
            cout << "Pacman moves: " << p << " -> " << move.pos1 << " -> " << move.pos2 
                 << " (double move!)" << endl;
        } else {
            cout << "Pacman moves: " << p << " -> " << move.pos1 << endl;
        }
        
        // Check for ghost eating on first move (before ghosts move)
        if (will_be_powered) {
            if (g1_alive && move.pos1 == g1) {
                cout << ">>> Pacman eats Ghost 1 on first move! <<<" << endl;
                move.new_g1 = DEAD;
            }
            if (g2_alive && move.pos1 == g2) {
                cout << ">>> Pacman eats Ghost 2 on first move! <<<" << endl;
                move.new_g2 = DEAD;
            }
        }
        
        // Check for ghost eating on second move (after ghosts move)
        g1_alive = (move.new_g1 != DEAD);
        g2_alive = (move.new_g2 != DEAD);
        
        if (will_be_powered) {
            // Check if Pacman catches ghost on second move (after ghosts move)
            bool catch1_move2 = g1_alive && (move.pos2 == move.new_g1);
            bool catch2_move2 = g2_alive && (move.pos2 == move.new_g2);
            
            // Check for clipping on second move
            bool clip1 = g1_alive && (move.pos1 == move.new_g1 && g1 == move.pos2);
            bool clip2 = g2_alive && (move.pos1 == move.new_g2 && g2 == move.pos2);
            
            if (catch1_move2 || clip1) {
                cout << ">>> Pacman eats Ghost 1 on second move! <<<" << endl;
                move.new_g1 = DEAD;
            }
            if (catch2_move2 || clip2) {
                cout << ">>> Pacman eats Ghost 2 on second move! <<<" << endl;
                move.new_g2 = DEAD;
            }
        }
        
        // Check for collision when not powered
        bool collision = false;
        if (!will_be_powered) {
            bool clip1 = g1_alive && (p == move.new_g1 && g1 == move.pos1);
            bool clip2 = g2_alive && (p == move.new_g2 && g2 == move.pos1);
            bool collide1 = g1_alive && (move.pos1 == move.new_g1 || clip1);
            bool collide2 = g2_alive && (move.pos1 == move.new_g2 || clip2);
            collision = collide1 || collide2;
        }
        
        // Decrement power timer
        int new_power = move.new_power;
        if (is_powered(new_power)) {
            new_power = decrement_power(new_power);
        }
        
        // Update state
        p = move.pos2;  // Final position after both moves
        g1 = move.new_g1;
        g2 = move.new_g2;
        power_state = new_power;
        
        print_game_state(p, g1, g2, power_state, step);
        
        // Check win/lose after move
        g1_alive = (g1 != DEAD);
        g2_alive = (g2 != DEAD);
        
        if (!g1_alive && !g2_alive) {
            cout << "PACMAN WINS! Both ghosts eliminated!" << endl;
            break;
        }
        
        // Check if Pacman was caught (collision when not powered)
        if (collision) {
            cout << "COLLISION: Pacman caught by ghost!" << endl;
            cout << "Game over - Pacman caught at step " << step << "!" << endl;
            break;
        }
        
        cout << "State value: Safety=" << (int)safety_value[p][g1][g2][power_state]
             << ", TTR_G=" << (int)ttr_g_value[p][g1][g2][power_state]
             << ", TTR_P=" << (int)ttr_p_value[p][g1][g2][power_state] << endl;
        cout << "----------------------------------------" << endl << endl;
    }
}

void print_usage(const char* program) {
    cout << "Usage: " << program << " [options]" << endl;
    cout << endl;
    cout << "Required:" << endl;
    cout << "  -v, --values FILE   Value table binary file" << endl;
    cout << endl;
    cout << "Optional:" << endl;
    cout << "  -p, --pacman POS    Initial Pacman position (default: 40)" << endl;
    cout << "  -1, --ghost1 POS    Initial Ghost1 position (default: 22)" << endl;
    cout << "  -2, --ghost2 POS    Initial Ghost2 position (default: 58)" << endl;
    cout << "  -m, --max-steps N   Maximum simulation steps (default: 100)" << endl;
    cout << "  -h, --help          Show this help message" << endl;
}

int main(int argc, char* argv[]) {
    string values_file = "";
    int start_p = 40;
    int start_g1 = 22;
    int start_g2 = 58;
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
    
    cout << "=== Pacman Simulation (Super Pellet) ===" << endl;
    cout << "Loading values from " << values_file << "..." << endl;
    
    if (!load_values(values_file)) {
        return 1;
    }
    
    cout << "Maze: " << MAZE_ROWS << "x" << MAZE_COLS << endl;
    cout << "Pellet position: " << pellet_pos << endl;
    cout << endl;
    
    // Validate positions
    if (!is_valid_pos(start_p)) {
        cerr << "Error: Invalid Pacman position " << start_p << endl;
        return 1;
    }
    if (!is_valid_pos(start_g1)) {
        cerr << "Error: Invalid Ghost1 position " << start_g1 << endl;
        return 1;
    }
    if (!is_valid_pos(start_g2)) {
        cerr << "Error: Invalid Ghost2 position " << start_g2 << endl;
        return 1;
    }
    
    simulate_optimal_play(start_p, start_g1, start_g2, max_steps);
    
    return 0;
}

