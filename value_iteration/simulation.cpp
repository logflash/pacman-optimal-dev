/* Simulation code for optimal Pacman play with two ghosts
 * Loads precomputed value functions and simulates optimal gameplay
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
#define SCARED_STEPS 6
#define MAZE_CELLS (MAZE_ROWS * MAZE_COLS)

uint32_t maze[MAZE_ROWS] = {0};
uint8_t safety_value[SCARED_STEPS+1][MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};
uint8_t ttr_value_g[SCARED_STEPS+1][MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};  // Time for ghosts to reach Pacman
uint8_t ttr_value_p[SCARED_STEPS+1][MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};  // Time for Pacman to reach ghosts

inline int row(int idx) { return idx / MAZE_COLS; }
inline int col(int idx) { return idx % MAZE_COLS; }
inline int is_free(int r, int c) { return (maze[r] >> c) & 1; }

inline void get_neighbors(int idx, int* neighbors, int &count) {
    int r = row(idx), c = col(idx);
    count = 0;
    int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
    for (int k = 0; k < 4; k++) {
        int nr = r + dr[k], nc = c + dc[k];
        if (nr >= 0 && nr < MAZE_ROWS && nc >= 0 && nc < MAZE_COLS && is_free(nr, nc)) {
            neighbors[count++] = nr * MAZE_COLS + nc;
        }
    }
}

// Binary file header structure
struct ValueFileHeader {
    char magic[4];          // "PVAL" for Pacman Value
    uint8_t version;        // File format version
    uint8_t value_type;     // 0 = safety, 1 = TTR_g, 2 = TTR_p
    uint16_t maze_rows;
    uint16_t maze_cols;
    uint8_t num_ghosts;
    uint8_t reserved[5];    // Padding for alignment
};

bool load_values(const string& filename, uint8_t expected_type) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: cannot open " << filename << endl;
        return false;
    }

    ValueFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (memcmp(header.magic, "PVAL", 4) != 0 || header.version != 1 || header.value_type != expected_type) {
        cerr << "Error: Invalid value file or type mismatch\n";
        return false;
    }
    if (header.maze_rows != MAZE_ROWS || header.maze_cols != MAZE_COLS) {
        cerr << "Error: Maze size mismatch\n";
        return false;
    }

    uint32_t file_maze[MAZE_ROWS];
    file.read(reinterpret_cast<char*>(file_maze), sizeof(maze));
    if (expected_type == 0) {
        memcpy(maze, file_maze, sizeof(maze));
    } else if (memcmp(maze, file_maze, sizeof(maze)) != 0) {
        cerr << "Error: Maze mismatch\n";
        return false;
    }

    if (expected_type == 0) {
        file.read(reinterpret_cast<char*>(safety_value), sizeof(safety_value));
    } else if (expected_type == 1) {
        file.read(reinterpret_cast<char*>(ttr_value_g), sizeof(ttr_value_g));
    } else if (expected_type == 2) {
        file.read(reinterpret_cast<char*>(ttr_value_p), sizeof(ttr_value_p));
    }

    if (!file) {
        cerr << "Error: Failed reading values\n";
        return false;
    }
    file.close();
    return true;
}

void print_game_state(int p, int g1, int g2, int scared, int step) {
    cout << "Step " << step << ":\n    ";
    for (int c = 0; c < MAZE_COLS; c++) {
        cout << hex << c << " " << dec;
    }
    cout << endl;
    bool caught = (p == g1 || p == g2);
    for (int r = 0; r < MAZE_ROWS; r++) {
        cout << setw(3) << setfill(' ') << hex << r << dec << " ";
        for (int c = 0; c < MAZE_COLS; c++) {
            int idx = r * MAZE_COLS + c;
            if (idx == p) cout << "P ";
            else if (idx == g1 && idx == g2) cout << "B ";
            else if (idx == g1) cout << "1 ";
            else if (idx == g2) cout << "2 ";
            else if (is_free(r, c)) cout << ". ";
            else cout << "# ";
        }
        cout << endl;
    }
    cout << "Pacman=" << p << ", Ghost1=" << g1 << ", Ghost2=" << g2 << ", Scared=" << scared;
    if (caught) cout << " (CAUGHT!)";
    cout << endl;
}

void print_values(int p, int g1, int g2, int scared) {
    cout << "Safety: " << (int)safety_value[scared][p][g1][g2]
         << ", TTR_G: " << (int)ttr_value_g[scared][p][g1][g2]
         << ", TTR_P: " << (int)ttr_value_p[scared][p][g1][g2] << endl;
}

// Intervention function: returns safety and optimal actions for all agents
struct InterventionResult {
    uint8_t safety;
    int pacman_move1;   // First Pacman move (in scared mode) or only move (normal mode)
    int pacman_move2;   // Second Pacman move (scared mode only, -1 in normal mode)
    int ghost1_move;
    int ghost2_move;
};

InterventionResult intervention(int p, int g1, int g2, int scared);  // Forward declaration

// Optimal move functions
pair<int, int> get_optimal_pacman_move_normal(int p, int g1, int g2, int scared_time) {
    int neighbors[4], n;
    get_neighbors(p, neighbors, n);
    if (n == 0) return {p, p};

    int best = p;
    uint8_t best_s = 0, best_t = 0;

    for (int i = 0; i < n; i++) {
        int np = neighbors[i];
        if (np == g1 || np == g2) continue;

        int g1_n[4], g1_count, g2_n[4], g2_count;
        get_neighbors(g1, g1_n, g1_count);
        get_neighbors(g2, g2_n, g2_count);

        uint8_t worst_s = 1, worst_t = 255;
        for (int gi1 = 0; gi1 < g1_count; gi1++) {
            for (int gi2 = 0; gi2 < g2_count; gi2++) {
                int ng1 = g1_n[gi1], ng2 = g2_n[gi2];
                bool clip = (p == ng1 && g1 == np) || (p == ng2 && g2 == np);
                uint8_t s = clip ? 0 : safety_value[scared_time][np][ng1][ng2];
                uint8_t t = clip ? 0 : ttr_value_g[scared_time][np][ng1][ng2];
                worst_s = min(worst_s, s);
                worst_t = min(worst_t, t);
            }
        }
        if (worst_s > best_s || (worst_s == best_s && worst_t > best_t)) {
            best_s = worst_s;
            best_t = worst_t;
            best = np;
        }
    }
    return {best, best};
}

pair<int, int> get_optimal_pacman_move_scared(int p, int g1, int g2, int scared_time) {
    int p_neighbors[4], p_count;
    get_neighbors(p, p_neighbors, p_count);
    if (p_count == 0) return {p, p};

    int best_p1 = p, best_p2 = p;
    uint8_t best_safety = 0;
    uint8_t best_ttr_p = 255;  // Lower is better
    uint8_t best_ttr_g = 0;    // Higher is better

    int g1_neighbors[4], g1_count;
    int g2_neighbors[4], g2_count;
    get_neighbors(g1, g1_neighbors, g1_count);
    get_neighbors(g2, g2_neighbors, g2_count);

    for (int i1 = 0; i1 < p_count; i1++) {
        int p1 = p_neighbors[i1];

        int p2_neighbors[5], p2_count;
        get_neighbors(p1, p2_neighbors, p2_count);
        p2_neighbors[p2_count++] = p1;  // Allow wait

        for (int i2 = 0; i2 < p2_count; i2++) {
            int p2 = p2_neighbors[i2];

            uint8_t worst_safety = 1;
            uint8_t worst_ttr_p = 0;    // Max over ghosts
            uint8_t worst_ttr_g = 255;  // Min over ghosts

            for (int gi1 = 0; gi1 < g1_count; gi1++) {
                for (int gi2 = 0; gi2 < g2_count; gi2++) {
                    int ng1 = g1_neighbors[gi1];
                    int ng2 = g2_neighbors[gi2];

                    bool pacman_catches_first_move = (p1 == g1 || p1 == g2);
                    bool pacman_catches_second_move = (p2 == ng1 || p2 == ng2);
                    bool pacman_clipped = (p1 == ng1 && g1 == p2) || (p1 == ng2 && g2 == p2);

                    bool pacman_catches = pacman_catches_first_move || pacman_catches_second_move;

                    uint8_t s, tp, tg;
                    if (pacman_catches || pacman_clipped) {
                        // Ghost gets caught
                        s = (scared_time > 1) ? safety_value[scared_time - 1][p2][ng1][ng2] : 1;
                        tp = 0;  // Ghost gets caught
                        tg = (scared_time > 1) ? ttr_value_g[scared_time - 1][p2][ng1][ng2] : 255;
                    } else {
                        // Normal state transition
                        s = safety_value[scared_time - 1][p2][ng1][ng2];
                        tp = ttr_value_p[scared_time - 1][p2][ng1][ng2];
                        tg = ttr_value_g[scared_time - 1][p2][ng1][ng2];
                    }

                    worst_safety = min(worst_safety, s);
                    worst_ttr_p = max(worst_ttr_p, tp);
                    worst_ttr_g = min(worst_ttr_g, tg);
                }
            }

            bool better = false;

            // Primary objective: maximize safety
            if (worst_safety > best_safety) {
                better = true;
            }
            // Secondary objective depends on safety
            else if (worst_safety == best_safety) {
                if (worst_safety == 1) {
                    // Safe: minimize time to eat a ghost
                    if (worst_ttr_p < best_ttr_p)
                        better = true;
                } else {
                    // Unsafe: maximize time until capture
                    if (worst_ttr_g > best_ttr_g)
                        better = true;
                }
            }

            if (better) {
                best_safety = worst_safety;
                best_ttr_p = worst_ttr_p;
                best_ttr_g = worst_ttr_g;
                best_p1 = p1;
                best_p2 = p2;
            }
        }
    }

    return {best_p1, best_p2};
}


pair<int, int> get_optimal_ghost_moves_normal(int p, int g1, int g2, int scared) {
    int g1_n[4], g1_count, g2_n[4], g2_count;
    get_neighbors(g1, g1_n, g1_count);
    get_neighbors(g2, g2_n, g2_count);

    int best_g1 = (g1_count > 0) ? g1_n[0] : g1;
    int best_g2 = (g2_count > 0) ? g2_n[0] : g2;
    uint8_t worst_s = 1, worst_t = 255;

    for (int i1 = 0; i1 < g1_count; i1++) {
        for (int i2 = 0; i2 < g2_count; i2++) {
            int ng1 = g1_n[i1], ng2 = g2_n[i2];
            if (ng1 == p || ng2 == p) {
                return {ng1, ng2};
            }
            uint8_t s = safety_value[scared][p][ng1][ng2];
            uint8_t t = ttr_value_g[scared][p][ng1][ng2];
            if (s < worst_s || (s == worst_s && t < worst_t)) {
                worst_s = s;
                worst_t = t;
                best_g1 = ng1;
                best_g2 = ng2;
            }
        }
    }
    return {best_g1, best_g2};
}

pair<int, int> get_optimal_ghost_moves_scared(int p, int g1, int g2, int scared, int np1, int np2) {
    int g1_n[4], g1_count;
    int g2_n[4], g2_count;
    get_neighbors(g1, g1_n, g1_count);
    get_neighbors(g2, g2_n, g2_count);

    int best_g1 = g1, best_g2 = g2;
    uint8_t best_t = 0;
    bool found = false;

    for (int i1 = 0; i1 < g1_count; i1++) {
        for (int i2 = 0; i2 < g2_count; i2++) {
            int ng1 = g1_n[i1];
            int ng2 = g2_n[i2];

            uint8_t t;
            if (np2 == ng1 || np2 == ng2) {
                t = 0;  // Pacman catches ghost - worst for ghost
            } else {
                t = ttr_value_p[scared - 1][np2][ng1][ng2];
            }

            if (!found || t > best_t) {
                best_t = t;
                best_g1 = ng1;
                best_g2 = ng2;
                found = true;
            }
        }
    }

    // Fallback: move to first neighbor if no valid option (should not happen)
    if (!found) {
        best_g1 = g1_n[0];
        best_g2 = g2_n[0];
    }

    return {best_g1, best_g2};
}

// Intervention function implementation
InterventionResult intervention(int p, int g1, int g2, int scared) {
    InterventionResult result;
    result.safety = safety_value[scared][p][g1][g2];

    if (scared == 0) {
        // Normal mode: single-step Pacman movement
        auto pacman_move = get_optimal_pacman_move_normal(p, g1, g2, scared);
        result.pacman_move1 = pacman_move.first;
        result.pacman_move2 = -1;  // No second move in normal mode

        // Get optimal ghost moves based on Pacman's next position
        int next_p = result.pacman_move1;
        auto ghost_moves = get_optimal_ghost_moves_normal(next_p, g1, g2, scared);
        result.ghost1_move = ghost_moves.first;
        result.ghost2_move = ghost_moves.second;

    } else {
        // Scared mode: two-step Pacman movement
        auto pacman_move = get_optimal_pacman_move_scared(p, g1, g2, scared);
        result.pacman_move1 = pacman_move.first;
        result.pacman_move2 = pacman_move.second;

        // Get optimal ghost moves based on both Pacman positions
        auto ghost_moves = get_optimal_ghost_moves_scared(p, g1, g2, scared, result.pacman_move1, result.pacman_move2);
        result.ghost1_move = ghost_moves.first;
        result.ghost2_move = ghost_moves.second;
    }

    return result;
}

// Simulation
void simulate_optimal_play(int p, int g1, int g2, int scared, int max_steps) {
    // Print initial state
    print_game_state(p, g1, g2, scared, 0);
    print_values(p, g1, g2, scared);

    for (int step = 1; step <= max_steps; step++) {
        // If Pacman is already caught
        if (p == g1 || p == g2) {
            cout << "Pacman caught at step " << step - 1 << "\n";
            break;
        }

        int next_p, next_p2, next_g1, next_g2;

        if (scared == 0) {
            // Normal mode: single-step Pacman
            auto move = get_optimal_pacman_move_normal(p, g1, g2, scared);
            next_p = move.first;

            // Ghosts move based on new Pacman position
            auto ghost_moves = get_optimal_ghost_moves_normal(next_p, g1, g2, scared);
            next_g1 = ghost_moves.first;
            next_g2 = ghost_moves.second;

            // Update all positions simultaneously
            p = next_p;
            g1 = next_g1;
            g2 = next_g2;

            // Print current state
            print_game_state(p, g1, g2, scared, step);
            print_values(p, g1, g2, scared);

            if (p == g1 || p == g2) {
                cout << "Pacman caught at step " << step << "\n";
                break;
            }

        } else {
            // Scared mode: two-step Pacman movement
            auto move = get_optimal_pacman_move_scared(p, g1, g2, scared);
            next_p = move.first;
            next_p2 = move.second;

            // Determine ghost moves considering both Pacman positions
            auto ghost_moves = get_optimal_ghost_moves_scared(p, g1, g2, scared, next_p, next_p2);
            next_g1 = ghost_moves.first;
            next_g2 = ghost_moves.second;

            // Check for clips before updating positions
            int old_p = p, old_g1 = g1, old_g2 = g2;

            // Update all positions simultaneously
            p = next_p2;   // Pacman completes second step
            g1 = next_g1;
            g2 = next_g2;
            scared--;

            // Check for catches and clips
            bool caught_g1 = (p == g1);
            bool caught_g2 = (p == g2);
            bool clipped_g1 = (next_p == next_g1 && old_g1 == next_p2);
            bool clipped_g2 = (next_p == next_g2 && old_g2 == next_p2);

            if (clipped_g1) {
                cout << "Ghost1 caught by clipping at step " << step << "!\n";
                g1 = g1;  // Mark as caught
            }
            if (clipped_g2) {
                cout << "Ghost2 caught by clipping at step " << step << "!\n";
                g2 = g2;  // Mark as caught
            }

            // Print current state
            print_game_state(p, g1, g2, scared, step);
            print_values(p, g1, g2, scared);

            if (caught_g1 || caught_g2) {
                cout << "Pacman caught ghost at step " << step << "\n";
                break;
            }
            if (clipped_g1 || clipped_g2) {
                cout << "Ghost caught at step " << step << "\n";
                break;
            }
        }
    }
}

void print_usage(const char* prog) {
    cout << "Usage: " << prog << " -s SAFETY -g TTR_G -p TTR_P [options]\n";
}

int main(int argc, char* argv[]) {
    string s_file, g_file, p_file;
    int p = 40, g1 = 22, g2 = 58, scared = 0, max_steps = 255;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "-s" || arg == "--safety") && i + 1 < argc) {
            s_file = argv[++i];
        } else if ((arg == "-g" || arg == "--ttr-g") && i + 1 < argc) {
            g_file = argv[++i];
        } else if ((arg == "-p" || arg == "--ttr-p") && i + 1 < argc) {
            p_file = argv[++i];
        } else if ((arg == "-0" || arg == "--pacman") && i + 1 < argc) {
            p = stoi(argv[++i]);
        } else if ((arg == "-1" || arg == "--ghost1") && i + 1 < argc) {
            g1 = stoi(argv[++i]);
        } else if ((arg == "-2" || arg == "--ghost2") && i + 1 < argc) {
            g2 = stoi(argv[++i]);
        } else if (arg == "--scared" && i + 1 < argc) {
            scared = stoi(argv[++i]);
        } else if ((arg == "-m" || arg == "--max-steps") && i + 1 < argc) {
            max_steps = stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (s_file.empty() || g_file.empty() || p_file.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    if (!load_values(s_file, 0)) return 1;
    if (!load_values(g_file, 1)) return 1;
    if (!load_values(p_file, 2)) return 1;

    simulate_optimal_play(p, g1, g2, scared, max_steps);
    return 0;
}
