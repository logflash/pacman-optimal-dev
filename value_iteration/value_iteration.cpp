/* Value iteration code for optimal Pacman play with two ghosts
 * Computes safety and TTR values and outputs to separate binary files
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
#define SCARED_STEPS 6
#define MAZE_CELLS (MAZE_ROWS * MAZE_COLS)

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

inline int is_free(int row, int col) {
    return (maze[row] >> col) & 1;
}

inline void get_neighbors(int idx, int* neighbors, int &count) {
    int r = row(idx), c = col(idx);
    count = 0;
    int dr[4] = {-1,1,0,0}, dc[4]={0,0,-1,1};
    for(int k=0;k<4;k++){
        int nr = r+dr[k], nc = c+dc[k];
        if(nr>=0 && nr<MAZE_ROWS && nc>=0 && nc<MAZE_COLS && is_free(nr, nc)) {
            neighbors[count++] = nr*MAZE_COLS+nc;
        }
    }
}

uint8_t safety_value[SCARED_STEPS+1][MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};
uint8_t ttr_value_g[SCARED_STEPS+1][MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};  // Time for ghosts to reach Pacman
uint8_t ttr_value_p[SCARED_STEPS+1][MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};  // Time for Pacman to reach ghosts

// Progress bar helper
void print_progress(int iter, int max_iters, double elapsed_sec, bool converged) {
    double progress = (double)(iter + 1) / max_iters;
    int bar_width = 40;
    int filled = (int)(progress * bar_width);

    // Estimate time remaining
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
    const int MAX_ITERS = 1000;
    auto start_time = high_resolution_clock::now();

    // Solve scared_time = 0 with full value iteration
    cout << "\nProcessing scared_time = 0 (full value iteration)" << endl;

    // Initialize value function for scared_time = 0
    #pragma omp parallel for collapse(2)
    for (int p = 0; p < MAZE_CELLS; p++) {
        for (int g1 = 0; g1 < MAZE_CELLS; g1++) {
            uint8_t* srow = safety_value[0][p][g1];
            uint8_t* trow_g = ttr_value_g[0][p][g1];
            uint8_t* trow_p = ttr_value_p[0][p][g1];

            for (int g2 = 0; g2 < MAZE_CELLS; g2++) {
                if (p == g1 || p == g2) {
                    srow[g2] = 0;
                    trow_g[g2] = 0;
                    trow_p[g2] = 0;
                } else {
                    srow[g2] = 1;
                    trow_g[g2] = 255;
                    trow_p[g2] = 255;
                }
            }
        }
    }

    bool converged;
    for (int iter = 0; iter < MAX_ITERS; iter++) {
        converged = true;

        #pragma omp parallel for collapse(2) schedule(dynamic) reduction(&&:converged)
        for (int p = 0; p < MAZE_CELLS; p++) {
            for (int g1 = 0; g1 < MAZE_CELLS; g1++) {
                uint8_t* srow = safety_value[0][p][g1];
                uint8_t* trow_g = ttr_value_g[0][p][g1];

                for (int g2 = 0; g2 < MAZE_CELLS; g2++) {
                    if (!is_free(row(p), col(p)) ||
                        !is_free(row(g1), col(g1)) ||
                        !is_free(row(g2), col(g2))) {
                        continue;
                    }

                    if (p == g1 || p == g2) {
                        srow[g2] = 0;
                        trow_g[g2] = 0;
                        continue;
                    }

                    int pac_neighbors[4], pac_n;
                    get_neighbors(p, pac_neighbors, pac_n);
                    if (pac_n == 0) continue;

                    uint8_t best_safety = 0;
                    uint8_t best_ttr_g = 0;

                    for (int pi = 0; pi < pac_n; pi++) {
                        int np = pac_neighbors[pi];

                        int g1_neighbors[4], g1_n;
                        get_neighbors(g1, g1_neighbors, g1_n);
                        if (g1_n == 0) continue;

                        int g2_neighbors[4], g2_n;
                        get_neighbors(g2, g2_neighbors, g2_n);
                        if (g2_n == 0) continue;

                        uint8_t worst_safety = 1;
                        uint8_t worst_ttr_g = 255;

                        for (int gi1 = 0; gi1 < g1_n; gi1++) {
                            for (int gi2 = 0; gi2 < g2_n; gi2++) {
                                int ng1 = g1_neighbors[gi1];
                                int ng2 = g2_neighbors[gi2];

                                bool clipping = (p == ng1 && g1 == np) || (p == ng2 && g2 == np);

                                uint8_t s, t_g;
                                if (clipping) {
                                    s = 0;
                                    t_g = 0;
                                } else {
                                    s = safety_value[0][np][ng1][ng2];
                                    t_g = ttr_value_g[0][np][ng1][ng2];
                                }

                                worst_safety = min(worst_safety, s);
                                worst_ttr_g = min(worst_ttr_g, t_g);
                            }
                        }

                        best_safety = max(best_safety, worst_safety);
                        uint8_t new_ttr_g = (worst_ttr_g >= 255) ? 255 : (worst_ttr_g + 1);
                        best_ttr_g = max(best_ttr_g, new_ttr_g);
                    }

                    if (best_safety != srow[g2] || best_ttr_g != trow_g[g2]) {
                        converged = false;
                        srow[g2] = best_safety;
                        trow_g[g2] = best_ttr_g;
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

    // Scared-mode value iteration (Pacman can move twice)
    for (int scared_time = 1; scared_time <= SCARED_STEPS; scared_time++) {

        // Initialize value function for this scared_time level
        #pragma omp parallel for collapse(2)
        for (int p = 0; p < MAZE_CELLS; p++) {
            for (int g1 = 0; g1 < MAZE_CELLS; g1++) {
                uint8_t* srow = safety_value[scared_time][p][g1];
                uint8_t* trow_g = ttr_value_g[scared_time][p][g1];
                uint8_t* trow_p = ttr_value_p[scared_time][p][g1];
                for (int g2 = 0; g2 < MAZE_CELLS; g2++) {
                    if (p == g1 || p == g2) {
                        // Caught states: Pacman caught a ghost
                        srow[g2] = 1;      // Safe (ghost is caught)
                        trow_g[g2] = 255;  // Ghosts can't catch Pacman anymore
                        trow_p[g2] = 0;    // Already caught
                    } else {
                        // Non-caught states: will be computed
                        srow[g2] = 0;      // Will be computed
                        trow_g[g2] = 0;    // Will be computed
                        trow_p[g2] = 255;  // Unknown/unreachable
                    }
                }
            }
        }

        bool converged;
        const int MAX_ITERS_SCARED = 500;
        int num_changes = 0;
        int num_catches_found = 0;
        for (int iter = 0; iter < MAX_ITERS_SCARED; iter++) {
            converged = true;
            num_changes = 0;
            num_catches_found = 0;

            #pragma omp parallel for collapse(2) schedule(dynamic) reduction(&&:converged)
            for (int p = 0; p < MAZE_CELLS; p++) {
                for (int g1 = 0; g1 < MAZE_CELLS; g1++) {
                    uint8_t* srow = safety_value[scared_time][p][g1];
                    uint8_t* trow_g = ttr_value_g[scared_time][p][g1];
                    uint8_t* trow_p = ttr_value_p[scared_time][p][g1];

                    for (int g2 = 0; g2 < MAZE_CELLS; g2++) {
                        if (!is_free(row(p), col(p)) || !is_free(row(g1), col(g1)) || !is_free(row(g2), col(g2)))
                            continue;
                        if (p == g1 || p == g2) {
                            srow[g2] = 1;
                            trow_g[g2] = 255;
                            trow_p[g2] = 0;
                            continue;
                        }

                        int pac_neighbors[4], pac_n;
                        get_neighbors(p, pac_neighbors, pac_n);
                        if (pac_n == 0) continue;

                        uint8_t best_safety = 0;    // Pacman can be safe if any move works
                        uint8_t best_ttr_g = 0;     // Max over worst-case ghost time
                        uint8_t best_ttr_p = 255;   // Min steps for Pacman to reach a ghost

                        // Pacman can move twice
                        for (int pi1 = 0; pi1 < pac_n; pi1++) {
                            int np1 = pac_neighbors[pi1];
                            int pac_neighbors2[5], pac_n2;
                            get_neighbors(np1, pac_neighbors2, pac_n2);
                            pac_neighbors2[pac_n2++] = np1;  // Allow wait on move 2

                            for (int pi2 = 0; pi2 < pac_n2; pi2++) {
                                int np2 = pac_neighbors2[pi2];

                                int g1_neighbors[4], g1_n;
                                int g2_neighbors[4], g2_n;
                                get_neighbors(g1, g1_neighbors, g1_n);
                                get_neighbors(g2, g2_neighbors, g2_n);

                                uint8_t worst_safe_for_combo = 1;    // Ghosts minimize safety
                                uint8_t worst_ttr_g_for_combo = 255;
                                uint8_t worst_ttr_p_for_combo = 0;

                                for (int gi1 = 0; gi1 < g1_n; gi1++) {
                                    for (int gi2 = 0; gi2 < g2_n; gi2++) {
                                        int ng1 = g1_neighbors[gi1];
                                        int ng2 = g2_neighbors[gi2];

                                        bool pacman_catches_first_move = (np1 == g1 || np1 == g2);
                                        bool pacman_catches_second_move = (np2 == ng1 || np2 == ng2);
                                        bool pacman_clipped = (np1 == ng1 && g1 == np2) || (np1 == ng2 && g2 == np2);

                                        bool pacman_catches = pacman_catches_first_move || pacman_catches_second_move;

                                        uint8_t s, t_g, t_p;
                                        if (pacman_catches || pacman_clipped) {
                                            // Ghost gets caught
                                            s = (scared_time > 1) ? safety_value[scared_time - 1][np2][ng1][ng2] : 1;
                                            t_p = 0;  // Ghost gets caught
                                            t_g = (scared_time > 1) ? ttr_value_g[scared_time - 1][np2][ng1][ng2] : 255;
                                            if (iter == 0) num_catches_found++;
                                        } else {
                                            // Normal state transition - look up values from next state
                                            s = safety_value[scared_time - 1][np2][ng1][ng2];
                                            t_g = ttr_value_g[scared_time - 1][np2][ng1][ng2];
                                            t_p = ttr_value_p[scared_time - 1][np2][ng1][ng2];
                                        }

                                        // Ghosts minimize safety and maximize TTR values
                                        worst_safe_for_combo = min(worst_safe_for_combo, s);
                                        worst_ttr_g_for_combo = min(worst_ttr_g_for_combo, t_g);
                                        worst_ttr_p_for_combo = max(worst_ttr_p_for_combo, t_p);
                                    }
                                }

                                best_safety = max(best_safety, worst_safe_for_combo);

                                uint8_t new_ttr_g = (worst_ttr_g_for_combo >= 255) ? 255 : (worst_ttr_g_for_combo + 1);
                                best_ttr_g = max(best_ttr_g, new_ttr_g);

                                uint8_t new_ttr_p = (worst_ttr_p_for_combo >= 255) ? 255 : (worst_ttr_p_for_combo + 1);
                                best_ttr_p = min(best_ttr_p, new_ttr_p);
                            }
                        }

                        if (srow[g2] != best_safety || trow_g[g2] != best_ttr_g || trow_p[g2] != best_ttr_p) {
                            srow[g2] = best_safety;
                            trow_g[g2] = best_ttr_g;
                            trow_p[g2] = best_ttr_p;
                            converged = false;
                            num_changes++;
                        }
                    }
                }
            }

            if (converged) {
                cout << "Completed value function for scared_time=" << scared_time << endl;
                break;
            }
        }

        // Count how many states have finite TTR_P values (for debugging)
        int finite_count = 0, caught_count = 0, intermediate_count = 0, total_count = 0;
        for (int p = 0; p < MAZE_CELLS; p++) {
            for (int g1 = 0; g1 < MAZE_CELLS; g1++) {
                for (int g2 = 0; g2 < MAZE_CELLS; g2++) {
                    if (is_free(row(p), col(p)) && is_free(row(g1), col(g1)) && is_free(row(g2), col(g2))) {
                        total_count++;
                        uint8_t val = ttr_value_p[scared_time][p][g1][g2];
                        if (val < 255) {
                            finite_count++;
                            if (val == 0) caught_count++;
                            else intermediate_count++;
                        }
                    }
                }
            }
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

bool save_values(const string& filename, uint8_t value_type) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Could not open " << filename << " for writing" << endl;
        return false;
    }

    // Write header
    ValueFileHeader header;
    memcpy(header.magic, "PVAL", 4);
    header.version = 1;
    header.value_type = value_type;
    header.maze_rows = MAZE_ROWS;
    header.maze_cols = MAZE_COLS;
    header.num_ghosts = 2;
    memset(header.reserved, 0, sizeof(header.reserved));

    file.write(reinterpret_cast<char*>(&header), sizeof(header));

    // Write maze data
    file.write(reinterpret_cast<char*>(maze), sizeof(maze));

    // Write value data
    if (value_type == 0) {
        file.write(reinterpret_cast<char*>(safety_value), sizeof(safety_value));
    } else if (value_type == 1) {
        file.write(reinterpret_cast<char*>(ttr_value_g), sizeof(ttr_value_g));
    } else if (value_type == 2) {
        file.write(reinterpret_cast<char*>(ttr_value_p), sizeof(ttr_value_p));
    }

    file.close();
    return true;
}

int main(int argc, char* argv[]) {
    string safety_file = "safety_0000.bin";
    string ttr_file_g = "g_ttr_0000.bin";
    string ttr_file_p = "p_ttr_0000.bin";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "-s" || arg == "--safety") && i + 1 < argc) {
            safety_file = argv[++i];
        } else if ((arg == "-g" || arg == "--ttr-g") && i + 1 < argc) {
            ttr_file_g = argv[++i];
        } else if ((arg == "-p" || arg == "--ttr-p") && i + 1 < argc) {
            ttr_file_p = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  -s, --safety FILE   Output file for safety values (default: safety_0000.bin)" << endl;
            cout << "  -g, --ttr-g FILE    Output file for ghost TTR values (default: g_ttr_0000.bin)" << endl;
            cout << "  -p, --ttr-p FILE    Output file for Pacman TTR values (default: p_ttr_0000.bin)" << endl;
            cout << "  -h, --help          Show this help message" << endl;
            return 0;
        }
    }

    cout << "=== Pacman Value Iteration ===" << endl;
    cout << "Maze: " << MAZE_ROWS << "x" << MAZE_COLS << ", Ghosts: 2" << endl;
    cout << "Output files:" << endl;
    cout << "  Safety:    " << safety_file << endl;
    cout << "  TTR (G):   " << ttr_file_g << endl;
    cout << "  TTR (P):   " << ttr_file_p << endl;
    cout << endl;

    auto start = high_resolution_clock::now();
    run_value_iteration();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << endl;
    cout << "Total computation time: " << duration.count() << " ms" << endl;

    // Save values to files
    cout << "Saving safety values to " << safety_file << "..." << endl;
    if (!save_values(safety_file, 0)) {
        return 1;
    }

    cout << "Saving ghost TTR values to " << ttr_file_g << "..." << endl;
    if (!save_values(ttr_file_g, 1)) {
        return 1;
    }

    cout << "Saving Pacman TTR values to " << ttr_file_p << "..." << endl;
    if (!save_values(ttr_file_p, 2)) {
        return 1;
    }

    cout << "Done!" << endl;

    return 0;
}

