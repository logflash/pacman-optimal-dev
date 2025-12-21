/* Value iteration code with two ghosts and Pacman, without super pellets
 * Outputs safety and TTR values to separate binary files
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
    // NOTE: Agents MUST move - they cannot stay in place
    // If somehow stuck (no neighbors), this would be an error in maze design
}

uint8_t safety_value[MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};
uint8_t ttr_value[MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};

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
    // Initial value function setup
    #pragma omp parallel for collapse(2)
    for (int p = 0; p < MAZE_CELLS; p++) {
        for (int g1 = 0; g1 < MAZE_CELLS; g1++) {
            uint8_t* srow = safety_value[p][g1];
            uint8_t* trow = ttr_value[p][g1];

            if (p == g1) {
                std::memset(srow, 0, MAZE_CELLS);
                std::memset(trow, 0, MAZE_CELLS);
            }
            else {
                std::memset(srow, 1, MAZE_CELLS);
                std::memset(trow, 0xFF, MAZE_CELLS);
                srow[p] = 0;
                trow[p] = 0;
            }
        }
    }

    const int MAX_ITERS = 1000;
    bool converged;
    auto start_time = high_resolution_clock::now();

    for(int iter=0; iter<MAX_ITERS; iter++){
        converged = true;

        #pragma omp parallel for collapse(2) schedule(dynamic) reduction(&&:converged)
        for(int p=0;p<MAZE_CELLS;p++){
            for(int g1=0;g1<MAZE_CELLS;g1++){
                uint8_t* srow = safety_value[p][g1];
                uint8_t* trow = ttr_value[p][g1];

                for(int g2=0; g2<MAZE_CELLS; g2++){
                    if(!is_free(row(p), col(p)) ||
                       !is_free(row(g1), col(g1)) ||
                       !is_free(row(g2), col(g2))) {
                        continue;
                    }

                    if(p==g1 || p==g2){
                        srow[g2] = 0;
                        trow[g2] = 0;
                        continue;
                    }

                    int pac_neighbors[4], pac_n;
                    get_neighbors(p, pac_neighbors, pac_n);

                    // Invalid state - agents must be able to move
                    if(pac_n==0) continue;

                    uint8_t best_safety = 0;
                    uint8_t best_ttr = 0;

                    for(int pi=0; pi<pac_n; pi++){
                        int np = pac_neighbors[pi];

                        int g1_neighbors[4], g1_n;
                        get_neighbors(g1, g1_neighbors, g1_n);
                        if(g1_n==0) continue; // Invalid state

                        int g2_neighbors[4], g2_n;
                        get_neighbors(g2, g2_neighbors, g2_n);
                        if(g2_n==0) continue; // Invalid state

                        uint8_t worst_safety = 1;
                        uint8_t worst_ttr = 255;

                        for(int gi1=0; gi1<g1_n; gi1++){
                            for(int gi2=0; gi2<g2_n; gi2++){
                                int ng1 = g1_neighbors[gi1];
                                int ng2 = g2_neighbors[gi2];

                                // Check for clipping/collision during movement:
                                // 1. Position swapping: Pacman and ghost trade places
                                bool clipping = (p == ng1 && g1 == np) ||
                                    (p == ng2 && g2 == np);

                                // If clipping occurs, treat as collision
                                // (safety=0, ttr=0)
                                uint8_t s, t;
                                if(clipping) {
                                    s = 0;
                                    t = 0;
                                } else {
                                    s = safety_value[np][ng1][ng2];
                                    t = ttr_value[np][ng1][ng2];
                                }

                                worst_safety = min(worst_safety, s);
                                worst_ttr = min(worst_ttr, t);
                            }
                        }

                        best_safety = max(best_safety, worst_safety);
                        uint8_t new_ttr = (worst_ttr >= 255) ? 255 : (worst_ttr + 1);
                        best_ttr = max(best_ttr, new_ttr);
                    }

                    if(best_safety != srow[g2] || best_ttr != trow[g2]){
                        converged = false;
                        srow[g2] = best_safety;
                        trow[g2] = best_ttr;
                    }
                }
            }
        }

        auto now = high_resolution_clock::now();
        double elapsed = duration_cast<milliseconds>(now - start_time).count() / 1000.0;
        print_progress(iter, MAX_ITERS, elapsed, converged);

        if(converged) {
            cout << endl;
            break;
        }
    }
}

// Binary file header structure
struct ValueFileHeader {
    char magic[4];          // "PVAL" for Pacman Value
    uint8_t version;        // File format version
    uint8_t value_type;     // 0 = safety, 1 = TTR
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
    } else {
        file.write(reinterpret_cast<char*>(ttr_value), sizeof(ttr_value));
    }

    file.close();
    return true;
}

int main(int argc, char* argv[]) {
    string safety_file = "safety_12x12_2ghosts.bin";
    string ttr_file = "ttr_12x12_2ghosts.bin";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "-s" || arg == "--safety") && i + 1 < argc) {
            safety_file = argv[++i];
        } else if ((arg == "-t" || arg == "--ttr") && i + 1 < argc) {
            ttr_file = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  -s, --safety FILE   Output file for safety values (default: safety_12x12_2ghosts.bin)" << endl;
            cout << "  -t, --ttr FILE      Output file for TTR values (default: ttr_12x12_2ghosts.bin)" << endl;
            cout << "  -h, --help          Show this help message" << endl;
            return 0;
        }
    }

    cout << "=== Pacman Value Iteration ===" << endl;
    cout << "Maze: " << MAZE_ROWS << "x" << MAZE_COLS << ", Ghosts: 2" << endl;
    cout << "Output files:" << endl;
    cout << "  Safety: " << safety_file << endl;
    cout << "  TTR:    " << ttr_file << endl;
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

    cout << "Saving TTR values to " << ttr_file << "..." << endl;
    if (!save_values(ttr_file, 1)) {
        return 1;
    }

    cout << "Done!" << endl;
    return 0;
}

