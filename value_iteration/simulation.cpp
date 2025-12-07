/* Simulation/Evaluation code for Pacman with two ghosts
 * Reads precomputed safety and TTR values from binary files
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>
using namespace std;

// Must match the maze used in value_iteration.cpp
#define MAZE_ROWS 12
#define MAZE_COLS 12
#define MAZE_CELLS (MAZE_ROWS * MAZE_COLS)

uint32_t maze[MAZE_ROWS] = {0};
uint8_t safety_value[MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};
uint8_t ttr_value[MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};

inline int row(int idx) { return idx / MAZE_COLS; }
inline int col(int idx) { return idx % MAZE_COLS; }

inline int is_free(int r, int c) {
    return (maze[r] >> c) & 1;
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

// Binary file header structure (must match value_iteration.cpp)
struct ValueFileHeader {
    char magic[4];          // "PVAL" for Pacman Value
    uint8_t version;        // File format version
    uint8_t value_type;     // 0 = safety, 1 = TTR
    uint16_t maze_rows;
    uint16_t maze_cols;
    uint8_t num_ghosts;
    uint8_t reserved[5];    // Padding for alignment
};

bool load_values(const string& filename, uint8_t expected_type) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Could not open " << filename << " for reading" << endl;
        return false;
    }

    // Read header
    ValueFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    // Validate header
    if (memcmp(header.magic, "PVAL", 4) != 0) {
        cerr << "Error: Invalid file format (bad magic)" << endl;
        return false;
    }

    if (header.version != 1) {
        cerr << "Error: Unsupported file version " << (int)header.version << endl;
        return false;
    }

    if (header.value_type != expected_type) {
        cerr << "Error: Wrong value type in file (expected " << (int)expected_type 
             << ", got " << (int)header.value_type << ")" << endl;
        return false;
    }

    // Validate maze dimensions match compiled-in size
    if (header.maze_rows != MAZE_ROWS || header.maze_cols != MAZE_COLS) {
        cerr << "Error: Maze size mismatch. File has " << header.maze_rows << "x" << header.maze_cols
             << ", but simulation compiled for " << MAZE_ROWS << "x" << MAZE_COLS << endl;
        return false;
    }

    // Read maze data
    uint32_t file_maze[MAZE_ROWS];
    file.read(reinterpret_cast<char*>(file_maze), sizeof(maze));

    if (expected_type == 0) {
        // First file - copy maze
        memcpy(maze, file_maze, sizeof(maze));
    } else {
        // Second file - verify maze matches
        if (memcmp(maze, file_maze, sizeof(maze)) != 0) {
            cerr << "Error: Maze data mismatch between files" << endl;
            return false;
        }
    }

    // Read value data
    size_t value_size = sizeof(safety_value);  // MAZE_CELLS^3
    if (expected_type == 0) {
        file.read(reinterpret_cast<char*>(safety_value), value_size);
    } else {
        file.read(reinterpret_cast<char*>(ttr_value), value_size);
    }

    if (!file) {
        cerr << "Error: Failed to read value data from " << filename << endl;
        return false;
    }

    file.close();
    return true;
}

void print_game_state(int p, int g1, int g2, int step) {
    cout << "Step " << step << ":" << endl;
    cout << "  ";
    for(int c=0; c<MAZE_COLS; c++) cout << c << " ";
    cout << endl;

    bool caught = (p == g1 || p == g2);

    for(int r=0; r<MAZE_ROWS; r++){
        cout << setw(3) << setfill(' ') << r << " ";
        for(int c=0; c<MAZE_COLS; c++){
            int idx = r * MAZE_COLS + c;
            if(caught && idx == p) {
                if(g1 == p && g2 == p) cout << "B ";
                else if(g1 == p) cout << "1 ";
                else if(g2 == p) cout << "2 ";
            }
            else if(idx == p) cout << "P ";
            else if(idx == g1 && idx == g2) cout << "B ";
            else if(idx == g1) cout << "1 ";
            else if(idx == g2) cout << "2 ";
            else if(is_free(r, c)) cout << ". ";
            else cout << "# ";
        }
        cout << endl;
    }
    cout << "Pacman=" << p << " [" << row(p) << "," << col(p) << "], "
         << "Ghost1=" << g1 << " [" << row(g1) << "," << col(g1) << "], "
         << "Ghost2=" << g2 << " [" << row(g2) << "," << col(g2) << "]";
    if(caught) {
        cout << " (CAUGHT!)";
    }
    cout << endl << endl;
}

// Extract optimal Pacman move based on maximizing worst-case value
int get_optimal_pacman_move(int p, int g1, int g2) {
    int pac_neighbors[4], pac_n;
    get_neighbors(p, pac_neighbors, pac_n);

    // Agents MUST move - they cannot stay
    if(pac_n == 0) {
        cout << "ERROR: Pacman has no valid moves!" << endl;
        return p;
    }

    int best_move = pac_neighbors[0]; // Default to first neighbor
    uint8_t best_safety = 0;
    uint8_t best_ttr = 0;

    for(int pi=0; pi<pac_n; pi++){
        int np = pac_neighbors[pi];

        // CRITICAL: Never move into a ghost's current position!
        if(np == g1 || np == g2) {
            continue; // Skip this move - it's an immediate collision
        }

        int g1_neighbors[4], g1_n;
        get_neighbors(g1, g1_neighbors, g1_n);

        int g2_neighbors[4], g2_n;
        get_neighbors(g2, g2_neighbors, g2_n);

        uint8_t worst_safety = 1;
        uint8_t worst_ttr = 255;

        for(int gi1=0; gi1<g1_n; gi1++){
            for(int gi2=0; gi2<g2_n; gi2++){
                int ng1 = g1_neighbors[gi1];
                int ng2 = g2_neighbors[gi2];
                uint8_t s = safety_value[np][ng1][ng2];
                uint8_t t = ttr_value[np][ng1][ng2];

                worst_safety = min(worst_safety, s);
                worst_ttr = min(worst_ttr, t);
            }
        }

        // Pacman prefers higher safety, then higher TTR
        if(worst_safety > best_safety ||
           (worst_safety == best_safety && worst_ttr > best_ttr)) {
            best_safety = worst_safety;
            best_ttr = worst_ttr;
            best_move = np;
        }
    }

    return best_move;
}

inline int manhattan_dist(int a, int b) {
    return abs(row(a) - row(b)) + abs(col(a) - col(b));
}

// Extract worst-case ghost moves (minimize Pacman's value)
void get_optimal_ghost_moves(int p, int g1, int g2, int& best_g1, int& best_g2) {
    int g1_neighbors[4], g1_n;
    get_neighbors(g1, g1_neighbors, g1_n);

    int g2_neighbors[4], g2_n;
    get_neighbors(g2, g2_neighbors, g2_n);

    // Agents MUST move - default to first neighbor
    best_g1 = (g1_n > 0) ? g1_neighbors[0] : g1;
    best_g2 = (g2_n > 0) ? g2_neighbors[0] : g2;
    uint8_t worst_safety = 1;
    uint8_t worst_ttr = 255;
    // Tiebreaker: each ghost's individual distance
    int best_g1_dist = 9999;
    int best_g2_dist = 9999;
    bool found_catch = false;
    int best_catch_g1_dist = 999;
    int best_catch_g2_dist = 999;

    for(int gi1=0; gi1<g1_n; gi1++){
        for(int gi2=0; gi2<g2_n; gi2++){
            int ng1 = g1_neighbors[gi1];
            int ng2 = g2_neighbors[gi2];

            bool catches = (ng1 == p || ng2 == p);

            if(catches) {
                // Among catching moves, prefer ones where both ghosts close in
                int dist1 = manhattan_dist(ng1, p);
                int dist2 = manhattan_dist(ng2, p);

                if(!found_catch || 
                   dist1 < best_catch_g1_dist ||
                   (dist1 == best_catch_g1_dist && dist2 < best_catch_g2_dist)) {
                    best_g1 = ng1;
                    best_g2 = ng2;
                    best_catch_g1_dist = dist1;
                    best_catch_g2_dist = dist2;
                    found_catch = true;
                }
                continue; // Skip value function check for catching moves
            }

            // If we've found a catch, don't consider non-catching moves
            if(found_catch) continue;

            uint8_t s = safety_value[p][ng1][ng2];
            uint8_t t = ttr_value[p][ng1][ng2];
            int g1_dist = manhattan_dist(ng1, p);
            int g2_dist = manhattan_dist(ng2, p);

            // Ghosts prefer lower safety, then lower TTR.
            // Tie-breaking: prefer moves that minimize individual ghost distances to Pacman,
            // then prefer lower position indices for g1, then g2.
            if(s < worst_safety || 
               (s == worst_safety && t < worst_ttr) ||
               (s == worst_safety && t == worst_ttr && g1_dist < best_g1_dist) ||
               (s == worst_safety && t == worst_ttr && g1_dist == best_g1_dist && g2_dist < best_g2_dist) ||
               (s == worst_safety && t == worst_ttr && g1_dist == best_g1_dist && g2_dist == best_g2_dist && ng1 < best_g1) ||
               (s == worst_safety && t == worst_ttr && g1_dist == best_g1_dist && g2_dist == best_g2_dist && ng1 == best_g1 && ng2 < best_g2)) {
                worst_safety = s;
                worst_ttr = t;
                best_g1_dist = g1_dist;
                best_g2_dist = g2_dist;
                best_g1 = ng1;
                best_g2 = ng2;
            }
        }
    }
}

void simulate_optimal_play(int start_p, int start_g1, int start_g2, int max_steps) {
    int p = start_p, g1 = start_g1, g2 = start_g2;

    cout << "========================================" << endl;
    cout << "OPTIMAL PLAY SIMULATION" << endl;
    cout << "========================================" << endl << endl;

    cout << "Initial state:" << endl;
    cout << "  Safety: " << (int)safety_value[p][g1][g2]
         << " (0=unsafe, 1=safe)" << endl;
    cout << "  TTR: " << (int)ttr_value[p][g1][g2] << " steps" << endl << endl;

    print_game_state(p, g1, g2, 0);

    for(int step=1; step<=max_steps; step++){
        if(p == g1 || p == g2) {
            cout << "Game over - Pacman caught at step " << step-1 << "!" << endl;
            break;
        }

        int new_p = get_optimal_pacman_move(p, g1, g2);
        int new_g1, new_g2;
        get_optimal_ghost_moves(new_p, g1, g2, new_g1, new_g2);

        // Check for clipping/collision during movement
        // Pacman and ghosts move simultaneously, so we need to check:
        // 1. If Pacman moves to where a ghost is moving
        // 2. If they swap positions (pass through each other)

        bool collision = false;
        string collision_reason = "";

        // Check if Pacman moves onto a ghost's new position
        if(new_p == new_g1 || new_p == new_g2) {
            collision = true;
            collision_reason = "Pacman moved to ghost position";
        }

        // Check for swap/clipping: Pacman and ghost trade positions
        if((p == new_g1 && g1 == new_p) || (p == new_g2 && g2 == new_p)) {
            collision = true;
            collision_reason = "Pacman and ghost swapped positions (clipping)";
        }

        p = new_p;
        g1 = new_g1;
        g2 = new_g2;

        print_game_state(p, g1, g2, step);

        if(collision || p == g1 || p == g2) {
            if(!collision_reason.empty()) {
                cout << "COLLISION: " << collision_reason << endl;
            }
            cout << "Game over - Pacman caught at step " << step << "!" << endl;
            break;
        }

        cout << "State value: Safety=" << (int)safety_value[p][g1][g2]
             << ", TTR=" << (int)ttr_value[p][g1][g2] << endl;
        cout << "----------------------------------------" << endl << endl;
    }
}

void print_usage(const char* program) {
    cout << "Usage: " << program << " [options]" << endl;
    cout << endl;
    cout << "Required:" << endl;
    cout << "  -s, --safety FILE   Safety value binary file" << endl;
    cout << "  -t, --ttr FILE      TTR value binary file" << endl;
    cout << endl;
    cout << "Optional:" << endl;
    cout << "  -p, --pacman POS    Initial Pacman position (default: 40)" << endl;
    cout << "  -1, --ghost1 POS    Initial Ghost1 position (default: 22)" << endl;
    cout << "  -2, --ghost2 POS    Initial Ghost2 position (default: 58)" << endl;
    cout << "  -m, --max-steps N   Maximum simulation steps (default: 255)" << endl;
    cout << "  -h, --help          Show this help message" << endl;
    cout << endl;
    cout << "Example:" << endl;
    cout << "  " << program << " -s safety_12x12_2ghosts.bin -t ttr_12x12_2ghosts.bin -p 40 -1 22 -2 58" << endl;
}

int main(int argc, char* argv[]) {
    string safety_file = "";
    string ttr_file = "";
    int start_p = 40;
    int start_g1 = 22;
    int start_g2 = 58;
    int max_steps = 255;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "-s" || arg == "--safety") && i + 1 < argc) {
            safety_file = argv[++i];
        } else if ((arg == "-t" || arg == "--ttr") && i + 1 < argc) {
            ttr_file = argv[++i];
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

    // Validate required arguments
    if (safety_file.empty() || ttr_file.empty()) {
        cerr << "Error: Both safety and TTR files are required" << endl << endl;
        print_usage(argv[0]);
        return 1;
    }

    cout << "=== Pacman Simulation ===" << endl;
    cout << "Loading value files..." << endl;

    // Load safety values first
    if (!load_values(safety_file, 0)) {
        return 1;
    }
    cout << "  Loaded safety values from " << safety_file << endl;

    // Load TTR values
    if (!load_values(ttr_file, 1)) {
        return 1;
    }
    cout << "  Loaded TTR values from " << ttr_file << endl;

    cout << "Maze: " << MAZE_ROWS << "x" << MAZE_COLS << endl;
    cout << endl;

    // Interesting test case: Pacman in center,
    // ghosts approaching from different chambers
    // Validate positions
    if (start_p < 0 || start_p >= MAZE_CELLS ||
        start_g1 < 0 || start_g1 >= MAZE_CELLS ||
        start_g2 < 0 || start_g2 >= MAZE_CELLS) {
        cerr << "Error: Invalid position (must be 0-" << MAZE_CELLS-1 << ")" << endl;
        return 1;
    }

    if (!is_free(row(start_p), col(start_p))) {
        cerr << "Error: Pacman position " << start_p << " is a wall" << endl;
        return 1;
    }
    if (!is_free(row(start_g1), col(start_g1))) {
        cerr << "Error: Ghost1 position " << start_g1 << " is a wall" << endl;
        return 1;
    }
    if (!is_free(row(start_g2), col(start_g2))) {
        cerr << "Error: Ghost2 position " << start_g2 << " is a wall" << endl;
        return 1;
    }

    simulate_optimal_play(start_p, start_g1, start_g2, max_steps);

    return 0;
}

