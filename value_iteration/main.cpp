/* Value iteration code with two ghosts and Pacman, without super pellets */

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <chrono>
#include <vector>
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

void print_maze_with_agents(int p, int g1, int g2, int step) {
    cout << "Step " << step << ":" << endl;
    cout << "  ";
    for(int c=0; c<MAZE_COLS; c++) cout << c << " ";
    cout << endl;

    bool caught = (p == g1 || p == g2);

    for(int r=0; r<MAZE_ROWS; r++){
        cout << r << " ";
        for(int c=0; c<MAZE_COLS; c++){
            int idx = r * MAZE_COLS + c;
            if(caught && idx == p) {
                // Pacman caught - show which ghost(s) caught him
                if(g1 == p && g2 == p) cout << "B "; // Both ghosts
                else if(g1 == p) cout << "1 ";
                else if(g2 == p) cout << "2 ";
            }
            else if(idx == p) cout << "P ";
            else if(idx == g1 && idx == g2) cout << "B "; // Both ghosts
            else if(idx == g1) cout << "1 ";
            else if(idx == g2) cout << "2 ";
            else if(is_free(r, c)) cout << ". ";
            else cout << "# ";
        }
        cout << endl;
    }
    cout << "Pacman=" << p << " [" << row(p) << "," << col(p) << "], "
         << "Ghost1=" << g1 << " [" << row(g1) << "," << col(g1) << "], "
         << "Ghost2=" << g2 << " [" << row(g2) << "," << col(g2) << "]" << endl;
    cout << endl;
}

uint8_t safety_value[MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};
uint8_t ttr_value[MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};

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
    int final_iter = 0;

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

        final_iter = iter + 1;
        if(converged) {
            cout << "Value iteration converged at iteration " << final_iter;
            cout << endl << endl;
            break;
        }
    }
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

        // Compute worst-case over ghost moves
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
    bool found_catch = false;
    int best_catch_distance = 999; // For tie-breaking catching moves

    for(int gi1=0; gi1<g1_n; gi1++){
        for(int gi2=0; gi2<g2_n; gi2++){
            int ng1 = g1_neighbors[gi1];
            int ng2 = g2_neighbors[gi2];

            // Check for immediate catch - either ghost catches
            bool catches = (ng1 == p || ng2 == p);

            if(catches) {
                // Among catching moves, prefer ones where both ghosts close in
                int dist1 = abs(row(ng1) - row(p)) + abs(col(ng1) - col(p));
                int dist2 = abs(row(ng2) - row(p)) + abs(col(ng2) - col(p));
                int total_dist = dist1 + dist2;

                if(!found_catch || total_dist < best_catch_distance) {
                    best_g1 = ng1;
                    best_g2 = ng2;
                    best_catch_distance = total_dist;
                    found_catch = true;
                }
                continue; // Skip value function check for catching moves
            }

            // If we've found a catch, don't consider non-catching moves
            if(found_catch) continue;

            uint8_t s = safety_value[p][ng1][ng2];
            uint8_t t = ttr_value[p][ng1][ng2];

            // Ghosts prefer lower safety, then lower TTR
            if(s < worst_safety || (s == worst_safety && t < worst_ttr)) {
                worst_safety = s;
                worst_ttr = t;
                best_g1 = ng1;
                best_g2 = ng2;
            }
        }
    }
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
                // Pacman caught - show which ghost(s) caught him
                if(g1 == p && g2 == p) cout << "B "; // Both ghosts
                else if(g1 == p) cout << "1 ";
                else if(g2 == p) cout << "2 ";
            }
            else if(idx == p) cout << "P ";
            else if(idx == g1 && idx == g2) cout << "B "; // Both ghosts
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
        // Check for collision before moves
        if(p == g1 || p == g2) {
            cout << "Game over - Pacman caught at step " << step-1 << "!" << endl;
            break;
        }

        // Get optimal moves
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

        // Update positions
        p = new_p;
        g1 = new_g1;
        g2 = new_g2;

        print_game_state(p, g1, g2, step);

        // Check for collision after moves
        if(collision || p == g1 || p == g2) {
            if(!collision_reason.empty()) {
                cout << "COLLISION: " << collision_reason << endl;
            }
            cout << "Game over - Pacman caught at step " << step << "!" << endl;
            break;
        }

        // Show current value
        cout << "State value: Safety=" << (int)safety_value[p][g1][g2]
             << ", TTR=" << (int)ttr_value[p][g1][g2] << endl;
        cout << "----------------------------------------" << endl << endl;
    }
}

int main() {
    auto start = high_resolution_clock::now();
    run_value_iteration();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Computation time: " << duration.count() << " ms" << endl << endl;

    // Interesting test case: Pacman in center,
    // ghosts approaching from different chambers
    int test_p = 40;   // Center
    int test_g1 = 22;  // Left side
    int test_g2 = 58;  // Right side

    simulate_optimal_play(test_p, test_g1, test_g2, 255);

    return 0;
}