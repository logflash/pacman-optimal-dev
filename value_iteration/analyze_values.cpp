/* Analysis tool for Pacman value tables
 * Loads precomputed values and generates statistics/insights
 * 
 * Analyses:
 *   1. Basic stats: safe/unsafe state counts, average TTR
 *   2. Position analysis: per-cell safety rates (heatmap)
 *   3. Threshold analysis: safety filter tuning
 *   4. Ghost configuration: fix Pacman, show dangerous ghost positions
 *   5. Critical positions: choke points where safety flips easily
 *   6. Export heatmaps for visualization
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <cmath>
#include <zlib.h>
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

// Single pellet (legacy format)
int pellet_pos = 64;
int power_duration = 10;

// Multi-pellet support
#define MAX_PELLETS 4
int num_pellets = 1;
int pellet_positions[MAX_PELLETS] = {64, 0, 0, 0};
int pellet_mask_bits = 1;
int power_timer_bits = 5;
int total_power_states = 32;
bool is_multi_pellet = false;

// Optimized format flags
bool is_optimized = false;
int num_pellet_masks = 2;  // 2^num_pellets

// Value tables - dynamically allocated for multi-pellet
uint8_t* safety_value_ptr = nullptr;
uint8_t* ttr_g_value_ptr = nullptr;
uint8_t* ttr_p_value_ptr = nullptr;
size_t table_size = 0;

// Legacy static arrays for single-pellet format
uint8_t safety_value_static[MAZE_CELLS][GHOST_STATES][GHOST_STATES][MAX_POWER_STATES];
uint8_t ttr_g_value_static[MAZE_CELLS][GHOST_STATES][GHOST_STATES][MAX_POWER_STATES];
uint8_t ttr_p_value_static[MAZE_CELLS][GHOST_STATES][GHOST_STATES][MAX_POWER_STATES];

// Precomputed valid positions
vector<int> valid_positions;

inline int power_expired() { return power_duration + 1; }
inline int power_states() { return power_duration + 2; }

// Multi-pellet encoding helpers
// For non-optimized format: state = (pellet_mask << power_timer_bits) | power_timer
// For optimized format: compact encoding (see below)

inline int encode_power_state(int pellet_mask, int power_timer) {
    if (is_optimized) {
        // Compact encoding
        if (power_timer == 0) {
            return pellet_mask;
        } else {
            return num_pellet_masks + pellet_mask * power_duration + (power_timer - 1);
        }
    } else {
        return (pellet_mask << power_timer_bits) | power_timer;
    }
}

inline void decode_power_state(int state, int& pellet_mask, int& power_timer) {
    if (is_optimized) {
        // Compact encoding
        if (state < num_pellet_masks) {
            pellet_mask = state;
            power_timer = 0;
        } else {
            int powered_idx = state - num_pellet_masks;
            pellet_mask = powered_idx / power_duration;
            power_timer = (powered_idx % power_duration) + 1;
        }
    } else {
        power_timer = state & ((1 << power_timer_bits) - 1);
        pellet_mask = state >> power_timer_bits;
    }
}

inline size_t value_index(int p, int g1, int g2, int pw) {
    return ((size_t)p * GHOST_STATES * GHOST_STATES + 
            (size_t)g1 * GHOST_STATES + 
            (size_t)g2) * total_power_states + pw;
}

// Unified value access (works for both formats)
inline uint8_t get_safety(int p, int g1, int g2, int pw) {
    if (is_multi_pellet) {
        return safety_value_ptr[value_index(p, g1, g2, pw)];
    } else {
        return safety_value_static[p][g1][g2][pw];
    }
}

inline uint8_t get_ttr_g(int p, int g1, int g2, int pw) {
    if (is_multi_pellet) {
        return ttr_g_value_ptr[value_index(p, g1, g2, pw)];
    } else {
        return ttr_g_value_static[p][g1][g2][pw];
    }
}

inline uint8_t get_ttr_p(int p, int g1, int g2, int pw) {
    if (is_multi_pellet) {
        return ttr_p_value_ptr[value_index(p, g1, g2, pw)];
    } else {
        return ttr_p_value_static[p][g1][g2][pw];
    }
}

inline int row(int idx) { return idx / MAZE_COLS; }
inline int col(int idx) { return idx % MAZE_COLS; }

inline bool is_free(int r, int c) {
    return r >= 0 && r < MAZE_ROWS && c >= 0 && c < MAZE_COLS && ((maze[r] >> c) & 1);
}

inline bool is_valid_pos(int idx) {
    return idx >= 0 && idx < MAZE_CELLS && is_free(row(idx), col(idx));
}

inline int manhattan_dist(int a, int b) {
    return abs(row(a) - row(b)) + abs(col(a) - col(b));
}

inline bool is_powered(int power_state) {
    return power_state >= 1 && power_state <= power_duration;
}

// BFS to compute shortest path distance
int bfs_distance(int start, int end) {
    if (start == end) return 0;
    if (!is_valid_pos(start) || !is_valid_pos(end)) return -1;
    
    vector<int> dist(MAZE_CELLS, -1);
    queue<int> q;
    q.push(start);
    dist[start] = 0;
    
    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};
    
    while (!q.empty()) {
        int curr = q.front();
        q.pop();
        
        int r = row(curr), c = col(curr);
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (is_free(nr, nc)) {
                int next = nr * MAZE_COLS + nc;
                if (dist[next] == -1) {
                    dist[next] = dist[curr] + 1;
                    if (next == end) return dist[next];
                    q.push(next);
                }
            }
        }
    }
    return -1;  // Unreachable
}

// File header for super pellet version (single pellet)
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

// File header for multi-pellet version
struct ValueFileHeaderMulti {
    char magic[4];
    uint8_t version;
    uint8_t value_type;
    uint16_t maze_rows;
    uint16_t maze_cols;
    uint8_t num_ghosts;
    uint8_t num_pellets;
    uint8_t power_duration;
    uint8_t reserved;
    uint16_t pellet_positions[MAX_PELLETS];
};

// File header for optimized version (compact encoding + compression)
struct ValueFileHeaderOpt {
    char magic[4];              // "PVAO"
    uint8_t version;
    uint8_t value_type;
    uint16_t maze_rows;
    uint16_t maze_cols;
    uint8_t num_ghosts;
    uint8_t num_pellets;
    uint8_t power_duration;
    uint8_t compression;        // 0 = none, 1 = gzip
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
    
    // Read magic to determine format
    char magic[4];
    file.read(magic, 4);
    file.seekg(0);
    
    if (memcmp(magic, "PVAO", 4) == 0) {
        // Optimized format (compact encoding + compression)
        is_multi_pellet = true;
        is_optimized = true;
        
        ValueFileHeaderOpt header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        
        if (header.version != 1) {
            cerr << "Error: Unsupported optimized version " << (int)header.version << endl;
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
        pellet_pos = pellet_positions[0];
        
        table_size = (size_t)MAZE_CELLS * GHOST_STATES * GHOST_STATES * total_power_states;
        
        cout << "Optimized format detected:" << endl;
        cout << "  Pellets: " << num_pellets << endl;
        cout << "  Power states (compact): " << total_power_states << endl;
        cout << "  Table size: " << (table_size / 1024 / 1024) << " MB per table" << endl;
        cout << "  Compression: " << (header.compression ? "gzip" : "none") << endl;
        
        // Read maze
        file.read(reinterpret_cast<char*>(maze), sizeof(maze));
        
        // Allocate tables
        try {
            safety_value_ptr = new uint8_t[table_size];
            ttr_g_value_ptr = new uint8_t[table_size];
            ttr_p_value_ptr = new uint8_t[table_size];
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
            memcpy(safety_value_ptr, uncompressed.data(), table_size);
            memcpy(ttr_g_value_ptr, uncompressed.data() + table_size, table_size);
            memcpy(ttr_p_value_ptr, uncompressed.data() + 2 * table_size, table_size);
            
            cout << "  Decompressed " << (compressed_size / 1024) << " KB -> " 
                 << (uncompressed_size / 1024 / 1024) << " MB" << endl;
        } else {
            // Read uncompressed
            file.read(reinterpret_cast<char*>(safety_value_ptr), table_size);
            file.read(reinterpret_cast<char*>(ttr_g_value_ptr), table_size);
            file.read(reinterpret_cast<char*>(ttr_p_value_ptr), table_size);
        }
        
    } else if (memcmp(magic, "PVAM", 4) == 0) {
        // Multi-pellet format (non-optimized)
        is_multi_pellet = true;
        is_optimized = false;
        
        ValueFileHeaderMulti header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        
        if (header.version != 1) {
            cerr << "Error: Unsupported multi-pellet version " << (int)header.version << endl;
            return false;
        }
        
        if (header.maze_rows != MAZE_ROWS || header.maze_cols != MAZE_COLS) {
            cerr << "Error: Maze size mismatch" << endl;
            return false;
        }
        
        num_pellets = header.num_pellets;
        power_duration = header.power_duration;
        num_pellet_masks = 1 << num_pellets;
        for (int i = 0; i < MAX_PELLETS; i++) {
            pellet_positions[i] = header.pellet_positions[i];
        }
        pellet_pos = pellet_positions[0];
        
        // Non-optimized uses wasteful encoding
        pellet_mask_bits = num_pellets;
        power_timer_bits = 5;
        total_power_states = (1 << pellet_mask_bits) * (1 << power_timer_bits);
        table_size = (size_t)MAZE_CELLS * GHOST_STATES * GHOST_STATES * total_power_states;
        
        cout << "Multi-pellet format detected:" << endl;
        cout << "  Pellets: " << num_pellets << endl;
        cout << "  Power states: " << total_power_states << endl;
        cout << "  Table size: " << (table_size / 1024 / 1024) << " MB per table" << endl;
        
        try {
            safety_value_ptr = new uint8_t[table_size];
            ttr_g_value_ptr = new uint8_t[table_size];
            ttr_p_value_ptr = new uint8_t[table_size];
        } catch (const bad_alloc& e) {
            cerr << "Error: Failed to allocate memory" << endl;
            return false;
        }
        
        file.read(reinterpret_cast<char*>(maze), sizeof(maze));
        file.read(reinterpret_cast<char*>(safety_value_ptr), table_size);
        file.read(reinterpret_cast<char*>(ttr_g_value_ptr), table_size);
        file.read(reinterpret_cast<char*>(ttr_p_value_ptr), table_size);
        
    } else if (memcmp(magic, "PVAS", 4) == 0) {
        // Single pellet format
        is_multi_pellet = false;
        is_optimized = false;
        num_pellets = 1;
        num_pellet_masks = 2;
        
        ValueFileHeaderSuper header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        
        if (header.version != 3) {
            cerr << "Error: Unsupported version " << (int)header.version << " (expected 3)" << endl;
            return false;
        }
        
        if (header.maze_rows != MAZE_ROWS || header.maze_cols != MAZE_COLS) {
            cerr << "Error: Maze size mismatch" << endl;
            return false;
        }
        
        pellet_pos = header.pellet_pos;
        pellet_positions[0] = pellet_pos;
        power_duration = header.power_duration;
        total_power_states = power_duration + 2;
        
        file.read(reinterpret_cast<char*>(maze), sizeof(maze));
        file.read(reinterpret_cast<char*>(safety_value_static), sizeof(safety_value_static));
        file.read(reinterpret_cast<char*>(ttr_g_value_static), sizeof(ttr_g_value_static));
        file.read(reinterpret_cast<char*>(ttr_p_value_static), sizeof(ttr_p_value_static));
        
    } else {
        cerr << "Error: Unknown file format (magic: " << string(magic, 4) << ")" << endl;
        return false;
    }
    
    if (!file) {
        cerr << "Error: Failed to read value data" << endl;
        return false;
    }
    
    // Build valid positions list
    valid_positions.clear();
    for (int i = 0; i < MAZE_CELLS; i++) {
        if (is_valid_pos(i)) {
            valid_positions.push_back(i);
        }
    }
    
    file.close();
    return true;
}

void free_tables() {
    if (is_multi_pellet) {
        delete[] safety_value_ptr;
        delete[] ttr_g_value_ptr;
        delete[] ttr_p_value_ptr;
        safety_value_ptr = ttr_g_value_ptr = ttr_p_value_ptr = nullptr;
    }
}

// ============================================================================
// ANALYSIS 1: Basic Statistics
// ============================================================================
void analyze_basic_stats() {
    cout << "\n========================================" << endl;
    cout << "BASIC STATISTICS" << endl;
    cout << "========================================\n" << endl;
    
    // Determine initial power state based on format
    int initial_power_state;
    if (is_multi_pellet) {
        // All pellets exist, not powered
        int all_pellets_mask = (1 << num_pellets) - 1;
        initial_power_state = encode_power_state(all_pellets_mask, 0);
    } else {
        initial_power_state = PELLET_EXISTS;
    }
    
    long long total_states = 0;
    long long safe_states = 0;
    long long unsafe_states = 0;
    
    long long total_ttr_g = 0;
    long long total_ttr_p = 0;
    int max_ttr_g = 0;
    int min_ttr_g_safe = 255;
    
    // Both ghosts alive
    for (int p : valid_positions) {
        for (int g1 : valid_positions) {
            for (int g2 : valid_positions) {
                if (p == g1 || p == g2) continue;  // Already caught
                
                total_states++;
                uint8_t s = get_safety(p, g1, g2, initial_power_state);
                uint8_t tg = get_ttr_g(p, g1, g2, initial_power_state);
                uint8_t tp = get_ttr_p(p, g1, g2, initial_power_state);
                
                if (s == 1) {
                    safe_states++;
                    total_ttr_g += tg;
                    total_ttr_p += tp;
                    max_ttr_g = max(max_ttr_g, (int)tg);
                    if (tg < 255) min_ttr_g_safe = min(min_ttr_g_safe, (int)tg);
                } else {
                    unsafe_states++;
                }
            }
        }
    }
    
    cout << "Valid maze positions: " << valid_positions.size() << endl;
    cout << "Number of pellets: " << num_pellets << endl;
    cout << "Pellet positions: ";
    for (int i = 0; i < num_pellets; i++) {
        cout << pellet_positions[i] << " [" << row(pellet_positions[i]) << "," << col(pellet_positions[i]) << "]";
        if (i < num_pellets - 1) cout << ", ";
    }
    cout << endl;
    cout << "Power duration: " << power_duration << " turns" << endl;
    cout << endl;
    
    cout << "Initial state (all pellets exist, not powered):" << endl;
    cout << "  Total valid states: " << total_states << endl;
    cout << "  Safe states (Pacman can win): " << safe_states 
         << " (" << fixed << setprecision(2) << (100.0 * safe_states / total_states) << "%)" << endl;
    cout << "  Unsafe states (ghosts win): " << unsafe_states 
         << " (" << fixed << setprecision(2) << (100.0 * unsafe_states / total_states) << "%)" << endl;
    
    if (safe_states > 0) {
        cout << endl;
        cout << "Among safe states:" << endl;
        cout << "  Average TTR_G (survival time): " << fixed << setprecision(1) 
             << (double)total_ttr_g / safe_states << " steps" << endl;
        cout << "  Max TTR_G: " << max_ttr_g << " steps" << endl;
        cout << "  Min TTR_G: " << min_ttr_g_safe << " steps" << endl;
        cout << "  Average TTR_P (time to catch ghosts): " << fixed << setprecision(1) 
             << (double)total_ttr_p / safe_states << " steps" << endl;
    }
    
    // Multi-pellet: analyze by number of pellets remaining
    if (is_multi_pellet && num_pellets > 1) {
        cout << "\n--- Safety by pellets remaining (not powered) ---" << endl;
        
        for (int pellet_count = 0; pellet_count <= num_pellets; pellet_count++) {
            long long safe = 0, total = 0;
            
            // Enumerate all pellet masks with exactly pellet_count bits set
            for (int mask = 0; mask < (1 << num_pellets); mask++) {
                if (__builtin_popcount(mask) != pellet_count) continue;
                
                int pw = encode_power_state(mask, 0);  // Not powered
                
                for (int p : valid_positions) {
                    for (int g1 : valid_positions) {
                        for (int g2 : valid_positions) {
                            if (p == g1 || p == g2) continue;
                            
                            total++;
                            if (get_safety(p, g1, g2, pw) == 1) safe++;
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
    
    // Analyze one-ghost-dead states
    cout << "\n--- One ghost dead, Pacman powered ---" << endl;
    
    long long one_dead_total = 0;
    long long one_dead_safe = 0;
    
    // For multi-pellet, check various pellet configurations when powered
    if (is_multi_pellet) {
        for (int mask = 0; mask < (1 << num_pellets); mask++) {
            for (int timer = 1; timer <= power_duration; timer++) {
                int pw = encode_power_state(mask, timer);
                
                for (int p : valid_positions) {
                    for (int g : valid_positions) {
                        if (p == g) continue;
                        
                        one_dead_total += 2;
                        if (get_safety(p, DEAD, g, pw) == 1) one_dead_safe++;
                        if (get_safety(p, g, DEAD, pw) == 1) one_dead_safe++;
                    }
                }
            }
        }
    } else {
        for (int pw = 1; pw <= power_duration; pw++) {
            for (int p : valid_positions) {
                for (int g : valid_positions) {
                    if (p == g) continue;
                    
                    one_dead_total += 2;
                    if (get_safety(p, DEAD, g, pw) == 1) one_dead_safe++;
                    if (get_safety(p, g, DEAD, pw) == 1) one_dead_safe++;
                }
            }
        }
    }
    
    cout << "  Total states: " << one_dead_total << endl;
    cout << "  Safe states: " << one_dead_safe 
         << " (" << fixed << setprecision(2) << (100.0 * one_dead_safe / one_dead_total) << "%)" << endl;
}

// ============================================================================
// ANALYSIS 2: Ghost Configuration Heatmap
// Fix Pacman position, show which ghost configurations are dangerous
// ============================================================================
void analyze_ghost_configuration(int pacman_pos = -1) {
    cout << "\n========================================" << endl;
    cout << "GHOST CONFIGURATION ANALYSIS" << endl;
    cout << "========================================\n" << endl;
    
    // Initial power state
    int power_state;
    if (is_multi_pellet) {
        int all_pellets_mask = (1 << num_pellets) - 1;
        power_state = encode_power_state(all_pellets_mask, 0);
    } else {
        power_state = PELLET_EXISTS;
    }
    
    // If no specific position, use the first pellet position (common starting point)
    if (pacman_pos < 0 || !is_valid_pos(pacman_pos)) {
        pacman_pos = pellet_positions[0];
    }
    
    cout << "Pacman fixed at position " << pacman_pos 
         << " [" << row(pacman_pos) << "," << col(pacman_pos) << "]" << endl;
    cout << "\nFor each ghost 1 position, showing % of ghost 2 positions that are safe:" << endl;
    
    // For each g1 position, compute safety rate across all g2 positions
    map<int, double> g1_safety_rate;
    
    for (int g1 : valid_positions) {
        if (g1 == pacman_pos) continue;
        
        long long safe = 0, total = 0;
        for (int g2 : valid_positions) {
            if (g2 == pacman_pos || g2 == g1) continue;
            
            total++;
            if (get_safety(pacman_pos, g1, g2, power_state) == 1) safe++;
        }
        
        g1_safety_rate[g1] = (total > 0) ? (100.0 * safe / total) : 0;
    }
    
    // Print as maze
    cout << "\nGhost 1 position danger map (higher % = safer for Pacman):" << endl;
    cout << "    ";
    for (int c = 0; c < MAZE_COLS; c++) cout << setw(5) << c;
    cout << endl;
    
    for (int r = 0; r < MAZE_ROWS; r++) {
        cout << setw(3) << r << " ";
        for (int c = 0; c < MAZE_COLS; c++) {
            int idx = r * MAZE_COLS + c;
            if (idx == pacman_pos) {
                cout << "  [P]";
            } else if (is_valid_pos(idx)) {
                cout << setw(5) << fixed << setprecision(0) << g1_safety_rate[idx];
            } else {
                cout << "    #";
            }
        }
        cout << endl;
    }
    
    // Find most dangerous ghost positions
    vector<pair<int, double>> sorted_positions(g1_safety_rate.begin(), g1_safety_rate.end());
    sort(sorted_positions.begin(), sorted_positions.end(),
         [](auto& a, auto& b) { return a.second < b.second; });
    
    cout << "\nMost dangerous ghost 1 positions (lowest safety for Pacman):" << endl;
    int show_count = min(10, (int)sorted_positions.size());
    for (int i = 0; i < show_count; i++) {
        auto& [pos, rate] = sorted_positions[i];
        int dist = bfs_distance(pacman_pos, pos);
        cout << "  " << pos << " [" << row(pos) << "," << col(pos) << "] - "
             << fixed << setprecision(1) << rate << "% safe, dist=" << dist << endl;
    }
}

// ============================================================================
// ANALYSIS 3: Per-Position Safety Rate
// ============================================================================
void analyze_position_safety() {
    cout << "\n========================================" << endl;
    cout << "PER-POSITION SAFETY ANALYSIS" << endl;
    cout << "========================================\n" << endl;
    
    // Initial power state
    int power_state;
    if (is_multi_pellet) {
        int all_pellets_mask = (1 << num_pellets) - 1;
        power_state = encode_power_state(all_pellets_mask, 0);
    } else {
        power_state = PELLET_EXISTS;
    }
    
    vector<pair<int, double>> position_rates;  // (position, safety_rate)
    
    for (int p : valid_positions) {
        long long safe = 0, total = 0;
        
        for (int g1 : valid_positions) {
            for (int g2 : valid_positions) {
                if (p == g1 || p == g2) continue;
                
                total++;
                if (get_safety(p, g1, g2, power_state) == 1) safe++;
            }
        }
        
        double rate = (total > 0) ? (100.0 * safe / total) : 0;
        position_rates.push_back({p, rate});
    }
    
    // Sort by safety rate
    sort(position_rates.begin(), position_rates.end(), 
         [](auto& a, auto& b) { return a.second > b.second; });
    
    cout << "Positions ranked by safety rate (pellet exists, both ghosts alive):" << endl;
    cout << setw(6) << "Pos" << setw(8) << "[r,c]" << setw(12) << "Safe %" << setw(18) << "Min Pellet Dist" << endl;
    cout << string(44, '-') << endl;
    
    for (auto& [pos, rate] : position_rates) {
        // Find minimum distance to any pellet
        int min_pellet_dist = 999;
        for (int i = 0; i < num_pellets; i++) {
            int d = bfs_distance(pos, pellet_positions[i]);
            if (d >= 0 && d < min_pellet_dist) min_pellet_dist = d;
        }
        cout << setw(6) << pos 
             << "  [" << row(pos) << "," << setw(2) << col(pos) << "]"
             << setw(11) << fixed << setprecision(1) << rate << "%"
             << setw(15) << min_pellet_dist << endl;
    }
    
    // Visual maze with safety rates
    cout << "\nMaze visualization (safety rate %):" << endl;
    cout << "    ";
    for (int c = 0; c < MAZE_COLS; c++) cout << setw(5) << c;
    cout << endl;
    
    // Build lookup
    map<int, double> rate_lookup;
    for (auto& [pos, rate] : position_rates) {
        rate_lookup[pos] = rate;
    }
    
    for (int r = 0; r < MAZE_ROWS; r++) {
        cout << setw(3) << r << " ";
        for (int c = 0; c < MAZE_COLS; c++) {
            int idx = r * MAZE_COLS + c;
            if (is_valid_pos(idx)) {
                double rate = rate_lookup[idx];
                // Check if this is a pellet position
                bool is_pellet = false;
                for (int i = 0; i < num_pellets; i++) {
                    if (idx == pellet_positions[i]) {
                        is_pellet = true;
                        break;
                    }
                }
                if (is_pellet) {
                    cout << " [" << setw(2) << (int)rate << "]";
                } else {
                    cout << setw(5) << fixed << setprecision(0) << rate;
                }
            } else {
                cout << "    #";
            }
        }
        cout << endl;
    }
    cout << "\n[##] = Pellet position" << endl;
}

// ============================================================================
// ANALYSIS 4: Threshold Sensitivity (for Safety Filter)
// ============================================================================
void analyze_threshold_sensitivity() {
    cout << "\n========================================" << endl;
    cout << "THRESHOLD SENSITIVITY ANALYSIS" << endl;
    cout << "(For safety filter tuning)" << endl;
    cout << "========================================\n" << endl;
    
    // Initial power state
    int power_state;
    if (is_multi_pellet) {
        int all_pellets_mask = (1 << num_pellets) - 1;
        power_state = encode_power_state(all_pellets_mask, 0);
    } else {
        power_state = PELLET_EXISTS;
    }
    
    // For each TTR_G threshold, compute intervention statistics
    cout << "If we intervene when TTR_G <= threshold:" << endl;
    cout << setw(10) << "Threshold" 
         << setw(15) << "Interventions"
         << setw(15) << "Necessary"
         << setw(12) << "Precision"
         << setw(15) << "Missed Danger" << endl;
    cout << string(67, '-') << endl;
    
    for (int threshold = 0; threshold <= 20; threshold++) {
        long long interventions = 0;      // Would intervene
        long long necessary = 0;          // Intervened AND was actually unsafe
        long long missed = 0;             // Didn't intervene but was unsafe
        long long total = 0;
        
        for (int p : valid_positions) {
            for (int g1 : valid_positions) {
                for (int g2 : valid_positions) {
                    if (p == g1 || p == g2) continue;
                    
                    total++;
                    uint8_t s = get_safety(p, g1, g2, power_state);
                    uint8_t tg = get_ttr_g(p, g1, g2, power_state);
                    
                    bool would_intervene = (tg <= threshold);
                    bool actually_unsafe = (s == 0);
                    
                    if (would_intervene) {
                        interventions++;
                        if (actually_unsafe) necessary++;
                    } else {
                        if (actually_unsafe) missed++;
                    }
                }
            }
        }
        
        double precision = (interventions > 0) ? (100.0 * necessary / interventions) : 0;
        
        cout << setw(10) << threshold
             << setw(15) << interventions
             << setw(15) << necessary
             << setw(11) << fixed << setprecision(1) << precision << "%"
             << setw(15) << missed << endl;
    }
    
    cout << "\nInterpretation:" << endl;
    cout << "  - Interventions: States where filter would override performance policy" << endl;
    cout << "  - Necessary: Interventions that prevented actual danger" << endl;
    cout << "  - Precision: % of interventions that were truly needed" << endl;
    cout << "  - Missed Danger: Unsafe states where filter didn't intervene (should be 0 for safety)" << endl;
    cout << "\nRecommendation: Choose threshold where Missed Danger = 0 and Precision is acceptable." << endl;
}

// ============================================================================
// ANALYSIS 5: Critical Positions (Choke Points)
// Positions where safety flips based on small ghost movements
// ============================================================================
void analyze_critical_positions() {
    cout << "\n========================================" << endl;
    cout << "CRITICAL POSITIONS ANALYSIS" << endl;
    cout << "(Choke points where safety is fragile)" << endl;
    cout << "========================================\n" << endl;
    
    // Initial power state
    int power_state;
    if (is_multi_pellet) {
        int all_pellets_mask = (1 << num_pellets) - 1;
        power_state = encode_power_state(all_pellets_mask, 0);
    } else {
        power_state = PELLET_EXISTS;
    }
    
    // For each Pacman position, count how many safe moves exist on average
    // A position is "critical" if it has few escape routes
    
    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};
    
    struct PositionStats {
        int pos;
        double avg_safe_moves;      // Average number of safe moves
        double safety_variance;     // How much safety varies with ghost positions
        int total_configs;
        int configs_with_zero_safe_moves;  // Trapped configurations
    };
    
    vector<PositionStats> position_stats;
    
    for (int p : valid_positions) {
        // Find valid moves from this position
        vector<int> valid_moves;
        valid_moves.push_back(p);  // Stay in place
        
        int r = row(p), c = col(p);
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (is_free(nr, nc)) {
                valid_moves.push_back(nr * MAZE_COLS + nc);
            }
        }
        
        // For each ghost configuration, count how many moves lead to safety
        long long total_safe_moves = 0;
        long long total_configs = 0;
        long long trapped_configs = 0;
        vector<int> safe_move_counts;
        
        for (int g1 : valid_positions) {
            for (int g2 : valid_positions) {
                if (p == g1 || p == g2) continue;
                
                total_configs++;
                
                // Count safe moves from this configuration
                int safe_moves = 0;
                for (int next_p : valid_moves) {
                    // After Pacman moves, is the resulting state safe?
                    // This is a simplification - we check if current state is safe
                    if (get_safety(p, g1, g2, power_state) == 1) {
                        // Check if moving to next_p keeps us safe
                        // For now, count moves that don't collide
                        if (next_p != g1 && next_p != g2) {
                            safe_moves++;
                        }
                    }
                }
                
                total_safe_moves += safe_moves;
                safe_move_counts.push_back(safe_moves);
                
                if (safe_moves == 0) {
                    trapped_configs++;
                }
            }
        }
        
        if (total_configs == 0) continue;
        
        double avg = (double)total_safe_moves / total_configs;
        
        // Compute variance
        double variance = 0;
        for (int sm : safe_move_counts) {
            variance += (sm - avg) * (sm - avg);
        }
        variance /= total_configs;
        
        position_stats.push_back({
            p, avg, variance, (int)total_configs, (int)trapped_configs
        });
    }
    
    // Sort by average safe moves (ascending = most critical first)
    sort(position_stats.begin(), position_stats.end(),
         [](auto& a, auto& b) { return a.avg_safe_moves < b.avg_safe_moves; });
    
    cout << "Positions ranked by average safe moves (fewer = more critical):" << endl;
    cout << setw(6) << "Pos" << setw(8) << "[r,c]" << setw(12) << "Avg Safe" 
         << setw(12) << "Variance" << setw(15) << "Trapped %" << endl;
    cout << string(53, '-') << endl;
    
    for (auto& ps : position_stats) {
        double trapped_pct = 100.0 * ps.configs_with_zero_safe_moves / ps.total_configs;
        cout << setw(6) << ps.pos 
             << "  [" << row(ps.pos) << "," << setw(2) << col(ps.pos) << "]"
             << setw(11) << fixed << setprecision(2) << ps.avg_safe_moves
             << setw(12) << fixed << setprecision(2) << ps.safety_variance
             << setw(14) << fixed << setprecision(1) << trapped_pct << "%" << endl;
    }
    
    // Visual maze showing critical positions
    cout << "\nMaze visualization (average safe moves):" << endl;
    cout << "    ";
    for (int c = 0; c < MAZE_COLS; c++) cout << setw(5) << c;
    cout << endl;
    
    map<int, double> pos_to_avg;
    for (auto& ps : position_stats) {
        pos_to_avg[ps.pos] = ps.avg_safe_moves;
    }
    
    for (int r = 0; r < MAZE_ROWS; r++) {
        cout << setw(3) << r << " ";
        for (int c = 0; c < MAZE_COLS; c++) {
            int idx = r * MAZE_COLS + c;
            if (is_valid_pos(idx)) {
                // Check if pellet
                bool is_pellet = false;
                for (int i = 0; i < num_pellets; i++) {
                    if (idx == pellet_positions[i]) {
                        is_pellet = true;
                        break;
                    }
                }
                if (is_pellet) {
                    cout << " [" << setw(2) << fixed << setprecision(0) << pos_to_avg[idx] << "]";
                } else {
                    cout << setw(5) << fixed << setprecision(1) << pos_to_avg[idx];
                }
            } else {
                cout << "    #";
            }
        }
        cout << endl;
    }
    cout << "\n[##] = Pellet position" << endl;
    cout << "Lower values = more critical/dangerous positions" << endl;
}


// ============================================================================
// EXPORT: CSV for visualization
// ============================================================================
void export_heatmap_csv(const string& filename) {
    cout << "\n========================================" << endl;
    cout << "EXPORTING HEATMAP DATA" << endl;
    cout << "========================================\n" << endl;
    
    ofstream file(filename);
    if (!file) {
        cerr << "Error: Could not open " << filename << " for writing" << endl;
        return;
    }
    
    // Export per-position safety rates
    file << "position,row,col,safety_rate,avg_ttr_g,avg_ttr_p,is_pellet" << endl;
    
    // Initial power state
    int power_state;
    if (is_multi_pellet) {
        int all_pellets_mask = (1 << num_pellets) - 1;
        power_state = encode_power_state(all_pellets_mask, 0);
    } else {
        power_state = PELLET_EXISTS;
    }
    
    for (int p : valid_positions) {
        long long safe = 0, total = 0;
        long long sum_ttr_g = 0, sum_ttr_p = 0;
        
        for (int g1 : valid_positions) {
            for (int g2 : valid_positions) {
                if (p == g1 || p == g2) continue;
                
                total++;
                uint8_t s = get_safety(p, g1, g2, power_state);
                uint8_t tg = get_ttr_g(p, g1, g2, power_state);
                uint8_t tp = get_ttr_p(p, g1, g2, power_state);
                
                if (s == 1) {
                    safe++;
                    sum_ttr_g += tg;
                    sum_ttr_p += tp;
                }
            }
        }
        
        double rate = (total > 0) ? (100.0 * safe / total) : 0;
        double avg_ttr_g = (safe > 0) ? ((double)sum_ttr_g / safe) : 0;
        double avg_ttr_p = (safe > 0) ? ((double)sum_ttr_p / safe) : 0;
        
        // Check if this is a pellet position
        int is_pellet = 0;
        for (int i = 0; i < num_pellets; i++) {
            if (p == pellet_positions[i]) {
                is_pellet = 1;
                break;
            }
        }
        
        file << p << "," << row(p) << "," << col(p) << ","
             << fixed << setprecision(2) << rate << ","
             << fixed << setprecision(2) << avg_ttr_g << ","
             << fixed << setprecision(2) << avg_ttr_p << ","
             << is_pellet << endl;
    }
    
    file.close();
    cout << "Exported position heatmap to: " << filename << endl;
}

void export_ghost_config_csv(const string& filename, int pacman_pos = -1) {
    ofstream file(filename);
    if (!file) {
        cerr << "Error: Could not open " << filename << " for writing" << endl;
        return;
    }
    
    // Initial power state
    int power_state;
    if (is_multi_pellet) {
        int all_pellets_mask = (1 << num_pellets) - 1;
        power_state = encode_power_state(all_pellets_mask, 0);
    } else {
        power_state = PELLET_EXISTS;
    }
    
    // If no specific position, use the first pellet position
    if (pacman_pos < 0 || !is_valid_pos(pacman_pos)) {
        pacman_pos = pellet_positions[0];
    }
    
    file << "pacman_pos,ghost1_pos,ghost1_row,ghost1_col,safety_rate,is_pacman" << endl;
    
    // For each g1 position, compute safety rate across all g2 positions
    for (int g1 : valid_positions) {
        if (g1 == pacman_pos) {
            // Mark Pacman position
            file << pacman_pos << "," << g1 << "," << row(g1) << "," << col(g1) << ",0,1" << endl;
            continue;
        }
        
        long long safe = 0, total = 0;
        for (int g2 : valid_positions) {
            if (g2 == pacman_pos || g2 == g1) continue;
            
            total++;
            if (get_safety(pacman_pos, g1, g2, power_state) == 1) safe++;
        }
        
        double rate = (total > 0) ? (100.0 * safe / total) : 0;
        file << pacman_pos << "," << g1 << "," << row(g1) << "," << col(g1) << "," 
             << fixed << setprecision(2) << rate << ",0" << endl;
    }
    
    file.close();
    cout << "Exported ghost configuration to: " << filename << endl;
}

void export_critical_csv(const string& filename) {
    ofstream file(filename);
    if (!file) {
        cerr << "Error: Could not open " << filename << " for writing" << endl;
        return;
    }
    
    // Initial power state
    int power_state;
    if (is_multi_pellet) {
        int all_pellets_mask = (1 << num_pellets) - 1;
        power_state = encode_power_state(all_pellets_mask, 0);
    } else {
        power_state = PELLET_EXISTS;
    }
    
    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};
    
    file << "position,row,col,avg_safe_moves,variance,trapped_pct,is_pellet" << endl;
    
    for (int p : valid_positions) {
        // Find valid moves from this position
        vector<int> valid_moves;
        valid_moves.push_back(p);
        
        int r = row(p), c = col(p);
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (is_free(nr, nc)) {
                valid_moves.push_back(nr * MAZE_COLS + nc);
            }
        }
        
        long long total_safe_moves = 0;
        long long total_configs = 0;
        long long trapped_configs = 0;
        vector<int> safe_move_counts;
        
        for (int g1 : valid_positions) {
            for (int g2 : valid_positions) {
                if (p == g1 || p == g2) continue;
                
                total_configs++;
                
                int safe_moves = 0;
                for (int next_p : valid_moves) {
                    if (get_safety(p, g1, g2, power_state) == 1) {
                        if (next_p != g1 && next_p != g2) {
                            safe_moves++;
                        }
                    }
                }
                
                total_safe_moves += safe_moves;
                safe_move_counts.push_back(safe_moves);
                
                if (safe_moves == 0) {
                    trapped_configs++;
                }
            }
        }
        
        if (total_configs == 0) continue;
        
        double avg = (double)total_safe_moves / total_configs;
        
        double variance = 0;
        for (int sm : safe_move_counts) {
            variance += (sm - avg) * (sm - avg);
        }
        variance /= total_configs;
        
        double trapped_pct = 100.0 * trapped_configs / total_configs;
        
        // Check if pellet
        int is_pellet = 0;
        for (int i = 0; i < num_pellets; i++) {
            if (p == pellet_positions[i]) {
                is_pellet = 1;
                break;
            }
        }
        
        file << p << "," << row(p) << "," << col(p) << ","
             << fixed << setprecision(3) << avg << ","
             << fixed << setprecision(3) << variance << ","
             << fixed << setprecision(2) << trapped_pct << ","
             << is_pellet << endl;
    }
    
    file.close();
    cout << "Exported critical positions to: " << filename << endl;
}

void export_threshold_csv(const string& filename) {
    ofstream file(filename);
    if (!file) {
        cerr << "Error: Could not open " << filename << " for writing" << endl;
        return;
    }
    
    file << "threshold,interventions,necessary,precision,missed_danger,total" << endl;
    
    // Initial power state
    int power_state;
    if (is_multi_pellet) {
        int all_pellets_mask = (1 << num_pellets) - 1;
        power_state = encode_power_state(all_pellets_mask, 0);
    } else {
        power_state = PELLET_EXISTS;
    }
    
    for (int threshold = 0; threshold <= 50; threshold++) {
        long long interventions = 0, necessary = 0, missed = 0, total = 0;
        
        for (int p : valid_positions) {
            for (int g1 : valid_positions) {
                for (int g2 : valid_positions) {
                    if (p == g1 || p == g2) continue;
                    
                    total++;
                    uint8_t s = get_safety(p, g1, g2, power_state);
                    uint8_t tg = get_ttr_g(p, g1, g2, power_state);
                    
                    bool would_intervene = (tg <= threshold);
                    bool actually_unsafe = (s == 0);
                    
                    if (would_intervene) {
                        interventions++;
                        if (actually_unsafe) necessary++;
                    } else {
                        if (actually_unsafe) missed++;
                    }
                }
            }
        }
        
        double precision = (interventions > 0) ? (100.0 * necessary / interventions) : 0;
        file << threshold << "," << interventions << "," << necessary << ","
             << fixed << setprecision(2) << precision << "," << missed << "," << total << endl;
    }
    
    file.close();
    cout << "Exported threshold analysis to: " << filename << endl;
}

// ============================================================================
// MAIN
// ============================================================================
void print_usage(const char* program) {
    cout << "Usage: " << program << " -v VALUES_FILE [options]" << endl;
    cout << endl;
    cout << "Required:" << endl;
    cout << "  -v, --values FILE   Value table binary file (.bin)" << endl;
    cout << endl;
    cout << "Analysis options (default: run all):" << endl;
    cout << "  --basic             Basic statistics only" << endl;
    cout << "  --position          Per-position safety rates (heatmap)" << endl;
    cout << "  --threshold         Threshold sensitivity for safety filter" << endl;
    cout << "  --ghost             Ghost configuration analysis" << endl;
    cout << "  --critical          Critical positions (choke points)" << endl;
    cout << "  --all               Run all analyses (default)" << endl;
    cout << endl;
    cout << "Export options:" << endl;
    cout << "  --export-heatmap FILE    Export position data to CSV" << endl;
    cout << "  --export-ghost FILE      Export ghost config data to CSV" << endl;
    cout << "  --export-critical FILE   Export critical positions to CSV" << endl;
    cout << "  --export-threshold FILE  Export threshold data to CSV" << endl;
    cout << "  --export-all PREFIX      Export all CSVs with given prefix" << endl;
    cout << endl;
    cout << "Other:" << endl;
    cout << "  --pacman-pos POS    Fix Pacman position for ghost analysis" << endl;
    cout << "  -h, --help          Show this help message" << endl;
}

int main(int argc, char* argv[]) {
    string values_file = "";
    bool run_basic = false, run_position = false;
    bool run_threshold = false, run_ghost = false, run_critical = false;
    bool run_all = false;
    int pacman_pos = -1;
    
    string export_heatmap = "", export_ghost = "", export_critical = "", export_threshold = "";
    string export_prefix = "";
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "-v" || arg == "--values") && i + 1 < argc) {
            values_file = argv[++i];
        } else if (arg == "--basic") {
            run_basic = true;
        } else if (arg == "--position") {
            run_position = true;
        } else if (arg == "--threshold") {
            run_threshold = true;
        } else if (arg == "--ghost") {
            run_ghost = true;
        } else if (arg == "--critical") {
            run_critical = true;
        } else if (arg == "--all") {
            run_all = true;
        } else if (arg == "--pacman-pos" && i + 1 < argc) {
            pacman_pos = atoi(argv[++i]);
        } else if (arg == "--export-heatmap" && i + 1 < argc) {
            export_heatmap = argv[++i];
        } else if (arg == "--export-ghost" && i + 1 < argc) {
            export_ghost = argv[++i];
        } else if (arg == "--export-critical" && i + 1 < argc) {
            export_critical = argv[++i];
        } else if (arg == "--export-threshold" && i + 1 < argc) {
            export_threshold = argv[++i];
        } else if (arg == "--export-all" && i + 1 < argc) {
            export_prefix = argv[++i];
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
    
    // Default to all analyses if none specified
    if (!run_basic && !run_position && !run_threshold && !run_ghost && !run_critical) {
        run_all = true;
    }
    
    if (run_all) {
        run_basic = run_position = run_threshold = run_ghost = run_critical = true;
    }
    
    // Load values
    cout << "=== Pacman Value Analysis Tool ===" << endl;
    cout << "Loading values from " << values_file << "..." << endl;
    
    if (!load_values(values_file)) {
        return 1;
    }
    
    cout << "Loaded successfully." << endl;
    cout << "  Maze: " << MAZE_ROWS << "x" << MAZE_COLS << endl;
    cout << "  Valid positions: " << valid_positions.size() << endl;
    cout << "  Pellet positions: ";
    for (int i = 0; i < num_pellets; i++) {
        cout << pellet_positions[i];
        if (i < num_pellets - 1) cout << ", ";
    }
    cout << endl;
    cout << "  Power duration: " << power_duration << endl;
    
    // Run analyses
    if (run_basic) analyze_basic_stats();
    if (run_ghost) analyze_ghost_configuration(pacman_pos);
    if (run_position) analyze_position_safety();
    if (run_threshold) analyze_threshold_sensitivity();
    if (run_critical) analyze_critical_positions();
    
    // Export CSVs
    if (!export_prefix.empty()) {
        export_heatmap = export_prefix + "_heatmap.csv";
        export_ghost = export_prefix + "_ghost.csv";
        export_critical = export_prefix + "_critical.csv";
        export_threshold = export_prefix + "_threshold.csv";
    }
    
    if (!export_heatmap.empty()) export_heatmap_csv(export_heatmap);
    if (!export_ghost.empty()) export_ghost_config_csv(export_ghost, pacman_pos);
    if (!export_critical.empty()) export_critical_csv(export_critical);
    if (!export_threshold.empty()) export_threshold_csv(export_threshold);
    
    cout << "\nAnalysis complete." << endl;
    
    // Cleanup
    free_tables();
    
    return 0;
}

