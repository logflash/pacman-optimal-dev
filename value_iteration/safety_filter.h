/* Safety filter using precomputed value tables
 *
 * Loads the value tables from value iteration and provides safety checks.
 * A state is safe if Pacman can guarantee survival (and eventually win)
 * against optimal ghost play.
 *
 * Usage:
 *   #include "safety_filter.h"
 *
 *   SafetyFilter filter;
 *   if (!filter.load("values_opt.bin.gz")) return 1;
 *
 *   // Check if state is safe
 *   bool safe = filter.is_safe(pacman, g1, g2, pellet_mask, power_timer);
 *
 *   // Get time-to-reach values
 *   int ttr_ghost = filter.get_ttr_ghost(pacman, g1, g2, pellet_mask, power_timer);
 *   int ttr_pacman = filter.get_ttr_pacman(pacman, g1, g2, pellet_mask, power_timer);
 */

 #ifndef SAFETY_FILTER_H
 #define SAFETY_FILTER_H
 
 #include <cstdint>
 #include <cstring>
 #include <fstream>
 #include <vector>
 #include <iostream>
 #include <zlib.h>
 
 #define SF_MAX_PELLETS 4
 #define SF_MAX_POWER_DURATION 30
 
 class SafetyFilter {
 public:
     // Maze configuration
     int maze_rows;
     int maze_cols;
     int maze_cells;
     uint32_t maze[32];  // Max 32 rows
 
     // Power state configuration
     int num_pellets;
     int power_duration;
     int pellet_positions[SF_MAX_PELLETS];
     int num_pellet_masks;
     int total_power_states;
 
     // Value tables
     uint8_t* safety_value;
     uint8_t* ttr_g_value;
     uint8_t* ttr_p_value;
     size_t table_size;
     bool loaded;
 
     SafetyFilter() : maze_rows(0), maze_cols(0), maze_cells(0),
                      num_pellets(0), power_duration(0), num_pellet_masks(0),
                      total_power_states(0), safety_value(nullptr),
                      ttr_g_value(nullptr), ttr_p_value(nullptr),
                      table_size(0), loaded(false) {
         memset(maze, 0, sizeof(maze));
         memset(pellet_positions, 0, sizeof(pellet_positions));
     }
 
     ~SafetyFilter() {
         free_tables();
     }
 
     void free_tables() {
         delete[] safety_value;
         delete[] ttr_g_value;
         delete[] ttr_p_value;
         safety_value = ttr_g_value = ttr_p_value = nullptr;
         loaded = false;
     }
 
     // ========================================================================
     // Power State Encoding
     // ========================================================================
 
     inline int encode_power_state(int pellet_mask, int power_timer) const {
         if (power_timer == 0) {
             return pellet_mask;
         } else {
             return num_pellet_masks + pellet_mask * power_duration + (power_timer - 1);
         }
     }
 
     inline void decode_power_state(int state, int& pellet_mask, int& power_timer) const {
         if (state < num_pellet_masks) {
             pellet_mask = state;
             power_timer = 0;
         } else {
             int powered_idx = state - num_pellet_masks;
             pellet_mask = powered_idx / power_duration;
             power_timer = (powered_idx % power_duration) + 1;
         }
     }
 
     inline bool is_powered(int power_timer) const {
         return power_timer >= 1 && power_timer <= power_duration;
     }
 
     // ========================================================================
     // Value Table Indexing
     // ========================================================================
 
     inline size_t value_index(int p, int g1, int g2, int pw) const {
         int ghost_states = maze_cells + 1;
         return ((size_t)p * ghost_states * ghost_states +
                 (size_t)g1 * ghost_states +
                 (size_t)g2) * total_power_states + pw;
     }
 
     // ========================================================================
     // File Loading
     // ========================================================================
 
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
         uint16_t pellet_positions[SF_MAX_PELLETS];
         uint32_t total_power_states;
         uint32_t uncompressed_size;
     };
 
     bool load(const std::string& filename) {
         std::ifstream file(filename, std::ios::binary);
         if (!file) {
             std::cerr << "SafetyFilter: Could not open " << filename << std::endl;
             return false;
         }
 
         ValueFileHeaderOpt header;
         file.read(reinterpret_cast<char*>(&header), sizeof(header));
 
         if (memcmp(header.magic, "PVAO", 4) != 0) {
             std::cerr << "SafetyFilter: Invalid file format" << std::endl;
             return false;
         }
 
         if (header.version != 1) {
             std::cerr << "SafetyFilter: Unsupported version" << std::endl;
             return false;
         }
 
         maze_rows = header.maze_rows;
         maze_cols = header.maze_cols;
         maze_cells = maze_rows * maze_cols;
         num_pellets = header.num_pellets;
         power_duration = header.power_duration;
         num_pellet_masks = 1 << num_pellets;
         total_power_states = header.total_power_states;
 
         for (int i = 0; i < SF_MAX_PELLETS; i++) {
             pellet_positions[i] = header.pellet_positions[i];
         }
 
         int ghost_states = maze_cells + 1;
         table_size = (size_t)maze_cells * ghost_states * ghost_states * total_power_states;
 
         // Read maze
         file.read(reinterpret_cast<char*>(maze), maze_rows * sizeof(uint32_t));
 
         // Allocate tables
         try {
             safety_value = new uint8_t[table_size];
             ttr_g_value = new uint8_t[table_size];
             ttr_p_value = new uint8_t[table_size];
         } catch (const std::bad_alloc& e) {
             std::cerr << "SafetyFilter: Failed to allocate memory" << std::endl;
             return false;
         }
 
         if (header.compression == 1) {
             uint32_t compressed_size;
             file.read(reinterpret_cast<char*>(&compressed_size), sizeof(compressed_size));
 
             std::vector<uint8_t> compressed(compressed_size);
             file.read(reinterpret_cast<char*>(compressed.data()), compressed_size);
 
             uLongf uncompressed_size = header.uncompressed_size;
             std::vector<uint8_t> uncompressed(uncompressed_size);
 
             int result = uncompress(uncompressed.data(), &uncompressed_size,
                                    compressed.data(), compressed_size);
 
             if (result != Z_OK) {
                 std::cerr << "SafetyFilter: Decompression failed" << std::endl;
                 free_tables();
                 return false;
             }
 
             memcpy(safety_value, uncompressed.data(), table_size);
             memcpy(ttr_g_value, uncompressed.data() + table_size, table_size);
             memcpy(ttr_p_value, uncompressed.data() + 2 * table_size, table_size);
         } else {
             file.read(reinterpret_cast<char*>(safety_value), table_size);
             file.read(reinterpret_cast<char*>(ttr_g_value), table_size);
             file.read(reinterpret_cast<char*>(ttr_p_value), table_size);
         }
 
         if (!file) {
             std::cerr << "SafetyFilter: Failed to read data" << std::endl;
             free_tables();
             return false;
         }
 
         loaded = true;
         return true;
     }
 
     // ========================================================================
     // Safety Checks
     // ========================================================================
 
     bool is_safe(int p, int g1, int g2, int pellet_mask, int power_timer) const {
         if (!loaded) return false;
 
         int pw = encode_power_state(pellet_mask, power_timer);
         size_t idx = value_index(p, g1, g2, pw);
 
         if (idx >= table_size) return false;
         return safety_value[idx] == 1;
     }
 
     int get_ttr_ghost(int p, int g1, int g2, int pellet_mask, int power_timer) const {
         if (!loaded) return 255;
 
         int pw = encode_power_state(pellet_mask, power_timer);
         size_t idx = value_index(p, g1, g2, pw);
 
         if (idx >= table_size) return 255;
         return ttr_g_value[idx];
     }
 
     int get_ttr_pacman(int p, int g1, int g2, int pellet_mask, int power_timer) const {
         if (!loaded) return 255;
 
         int pw = encode_power_state(pellet_mask, power_timer);
         size_t idx = value_index(p, g1, g2, pw);
 
         if (idx >= table_size) return 255;
         return ttr_p_value[idx];
     }
 
     // ========================================================================
     // Utility
     // ========================================================================
 
     inline int row(int idx) const { return idx / maze_cols; }
     inline int col(int idx) const { return idx % maze_cols; }
 
     inline bool is_free(int r, int c) const {
         return r >= 0 && r < maze_rows && c >= 0 && c < maze_cols && ((maze[r] >> c) & 1);
     }
 
     inline bool is_valid_pos(int idx) const {
         return idx >= 0 && idx < maze_cells && is_free(row(idx), col(idx));
     }
 
     int get_pellet_at(int pos, int pellet_mask) const {
         for (int i = 0; i < num_pellets; i++) {
             if ((pellet_mask >> i) & 1 && pos == pellet_positions[i]) {
                 return i;
             }
         }
         return -1;
     }
 
     int all_pellets_mask() const {
         return num_pellet_masks - 1;
     }
 
     const uint32_t* get_maze() const {
         return maze;
     }
 
     int get_maze_rows() const { return maze_rows; }
     int get_maze_cols() const { return maze_cols; }
 };
 
 #endif // SAFETY_FILTER_H