/* Forward Reachable Set computation for Pac-Man ghosts
 *
 * Computes the set of positions each ghost can reach within k timesteps.
 * Used as a filter for A*: reject Pacman paths that enter the union of
 * both ghosts' reachable sets.
 *
 * Usage:
 *   #include "forward_reachable.h"
 *
 *   ForwardReachableSet frs1, frs2;
 *   frs1.compute(maze, ghost1_pos, k);
 *   frs2.compute(maze, ghost2_pos, k);
 *
 *   // Check if position is dangerous at time t
 *   bool dangerous = frs1.can_reach(pos, t) || frs2.can_reach(pos, t);
 */

 #ifndef FORWARD_REACHABLE_H
 #define FORWARD_REACHABLE_H
 
 #include <cstdint>
 #include <bitset>
 #include <vector>
 
 #define FRS_MAX_CELLS 256
 
 class ForwardReachableSet {
 public:
     int horizon;
     int maze_rows;
     int maze_cols;
     int maze_cells;
     std::vector<std::bitset<FRS_MAX_CELLS>> reachable;
 
     ForwardReachableSet() : horizon(0), maze_rows(0), maze_cols(0), maze_cells(0) {}
 
     inline int row(int idx) const { return idx / maze_cols; }
     inline int col(int idx) const { return idx % maze_cols; }
 
     inline bool is_free(const uint32_t* maze, int r, int c) const {
         return r >= 0 && r < maze_rows && c >= 0 && c < maze_cols && ((maze[r] >> c) & 1);
     }
 
     inline bool is_valid_pos(const uint32_t* maze, int idx) const {
         return idx >= 0 && idx < maze_cells && is_free(maze, row(idx), col(idx));
     }
 
     inline void get_neighbors(const uint32_t* maze, int idx, int* neighbors, int& count) const {
         int r = row(idx), c = col(idx);
         count = 0;
         int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
         for (int i = 0; i < 4; i++) {
             int nr = r + dr[i], nc = c + dc[i];
             if (is_free(maze, nr, nc)) {
                 neighbors[count++] = nr * maze_cols + nc;
             }
         }
     }
 
     void compute(const uint32_t* maze, int rows, int cols, int start_pos, int k) {
         maze_rows = rows;
         maze_cols = cols;
         maze_cells = rows * cols;
         horizon = k;
 
         reachable.clear();
         reachable.resize(k + 1);
 
         if (start_pos < 0 || start_pos >= maze_cells || !is_valid_pos(maze, start_pos)) {
             return;
         }
 
         reachable[0].set(start_pos);
 
         int neighbors[4], count;
         for (int t = 0; t < k; t++) {
             for (int pos = 0; pos < maze_cells; pos++) {
                 if (reachable[t].test(pos)) {
                     get_neighbors(maze, pos, neighbors, count);
                     for (int i = 0; i < count; i++) {
                         reachable[t + 1].set(neighbors[i]);
                     }
                 }
             }
         }
     }
 
     bool can_reach(int pos, int t) const {
         if (t < 0 || t > horizon || pos < 0 || pos >= maze_cells) return false;
         return reachable[t].test(pos);
     }
 
     const std::bitset<FRS_MAX_CELLS>& at_time(int t) const {
         static std::bitset<FRS_MAX_CELLS> empty;
         if (t < 0 || t > horizon) return empty;
         return reachable[t];
     }
 
     int count_at_time(int t) const {
         if (t < 0 || t > horizon) return 0;
         return reachable[t].count();
     }
 };
 
 // Compute union of two FRS: positions reachable by either ghost at each timestep
 inline std::vector<std::bitset<FRS_MAX_CELLS>> compute_frs_union(
     const ForwardReachableSet& frs1,
     const ForwardReachableSet& frs2,
     int k
 ) {
     std::vector<std::bitset<FRS_MAX_CELLS>> danger(k + 1);
     for (int t = 0; t <= k; t++) {
         danger[t] = frs1.at_time(t) | frs2.at_time(t);
     }
     return danger;
 }
 
 // Check if position is dangerous (reachable by either ghost) at time t
 inline bool is_frs_dangerous(
     const ForwardReachableSet& frs1,
     const ForwardReachableSet& frs2,
     int pos,
     int t
 ) {
     return frs1.can_reach(pos, t) || frs2.can_reach(pos, t);
 }
 
 #endif // FORWARD_REACHABLE_H