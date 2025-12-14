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
#define SCARED_STEPS 6

uint32_t maze[MAZE_ROWS] = {0};
uint8_t safety_value[SCARED_STEPS+1][MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};
uint8_t ttr_value_g[SCARED_STEPS+1][MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};
uint8_t ttr_value_p[SCARED_STEPS+1][MAZE_CELLS][MAZE_CELLS][MAZE_CELLS] = {0};

inline int row(int idx) { return idx / MAZE_COLS; }
inline int col(int idx) { return idx % MAZE_COLS; }
inline int is_free(int r, int c) { return (maze[r] >> c) & 1; }

inline void get_neighbors(int idx, int* neighbors, int &count) {
    int r = row(idx), c = col(idx);
    count = 0;
    int dr[4] = {-1,1,0,0}, dc[4]={0,0,-1,1};
    for(int k=0;k<4;k++){
        int nr = r+dr[k], nc = c+dc[k];
        if(nr>=0 && nr<MAZE_ROWS && nc>=0 && nc<MAZE_COLS && is_free(nr,nc))
            neighbors[count++] = nr*MAZE_COLS+nc;
    }
}

struct ValueFileHeader {
    char magic[4];
    uint8_t version;
    uint8_t value_type;
    uint16_t maze_rows;
    uint16_t maze_cols;
    uint8_t num_ghosts;
    uint8_t reserved[5];
};

bool load_values(const string& filename, uint8_t expected_type) {
    ifstream file(filename, ios::binary);
    if(!file){ cerr<<"Error: cannot open "<<filename<<endl; return false; }

    ValueFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if(memcmp(header.magic,"PVAL",4)!=0 || header.version!=1 || header.value_type!=expected_type){
        cerr<<"Error: Invalid value file or type mismatch\n"; return false;
    }
    if(header.maze_rows!=MAZE_ROWS || header.maze_cols!=MAZE_COLS){
        cerr<<"Error: Maze size mismatch\n"; return false;
    }

    uint32_t file_maze[MAZE_ROWS];
    file.read(reinterpret_cast<char*>(file_maze), sizeof(maze));
    if(expected_type==0) memcpy(maze,file_maze,sizeof(maze));
    else if(memcmp(maze,file_maze,sizeof(maze))!=0){ cerr<<"Error: Maze mismatch\n"; return false; }

    if(expected_type==0) file.read(reinterpret_cast<char*>(safety_value), sizeof(safety_value));
    else if(expected_type==1) file.read(reinterpret_cast<char*>(ttr_value_g), sizeof(ttr_value_g));
    else if(expected_type==2) file.read(reinterpret_cast<char*>(ttr_value_p), sizeof(ttr_value_p));

    if(!file){ cerr<<"Error: Failed reading values\n"; return false; }
    file.close(); return true;
}

void print_game_state(int p,int g1,int g2,int scared,int step){
    cout<<"Step "<<step<<":\n    ";
    for(int c=0;c<MAZE_COLS;c++) cout<<hex<<c<<" "<<dec; cout<<endl;
    bool caught=(p==g1||p==g2);
    for(int r=0;r<MAZE_ROWS;r++){
        cout<<setw(3)<<setfill(' ')<<hex<<r<<dec<<" ";
        for(int c=0;c<MAZE_COLS;c++){
            int idx=r*MAZE_COLS+c;
            if(idx==p) cout<<"P ";
            else if(idx==g1 && idx==g2) cout<<"B ";
            else if(idx==g1) cout<<"1 ";
            else if(idx==g2) cout<<"2 ";
            else if(is_free(r,c)) cout<<". ";
            else cout<<"# ";
        }
        cout<<endl;
    }
    cout<<"Pacman="<<p<<", Ghost1="<<g1<<", Ghost2="<<g2<<", Scared="<<scared;
    if(caught) cout<<" (CAUGHT!)";
    cout<<endl;
}

// --- Print TTR and Safety values for current positions ---
void print_values(int p,int g1,int g2,int scared){
    cout<<"Safety: "<<(int)safety_value[scared][p][g1][g2]
        <<", TTR_G: "<<(int)ttr_value_g[scared][p][g1][g2]
        <<", TTR_P: "<<(int)ttr_value_p[scared][p][g1][g2]<<endl;
}

// --- Intervention function: returns safety and optimal actions for all agents ---
struct InterventionResult {
    uint8_t safety;
    int pacman_move1;   // First Pacman move (in scared mode) or only move (normal mode)
    int pacman_move2;   // Second Pacman move (scared mode only, -1 in normal mode)
    int ghost1_move;
    int ghost2_move;
};

InterventionResult intervention(int p, int g1, int g2, int scared);  // Forward declaration

// --- Optimal move functions ---
pair<int,int> get_optimal_pacman_move_normal(int p,int g1,int g2,int scared_time){
    int neighbors[4], n; get_neighbors(p,neighbors,n);
    if(n==0) return {p,p};
    int best=p; uint8_t best_s=0, best_t=0;
    for(int i=0;i<n;i++){
        int np=neighbors[i];
        if(np==g1||np==g2) continue;

        int g1_n[4], g1_count, g2_n[4], g2_count;
        get_neighbors(g1,g1_n,g1_count); get_neighbors(g2,g2_n,g2_count);

        uint8_t worst_s=1, worst_t=255;
        for(int gi1=0;gi1<g1_count;gi1++){
            for(int gi2=0;gi2<g2_count;gi2++){
                int ng1=g1_n[gi1], ng2=g2_n[gi2];
                bool clip=(p==ng1&&g1==np)||(p==ng2&&g2==np);
                uint8_t s=clip?0:safety_value[scared_time][np][ng1][ng2];
                uint8_t t=clip?0:ttr_value_g[scared_time][np][ng1][ng2];
                worst_s=min(worst_s,s); worst_t=min(worst_t,t);
            }
        }
        if(worst_s>best_s || (worst_s==best_s && worst_t>best_t)){
            best_s=worst_s; best_t=worst_t; best=np;
        }
    }
    return {best,best};
}

pair<int,int> get_optimal_pacman_move_scared(int p,int g1,int g2,int scared_time){
    int neighbors1[4], n1; get_neighbors(p,neighbors1,n1);
    if(n1==0) return {p,p};
    int best1=p,best2=p; uint8_t min_max=255;
    for(int i1=0;i1<n1;i1++){
        int np1=neighbors1[i1];
        int neighbors2[5], n2; get_neighbors(np1,neighbors2,n2); neighbors2[n2++]=np1;
        for(int i2=0;i2<n2;i2++){
            int np2=neighbors2[i2];
            if(np1==g1||np1==g2||np2==g1||np2==g2) return {np1,np2};

            int g1_n[4], g1_count, g2_n[4], g2_count;
            get_neighbors(g1,g1_n,g1_count); get_neighbors(g2,g2_n,g2_count);
            uint8_t worst=0;
            for(int gi1=0;gi1<g1_count;gi1++){
                for(int gi2=0;gi2<g2_count;gi2++){
                    int ng1=g1_n[gi1], ng2=g2_n[gi2];
                    bool clip=(np1==ng1 && g1==np2)||(np1==ng2 && g2==np2);
                    uint8_t t=clip?0:ttr_value_p[scared_time-1][np2][ng1][ng2];
                    if(t>worst) worst=t;
                }
            }
            if(worst<min_max){ min_max=worst; best1=np1; best2=np2; }
            // printf("Pacman Moves: [%d %d] - t = %d\n", np1, np2, worst);
        }
    }
    return {best1,best2};
}

void get_optimal_ghost_moves_normal(int p,int g1,int g2,int scared,int &best_g1,int &best_g2){
    int g1_n[4], g1_count, g2_n[4], g2_count;
    get_neighbors(g1,g1_n,g1_count); get_neighbors(g2,g2_n,g2_count);
    best_g1=(g1_count>0)?g1_n[0]:g1; best_g2=(g2_count>0)?g2_n[0]:g2;
    uint8_t worst_s=1, worst_t=255;
    for(int i1=0;i1<g1_count;i1++){
        for(int i2=0;i2<g2_count;i2++){
            int ng1=g1_n[i1], ng2=g2_n[i2];
            if(ng1==p||ng2==p){ best_g1=ng1; best_g2=ng2; return; }
            uint8_t s=safety_value[scared][p][ng1][ng2];
            uint8_t t=ttr_value_g[scared][p][ng1][ng2];
            if(s<worst_s || (s==worst_s && t<worst_t)){
                worst_s=s; worst_t=t; best_g1=ng1; best_g2=ng2;
            }
        }
    }
}

void get_optimal_ghost_moves_scared(int p,int g1,int g2,int scared,int np1,int np2,int &best_g1,int &best_g2){
    int g1_n[4], g1_count, g2_n[4], g2_count;
    get_neighbors(g1,g1_n,g1_count); get_neighbors(g2,g2_n,g2_count);
    best_g1=(g1_count>0)?g1_n[0]:g1; best_g2=(g2_count>0)?g2_n[0]:g2;
    uint8_t best=0;
    for(int i1=0;i1<g1_count;i1++){
        for(int i2=0;i2<g2_count;i2++){
            int ng1=g1_n[i1], ng2=g2_n[i2];
            if(np1==g1||np1==g2||np2==ng1||np2==ng2) continue;
            bool clip=(np1==ng1 && g1==np2)||(np1==ng2 && g2==np2);
            // If clipping, ghost gets caught (bad for ghost, so t=0)
            uint8_t t=clip?0:ttr_value_p[scared-1][np2][ng1][ng2];
            if(t>best){ best=t; best_g1=ng1; best_g2=ng2; }
        }
    }
}

// --- Intervention function implementation ---
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
        get_optimal_ghost_moves_normal(next_p, g1, g2, scared, result.ghost1_move, result.ghost2_move);

    } else {
        // Scared mode: two-step Pacman movement
        auto pacman_move = get_optimal_pacman_move_scared(p, g1, g2, scared);
        result.pacman_move1 = pacman_move.first;
        result.pacman_move2 = pacman_move.second;

        // Get optimal ghost moves based on both Pacman positions
        get_optimal_ghost_moves_scared(p, g1, g2, scared, result.pacman_move1, result.pacman_move2,
                                       result.ghost1_move, result.ghost2_move);
    }

    return result;
}

// --- Simulation ---
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
            get_optimal_ghost_moves_normal(next_p, g1, g2, scared, next_g1, next_g2);

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
            get_optimal_ghost_moves_scared(p, g1, g2, scared, next_p, next_p2, next_g1, next_g2);

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

void print_usage(const char* prog){ cout<<"Usage: "<<prog<<" -s SAFETY -g TTR_G -p TTR_P [options]\n"; }

int main(int argc,char* argv[]){
    string s_file,g_file,p_file;
    int p=40,g1=22,g2=58,scared=0,max_steps=255;

    for(int i=1;i<argc;i++){
        string arg=argv[i];
        if((arg=="-s"||arg=="--safety") && i+1<argc) s_file=argv[++i];
        else if((arg=="-g"||arg=="--ttr-g") && i+1<argc) g_file=argv[++i];
        else if((arg=="-p"||arg=="--ttr-p") && i+1<argc) p_file=argv[++i];
        else if((arg=="-0"||arg=="--pacman") && i+1<argc) p=stoi(argv[++i]);
        else if((arg=="-1"||arg=="--ghost1") && i+1<argc) g1=stoi(argv[++i]);
        else if((arg=="-2"||arg=="--ghost2") && i+1<argc) g2=stoi(argv[++i]);
        else if(arg=="--scared" && i+1<argc) scared=stoi(argv[++i]);
        else if((arg=="-m"||arg=="--max-steps") && i+1<argc) max_steps=stoi(argv[++i]);
        else if(arg=="-h"||arg=="--help") { print_usage(argv[0]); return 0; }
    }

    if(s_file.empty()||g_file.empty()||p_file.empty()){ print_usage(argv[0]); return 1; }
    if(!load_values(s_file,0)) return 1;
    if(!load_values(g_file,1)) return 1;
    if(!load_values(p_file,2)) return 1;

    simulate_optimal_play(p,g1,g2,scared,max_steps);
    return 0;
}
