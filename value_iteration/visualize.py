#!/usr/bin/env python3
"""
Combined Visualization Script for Pac-Man Value Iteration Analysis

Generates plots for:
1. Survival rate comparison (bar chart)
2. Intervention rates (FRS vs Safety filter)
3. Survival rate vs initial ghost distance
4. Win time distribution
5. Death location heatmap (3x2 grid: strategies x ghost types)
6. Safety heatmap (starting position analysis)
7. Critical positions (choke points)

Usage:
    python visualize.py <eval_results.csv> [--analysis-prefix PREFIX] [--output-dir DIR]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

# Global style for column-style report
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 17
plt.rcParams['axes.titlesize'] = 21
plt.rcParams['axes.labelsize'] = 19
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'gray'
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# Relative font size for bar annotations
BAR_LABEL_FONTSIZE = 15

# Maze dimensions
MAZE_ROWS = 12
MAZE_COLS = 12

# Consistent color palette for strategies
STRATEGY_COLORS = {
    'Optimal Pacman': '#2ecc71',           # Green
    'A* + Safety Heuristic': '#3498db',    # Blue
    'A* + FRS Filter': '#f39c12',          # Orange
    'A* + Safety Filter': '#9b59b6',       # Purple
}

# Consistent color palette for ghost types
GHOST_COLORS = {
    'Optimal Ghost': '#e74c3c',     # Red
    'Greedy BFS Ghost': '#3498db',  # Blue
}

# Strategy name mapping for consistent display
STRATEGY_NAMES = {
    'Optimal': 'Optimal Pacman',
    'A*_Safety_Heuristic': 'A* + Safety Heuristic',
    'A*_FRS_Filter': 'A* + FRS Filter',
    'A*_Safety_Filter': 'A* + Safety Filter',
}

# Ghost name mapping for consistent display
GHOST_NAMES = {
    'Optimal': 'Optimal Ghost',
    'Greedy_BFS': 'Greedy BFS Ghost',
}

# Core strategies to display (A* approaches only, excluding Optimal baseline for some plots)
ASTAR_STRATEGIES = ['A*_Safety_Heuristic', 'A*_FRS_Filter', 'A*_Safety_Filter']
CORE_STRATEGIES = ['Optimal'] + ASTAR_STRATEGIES

# Use RdYlGn colormap for safety (red=unsafe, green=safe)
SAFETY_CMAP = 'RdYlGn'


def get_strategy_color(strategy):
    """Get consistent color for a strategy."""
    name = STRATEGY_NAMES.get(strategy, strategy)
    return STRATEGY_COLORS.get(name, '#95a5a6')


def get_display_name(strategy):
    """Get display name for a strategy."""
    return STRATEGY_NAMES.get(strategy, strategy)


def get_ghost_display_name(ghost):
    """Get display name for a ghost strategy."""
    return GHOST_NAMES.get(ghost, ghost)


def get_ghost_color(ghost):
    """Get consistent color for a ghost strategy."""
    name = GHOST_NAMES.get(ghost, ghost)
    return GHOST_COLORS.get(name, '#95a5a6')


def load_data(csv_file):
    """Load evaluation results from CSV."""
    return pd.read_csv(csv_file)


# =============================================================================
# Evaluation Plots
# =============================================================================

def plot_survival_comparison(df, output_dir):
    """Bar chart comparing survival rates across strategies and ghost types."""
    ghost_types = df['ghost_strategy'].unique()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    for idx, ghost in enumerate(ghost_types):
        ax = axes[idx]
        subset = df[(df['ghost_strategy'] == ghost) & (df['pacman_strategy'].isin(CORE_STRATEGIES))]
        
        strategies = [get_display_name(s) for s in subset['pacman_strategy']]
        survival = subset['survival_rate'].values * 100
        win = subset['win_rate'].values * 100
        colors = [get_strategy_color(s) for s in subset['pacman_strategy']]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, survival, width, label='Survival %', 
                       color=colors, alpha=0.9, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, win, width, label='Win %', 
                       color=colors, alpha=0.5, hatch='//', edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Pac-Man Strategy')
        ax.set_ylabel('Rate (%)')
        ax.set_title(f'Performance vs {get_ghost_display_name(ghost)}')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=30, ha='right')
        ax.legend(loc='lower right', fancybox=True, shadow=True)
        ax.set_ylim(0, 115)
        
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'survival_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: survival_comparison.png")


def plot_intervention_rates(df, output_dir):
    """Plot intervention rates for FRS and Safety filter strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 14))
    
    # FRS interventions
    ax = axes[0]
    frs_data = df[df['pacman_strategy'] == 'A*_FRS_Filter']
    if len(frs_data) > 0:
        ghosts = [get_ghost_display_name(g) for g in frs_data['ghost_strategy']]
        intervention_rate = frs_data['frs_intervention_rate'].values * 100
        colors = [get_ghost_color(g) for g in frs_data['ghost_strategy']]
        
        bars = ax.bar(ghosts, intervention_rate, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Ghost Strategy')
        ax.set_ylabel('Intervention Rate (% of ticks)')
        ax.set_title('A* + FRS Filter: Intervention Rate\n(% of ticks where FRS blocked preferred move)')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
        # Set ylim with headroom for labels
        max_val = max(intervention_rate) if len(intervention_rate) > 0 else 1
        ax.set_ylim(0, max_val * 1.25)
    else:
        ax.text(0.5, 0.5, 'No FRS Filter data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title('A* + FRS Filter: Intervention Rate')
    
    # Safety filter interventions
    ax = axes[1]
    safety_data = df[df['pacman_strategy'] == 'A*_Safety_Filter']
    if len(safety_data) > 0:
        ghosts = [get_ghost_display_name(g) for g in safety_data['ghost_strategy']]
        intervention_rate = safety_data['safety_intervention_rate'].values * 100
        fallback_rate = safety_data['safety_fallback_rate'].values * 100
        
        x = np.arange(len(ghosts))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, intervention_rate, width, 
                       label='Intervention (unsafe move blocked)', 
                       color='#f39c12', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, fallback_rate, width, 
                       label='Fallback (no safe move exists)', 
                       color='#e74c3c', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Ghost Strategy')
        ax.set_ylabel('Rate (% of ticks)')
        ax.set_title('A* + Safety Filter: Intervention Rates\n(% of ticks where filter acted)')
        ax.set_xticks(x)
        ax.set_xticklabels(ghosts)
        ax.legend(loc='lower right', fancybox=True, shadow=True)
        
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
        for bar in bars2:
            height = bar.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.2f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
        # Set ylim with headroom for labels
        max_val = max(max(intervention_rate), max(fallback_rate)) if len(intervention_rate) > 0 else 1
        ax.set_ylim(0, max_val * 1.3)
    else:
        ax.text(0.5, 0.5, 'No Safety Filter data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title('A* + Safety Filter: Intervention Rates')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intervention_rates.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: intervention_rates.png")


def plot_survival_by_distance(df, output_dir):
    """Plot survival rate vs initial ghost distance."""
    distance_labels = ['0-2', '3-5', '6-8', '9+']
    distance_cols = ['survival_dist_0_2', 'survival_dist_3_5', 
                     'survival_dist_6_8', 'survival_dist_9plus']
    
    ghost_types = df['ghost_strategy'].unique()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    for idx, ghost in enumerate(ghost_types):
        ax = axes[idx]
        subset = df[(df['ghost_strategy'] == ghost) & (df['pacman_strategy'].isin(CORE_STRATEGIES))]
        
        x = np.arange(len(distance_labels))
        width = 0.18
        
        for i, (_, row) in enumerate(subset.iterrows()):
            strategy = row['pacman_strategy']
            values = [row[col] for col in distance_cols]
            offset = (i - len(subset)/2 + 0.5) * width
            color = get_strategy_color(strategy)
            bars = ax.bar(x + offset, values, width, label=get_display_name(strategy), 
                          color=color, edgecolor='black', linewidth=1)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.0f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 2), textcoords="offset points",
                               ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE - 2)
        
        ax.set_xlabel('Initial Min Ghost Distance (cells)')
        ax.set_ylabel('Survival Rate (%)')
        ax.set_title(f'Survival by Starting Distance vs {get_ghost_display_name(ghost)}')
        ax.set_xticks(x)
        ax.set_xticklabels(distance_labels)
        ax.legend(loc='lower right', fancybox=True, shadow=True)
        ax.set_ylim(0, 115)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'survival_by_distance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: survival_by_distance.png")


def plot_win_time_distribution(wins_csv, output_dir):
    """Plot win time distribution as grouped bar chart with median and count."""
    if not os.path.exists(wins_csv):
        print("  Skipping win time distribution (no wins CSV)")
        return
    
    df = pd.read_csv(wins_csv)
    
    # Get strategy columns and filter to core strategies
    all_strategy_cols = [c for c in df.columns if c != 'index']
    
    strategy_cols = []
    for col in all_strategy_cols:
        parts = col.split('_vs_')
        if len(parts) == 2:
            pac_strategy = parts[0]
            if pac_strategy in CORE_STRATEGIES and df[col].notna().sum() > 0:
                strategy_cols.append(col)
    
    if not strategy_cols:
        strategy_cols = [c for c in all_strategy_cols if df[c].notna().sum() > 0]
    
    if not strategy_cols:
        print("  Skipping win time distribution (no wins)")
        return
    
    # Collect stats for each strategy/ghost combination
    stats_data = []
    for col in strategy_cols:
        times = df[col].dropna().values
        if len(times) > 0:
            parts = col.split('_vs_')
            if len(parts) == 2:
                stats_data.append({
                    'strategy': parts[0],
                    'ghost': parts[1],
                    'median': np.median(times),
                    'mean': np.mean(times),
                    'count': len(times),
                    'min': np.min(times),
                    'max': np.max(times)
                })
    
    if not stats_data:
        print("  Skipping win time distribution (no data)")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    for idx, ghost_type in enumerate(['Optimal', 'Greedy_BFS']):
        ax = axes[idx]
        ghost_stats = [s for s in stats_data if s['ghost'] == ghost_type]
        
        if not ghost_stats:
            ax.text(0.5, 0.5, f'No wins vs {get_ghost_display_name(ghost_type)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Win Statistics vs {get_ghost_display_name(ghost_type)}')
            continue
        
        strategies = [s['strategy'] for s in ghost_stats]
        medians = [s['median'] for s in ghost_stats]
        counts = [s['count'] for s in ghost_stats]
        colors = [get_strategy_color(s) for s in strategies]
        labels = [get_display_name(s) for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.6
        
        bars = ax.bar(x, medians, width, color=colors, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Pac-Man Strategy')
        ax.set_ylabel('Median Win Time (ticks)')
        ax.set_title(f'Win Statistics vs {get_ghost_display_name(ghost_type)}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        
        for i, (bar, median, count) in enumerate(zip(bars, medians, counts)):
            height = bar.get_height()
            ax.annotate(f'{median:.0f} ticks\n({count} wins)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
        
        # Set ylim with headroom for labels
        max_val = max(medians) if medians else 1
        ax.set_ylim(0, max_val * 1.35)
    
    plt.suptitle('Win Time Statistics by Strategy', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_time_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: win_time_distribution.png")


def plot_death_heatmap(deaths_csv, output_dir):
    """Plot death location heatmap in 3x2 grid (strategies x ghost types) with cell values."""
    if not os.path.exists(deaths_csv):
        print("  Skipping death heatmap (no deaths CSV)")
        return
    
    df = pd.read_csv(deaths_csv)
    
    # Get all valid positions from the deaths CSV
    valid_positions = set()
    for _, row in df.iterrows():
        valid_positions.add((int(row['row']), int(row['col'])))
    
    # Define the 3 A* strategies and 2 ghost types for the grid
    strategies = ASTAR_STRATEGIES
    ghost_types = ['Optimal', 'Greedy_BFS']
    
    # Create 3 rows x 2 columns figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    
    for row_idx, strategy in enumerate(strategies):
        for col_idx, ghost in enumerate(ghost_types):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('#333333')  # Dark background for walls
            ax.grid(False)
            
            col_name = f"{strategy}_vs_{ghost}"
            
            if col_name not in df.columns:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, color='white')
                ax.set_title(f'{get_display_name(strategy)}\nvs {get_ghost_display_name(ghost)}', 
                            fontsize=14, fontweight='bold')
                continue
            
            # Create grid with NaN for walls
            grid = np.full((MAZE_ROWS, MAZE_COLS), np.nan)
            for _, data_row in df.iterrows():
                r, c = int(data_row['row']), int(data_row['col'])
                grid[r, c] = data_row[col_name]
            
            # Get max value for colormap scaling (excluding NaN)
            max_val = np.nanmax(grid) if np.nanmax(grid) > 0 else 1
            
            # Plot heatmap (green=0 deaths, red=many deaths)
            im = ax.imshow(grid, cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=max_val)
            
            # Draw walls and add text values
            for r in range(MAZE_ROWS):
                for c in range(MAZE_COLS):
                    if (r, c) not in valid_positions:
                        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                                  fill=True, facecolor='#333333', edgecolor='#333333'))
                    else:
                        val = grid[r, c]
                        if not np.isnan(val):
                            # Always use black text for readability
                            ax.text(c, r, f'{int(val)}', ha='center', va='center', 
                                   fontsize=8, color='black', fontweight='bold')
            
            # Title shows strategy and ghost type
            if row_idx == 0:
                ax.set_title(f'{get_ghost_display_name(ghost)}', fontsize=16, fontweight='bold')
            
            # Row labels on the left
            if col_idx == 0:
                ax.set_ylabel(f'{get_display_name(strategy)}', fontsize=14, fontweight='bold')
            
            ax.set_xticks(range(MAZE_COLS))
            ax.set_yticks(range(MAZE_ROWS))
            ax.tick_params(labelsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Deaths', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle('Death Location Heatmaps by Strategy and Ghost Type', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'death_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: death_heatmap.png")


# =============================================================================
# Analysis Plots (from value table analysis)
# =============================================================================

def plot_safety_heatmap(csv_file, output_dir):
    """Generate maze heatmap showing safety rate per position."""
    if not os.path.exists(csv_file):
        print(f"  Skipping safety heatmap ({csv_file} not found)")
        return
    
    df = pd.read_csv(csv_file)
    
    # Create grid
    maze = np.full((MAZE_ROWS, MAZE_COLS), np.nan)
    for _, row in df.iterrows():
        maze[int(row['row']), int(row['col'])] = row['safety_rate']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#333333')
    ax.grid(False)
    
    im = ax.imshow(maze, cmap=SAFETY_CMAP, vmin=0, vmax=100, aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Safety Rate (%)', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    
    # Mark walls and add text
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if np.isnan(maze[r, c]):
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                          fill=True, facecolor='#333333', edgecolor='#333333'))
    
    for _, row in df.iterrows():
        r, c = int(row['row']), int(row['col'])
        rate = row['safety_rate']
        is_pellet = row['is_pellet'] == 1
        
        text_color = 'white' if rate < 50 else 'black'
        
        if is_pellet:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                       fill=False, edgecolor='#3498db', linewidth=3))
            ax.text(c, r, f'{rate:.0f}*', ha='center', va='center', 
                   fontsize=11, fontweight='bold', color=text_color)
        else:
            ax.text(c, r, f'{rate:.0f}', ha='center', va='center', 
                   fontsize=10, color=text_color)
    
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('Pac-Man Safety Rate by Starting Position\n(Both ghosts alive, pellet(s) exist)')
    ax.set_xticks(range(maze.shape[1]))
    ax.set_yticks(range(maze.shape[0]))
    
    pellet_patch = mpatches.Patch(edgecolor='#3498db', facecolor='white', 
                                  linewidth=2, label='Pellet Position')
    ax.legend(handles=[pellet_patch], loc='upper left', bbox_to_anchor=(1.15, 1),
              fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'safety_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("  Created: safety_heatmap.png")


def plot_critical_positions(csv_file, output_dir):
    """Plot critical positions (choke points) heatmap with cell values."""
    if not os.path.exists(csv_file):
        print(f"  Skipping critical positions ({csv_file} not found)")
        return
    
    df = pd.read_csv(csv_file)
    
    # Create grids
    maze = np.full((MAZE_ROWS, MAZE_COLS), np.nan)
    trapped_maze = np.full((MAZE_ROWS, MAZE_COLS), np.nan)
    
    for _, row in df.iterrows():
        maze[int(row['row']), int(row['col'])] = row['avg_safe_moves']
        trapped_maze[int(row['row']), int(row['col'])] = row['trapped_pct']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
    
    # Plot 1: Average safe moves
    ax1.set_facecolor('#333333')
    ax1.grid(False)
    max_val = np.nanmax(maze) if np.nanmax(maze) > 0 else 1
    im1 = ax1.imshow(maze, cmap='RdYlGn', vmin=0, vmax=max_val, aspect='equal')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Avg Safe Moves', fontsize=14)
    cbar1.ax.tick_params(labelsize=12)
    
    # Mark pellets, walls, and add text values
    for _, row in df.iterrows():
        r, c = int(row['row']), int(row['col'])
        val = row['avg_safe_moves']
        if row['is_pellet'] == 1:
            ax1.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                       fill=False, edgecolor='#3498db', linewidth=3))
        # Add text value
        text_color = 'white' if val < max_val * 0.5 else 'black'
        ax1.text(c, r, f'{val:.1f}', ha='center', va='center', 
                fontsize=8, color=text_color)
    
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if np.isnan(maze[r, c]):
                ax1.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                          fill=True, facecolor='#333333', edgecolor='#333333'))
    
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.set_title('Average Safe Moves per Position\n(Higher = Safer)')
    ax1.set_xticks(range(maze.shape[1]))
    ax1.set_yticks(range(maze.shape[0]))
    
    # Plot 2: Trapped percentage
    ax2.set_facecolor('#333333')
    ax2.grid(False)
    im2 = ax2.imshow(trapped_maze, cmap='Reds', vmin=0, vmax=100, aspect='equal')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Trapped %', fontsize=14)
    cbar2.ax.tick_params(labelsize=12)
    
    for _, row in df.iterrows():
        r, c = int(row['row']), int(row['col'])
        val = row['trapped_pct']
        if row['is_pellet'] == 1:
            ax2.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                       fill=False, edgecolor='#3498db', linewidth=3))
        # Add text value
        text_color = 'white' if val > 50 else 'black'
        ax2.text(c, r, f'{val:.0f}', ha='center', va='center', 
                fontsize=8, color=text_color)
    
    for r in range(trapped_maze.shape[0]):
        for c in range(trapped_maze.shape[1]):
            if np.isnan(trapped_maze[r, c]):
                ax2.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                          fill=True, facecolor='#333333', edgecolor='#333333'))
    
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.set_title('Trapped Configurations %\n(Higher = More Dangerous)')
    ax2.set_xticks(range(trapped_maze.shape[1]))
    ax2.set_yticks(range(trapped_maze.shape[0]))
    
    plt.suptitle('Critical Position Analysis', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'critical_positions.png'), bbox_inches='tight')
    plt.close()
    print("  Created: critical_positions.png")


# =============================================================================
# Summary
# =============================================================================

def create_summary_table(df, output_dir):
    """Create a summary markdown table."""
    summary = []
    
    summary.append("# A* Evaluation Results Summary\n\n")
    
    summary.append("## Strategies Evaluated\n\n")
    summary.append("1. **Optimal Pacman**: Value iteration optimal policy\n")
    summary.append("2. **A* + Safety Heuristic**: A* with large penalty for unsafe positions\n")
    summary.append("3. **A* + FRS Filter**: A* that avoids ghost-reachable positions\n")
    summary.append("4. **A* + Safety Filter**: A* that only allows provably safe moves\n\n")
    
    summary.append("## Key Findings\n\n")
    
    for ghost in df['ghost_strategy'].unique():
        subset = df[(df['ghost_strategy'] == ghost) & (df['pacman_strategy'].isin(CORE_STRATEGIES))]
        if len(subset) == 0:
            continue
        best = subset.loc[subset['survival_rate'].idxmax()]
        summary.append(f"### Best vs {get_ghost_display_name(ghost)}\n\n")
        summary.append(f"- **Strategy**: {get_display_name(best['pacman_strategy'])}\n")
        summary.append(f"- **Survival Rate**: {best['survival_rate']*100:.1f}%\n")
        summary.append(f"- **Win Rate**: {best['win_rate']*100:.1f}%\n")
        summary.append(f"- **Avg Survival**: {best['avg_survival_time']:.1f} ticks\n\n")
    
    summary.append("## Full Results Table\n\n")
    summary.append("| Strategy | Ghost | Survival % | Win % | Avg Time | Interventions |\n")
    summary.append("|----------|-------|------------|-------|----------|---------------|\n")
    
    for _, row in df[df['pacman_strategy'].isin(CORE_STRATEGIES)].iterrows():
        intervention_info = ""
        if row['frs_intervention_rate'] > 0:
            intervention_info = f"FRS: {row['frs_intervention_rate']*100:.2f}%"
        elif row['safety_intervention_rate'] > 0:
            intervention_info = f"Safety: {row['safety_intervention_rate']*100:.2f}%"
        else:
            intervention_info = "-"
        
        summary.append(f"| {get_display_name(row['pacman_strategy'])} | {get_ghost_display_name(row['ghost_strategy'])} | "
                      f"{row['survival_rate']*100:.1f}% | {row['win_rate']*100:.1f}% | "
                      f"{row['avg_survival_time']:.1f} | {intervention_info} |\n")
    
    with open(os.path.join(output_dir, 'summary.md'), 'w') as f:
        f.writelines(summary)
    
    print("  Created: summary.md")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize Pac-Man evaluation and analysis results')
    parser.add_argument('eval_csv', nargs='?', help='Evaluation results CSV file')
    parser.add_argument('--analysis-prefix', type=str, help='Prefix for analysis CSV files (e.g., "analysis_4p")')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for figures')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Process evaluation results if provided
    if args.eval_csv and os.path.exists(args.eval_csv):
        print(f"Loading evaluation data from {args.eval_csv}...")
        df = load_data(args.eval_csv)
        print(f"  Found {len(df)} evaluation results")
        
        base_name = args.eval_csv.rsplit('.', 1)[0]
        deaths_csv = base_name + "_deaths.csv"
        wins_csv = base_name + "_wins.csv"
        
        print(f"\nGenerating evaluation visualizations in {output_dir}/...")
        
        plot_survival_comparison(df, output_dir)
        plot_intervention_rates(df, output_dir)
        plot_survival_by_distance(df, output_dir)
        plot_win_time_distribution(wins_csv, output_dir)
        plot_death_heatmap(deaths_csv, output_dir)
        create_summary_table(df, output_dir)
    
    # Process analysis files if prefix provided
    if args.analysis_prefix:
        print(f"\nGenerating analysis visualizations from {args.analysis_prefix}_*.csv...")
        
        heatmap_file = f"{args.analysis_prefix}_heatmap.csv"
        critical_file = f"{args.analysis_prefix}_critical.csv"
        
        plot_safety_heatmap(heatmap_file, output_dir)
        plot_critical_positions(critical_file, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

