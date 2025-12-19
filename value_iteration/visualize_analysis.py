#!/usr/bin/env python3
"""
Visualization script for Pacman value analysis data.
Generates publication-quality figures from CSV exports.

Usage:
    python visualize_analysis.py --prefix analysis
    python visualize_analysis.py --heatmap analysis_heatmap.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

# Global style for column-style report (larger fonts) - matching visualize_eval.py
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'gray'
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# Consistent color palette matching visualize_eval.py
STRATEGY_COLORS = {
    'Optimal Pacman': '#2ecc71',           # Green
    'A* + Safety Heuristic': '#3498db',    # Blue
    'A* + FRS Filter': '#f39c12',          # Orange
    'A* + Safety Filter': '#9b59b6',       # Purple
}

# Use RdYlGn colormap for safety (red=unsafe, green=safe) - matches death heatmap style
SAFETY_CMAP = 'RdYlGn'


MAZE_ROWS = 12
MAZE_COLS = 12

def load_maze_from_heatmap(df):
    """Reconstruct maze grid from heatmap data."""
    # Use full 12x12 maze dimensions to show outer walls
    maze = np.full((MAZE_ROWS, MAZE_COLS), np.nan)
    
    for _, row in df.iterrows():
        maze[int(row['row']), int(row['col'])] = row['safety_rate']
    
    return maze


def plot_safety_heatmap(csv_file, output_file=None):
    """Generate maze heatmap showing safety rate per position."""
    df = pd.read_csv(csv_file)
    maze = load_maze_from_heatmap(df)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#333333')  # Dark background for walls
    
    # Turn off grid for heatmap
    ax.grid(False)
    
    # Plot heatmap
    im = ax.imshow(maze, cmap=SAFETY_CMAP, vmin=0, vmax=100, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Safety Rate (%)', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    
    # Mark walls (so they're behind text)
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if np.isnan(maze[r, c]):
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                          fill=True, facecolor='#333333', edgecolor='#333333'))
    
    # Add text annotations and pellet markers
    for _, row in df.iterrows():
        r, c = int(row['row']), int(row['col'])
        rate = row['safety_rate']
        is_pellet = row['is_pellet'] == 1
        
        # Choose text color based on background
        text_color = 'white' if rate < 50 else 'black'
        
        if is_pellet:
            # Highlight pellet position
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                       fill=False, edgecolor='#3498db', linewidth=3))
            ax.text(c, r, f'{rate:.0f}*', ha='center', va='center', 
                   fontsize=11, fontweight='bold', color=text_color)
        else:
            ax.text(c, r, f'{rate:.0f}', ha='center', va='center', 
                   fontsize=10, color=text_color)
    
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('Pacman Safety Rate by Starting Position\n(Both ghosts alive, pellet(s) exist)')
    
    # Set ticks
    ax.set_xticks(range(maze.shape[1]))
    ax.set_yticks(range(maze.shape[0]))
    
    # Legend positioned outside the plot for clarity
    pellet_patch = mpatches.Patch(edgecolor='#3498db', facecolor='white', 
                                  linewidth=2, label='Pellet Position')
    ax.legend(handles=[pellet_patch], loc='upper left', bbox_to_anchor=(1.15, 1),
              fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_ghost_config(csv_file, output_file=None):
    """Plot ghost proximity danger analysis (Pacman fixed).
    
    Shows how Pacman's safety changes based on where Ghost 1 is positioned,
    with a focus on distance-based insights.
    """
    df = pd.read_csv(csv_file)
    
    # Get Pacman position
    pacman_row_df = df[df['is_pacman'] == 1]
    if len(pacman_row_df) > 0:
        pacman_pos = int(pacman_row_df['ghost1_pos'].iloc[0])
        pacman_r = int(pacman_row_df['ghost1_row'].iloc[0])
        pacman_c = int(pacman_row_df['ghost1_col'].iloc[0])
    else:
        pacman_pos = -1
        pacman_r, pacman_c = 5, 4  # Default center
    
    # Filter out Pacman position for the heatmap
    df_ghosts = df[df['is_pacman'] == 0].copy()
    
    if len(df_ghosts) == 0:
        print("Warning: No ghost position data found")
        return
    
    # Compute Manhattan distance from Pacman
    df_ghosts['distance'] = abs(df_ghosts['ghost1_row'] - pacman_r) + abs(df_ghosts['ghost1_col'] - pacman_c)
    
    # Vertical layout: 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(10, 14))
    
    # ===== Top plot: Maze heatmap =====
    ax1 = axes[0]
    ax1.set_facecolor('#333333')  # Dark background for walls
    ax1.grid(False)
    
    # Use full 12x12 maze dimensions
    maze = np.full((MAZE_ROWS, MAZE_COLS), np.nan)
    
    for _, row in df_ghosts.iterrows():
        maze[int(row['ghost1_row']), int(row['ghost1_col'])] = row['safety_rate']
    
    # Use consistent colormap with safety heatmap
    im = ax1.imshow(maze, cmap=SAFETY_CMAP, vmin=0, vmax=100, aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Pacman Safety Rate (%)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Mark walls (behind other elements)
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if np.isnan(maze[r, c]) and not (r == pacman_r and c == pacman_c):
                ax1.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                          fill=True, facecolor='#333333', edgecolor='#333333'))
    
    # Add text annotations for safety values
    for _, row in df_ghosts.iterrows():
        r, c = int(row['ghost1_row']), int(row['ghost1_col'])
        rate = row['safety_rate']
        text_color = 'white' if rate < 50 else 'black'
        ax1.text(c, r, f'{rate:.0f}', ha='center', va='center', 
                fontsize=8, color=text_color)
    
    # Mark Pacman position
    ax1.add_patch(plt.Rectangle((pacman_c-0.5, pacman_r-0.5), 1, 1, 
                              fill=True, facecolor='#f1c40f', edgecolor='black', linewidth=2))
    ax1.text(pacman_c, pacman_r, 'P', ha='center', va='center', fontsize=15, fontweight='bold', color='black')
    
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.set_title(f'Safety by Ghost 1 Position\n(Pacman at [{pacman_r},{pacman_c}], avg over Ghost 2)')
    ax1.set_xticks(range(maze.shape[1]))
    ax1.set_yticks(range(maze.shape[0]))
    
    # Legend for Pacman marker
    pacman_patch = mpatches.Patch(facecolor='#f1c40f', edgecolor='black', 
                                  linewidth=2, label='Pacman Position')
    ax1.legend(handles=[pacman_patch], loc='upper left', bbox_to_anchor=(1.02, 1),
              fancybox=True, shadow=True)
    
    # ===== Bottom plot: Distance vs Safety scatter =====
    ax2 = axes[1]
    
    # Group by distance and compute stats
    dist_stats = df_ghosts.groupby('distance').agg({
        'safety_rate': ['mean', 'std', 'count']
    }).reset_index()
    dist_stats.columns = ['distance', 'mean_safety', 'std_safety', 'count']
    
    # Scatter plot with all points
    ax2.scatter(df_ghosts['distance'], df_ghosts['safety_rate'], 
               alpha=0.3, s=30, c='#3498db', label='Individual positions')
    
    # Line plot of means
    ax2.plot(dist_stats['distance'], dist_stats['mean_safety'], 
            color='#e74c3c', linewidth=2, marker='o', markersize=8, label='Mean safety')
    
    # Error bars
    ax2.fill_between(dist_stats['distance'], 
                    dist_stats['mean_safety'] - dist_stats['std_safety'],
                    dist_stats['mean_safety'] + dist_stats['std_safety'],
                    alpha=0.2, color='#e74c3c')
    
    ax2.set_xlabel('Manhattan Distance from Pacman')
    ax2.set_ylabel('Pacman Safety Rate (%)')
    ax2.set_title('Safety vs Ghost 1 Distance')
    ax2.legend(loc='lower right', fancybox=True, shadow=True)
    ax2.set_xlim(0, dist_stats['distance'].max() + 1)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    # Add annotation for key insight
    min_safe_dist = dist_stats[dist_stats['mean_safety'] > 50]['distance'].min() if len(dist_stats[dist_stats['mean_safety'] > 50]) > 0 else None
    if min_safe_dist is not None:
        ax2.axvline(x=min_safe_dist, color='#2ecc71', linestyle='--', alpha=0.7, linewidth=2)
        ax2.text(min_safe_dist + 0.2, 90, f'Safe zone: dist ≥ {min_safe_dist}', 
                fontsize=12, color='#2ecc71', fontweight='bold')
    
    plt.suptitle('Ghost Proximity Analysis', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_critical_positions(csv_file, output_file=None):
    """Plot critical positions (choke points) heatmap."""
    df = pd.read_csv(csv_file)
    
    # Use full 12x12 maze dimensions
    maze = np.full((MAZE_ROWS, MAZE_COLS), np.nan)
    trapped_maze = np.full((MAZE_ROWS, MAZE_COLS), np.nan)
    
    for _, row in df.iterrows():
        maze[int(row['row']), int(row['col'])] = row['avg_safe_moves']
        trapped_maze[int(row['row']), int(row['col'])] = row['trapped_pct']
    
    # Vertical layout: 2 rows, 1 column
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
    
    # Plot 1: Average safe moves
    ax1.set_facecolor('#333333')
    ax1.grid(False)
    im1 = ax1.imshow(maze, cmap='RdYlGn', vmin=0, vmax=maze[~np.isnan(maze)].max(), aspect='equal')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Avg Safe Moves', fontsize=14)
    cbar1.ax.tick_params(labelsize=12)
    
    # Mark pellets and walls
    for _, row in df.iterrows():
        r, c = int(row['row']), int(row['col'])
        if row['is_pellet'] == 1:
            ax1.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                       fill=False, edgecolor='#3498db', linewidth=3))
    
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
        if row['is_pellet'] == 1:
            ax2.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                       fill=False, edgecolor='#3498db', linewidth=3))
    
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
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_threshold_analysis(csv_file, output_file=None):
    """Plot threshold sensitivity analysis for safety filter tuning."""
    df = pd.read_csv(csv_file)
    
    # Vertical layout: 2 rows, 1 column
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Interventions and Missed Dangers
    ax1.fill_between(df['threshold'], 0, df['interventions'], 
                     alpha=0.3, color='#3498db', label='Interventions')
    ax1.plot(df['threshold'], df['interventions'], color='#3498db', linewidth=2)
    ax1.plot(df['threshold'], df['missed_danger'], color='#e74c3c', linewidth=2, 
             marker='o', markersize=4, label='Missed Dangers')
    
    ax1.set_xlabel('TTR Threshold')
    ax1.set_ylabel('Number of States')
    ax1.set_title('Safety Filter Intervention Analysis')
    ax1.legend(loc='upper left', fancybox=True, shadow=True)
    ax1.set_xlim(0, df['threshold'].max())
    
    # Add secondary y-axis for missed danger percentage
    ax1b = ax1.twinx()
    total = df['total'].iloc[0]
    ax1b.plot(df['threshold'], 100 * df['missed_danger'] / total, 
              color='#e74c3c', linestyle='--', alpha=0.5)
    ax1b.set_ylabel('Missed Danger (%)', color='#e74c3c')
    ax1b.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Plot 2: Precision and Intervention Rate
    intervention_rate = 100 * df['interventions'] / df['total']
    ax2.plot(df['threshold'], df['precision'], color='#2ecc71', linewidth=2, 
             marker='s', markersize=4, label='Precision (%)')
    ax2.plot(df['threshold'], intervention_rate, color='#3498db', linewidth=2,
             marker='o', markersize=4, label='Intervention Rate (%)')
    
    ax2.set_xlabel('TTR Threshold')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Filter Precision and Intervention Rate')
    ax2.legend(loc='center right', fancybox=True, shadow=True)
    ax2.set_xlim(0, df['threshold'].max())
    ax2.set_ylim(0, 105)
    
    # Highlight recommended threshold region
    safe_thresholds = df[df['missed_danger'] == 0]['threshold']
    if len(safe_thresholds) > 0:
        min_safe = safe_thresholds.min()
        ax1.axvline(x=min_safe, color='#2ecc71', linestyle=':', linewidth=2,
                   label=f'Min safe threshold: {min_safe}')
        ax2.axvline(x=min_safe, color='#2ecc71', linestyle=':', linewidth=2)
    
    plt.suptitle('Safety Filter Threshold Tuning', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Pacman value analysis data')
    parser.add_argument('--prefix', type=str, help='Prefix for all CSV files (e.g., "analysis")')
    parser.add_argument('--heatmap', type=str, help='Heatmap CSV file')
    parser.add_argument('--ghost', type=str, help='Ghost configuration CSV file')
    parser.add_argument('--critical', type=str, help='Critical positions CSV file')
    parser.add_argument('--threshold', type=str, help='Threshold analysis CSV file')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for figures')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                       help='Output format')
    
    args = parser.parse_args()
    
    # Determine file paths
    if args.prefix:
        heatmap_file = f"{args.prefix}_heatmap.csv"
        ghost_file = f"{args.prefix}_ghost.csv"
        critical_file = f"{args.prefix}_critical.csv"
        threshold_file = f"{args.prefix}_threshold.csv"
    else:
        heatmap_file = args.heatmap
        ghost_file = args.ghost
        critical_file = args.critical
        threshold_file = args.threshold
    
    output_dir = args.output_dir
    fmt = args.format
    
    # Generate individual plots
    if heatmap_file and os.path.exists(heatmap_file):
        plot_safety_heatmap(heatmap_file, 
                           os.path.join(output_dir, f'safety_heatmap.{fmt}'))
    
    if ghost_file and os.path.exists(ghost_file):
        plot_ghost_config(ghost_file,
                         os.path.join(output_dir, f'ghost_config.{fmt}'))
    
    if critical_file and os.path.exists(critical_file):
        plot_critical_positions(critical_file,
                               os.path.join(output_dir, f'critical_positions.{fmt}'))
    
    if threshold_file and os.path.exists(threshold_file):
        plot_threshold_analysis(threshold_file,
                               os.path.join(output_dir, f'threshold_analysis.{fmt}'))
    
    print("\nAll visualizations generated successfully!")


if __name__ == '__main__':
    main()
