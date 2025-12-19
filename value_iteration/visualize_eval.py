#!/usr/bin/env python3
"""
Visualization for A* Evaluation Results

Generates plots for:
1. Survival rate comparison
2. Intervention rates (FRS vs Safety filter)
3. Survival rate vs initial ghost distance
4. Time-to-first-pellet distribution
5. Death location heatmap
6. Safety state analysis
7. Win time distribution
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

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

# Relative font size for bar annotations (scales with figure)
BAR_LABEL_FONTSIZE = 15

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
    'A*_Basic': 'A* Basic',
    'A*_Safety_Heuristic': 'A* + Safety Heuristic',
    'A*_FRS_Filter': 'A* + FRS Filter',
    'A*_Safety_Filter': 'A* + Safety Filter',
}

# Ghost name mapping for consistent display
GHOST_NAMES = {
    'Optimal': 'Optimal Ghost',
    'Greedy_BFS': 'Greedy BFS Ghost',
}

# Core strategies to display (excluding A* Basic which is just for reference)
CORE_STRATEGIES = ['Optimal', 'A*_Safety_Heuristic', 'A*_FRS_Filter', 'A*_Safety_Filter']

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
    df = pd.read_csv(csv_file)
    return df

def plot_survival_comparison(df, output_dir):
    """Bar chart comparing survival rates across strategies and ghost types."""
    ghost_types = df['ghost_strategy'].unique()
    
    # Vertical layout: 2 rows, 1 column
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
        ax.legend(loc='upper right', fancybox=True, shadow=True)
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
    # Vertical layout: 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
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
        ax.legend(loc='upper right', fancybox=True, shadow=True)
        
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
        for bar in bars2:
            height = bar.get_height()
            if height > 0.1:
                ax.annotate(f'{height:.2f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
    
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
    
    # Vertical layout: 2 rows, 1 column
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

def plot_first_pellet_timing(df, output_dir):
    """Plot average time to first pellet, split by ghost type."""
    # Use core strategies only
    subset = df[df['pacman_strategy'].isin(CORE_STRATEGIES)]
    
    if len(subset) == 0 or 'avg_first_pellet_tick' not in subset.columns:
        print("  Skipping first pellet timing (no data)")
        return
    
    # Vertical layout: 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    for idx, ghost_type in enumerate(['Optimal', 'Greedy_BFS']):
        ax = axes[idx]
        ghost_subset = subset[subset['ghost_strategy'] == ghost_type]
        
        labels = []
        values = []
        colors = []
        
        for _, row in ghost_subset.iterrows():
            if row['avg_first_pellet_tick'] >= 0:
                strategy_name = get_display_name(row['pacman_strategy'])
                labels.append(strategy_name)
                values.append(row['avg_first_pellet_tick'])
                colors.append(get_strategy_color(row['pacman_strategy']))
        
        if not values:
            ax.text(0.5, 0.5, f'No data vs {get_ghost_display_name(ghost_type)}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Time to First Pellet vs {get_ghost_display_name(ghost_type)}')
            continue
        
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=1)
        ax.set_xlabel('Pac-Man Strategy')
        ax.set_ylabel('Average Tick of First Pellet')
        ax.set_title(f'Time to First Pellet vs {get_ghost_display_name(ghost_type)}\n(Later = more strategic waiting)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'first_pellet_timing.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: first_pellet_timing.png")

def plot_death_heatmap(deaths_csv, maze_csv, output_dir):
    """Plot death location heatmap with walls visualized."""
    if not os.path.exists(deaths_csv):
        print("  Skipping death heatmap (no deaths CSV)")
        return
    
    df = pd.read_csv(deaths_csv)
    
    # Get all valid positions from the deaths CSV
    valid_positions = set()
    for _, row in df.iterrows():
        valid_positions.add((int(row['row']), int(row['col'])))
    
    # Get strategy columns and filter to core strategies
    all_strategy_cols = [c for c in df.columns if c not in ['row', 'col']]
    
    # Map column names to display names and filter
    strategy_cols = []
    display_names = []
    for col in all_strategy_cols:
        parts = col.split('_vs_')
        if len(parts) == 2:
            pac_strategy = parts[0]
            ghost_strategy = parts[1]
            if pac_strategy in CORE_STRATEGIES:
                strategy_cols.append(col)
                display_names.append(f"{get_display_name(pac_strategy)}\nvs {get_ghost_display_name(ghost_strategy)}")
    
    if not strategy_cols:
        strategy_cols = all_strategy_cols
        display_names = [c.replace('_vs_', '\nvs ') for c in strategy_cols]
    
    n_strategies = len(strategy_cols)
    cols = min(4, n_strategies)
    rows = (n_strategies + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_strategies == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    for idx, (strategy, display_name) in enumerate(zip(strategy_cols, display_names)):
        ax = axes[idx]
        ax.set_facecolor('#333333')  # Dark background for walls
        
        # Create 12x12 grid with NaN for walls
        grid = np.full((12, 12), np.nan)
        for _, row in df.iterrows():
            r, c = int(row['row']), int(row['col'])
            grid[r, c] = row[strategy]
        
        # Plot heatmap without grid (green=0 deaths, red=many deaths)
        im = ax.imshow(grid, cmap='RdYlGn_r', interpolation='nearest')
        ax.grid(False)  # Remove grid overlay
        
        # Draw walls as dark rectangles
        for r in range(12):
            for c in range(12):
                if (r, c) not in valid_positions:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                              fill=True, facecolor='#333333', edgecolor='#333333'))
        
        ax.set_title(display_name, fontsize=16, fontweight='bold')
        ax.set_xlabel('Column', fontsize=14)
        ax.set_ylabel('Row', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Deaths', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    
    # Hide unused subplots
    for idx in range(n_strategies, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Death Location Heatmaps', fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'death_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: death_heatmap.png")

def plot_safety_states(df, output_dir):
    """Plot percentage of time in safe vs unsafe states."""
    if 'pct_in_safe_state' not in df.columns:
        print("  Skipping safety states (no data)")
        return
    
    ghost_types = df['ghost_strategy'].unique()
    
    # Vertical layout: 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    for idx, ghost in enumerate(ghost_types):
        ax = axes[idx]
        subset = df[(df['ghost_strategy'] == ghost) & (df['pacman_strategy'].isin(CORE_STRATEGIES))]
        
        strategies = [get_display_name(s) for s in subset['pacman_strategy']]
        safe_pct = subset['pct_in_safe_state'].values
        unsafe_pct = subset['pct_in_unsafe_state'].values
        colors = [get_strategy_color(s) for s in subset['pacman_strategy']]
        
        x = np.arange(len(strategies))
        width = 0.6
        
        bars1 = ax.bar(x, safe_pct, width, label='Safe', color='#2ecc71', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x, unsafe_pct, width, bottom=safe_pct, label='Unsafe', 
                       color='#e74c3c', edgecolor='black', linewidth=1)
        
        for i, (bar, safe_val) in enumerate(zip(bars1, safe_pct)):
            ax.annotate(f'{safe_val:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, safe_val / 2),
                       ha='center', va='center', fontsize=BAR_LABEL_FONTSIZE, color='white')
        
        ax.set_xlabel('Pac-Man Strategy')
        ax.set_ylabel('% of Time')
        ax.set_title(f'Safety State Distribution vs {get_ghost_display_name(ghost)}')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=30, ha='right')
        ax.legend(loc='upper right', fancybox=True, shadow=True)
        ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'safety_states.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: safety_states.png")

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
    
    # Create vertical layout with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
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
        
        # Add median value and win count on top of bars
        for i, (bar, median, count) in enumerate(zip(bars, medians, counts)):
            height = bar.get_height()
            ax.annotate(f'{median:.0f} ticks\n({count} wins)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
    
    plt.suptitle('Win Time Statistics by Strategy', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_time_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: win_time_distribution.png")

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
    
    # Best strategy per ghost type
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
    summary.append("| Strategy | Ghost | Survival % | Win % | Avg Time | Safe % | Interventions |\n")
    summary.append("|----------|-------|------------|-------|----------|--------|---------------|\n")
    
    for _, row in df[df['pacman_strategy'].isin(CORE_STRATEGIES)].iterrows():
        intervention_info = ""
        if row['frs_intervention_rate'] > 0:
            intervention_info = f"FRS: {row['frs_intervention_rate']*100:.2f}%"
        elif row['safety_intervention_rate'] > 0:
            intervention_info = f"Safety: {row['safety_intervention_rate']*100:.2f}%"
        else:
            intervention_info = "-"
        
        safe_pct = row.get('pct_in_safe_state', 0)
        
        summary.append(f"| {get_display_name(row['pacman_strategy'])} | {get_ghost_display_name(row['ghost_strategy'])} | "
                      f"{row['survival_rate']*100:.1f}% | {row['win_rate']*100:.1f}% | "
                      f"{row['avg_survival_time']:.1f} | {safe_pct:.1f}% | {intervention_info} |\n")
    
    with open(os.path.join(output_dir, 'summary.md'), 'w') as f:
        f.writelines(summary)
    
    print("  Created: summary.md")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_eval.py <eval_results.csv> [output_dir]")
        print("\nGenerates visualizations from A* evaluation results.")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_file) or '.'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {csv_file}...")
    df = load_data(csv_file)
    print(f"  Found {len(df)} evaluation results")
    
    # Derive additional CSV filenames
    base_name = csv_file.rsplit('.', 1)[0]
    deaths_csv = base_name + "_deaths.csv"
    wins_csv = base_name + "_wins.csv"
    
    print(f"\nGenerating visualizations in {output_dir}/...")
    
    # Core plots
    plot_survival_comparison(df, output_dir)
    plot_intervention_rates(df, output_dir)
    
    # New analysis plots
    plot_survival_by_distance(df, output_dir)
    plot_first_pellet_timing(df, output_dir)
    plot_death_heatmap(deaths_csv, None, output_dir)
    plot_safety_states(df, output_dir)
    plot_win_time_distribution(wins_csv, output_dir)
    
    # Summary
    create_summary_table(df, output_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
