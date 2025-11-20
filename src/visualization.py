import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict

def save_metrics_plot(results: Dict, filename: str):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    window = 100 # Smoothing window
    
    for name, stats in results.items():
        if len(stats['rewards']) < window: continue
        r = pd.Series(stats['rewards']).rolling(window).mean()
        s = pd.Series(stats['steps']).rolling(window).mean()
        e = pd.Series(stats['errors']).rolling(window).mean()
        
        ax1.plot(r, label=name); ax2.plot(s, label=name); ax3.plot(e, label=name)
    
    ax1.set_title("Reward Convergence"); ax1.grid(True); ax1.legend()
    ax2.set_title("Steps to Goal"); ax2.grid(True)
    ax3.set_title("MSE vs Theoretical Q* (Log Scale)"); ax3.set_yscale('log'); ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_heatmap(q_table: np.ndarray, env, title: str, filename: str):
    v_grid = np.max(q_table, axis=2)
    policy = np.argmax(q_table, axis=2)
    arrows = ['↑', '↓', '←', '→']
    
    mask = np.zeros_like(v_grid, dtype=bool)
    for (r,c) in env.walls: mask[r,c] = True
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(v_grid, mask=mask, cmap='viridis', cbar=True, cbar_kws={'label': 'Value V(s)'})
    
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if (r,c) in env.cliff: 
                plt.text(c+0.5, r+0.5, 'CLIFF', color='red', ha='center', va='center', weight='bold')
            elif (r,c) == env.goal_state: 
                plt.text(c+0.5, r+0.5, 'GOAL', color='white', ha='center', va='center', weight='bold')
            elif not mask[r,c]:
                plt.text(c+0.5, r+0.5, arrows[policy[r,c]], ha='center', va='center', fontsize=8)
                
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def save_action_fear_map(q_table: np.ndarray, env, action_idx: int, title: str, filename: str):
    """Visualizes specific action Q-values to show risk aversion (e.g. Q(s, DOWN))."""
    q_vals = q_table[:, :, action_idx]
    mask = np.zeros_like(q_vals, dtype=bool)
    for (r,c) in env.walls: mask[r,c] = True
    
    plt.figure(figsize=(8, 6))
    # Center color map at -50 to highlight safe (green) vs fatal (red)
    sns.heatmap(q_vals, mask=mask, cmap='RdYlGn', center=-50, cbar_kws={'label': 'Q-Value'})
    plt.title(title)
    plt.savefig(filename)
    plt.close()