# import numpy as np
# import matplotlib.pyplot as plt

# # --- 3. VISUALIZATION (Updated for Cliff) ---

# def visualize_cliff_policy(q_table, env, title):
#     arrows = ['\u2191', '\u2193', '\u2190', '\u2192']
#     policy = np.argmax(q_table, axis=2)
    
#     grid_viz = np.full((env.grid_size, env.grid_size), 0.9)
    
#     for (r, c) in env.walls:
#         grid_viz[r, c] = 0.0
        
#     # Mark the Cliff in RED
#     for (r, c) in env.cliff:
#         grid_viz[r, c] = 0.5 # different color value for logic
        
#     grid_viz[env.start_state] = 0.3
#     grid_viz[env.goal_state] = 0.6
    
#     plt.figure(figsize=(10, 10))
#     # Custom colormap to make cliff red
#     plt.imshow(grid_viz, cmap='gray', vmin=0, vmax=1)
    
#     # Overlay Red rectangles for cliff
#     for (r, c) in env.cliff:
#         plt.gca().add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='red', alpha=0.6))
#         plt.text(c, r, "CLIFF", ha='center', va='center', color='white', fontsize=6, weight='bold')

#     for r in range(env.grid_size):
#         for c in range(env.grid_size):
#             if (r, c) in env.walls or (r, c) == env.goal_state or (r,c) in env.cliff:
#                 continue
            
#             action = policy[r, c]
#             arrow = arrows[action]
#             color = 'blue' if 'SARSA' in title else 'orange'
#             plt.text(c, r, arrow, ha='center', va='center', color=color, fontsize=10, weight='bold')
            
#     plt.title(title, fontsize=16)
#     plt.xticks([]); plt.yticks([])
#     plt.show()

# def plot_rewards(sarsa_stats, q_stats, window=100):
#     s_smooth = np.convolve(sarsa_stats['rewards'], np.ones(window)/window, mode='valid')
#     q_smooth = np.convolve(q_stats['rewards'], np.ones(window)/window, mode='valid')
    
#     plt.figure(figsize=(12,6))
#     plt.plot(s_smooth, label='SARSA Rewards')
#     plt.plot(q_smooth, label='Q-Learning Rewards')
#     plt.ylabel('Total Reward per Episode')
#     plt.xlabel('Episode')
#     plt.title('SARSA vs Q-Learning (Cliff Walking)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()



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