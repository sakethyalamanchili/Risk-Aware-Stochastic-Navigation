# import numpy as np
# import matplotlib.pyplot as plt
# from environment import CliffWalkingEnvironment
# from algorithms import train_q_learning, train_sarsa

# # --- VISUALIZATION FUNCTIONS (Copied for a standalone main) ---

# def visualize_cliff_policy(q_table, env, title):
#     arrows = ['\u2191', '\u2193', '\u2190', '\u2192']
#     policy = np.argmax(q_table, axis=2)
#     grid_viz = np.full((env.grid_size, env.grid_size), 0.9)
    
#     for (r, c) in env.walls: grid_viz[r, c] = 0.0
#     for (r, c) in env.cliff: grid_viz[r, c] = 0.5 
        
#     grid_viz[env.start_state] = 0.3
#     grid_viz[env.goal_state] = 0.6
    
#     plt.figure(figsize=(10, 10))
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
#     plt.ylabel('Total Reward per Episode (Smoothed)')
#     plt.xlabel('Episode')
#     plt.title('SARSA vs Q-Learning (Stochastic Cliff Walking)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # --- MAIN EXECUTION ---

# if __name__ == "__main__":
#     # Hyperparameters
#     PARAMS = {
#         'num_episodes': 5000,
#         'alpha': 0.1,
#         'gamma': 0.99,
#         'epsilon_start': 1.0,
#         'epsilon_decay': 0.999,
#         # Crucial for Stochasticity: 
#         # A high epsilon_min proves SARSA needs a safer path.
#         'epsilon_min': 0.1, 
#         'max_steps': 100
#     }
    
#     # 1. Initialize the Environment (Stochasticity is set in environment.py)
#     env = CliffWalkingEnvironment()
    
#     print(f"--- Starting Stochastic Tabular RL Experiment ---")
#     print(f"Action Slip Probability: {env.action_slip_prob*100:.0f}%")
    
#     # 2. Train Algorithms
#     print("\nTraining SARSA (On-Policy)...")
#     s_q, s_stats = train_sarsa(env, PARAMS)
    
#     print("Training Q-Learning (Off-Policy)...")
#     q_q, q_stats = train_q_learning(env, PARAMS)
    
#     # 3. Generate Visualizations
#     plot_rewards(s_stats, q_stats)
#     visualize_cliff_policy(s_q, env, "SARSA (Safe Path in Stochastic Env)")
#     visualize_cliff_policy(q_q, env, "Q-Learning (Risky Path in Stochastic Env)")


import os
import sys
import pandas as pd

# Robust path setup to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from environment import CliffWalkingEnvironment
from algorithms import value_iteration, train_agent, train_double_q
from visualization import save_metrics_plot, save_heatmap, save_action_fear_map

if __name__ == "__main__":
    # --- PATH FIX START ---
    # 1. Get the directory where this script lives (src/)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 2. Get the project root (one level up from src/)
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # 3. Define absolute paths for output
    ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    
    # 4. Create directories if they don't exist
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Project Root detected at: {PROJECT_ROOT}")
    print(f"Saving artifacts to: {ASSETS_DIR}")
    # --- PATH FIX END ---

    env = CliffWalkingEnvironment(action_slip_prob=0.2, wind_prob=0.15)
    print(f"--- EXPERIMENT START: {env.grid_size}x{env.grid_size} Grid | Slip: 20% | Wind: 15% ---")

    # 2. Compute Theoretical Ground Truth
    print("\n[1/4] Computing Theoretical Ground Truth (Value Iteration)...")
    opt_Q, opt_V = value_iteration(env)
    
    # Save Baseline Artifacts using FIXED paths
    save_heatmap(opt_Q, env, "Theoretical Optimal Policy (V*)", os.path.join(ASSETS_DIR, "Figure_1_Theoretical.png"))
    pd.DataFrame(opt_Q.reshape(169, 4)).to_csv(os.path.join(DATA_DIR, "Theoretical_Q.csv"))
    
    # 3. Train Algorithms
    print("\n[2/4] Training Agents (5000 Episodes)...")
    params = {
        'num_episodes': 5000, 
        'max_steps': 200, 
        'alpha': 0.1, 
        'gamma': 0.99, 
        'epsilon_start': 1.0, 
        'epsilon_min': 0.05, 
        'epsilon_decay': 0.999
    }
    
    results = {}
    print("  > Training SARSA...")
    s_q, s_stats = train_agent(env, params, 'sarsa', opt_Q)
    results['SARSA'] = s_stats
    
    print("  > Training Q-Learning...")
    q_q, q_stats = train_agent(env, params, 'q_learning', opt_Q)
    results['Q-Learning'] = q_stats
    
    print("  > Training Double Q-Learning...")
    dq_q, dq_stats = train_double_q(env, params, opt_Q)
    results['Double-Q'] = dq_stats
    
    # 4. Generate Visualization Artifacts
    print("\n[3/4] Generating Visualization Artifacts...")
    save_metrics_plot(results, os.path.join(ASSETS_DIR, "Figure_2_Convergence.png"))
    
    # Policy Maps
    save_heatmap(s_q, env, "SARSA Final Policy (Safe)", os.path.join(ASSETS_DIR, "Figure_3_SARSA_Policy.png"))
    save_heatmap(q_q, env, "Q-Learning Final Policy (Risky)", os.path.join(ASSETS_DIR, "Figure_4_QLearning_Policy.png"))
    save_heatmap(dq_q, env, "Double Q-Learning Policy", os.path.join(ASSETS_DIR, "Figure_5_DoubleQ_Policy.png"))
    
    # Risk Analysis (Fear Maps for DOWN Action)
    save_action_fear_map(s_q, env, 1, "SARSA Risk Perception (Action: DOWN)", os.path.join(ASSETS_DIR, "Figure_6_SARSA_Fear.png"))
    save_action_fear_map(q_q, env, 1, "Q-Learning Risk Perception (Action: DOWN)", os.path.join(ASSETS_DIR, "Figure_7_QLearning_Fear.png"))
    
    # 5. Quantitative Validation
    print("\n[4/4] Quantitative Risk Validation (Console Check):")
    danger_state = (5, 3)
    action_down = 1
    
    th_val = opt_Q[danger_state][action_down]
    sarsa_val = s_q[danger_state][action_down]
    q_val = q_q[danger_state][action_down]
    
    print(f"  State {danger_state} | Action: DOWN")
    print(f"  > Theoretical Limit (Q*): {th_val:.2f}")
    print(f"  > Q-Learning Est. (Q*):   {q_val:.2f}")
    print(f"  > SARSA Est. (Q_pi):      {sarsa_val:.2f}")
    print(f"  > Risk Premium (Safety):  {abs(sarsa_val - th_val):.2f}")

    print(f"\nDone. All artifacts saved to:\n  {ASSETS_DIR}\n  {DATA_DIR}")