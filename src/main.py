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