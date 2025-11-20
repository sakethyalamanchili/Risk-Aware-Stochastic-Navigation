import numpy as np
import random
from typing import Tuple, Dict, Any

def create_q_table(env) -> np.ndarray:
    return np.zeros((env.grid_size, env.grid_size, 4))

def get_greedy_action(q_table: np.ndarray, state: Tuple[int, int]) -> int:
    r, c = state
    vals = q_table[r, c]
    # Random tie-breaking to avoid directional bias
    return np.random.choice(np.flatnonzero(vals == vals.max()))

def choose_action(q_table: np.ndarray, state: Tuple[int, int], epsilon: float) -> int:
    if random.random() < epsilon:
        return random.choice([0, 1, 2, 3])
    return get_greedy_action(q_table, state)

def value_iteration(env, gamma=0.99, theta=1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Computes Theoretical V* and Q* using Dynamic Programming."""
    V = np.zeros((env.grid_size, env.grid_size))
    Q = np.zeros((env.grid_size, env.grid_size, 4))
    
    print("  > Running Value Iteration...")
    while True:
        delta = 0
        for r in range(env.grid_size):
            for c in range(env.grid_size):
                if (r,c) in env.walls: continue
                v_old = V[r, c]
                for a in range(4):
                    transitions = env.get_transition_probabilities((r,c), a)
                    # Bellman Expectation Equation
                    Q[r,c,a] = sum([p * (rew + gamma * (V[ns] if not d else 0)) for p,ns,rew,d in transitions])
                V[r,c] = max(Q[r,c])
                delta = max(delta, abs(v_old - V[r,c]))
        if delta < theta: break
    return Q, V

def train_agent(env, params: Dict[str, Any], mode='sarsa', optimal_Q=None) -> Tuple[np.ndarray, Dict]:
    """Generic trainer for SARSA and Q-Learning."""
    q = create_q_table(env)
    stats = {'rewards': [], 'errors': [], 'steps': []}
    eps = params['epsilon_start']
    
    for _ in range(params['num_episodes']):
        s = env.reset()
        a = choose_action(q, s, eps)
        done, rew_sum, steps = False, 0, 0
        
        while not done and steps < params['max_steps']:
            ns, rew, done = env.step(a)
            na = choose_action(q, ns, eps)
            
            target = rew
            if not done:
                if mode == 'sarsa':
                    target += params['gamma'] * q[ns][na]
                elif mode == 'q_learning':
                    target += params['gamma'] * np.max(q[ns])
            
            # Update Rule
            q[s][a] += params['alpha'] * (target - q[s][a])
            s, a = ns, na
            rew_sum += rew; steps += 1
            
        stats['rewards'].append(rew_sum)
        stats['steps'].append(steps)
        
        # Calculate MSE if ground truth is available
        if optimal_Q is not None:
            stats['errors'].append(np.mean((q - optimal_Q)**2))
            
        eps = max(params['epsilon_min'], eps * params['epsilon_decay'])
        
    return q, stats

def train_double_q(env, params: Dict[str, Any], optimal_Q=None) -> Tuple[np.ndarray, Dict]:
    q1, q2 = create_q_table(env), create_q_table(env)
    stats = {'rewards': [], 'errors': [], 'steps': []}
    eps = params['epsilon_start']
    
    for _ in range(params['num_episodes']):
        s = env.reset()
        done, rew_sum, steps = False, 0, 0
        while not done and steps < params['max_steps']:
            a = choose_action(q1 + q2, s, eps)
            ns, rew, done = env.step(a)
            
            if np.random.rand() < 0.5:
                best_next = np.argmax(q1[ns])
                q1[s][a] += params['alpha'] * (rew + params['gamma'] * q2[ns][best_next] * (not done) - q1[s][a])
            else:
                best_next = np.argmax(q2[ns])
                q2[s][a] += params['alpha'] * (rew + params['gamma'] * q1[ns][best_next] * (not done) - q2[s][a])
                
            s = ns; rew_sum += rew; steps += 1
            
        stats['rewards'].append(rew_sum); stats['steps'].append(steps)
        if optimal_Q is not None:
            stats['errors'].append(np.mean(((q1+q2)/2 - optimal_Q)**2))
        eps = max(params['epsilon_min'], eps * params['epsilon_decay'])
        
    return (q1+q2)/2, stats