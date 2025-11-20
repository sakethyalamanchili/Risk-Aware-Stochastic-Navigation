import numpy as np
from typing import Tuple, List, Set, Dict

class CliffWalkingEnvironment:
    """
    A tabular Grid World environment compliant with OpenAI Gym interfaces.
    
    Dynamics:
    - Grid: 13x13 (State space size: 169)
    - Actions: 4 (Up, Down, Left, Right)
    - Stochasticity: 20% Action Slip
    - Wind Force: 15% probability of downward displacement in 'Wind Zones'.
    """
    
    def __init__(self, action_slip_prob: float = 0.2, wind_prob: float = 0.15):
        self.grid_size = 13
        self.start_state = (11, 1)
        self.goal_state = (1, 11)
        self.action_slip_prob = action_slip_prob
        self.wind_prob = wind_prob
        
        self.walls = self._create_walls()
        self.cliff = {(6, c) for c in range(2, 6)}
        self.wind_zone = {(r, c) for r in [4, 5] for c in range(2, 6)}
        self.state = self.start_state
        self.actions = [0, 1, 2, 3] # Up, Down, Left, Right

    def _create_walls(self) -> Set[Tuple[int, int]]:
        walls = set()
        mid = self.grid_size // 2
        # Boundary walls
        for i in range(self.grid_size):
            walls.add((0, i)); walls.add((self.grid_size - 1, i))
            walls.add((i, 0)); walls.add((i, self.grid_size - 1))
        # Internal structure (Four Rooms)
        for i in range(1, self.grid_size - 1):
            if i not in [3, 9]: 
                walls.add((mid, i))
                walls.add((i, mid))
        walls.add((mid, mid))
        return walls

    def reset(self) -> Tuple[int, int]:
        self.state = self.start_state
        return self.state

    def _get_next_pos(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        r, c = state
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[action]
        return r + dr, c + dc

    def _constrain(self, r: int, c: int, old_r: int, old_c: int) -> Tuple[int, int]:
        """Enforces boundaries and wall collisions."""
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size): return old_r, old_c
        if (r, c) in self.walls: return old_r, old_c
        return r, c

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        curr_r, curr_c = self.state
        
        # 1. Stochastic Slip Dynamics
        if np.random.rand() < self.action_slip_prob:
            action = np.random.choice(self.actions)
            
        # 2. Deterministic Movement
        tr, tc = self._get_next_pos((curr_r, curr_c), action)
        next_r, next_c = self._constrain(tr, tc, curr_r, curr_c)
        
        # 3. Wind Force Dynamics
        if (next_r, next_c) in self.wind_zone and np.random.rand() < self.wind_prob:
            wr, wc = self._get_next_pos((next_r, next_c), 1) # Wind pushes DOWN
            next_r, next_c = self._constrain(wr, wc, next_r, next_c)

        # 4. Terminal Checks
        fell = (next_r, next_c) in self.cliff
        hit_wall = (next_r == curr_r and next_c == curr_c)
        
        if fell:
            self.state = self.start_state
            return self.state, -100.0, False
        
        self.state = (next_r, next_c)
        
        if self.state == self.goal_state:
            return self.state, 100.0, True
        
        return self.state, (-1.0 if hit_wall else -0.1), False

    def get_transition_probabilities(self, state: Tuple[int, int], action: int) -> List[Tuple[float, Tuple[int, int], float, bool]]:
        """Model logic for Value Iteration (P(s'|s,a))."""
        if state == self.goal_state: return [(1.0, state, 0.0, True)]
        if state in self.cliff: return [(1.0, self.start_state, 0.0, False)]
        
        outcomes = {}
        # Probability distribution over intended vs slipped actions
        p_act = {a: (self.action_slip_prob/4) for a in self.actions}
        p_act[action] += (1 - self.action_slip_prob)
        
        for act, p in p_act.items():
            tr, tc = self._get_next_pos(state, act)
            nr, nc = self._constrain(tr, tc, *state)
            
            # Wind branching
            branches = []
            if (nr, nc) in self.wind_zone:
                wr, wc = self._get_next_pos((nr, nc), 1)
                wr, wc = self._constrain(wr, wc, nr, nc)
                branches = [(1-self.wind_prob, nr, nc), (self.wind_prob, wr, wc)]
            else:
                branches = [(1.0, nr, nc)]
                
            for p_wind, fr, fc in branches:
                final_p = p * p_wind
                if (fr, fc) in self.cliff:
                    # Falling is terminal-like for the transition, but in this env it resets
                    outcomes[(self.start_state, -100.0, False)] = outcomes.get((self.start_state, -100.0, False), 0) + final_p
                elif (fr, fc) == self.goal_state:
                    outcomes[((fr, fc), 100.0, True)] = outcomes.get(((fr, fc), 100.0, True), 0) + final_p
                else:
                    rew = -1.0 if (fr,fc)==state else -0.1
                    outcomes[((fr, fc), rew, False)] = outcomes.get(((fr, fc), rew, False), 0) + final_p
                    
        return [(prob, ns, r, d) for (ns, r, d), prob in outcomes.items()]