import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ParkingEnv(gym.Env):
    def __init__(self, size=4, max_steps=10):
        super().__init__()

        self.size = size
        self.total_slots = size * size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Entry point (top-left corner)
        self.entry_row = 0
        self.entry_col = 0

        # Action: choose a slot
        self.action_space = spaces.Discrete(self.total_slots)

        # Observation: parking grid
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.size, self.size), dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 0 = empty, 1 = occupied
        self.grid = np.random.choice([0, 1], size=(self.size, self.size), p=[0.7, 0.3])
        self.current_step = 0

        return self.grid.copy(), {}

    def step(self, action):
        row = action // self.size
        col = action % self.size

        reward = -0.1
        done = False
        
        self.current_step += 1

        if self.grid[row][col] == 0:
            self.grid[row][col] = 1
            reward = 10
            
            # Penalize far slots from entry point
            distance = abs(row - self.entry_row) + abs(col - self.entry_col)
            reward -= 0.5 * distance
        else:
            reward = -5

        # Episode ends when max steps reached or all slots full
        if self.current_step >= self.max_steps or np.all(self.grid == 1):
            done = True

        return self.grid.copy(), reward, done, False, {}

    def render(self):
        print("\nParking Grid (0=empty, 1=occupied):")
        print(self.grid)
        occupancy = np.sum(self.grid) / self.total_slots * 100
        print(f"Occupancy: {occupancy:.1f}%")
        print(f"Entry Point: ({self.entry_row}, {self.entry_col})")
        print(f"Current Step: {self.current_step}/{self.max_steps}")
        print("\n")
        