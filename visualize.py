import torch
import time
import numpy as np

from agent.dqn import DQN
from env.parking_env import ParkingEnv

# Load environment
env = ParkingEnv(size=4)

# Load model
model = DQN(16, 16)
model.load_state_dict(torch.load("agent/model.pth"))
model.eval()


def print_grid(grid, action=None):
    print("\nParking Lot:\n")
    
    for i in range(len(grid)):
        row = ""
        for j in range(len(grid[i])):
            idx = i * len(grid) + j

            if action == idx:
                row += "🅿️ "   # AI chosen slot
            elif grid[i][j] == 1:
                row += "🚗 "   # occupied
            else:
                row += "⬜ "   # empty

        print(row)
    print("\n")


# Run demo with multiple cars
for i in range(5):  # simulate 5 cars
    print(f"\n🚗 Car {i+1} arriving...")

    state, _ = env.reset()
    print_grid(state)

    time.sleep(1)

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = torch.argmax(model(state_tensor)).item()

    print("AI is choosing best slot...\n")
    time.sleep(1)

    next_state, reward, done, _, _ = env.step(action)

    print("AI Assigned Slot:")
    print_grid(next_state, action)

    row = action // env.size
    col = action % env.size

    print(f"Chosen Slot: {action} (Row: {row}, Col: {col})")
    print(f"Reward: {reward:.2f}")
    print("-" * 40)

    time.sleep(1)

print("\n✅ Simulation Complete!")