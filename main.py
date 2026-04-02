from env.parking_env import ParkingEnv

env = ParkingEnv(size=4)

state, _ = env.reset()
print("Initial State:")
env.render()

action = env.action_space.sample()
print("Random Action:", action)

state, reward, done, _, _ = env.step(action)

print("After Action:")
env.render()
print("Reward:", reward)

from fastapi import FastAPI
import torch

from agent.dqn import DQN
from env.parking_env import ParkingEnv

app = FastAPI()

# Initialize environment
env = ParkingEnv(size=4)

# Load trained model
model = DQN(16, env.action_space.n)
model.load_state_dict(torch.load("agent/model.pth"))
model.eval()

# Store current state globally
current_state = None


# ✅ RESET endpoint
@app.post("/reset")
def reset():
    global current_state
    state, _ = env.reset()
    current_state = state

    return {
        "state": state.tolist()
    }


# ✅ STEP endpoint
@app.post("/step")
def step():
    global current_state

    state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
    action = torch.argmax(model(state_tensor)).item()

    next_state, reward, done, _, _ = env.step(action)
    current_state = next_state

    return {
        "action": action,
        "next_state": next_state.tolist(),
        "reward": reward,
        "done": done
    }


# ✅ STATE endpoint
@app.get("/state")
def get_state():
    global current_state

    return {
        "state": current_state.tolist() if current_state is not None else None
    }