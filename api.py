from fastapi import FastAPI
import torch
import numpy as np
from agent.dqn import DQN
from env.parking_env import ParkingEnv

app = FastAPI()

env = ParkingEnv(size=4)
model = DQN(16, env.action_space.n)
model.load_state_dict(torch.load("agent/model.pth", map_location="cpu"))
model.eval()

state, _ = env.reset()

@app.api_route("/reset", methods=["GET", "POST"])
def reset():
    global state
    state, _ = env.reset()
    return {"state": state.tolist()}

@app.api_route("/step", methods=["GET", "POST"])

# Root route
@app.get("/")
def read_root():
    return {"message": "Hybrid AI Parking System is running!"}

@app.api_route("/step", methods=["GET", "POST"])

def step():
    global state

    state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor).numpy().flatten()

    valid_actions = (state.flatten() == 0)

    if not valid_actions.any():
        return {"done": True, "reward": 0}

    masked_q = np.where(valid_actions, q_values, -1e9)
    action = int(np.argmax(masked_q))

    step_output = env.step(action)

    if len(step_output) == 5:
        next_state, reward, done, _, _ = step_output
    else:
        next_state, reward, done, _ = step_output

    state = next_state

    return {
        "state": state.tolist(),
        "reward": float(reward),
        "done": bool(done),
        "action": action
    }