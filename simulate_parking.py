import os
import requests
import time
import random  # if you want random actions

BASE_URL = os.environ.get("PARKING_API_URL", "http://localhost:7860")
TIMEOUT = 5  # seconds

# -------------------------------
# Function definitions
# -------------------------------
def print_state(state):
    print("Parking Lot State:")
    for row in state:
        print(" ".join(str(cell) for cell in row))
    print("-" * 20)

def reset_env(session):
    try:
        resp = session.get(f"{BASE_URL}/reset", timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state")
        if state is None:
            raise ValueError("Missing 'state' in reset response")
        return state
    except Exception as e:
        print(f"Error resetting environment: {e}")
        return None

def step_env(session, action):
    try:
        resp = session.post(f"{BASE_URL}/step", json={"action": action}, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        next_state = data.get("state")
        reward = data.get("reward")
        done = data.get("done")
        return next_state, reward, done
    except Exception as e:
        print(f"Error stepping environment: {e}")
        return None, None, True

# -------------------------------
# Main simulation loop
# -------------------------------
def main():
    session = requests.Session()
    state = reset_env(session)
    if state is None:
        return
    print_state(state)
    
    done = False
    while not done:
        action = random.randint(0, 3)  # automatic random action, adjust range to your env
        next_state, reward, done = step_env(session, action)
        if next_state is None:
            break
        print(f"Action taken: {action}, Reward: {reward}, Done: {done}")
        print_state(next_state)

if __name__ == "__main__":
    main()