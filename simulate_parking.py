import os
import requests
import time
import random

# Base URL for your FastAPI container
BASE_URL = os.environ.get("PARKING_API_URL", "http://localhost:7860")
TIMEOUT = 5  # seconds


def print_state(state):
    print("Parking Lot State:")
    for row in state:
        print(" ".join(str(cell) for cell in row))
    print("-" * 20)


def reset_env(session):
    """Reset the parking lot environment via API"""
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
    """Take a step/action in the environment via API"""
    try:
        resp = session.post(f"{BASE_URL}/step", json={"action": action}, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state")
        reward = data.get("reward")
        done = data.get("done")
        return state, reward, done
    except Exception as e:
        print(f"Error performing step: {e}")
        return None, None, None


def main():
    session = requests.Session()

    # Reset the environment first
    state = reset_env(session)
    if state is None:
        return

    print("Initial State:")
    print_state(state)

    done = False
    step_count = 0
    MAX_STEPS = 20  # max steps to run

    while not done and step_count < MAX_STEPS:
        # Random action for demonstration (0 to 15 for 4x4 parking lot)
        action = random.randint(0, 15)

        # Take the step
        state, reward, done = step_env(session, action)
        if state is None:
            break

        step_count += 1
        print(f"Step {step_count} - Action: {action}, Reward: {reward}, Done: {done}")
        print_state(state)
        time.sleep(1)  # pause so you can see each step


if __name__ == "__main__":
    main()