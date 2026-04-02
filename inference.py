import requests
import time
import json

BASE_URL = "https://pixelpirates1-hybrid-ai-parking.hf.space"

def run_episode():
    print("[START] Running parking simulation episode...")

    try:
        # Reset environment
        res = requests.post(f"{BASE_URL}/reset")
        res.raise_for_status()
        state = res.json()["state"]
        print("[RESET] Environment reset successfully")

        total_reward = 0
        step_count = 0

        for step in range(10):
            res = requests.post(f"{BASE_URL}/step")
            res.raise_for_status()
            data = res.json()

            reward = data.get("reward", 0)
            done = data.get("done", False)
            action = data.get("action", "N/A")

            total_reward += reward
            step_count += 1

            print(f"[STEP {step+1}] Action: {action} | Reward: {reward:.2f} | Total: {total_reward:.2f}")

            if done:
                print("[DONE] Episode ended early")
                break

            time.sleep(0.2)

        print(f"\n[END] Episode Complete")
        print(f"  Total Steps: {step_count}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Avg Reward per Step: {total_reward/step_count:.2f}")

    except requests.exceptions.ConnectionError:
        print("[ERROR] Failed to connect to API. Make sure server is running!")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
    except json.JSONDecodeError:
        print("[ERROR] Invalid JSON response from server")


def run_multiple_episodes(num_episodes=5):
    print(f"[START] Running {num_episodes} episodes...\n")
    
    all_rewards = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")
        run_episode()
        time.sleep(1)
    
    print(f"\n[SUMMARY] All episodes completed!")


if __name__ == "__main__":
    # Run single episode
    run_episode()
    
    # Uncomment to run multiple episodes
    # run_multiple_episodes(5)