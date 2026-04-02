import torch
import torch.optim as optim
import random
import numpy as np

from env.parking_env import ParkingEnv
from agent.dqn import DQN

# Create environment
env = ParkingEnv(size=4)

# Model
model = DQN(16, env.action_space.n)
optimizer = optim.Adam(model.parameters(), lr=0.001)

gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

def choose_action(state):
    global epsilon
    if random.random() < epsilon:
        return env.action_space.sample()
    
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    q_values = model(state)
    return torch.argmax(q_values).item()

# Training loop
episode_rewards = []

for episode in range(300):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = choose_action(state)

        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        q_values = model(state_tensor)
        next_q = model(next_tensor).detach()

        target = q_values.clone()
        
        # FIX: Handle done flag correctly
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + gamma * torch.max(next_q)

        loss = torch.nn.functional.mse_loss(q_values, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    episode_rewards.append(episode_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episode {episode}, Epsilon {epsilon:.2f}, Avg Reward: {avg_reward:.2f}")

# Save model
torch.save(model.state_dict(), "agent/model.pth")
print("Training Done ✅")