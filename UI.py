import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from agent.dqn import DQN
from env.parking_env import ParkingEnv

# Page config
st.set_page_config(page_title="AI Parking System", layout="wide")

# Title + Description
st.title("🚗 Smart AI Parking Optimization System")
st.markdown(
    "This system uses Deep Q-Learning to intelligently assign optimal parking slots, "
    "minimizing congestion and maximizing efficiency."
)

# Load model and environment
@st.cache_resource
def load_model():
    env = ParkingEnv(size=4)
    model = DQN(16, 16)
    model.load_state_dict(torch.load("agent/model.pth", map_location=torch.device("cpu")))
    model.eval()
    return env, model

env, model = load_model()

# Sidebar controls
st.sidebar.header("Controls")
num_cars = st.sidebar.slider("Number of Cars to Simulate", 1, 10, 5)

# Main button
if st.button("🚀 Simulate Parking"):
    st.subheader("Simulation Results")
    
    total_reward = 0
    successful_parks = 0
    
    for car_num in range(num_cars):
        col1, col2, col3 = st.columns(3)
        
        # Reset environment for each car
        state, _ = env.reset()
        
        # ✅ FIX: flatten state
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
        
        # Model prediction
        with torch.no_grad():
            q_values = model(state_tensor).numpy().flatten()
        
        # Mask invalid actions
        valid_actions = (state.flatten() == 0)
        masked_q = np.where(valid_actions, q_values, -1e9)
        action = int(np.argmax(masked_q))
        
        # ✅ FIX: safe env.step()
        step_output = env.step(action)
        if len(step_output) == 5:
            next_state, reward, done, _, _ = step_output
        else:
            next_state, reward, done, _ = step_output
        
        # Calculate metrics
        row = action // env.size
        col_idx = action % env.size
        
        total_reward += reward
        if reward > 5:
            successful_parks += 1
        
        occupancy = np.mean(next_state) * 100
        distance = abs(row - env.entry_row) + abs(col_idx - env.entry_col)
        
        # 🟢 Column 1: Grid
        with col1:
            st.write(f"**Car {car_num + 1}**")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(next_state, cmap="RdYlGn_r", vmin=0, vmax=1)
            
            ax.set_xticks(range(env.size))
            ax.set_yticks(range(env.size))
            
            # Highlight selected slot
            circle = plt.Circle((col_idx, row), 0.3, color='blue', fill=False, linewidth=2)
            ax.add_patch(circle)
            
            st.pyplot(fig)
            plt.close(fig)  # ✅ prevent memory leak
        
        # 🟢 Column 2: Decision info
        with col2:
            st.metric("Selected Slot", f"({row}, {col_idx})")
            st.metric("Reward", f"{reward:.2f}")
            status = "✅ Success" if reward > 5 else "❌ Failed"
            st.write(status)
        
        # 🟢 Column 3: Analytics
        with col3:
            st.metric("Occupancy", f"{occupancy:.1f}%")
            st.metric("Distance from Entry", distance)
    
    # 📊 Summary Section
    st.divider()
    st.subheader("📊 Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Total Reward", f"{total_reward:.2f}")
    
    with summary_col2:
        st.metric("Successful Parks", f"{successful_parks}/{num_cars}")
    
    with summary_col3:
        success_rate = (successful_parks / num_cars * 100) if num_cars > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # 🔥 Bonus: Progress bar
    st.progress(successful_parks / num_cars)