import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from agent.dqn import DQN
from env.parking_env import ParkingEnv

st.set_page_config(page_title="AI Parking System", layout="wide")
st.title("🚗 AI Parking System")

# Load model and environment
@st.cache_resource
def load_model():
    env = ParkingEnv(size=4)
    model = DQN(16, 16)
    model.load_state_dict(torch.load("agent/model.pth"))
    model.eval()
    return env, model

env, model = load_model()

# Sidebar controls
st.sidebar.header("Controls")
num_cars = st.sidebar.slider("Number of Cars to Simulate", 1, 10, 5)

if st.button("🚀 Simulate Parking", key="main_button"):
    st.subheader("Simulation Results")
    
    total_reward = 0
    successful_parks = 0
    
    for car_num in range(num_cars):
        col1, col2, col3 = st.columns(3)
        
        state, _ = env.reset()
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = torch.argmax(model(state_tensor)).item()
        
        next_state, reward, done, _, _ = env.step(action)
        
        row = action // env.size
        col_idx = action % env.size
        
        total_reward += reward
        if reward > 5:
            successful_parks += 1
        
        # Display parking grid
        with col1:
            st.write(f"**Car {car_num + 1}**")
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(next_state, cmap="RdYlGn_r", vmin=0, vmax=1)
            ax.set_xticks(range(env.size))
            ax.set_yticks(range(env.size))
            
            # Highlight selected slot
            circle = plt.Circle((col_idx, row), 0.3, color='blue', fill=False, linewidth=3)
            ax.add_patch(circle)
            
            plt.colorbar(im, ax=ax, label="0=Empty, 1=Occupied")
            st.pyplot(fig)
        
        # Display info
        with col2:
            st.metric("Selected Slot", f"({row}, {col_idx})")
            st.metric("Reward", f"{reward:.2f}")
            status = "✅ Success" if reward > 5 else "❌ Failed"
            st.write(status)
        
        with col3:
            occupancy = np.sum(next_state) / env.total_slots * 100
            st.metric("Occupancy", f"{occupancy:.1f}%")
            distance = abs(row - env.entry_row) + abs(col_idx - env.entry_col)
            st.metric("Distance from Entry", distance)
    
    # Summary
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