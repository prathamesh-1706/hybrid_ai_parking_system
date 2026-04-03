import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt

from agent.dqn import DQN
from env.parking_env import ParkingEnv

# Initialize
env = ParkingEnv(size=4)
model = DQN(16, env.action_space.n)
model.load_state_dict(torch.load("agent/model.pth"))
model.eval()

state, _ = env.reset()

def visualize_grid(grid, action=None):
    """Visualize parking grid with matplotlib"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create heatmap
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1)
    
    # Highlight selected action
    if action is not None:
        row = action // env.size
        col = action % env.size
        circle = plt.Circle((col, row), 0.3, color='blue', fill=False, linewidth=3)
        ax.add_patch(circle)
    
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title("Parking Lot (🟩=Empty, 🟥=Occupied, 🔵=Selected)")
    
    plt.colorbar(im, ax=ax, label="0=Empty, 1=Occupied")
    plt.tight_layout()
    
    return fig

def step_fn():
    global state
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = torch.argmax(model(state_tensor)).item()

    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    
    row = action // env.size
    col = action % env.size
    
    info = f"""
    ✅ **Action Executed**
    - Selected Slot: {action} (Row: {row}, Col: {col})
    - Reward: {reward:.2f}
    - Episode Done: {done}
    """
    
    fig = visualize_grid(state, action)
    
    return fig, info

def reset_fn():
    global state
    state, _ = env.reset()
    
    fig = visualize_grid(state)
    info = "🔄 **Environment Reset Successfully!**"
    
    return fig, info

# Gradio Interface
with gr.Blocks(title="🚗 AI Parking System") as demo:
    gr.Markdown("# 🚗 Hybrid AI Parking System")
    gr.Markdown("AI-powered optimal parking spot allocation using Deep Q-Learning")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Controls")
            reset_btn = gr.Button("🔄 Reset Environment", size="lg")
            step_btn = gr.Button("▶️ Next Step", size="lg", variant="primary")
        
        with gr.Column():
            gr.Markdown("### Parking Visualization")
            plot_output = gr.Plot(label="Parking Grid")
    
    info_output = gr.Markdown()
    
    # Button actions
    reset_btn.click(reset_fn, outputs=[plot_output, info_output])
    step_btn.click(step_fn, outputs=[plot_output, info_output])
    
    # Initial display
    demo.load(reset_fn, outputs=[plot_output, info_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)