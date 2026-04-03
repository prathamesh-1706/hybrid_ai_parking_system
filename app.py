import gradio as gr
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ✅ FIX for HF (no GUI backend)
import matplotlib.pyplot as plt

from agent.dqn import DQN
from env.parking_env import ParkingEnv

# Initialize
env = ParkingEnv(size=4)

# ✅ Safe model loading
try:
    model = DQN(16, env.action_space.n)
    model.load_state_dict(torch.load("agent/model.pth", map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    print("Model loading error:", e)
    model = None

state, _ = env.reset()


def visualize_grid(grid, action=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1)

    if action is not None:
        row = action // env.size
        col = action % env.size
        circle = plt.Circle((col, row), 0.3, color='blue', fill=False, linewidth=3)
        ax.add_patch(circle)

    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title("Parking Lot")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    plt.close(fig)   # ✅ ADD THIS LINE

    return fig

def step_fn():
    global state

    if model is None:
        return None, "❌ Model not loaded"

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = torch.argmax(model(state_tensor)).item()

    next_state, reward, done, _, _ = env.step(action)
    state = next_state

    row = action // env.size
    col = action % env.size

    info = f"""
    ✅ Action: {action}  
    📍 Row: {row}, Col: {col}  
    💰 Reward: {reward:.2f}  
    🏁 Done: {done}
    """

    fig = visualize_grid(state, action)
    return fig, info


def reset_fn():
    global state
    state, _ = env.reset()

    fig = visualize_grid(state)
    return fig, "🔄 Environment Reset"


# UI
with gr.Blocks() as demo:
    gr.Markdown("# 🚗 Hybrid AI Parking System")

    with gr.Row():
        reset_btn = gr.Button("Reset")
        step_btn = gr.Button("Next Step")

    plot_output = gr.Plot()
    info_output = gr.Markdown()

    reset_btn.click(reset_fn, outputs=[plot_output, info_output])
    step_btn.click(step_fn, outputs=[plot_output, info_output])

    demo.load(reset_fn, outputs=[plot_output, info_output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)