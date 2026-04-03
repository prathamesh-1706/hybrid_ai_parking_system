import gradio as gr
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ✅ FIX for HF (no GUI backend)
import matplotlib.pyplot as plt
import io
from PIL import Image

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

from agent.dqn import DQN
from env.parking_env import ParkingEnv

# Initialize
env = ParkingEnv(size=4)

# ✅ Safe model loading
try:
    model = DQN(16, env.action_space.n)
    model.load_state_dict(torch.load("agent/model.pth", map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    model = None

state, _ = env.reset()
episode_reward = 0
episode_steps = 0


def visualize_grid(grid, action=None):
    """Visualize parking grid with matplotlib"""
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1)

    if action is not None:
        row = action // env.size
        col = action % env.size
        circle = plt.Circle((col, row), 0.3, color='blue', fill=False, linewidth=3)
        ax.add_patch(circle)
        ax.text(col, row, '✓', ha='center', va='center', color='blue', fontsize=16)

    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title("Parking Lot (🟩=Empty, 🟥=Occupied)", fontsize=14)

    plt.colorbar(im, ax=ax, label="0=Empty, 1=Occupied")
    plt.tight_layout()
    plt.close(fig)

    return fig

def step_fn():
    global state, episode_reward, episode_steps

    if model is None:
        return None, "❌ Model not loaded"

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = torch.argmax(model(state_tensor)).item()

    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    episode_reward += reward
    episode_steps += 1

    row = action // env.size
    col = action % env.size

    # Status indicator
    status = "✅ Success!" if reward > 5 else "⚠️ Occupied or Penalty"

    info = f"""
    ### Last Action
    - **Selected Slot**: {action} (Row: {row}, Col: {col})
    - **Reward**: {reward:.2f}
    - **Status**: {status}
    - **Episode Reward**: {episode_reward:.2f}
    - **Steps**: {episode_steps}
    - **Done**: {"🏁 Yes" if done else "▶️ No"}
    """

    fig = visualize_grid(state, action)
    img = fig_to_image(fig)

    return img, info


def reset_fn():
    global state, episode_reward, episode_steps
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0

    fig = visualize_grid(state)
    img = fig_to_image(fig)

    return img, "🔄 **Environment Reset Successfully!**"


# UI
with gr.Blocks(title="🚗 AI Parking System") as demo:
    gr.Markdown("# 🚗 Hybrid AI Parking System")
    gr.Markdown("*AI-powered optimal parking spot allocation using Deep Q-Learning*")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎮 Controls")
            reset_btn = gr.Button("🔄 Reset", size="lg", variant="secondary")
            step_btn = gr.Button("▶️ Next Step", size="lg", variant="primary")
        
        with gr.Column(scale=2):
            plot_output = gr.Image(label="Parking Visualization")

    info_output = gr.Markdown(label="Information")

    reset_btn.click(reset_fn, outputs=[plot_output, info_output])
    step_btn.click(step_fn, outputs=[plot_output, info_output])

    demo.load(reset_fn, outputs=[plot_output, info_output])

    gr.Markdown("""
    ---
    ### 📊 How it works:
    1. Click **Reset** to initialize a new parking lot
    2. Click **Next Step** to let AI assign the best parking spot
    3. Watch the reward and episode statistics update
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)