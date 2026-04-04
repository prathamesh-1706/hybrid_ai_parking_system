import gradio as gr
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image

from agent.dqn import DQN
from env.parking_env import ParkingEnv

# Initialize
env = ParkingEnv(size=4)

# Load model
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


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    plt.close(fig)  # ✅ prevent memory leak
    return np.array(img)


def visualize_grid(grid, action=None):
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1)

    if action is not None:
        row = action // env.size
        col = action % env.size
        ax.add_patch(plt.Circle((col, row), 0.3, color='blue', fill=False, linewidth=2))
        ax.text(col, row, '✓', ha='center', va='center', color='blue', fontsize=14)

    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title("Parking Lot")

    return fig


def step_fn():
    global state, episode_reward, episode_steps

    try:
        if model is None:
            return None, "❌ Model not loaded"

        # ✅ FIXED indentation + flatten
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor).cpu().numpy().flatten()

        valid_actions = (state.flatten() == 0)

        if not valid_actions.any():
            fig = visualize_grid(state)
            return fig_to_image(fig), "🚫 No available parking slots!"

        masked_q = np.where(valid_actions, q_values, -1e9)
        action = int(np.argmax(masked_q))

        # Safe env step
        step_output = env.step(action)

        if len(step_output) == 5:
            next_state, reward, done, _, _ = step_output
        else:
            next_state, reward, done, _ = step_output

        state = next_state
        episode_reward += reward
        episode_steps += 1

        row = action // env.size
        col = action % env.size

        status = "✅ Success!" if reward > 5 else "⚠️ Penalty"

        info = f"""
### 🤖 AI Decision
- Slot: {action} (Row {row}, Col {col})
- Reward: {reward:.2f}
- Total Reward: {episode_reward:.2f}
- Steps: {episode_steps}
- Done: {done}
"""

        fig = visualize_grid(state, action)
        return fig_to_image(fig), info

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        return None, f"❌ ERROR:\n```\n{error_msg}\n```"


def reset_fn():
    global state, episode_reward, episode_steps

    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0

    fig = visualize_grid(state)
    return fig_to_image(fig), "🔄 Reset successful"


# UI
with gr.Blocks() as demo:
    gr.Markdown("# 🚗 Hybrid AI Parking System")

    with gr.Row():
        with gr.Column():
            reset_btn = gr.Button("🔄 Reset")
            step_btn = gr.Button("▶️ Next Step")

        with gr.Column():
            plot_output = gr.Image(type="numpy")

    info_output = gr.Markdown()

    reset_btn.click(reset_fn, outputs=[plot_output, info_output])
    step_btn.click(step_fn, outputs=[plot_output, info_output])

    demo.load(reset_fn, outputs=[plot_output, info_output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)