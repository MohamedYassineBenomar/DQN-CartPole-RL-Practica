"""
Utilidades: ReplayBuffer (eficiente en memoria para imágenes), funciones de plotting y GIF.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class ReplayBuffer:
    """
    Buffer de experiencia para observaciones de imágenes.
    Almacena LazyFrames de Gymnasium (eficiente en memoria).
    Las observaciones se mantienen como uint8 para ahorrar memoria.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([np.array(t[0]) for t in batch], dtype=np.uint8)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([np.array(t[3]) for t in batch], dtype=np.uint8)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def plot_learning_curves(rewards, losses, epsilons, window=50):
    """
    Curvas de aprendizaje: recompensa (con media móvil), pérdida, epsilon.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Recompensa por episodio + media móvil
    axes[0].plot(rewards, alpha=0.3, color='blue', label='Recompensa por episodio')
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(range(window - 1, len(rewards)), moving_avg, color='red',
                     linewidth=2, label=f'Media móvil ({window} ep.)')
    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_title('Recompensa por Episodio')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Pérdida
    if losses:
        axes[1].plot(losses, alpha=0.5, color='orange')
    axes[1].set_xlabel('Actualización')
    axes[1].set_ylabel('Pérdida (Huber)')
    axes[1].set_title('Pérdida durante Entrenamiento')
    axes[1].grid(True, alpha=0.3)

    # Epsilon decay
    axes[2].plot(epsilons, color='green')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Epsilon')
    axes[2].set_title('Decaimiento de Epsilon')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison_curves(rewards_dict, window=50):
    """Compara curvas de recompensa de múltiples agentes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for i, (name, rewards) in enumerate(rewards_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(rewards, alpha=0.15, color=color)
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(rewards)), moving_avg,
                    color=color, linewidth=2, label=f'{name} (media móvil)')

    ax.set_xlabel('Episodio')
    ax.set_ylabel('Recompensa')
    ax.set_title('Comparación de Curvas de Aprendizaje')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_evaluation(trained_rewards, random_rewards):
    """Boxplot comparativo entre agente entrenado y baseline aleatorio."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([trained_rewards, random_rewards],
                    labels=['Agente DQN', 'Baseline Aleatorio'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    ax.set_ylabel('Recompensa Total')
    ax.set_title('Evaluación: DQN vs Baseline Aleatorio')
    ax.grid(True, alpha=0.3, axis='y')

    for i, data_set in enumerate([trained_rewards, random_rewards]):
        stats_text = f'μ={np.mean(data_set):.1f}\nσ={np.std(data_set):.1f}'
        ax.text(i + 1, max(data_set) + 1, stats_text, ha='center', fontsize=9)

    plt.tight_layout()
    return fig


def create_agent_gif(agent, filepath='agent_pong.gif', n_frames=500):
    """Crea un GIF del agente entrenado jugando Pong."""
    import imageio
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
    import ale_py
    gym.register_envs(ale_py)

    # Env con rgb_array para capturar frames + preprocessing para el agente
    env_rgb = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    env_agent = AtariPreprocessing(env_rgb, noop_max=30, frame_skip=4,
                                   screen_size=84, terminal_on_life_loss=False,
                                   grayscale_obs=True, grayscale_newaxis=False,
                                   scale_obs=False)
    env_agent = FrameStackObservation(env_agent, stack_size=4)

    # Wrapper de acciones inline
    action_map = {0: 0, 1: 2, 2: 3}

    frames = []
    state, _ = env_agent.reset()

    for _ in range(n_frames):
        # Capturar frame RGB del render
        frame = env_agent.render()
        if frame is not None:
            frames.append(frame)

        agent_action = agent.select_action(state, epsilon=0.0)
        env_action = action_map[agent_action]
        state, _, terminated, truncated, _ = env_agent.step(env_action)

        if terminated or truncated:
            state, _ = env_agent.reset()

    env_agent.close()

    if frames:
        imageio.mimsave(filepath, frames, fps=30, loop=0)
        print(f"GIF guardado en: {filepath} ({len(frames)} frames)")

    return filepath
