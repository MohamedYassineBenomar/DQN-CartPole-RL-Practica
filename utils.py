"""
Utilidades para el proyecto DQN: ReplayBuffer, funciones de plotting y creación de GIFs.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """Buffer circular para almacenar transiciones (s, a, r, s', done)."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return (
            np.array(batch.state, dtype=np.float32),
            np.array(batch.action, dtype=np.int64),
            np.array(batch.reward, dtype=np.float32),
            np.array(batch.next_state, dtype=np.float32),
            np.array(batch.done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def plot_learning_curves(rewards, losses, epsilons, window=50):
    """
    Genera gráficas de curvas de aprendizaje: recompensa (con media móvil),
    pérdida y decaimiento de epsilon.
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

    # Pérdida por episodio
    axes[1].plot(losses, alpha=0.5, color='orange')
    axes[1].set_xlabel('Episodio')
    axes[1].set_ylabel('Pérdida (MSE)')
    axes[1].set_title('Pérdida durante Entrenamiento')
    axes[1].grid(True, alpha=0.3)

    # Epsilon decay
    axes[2].plot(epsilons, color='green')
    axes[2].set_xlabel('Episodio')
    axes[2].set_ylabel('Epsilon')
    axes[2].set_title('Decaimiento de Epsilon')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison_curves(rewards_dict, window=50):
    """
    Compara curvas de recompensa de múltiples agentes en un solo gráfico.
    rewards_dict: dict con nombre -> lista de recompensas.
    """
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
    data = [trained_rewards, random_rewards]
    bp = ax.boxplot(data, labels=['Agente DQN', 'Baseline Aleatorio'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    ax.set_ylabel('Recompensa Total')
    ax.set_title('Evaluación: DQN vs Baseline Aleatorio')
    ax.grid(True, alpha=0.3, axis='y')

    # Añadir estadísticas como texto
    for i, (data_set, name) in enumerate([(trained_rewards, 'DQN'), (random_rewards, 'Random')]):
        stats_text = f'μ={np.mean(data_set):.1f}\nσ={np.std(data_set):.1f}'
        ax.text(i + 1, np.max(data_set) + 5, stats_text, ha='center', fontsize=9)

    plt.tight_layout()
    return fig


def create_agent_gif(env, agent, filepath='agent_playing.gif', max_steps=500):
    """Crea un GIF del agente entrenado jugando un episodio."""
    import imageio

    frames = []
    state, _ = env.reset()
    frames.append(env.render())

    for _ in range(max_steps):
        action = agent.select_action(state, greedy=True)
        state, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break

    imageio.mimsave(filepath, frames, fps=30, loop=0)
    print(f"GIF guardado en: {filepath} ({len(frames)} frames)")
    return filepath
