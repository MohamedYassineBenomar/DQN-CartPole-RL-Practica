"""
Entorno Atari Pong (ALE/Pong-v5) con preprocessing para DQN.
Pipeline: RGB(210x160x3) → Grayscale(84x84) → FrameStackObservation(4) → ActionMapping(6→3)
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

# Registrar entornos ALE en Gymnasium
gymnasium = gym
gymnasium.register_envs(ale_py)


class PongActionWrapper(gym.ActionWrapper):
    """
    Reduce el espacio de acciones de Pong de 6 a 3 acciones útiles.
    0 → NOOP (no hacer nada)
    1 → UP (mover arriba)
    2 → DOWN (mover abajo)
    """
    _action_map = {0: 0, 1: 2, 2: 3}

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, act):
        return self._action_map[int(act)]


def make_pong_env(render_mode=None, terminal_on_life_loss=True):
    """
    Crea el entorno Pong con el pipeline completo de preprocessing.

    Pipeline:
    1. ALE/Pong-v5 (210x160x3 RGB)
    2. AtariPreprocessing: grayscale, resize 84x84, frame skip 4, max-pool
    3. FrameStackObservation: apilar 4 frames → (4, 84, 84)
    4. PongActionWrapper: 6 acciones → 3 (NOOP, UP, DOWN)

    Returns:
        env con observation_space=(4, 84, 84) uint8, action_space=Discrete(3)
    """
    env = gym.make("ALE/Pong-v5", render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    env = PongActionWrapper(env)
    return env


def describe_env(env):
    """Describe el entorno Pong: espacios, recompensas, acciones."""
    print("=" * 60)
    print("ANÁLISIS DEL ENTORNO: ALE/Pong-v5")
    print("=" * 60)
    print(f"\nObservación (preprocesada): {env.observation_space}")
    print(f"  Forma: {env.observation_space.shape}")
    print(f"  Tipo: uint8 [0, 255]")
    print(f"  4 frames apilados de 84x84 píxeles (grayscale)")
    print(f"\nEspacio de acciones: {env.action_space}")
    print(f"  0 = NOOP (no hacer nada)")
    print(f"  1 = UP (mover arriba)")
    print(f"  2 = DOWN (mover abajo)")
    print(f"\nRecompensa:")
    print(f"  +1 cuando la bola pasa al oponente")
    print(f"  -1 cuando la bola te pasa a ti")
    print(f"  0 en cualquier otro momento")
    print(f"\nPartida: primer jugador en llegar a 21 puntos gana")
    print(f"  Rango de recompensa total: [-21, +21]")
    print("=" * 60)

    return {
        'observation_shape': env.observation_space.shape,
        'num_actions': env.action_space.n,
        'reward_range': (-21, 21),
    }


def evaluate_agent(env, agent, n_episodes=30, seed=42):
    """Evalúa el agente entrenado (greedy) durante n episodios completos."""
    rewards = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i)
        episode_reward = 0
        while True:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)
    return rewards


def evaluate_random(env, n_episodes=30, seed=42):
    """Evalúa un agente aleatorio como baseline."""
    rewards = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i)
        episode_reward = 0
        while True:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)
    return rewards
