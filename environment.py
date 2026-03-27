"""
Utilidades del entorno CartPole-v1: creación, análisis, evaluación y baseline aleatorio.
"""

import numpy as np
import gymnasium as gym


def make_env(render_mode=None):
    """Crea y devuelve el entorno CartPole-v1."""
    return gym.make("CartPole-v1", render_mode=render_mode)


def describe_env(env):
    """Describe el entorno: espacios de estados/acciones, recompensas, límites."""
    obs_space = env.observation_space
    act_space = env.action_space

    info = {
        'observation_shape': obs_space.shape,
        'observation_low': obs_space.low,
        'observation_high': obs_space.high,
        'num_actions': act_space.n,
        'max_steps': env.spec.max_episode_steps,
    }

    state_names = [
        'Posición del carro',
        'Velocidad del carro',
        'Ángulo del palo',
        'Velocidad angular del palo',
    ]

    print("=" * 60)
    print("ANÁLISIS DEL ENTORNO: CartPole-v1")
    print("=" * 60)
    print(f"\nEspacio de observación: {obs_space}")
    print(f"  Forma: {obs_space.shape}")
    print(f"\n{'Variable':<30} {'Mínimo':>10} {'Máximo':>10}")
    print("-" * 50)
    for name, low, high in zip(state_names, obs_space.low, obs_space.high):
        print(f"  {name:<28} {low:>10.2f} {high:>10.2f}")

    print(f"\nEspacio de acciones: {act_space}")
    print(f"  0 = Empujar a la izquierda")
    print(f"  1 = Empujar a la derecha")

    print(f"\nRecompensa: +1 por cada paso que el palo se mantiene en pie")
    print(f"Máximo pasos por episodio: {env.spec.max_episode_steps}")
    print(f"\nCondiciones de terminación:")
    print(f"  - Ángulo del palo > ±12° (±0.2095 rad)")
    print(f"  - Posición del carro > ±2.4")
    print(f"  - Episodio alcanza {env.spec.max_episode_steps} pasos (truncado)")
    print("=" * 60)

    return info


def run_random_episode(env, seed=None, verbose=False):
    """Ejecuta un episodio con acciones aleatorias. Retorna recompensa total y pasos."""
    state, _ = env.reset(seed=seed)
    total_reward = 0
    steps = 0
    states_log = [state.copy()]

    while True:
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        states_log.append(state.copy())

        if terminated or truncated:
            break

    if verbose:
        print(f"\nEpisodio aleatorio (seed={seed}):")
        print(f"  Pasos: {steps}, Recompensa total: {total_reward}")
        print(f"\n  Primeros 10 estados:")
        print(f"  {'Paso':<6} {'Pos Carro':>10} {'Vel Carro':>10} {'Ángulo':>10} {'Vel Ang':>10}")
        print(f"  {'-'*46}")
        for i, s in enumerate(states_log[:10]):
            print(f"  {i:<6} {s[0]:>10.4f} {s[1]:>10.4f} {s[2]:>10.4f} {s[3]:>10.4f}")

    return total_reward, steps, states_log


def evaluate_agent(env, agent, n_episodes=100, seed=42):
    """Evalúa el agente entrenado (modo greedy, epsilon=0) durante n episodios."""
    rewards = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i)
        episode_reward = 0
        while True:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)
    return rewards


def evaluate_random(env, n_episodes=100, seed=42):
    """Evalúa un agente aleatorio como baseline durante n episodios."""
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
