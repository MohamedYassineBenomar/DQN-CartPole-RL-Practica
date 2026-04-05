"""Visualiza al agente DQN jugando Pong.

Ejemplos:
    python3 watch.py
    python3 watch.py --model-path models/dqn_pong.pth --games 5
    python3 watch.py --games 0   (infinito)
"""

import argparse
import os
import time

from agent import DQNAgent, DQNConfig
from environment import make_pong_env


def parse_args():
    parser = argparse.ArgumentParser(description="Watch DQN agent play Pong")
    parser.add_argument("--model-path", type=str, default="models/dqn_pong.pth")
    parser.add_argument("--games", type=int, default=5, help="Games to play (0=infinite)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        print("Train first with: python3 train.py")
        return

    env = make_pong_env(render_mode="human", terminal_on_life_loss=False)
    agent = DQNAgent(config=DQNConfig())
    agent.load(args.model_path)

    print(f"Watching agent play Pong... (close window to quit)")

    game_num = 0
    while args.games == 0 or game_num < args.games:
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(0.01)
            if terminated or truncated:
                break

        game_num += 1
        print(f"Game {game_num}: Score = {total_reward:.0f}")
        time.sleep(0.5)

    env.close()


if __name__ == "__main__":
    main()
