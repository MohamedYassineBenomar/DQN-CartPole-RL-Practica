"""Entrena el agente DQN en Atari Pong.

Ejemplos:
    python3 train.py
    python3 train.py --total-steps 200000 --render
    python3 train.py --double-dqn --checkpoint-every 25000
    python3 train.py --resume models/dqn_pong_checkpoint.pth
"""

import argparse
import os
import random
import time

import numpy as np
import torch

from agent import DQNAgent, DQNConfig
from environment import make_pong_env


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN agent on Atari Pong")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50_000, help="Save every N steps")
    parser.add_argument("--model-path", type=str, default="models/dqn_pong.pth")
    parser.add_argument("--double-dqn", action="store_true", default=True)
    parser.add_argument("--no-double-dqn", dest="double_dqn", action="store_false")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--min-buffer", type=int, default=10_000)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)

    env = make_pong_env(render_mode="human" if args.render else None)
    config = DQNConfig(
        lr=args.lr,
        total_steps=args.total_steps,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
        train_frequency=args.train_freq,
        epsilon_decay_steps=args.epsilon_decay_steps,
        use_double_dqn=args.double_dqn,
    )
    agent = DQNAgent(config=config)

    print(f"Training DQN on Pong | Device: {agent.device}")
    print(f"  steps={args.total_steps}, double_dqn={args.double_dqn}, lr={args.lr}")
    print(f"  buffer={args.buffer_size}, batch={args.batch_size}, train_freq={args.train_freq}")

    # Resume from checkpoint if specified
    start_step = 0
    episode_count = 0
    rewards_history = []
    losses_history = []
    epsilons_history = []

    if args.resume and os.path.exists(args.resume):
        checkpoint = agent.load_checkpoint(args.resume)
        start_step = checkpoint['step']
        episode_count = checkpoint['episode']
        rewards_history = checkpoint['rewards_history']
        losses_history = checkpoint['losses_history']
        epsilons_history = checkpoint.get('epsilons_history', [])
        print(f"Resuming from step {start_step}, episode {episode_count}")

    state, _ = env.reset(seed=args.seed)
    episode_reward = 0
    episode_start_time = time.time()

    try:
        for step in range(start_step, config.total_steps):
            epsilon = agent.get_epsilon(step)

            # Select and execute action
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            # Episode ended
            if done:
                rewards_history.append(episode_reward)
                epsilons_history.append(epsilon)
                episode_count += 1
                elapsed = time.time() - episode_start_time

                if episode_count % 5 == 0:
                    avg = np.mean(rewards_history[-100:]) if rewards_history else 0
                    print(f"Step {step:>7d} | Ep {episode_count:>4d} | "
                          f"Reward: {episode_reward:>5.0f} | Avg100: {avg:>6.1f} | "
                          f"Eps: {epsilon:.4f} | {elapsed:.1f}s")

                episode_reward = 0
                episode_start_time = time.time()
                state, _ = env.reset()

            # Train
            if step >= config.min_buffer_size and step % config.train_frequency == 0:
                loss = agent.update()
                if loss > 0:
                    losses_history.append(loss)

            # Update target network
            if step % config.target_update_freq == 0 and step > 0:
                agent.update_target_network()

            # Save checkpoint
            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0 and step > start_step:
                checkpoint_path = args.model_path.replace('.pth', '_checkpoint.pth')
                agent.save_checkpoint(checkpoint_path, step, episode_count,
                                      rewards_history, losses_history, epsilons_history)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Save final model
        agent.save(args.model_path)
        checkpoint_path = args.model_path.replace('.pth', '_checkpoint.pth')
        agent.save_checkpoint(checkpoint_path, step, episode_count,
                              rewards_history, losses_history, epsilons_history)
        env.close()

        print(f"\nTraining complete!")
        print(f"  Total steps: {step}")
        print(f"  Total episodes: {episode_count}")
        if rewards_history:
            print(f"  Final avg reward (100 ep): {np.mean(rewards_history[-100:]):.1f}")
        print(f"  Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()
