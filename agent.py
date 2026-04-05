"""
Agente DQN con CNN para Atari Pong.
Arquitectura Nature DQN (2015) con soporte para Double DQN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from utils import ReplayBuffer


@dataclass
class DQNConfig:
    """Hiperparámetros del agente DQN para Atari Pong."""
    lr: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 100_000
    buffer_size: int = 100_000
    batch_size: int = 32
    target_update_freq: int = 1_000
    total_steps: int = 500_000
    min_buffer_size: int = 10_000
    train_frequency: int = 4
    grad_clip: float = 10.0
    use_double_dqn: bool = True
    action_dim: int = 3


class QNetworkCNN(nn.Module):
    """
    Red neuronal convolucional para aproximar Q(s, a) — Arquitectura Nature DQN.

    Input:  (batch, 4, 84, 84) uint8 → normalizado a float32 [0, 1]
    Conv1:  Conv2d(4, 32, 8, stride=4) + ReLU  → (batch, 32, 20, 20)
    Conv2:  Conv2d(32, 64, 4, stride=2) + ReLU → (batch, 64, 9, 9)
    Conv3:  Conv2d(64, 64, 3, stride=1) + ReLU → (batch, 64, 7, 7)
    FC1:    Linear(3136, 512) + ReLU
    FC2:    Linear(512, action_dim) → Q-valores
    """

    def __init__(self, action_dim=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        # Normalizar uint8 [0,255] → float32 [0,1]
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DQNAgent:
    """
    Agente DQN con CNN, Experience Replay, Target Network y política ε-greedy.
    Soporta DQN estándar y Double DQN.
    """

    def __init__(self, config=None):
        if config is None:
            config = DQNConfig()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Redes Q y Target
        self.q_network = QNetworkCNN(config.action_dim).to(self.device)
        self.target_network = QNetworkCNN(config.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)

        # Experience Replay
        self.replay_buffer = ReplayBuffer(config.buffer_size)

    def get_epsilon(self, step):
        """Schedule lineal de epsilon: de epsilon_start a epsilon_end en epsilon_decay_steps."""
        fraction = min(step / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def select_action(self, state, epsilon=None):
        """Selecciona acción con política ε-greedy."""
        if epsilon is None:
            epsilon = self.config.epsilon_end

        if np.random.random() < epsilon:
            return np.random.randint(self.config.action_dim)

        state_array = np.array(state)
        state_tensor = torch.as_tensor(state_array, dtype=torch.uint8).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()

    def update(self):
        """Actualiza la red Q con un minibatch del buffer."""
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        states = torch.as_tensor(states, dtype=torch.uint8).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.uint8).to(self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32).to(self.device)

        # Q-valores actuales
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calcular targets
        with torch.no_grad():
            if self.config.use_double_dqn:
                best_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, best_actions).squeeze(1)
            else:
                next_q = self.target_network(next_states).max(dim=1)[0]
            targets = rewards + self.config.gamma * next_q * (1 - dones)

        # Huber loss + backprop
        loss = F.smooth_l1_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copia los pesos de la red Q a la target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath):
        """Guarda los pesos del modelo."""
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Modelo guardado en: {filepath}")

    def load(self, filepath):
        """Carga los pesos de un modelo entrenado."""
        self.q_network.load_state_dict(
            torch.load(filepath, map_location=self.device, weights_only=True)
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"Modelo cargado desde: {filepath}")

    def save_checkpoint(self, filepath, step, episode, rewards_history, losses_history, epsilons_history):
        """Guarda checkpoint completo para reanudar entrenamiento."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step,
            'episode': episode,
            'rewards_history': rewards_history,
            'losses_history': losses_history,
            'epsilons_history': epsilons_history,
        }, filepath)
        print(f"Checkpoint guardado en: {filepath} (step {step})")

    def load_checkpoint(self, filepath):
        """Carga checkpoint completo para reanudar entrenamiento."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint cargado desde: {filepath} (step {checkpoint['step']})")
        return checkpoint
