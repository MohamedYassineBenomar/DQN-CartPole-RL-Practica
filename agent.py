"""
Agente DQN con soporte para Double DQN.
Incluye QNetwork (red neuronal), DQNAgent (agente completo) y DQNConfig (hiperparámetros).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from utils import ReplayBuffer


@dataclass
class DQNConfig:
    """Configuración de hiperparámetros del agente DQN."""
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 10_000
    batch_size: int = 64
    target_update_freq: int = 10
    hidden_dims: tuple = (128, 128)
    max_episodes: int = 600
    min_buffer_size: int = 1000
    grad_clip: float = 1.0
    use_double_dqn: bool = False


class QNetwork(nn.Module):
    """
    Red neuronal para aproximar Q(s, a).
    Arquitectura: Linear → ReLU → Linear → ReLU → Linear
    """

    def __init__(self, state_dim=4, action_dim=2, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    Agente DQN completo con Experience Replay, Target Network y política ε-greedy.
    Soporta tanto DQN estándar como Double DQN.
    """

    def __init__(self, state_dim, action_dim, config=None):
        if config is None:
            config = DQNConfig()

        self.config = config
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Redes Q y Target
        self.q_network = QNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)

        # Experience Replay Buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)

        # Epsilon para política ε-greedy
        self.epsilon = config.epsilon_start

    def select_action(self, state, greedy=False):
        """Selecciona una acción usando política ε-greedy (o greedy si se indica)."""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()

    def update(self, batch_size=None):
        """Realiza un paso de actualización de la red Q usando un minibatch del buffer."""
        if batch_size is None:
            batch_size = self.config.batch_size

        if len(self.replay_buffer) < batch_size:
            return 0.0

        # Muestrear del buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-valores actuales: Q(s, a)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calcular targets
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: la red online selecciona la acción, la target la evalúa
                best_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, best_actions).squeeze(1)
            else:
                # DQN estándar: max Q del target network
                next_q = self.target_network(next_states).max(dim=1)[0]

            targets = rewards + self.config.gamma * next_q * (1 - dones)

        # Pérdida y backpropagation
        loss = F.mse_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copia los pesos de la red Q a la target network (hard update)."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decae epsilon multiplicativamente hasta el mínimo."""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def save(self, filepath):
        """Guarda los pesos del modelo entrenado."""
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Modelo guardado en: {filepath}")

    def load(self, filepath):
        """Carga los pesos de un modelo entrenado."""
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"Modelo cargado desde: {filepath}")
