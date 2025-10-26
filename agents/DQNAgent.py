"""Deep Q-Network (DQN) Agent"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from typing import List
from config import base_config, experiement_config as exp_config
from .agent_protocol import AgentProtocol
from experience import ReplayBuffer


class DQNNetwork(nn.Module):
    """3-layer Deep Q-Network"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()

        # Encapsulate all layers in nn.Sequential
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                # Works well with ReLU (xavier_uniform), no vanishing gradients problem? check
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)


class DQNAgent(AgentProtocol):
    """Deep Q-Network Agent"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 32,
        memory_size: int = 10000,
        target_update_freq: int = 100,
        hidden_size: int = 128,
        device: str = None,
    ):
        """
        Initialize DQN Agent

        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            batch_size: Batch size for training
            memory_size: Size of replay buffer
            target_update_freq: Frequency to update target network
            hidden_size: Hidden layer size
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Experience replay buffer
        self.memory = ReplayBuffer(memory_size)

        # Training tracking
        self.step_count = 0
        self.loss_history = []

    def _state_to_tensor(self, state: int) -> torch.Tensor:
        """Convert state to tensor"""
        # Convert state index to one-hot encoding
        state_tensor = torch.zeros(self.state_size, dtype=torch.float32)
        state_tensor[state] = 1.0
        return state_tensor.unsqueeze(0).to(self.device)

    def act(self, state: int, epsilon: float = None, tolerance: float = 0.001) -> int:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            # Explore: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit: best action according to Q-network
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        goal_state: np.ndarray = None,
    ) -> None:
        """Update agent's knowledge based on experience"""
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str) -> None:
        """Save agent parameters to file"""
        save_dict = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "loss_history": self.loss_history,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
        }

        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        """Load agent parameters from file"""
        save_dict = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(save_dict["q_network_state_dict"])
        self.target_network.load_state_dict(save_dict["target_network_state_dict"])
        self.optimizer.load_state_dict(save_dict["optimizer_state_dict"])
        self.epsilon = save_dict["epsilon"]
        self.step_count = save_dict["step_count"]
        self.loss_history = save_dict["loss_history"]

        # Verify parameters match
        assert self.state_size == save_dict["state_size"]
        assert self.action_size == save_dict["action_size"]

    def get_q_values(self, state: int) -> np.ndarray:
        """Get Q-values for a given state"""
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()

    def get_loss_history(self) -> List[float]:
        """Get training loss history"""
        return self.loss_history.copy()
