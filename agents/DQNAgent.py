"""Deep Q-Network (DQN) Agent"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from typing import List
from .agent_protocol import AgentProtocol
from .convergence_tracking import ConvergenceTrackingMixin
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


class DQNAgent(AgentProtocol, ConvergenceTrackingMixin):
    """Deep Q-Network Agent"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float,
        batch_size: int = 32,
        memory_size: int = 10000,
        target_update_freq: int = 100,
        hidden_size: int = 128,
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

        # Networks
        self.q_network = DQNNetwork(state_size, action_size, hidden_size)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Experience replay buffer
        self.memory = ReplayBuffer(memory_size)

        # Training tracking
        self.step_count = 0
        self.loss_history = []

        # Initialize convergence tracking mixin
        self.__init_convergence_tracking__()

    def _state_to_tensor(self, state: int) -> torch.Tensor:
        """Convert state to tensor"""
        # Convert state index to one-hot encoding
        state_tensor = torch.zeros(self.state_size, dtype=torch.float32)
        state_tensor[state] = 1.0
        return state_tensor.unsqueeze(0)

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
        goal_state: tuple = None,
        info: dict = None,
    ) -> None:
        """Update agent's knowledge based on experience"""
        # Always add experience to replay buffer
        # If memory is HER wrapper, it will handle HER logic
        # If memory is regular ReplayBuffer, it will add normally
        if hasattr(self.memory, "add_experience") and goal_state is not None:
            # HER wrapper - use add_experience with goal info
            self.memory.add_experience(
                state, action, reward, next_state, done, {"goal_state": goal_state}
            )
        else:
            # Regular ReplayBuffer - use standard add method
            self.memory.add(state, action, reward, next_state, done)

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Train the network if we have enough experiences
        if len(self.memory) >= self.batch_size:
            self._train_network()

    def _train_network(self) -> None:
        """Internal method to train the DQN network"""
        # Sample experiences from replay buffer
        experiences = self.memory.sample(self.batch_size)

        # Convert experiences to tensors
        states = torch.stack([self._state_to_tensor(exp.state).squeeze(0) for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        next_states = torch.stack(
            [self._state_to_tensor(exp.next_state).squeeze(0) for exp in experiences]
        )
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Track loss
        self.loss_history.append(loss.item())

    def save(self, path: str) -> None:
        """Save agent parameters to file"""
        save_dict = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "loss_history": self.loss_history,
            "q_value_history": self.q_value_history,
            "episode_q_stats": self.episode_q_stats,
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
        save_dict = torch.load(path, map_location="cpu")

        self.q_network.load_state_dict(save_dict["q_network_state_dict"])
        self.target_network.load_state_dict(save_dict["target_network_state_dict"])
        self.optimizer.load_state_dict(save_dict["optimizer_state_dict"])
        self.epsilon = save_dict["epsilon"]
        self.step_count = save_dict["step_count"]
        self.loss_history = save_dict["loss_history"]

        # Load convergence tracking data (with backward compatibility)
        self.q_value_history = save_dict.get("q_value_history", [])
        self.episode_q_stats = save_dict.get("episode_q_stats", [])

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
