"""Replay Buffer implementation for DQN Agent"""

import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Optional, Tuple
import torch


# Experience tuple for replay buffer
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN Agent

    This buffer stores experiences and provides sampling functionality for training.
    It's designed to be used as an attribute of the DQN agent.
    """

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed for reproducible sampling
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.seed = seed

        if seed is not None:
            random.seed(seed)

    def add(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """
        Add a new experience to the buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode is finished
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences from the buffer

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)

    def sample_tensors(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch and convert to PyTorch tensors

        Args:
            batch_size: Number of experiences to sample
            device: Device to place tensors on

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        experiences = self.sample(batch_size)

        # Convert to tensors
        states = torch.tensor([exp.state for exp in experiences], dtype=torch.long, device=device)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long, device=device)
        rewards = torch.tensor(
            [exp.reward for exp in experiences], dtype=torch.float32, device=device
        )
        next_states = torch.tensor(
            [exp.next_state for exp in experiences], dtype=torch.long, device=device
        )
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool, device=device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current number of experiences in buffer"""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough experiences for sampling

        Args:
            batch_size: Required batch size

        Returns:
            True if buffer has enough experiences
        """
        return len(self.buffer) >= batch_size

    def is_full(self) -> bool:
        """Check if buffer is at capacity"""
        return len(self.buffer) >= self.capacity

    def clear(self) -> None:
        """Clear all experiences from the buffer"""
        self.buffer.clear()

    def get_stats(self) -> dict:
        """
        Get statistics about the replay buffer

        Returns:
            Dictionary with buffer statistics
        """
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "avg_reward": 0.0,
                "success_rate": 0.0,
            }

        rewards = [exp.reward for exp in self.buffer]
        dones = [exp.done for exp in self.buffer]

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "avg_reward": np.mean(rewards),
            "success_rate": np.mean(dones) if dones else 0.0,
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }

    def get_recent_experiences(self, n: int = 100) -> List[Experience]:
        """
        Get the most recent n experiences

        Args:
            n: Number of recent experiences to return

        Returns:
            List of recent experiences
        """
        return list(self.buffer)[-n:] if n <= len(self.buffer) else list(self.buffer)

    def get_episode_experiences(self) -> List[List[Experience]]:
        """
        Get experiences grouped by episodes

        Returns:
            List of episodes, where each episode is a list of experiences
        """
        episodes = []
        current_episode = []

        for exp in self.buffer:
            current_episode.append(exp)
            if exp.done:
                episodes.append(current_episode)
                current_episode = []

        # Add incomplete episode if exists
        if current_episode:
            episodes.append(current_episode)

        return episodes


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer

    Samples experiences based on their TD error (temporal difference error)
    Higher TD error experiences are sampled more frequently.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        seed: Optional[int] = None,
    ):
        """
        Initialize prioritized replay buffer

        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction exponent
            beta_increment: How much to increment beta per sample
            seed: Random seed for reproducible sampling
        """
        super().__init__(capacity, seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Add experience with maximum priority"""
        super().add(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int) -> Tuple[List[Experience], List[float], List[int]]:
        """
        Sample experiences based on priorities

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (experiences, importance_weights, indices)
        """
        if len(self.buffer) < batch_size:
            experiences = list(self.buffer)
            weights = [1.0] * len(experiences)
            indices = list(range(len(experiences)))
            return experiences, weights, indices

        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Get experiences
        experiences = [self.buffer[i] for i in indices]

        return experiences, weights.tolist(), indices.tolist()

    def update_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        """
        Update priorities based on TD errors

        Args:
            indices: Indices of experiences to update
            td_errors: TD errors for those experiences
        """
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                priority = (abs(td_error) + 1e-6) ** self.alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

    def increment_beta(self) -> None:
        """Increment beta for importance sampling correction"""
        self.beta = min(1.0, self.beta + self.beta_increment)
