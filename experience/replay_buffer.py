"""Replay Buffer implementation for DQN Agent"""

import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Optional


# Experience tuple for replay buffer
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN Agent
    """

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
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

    def __len__(self) -> int:
        """Return current number of experiences in buffer"""
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer

    Samples experiences based on their TD error (temporal difference error)
    Higher TD error experiences are sampled more frequently.
    """

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        super().__init__(capacity, seed)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.5  # Priority exponent
        self.beta = 0.4  # Importance sampling exponent
        self.max_priority = 1.0  # Maximum priority
        self.min_priority = 0.01  # Minimum priority
