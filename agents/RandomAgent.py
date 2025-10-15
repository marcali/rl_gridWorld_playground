"""Random Agent - Takes completely random actions for baseline comparison"""

import numpy as np
from .base import AgentProtocol


class RandomAgent(AgentProtocol):
    """Agent that takes completely random actions - used as baseline"""

    def __init__(self, n_states, n_actions):
        """
        Initialize random agent

        Args:
            n_states: Number of states (not used, but kept for consistency)
            n_actions: Number of possible actions
        """
        self.n_actions = n_actions

    def act(self, state, epsilon=None, tolerance=None):
        """
        Select a completely random action

        Args:
            state: Current state (ignored)
            epsilon: Exploration rate (ignored)
            tolerance: Q-value tolerance (ignored)

        Returns:
            Random action (integer from 0 to n_actions-1)
        """
        return np.random.randint(self.n_actions)

    def learn(self, state, action, reward, next_state, done):
        """
        Random agent doesn't learn - this method does nothing

        Args:
            state: Current state (ignored)
            action: Action taken (ignored)
            reward: Reward received (ignored)
            next_state: Next state (ignored)
            done: Episode done flag (ignored)
        """
        pass
