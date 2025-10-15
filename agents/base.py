"""Base protocol for all agents"""

from abc import ABC, abstractmethod
from typing import Union


class AgentProtocol(ABC):
    """Abstract base class defining the interface all agents must implement"""

    @abstractmethod
    def act(self, state: int, epsilon: float = 0.1, tolerance: float = 0.001) -> int:
        """
        Select an action based on the current state

        Args:
            state: Current state index
            epsilon: Exploration rate (0.0 = no exploration, 1.0 = full exploration)
            tolerance: Tolerance for tie-breaking in action selection

        Returns:
            Selected action (integer from 0 to n_actions-1)
        """
        pass

    @abstractmethod
    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """
        Update agent's knowledge based on experience

        Args:
            state: State where action was taken
            action: Action that was taken
            reward: Reward received for taking action in state
            next_state: State reached after taking action
            done: Whether the episode is finished
        """
        pass
