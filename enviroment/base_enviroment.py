"""Abstract base class for RL environments"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import numpy as np


class BaseEnvironment(ABC):
    """Abstract base class defining the interface for any RL environment"""

    def __init__(self):
        """Initialize the environment"""
        self.step_count = 0
        self.current_state = None

    @abstractmethod
    def reset(self) -> Any:
        """
        Reset the environment to its initial state

        Returns:
            observation: Initial observation of the environment
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool]:
        """
        Execute one step in the environment

        Args:
            action: Action to take (environment-specific)

        Returns:
            observation: Current observation of the environment
            reward: Reward for the transition
            done: Whether the episode is finished
        """
        pass

    @abstractmethod
    def render(
        self, path: Optional[list] = None, title: Optional[str] = None, show_agent: bool = True
    ):
        """
        Visualize the current environment state

        Args:
            path: List of states representing a path to overlay
            title: Title for the plot
            show_agent: Whether to show the current agent position

        Returns:
            matplotlib figure or visualization object
        """
        pass

    @abstractmethod
    def render_path(self, path: list, title: str = "Agent Path", save_path: Optional[str] = None):
        """
        Visualize a complete path taken by the agent

        Args:
            path: List of states representing the path
            title: Title for the plot
            save_path: If provided, saves figure to this path instead of showing

        Returns:
            matplotlib figure or visualization object
        """
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Return the number of possible actions"""
        pass

    @property
    @abstractmethod
    def state_space_size(self) -> int:
        """Return the number of possible states"""
        pass

    @property
    def current_observation(self) -> Any:
        """Get the current observation"""
        return self.current_state

    @property
    def episode_step_count(self) -> int:
        """Get the current step count for this episode"""
        return self.step_count
