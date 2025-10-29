"""Abstract base class for RL environments

This base class defines the minimal contract any concrete environment must
implement. In addition to the usual reset/step/render API, environments in
this codebase are expected to be built from explicit MDP components:

Required MDP components that every environment must provide:
- Reward term: computes scalar rewards for transitions
- Observation term: converts internal state into agent observations
- Done term: determines episode termination

Optional components supported by the framework:
- Curriculum manager/rules: adjusts difficulty or rules over episodes/steps
- Event manager/terms: injects stochastic events or dynamics each episode

Concrete implementations should expose these as attributes so downstream code
can interact with them in a consistent way:
- self.rew_term (required)
- self.obs_term (required)
- self.done_term (required)
- self.curriculum_rules or self.curriculum_manager (optional)
- self.event_manager (optional)
"""

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
