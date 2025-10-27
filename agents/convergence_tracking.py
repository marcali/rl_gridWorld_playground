"""Convergence tracking mixin for RL agents"""

import numpy as np
from typing import Dict, List, Optional, Any


class ConvergenceTrackingMixin:
    """
    Mixin class providing convergence tracking functionality for RL agents.

    This mixin adds Q-value statistics logging and convergence metrics calculation
    to any agent that implements the required interface methods.

    Required methods from the agent:
    - get_q_values(state: int) -> np.ndarray: Get Q-values for a specific state
    - state_size: int: Number of states in the environment
    """

    def __init_convergence_tracking__(self):
        """Initialize convergence tracking attributes"""
        self.q_value_history: List[Dict[str, Any]] = []
        self.episode_q_stats: List[Dict[str, Any]] = []

    def log_q_value_stats(self, episode: int) -> Dict[str, Any]:
        """
        Log Q-value statistics for convergence tracking

        Args:
            episode: Current episode number

        Returns:
            Dictionary containing Q-value statistics
        """
        # Calculate Q-value statistics for all states
        all_q_values = []
        non_zero_states = 0

        for state in range(self.state_size):
            q_values = self.get_q_values(state)
            all_q_values.extend(q_values)
            # Check if any Q-value for this state is non-zero
            if np.any(q_values != 0):
                non_zero_states += 1

        all_q_values = np.array(all_q_values)

        q_stats = {
            "episode": episode,
            "mean_q": np.mean(all_q_values),
            "std_q": np.std(all_q_values),
            "max_q": np.max(all_q_values),
            "min_q": np.min(all_q_values),
            "q_range": np.max(all_q_values) - np.min(all_q_values),
            "non_zero_states": non_zero_states,
            "total_states": self.state_size,
        }
        self.episode_q_stats.append(q_stats)
        return q_stats

    def get_convergence_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get convergence-related metrics

        Returns:
            Dictionary containing convergence metrics, or None if insufficient data
        """
        if len(self.episode_q_stats) < 2:
            return None

        # Calculate Q-value change over time
        recent_episodes = min(100, len(self.episode_q_stats))
        recent_mean_q = [stats["mean_q"] for stats in self.episode_q_stats[-recent_episodes:]]

        if len(recent_mean_q) < 2:
            return None

        # Calculate stability (how much Q-values are changing)
        q_std_change = np.std(np.diff(recent_mean_q))

        # Calculate convergence rate (how quickly Q-values are stabilizing)
        convergence_rate = 1.0 / (1.0 + q_std_change) if q_std_change > 0 else 1.0

        return {
            "q_value_stability": q_std_change,
            "convergence_rate": convergence_rate,
            "current_mean_q": recent_mean_q[-1],
            "q_value_trend": (
                np.polyfit(range(len(recent_mean_q)), recent_mean_q, 1)[0]
                if len(recent_mean_q) > 1
                else 0
            ),
        }

    def get_state_value_map(self) -> np.ndarray:
        """
        Get the value of each state (max Q-value for each state)

        Returns:
            Array containing the maximum Q-value for each state
        """
        state_values = np.zeros(self.state_size)
        for state in range(self.state_size):
            q_values = self.get_q_values(state)
            state_values[state] = np.max(q_values)
        return state_values
