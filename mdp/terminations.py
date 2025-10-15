"""Termination conditions for the MDP environment"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class DoneTerm:
    """Defines when an episode should terminate"""

    def __init__(self):
        pass

    def __call__(self, state, goal_state, max_steps, step_count):
        """
        Check if episode should terminate

        Args:
            state: Current state of the environment (as [y, x])
            goal_state: Goal state of the environment (as [y, x])
            max_steps: Maximum number of steps per episode
            step_count: Current step count

        Returns:
            bool: True if gent reached the goal
        """
        # Check if agent reached the goal
        if (state[0] == goal_state[0] and state[1] == goal_state[1]) or step_count >= max_steps:
            return True

        return False
