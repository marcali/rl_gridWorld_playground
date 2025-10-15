"""Reward functions for the MDP environment"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class RewTerm:
    """Defines reward calculation for state transitions"""

    def __init__(
        self,
        reward_goal=config.REWARD_GOAL,
        reward_step=config.REWARD_STEP,
        reward_collision=config.REWARD_COLLISION,
    ):
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_collision = reward_collision

    def __call__(self, state, action, next_state, goal_state):
        """
        Calculate reward for a transition

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            env: Environment instance

        Returns:
            float: Reward value
        """
        reward = 0.0

        # Check if agent reached the goal
        if next_state[0] == goal_state[0] and next_state[1] == goal_state[1]:
            reward += self.reward_goal
            # goal reached
            return reward

        # Check if agent hit a wall (position didn't change despite taking action)
        if state[0] == next_state[0] and state[1] == next_state[1]:
            # Position didn't change, likely hit a wall
            reward += self.reward_collision
        else:
            # Agent moved successfully, apply step penalty
            reward += self.reward_step

        return reward
