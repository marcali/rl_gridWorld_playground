"""Goal extraction utilities for GridWorld environment with HER"""

import numpy as np
from typing import Tuple, Any


def extract_achieved_goal_from_state(state: int, grid_size: int) -> Tuple[int, int]:
    """
    Extract achieved goal (current position) from flattened state index

    Args:
        state: Flattened state index
        grid_size: Size of the grid

    Returns:
        Tuple of (y, x) coordinates representing current position
    """
    y = state // grid_size
    x = state % grid_size
    return (y, x)


def create_goal_conditioned_state_simple(
    current_state: int, goal_pos: Tuple[int, int], grid_size: int
) -> int:
    """
    Create a goal-conditioned state by concatenating current state and goal information

    Since the current system only provides current state as observation,
    we'll create a simple concatenation approach that works with the existing structure.

    Args:
        current_state: Current state as flattened index
        goal_pos: Goal position as (y, x) tuple
        grid_size: Size of the grid

    Returns:
        Goal-conditioned state as flattened index
    """
    # Convert goal position to flattened index
    goal_y, goal_x = goal_pos
    goal_index = goal_y * grid_size + goal_x

    # Create combined state: current_state * grid_size^2 + goal_index
    # This ensures each (current_state, goal) pair has a unique index
    combined_state = current_state * (grid_size * grid_size) + goal_index

    return combined_state


class GridWorldGoalExtractor:
    """
    Goal extraction utilities specifically for GridWorld environment
    Compatible with current observation structure that only provides current state
    """

    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    def extract_achieved_goal(self, state: int) -> Tuple[int, int]:
        """Extract achieved goal from state"""
        return extract_achieved_goal_from_state(state, self.grid_size)

    def extract_desired_goal(self, state: int, goal_state: Tuple[int, int]) -> Tuple[int, int]:
        """Extract desired goal from goal_state parameter"""
        return goal_state

    def substitute_goal(self, state: int, new_goal: Tuple[int, int]) -> int:
        """Substitute goal in state representation"""
        return create_goal_conditioned_state_simple(state, new_goal, self.grid_size)

    def create_goal_conditioned_state(self, current_state: int, goal_pos: Tuple[int, int]) -> int:
        """Create goal-conditioned state"""
        return create_goal_conditioned_state_simple(current_state, goal_pos, self.grid_size)
